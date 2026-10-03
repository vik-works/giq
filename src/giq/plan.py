# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Before anything is fetched, giq plans (ADR-005 D3).

The plan is what :mod:`giq.availability` knows about this machine plus what
only the Hub knows: how large the missing files are, whether the repository
is gated and the token can open it, and the licence. One function behind
``giq add --dry-run``, ``GET /recipes/{name}/plan`` and the dashboard's Add
view. Without network the Hub's checks read "unknown"; they never fail the
plan on their own.

It also lists the transfers a fetch carries out, so the fetch does exactly
what the plan showed.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from giq.availability import Availability, Check, availability_of, checks
from giq.recipes.schema import Recipe
from giq.registry import get_recipe

logger = logging.getLogger(__name__)

# Licences that ask nothing of a commercial user beyond attribution.
PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "openrail"}

# GGUF shard names: the first shard brings its siblings.
_SHARD_RE = re.compile(r"^(.*)-00001-of-(\d{5})\.gguf$")


@dataclass
class Transfer:
    """One location's files, from one repository at one revision."""

    part: str | None
    repo: str
    revision: str | None
    # The file a single-file source names; None is the whole repository.
    file: str | None
    # Where the files go: the file or directory the recipe loads, or None
    # for a repository the children load from the Hugging Face cache.
    dest: Path | None
    # (file in the repository, bytes), as the Hub lists them; empty when the
    # Hub did not answer.
    files: list[tuple[str, int]] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        return sum(size for _, size in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "repo": self.repo,
            "revision": self.revision,
            "file": self.file,
            "dest": str(self.dest) if self.dest else None,
            "files": len(self.files),
            "bytes": self.bytes,
        }


@dataclass
class Plan:
    recipe: str
    availability: Availability
    checks: list[Check]
    transfers: list[Transfer]
    # Can the service write where the files go (else: `giq add` as the operator)?
    service_can_fetch: bool

    @property
    def download_bytes(self) -> int:
        return sum(t.bytes for t in self.transfers)

    @property
    def can_fetch(self) -> bool:
        """Nothing but the missing weights stands in the way."""
        return self.availability == "fetchable" and not any(c.status == "fail" for c in self.checks)

    @property
    def command(self) -> str:
        return f"giq add {self.recipe}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "availability": self.availability,
            "can_fetch": self.can_fetch,
            "service_can_fetch": self.service_can_fetch,
            "command": self.command,
            "download_bytes": self.download_bytes,
            "checks": [c.to_dict() for c in self.checks],
            "transfers": [t.to_dict() for t in self.transfers],
        }


def _blocks(recipe: Recipe) -> dict[str | None, Any]:
    """Part (None for the main weights) -> its weights block."""
    if recipe.weights is None:
        return {}
    return {None: recipe.weights, **recipe.weights.parts}


def transfers(recipe: Recipe) -> list[Transfer]:
    """What a fetch of ``recipe`` moves: one per missing location with a source."""
    from giq.storage import missing
    from giq.weights import hub_file, hub_repo

    blocks = _blocks(recipe)
    out = []
    for loc in missing(recipe.name):
        block = blocks.get(loc.part)
        if block is None or (repo := hub_repo(block.source)) is None:
            continue
        out.append(
            Transfer(
                part=loc.part,
                repo=repo,
                revision=block.revision,
                file=hub_file(block.source),
                dest=Path(loc.path) if loc.path else None,
            )
        )
    return out


def _wanted(siblings: list[tuple[str, int]], file: str | None) -> list[tuple[str, int]]:
    """The repository's files a transfer takes: all, or one file and its shards."""
    if file is None:
        return siblings
    m = _SHARD_RE.match(file)
    if m is None:
        return [(name, size) for name, size in siblings if name == file]
    shard = re.compile(rf"^{re.escape(m.group(1))}-\d{{5}}-of-{m.group(2)}\.gguf$")
    return [(name, size) for name, size in siblings if shard.match(name)]


def _opens(repo: str, token: str | None) -> bool:
    """Can ``token`` download from gated ``repo``?"""
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError

    if token is None:
        return False
    try:
        auth_check(repo, token=token)
    except GatedRepoError:
        return False
    return True


def _hub(plan_checks: list[Check], moves: list[Transfer], recipe: Recipe) -> None:
    """Fill in sizes, and check access and licence, from the Hub."""
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    api = HfApi()
    token = get_token()
    licences: set[str] = set()
    for move in moves:
        try:
            info = api.model_info(move.repo, revision=move.revision, files_metadata=True)
        except RepositoryNotFoundError:
            plan_checks.append(
                Check("access", "fail", f"{move.repo} is not on the Hub, or is private")
            )
            continue
        except RevisionNotFoundError:
            plan_checks.append(
                Check("access", "fail", f"{move.repo} has no revision {move.revision}")
            )
            continue
        except Exception as e:  # offline, a proxy, the Hub down: the plan still stands
            plan_checks.append(
                Check(
                    "access",
                    "warn",
                    f"the Hub did not answer ({type(e).__name__}): "
                    "download size and access are unknown",
                )
            )
            continue
        siblings = [(s.rfilename, s.size or 0) for s in info.siblings or []]
        move.files = _wanted(siblings, move.file)
        if move.file is not None and not move.files:
            plan_checks.append(Check("access", "fail", f"{move.repo} has no file {move.file}"))
        if info.gated and not _opens(move.repo, token):
            plan_checks.append(
                Check(
                    "access",
                    "fail",
                    f"{move.repo} is gated: accept its terms at "
                    f"https://huggingface.co/{move.repo} and set HF_TOKEN"
                    + ("" if token is None else " to a token of that account"),
                )
            )
        card = info.card_data.to_dict() if info.card_data else {}
        if licence := card.get("license"):
            licences.add(str(licence))
    declared = {b.licence for b in _blocks(recipe).values() if b is not None and b.licence}
    for licence in sorted(licences | declared):
        if licence.lower() not in PERMISSIVE:
            plan_checks.append(
                Check("licence", "warn", f"licence {licence}: read it before using the output")
            )
    if licences | declared and all(lic.lower() in PERMISSIVE for lic in licences | declared):
        plan_checks.append(Check("licence", "ok", ", ".join(sorted(licences | declared))))


def _existing(path: Path) -> Path:
    """The nearest ancestor of ``path`` that exists: where its mount is."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _destination(move: Transfer) -> Path:
    from giq.storage import hf_cache_dir

    return move.dest if move.dest is not None else hf_cache_dir()


def _disk(plan_checks: list[Check], moves: list[Transfer]) -> bool:
    """Room where the files go; returns whether the service may write there."""
    need: dict[int, tuple[Path, int]] = {}
    writable = True
    for move in moves:
        at = _existing(_destination(move))
        writable = writable and os.access(at, os.W_OK)
        device = at.stat().st_dev
        place, total = need.get(device, (at, 0))
        need[device] = (place, total + move.bytes)
    for place, total in need.values():
        st = os.statvfs(place)
        free = st.f_bavail * st.f_frsize
        if total > free:
            plan_checks.append(
                Check(
                    "disk",
                    "fail",
                    f"{total / 1e9:.1f} GB to fetch, {free / 1e9:.1f} GB free under {place}",
                )
            )
        elif total:
            plan_checks.append(
                Check(
                    "disk", "ok", f"{total / 1e9:.1f} GB of {free / 1e9:.1f} GB free under {place}"
                )
            )
    return writable


def _already_here(plan_checks: list[Check], recipe: Recipe) -> None:
    """Weights another recipe already brought: shared parts are fetched once."""
    from giq.storage import missing
    from giq.weights import locations

    gone = {(loc.part, loc.path, loc.repo) for loc in missing(recipe.name)}
    present = [loc for loc in locations(recipe.name) if (loc.part, loc.path, loc.repo) not in gone]
    if present and gone:
        names = ", ".join(loc.part or "weights" for loc in present)
        plan_checks.append(Check("weights", "ok", f"already on disk: {names}"))


def _pinned(plan_checks: list[Check], moves: list[Transfer]) -> None:
    unpinned = [m.repo for m in moves if m.revision is None and m.dest is not None]
    if unpinned:
        plan_checks.append(
            Check(
                "pinned",
                "warn",
                f"no revision for {', '.join(unpinned)}: the files are whatever its main "
                "branch holds now, not the ones the recipe's VRAM figure was measured with",
            )
        )


def plan(name: str, *, online: bool = True) -> Plan:
    """The plan for recipe ``name``. Raises KeyError for an unknown recipe."""
    recipe = get_recipe(name)
    if recipe is None:
        raise KeyError(name)
    found = checks(recipe)
    avail = availability_of(found)
    moves = transfers(recipe) if avail != "ready" else []
    if moves:
        if online:
            _hub(found, moves, recipe)
        _already_here(found, recipe)
        _pinned(found, moves)
    writable = _disk(found, moves) if moves else True
    return Plan(recipe.name, avail, found, moves, service_can_fetch=writable)
