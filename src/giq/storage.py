# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Disk accounting and deletion for weights.

Every checkpoint the recipes name (``giq.weights.inventory``) resolves to its
files on disk: a GGUF with its shards, a safetensors component, a checkpoint
directory, an HF-cache repository directory. One checkpoint loaded by two
recipes — the stt recipe faster-whisper-large-v3 and the audio resident share
a faster-whisper snapshot — is one item, sized and deleted once. Where a
recipe's files are is its own to say (``giq.weights``).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from giq.paths import hf_home
from giq.weights import Location, WeightsItem, inventory, locations

# GGUF shard names: model-00001-of-00004.gguf → glob the whole set.
_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-(\d{5})$")


class StorageError(Exception):
    """Deletion refused or failed; .status carries the HTTP code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def hf_cache_dir() -> Path:
    """HF hub cache root, honoring the usual env overrides.

    ``paths.hf_home`` is also what giq exports to its children, so the
    catalog looks where they download.
    """
    if explicit := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(explicit)
    return hf_home() / "hub"


def _on_disk(loc: Location, hub: Path) -> Path:
    """A path as it is, a repository as its HF-cache directory."""
    if loc.path:
        return Path(loc.path)
    return hub / f"models--{str(loc.repo).replace('/', '--')}"


def _expand_gguf(path: Path) -> list[Path]:
    """A sharded GGUF's siblings count as part of the model; anything else is itself.

    A checkpoint directory is counted whole already, and its own
    ``model-00001-of-00003.safetensors`` shards are not the GGUF set this
    globs for.
    """
    if path.suffix != ".gguf":
        return [path]
    m = _SHARD_RE.match(path.stem)
    if not m:
        return [path]
    siblings = sorted(path.parent.glob(f"{m.group(1)}-*-of-{m.group(2)}{path.suffix}"))
    return siblings or [path]


def _path_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def _mount_point(p: Path) -> Path:
    while not os.path.ismount(p):
        p = p.parent
    return p


def _item_paths(item: WeightsItem, hub: Path) -> list[Path]:
    """Every file or directory a weights item is on disk, shards included."""
    loc = Location(None, path=item.path, repo=item.repo)
    return [p.expanduser().resolve() for p in _expand_gguf(_on_disk(loc, hub))]


def weights_report() -> list[dict]:
    """Every checkpoint the recipes name, with its size, presence and users.

    Each checkpoint is counted once however many recipes load it, and
    deleting it is one act whatever uses it.
    """
    hub = hf_cache_dir()
    out = []
    for item in sorted(inventory(), key=lambda i: i.repo or i.path or ""):
        paths = _item_paths(item, hub)
        existing = [p for p in paths if p.exists()]
        out.append(
            {
                "id": item.id,
                "path": item.path,
                "repo": item.repo,
                "format": item.format,
                "source": item.source,
                "revision": item.revision,
                "licence": item.licence,
                "recipes": list(item.recipes),
                "used_by": list(item.used_by),
                "on_disk": bool(paths) and len(existing) == len(paths),
                "size_bytes": sum(_path_size(p) for p in existing),
                "mount": str(_mount_point(existing[0])) if existing else None,
            }
        )
    return out


def disk_report() -> list[dict]:
    """Per mount: its size, what is free, and how much of it is weights.

    Each file is counted once, however many recipes load it; ``other_bytes``
    is what the mount holds besides the weights and is what an operator
    looks at before blaming the models for a full disk.
    """
    hub = hf_cache_dir()
    paths = {p for item in inventory() for p in _item_paths(item, hub) if p.exists()}
    disks: dict[Path, dict] = {}
    for p in paths:
        mount = _mount_point(p)
        if mount not in disks:
            st = os.statvfs(mount)
            disks[mount] = {
                "mount": str(mount),
                "total_bytes": st.f_blocks * st.f_frsize,
                "free_bytes": st.f_bavail * st.f_frsize,
                "models_bytes": 0,
            }
        disks[mount]["models_bytes"] += _path_size(p)
    for d in disks.values():
        d["other_bytes"] = max(0, d["total_bytes"] - d["free_bytes"] - d["models_bytes"])
    return sorted(disks.values(), key=lambda d: d["mount"])


def delete_weights(weights_id: str, *, busy: set[str]) -> dict:
    """Delete one checkpoint from disk; returns what happened per path.

    Refused while any recipe that uses it is in ``busy`` (resident or loaded):
    evicting is scheduler policy, not a disk operation, and a resident would
    only fail to reload. Recipes that used it stay in the catalog, uninstalled.
    """
    item = next((i for i in inventory() if i.id == weights_id), None)
    if item is None:
        raise StorageError(f"no weights {weights_id}", status=404)
    if blocking := sorted(set(item.recipes) & busy):
        raise StorageError(
            f"{', '.join(blocking)} {'uses' if len(blocking) == 1 else 'use'} these weights "
            "and is resident or loaded — set it to on-demand or off first",
            status=409,
        )
    deleted, missing = [], []
    freed = 0
    for p in _item_paths(item, hf_cache_dir()):
        if not p.exists():
            missing.append(str(p))
            continue
        size = _path_size(p)
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError as e:
            # A hardened service mounts the model store read-only (the systemd
            # unit's ProtectSystem=strict); weights are the operator's to
            # delete, and a 500 would read like a giq bug.
            raise StorageError(
                f"cannot delete {p}: {e.strerror or e}. The model store is read-only "
                "for the service; delete the files as the operator"
                + (f" (already deleted: {', '.join(deleted)})" if deleted else ""),
                status=409,
            ) from e
        freed += size
        deleted.append(str(p))
    return {
        "id": item.id,
        "recipes": list(item.recipes),
        "deleted": deleted,
        "missing": missing,
        "freed_bytes": freed,
    }


def on_disk(loc: Location) -> Path:
    """Where a location's files are, or go: a path as it is, a repository as
    its HF-cache directory."""
    return _on_disk(loc, hf_cache_dir())


def missing(name: str) -> list[Location]:
    """Recipe ``name``'s locations that are not (wholly) on disk.

    A presence check, not a size: the catalog asks it for every recipe on
    every poll, and walking a checkpoint directory to sum it is the storage
    report's job.
    """
    hub = hf_cache_dir()
    return [
        loc
        for loc in locations(name)
        if not all(p.exists() for p in _expand_gguf(_on_disk(loc, hub)))
    ]


def installed(name: str) -> bool:
    """Are recipe ``name``'s weights on disk? Every file, part and shard."""
    return bool(locations(name)) and not missing(name)
