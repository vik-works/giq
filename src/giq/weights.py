# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Where a model's weights are, as its recipe file says (ADR-002).

Workers used to find their weights through tables of their own — a snapshot
directory name per model, hard-coded next to the worker — so a recipe
file could add a model to the catalog that its worker then refused to load.
Every worker now asks here, and the answer comes from the recipe:
``weights.path`` for the main weights, ``weights.parts.<name>`` for the other
files a model needs (an image model's text encoder and VAE, OCR's layout
model). A relative path is taken under the models directory; ``~`` and
absolute paths are used as written.

The environment variables that located these snapshots before recipe
files existed still work, and outrank the recipe, as an environment
variable outranks a file everywhere else in giq:

- a directory for one model's weights (``DIR_OVERRIDES``), which applies to
  that model name only — an operator's second OCR recipe is not redirected
  by a variable documented for the built-in;
- a root for one modality's relative paths (its registered
  ``weights_root_env``), in place of the models directory.

Models that load by Hugging Face repository rather than by path (the audio
and speech workers) record it as ``source: hf:org/repo``; :func:`hub_repo`
reads it, and the storage catalog finds the download in the HF cache by it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from giq import recipes
from giq.paths import model_path
from giq.recipes.schema import Recipe

# A directory that replaces one recipe's weights: (recipe, part) -> variable;
# part None is the main weights.
DIR_OVERRIDES: dict[tuple[str, str | None], str] = {
    ("unlimited-ocr", None): "GIQ_OCR_MODEL_DIR",
    ("glm-ocr", None): "GIQ_GLM_OCR_MODEL_DIR",
    ("glm-ocr", "layout"): "GIQ_GLM_LAYOUT_DIR",
}

HF_PREFIX = "hf:"


def recipe_of(name: str) -> Recipe | None:
    """The current recipe ``name`` means, by name or alias."""
    return recipes.current().get(str(name))


def _env(var: str | None) -> str | None:
    return os.environ.get(var) if var else None


def resolve_path(modality: str, raw: str) -> str:
    """A path as a recipe writes it, made absolute for a recipe of ``modality``."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    from giq import plugins

    spec = plugins.modality(str(modality))
    if root := _env(spec.weights_root_env if spec is not None else None):
        return str(Path(root).expanduser() / p)
    return model_path(p)


def path_of(name: str, part: str | None = None) -> str | None:
    """Absolute path of a recipe's main weights (``part=None``) or of one part.

    None when neither an override nor the recipe gives one — the recipe is
    unknown, has no such part, or loads by repository rather than by path.
    """
    recipe = recipe_of(name)
    canonical = recipe.name if recipe is not None else str(name)
    if override := _env(DIR_OVERRIDES.get((canonical, part))):
        return str(Path(override).expanduser())
    if recipe is None or recipe.weights is None:
        return None
    if part is None:
        raw = recipe.weights.path
    else:
        piece = recipe.weights.parts.get(part)
        raw = piece.path if piece is not None else None
    return resolve_path(recipe.modality, raw) if raw else None


def require_path(name: str, part: str | None = None) -> str:
    """:func:`path_of`, raising when there is none — at construction, not at spawn."""
    path = path_of(name, part)
    if path is None:
        what = "weights.path" if part is None else f"weights.parts.{part}"
        if recipe_of(name) is None:
            raise ValueError(f"unknown recipe {name!r}: no recipe file defines it")
        raise ValueError(f"{name} has no {what} in its recipe file")
    return path


def hub_repo(source: str | None) -> str | None:
    """``org/repo`` of an ``hf:org/repo`` source, else None."""
    if source and source.startswith(HF_PREFIX):
        return source[len(HF_PREFIX) :] or None
    return None


def load_ref(name: str, part: str | None = None) -> str | None:
    """What a library that takes "a path or a repo id" should load.

    The path when the recipe gives one (or an override does), else the
    repository of its ``hf:`` source, else None.
    """
    if path := path_of(name, part):
        return path
    recipe = recipe_of(name)
    if recipe is None or recipe.weights is None:
        return None
    if part is None:
        return hub_repo(recipe.weights.source)
    piece = recipe.weights.parts.get(part)
    return hub_repo(piece.source) if piece is not None else None


IMAGE_PARTS = ("diffusion", "text_encoder", "vae", "lora")
_REQUIRED_IMAGE_PARTS = ("diffusion", "text_encoder", "vae")


@dataclass(frozen=True)
class ImageFiles:
    """An image recipe's files, as sd-server takes them."""

    diffusion: str
    text_encoder: str
    vae: str
    lora: str | None = None


def image_files(name: str) -> ImageFiles:
    """An image recipe's ``weights.parts``, paths resolved like any weights."""
    recipe = recipe_of(name)
    if recipe is None:
        raise ValueError(f"unknown recipe {name!r}: no recipe file defines it")
    files = {part: path_of(recipe.name, part) for part in IMAGE_PARTS}
    if missing := [p for p in _REQUIRED_IMAGE_PARTS if not files[p]]:
        raise ValueError(
            f"{recipe.name} names no {', '.join(f'weights.parts.{p}' for p in missing)} "
            "in its recipe file"
        )
    return ImageFiles(
        diffusion=str(files["diffusion"]),
        text_encoder=str(files["text_encoder"]),
        vae=str(files["vae"]),
        lora=files["lora"],
    )


@dataclass(frozen=True)
class Location:
    """One thing on disk a model needs: a path, or an HF repository whose
    download lives in the HF cache."""

    part: str | None
    path: str | None = None
    repo: str | None = None


def locations(name: str) -> list[Location]:
    """Every file, snapshot or repository recipe ``name`` loads, main weights first."""
    recipe = recipe_of(name)
    weights = recipe.weights if recipe is not None else None
    parts: list[tuple[str | None, str | None]] = [
        (None, weights.source if weights else None),
        *((part, p.source) for part, p in (weights.parts.items() if weights else ())),
    ]
    out = []
    for part, source in parts:
        if path := path_of(name, part):
            out.append(Location(part, path=path))
        elif repo := hub_repo(source):
            out.append(Location(part, repo=repo))
    return out


# --- the inventory (ADR-003) ---------------------------------------------------
#
# Weights are a thing of their own: one checkpoint, however many recipes use
# it. Recipes still write their weights inline — one file per new model — and
# the inventory is built from them, keyed by where the files are. A recipe's
# parts (an image model's diffusion model, text encoder and VAE) are weights
# too, so an encoder two recipes share is one item, deleted once.

_PROVENANCE = ("format", "source", "revision", "licence")


@dataclass(frozen=True)
class WeightsItem:
    """One checkpoint on this machine (or that should be), and who uses it."""

    # A short, stable hash of the location: URL-safe, and the same across
    # restarts as long as the files stay where they are.
    id: str
    # Exactly one of these: an absolute path (a file or a directory), or the
    # ``org/repo`` whose download lives in the HF cache.
    path: str | None
    repo: str | None
    format: str | None = None
    source: str | None = None
    revision: str | None = None
    licence: str | None = None
    # "recipe" for main weights, "recipe:part" for a part, by name.
    used_by: tuple[str, ...] = ()

    @property
    def recipes(self) -> tuple[str, ...]:
        """The recipes that load these weights, each once."""
        return tuple(dict.fromkeys(u.partition(":")[0] for u in self.used_by))


def weights_id(location: Location) -> str:
    import hashlib

    key = f"path:{location.path}" if location.path else f"repo:{location.repo}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _declared(recipe: Recipe) -> list[tuple[str | None, Any]]:
    """(part, its weights block) for the main weights and every part."""
    if recipe.weights is None:
        return []
    main = [(None, recipe.weights)] if recipe.weights.path or recipe.weights.source else []
    return main + list(recipe.weights.parts.items())


def _provenance(block: Any, main: Any = None) -> dict[str, str]:
    """What a weights block says its files are.

    A part that states no licence of its own is under the recipe's: flux_klein
    declares apache-2.0 once over all three files and names the VAE's source
    only because it differs. Format, source and revision describe one
    download and are not inherited.
    """
    own = {k: v for k in _PROVENANCE if (v := getattr(block, k, None)) is not None}
    if main is not None and main is not block and "licence" not in own and main.licence:
        own["licence"] = main.licence
    return own


def inventory() -> list[WeightsItem]:
    """Every checkpoint the current recipes name, each once, by location."""
    found: dict[str, dict[str, Any]] = {}
    for recipe in sorted(recipes.current().recipes.values(), key=lambda r: r.name):
        declared = dict(_declared(recipe))
        for loc in locations(recipe.name):
            item = found.setdefault(
                weights_id(loc), {"location": loc, "provenance": {}, "used_by": []}
            )
            block = declared.get(loc.part)
            for key, value in _provenance(block, recipe.weights).items():
                item["provenance"].setdefault(key, value)
            item["used_by"].append(recipe.name if loc.part is None else f"{recipe.name}:{loc.part}")
    return [
        WeightsItem(
            id=wid,
            path=item["location"].path,
            repo=item["location"].repo,
            used_by=tuple(item["used_by"]),
            **item["provenance"],
        )
        for wid, item in found.items()
    ]


def provenance_conflicts(candidates: Iterable[Recipe]) -> list[str]:
    """Recipes that describe one checkpoint differently.

    Two recipes over the same files must agree on what the files are; which
    file's licence the catalog showed would otherwise be an accident of load
    order. Keyed by the declared location — environment overrides redirect
    one recipe's files at run time and are not a statement about the files.
    A value one recipe leaves out is not a disagreement.
    """
    seen: dict[str, tuple[str, dict[str, str]]] = {}
    problems = []
    for recipe in candidates:
        for part, block in _declared(recipe):
            if block.path:
                key = f"path:{resolve_path(recipe.modality, block.path)}"
            elif repo := hub_repo(block.source):
                key = f"repo:{repo}"
            else:
                continue
            who = recipe.name if part is None else f"{recipe.name}:{part}"
            mine = _provenance(block, recipe.weights)
            if key not in seen:
                seen[key] = (who, mine)
                continue
            other, theirs = seen[key]
            for field in sorted(set(mine) & set(theirs)):
                if mine[field] != theirs[field]:
                    problems.append(
                        f"{who} and {other} use the same weights but disagree on {field} "
                        f"({mine[field]!r} vs {theirs[field]!r})"
                    )
            seen[key] = (other, {**mine, **theirs})
    return problems
