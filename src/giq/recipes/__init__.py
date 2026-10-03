# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Model recipes: every model giq serves, loaded from YAML files (ADR-002).

Two sources, read in order:

- the built-in recipes, one file each next to this module
  (``<name>.yaml``), shipped in the package;
- the operator's, ``*.yaml``/``*.yml`` directly in
  :func:`giq.paths.recipes_dir`, which add recipes or replace the built-in
  of the same name.

A recipe's name is what a client sends as ``model`` and the key every other
part of giq uses for it, so names — and aliases — are unique across
modalities (ADR-003).

They are validated into one immutable :class:`Snapshot`. Consumers read the
current snapshot and never hold on to its parts across a reload: the
registry serves it as the catalog, the llm adapter keeps its per-model tables.

The two sources fail differently. A broken built-in is a bug in giq — it
raises, and the test suite catches it. A broken operator file must not take
the service down: it is logged as an error and skipped, so the built-in of
that name (if any) keeps serving — the same rule the registry applies to a
broken config.yaml overlay.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from giq.recipes.schema import Recipe

logger = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).parent
SUFFIXES = (".yaml", ".yml")

# A recipe's name: unique across modalities (ADR-003).
Key = str


class RecipeError(ValueError):
    """A recipe file that cannot be used, with the file named."""

    def __init__(self, message: str, file: Path | None = None, detail: str | None = None):
        super().__init__(message)
        # The file and the complaint apart, for the API and the dashboard.
        self.file = file
        self.detail = detail if detail is not None else message


def _file_error(path: Path, detail: str) -> RecipeError:
    return RecipeError(f"{path}: {detail}", file=path, detail=detail)


@dataclass(frozen=True)
class LoadError:
    """An operator file left out of the snapshot, and why."""

    # None when the problem is not one file's (two files defining one recipe).
    file: str | None
    message: str

    def __str__(self) -> str:
        return f"{self.file}: {self.message}" if self.file else self.message


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a key given twice in one mapping.

    PyYAML otherwise keeps the last one, which is a silently ignored setting
    of exactly the kind strict validation exists to prevent.
    """


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _format(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        where = ".".join(str(p) for p in e["loc"]) or "(file)"
        parts.append(f"{where}: {e['msg']}")
    return "; ".join(parts)


def load_file(path: Path) -> Recipe:
    """Parse and validate one recipe file."""
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as e:
        raise _file_error(path, str(e)) from e
    if not isinstance(data, dict):
        raise _file_error(path, "expected a mapping at the top level")
    try:
        return Recipe.model_validate(data)
    except ValidationError as e:
        raise _file_error(path, _format(e)) from e


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix in SUFFIXES and p.is_file())


def _alias_clashes(recipes: Iterable[Recipe]) -> list[str]:
    """Names a client could send that would mean two recipes."""
    owner: dict[str, str] = {}
    clashes = []
    for recipe in recipes:
        for name in (recipe.name, *recipe.aliases):
            if name in owner and owner[name] != recipe.name:
                clashes.append(f"{name} names both {owner[name]} and {recipe.name}")
            owner.setdefault(name, recipe.name)
    return clashes


def _from_plugins(found: dict[Key, tuple[Recipe, Path]]) -> None:
    """Add the recipes installed plugins ship (ADR-004), into ``found``.

    A plugin's file that fails, or that takes a name already taken, is
    logged and left out: one plugin's mistake does not stop giq, and a
    plugin adds recipes, it never replaces one.
    """
    from giq import plugins

    for directory in plugins.recipe_dirs():
        for path in _files(directory):
            try:
                recipe = load_file(path)
            except RecipeError as e:
                logger.error(f"plugin recipe left out: {e}")
                continue
            taken = {n for r, _ in found.values() for n in (r.name, *r.aliases)}
            if clash := sorted({recipe.name, *recipe.aliases} & taken):
                logger.error(f"plugin recipe {path} left out: {', '.join(clash)} already taken")
                continue
            found[recipe.name] = (recipe, path)


@lru_cache(maxsize=1)
def builtin() -> Mapping[Key, tuple[Recipe, Path]]:
    """The shipped recipes: core's, then each plugin's.

    A problem in core's own raises: it is a bug, not a setting.
    """
    found: dict[Key, tuple[Recipe, Path]] = {}
    for path in _files(BUILTIN_DIR):
        recipe = load_file(path)
        if path.stem != recipe.name:
            raise _file_error(path, "a built-in is named <name>.yaml")
        found[recipe.name] = (recipe, path)
    _from_plugins(found)
    if clashes := _alias_clashes(recipe for recipe, _ in found.values()):
        raise RecipeError(f"built-in recipes: {'; '.join(clashes)}")
    from giq.weights import provenance_conflicts

    if conflicts := provenance_conflicts(recipe for recipe, _ in found.values()):
        raise RecipeError(f"built-in recipes: {'; '.join(conflicts)}")
    return MappingProxyType(found)


def _operator(directory: Path, errors: list[LoadError]) -> dict[Key, tuple[Recipe, Path]]:
    """The operator's recipes; files that fail are reported and left out."""
    if not directory.is_dir():
        return {}
    try:
        files = _files(directory)
    except OSError as e:
        errors.append(LoadError(str(directory), str(e)))
        return {}
    loaded: dict[Key, list[tuple[Recipe, Path]]] = {}
    for path in files:
        try:
            recipe = load_file(path)
        except RecipeError as e:
            errors.append(LoadError(str(e.file or path), e.detail))
            continue
        loaded.setdefault(recipe.name, []).append((recipe, path))
    found: dict[Key, tuple[Recipe, Path]] = {}
    for key, entries in loaded.items():
        if len(entries) > 1:
            # Neither file wins: which one would is an accident of sorting.
            where = ", ".join(str(p) for _, p in entries)
            errors.append(LoadError(None, f"{key} is defined more than once: {where}"))
            continue
        found[key] = entries[0]
    return found


@dataclass(frozen=True)
class Snapshot:
    """Every recipe giq serves, validated together; never mutated."""

    recipes: Mapping[Key, Recipe]
    # Which file each recipe came from.
    sources: Mapping[Key, Path]
    # Operator files that were left out, one message each.
    errors: tuple[str, ...] = ()
    # The same, with the file and the complaint apart.
    problems: tuple[LoadError, ...] = ()
    # The operator directory this snapshot read; None if none was given.
    operator_dir: Path | None = None

    def get(self, name: str) -> Recipe | None:
        """The recipe a client means by ``name``: its own name or an alias."""
        recipe = self.recipes.get(str(name))
        if recipe is not None:
            return recipe
        return next((r for r in self.recipes.values() if name in r.aliases), None)

    def serving(self, modality: str) -> list[Recipe]:
        """Every recipe that serves ``modality``, by name."""
        return sorted(
            (r for r in self.recipes.values() if r.serves(modality)), key=lambda r: r.name
        )


def load(operator_dir: Path | None = None) -> Snapshot:
    """Built-ins, then the operator directory over them."""
    if operator_dir is None:
        from giq.paths import recipes_dir

        operator_dir = recipes_dir()
    from giq.weights import provenance_conflicts

    merged = dict(builtin())
    errors: list[LoadError] = []
    for key, (recipe, path) in sorted(_operator(operator_dir, errors).items()):
        candidate = {**{k: i for k, (i, _) in merged.items()}, key: recipe}
        if clashes := _alias_clashes(candidate.values()):
            errors.append(LoadError(str(path), "; ".join(clashes)))
            continue
        if conflicts := provenance_conflicts(candidate.values()):
            errors.append(LoadError(str(path), "; ".join(conflicts)))
            continue
        if key in merged:
            logger.info(f"recipes: {recipe.name} from {path} replaces the built-in")
        else:
            logger.info(f"recipes: {recipe.name} from {path}")
        merged[key] = (recipe, path)
    for problem in errors:
        logger.error(f"recipes: ignoring {problem}")
    return Snapshot(
        recipes=MappingProxyType({k: i for k, (i, _) in merged.items()}),
        sources=MappingProxyType({k: p for k, (_, p) in merged.items()}),
        errors=tuple(str(e) for e in errors),
        problems=tuple(errors),
        operator_dir=operator_dir,
    )


def describe(snapshot: Snapshot) -> dict:
    """The operator's side of a snapshot, as ``GET /storage`` reports it.

    Which directory was read, which files in it are serving (and which of
    them replace a built-in), and which were left out and why — the last
    being the one thing an operator otherwise only learns from the log.
    """
    shipped = builtin()
    files = [
        {
            "file": str(path),
            "name": key,
            "modalities": list(snapshot.recipes[key].modalities),
            "replaces_builtin": key in shipped,
        }
        for key, path in sorted(snapshot.sources.items(), key=lambda kv: str(kv[1]))
        if key not in shipped or shipped[key][1] != path
    ]
    return {
        "dir": str(snapshot.operator_dir) if snapshot.operator_dir is not None else None,
        "builtin_dir": str(BUILTIN_DIR),
        "files": files,
        "overrides": [f["name"] for f in files if f["replaces_builtin"]],
        "errors": [{"file": p.file, "message": p.message} for p in snapshot.problems],
    }


# --- the current snapshot -----------------------------------------------------

_lock = threading.Lock()
_current: Snapshot | None = None
_subscribers: list[Callable[[Snapshot], None]] = []


def current() -> Snapshot:
    """The snapshot in force, loaded on first use."""
    global _current
    with _lock:
        if _current is None:
            _current = load()
        return _current


def reload(operator_dir: Path | None = None) -> Snapshot:
    """Load again and swap the snapshot in; subscribers see the new one.

    Built-ins are parsed once per process — they are part of the installed
    package — so a reload reads only the operator directory.
    """
    global _current
    snapshot = load(operator_dir)
    with _lock:
        _current = snapshot
        subscribers = list(_subscribers)
    for fn in subscribers:
        fn(snapshot)
    return snapshot


def subscribe(fn: Callable[[Snapshot], None]) -> None:
    """Call ``fn`` with the current snapshot now and after every reload."""
    with _lock:
        _subscribers.append(fn)
    fn(current())


__all__ = [
    "BUILTIN_DIR",
    "Recipe",
    "RecipeError",
    "LoadError",
    "Snapshot",
    "builtin",
    "current",
    "describe",
    "load",
    "load_file",
    "reload",
    "subscribe",
]
