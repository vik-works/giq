# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The plugin registry: core's own registrations plus every installed plugin.

Core's built-ins (:mod:`giq.builtins`) register first, then each plugin found
under the ``giq.plugins`` entry-point group, in name order. A plugin is
refused whole, never half-loaded, when:

- it was written for another :data:`giq.plugin.API_VERSION`;
- its entry point does not load, or does not name a :class:`Plugin`;
- it registers an engine, alias or modality name that is already taken
  (plugins add, they never override; ADR-004 D5);
- an adapter names an engine or modality nobody registered.

A refused plugin is logged and listed by :func:`status` with the reason; one
broken plugin does not stop giq or the others. The registry is built on first
use and kept; :func:`reset` throws it away (tests).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from giq.plugin import API_VERSION, SELF, AdapterFactory, Engine, Modality, Plugin

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "giq.plugins"


@dataclass(frozen=True)
class PluginStatus:
    """One plugin as /status reports it."""

    name: str
    version: str
    # "builtin", or the distribution that installed it.
    source: str
    loaded: bool
    reason: str | None = None
    engines: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()


@dataclass
class _Registry:
    engines: dict[str, Engine] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    modalities: dict[str, Modality] = field(default_factory=dict)
    adapters: dict[tuple[str, str], AdapterFactory] = field(default_factory=dict)
    recipe_dirs: list[Path] = field(default_factory=list)
    plugins: list[PluginStatus] = field(default_factory=list)
    loaded: list[Plugin] = field(default_factory=list)

    def refusal(self, plugin: Plugin) -> str | None:
        """Why ``plugin`` cannot join this registry, or None."""
        if plugin.name in _excluded:
            return _excluded[plugin.name]
        if plugin.api_version != API_VERSION:
            return f"written for plugin API {plugin.api_version}, giq speaks {API_VERSION}"
        taken = set(self.engines) | set(self.aliases)
        for engine in plugin.engines:
            for name in (engine.name, *engine.aliases):
                if name in taken or name == SELF:
                    return f"engine name {name!r} is already registered"
                taken.add(name)
        modalities = set(self.modalities)
        for modality in plugin.modalities:
            if modality.name in modalities:
                return f"modality {modality.name!r} is already registered"
            modalities.add(modality.name)
        engines = set(self.engines) | {e.name for e in plugin.engines}
        for engine, modality in plugin.adapters:
            if (engine, modality) in self.adapters:
                return f"an adapter for {engine} / {modality} is already registered"
            if engine not in engines:
                return f"its adapter names engine {engine!r}, which nobody registers"
            if modality not in modalities:
                return f"its adapter names modality {modality!r}, which nobody registers"
        return None

    def add(self, plugin: Plugin, source: str) -> None:
        if reason := self.refusal(plugin):
            logger.error(f"plugin {plugin.name} ({source}) not loaded: {reason}")
            self.plugins.append(PluginStatus(plugin.name, plugin.version, source, False, reason))
            return
        for engine in plugin.engines:
            self.engines[engine.name] = engine
            for alias in engine.aliases:
                self.aliases[alias] = engine.name
        for modality in plugin.modalities:
            self.modalities[modality.name] = modality
        self.adapters.update(plugin.adapters)
        if plugin.recipes is not None:
            self.recipe_dirs.append(plugin.recipes)
        self.loaded.append(plugin)
        self.plugins.append(
            PluginStatus(
                plugin.name,
                plugin.version,
                source,
                True,
                engines=tuple(e.name for e in plugin.engines),
                modalities=tuple(m.name for m in plugin.modalities),
            )
        )
        if source != "builtin":
            logger.info(
                f"plugin {plugin.name} {plugin.version} ({source}): "
                f"engines {', '.join(e.name for e in plugin.engines) or '-'}, "
                f"modalities {', '.join(m.name for m in plugin.modalities) or '-'}"
            )


def _installed() -> list[tuple[str, str, Any]]:
    """(entry point name, distribution, loaded object or the error) per entry point."""
    from importlib.metadata import entry_points

    found = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):
        dist = ep.dist.name if ep.dist is not None else ep.value
        try:
            found.append((ep.name, dist, ep.load()))
        except Exception as e:  # a plugin that cannot import must not stop giq
            found.append((ep.name, dist, e))
    return found


def _build() -> _Registry:
    from giq.builtins import PLUGINS

    registry = _Registry()
    for plugin in PLUGINS:
        registry.add(plugin, "builtin")
    for name, dist, obj in _installed():
        if isinstance(obj, Plugin):
            registry.add(obj, dist)
            continue
        reason = (
            f"its entry point does not load: {type(obj).__name__}: {obj}"
            if isinstance(obj, Exception)
            else f"its entry point names a {type(obj).__name__}, not a giq.plugin.Plugin"
        )
        logger.error(f"plugin {name} ({dist}) not loaded: {reason}")
        registry.plugins.append(PluginStatus(name, "", dist, False, reason))
    return registry


_lock = threading.RLock()
_registry: _Registry | None = None
# Plugins refused after the registry was built (a route clash, found when
# routes are mounted), by name, with the reason; the next build leaves them
# out whole.
_excluded: dict[str, str] = {}


def _get() -> _Registry:
    global _registry
    with _lock:
        if _registry is None:
            _registry = _build()
        return _registry


def reset() -> None:
    """Forget the registry and any refusals; the next lookup builds it again."""
    global _registry
    with _lock:
        _registry = None
        _excluded.clear()


def _import(path: str) -> Any:
    """``"module:attribute"`` -> the object."""
    import importlib

    module, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module), attribute)


def mount(app: Any) -> None:
    """Mount every loaded plugin's routers on ``app``, core's first.

    A plugin whose routers do not import, or that would serve a method and
    path already served, is refused whole (its routes are not mounted, and
    the registry is rebuilt without its engines and modalities): plugins add
    routes, they never take one over.
    """
    global _registry
    served: set[tuple[str, str]] = {
        (method, route.path)
        for route in app.routes
        for method in (getattr(route, "methods", None) or ())
    }
    refused = False
    for plugin in list(_get().loaded):
        try:
            routers = [_import(path) for path in plugin.routers]
        except Exception as e:
            reason = f"its routes do not import: {type(e).__name__}: {e}"
        else:
            routes = {
                (method, route.path)
                for router in routers
                for route in router.routes
                for method in (getattr(route, "methods", None) or ())
            }
            if not (clash := sorted(routes & served)):
                for router in routers:
                    app.include_router(router)
                served |= routes
                continue
            reason = "it would take over routes already served: " + ", ".join(
                f"{m} {p}" for m, p in clash[:3]
            )
        logger.error(f"plugin {plugin.name} not loaded: {reason}")
        with _lock:
            _excluded[plugin.name] = reason
        refused = True
    if refused:
        with _lock:
            _registry = None


# --- lookups -----------------------------------------------------------------


def engines() -> dict[str, Engine]:
    return dict(_get().engines)


def engine(name: str) -> Engine | None:
    """The engine called ``name`` (or one of its old spellings)."""
    reg = _get()
    return reg.engines.get(reg.aliases.get(name, name))


def aliases() -> dict[str, str]:
    """Old engine spellings -> the engine's one name."""
    return dict(_get().aliases)


def modalities() -> dict[str, Modality]:
    return dict(_get().modalities)


def modality(name: str) -> Modality | None:
    return _get().modalities.get(str(name))


def adapter(engine_name: str, modality_name: str) -> AdapterFactory | None:
    """The factory for recipes of ``engine_name`` serving ``modality_name``."""
    return _get().adapters.get((engine_name, str(modality_name)))


def engines_for(modality_name: str) -> frozenset[str]:
    """The engines some adapter runs ``modality_name`` on."""
    return frozenset(e for e, m in _get().adapters if m == str(modality_name))


def recipe_dirs() -> list[Path]:
    """Plugins' built-in recipe directories, in registration order."""
    return list(_get().recipe_dirs)


def status() -> list[PluginStatus]:
    return list(_get().plugins)


def loaded() -> list[Plugin]:
    """The plugins that joined, in registration order."""
    return list(_get().loaded)


# --- the curated index (ADR-005 D5) --------------------------------------------


def curated() -> list[dict[str, Any]]:
    """The curated plugins as ``giq/plugins.json`` lists them, data only."""
    import json
    from importlib.resources import files

    return json.loads(files("giq").joinpath("plugins.json").read_text())["plugins"]


def install_command(package: str) -> str:
    """How to install ``package`` into the interpreter giq runs on."""
    import sys

    return f"uv pip install --python {sys.executable} {package}"


def catalog() -> list[dict[str, Any]]:
    """Installed plugins and the curated ones that are not, for /plugins.

    A curated plugin that is installed is listed once, as installed; one
    that is not carries its install command. Plugins are installed by the
    operator, from the command line: installing one changes giq's own
    environment and needs a restart (ADR-005 D5).
    """
    from dataclasses import asdict

    installed = {s.name: s for s in status()}
    out = []
    for entry in curated():
        state = installed.pop(entry["name"], None)
        out.append(
            {
                **entry,
                "curated": True,
                "installed": state is not None and state.loaded,
                "status": asdict(state) if state is not None else None,
                "install": None if state is not None else install_command(entry["package"]),
            }
        )
    for state in installed.values():
        out.append(
            {
                "name": state.name,
                "package": state.source,
                "summary": "",
                "engines": list(state.engines),
                "modalities": list(state.modalities),
                "recipes": [],
                "needs": "",
                "curated": False,
                "installed": state.loaded,
                "status": asdict(state),
                "install": None,
            }
        )
    return out


# --- dashboard UI (ADR-004 D6) -------------------------------------------------


def curated_ui(name: str) -> Path | None:
    """A curated plugin's dashboard UI. The curated plugins are released
    with core (ADR-004 D7), and their panels are built with the dashboard
    (frontend/plugins/<name>, `make ui`) into its tree, so the UI tarball
    and core's wheel carry them. None without a build: the plugin serves,
    its sandbox panels are absent. A third-party plugin ships its own."""
    ui = Path(__file__).parent / "static" / "ui" / "plugins" / name
    return ui if (ui / "manifest.json").is_file() else None


def ui_dir(name: str) -> Path | None:
    """The UI directory of loaded plugin ``name``, if it ships one."""
    plugin = next((p for p in loaded() if p.name == name), None)
    return plugin.ui if plugin is not None and plugin.ui is not None else None


def ui_manifests() -> list[dict[str, Any]]:
    """Every loaded plugin's UI, as /capabilities lists it.

    A manifest that is missing, unreadable or written for another
    API_VERSION leaves that plugin without a UI, logged; its engines,
    modalities and routes still serve.
    """
    import json

    out = []
    for plugin in loaded():
        if plugin.ui is None:
            continue
        try:
            manifest = json.loads((plugin.ui / "manifest.json").read_text())
        except (OSError, ValueError) as e:
            logger.error(f"plugin {plugin.name}: its UI manifest does not load: {e}")
            continue
        if manifest.get("api_version") != API_VERSION:
            logger.error(
                f"plugin {plugin.name}: UI written for API {manifest.get('api_version')}, "
                f"giq speaks {API_VERSION}; its panels are left out"
            )
            continue
        out.append({"plugin": plugin.name, "base": f"/plugins/{plugin.name}/ui/", **manifest})
    return out
