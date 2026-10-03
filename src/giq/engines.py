# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Declared inference engines: which binary runs a model, and which build it is.

giq knows everything about model *files* — path, VRAM, measured or estimated.
This module does the same for the runtimes that execute them, because two
failures follow from leaving that implicit.

An implicit binary drifts. A bare ``llama-server`` resolved through PATH for
some models and an absolute path for others means two builds can serve side
by side; they may be byte-identical today, and the first rebuild of either
splits them silently.

And an unrecorded build is invisible. A source tree can be damaged, or pulled
and never rebuilt, while the installed binary keeps working — nothing in the
serving path would tell you which build is actually running.

So: one declared path per engine, and a version probe whose answer is
surfaced. The built-in default for a native engine is its build under the
engines directory (``$GIQ_HOME/engines/<engine>/bin/<binary>``) when one is
there, else whatever PATH resolves *once*, when the table is built;
config.yaml's ``engines:`` or an env var pins it explicitly. A missing binary
fails loudly at spawn rather than resolving to a stranger. This is
deliberately *identity*, not management — giq does not fetch, build or
upgrade engines. If that is ever wanted, this is its prerequisite.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from dataclasses import dataclass

from giq import plugins

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSpec:
    """One runtime giq executes models with."""

    name: str
    binary: str
    # How to ask it what it is. sd-server has no --version; its banner line
    # carries the commit and it prints that for --help too.
    version_args: tuple[str, ...] = ("--version",)
    detail: str = ""


def _builtin() -> tuple[EngineSpec, ...]:
    """Every registered engine that runs as an executable of its own.

    Resolved when the table is built, not at import, so env and PATH set
    after import still count.
    """
    specs = []
    for engine in plugins.engines().values():
        if engine.binary is None:
            continue
        default = engine.binary.default
        specs.append(
            EngineSpec(
                engine.name,
                default if isinstance(default, str) else default(),
                engine.binary.version_args,
                engine.binary.detail or engine.detail,
            )
        )
    return tuple(specs)


def runtime_of(engine: str) -> str:
    """The executable a recipe's `engine:` runs on: its own name, or SELF for
    the engines that run in giq's interpreter. An unknown name is itself."""
    spec = plugins.engine(engine)
    return spec.runtime if spec is not None else engine


_warned_aliases: set[tuple[str, str]] = set()


def canonical_engine(name: str, where: str) -> str:
    """The engine's one name; an old spelling is translated, and said once per place.

    There is one name per engine — the one /engines, the catalog and the
    dashboard show — and recipe files and config.yaml all use it. Old
    spellings come from the registry: `sdcpp` is what config.yaml's
    image_models said, `transformers-4.57` the interpreter Unlimited-OCR ran
    on until it moved to giq's own transformers.
    """
    canonical = plugins.aliases().get(name)
    if canonical is None:
        return name
    if (name, where) not in _warned_aliases:
        _warned_aliases.add((name, where))
        logger.warning(f"{where}: engine {name!r} is deprecated, write {canonical!r}")
    return canonical


def _env_override(name: str) -> str | None:
    """The environment variable that outranks config.yaml for engine ``name``."""
    spec = plugins.engine(name)
    return spec.binary.env if spec is not None and spec.binary is not None else None


_lock = threading.Lock()
_engines: dict[str, EngineSpec] | None = None
_versions: dict[str, dict[str, object]] = {}


def _build() -> dict[str, EngineSpec]:
    engines = {spec.name: spec for spec in _builtin()}
    try:
        from giq.config import get_config

        for name, binary in (get_config().engines or {}).items():
            name = canonical_engine(name, "config.yaml engines")
            base = engines.get(name)
            engines[name] = (
                EngineSpec(name, binary, base.version_args, base.detail)
                if base
                else EngineSpec(name, binary)
            )
    except Exception as e:  # config problems must not take the service down
        logger.warning(f"engines: config overlay failed, using built-ins: {e}")
    for name in list(engines):
        env = _env_override(name)
        override = os.environ.get(env) if env else None
        if override:
            engines[name] = EngineSpec(
                name, override, engines[name].version_args, engines[name].detail
            )
    return engines


def all_engines() -> dict[str, EngineSpec]:
    global _engines
    with _lock:
        if _engines is None:
            _engines = _build()
        return _engines


def reload_engines() -> dict[str, EngineSpec]:
    """Rebuild from config (tests, and after a config reload)."""
    global _engines, _versions
    with _lock:
        _engines = None
        _versions = {}
    return all_engines()


def binary_for(name: str) -> str:
    """The declared executable for an engine. Raises only if it is undeclared.

    Deliberately does NOT check the file exists: this is called while building
    a worker's config, which happens in tests and on machines that have no
    engines installed. Existence is checked at spawn by ``require_binary``,
    which is where a missing binary is actually a problem — and where the
    error can say what to do about it.
    """
    spec = all_engines().get(name)
    if spec is None:
        raise ValueError(f"unknown engine {name!r} (declared: {', '.join(sorted(all_engines()))})")
    return spec.binary


def require_binary(name: str) -> str:
    """The executable, verified present. Raises loudly if it is not.

    Loud on purpose: what this replaces was a bare name resolving quietly
    through PATH to a different build than the one intended.
    """
    binary = binary_for(name)
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"engine {name!r} binary not found at {binary} — declare the right path under "
            f"`engines:` in config.yaml, or set {_env_override(name) or 'the override'}"
        )
    return binary


def probe(name: str, refresh: bool = False) -> dict[str, object]:
    """Ask an engine what build it is. Cached; never raises.

    Returns {name, binary, present, version, error}. `version` is the first
    line of output that mentions a version or commit, which is what both
    llama-server and sd-server put there.
    """
    with _lock:
        if not refresh and name in _versions:
            return _versions[name]
    spec = all_engines().get(name)
    if spec is None:
        return {
            "name": name,
            "binary": None,
            "present": False,
            "version": None,
            "error": "not declared",
        }
    out: dict[str, object] = {
        "name": name,
        "binary": spec.binary,
        "detail": spec.detail,
        "present": os.path.exists(spec.binary),
        "version": None,
        "error": None,
    }
    if out["present"]:
        try:
            result = subprocess.run(
                [spec.binary, *spec.version_args], capture_output=True, text=True, timeout=20
            )
            lines = [
                ln.strip()
                for ln in (result.stdout + "\n" + result.stderr).splitlines()
                if ln.strip()
            ]
            for line in lines:
                low = line.lower()
                if ("version" in low or "commit" in low) and "usage" not in low:
                    out["version"] = line
                    break
            else:
                # `python -V` says "Python 3.11.14" and matches neither word;
                # the first line of output identifies it well enough.
                out["version"] = lines[0] if lines else None
        except Exception as e:
            out["error"] = str(e)
    else:
        out["error"] = "binary not found"
    with _lock:
        _versions[name] = out
    return out


def probe_all(refresh: bool = False) -> list[dict[str, object]]:
    """Every declared engine's identity, for /engines and the dashboard."""
    return [probe(name, refresh) for name in sorted(all_engines())]
