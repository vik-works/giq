# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The ``giq`` command: ``giq serve`` (the default), ``giq init`` and
``giq prepare``.

``serve`` is imported only when serving, so ``giq init`` — run once by an
installer as the service user, before a config exists — never builds the
FastAPI app.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib import resources
from pathlib import Path

from giq import paths

TEMPLATE = "config.example.yaml"


def template_text() -> str:
    """The commented config template shipped in the package."""
    return resources.files("giq").joinpath("templates", TEMPLATE).read_text(encoding="utf-8")


def init_home(home: Path) -> list[str]:
    """Create a GIQ_HOME tree and its config.yaml. Never overwrites anything.

    The config is written first and then read, so a pre-existing config whose
    ``paths:`` block moves a directory elsewhere gets that directory created
    where it points. Returns a line per thing it created.
    """
    from giq.config import reload_config

    created: list[str] = []
    home = home.expanduser().absolute()
    os.environ["GIQ_HOME"] = str(home)
    if not home.is_dir():
        home.mkdir(parents=True)
        created.append(f"{home}/")

    config = paths.config_file().absolute()
    if not config.exists():
        config.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: a config that appears between the check and the
        # write is someone else's and stays theirs.
        with open(config, "x", encoding="utf-8") as f:
            f.write(template_text())
        created.append(str(config))
    reload_config()

    # ADR-003 renamed the instance files recipes. Moved rather than left for
    # the fallback in paths.recipes_dir: the systemd unit grants write access
    # to $GIQ_HOME/recipes by name, and a home that only has instances/ would
    # keep the service from starting.
    old, new = home / "instances", home / "recipes"
    if old.is_dir() and not new.exists():
        old.rename(new)
        created.append(f"{new}/ (was {old.name}/)")

    dirs = [paths.models_dir(), paths.recipes_dir(), paths.state_dir()]
    engines, cache = paths.engines_dir(), paths.cache_dir()
    dirs += [d for d in (engines, cache) if d is not None]
    if cache is not None:
        dirs.append(paths.hf_home())
    for d in dirs:
        if not d.is_dir():
            d.mkdir(parents=True)
            created.append(f"{d}/")
    return created


def _init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="giq init",
        description="Create the GIQ_HOME data tree (config.yaml, models/, recipes/, "
        "engines/, state/, cache/). Existing files and directories are left as they are.",
    )
    parser.add_argument(
        "--home",
        default=os.environ.get("GIQ_HOME"),
        help="Data root to create (default: $GIQ_HOME)",
    )
    args = parser.parse_args(argv)
    if not args.home:
        parser.error("no data root: pass --home PATH or set GIQ_HOME")
    created = init_home(Path(args.home))
    for line in created:
        print(f"created {line}")
    if not created:
        print(f"{paths.home()}: nothing to do, everything exists")
    for key, value in paths.resolved().items():
        print(f"  {key:9} {value if value is not None else '-'}")
    return 0


def _prepare(argv: list[str]) -> int:
    """``giq prepare <engine> …``: an engine's one-off build steps, if it has any."""
    from giq import plugins

    with_steps = sorted(n for n, e in plugins.engines().items() if e.prepare is not None)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: giq prepare <engine> [options]\n\n"
            "One-off build steps an engine needs before it serves. Engines with steps: "
            + (", ".join(with_steps) or "none installed"),
            file=sys.stderr if not argv else sys.stdout,
        )
        return 0 if argv else 2
    engine = plugins.engine(argv[0])
    if engine is None or engine.prepare is None:
        print(
            f"giq prepare: {argv[0]!r} has no build steps (engines with steps: "
            f"{', '.join(with_steps) or 'none installed'})",
            file=sys.stderr,
        )
        return 2
    return engine.prepare(argv[1:])


def main(argv: list[str] | None = None) -> None:
    """Dispatch ``giq [serve|init|prepare] …``; bare options mean ``serve``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "init":
        sys.exit(_init(args[1:]))
    if args and args[0] == "prepare":
        sys.exit(_prepare(args[1:]))
    if args and args[0] == "serve":
        args = args[1:]
    from giq.main import cli

    cli(args)
