# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""``giq prepare skintokens``: what the engine needs that `uv sync` cannot give.

After ``cd envs/skintokens && uv sync`` the environment has torch,
flash-attn, Blender's bpy module and the rest, all from an index or a
pinned wheel. SkinTokens itself is not a package: this clones it at a
pinned commit beside the environment (or into ``GIQ_SKINTOKENS_SOURCE``),
skipped when the checkout is already there, and then checks that the
environment imports what the child does, without touching a card.

The checkout is fetched here rather than shipped with giq because part of
it is not under SkinTokens' own MIT licence: it carries a copy of
Michelangelo's shape encoder, whose upstream licence is GPL-3.0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SKINTOKENS = (
    "https://github.com/VAST-AI-Research/SkinTokens.git",
    "273b691d35989d71cd17ff2895fdc735097b92d1",
)
# A file only a SkinTokens checkout has at its root.
MARKER = "demo.py"
# What the child imports that a broken environment would fail on first.
IMPORTS = "import torch, flash_attn, transformers, lightning, open3d, bpy"


def _say(message: str) -> None:
    print(f"prepare skintokens: {message}", flush=True)


def _run(command: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def _head(source: Path) -> str | None:
    probe = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


def checkout(source: Path) -> bool:
    """SkinTokens at the pinned commit in ``source``; False when a folder
    that is not a checkout is in the way."""
    url, commit = SKINTOKENS
    if (source / MARKER).exists():
        head = _head(source)
        if head != commit:
            # An operator's own checkout is theirs: said, not changed.
            _say(f"{source} is at {head or 'an unknown commit'}, not the pinned {commit[:10]}")
        return True
    if source.exists() and any(source.iterdir()):
        _say(f"{source} is in the way and is not a SkinTokens checkout: move it aside")
        return False
    _say(f"cloning SkinTokens at {commit[:10]} into {source}")
    _run(["git", "clone", "--quiet", url, str(source)])
    _run(["git", "checkout", "--quiet", commit], cwd=str(source))
    return True


def main(argv: list[str]) -> int:
    from giq.engines import require_binary
    from giq_rig.adapter import source_dir

    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    try:
        python = require_binary("skintokens")
    except Exception as e:  # noqa: BLE001 — say how to get the interpreter
        _say(f"{e}\n  set it up first: cd envs/skintokens && uv sync")
        return 1

    if not checkout(Path(source_dir())):
        return 1

    _say("checking that the environment imports what the child does")
    # No card is needed to import any of it, so none is shown.
    try:
        _run([python, "-c", IMPORTS], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    except subprocess.CalledProcessError:
        _say("the environment is incomplete: cd envs/skintokens && uv sync")
        return 1

    _say("done; the recipe's weights come with `giq add skintokens`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
