# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-rig: a 3D mesh skinned to a skeleton with SkinTokens (ADR-004).

SkinTokens on its own interpreter (envs/skintokens: its torch, flash-attn
and Blender's bpy module) in a child process; the mesh2rig modality; the
built-in recipe. Not part of giq-defaults: the engine's code is a checkout
`giq prepare skintokens` fetches, and part of it is GPL-3.0 (see
docs/engines.md), which the operator fetches for themselves rather than
giq shipping it.
"""

from __future__ import annotations

from pathlib import Path

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin
from giq_rig.params import SkinTokensParams

__version__ = "0.12.0"


def _rig(ctx: AdapterContext):
    from giq_rig.adapter import RigAdapter, RigConfig

    return RigAdapter(config=RigConfig(model=ctx.recipe), device=ctx.device)


def _start_budget(_params) -> float:
    from giq_rig.adapter import START_BUDGET_SECONDS

    return START_BUDGET_SECONDS


def _prepare(argv: list[str]) -> int:
    from giq_rig.prepare import main

    return main(argv)


def _rig_plugin() -> Plugin:
    return Plugin(
        name="giq-rig",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "skintokens",
                Binary(
                    lambda: env_python("skintokens"),
                    env="GIQ_SKINTOKENS_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; print('torch', m.version('torch'), "
                        "'transformers', m.version('transformers'), 'bpy', m.version('bpy'))",
                    ),
                    detail="python + torch + flash-attn + bpy, for the mesh2rig child "
                    "(envs/skintokens, then `giq prepare skintokens`)",
                ),
                params=SkinTokensParams,
                prepare=_prepare,
                start_budget=_start_budget,
            ),
        ),
        modalities=(
            # A rig is seconds once the model is loaded; a cold start loads
            # 1.6 GB of checkpoints first.
            Modality(
                "mesh2rig",
                label="Mesh to rig",
                icon="cube",
                job_timeout=600.0,
                # The transformer's architecture, from the repository of the
                # language model it was shaped after (its config only).
                parts=frozenset({"llm"}),
                payload_keys=("glb_b64",),
            ),
        ),
        adapters={
            ("skintokens", "mesh2rig"): _rig,
        },
        recipes=Path(__file__).parent / "recipes",
    )


plugin = _rig_plugin()
