# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-depth: monocular depth maps (ADR-004).

Depth Anything V2 on the transformers engine in a child process; POST
/depth; the built-in depth recipe."""

from __future__ import annotations

from pathlib import Path

from giq.plugin import API_VERSION, AdapterContext, Modality, Plugin

__version__ = "0.6.0"


def _depth(ctx: AdapterContext):
    from giq_depth.adapter import DepthAdapter, DepthConfig

    return DepthAdapter(config=DepthConfig(model=ctx.recipe), device=ctx.device)


def _depth_plugin() -> Plugin:
    return Plugin(
        name="giq-depth",
        api_version=API_VERSION,
        version=__version__,
        modalities=(
            Modality(
                "depth",
                label="Depth",
                icon="mountains",
                payload_keys=("image_b64",),
                weights_root_env="GIQ_DEPTH_MODELS_DIR",
            ),
        ),
        adapters={("transformers", "depth"): _depth},
        recipes=Path(__file__).parent / "recipes",
        routers=("giq_depth.api:router",),
    )


plugin = _depth_plugin()
