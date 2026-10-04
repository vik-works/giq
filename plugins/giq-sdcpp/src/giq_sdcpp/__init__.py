# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-sdcpp: image generation and editing with stable-diffusion.cpp (ADR-004).

sd-server is a binary, so this package brings no Python dependency of its
own: the engine, the text2image and image_edit modalities, their sandbox
panels and the built-in image recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from giq.paths import engine_binary
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin, SmokeTest
from giq.plugins import curated_ui

__version__ = "0.6.1"

_IMAGE_PARTS = frozenset({"diffusion", "text_encoder", "vae", "lora"})


def _sdcpp(ctx: AdapterContext):
    from giq_sdcpp.adapter import SdCppAdapter, SdCppConfig

    return SdCppAdapter(config=SdCppConfig(model=ctx.recipe, device=ctx.device))


def _sdcpp_start_budget(recipe: Any) -> float:
    from giq_sdcpp.adapter import READY_TIMEOUT_SECONDS

    return READY_TIMEOUT_SECONDS


def _image_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[{"id": "smoke-image", "prompt": "a tiny test pattern, colorful geometric shapes"}],
        timeout=700.0,
        summary=lambda result: {"seed": result.get("seed"), "image_b64": result.get("image_b64")},
        evicts=True,
    )


def _sdcpp_plugin() -> Plugin:
    return Plugin(
        name="giq-sdcpp",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "sd.cpp",
                # sd-server has no --version; its banner carries the commit
                # and it prints that for --help too.
                Binary(
                    lambda: engine_binary("sd.cpp", "sd-server"),
                    env="GIQ_SDCPP_BINARY",
                    version_args=("--help",),
                    detail="sd-server (stable-diffusion.cpp)",
                ),
                aliases=("sdcpp",),
                start_budget=_sdcpp_start_budget,
                stale_pattern=r"sd-server .*--listen-port {port}\b",
            ),
        ),
        modalities=(
            Modality(
                "text2image",
                label="Text to image",
                icon="image",
                parts=_IMAGE_PARTS,
                smoke_test=_image_smoke,
            ),
            Modality(
                "image_edit",
                label="Image edit",
                icon="paint-brush",
                parts=_IMAGE_PARTS,
                payload_keys=("reference_image_b64",),
            ),
        ),
        # One sd-server serves both: flux_klein renders and edits without a reload.
        adapters={("sd.cpp", "text2image"): _sdcpp, ("sd.cpp", "image_edit"): _sdcpp},
        recipes=Path(__file__).parent / "recipes",
        ui=curated_ui("giq-sdcpp"),
    )


plugin = _sdcpp_plugin()
