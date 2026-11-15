# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""sd.cpp's recipe parameters: the schema an sd.cpp recipe's `params` is
validated against. Registered with the engine (giq_sdcpp.plugin)."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from giq.recipes.schema import EngineParams


class SdCppParams(EngineParams):
    """What an sd.cpp recipe may set. See docs/engines.md."""

    # The largest render a request may ask for, in pixels. Peak VRAM grows
    # with the render's area, so the budget belongs to a recipe and is
    # measured with its VRAM figure. Unset = the configured image_generation
    # area: a request may change the aspect, not grow past what was measured.
    max_render_pixels: int | None = Field(default=None, ge=256 * 256)

    # Sampling, where the global image_generation defaults don't suit the
    # model: Qwen-Image-Edit is distilled for neither 8 steps nor CFG 1
    # unless a Lightning LoRA rides along. Unset = the configured default.
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg_scale: float | None = Field(default=None, ge=1.0, le=20.0)
    # The flow models' timestep shift; unset = sd.cpp's per-model default.
    flow_shift: float | None = Field(default=None, gt=0.0, le=20.0)
    # sd-server's --model-args, key=value each: switches a model needs at
    # load, such as Qwen-Image-Edit-2511's qwen_image_zero_cond_t.
    model_args: tuple[str, ...] = ()
    # How strongly weights.parts.lora applies to every render.
    lora_strength: float = Field(default=1.0, gt=0.0, le=2.0)
    # How strongly weights.parts.lora_camera applies to a task that asks for
    # it. 0.9 is what the multiple-angles LoRA's card recommends.
    lora_camera_strength: float = Field(default=0.9, gt=0.0, le=2.0)

    @field_validator("model_args")
    @classmethod
    def _key_value(cls, args: tuple[str, ...]) -> tuple[str, ...]:
        # One flag value carries them all, comma-joined: a comma or a space
        # in one would split it into arguments the recipe never named.
        for arg in args:
            if not _MODEL_ARG.fullmatch(arg):
                raise ValueError(f"model_args entry {arg!r} is not key=value")
        return args


_MODEL_ARG = re.compile(r"[a-z][a-z0-9_]*=[A-Za-z0-9_.+-]+")
