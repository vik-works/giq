# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""sd.cpp's recipe parameters: the schema an sd.cpp recipe's `params` is
validated against. Registered with the engine (giq_sdcpp.plugin)."""

from __future__ import annotations

from pydantic import Field

from giq.recipes.schema import EngineParams


class SdCppParams(EngineParams):
    """What an sd.cpp recipe may set. See docs/engines.md."""

    # The largest render a request may ask for, in pixels. Peak VRAM grows
    # with the render's area, so the budget belongs to a recipe and is
    # measured with its VRAM figure. Unset = the configured image_generation
    # area: a request may change the aspect, not grow past what was measured.
    max_render_pixels: int | None = Field(default=None, ge=256 * 256)
