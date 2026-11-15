# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Pixal3D's recipe parameters: the schema a pixal3d recipe's `params` is
validated against. Registered with the engine (giq_3d.plugin)."""

from __future__ import annotations

from typing import Literal

from giq.recipes.schema import EngineParams


class Pixal3DParams(EngineParams):
    """What a pixal3d recipe may set. See docs/engines.md."""

    # The voxel grid the cascade ends at. 1536 adds about 4% faces for 3.5
    # times the generation time and 10 GB more VRAM, so it is a recipe's
    # choice, not a request's: the figure in `vram` belongs to one of them.
    resolution: Literal[1024, 1536] = 1024
    # Keep the flow models on the CPU and move each to the card for its
    # stage. Measured on an RTX 5090 at 1024 this is the faster mode as well
    # as the smaller one (21 s against 84 s, 15.9 GB against 31.4 GB): with
    # everything resident the card is full and the allocator thrashes. Off
    # only for a card with room for all of it at once.
    low_vram: bool = True
