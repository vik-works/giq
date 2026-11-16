# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Pixal3D's recipe parameters: the schema a pixal3d recipe's `params` is
validated against. Registered with the engine (giq_3d.plugin)."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from giq.recipes.schema import EngineParams


class Pixal3DParams(EngineParams):
    """What a pixal3d recipe may set. See docs/engines.md."""

    # Which model generates. `pixal3d` keeps the picture: the mesh is posed
    # as it was taken, its surface close to it. `trellis2`, the backbone
    # Pixal3D is built on, stands the object upright in a frame of its own
    # and takes more liberties with the surface. They share the engine's
    # environment and extensions, and nothing else.
    pipeline: Literal["pixal3d", "trellis2"] = "pixal3d"

    # The voxel grid the cascade ends at. 1536 adds about 4% faces for 3.5
    # times the generation time and 10 GB more VRAM, so it is a recipe's
    # choice, not a request's: the figure in `vram` belongs to one of them.
    resolution: Literal[1024, 1536] = 1024
    # What a task brings. `single`: one picture, and a mesh posed the way it
    # was taken. `multi`: four views a quarter turn apart, and a mesh upright
    # in their frame. They are different checkpoints of the one repository,
    # each 23 GB in RAM, so a recipe loads one set and a second recipe the
    # other.
    views: Literal["single", "multi"] = "single"
    # Keep the flow models on the CPU and move each to the card for its
    # stage. Measured on an RTX 5090 at 1024 this is the faster mode as well
    # as the smaller one (21 s against 84 s, 15.9 GB against 31.4 GB): with
    # everything resident the card is full and the allocator thrashes. Off
    # only for a card with room for all of it at once.
    low_vram: bool = True
    # Load the recipe's `segmenter` part beside the generator, so a task may
    # ask for `parts`. The segmenter is TRELLIS.2's texture model taught to
    # paint parts, and runs on its decoders: it is a trellis2 recipe's.
    segment: bool = False

    @model_validator(mode="after")
    def _one_picture_for_trellis2(self) -> Pixal3DParams:
        if self.pipeline == "trellis2" and self.views == "multi":
            raise ValueError("trellis2 takes one picture: `views: multi` is pixal3d's")
        if self.segment and self.pipeline != "trellis2":
            raise ValueError("`segment` is trellis2's: the segmenter runs on its decoders")
        return self
