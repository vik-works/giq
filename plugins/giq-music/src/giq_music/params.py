# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step's recipe parameters: the schema an acestep recipe's `params` is
validated against. Registered with the engine (giq_music.plugin)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from giq.recipes.schema import EngineParams


class AceStepParams(EngineParams):
    """What an acestep recipe may set. See docs/engines.md."""

    # The folders of weights.path the child loads: ACE-Step's own checkpoint
    # layout, so the names are ACE-Step's (they pick the model code it syncs).
    dit: str = Field(default="acestep-v15-turbo", pattern=r"^acestep-v15-[a-z0-9-]+$")
    # The 5Hz planner LM; null loads none, and every task renders unplanned.
    lm: str | None = Field(default="acestep-5Hz-lm-1.7B", pattern=r"^acestep-5Hz-lm-[0-9.]+B$")
    # Diffusion steps: 8 for the turbo DiTs (ACE-Step caps them there), 50
    # for base and sft.
    steps: int = Field(default=8, ge=1, le=200)
    # Classifier-free guidance: how strictly base and sft follow the prompt.
    # The turbo DiTs are distilled without it, and ACE-Step sets theirs to 1.
    guidance: float = Field(default=7.0, ge=1.0, le=20.0)
    # What the DiT does with a task's source audio. Every DiT covers; adding
    # a track (lego) and filling in the missing ones (complete) are the base
    # models'.
    tasks: tuple[Literal["cover", "lego", "complete"], ...] = ("cover",)
