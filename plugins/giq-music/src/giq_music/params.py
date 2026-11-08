# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step's recipe parameters: the schema an acestep recipe's `params` is
validated against. Registered with the engine (giq_music.plugin)."""

from __future__ import annotations

from pydantic import Field

from giq.recipes.schema import EngineParams


class AceStepParams(EngineParams):
    """What an acestep recipe may set. See docs/engines.md."""

    # The folders of weights.path the child loads: ACE-Step's own checkpoint
    # layout, so the names are ACE-Step's (they pick the model code it syncs).
    dit: str = Field(default="acestep-v15-turbo", pattern=r"^acestep-v15-[a-z0-9-]+$")
    # The 5Hz planner LM; null loads none, and every task renders unplanned.
    lm: str | None = Field(default="acestep-5Hz-lm-1.7B", pattern=r"^acestep-5Hz-lm-[0-9.]+B$")
    # Diffusion steps: 8 for the turbo DiTs, 32-64 for base and sft.
    steps: int = Field(default=8, ge=1, le=200)


class StableAudioParams(EngineParams):
    """What a stable-audio recipe may set. See docs/engines.md."""

    # Sampling steps: 8 for the post-trained models, ~50 for the -base ones.
    steps: int = Field(default=8, ge=1, le=200)
    # Classifier-free guidance; the post-trained models ignore it.
    cfg_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    # The longest track a task may ask for: the model's own limit (120 s for
    # the small models, 380 s for medium), or less to bound a job's VRAM.
    max_seconds: float = Field(default=120.0, ge=1.0, le=380.0)
    # The length of a task that names none.
    default_seconds: float = Field(default=30.0, ge=1.0, le=380.0)
