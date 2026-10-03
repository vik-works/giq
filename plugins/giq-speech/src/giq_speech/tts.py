# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""TTS worker using Kokoro.

The pipeline loads in a child process so its CUDA context is released on stop.
Kokoro was the last CUDA worker still running in giq's own process, which meant
it never gave its VRAM back: the runner logged "Kokoro TTS stopped" and
``active_worker`` went None while ~968 MiB stayed held until giq restarted.
See ``giq.adapters._subprocess`` for the mechanism; child entry point is
``giq_speech._tts_child``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from giq.adapters._subprocess import SubprocessAdapter
from giq.models import JobResult
from giq.registry import vram_for
from giq.weights import hub_repo, recipe_of

logger = logging.getLogger(__name__)

# Voice mapping (OpenAI-style -> Kokoro voices)
VOICE_MAP = {
    "alloy": "af_heart",
    "echo": "am_adam",
    "fable": "bf_emma",
    "onyx": "bm_george",
    "nova": "af_bella",
    "shimmer": "af_sky",
}

# Available Kokoro voices
KOKORO_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]

# Kokoro emits 24 kHz audio.
KOKORO_SAMPLE_RATE = 24000


@dataclass
class TtsConfig:
    """Configuration for TTS worker."""

    model: str = "kokoro"
    lang_code: str = "a"  # American English


class TtsAdapter(SubprocessAdapter):
    """TTS worker — Kokoro runs in a child process for CUDA isolation."""

    child_module: ClassVar[str] = "giq_speech._tts_child"
    modality: ClassVar[str] = "tts"

    def __init__(self, config: TtsConfig, device: str | None = None):
        super().__init__(config, device)

    def child_args(self) -> list[str]:
        args = ["--lang-code", self.config.lang_code]
        # kokoro fetches its model and voices by repository (hf_hub_download
        # per file), so the recipe's hf: source is what it can take.
        recipe = recipe_of(self.config.model)
        if recipe and recipe.weights and (repo := hub_repo(recipe.weights.source)):
            args += ["--repo", repo]
        return args

    @property
    def estimated_vram_gb(self) -> float:
        # Registry, not a literal: this was hardcoded 0.5 while kokoro actually
        # holds ~0.95GB, so the VRAM gate admitted it into space it did not fit
        # in and it OOM'd mid-synthesis with the resident set loaded.
        return vram_for(self.config.model, default=1.0)

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[JobResult]:
        """Hydrate the raw result dicts back into ``JobResult`` recipes."""
        result_dicts = await super().run_batch(tasks, params)
        return [JobResult(**r) for r in result_dicts]
