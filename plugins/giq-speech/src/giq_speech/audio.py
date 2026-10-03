# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Audio (ASR + diarization) and speaker-embedding workers.

Both run their models in child processes under giq's own interpreter (no
whisperx, whose torch~=2.8 pin would force a second venv; see _audio_child).
Both are resident-set members: loaded whenever the GPU is idle, evicted
(child killed → CUDA context gone, true-zero VRAM floor) when image jobs need
the space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from giq.adapters._subprocess import SubprocessAdapter
from giq.weights import load_ref, recipe_of


def _refs(env: dict[str, str], worker: str, model: str, wanted: dict[str, str | None]) -> None:
    """Hand the child what the recipe loads, by the variables it reads.

    ``wanted`` maps a variable to a part (None: the main weights). A variable
    already set in giq's environment wins — the operator said so, and the
    child reads it the same way it always has.
    """
    for var, part in wanted.items():
        if var not in os.environ and (ref := load_ref(model, part)):
            env[var] = ref


# whisper-large-v3 fp16 + pyannote diarization measured ~3.9GB resident.
AUDIO_VRAM_GB = 4.0
# ECAPA is ~20MB; the child's CUDA context dominates.
EMBED_VRAM_GB = 0.6


@dataclass
class AudioConfig:
    model: str = "whisper-large-v3"


class AudioAdapter(SubprocessAdapter):
    """Transcription + diarization (faster-whisper + pyannote)."""

    child_module: ClassVar[str] = "giq_speech._audio_child"
    modality: ClassVar[str] = "audio"
    # Diarizing an hours-long recording takes minutes; allow close to a 900s
    # client batch budget rather than the 600s subprocess default.
    run_batch_timeout: ClassVar[float] = 860.0

    def __init__(self, config: AudioConfig, device: str | None = None):
        super().__init__(config, device)

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        _refs(
            env,
            "audio",
            self.config.model,
            {"GIQ_AUDIO_WHISPER_MODEL": "asr", "GIQ_AUDIO_DIAR_MODEL": "diarization"},
        )
        return env

    @property
    def estimated_vram_gb(self) -> float:
        return AUDIO_VRAM_GB

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        # Results are plain transcription dicts; no Pydantic hydration — the
        # API layer serves them as-is, and the shape is a served contract.
        return await super().run_batch(tasks, params)


@dataclass
class EmbedConfig:
    model: str = "ecapa-tdnn"


class EmbedAdapter(SubprocessAdapter):
    """Speaker voiceprints (speechbrain ECAPA-TDNN), stateless clip → vector."""

    child_module: ClassVar[str] = "giq_speech._embed_child"
    modality: ClassVar[str] = "embed"

    def __init__(self, config: EmbedConfig, device: str | None = None):
        super().__init__(config, device)

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        _refs(env, "embed", self.config.model, {"GIQ_EMBED_MODEL": None})
        recipe = recipe_of(self.config.model)
        revision = recipe.weights.revision if recipe and recipe.weights else None
        if revision and "GIQ_EMBED_REVISION" not in os.environ:
            env["GIQ_EMBED_REVISION"] = revision
        return env

    @property
    def estimated_vram_gb(self) -> float:
        return EMBED_VRAM_GB
