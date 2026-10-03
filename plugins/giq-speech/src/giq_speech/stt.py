# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""STT worker using faster-whisper."""

import asyncio
import base64
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from giq.models import JobResult
from giq.registry import vram_for
from giq.weights import load_ref

logger = logging.getLogger(__name__)


def model_ref(model: str) -> str:
    """What faster-whisper loads for ``model``: the recipe's weights.path
    or the repository of its ``hf:`` source. A name without a recipe
    falls through to faster-whisper's own table of size names."""
    return load_ref(model) or model


@dataclass
class SttConfig:
    """Configuration for STT worker."""

    # A recipe name: faster-whisper-{tiny,base,small,medium,large-v3}.
    model: str = "faster-whisper-base"
    device: str = "cuda"
    compute_type: str = "float16"
    # GPU UUID this model is bound to. Unlike every other CUDA worker, STT
    # runs in giq's own process, so it cannot be pinned with
    # CUDA_VISIBLE_DEVICES at spawn — there is no spawn. faster-whisper takes
    # a CUDA ordinal instead, and giq's process sees every card, so the card's
    # own index is the right ordinal.
    gpu_device: str | None = None


@dataclass
class SttAdapter:
    """STT worker using faster-whisper."""

    config: SttConfig
    # Runs in giq's own process: the VRAM it holds is giq's own pid's.
    in_process: ClassVar[bool] = True
    _model: object = field(default=None, repr=False)
    _ready: bool = field(default=False, repr=False)

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._ready and self._model is not None

    @property
    def is_ready(self) -> bool:
        """Check if worker is ready to accept requests."""
        return self.is_running

    @property
    def pid(self) -> int | None:
        """None: this worker runs in giq's own process, so its VRAM cannot be
        told apart from anything else giq has allocated. (It also means its
        CUDA context is never reclaimed on stop — see the subprocess-isolated
        workers for the pattern that fixes that.)"""
        return None

    @property
    def estimated_vram_gb(self) -> float:
        """Estimated VRAM usage."""
        return vram_for(self.config.model, default=2.0)

    async def start(self) -> None:
        """Load Whisper model."""
        if self._ready:
            return

        logger.info(f"Loading faster-whisper model: {self.config.model}")

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)

        self._ready = True
        logger.info(f"Whisper STT ready: {self.config.model}")

    def _load_model(self):
        """Load model (blocking)."""
        from faster_whisper import WhisperModel

        from giq.gpus import resolve_device
        from giq.vram import device_for_recipe

        uuid = self.config.gpu_device or device_for_recipe(self.config.model)
        gpu = resolve_device(uuid) if uuid else None
        kwargs = {}
        if gpu is not None and self.config.device == "cuda":
            kwargs["device_index"] = gpu.index
        return WhisperModel(
            model_ref(self.config.model),
            device=self.config.device,
            compute_type=self.config.compute_type,
            **kwargs,
        )

    async def stop(self) -> None:
        """Unload model."""
        self._model = None
        self._ready = False
        logger.info("Whisper STT stopped")

    def _transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """Transcribe audio bytes (blocking)."""
        # Write to temp file (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            segments, info = self._model.transcribe(
                temp_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            # Collect all segments
            text = " ".join(segment.text.strip() for segment in segments)

            logger.info(f"Transcribed: {info.duration:.1f}s, lang={info.language}")
            return text

        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def run_batch(self, tasks: list[dict], params: dict | None = None) -> list[JobResult]:
        """Run batch of STT tasks."""
        if not self._ready:
            raise RuntimeError("Worker not ready")

        results = []
        loop = asyncio.get_event_loop()

        for task in tasks:
            task_id = task.get("id", "unknown")
            audio_b64 = task.get("audio_b64", "")
            language = task.get("language")  # None = auto-detect

            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_b64)

                # Transcribe in executor
                text = await loop.run_in_executor(None, self._transcribe, audio_bytes, language)

                results.append(
                    JobResult(
                        id=task_id,
                        output=text,
                    )
                )

            except Exception as e:
                logger.error(f"STT task {task_id} failed: {e}")
                results.append(JobResult(id=task_id, output="", error=str(e)))

        return results
