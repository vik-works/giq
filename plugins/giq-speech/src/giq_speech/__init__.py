# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-speech: speech to text with speaker diarization, text to speech and
speaker voiceprints (ADR-004).

faster-whisper with pyannote, faster-whisper alone, speechbrain's ECAPA-TDNN
and Kokoro, each in a child process; the OpenAI-compatible /v1/audio routes;
the sandbox panels; the built-in speech recipes."""

from __future__ import annotations

from pathlib import Path

from giq.plugin import API_VERSION, AdapterContext, Engine, Modality, Plugin, SmokeTest
from giq.plugins import curated_ui

__version__ = "0.6.1"


def _audio(ctx: AdapterContext):
    from giq_speech.audio import AudioAdapter, AudioConfig

    return AudioAdapter(config=AudioConfig(model=ctx.recipe), device=ctx.device)


def _embed(ctx: AdapterContext):
    from giq_speech.audio import EmbedAdapter, EmbedConfig

    return EmbedAdapter(config=EmbedConfig(model=ctx.recipe), device=ctx.device)


def _tts(ctx: AdapterContext):
    from giq_speech.tts import TtsAdapter, TtsConfig

    return TtsAdapter(config=TtsConfig(model=ctx.recipe), device=ctx.device)


def _stt(ctx: AdapterContext):
    from giq_speech.stt import SttAdapter, SttConfig

    return SttAdapter(config=SttConfig(model=ctx.recipe, gpu_device=ctx.device))


def _beep_wav_b64(seconds: float = 1.0, freq: float = 440.0) -> str:
    """A tiny synthesized test tone, base64 WAV (no dependencies)."""
    import base64
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    rate = 16000
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(
                        6000
                        * math.sin(2 * math.pi * freq * t / rate)
                        * (0.5 + 0.5 * math.sin(2 * math.pi * 3 * t / rate))
                    ),
                )
                for t in range(int(rate * seconds))
            )
        )
    return base64.b64encode(buf.getvalue()).decode()


def _audio_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[
            {"id": "smoke-audio", "audio_b64": _beep_wav_b64(), "language": "en", "diarize": True}
        ],
        timeout=180.0,
        summary=lambda result: {
            "output": result.get("text", "") or "(silence: VAD gated the test tone)",
            "language": result.get("language"),
        },
    )


def _embed_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[{"id": "smoke-embed", "audio_b64": _beep_wav_b64()}],
        timeout=120.0,
        summary=lambda result: {"dim": result.get("dim")},
    )


def _speech_plugin() -> Plugin:
    return Plugin(
        name="giq-speech",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine("faster-whisper+pyannote"),
            Engine("faster-whisper"),
            Engine("speechbrain"),
            Engine("kokoro"),
        ),
        modalities=(
            # GPU-heavy and serial. Diarizing an hours-long recording takes
            # minutes of GPU time: a full batch budget, just under 15 minutes.
            Modality(
                "audio",
                label="Speech recognition",
                icon="waveform",
                job_timeout=870.0,
                parts=frozenset({"asr", "diarization"}),
                payload_keys=("audio_b64",),
                smoke_test=_audio_smoke,
            ),
            Modality("stt", label="Speech to text", icon="waveform", payload_keys=("audio_b64",)),
            Modality(
                "embed",
                label="Voiceprint",
                icon="vector",
                lane_width=2,
                payload_keys=("audio_b64",),
                smoke_test=_embed_smoke,
            ),
            Modality("tts", label="Text to speech", icon="speaker"),
        ),
        adapters={
            ("faster-whisper+pyannote", "audio"): _audio,
            ("faster-whisper", "stt"): _stt,
            ("speechbrain", "embed"): _embed,
            ("kokoro", "tts"): _tts,
        },
        recipes=Path(__file__).parent / "recipes",
        routers=("giq_speech.api:router",),
        ui=curated_ui("giq-speech"),
    )


plugin = _speech_plugin()
