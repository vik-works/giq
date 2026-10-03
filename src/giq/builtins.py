# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq's own registrations, through the same contract a plugin uses (ADR-004).

Core registers the `llm` modality, the llama.cpp engine and the
`transformers` engine name. Everything else is grouped as the curated plugin
it belongs to. These groups ship inside giq for now and become packages of
their own (D7) without their registrations changing.

`transformers` is core's although core runs no transformers model: OCR and
depth both name it, and two plugins registering one engine would clash. It
declares only that a recipe runs in giq's own interpreter. The library
itself is a dependency of the plugins that use it.

Every adapter import is deferred to the factory, so building the registry
imports no engine and no ML library.
"""

from __future__ import annotations

from typing import Any

from giq.paths import engine_binary, env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin, SmokeTest

# --- core: llm, llama.cpp, transformers ----------------------------------------


def _llamacpp_start_budget(recipe: Any) -> float:
    from giq.adapters.llama_cpp import DEFAULT_READY_TIMEOUT, MODEL_READY_TIMEOUT

    return float(MODEL_READY_TIMEOUT.get(recipe.name, DEFAULT_READY_TIMEOUT))


def _llamacpp(ctx: AdapterContext):
    from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig

    return LlamaCppAdapter(
        config=LlamaCppConfig(model=ctx.recipe, model_path=ctx.model_path, device=ctx.device)
    )


def _llm_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[
            {
                "id": "smoke-llm",
                "user": "Reply with exactly: PONG",
                "params": {"max_tokens": 8, "enable_thinking": False},
            }
        ],
        timeout=120.0,
        summary=lambda result: {"output": (result.get("output") or "")[:200]},
    )


def _core() -> Plugin:
    from giq.recipes.schema import LlamaCppParams, llamacpp_consistent

    return Plugin(
        name="giq",
        api_version=API_VERSION,
        engines=(
            Engine(
                "llama.cpp",
                Binary(
                    lambda: engine_binary("llama.cpp", "llama-server"),
                    env="GIQ_LLAMA_BINARY",
                    detail="llama-server, one process per model on its card's port",
                ),
                params=LlamaCppParams,
                request_defaults=None,
                validate=llamacpp_consistent,
                start_budget=_llamacpp_start_budget,
                # \b: --port 8089 must not also match --port 80891.
                stale_pattern=r"llama-server .*--port {port}\b",
            ),
            Engine("transformers", aliases=("transformers-4.57",)),
        ),
        modalities=(
            # llama-server's 4 slots: a client fanning out a few chat calls
            # at once gets them served in parallel.
            Modality(
                "llm",
                label="LLM",
                icon="chat",
                lane_width=4,
                # llama.cpp's vision projector.
                parts=frozenset({"mmproj"}),
                smoke_test=_llm_smoke,
            ),
        ),
        adapters={("llama.cpp", "llm"): _llamacpp},
        routers=(
            "giq.api.router:router",
            "giq.api.openai_compat:router",
            "giq.api.openai_responses:router",
            "giq.api.stats_api:router",
            "giq.api.recipes_api:router",
        ),
    )


# --- giq-vllm ------------------------------------------------------------------


def _vllm(ctx: AdapterContext):
    import logging

    from giq.adapters.vllm import VllmAdapter, VllmConfig

    if ctx.model_path:
        # A path override names a GGUF for llama-server; a vllm model's
        # weights are part of its declaration.
        logging.getLogger("giq.runner").warning(
            f"llm/{ctx.recipe}: model_path ignored, vllm serves its declared weights"
        )
    return VllmAdapter(config=VllmConfig(model=ctx.recipe, device=ctx.device))


def _vllm_sweep(ports: set[int]) -> None:
    # A vllm server runs in a transient systemd scope, which outlives a giq
    # that died without stopping it; the scope takes its engine core down
    # with it, which a pattern on the API server's command line would miss.
    from giq.adapters.vllm import scope_unit, stop_scope

    for port in sorted(ports):
        stop_scope(scope_unit(port))


def _vllm_prepare(argv: list[str]) -> int:
    from giq.adapters.vllm import cli_prepare

    return cli_prepare(argv)


def _vllm_plugin() -> Plugin:
    from giq.recipes.schema import (
        ENGINE_PROFILES,
        VLLM_REQUEST_DEFAULTS,
        VllmParams,
        derived_vram,
        vllm_consistent,
    )

    return Plugin(
        name="giq-vllm",
        api_version=API_VERSION,
        engines=(
            Engine(
                "vllm",
                # `vllm serve` runs from this env's console script, next to the
                # interpreter; the interpreter is what is declared and probed,
                # because asking the script for its version imports torch and
                # takes longer than the probe allows.
                Binary(
                    lambda: env_python("vllm"),
                    env="GIQ_VLLM_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; "
                        "print('vllm', m.version('vllm'), 'torch', m.version('torch'), "
                        "'flashinfer', m.version('flashinfer-python'))",
                    ),
                    detail="vllm serve, one process per model on its card's port (envs/vllm)",
                ),
                params=VllmParams,
                profiles=ENGINE_PROFILES["vllm"],
                request_defaults=VLLM_REQUEST_DEFAULTS,
                validate=vllm_consistent,
                derive_vram=derived_vram,
                # max_num_seqs is how many requests vllm really runs at once.
                lanes=lambda params: params.max_num_seqs,
                start_budget=lambda recipe: float(recipe.params.ready_timeout),
                stale_pattern=r"vllm serve .*--port {port}\b",
                sweep=_vllm_sweep,
                prepare=_vllm_prepare,
            ),
        ),
        adapters={("vllm", "llm"): _vllm},
    )


# --- giq-sdcpp -----------------------------------------------------------------

_IMAGE_PARTS = frozenset({"diffusion", "text_encoder", "vae", "lora"})


def _sdcpp(ctx: AdapterContext):
    from giq.adapters.sdcpp import SdCppAdapter, SdCppConfig

    return SdCppAdapter(config=SdCppConfig(model=ctx.recipe, device=ctx.device))


def _sdcpp_start_budget(recipe: Any) -> float:
    from giq.adapters.sdcpp import READY_TIMEOUT_SECONDS

    return READY_TIMEOUT_SECONDS


def _image_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[{"id": "smoke-image", "prompt": "a tiny test pattern, colorful geometric shapes"}],
        timeout=700.0,
        summary=lambda result: {"seed": result.get("seed"), "image_b64": result.get("image_b64")},
        evicts=True,
    )


def _sdcpp_plugin() -> Plugin:
    return Plugin(
        name="giq-sdcpp",
        api_version=API_VERSION,
        engines=(
            Engine(
                "sd.cpp",
                # sd-server has no --version; its banner carries the commit
                # and it prints that for --help too.
                Binary(
                    lambda: engine_binary("sd.cpp", "sd-server"),
                    env="GIQ_SDCPP_BINARY",
                    version_args=("--help",),
                    detail="sd-server (stable-diffusion.cpp)",
                ),
                aliases=("sdcpp",),
                start_budget=_sdcpp_start_budget,
                stale_pattern=r"sd-server .*--listen-port {port}\b",
            ),
        ),
        modalities=(
            Modality(
                "text2image",
                label="Text to image",
                icon="image",
                parts=_IMAGE_PARTS,
                smoke_test=_image_smoke,
            ),
            Modality(
                "image_edit",
                label="Image edit",
                icon="paint-brush",
                parts=_IMAGE_PARTS,
                payload_keys=("reference_image_b64",),
            ),
        ),
        # One sd-server serves both: flux_klein renders and edits without a reload.
        adapters={("sd.cpp", "text2image"): _sdcpp, ("sd.cpp", "image_edit"): _sdcpp},
    )


# --- giq-speech ----------------------------------------------------------------


def _audio(ctx: AdapterContext):
    from giq.adapters.audio import AudioAdapter, AudioConfig

    return AudioAdapter(config=AudioConfig(model=ctx.recipe), device=ctx.device)


def _embed(ctx: AdapterContext):
    from giq.adapters.audio import EmbedAdapter, EmbedConfig

    return EmbedAdapter(config=EmbedConfig(model=ctx.recipe), device=ctx.device)


def _tts(ctx: AdapterContext):
    from giq.adapters.tts import TtsAdapter, TtsConfig

    return TtsAdapter(config=TtsConfig(model=ctx.recipe), device=ctx.device)


def _stt(ctx: AdapterContext):
    from giq.adapters.stt import SttAdapter, SttConfig

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
        routers=("giq.api.audio_api:router",),
    )


# --- giq-ocr, giq-depth ----------------------------------------------------------


def _ocr(ctx: AdapterContext):
    from giq.adapters.ocr import OcrAdapter, OcrConfig

    return OcrAdapter(config=OcrConfig(model=ctx.recipe), device=ctx.device)


def _depth(ctx: AdapterContext):
    from giq.adapters.depth import DepthAdapter, DepthConfig

    return DepthAdapter(config=DepthConfig(model=ctx.recipe), device=ctx.device)


def _ocr_plugin() -> Plugin:
    return Plugin(
        name="giq-ocr",
        api_version=API_VERSION,
        modalities=(
            # A long PDF is several passes of a few minutes each (~80 tok/s,
            # up to 32k tokens a pass). Matches OcrAdapter.run_batch_timeout.
            Modality(
                "ocr",
                label="Document OCR",
                icon="file-text",
                job_timeout=3600.0,
                parts=frozenset({"layout"}),
                payload_keys=("pdf_b64", "images_b64"),
            ),
        ),
        adapters={("transformers", "ocr"): _ocr},
        routers=("giq.api.ocr_api:router",),
    )


def _depth_plugin() -> Plugin:
    return Plugin(
        name="giq-depth",
        api_version=API_VERSION,
        modalities=(
            Modality(
                "depth",
                label="Depth",
                icon="mountains",
                payload_keys=("image_b64",),
                weights_root_env="GIQ_DEPTH_MODELS_DIR",
            ),
        ),
        adapters={("transformers", "depth"): _depth},
        routers=("giq.api.depth_api:router",),
    )


class _Plugins:
    """Built lazily: the declarations import the recipe schema, which itself
    looks names up in the registry."""

    def __iter__(self):
        return iter(
            (
                _core(),
                _vllm_plugin(),
                _sdcpp_plugin(),
                _speech_plugin(),
                _ocr_plugin(),
                _depth_plugin(),
            )
        )


PLUGINS = _Plugins()
