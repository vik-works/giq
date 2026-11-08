# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-music: text to music and sound with ACE-Step 1.5 and Stable Audio 3
(ADR-004).

Each engine on its own interpreter, in a child process: ACE-Step
(envs/acestep: it pins its own torch and transformers), where a 5Hz LM plans
the song and a DiT renders it, and Stable Audio 3 (envs/stable-audio: torch
2.7), a DiT that renders a prompt. The text2music and text2sfx modalities;
the sandbox panels; the built-in recipes.
"""

from __future__ import annotations

from pathlib import Path

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin, SmokeTest
from giq.plugins import curated_ui
from giq_music.params import AceStepParams, StableAudioParams

__version__ = "0.7.0"


def _music(ctx: AdapterContext):
    from giq_music.adapter import MusicAdapter, MusicConfig

    return MusicAdapter(config=MusicConfig(model=ctx.recipe), device=ctx.device)


def _stable_audio(ctx: AdapterContext):
    from giq_music.adapter import MusicConfig, StableAudioAdapter

    return StableAudioAdapter(config=MusicConfig(model=ctx.recipe), device=ctx.device)


def _smoke(kind: str, prompt: str) -> SmokeTest:
    return SmokeTest(
        tasks=[{"id": f"smoke-{kind}", "prompt": prompt, "duration": 10}],
        timeout=600.0,
        summary=lambda result: {
            "duration": result.get("duration"),
            "format": result.get("format"),
            "audio_b64": result.get("audio_b64"),
        },
    )


def _music_smoke() -> SmokeTest:
    return _smoke("music", "a short calm piano phrase")


def _sfx_smoke() -> SmokeTest:
    return _smoke("sfx", "a door creaking open in an empty hall")


def _music_plugin() -> Plugin:
    return Plugin(
        name="giq-music",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "acestep",
                Binary(
                    lambda: env_python("acestep"),
                    env="GIQ_ACESTEP_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; print('ace-step', m.version('ace-step'), "
                        "'torch', m.version('torch'))",
                    ),
                    detail="python + torch + ACE-Step, for the text2music child (envs/acestep)",
                ),
                params=AceStepParams,
            ),
            Engine(
                "stable-audio",
                Binary(
                    lambda: env_python("stable-audio"),
                    env="GIQ_STABLE_AUDIO_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; print('stable-audio-3', "
                        "m.version('stable-audio-3'), 'torch', m.version('torch'))",
                    ),
                    detail="python + torch + Stable Audio 3, for its child (envs/stable-audio)",
                ),
                params=StableAudioParams,
            ),
        ),
        modalities=(
            # A track is seconds to render on a fast card, but a batch of
            # ten-minute songs with the planner on is minutes of GPU time.
            Modality(
                "text2music",
                label="Text to music",
                icon="music-notes",
                job_timeout=900.0,
                # An ACE-Step DiT that is a repository of its own (the XL ones).
                parts=frozenset({"dit"}),
                # A task's own audio: a reference to sound like, a source to build on.
                payload_keys=("reference_audio_b64", "source_audio_b64"),
                smoke_test=_music_smoke,
            ),
            # Sound effects: a few seconds of foley or ambience per task.
            Modality(
                "text2sfx",
                label="Text to sound",
                icon="bell",
                job_timeout=600.0,
                smoke_test=_sfx_smoke,
            ),
        ),
        adapters={
            ("acestep", "text2music"): _music,
            ("stable-audio", "text2music"): _stable_audio,
            ("stable-audio", "text2sfx"): _stable_audio,
        },
        recipes=Path(__file__).parent / "recipes",
        ui=curated_ui("giq-music"),
    )


plugin = _music_plugin()
