# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-music: text to music with ACE-Step 1.5 (ADR-004).

ACE-Step on its own interpreter (envs/acestep: it pins its own torch and
transformers) in a child process: a 5Hz LM plans the song, a DiT renders
it; the text2music modality; the sandbox panel; the built-in recipes.
"""

from __future__ import annotations

from pathlib import Path

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin, SmokeTest
from giq.plugins import curated_ui
from giq_music.params import AceStepParams

__version__ = "0.13.0"


def _music(ctx: AdapterContext):
    from giq_music.adapter import MusicAdapter, MusicConfig

    return MusicAdapter(config=MusicConfig(model=ctx.recipe), device=ctx.device)


def _music_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[{"id": "smoke-music", "prompt": "a short calm piano phrase", "duration": 10}],
        timeout=600.0,
        summary=lambda result: {
            "duration": result.get("duration"),
            "format": result.get("format"),
            "audio_b64": result.get("audio_b64"),
        },
    )


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
        ),
        adapters={
            ("acestep", "text2music"): _music,
        },
        recipes=Path(__file__).parent / "recipes",
        ui=curated_ui("giq-music"),
    )


plugin = _music_plugin()
