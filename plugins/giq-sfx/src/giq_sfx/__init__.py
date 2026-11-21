# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-sfx: text to sound effects with MOSS-SoundEffect v2 (ADR-004).

MOSS-SoundEffect on its own interpreter (envs/mosssfx: it pins its own
transformers, diffusers and numpy) in a child process: a DiT renders a
sound from a prompt; the text2sfx modality; the sandbox panel; the built-in
recipe.
"""

from __future__ import annotations

from pathlib import Path

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin, SmokeTest
from giq.plugins import curated_ui
from giq_sfx.params import MossSfxParams

__version__ = "0.14.0"


def _sfx(ctx: AdapterContext):
    from giq_sfx.adapter import SfxConfig, sfx_adapter

    return sfx_adapter(SfxConfig(model=ctx.recipe), device=ctx.device)


def _sfx_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[{"id": "smoke-sfx", "prompt": "a wooden door creaks open slowly", "duration": 4}],
        # A cold load, and the warm-up compile when the recipe asks for one.
        timeout=1500.0,
        summary=lambda result: {
            "duration": result.get("duration"),
            "format": result.get("format"),
            "seed": result.get("seed"),
            "audio_b64": result.get("audio_b64"),
        },
    )


def _sfx_plugin() -> Plugin:
    return Plugin(
        name="giq-sfx",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "moss-sfx",
                Binary(
                    lambda: env_python("mosssfx"),
                    env="GIQ_MOSSSFX_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; print('moss-soundeffect-v2', "
                        "m.version('moss-soundeffect-v2'), 'torch', m.version('torch'))",
                    ),
                    detail=(
                        "python + torch + MOSS-SoundEffect, for the text2sfx child (envs/mosssfx)"
                    ),
                ),
                params=MossSfxParams,
                start_budget=lambda recipe: recipe.params.ready_timeout,
            ),
        ),
        modalities=(
            # A sound is seconds to tens of seconds of DiT passes whatever its
            # length (MOSS always denoises 30 s); a batch of four on a slow
            # card is minutes.
            Modality(
                "text2sfx",
                label="Text to sound effects",
                icon="bell-ringing",
                job_timeout=900.0,
                smoke_test=_sfx_smoke,
            ),
        ),
        adapters={
            ("moss-sfx", "text2sfx"): _sfx,
        },
        recipes=Path(__file__).parent / "recipes",
        ui=curated_ui("giq-sfx"),
    )


plugin = _sfx_plugin()
