# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-imajev: typed photo+record decisions (ADR-004).

imajev on its own interpreter (envs/imajev, torch + PEFT LoRA + a decision
readout over a Qwen3.5 checkpoint) in a child process; POST /decide; the
built-in imajev recipe.
"""

from __future__ import annotations

from pathlib import Path

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin

__version__ = "0.6.0"


def _decide(ctx: AdapterContext):
    from giq_imajev.adapter import DecideAdapter, DecideConfig

    return DecideAdapter(config=DecideConfig(model=ctx.recipe), device=ctx.device)


def _imajev_plugin() -> Plugin:
    return Plugin(
        name="giq-imajev",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "imajev",
                Binary(
                    lambda: env_python("imajev"),
                    env="GIQ_IMAJEV_PYTHON",
                    version_args=(
                        "-c",
                        "import transformers, peft; print('imajev', transformers.__version__, "
                        "'peft', peft.__version__)",
                    ),
                    detail="python + torch + PEFT, for the decide child (envs/imajev)",
                ),
            ),
        ),
        modalities=(
            # A decide batch is up to max_batch tasks of up to 8 questions at
            # up to 4 rotations: seconds per question, minutes for the batch.
            Modality(
                "decide",
                label="Decide",
                icon="eye",
                job_timeout=900.0,
                parts=frozenset({"base"}),
                payload_keys=("images_b64",),
                weights_root_env="GIQ_DECIDE_MODELS_DIR",
            ),
        ),
        adapters={("imajev", "decide"): _decide},
        recipes=Path(__file__).parent / "recipes",
        routers=("giq_imajev.api:router",),
    )


plugin = _imajev_plugin()
