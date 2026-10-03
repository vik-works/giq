# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Decide worker: typed photo+record decisions (imajev, Jev contract + images).

The child (``_decide_child``) loads the base Qwen3.5 checkpoint plus the
PEFT LoRA and the decision readout the recipe names, with the hub
disabled so nothing is ever fetched. Per task it compiles the state's
questions (``vision_decision.jev_api``), scores one forward per question per
rotation (``torch_decision.TorchDecision.candidate_logits``), averages
rotations (``vision_decision.scoring.combine_rotations``), applies the
shipped temperature (``vision_decision.calibration``), and returns Jev's
answer shapes plus ``unknown_probability`` / ``abstained``.

Sleepy model: loads on demand, evictable. Runs on the imajev
interpreter, like the vllm engine's own interpreter: the child imports only
the top-level ``giq_imajev_child`` package plus ``giq_child``, never giq.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from giq.adapters._subprocess import SubprocessAdapter
from giq.models import DecideResult
from giq.registry import vram_for
from giq.weights import require_path

# Which declared engine runs the child.
ENGINE = "imajev"

# Upper bound for an unregistered model; the registry has the measured ones.
DECIDE_VRAM_GB = 14.0


@dataclass
class DecideConfig:
    model: str = "imajev-2b"


class DecideAdapter(SubprocessAdapter):
    """imajev typed decisions in a child process on the ``imajev`` interpreter."""

    child_module: ClassVar[str] = "giq_imajev_child._decide_child"
    modality: ClassVar[str] = "decide"

    def __init__(self, config: DecideConfig, device: str | None = None):
        super().__init__(config, device)
        # The recipe's weights.path (the adapter dir) plus its base part
        # (the Qwen checkpoint). Unknown model or missing weights: fail
        # here, not at spawn.
        self.weights = require_path(config.model)
        self.base = require_path(config.model, "base")

    def child_args(self) -> list[str]:
        return ["--model", self.config.model, "--weights", self.weights, "--base", self.base]

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=DECIDE_VRAM_GB)

    def _command(self) -> list[str]:
        from giq.engines import require_binary

        return [require_binary(ENGINE), "-u", "-m", self.child_module, *self.child_args()]

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        # The imajev checkout (TorchDecision + scoring) rides on PYTHONPATH:
        # GIQ_IMAJEV_REPO, else the /projects checkout this was built for.
        # The plugin's own child package and the giq_child protocol travel
        # the same way, so the child imports no giq.
        import giq_child

        repo = os.environ.get("GIQ_IMAJEV_REPO", "/projects/imajev")
        path = os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1]),
                str(Path(giq_child.__file__).resolve().parent.parent),
                os.path.join(repo, "src"),
                os.path.join(repo, "scripts"),
            ]
        )
        env["PYTHONPATH"] = path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[DecideResult]:
        raw_results = await super().run_batch(tasks, params)
        return [hydrate(r) for r in raw_results]


def hydrate(result: dict[str, Any]) -> DecideResult:
    """Child result dict → DecideResult; an error envelope stays an error."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return DecideResult(id=rid, error=str(result["error"]))
    return DecideResult.model_validate({**result, "id": rid})
