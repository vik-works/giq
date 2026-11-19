# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The text-to-sound-effects worker: MOSS-SoundEffect v2 in a child process
on its own interpreter.

The child (``giq_sfx_child._sfx_child``) loads the recipe's weights — MOSS's
diffusers-style layout: the DiT, the DAC VAE, the Qwen3 text encoder — with
the hub disabled and the layout checked first, so nothing is ever fetched.
Per task it renders one sound from a prompt.

Sleepy model: loads on demand, evictable. Runs on the mosssfx interpreter,
like the vllm engine's own: the child imports only the top-level
``giq_sfx_child`` package plus ``giq_child``, never giq.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from giq.adapters._subprocess import SubprocessAdapter
from giq.registry import get_recipe, vram_for
from giq.weights import require_path
from giq_sfx.params import COMPILE_READY_TIMEOUT, MossSfxParams

# Upper bound for an unregistered model; the registry has the recipe's.
SFX_VRAM_GB = 20.0


class SfxResult(BaseModel):
    """One sound, and what it was rendered with: the task's values, the
    recipe's where it gave none, and the seed it drew when it named none."""

    id: str
    audio_b64: str | None = None
    format: str | None = None
    sample_rate: int | None = None
    duration: float | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    error: str | None = None


@dataclass
class SfxConfig:
    model: str = "moss-soundeffect-v2"


class SfxAdapter(SubprocessAdapter):
    """MOSS-SoundEffect text to sound effects in a child process on the
    ``moss-sfx`` interpreter."""

    child_module: ClassVar[str] = "giq_sfx_child._sfx_child"
    modality: ClassVar[str] = "text2sfx"
    engine: ClassVar[str] = "moss-sfx"
    # A batch of four on a slow card is minutes of DiT passes; just under the
    # modality's 900 s job_timeout, so a stuck child is reported as such
    # before the job is cut off.
    run_batch_timeout: ClassVar[float] = 870.0

    def __init__(self, config: SfxConfig, device: str | None = None):
        super().__init__(config, device)
        # Unknown model or missing weights: fail here, not at spawn.
        self.weights = require_path(config.model)
        recipe = get_recipe(config.model)
        self.params = (
            recipe.params
            if recipe and isinstance(recipe.params, MossSfxParams)
            else MossSfxParams()
        )

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=SFX_VRAM_GB)

    def child_args(self) -> list[str]:
        p = self.params
        args = ["--model", self.config.model, "--weights", self.weights]
        args += ["--steps", str(p.steps), "--guidance", str(p.guidance), "--shift", str(p.shift)]
        return args + (["--compile"] if p.compile else [])

    def _command(self) -> list[str]:
        from giq.engines import require_binary

        return [require_binary(self.engine), "-u", "-m", self.child_module, *self.child_args()]

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        # The engine's interpreter has the engine installed but not giq: the
        # plugin's child package and the giq_child protocol travel on
        # PYTHONPATH, so the child imports no giq.
        import giq_child

        path = os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1]),
                str(Path(giq_child.__file__).resolve().parent.parent),
            ]
        )
        env["PYTHONPATH"] = path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[SfxResult]:
        raw_results = await super().run_batch(tasks, params)
        return [hydrate(r) for r in raw_results]


class CompiledSfxAdapter(SfxAdapter):
    """The same, for a recipe that compiles the DiT: the child compiles it at
    load, which takes minutes on a cold inductor cache."""

    ready_timeout: ClassVar[float] = COMPILE_READY_TIMEOUT


def sfx_adapter(config: SfxConfig, device: str | None = None) -> SfxAdapter:
    """The adapter for the recipe: compiled or not decides how long it may start."""
    recipe = get_recipe(config.model)
    params = recipe.params if recipe is not None else None
    if isinstance(params, MossSfxParams) and params.compile:
        return CompiledSfxAdapter(config, device)
    return SfxAdapter(config, device)


def hydrate(result: dict[str, Any]) -> SfxResult:
    """Child result dict → SfxResult; an error envelope stays an error."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return SfxResult(id=rid, error=str(result["error"]))
    return SfxResult.model_validate({**result, "id": rid})
