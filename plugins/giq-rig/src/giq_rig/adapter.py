# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The mesh-to-rig worker: SkinTokens in a child process on its own
interpreter.

The child (``giq_rig_child._rig_child``) loads TokenRig and the skin
tokenizer from weights.path and the transformer's shape from
weights.parts.llm, once. Per task it gives a mesh a skeleton and skin
weights and hands back the same mesh, textures and scale kept, skinned to
that skeleton as GLB.

SkinTokens is a checkout, not a package: its code is imported from the
source directory ``giq prepare skintokens`` cloned, which the child is told
about.

Sleepy model: loads on demand, evictable. Runs on the skintokens
interpreter: the child imports only the top-level ``giq_rig_child`` package
plus ``giq_child``, never giq.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from giq.adapters._subprocess import SubprocessAdapter
from giq.paths import repo_root
from giq.registry import get_recipe, vram_for
from giq.weights import path_of, require_path
from giq_rig.params import SkinTokensParams

# Upper bound for an unregistered model; the registry has the measured one.
RIG_VRAM_GB = 6.0
# What a start may take: the checkpoints are 1.6 GB and nothing is compiled
# at load, so the protocol's default leaves room for a slow disk.
START_BUDGET_SECONDS = 300.0
# A rig is seconds once the model is loaded; a batch of four large meshes
# with beam search stays well inside this. Just under the modality's 600 s
# job_timeout, so a stuck child is reported as such before the job is cut off.
RUN_BATCH_SECONDS = 570.0
SOURCE_ENV = "GIQ_SKINTOKENS_SOURCE"


def source_dir() -> str:
    """Where SkinTokens' code is: ``GIQ_SKINTOKENS_SOURCE``, else the checkout
    ``giq prepare skintokens`` makes next to the engine's environment."""
    return os.environ.get(SOURCE_ENV) or str(repo_root() / "envs" / "skintokens" / "SkinTokens")


class RigResult(BaseModel):
    """One mesh, skinned: the GLB and how many bones its skeleton got.

    The count is the model's to choose and differs from mesh to mesh and
    from seed to seed."""

    id: str
    glb_b64: str | None = None
    format: str | None = None
    bones: int | None = None
    seed: int | None = None
    error: str | None = None


@dataclass
class RigConfig:
    model: str = "skintokens"


class RigAdapter(SubprocessAdapter):
    """SkinTokens mesh to rig in a child process on the ``skintokens`` interpreter."""

    modality: ClassVar[str] = "mesh2rig"
    child_module: ClassVar[str] = "giq_rig_child._rig_child"
    engine: ClassVar[str] = "skintokens"
    run_batch_timeout: ClassVar[float] = RUN_BATCH_SECONDS
    ready_timeout: ClassVar[float] = START_BUDGET_SECONDS

    def __init__(self, config: RigConfig, device: str | None = None):
        super().__init__(config, device)
        # Unknown model or missing weights: fail here, not at spawn.
        self.weights = require_path(config.model)
        recipe = get_recipe(config.model)
        self.params = (
            recipe.params
            if recipe and isinstance(recipe.params, SkinTokensParams)
            else SkinTokensParams()
        )

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=RIG_VRAM_GB)

    def _command(self) -> list[str]:
        from giq.engines import require_binary

        return [require_binary(self.engine), "-u", "-m", self.child_module, *self.child_args()]

    def child_args(self) -> list[str]:
        model = self.config.model
        llm = path_of(model, "llm")
        if not llm:
            raise ValueError(f"{model} names no weights.parts.llm (the transformer's config.json)")
        return [
            "--model",
            model,
            "--weights",
            self.weights,
            "--checkpoint",
            self.params.checkpoint,
            "--llm",
            llm,
            "--source",
            source_dir(),
        ]

    def _spawn_env(self) -> dict[str, str]:
        env = super()._spawn_env()
        # The engine's interpreter has the engine's dependencies but not giq:
        # the plugin's child package and the giq_child protocol travel on
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

    # Typed list[Any] rather than list[RigResult]: the base class promises
    # dicts, and a list of something narrower would not be one of those.
    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[Any]:
        """The child's results, each a :class:`RigResult`."""
        raw_results = await super().run_batch(tasks, params)
        return [hydrate(r) for r in raw_results]


def hydrate(result: dict[str, Any]) -> RigResult:
    """Child result dict → RigResult; an error envelope stays an error."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return RigResult(id=rid, error=str(result["error"]))
    return RigResult.model_validate({**result, "id": rid})
