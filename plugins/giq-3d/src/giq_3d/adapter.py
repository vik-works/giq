# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The image-to-mesh worker: Pixal3D in a child process on its own
interpreter.

The child (``giq_3d_child._mesh_child``) loads the recipe's flow models and
decoders from weights.path, and the DINOv3 encoder, the camera estimator
and the background remover from weights.parts (encoder, camera, matting).
Per task it turns one image, or four views round the object for a recipe
with ``views: multi``, into a mesh and exports it as GLB: at full detail, or
as a game-ready low-poly asset with a normal map (``mode``, see the child).

Pixal3D is a checkout, not a package: its code is imported from the source
directory ``giq prepare pixal3d`` cloned, which the child is told about.

Sleepy model: loads on demand, evictable. Runs on the pixal3d interpreter:
the child imports only the top-level ``giq_3d_child`` package plus
``giq_child``, never giq.
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
from giq_3d.params import Pixal3DParams

# Upper bound for an unregistered model; the registry has the measured ones.
MESH_VRAM_GB = 26.0
# What a start may take, the first one included (see MeshAdapter.ready_timeout).
START_BUDGET_SECONDS = 900.0
# The driver keeps the kernels it compiled in a cache of 1 GiB by default,
# which the other engines on a machine already fill; with no room, every
# start compiles them again.
KERNEL_CACHE_BYTES = 4 * 1024**3


def source_dir() -> str:
    """Where Pixal3D's code is: ``GIQ_PIXAL3D_SOURCE``, else the checkout
    ``giq prepare pixal3d`` makes next to the engine's environment."""
    return os.environ.get("GIQ_PIXAL3D_SOURCE") or str(repo_root() / "envs" / "pixal3d" / "Pixal3D")


class MeshResult(BaseModel):
    """One asset, and what the export made of it.

    ``faces`` and ``texture_size`` are what came out, which for a low-poly
    export is the request's target or just under it."""

    id: str
    glb_b64: str | None = None
    format: str | None = None
    mode: str | None = None
    faces: int | None = None
    vertices: int | None = None
    texture_size: int | None = None
    seed: int | None = None
    error: str | None = None


@dataclass
class MeshConfig:
    model: str = "pixal3d"


class MeshAdapter(SubprocessAdapter):
    """Pixal3D image to mesh in a child process on the ``pixal3d`` interpreter."""

    modality: ClassVar[str] = "image2mesh"
    child_module: ClassVar[str] = "giq_3d_child._mesh_child"
    engine: ClassVar[str] = "pixal3d"
    # A batch of two at 1536 with game-ready exports is minutes; just under
    # the modality's 900 s job_timeout, so a stuck child is reported as such
    # before the job is cut off.
    run_batch_timeout: ClassVar[float] = 870.0
    # A warm start is about 45 s. The first start after `giq prepare pixal3d`
    # is not: the extensions are built as PTX for a card the toolkit may not
    # know, and the driver compiles every kernel for it on first load.
    # Measured on an RTX 5090, that first start ran past 300 s.
    ready_timeout: ClassVar[float] = START_BUDGET_SECONDS

    def __init__(self, config: MeshConfig, device: str | None = None):
        super().__init__(config, device)
        # Unknown model or missing weights: fail here, not at spawn.
        self.weights = require_path(config.model)
        recipe = get_recipe(config.model)
        self.params = (
            recipe.params
            if recipe and isinstance(recipe.params, Pixal3DParams)
            else Pixal3DParams()
        )

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=MESH_VRAM_GB)

    def _command(self) -> list[str]:
        from giq.engines import require_binary

        return [require_binary(self.engine), "-u", "-m", self.child_module, *self.child_args()]

    def child_args(self) -> list[str]:
        model = self.config.model
        args = ["--model", model, "--weights", self.weights, "--source", source_dir()]
        for part in ("encoder", "camera", "matting"):
            path = path_of(model, part)
            if not path:
                raise ValueError(f"{model} names no weights.parts.{part}")
            args += [f"--{part}", path]
        args += ["--resolution", str(self.params.resolution), "--views", self.params.views]
        return args + (["--low-vram"] if self.params.low_vram else [])

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
        env.setdefault("CUDA_CACHE_MAXSIZE", str(KERNEL_CACHE_BYTES))
        return env

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[MeshResult]:
        raw_results = await super().run_batch(tasks, params)
        return [hydrate(r) for r in raw_results]


def hydrate(result: dict[str, Any]) -> MeshResult:
    """Child result dict → MeshResult; an error envelope stays an error."""
    rid = str(result.get("id", "unknown"))
    if result.get("error"):
        return MeshResult(id=rid, error=str(result["error"]))
    return MeshResult.model_validate({**result, "id": rid})
