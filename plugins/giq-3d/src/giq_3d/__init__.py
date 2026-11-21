# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-3d: one image to a textured 3D mesh with Pixal3D (ADR-004).

Pixal3D on its own interpreter (envs/pixal3d: its torch, and the compiled
extensions `giq prepare pixal3d` builds) in a child process; the image2mesh
modality, and mesh2texture, which paints a mesh the client brings; the
sandbox panel; the built-in recipes. Not part of giq-defaults: the engine takes a
build step, and its encoder comes under Meta's DINOv3 licence, which is the
operator's to accept.
"""

from __future__ import annotations

from pathlib import Path

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin
from giq.plugins import curated_ui
from giq_3d.params import Pixal3DParams

__version__ = "0.14.0"


def _mesh(ctx: AdapterContext):
    from giq_3d.adapter import MeshAdapter, MeshConfig

    return MeshAdapter(config=MeshConfig(model=ctx.recipe), device=ctx.device)


def _start_budget(_params) -> float:
    from giq_3d.adapter import START_BUDGET_SECONDS

    return START_BUDGET_SECONDS


def _prepare(argv: list[str]) -> int:
    from giq_3d.prepare import main

    return main(argv)


def _mesh_plugin() -> Plugin:
    return Plugin(
        name="giq-3d",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "pixal3d",
                Binary(
                    lambda: env_python("pixal3d"),
                    env="GIQ_PIXAL3D_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; print('torch', m.version('torch'), "
                        "'o-voxel', m.version('o_voxel'), 'natten', m.version('natten'))",
                    ),
                    detail="python + torch + Pixal3D's extensions, for the image2mesh child "
                    "(envs/pixal3d, then `giq prepare pixal3d`)",
                ),
                params=Pixal3DParams,
                prepare=_prepare,
                start_budget=_start_budget,
            ),
        ),
        modalities=(
            # Generation is half a minute on a fast card; the mesh clean-up
            # and texture bake after it are a minute more, mostly on the CPU,
            # and a cold start loads 23 GB of weights first.
            Modality(
                "image2mesh",
                label="Image to 3D",
                icon="cube",
                job_timeout=900.0,
                # The DINOv3 encoder, the camera estimator, the background
                # remover, TRELLIS.2's structure decoder and a part
                # segmenter, each a repository of its own with its own licence.
                parts=frozenset({"encoder", "camera", "matting", "structure", "segmenter"}),
                payload_keys=("image_b64", "views_b64"),
            ),
            # A mesh the client brings, painted from a picture: TRELLIS.2's
            # texture stage on its geometry, the vertices and faces kept. It
            # runs in a trellis2 recipe's image2mesh child, so its parts are
            # that child's.
            Modality(
                "mesh2texture",
                label="Mesh to texture",
                icon="cube",
                job_timeout=900.0,
                parts=frozenset({"encoder", "matting", "structure", "segmenter"}),
                payload_keys=("mesh_b64", "image_b64", "shape_b64"),
            ),
        ),
        adapters={
            ("pixal3d", "image2mesh"): _mesh,
            ("pixal3d", "mesh2texture"): _mesh,
        },
        recipes=Path(__file__).parent / "recipes",
        ui=curated_ui("giq-3d"),
    )


plugin = _mesh_plugin()
