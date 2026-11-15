# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Image-to-mesh child process: Pixal3D, and the export of what it makes.

Invoked by ``MeshAdapter`` via ``<pixal3d python> -u -m
giq_3d_child._mesh_child --model <recipe> --weights <checkpoint root>
--source <Pixal3D checkout> --encoder <DINOv3 folder> --camera <MoGe folder>
--matting <BiRefNet folder> --resolution <1024|1536> [--low-vram]``. Loads
the flow models, decoders and the four DINOv3 feature extractors once.

Pixal3D is a checkout, not a package: its code is imported from ``--source``.
Three things are put in place before it is: giq's UV rasteriser under the
name ``nvdiffrast.torch`` (the export's texture bake is all Pixal3D asks of
nvdiffrast, whose licence allows research and evaluation only); timm's
DINOv3 file for every feature extractor, its parameters renamed as it loads
(upstream names a mirror on the hub); and BiRefNet's own weights for the
background remover (upstream names RMBG-2.0, which is non-commercial). The
hub is offline, so anything missing fails here rather than being fetched.

Per task: the image is cut out of its background unless it already carries
an alpha channel, the camera it was taken with is estimated (or given, as
``fov``), the cascade generates the mesh, and it is exported as GLB by
``mode``:

  ``game`` (default)  a low-poly cage of ``faces`` triangles with the asset
                      baked onto it, normal map included (see ``lowpoly``);
  ``full``            Pixal3D's own export: the generated mesh simplified to
                      ``faces``, base colour, metallic and roughness only.

Wire protocol: see giq_child (``run_ipc_child_loop``). This module is a
top-level package importing no giq: the pixal3d interpreter carries torch
and Pixal3D's extensions, never giq's venv. The parsing below is pure, so
giq's suite tests it without Pixal3D.
"""

import base64
import binascii
import io
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# Lets the allocator hand freed blocks back between the cascade's stages,
# which is most of what keeps low-VRAM mode at its measured peak.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from giq_child import (  # noqa: E402
    reserve_ipc_stdout,
    run_ipc_child_loop,
    write_startup_error,
)

MODES = ("game", "full")
# What an export is asked for when the task does not say, and how far a task
# may go. A game cage under 200 faces loses the silhouette; the stock export
# above a million faces passes nvdiffrast's old limit for no visible gain.
FACES = {"game": (3000, 200, 50000), "full": (100000, 5000, 1000000)}
TEXTURE_SIZES = (1024, 2048, 4096)
TEXTURE_SIZE = {"game": 2048, "full": 4096}
# Voxels (of the cage's 384 grid) an opening may be wide and still be closed
# in a game cage: 4 seals vent slots and keeps a carry handle open.
CLOSE, MAX_CLOSE = 4, 8
# Pixal3D's own cap on the tokens of the high-resolution stage.
MAX_TOKENS = 49152
# The turn Pixal3D gives its export so the asset faces the viewer it was drawn for.
ROTATION = ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1))
MAX_IMAGE_BYTES = 40 * 1024 * 1024


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"'{name}' must be a whole number")
    return int(value)


def task_inputs(task: dict) -> dict:
    """A task's request, checked. Raises ValueError naming the field that is wrong."""
    image = task.get("image_b64")
    if not isinstance(image, str) or not image:
        raise ValueError("task needs an 'image_b64' of the object")
    try:
        data = base64.b64decode(image.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("'image_b64' is not base64") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"'image_b64' is larger than {MAX_IMAGE_BYTES // 2**20} MB")

    mode = task.get("mode", "game")
    if mode not in MODES:
        raise ValueError(f"'mode' must be one of {', '.join(MODES)}, got {mode!r}")
    default, low, high = FACES[mode]
    faces = default if task.get("faces") is None else _integer(task["faces"], "faces")
    if not low <= faces <= high:
        raise ValueError(f"'faces' must be {low}-{high} for a {mode} export, got {faces}")
    size = task.get("texture_size")
    size = TEXTURE_SIZE[mode] if size is None else _integer(size, "texture_size")
    if size not in TEXTURE_SIZES:
        raise ValueError(f"'texture_size' must be one of {TEXTURE_SIZES}, got {size}")
    close = CLOSE if task.get("close") is None else _integer(task["close"], "close")
    if not 0 <= close <= MAX_CLOSE:
        raise ValueError(f"'close' must be 0-{MAX_CLOSE}, got {close}")
    seed = 42 if task.get("seed") is None else _integer(task["seed"], "seed")
    if not 0 <= seed < 2**32:
        raise ValueError("'seed' must be 0 to 4294967295")
    fov = task.get("fov")
    if fov is not None:
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not 0.05 <= fov <= 2.5:
            raise ValueError("'fov' is the camera's horizontal field of view in radians, 0.05-2.5")
        fov = float(fov)
    return {
        "image": data,
        "mode": mode,
        "faces": faces,
        "texture_size": size,
        "close": close,
        "seed": seed,
        "fov": fov,
    }


def _load(args):
    """Pixal3D's pipeline, with the three substitutions the module docstring names."""
    for path, what in (
        (args.source, "the Pixal3D checkout"),
        (args.weights, "the weights"),
        (os.path.join(args.encoder, "model.safetensors"), "the DINOv3 encoder"),
        (os.path.join(args.camera, "model.pt"), "the MoGe checkpoint"),
        (args.matting, "the background remover"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} is missing: {path}")
    if not os.path.isfile(os.path.join(args.source, "inference.py")):
        raise FileNotFoundError(f"{args.source} is not a Pixal3D checkout (no inference.py)")

    from giq_3d_child import uvraster

    uvraster.install()
    sys.path.insert(0, args.source)

    from pixal3d.pipelines import rembg

    upstream = rembg.BiRefNet

    class LocalMatting(upstream):
        """BiRefNet from the recipe's folder, whatever the pipeline's config names."""

        def __init__(self, model_name: str | None = None):
            super().__init__(model_name=args.matting)

    rembg.BiRefNet = LocalMatting

    import inference as px
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    from giq_3d_child.dinov3 import TimmDINOv3

    image_conditioned_proj.DINOv3ViTModel = TimmDINOv3
    for config in px.IMAGE_COND_CONFIGS.values():
        config["model_name"] = args.encoder
    pipeline = px.init_pipeline(args.weights, low_vram=args.low_vram)
    return px, pipeline


def _camera(px, args, image, fov):
    """The camera the picture was taken with: given, or estimated by MoGe."""
    import tempfile

    import torch

    resolution = 512
    if fov is not None:
        distance = px.distance_from_fov(
            fov, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, resolution - 1]), 1.0, resolution
        )["distance_from_x"]
        return {"camera_angle_x": fov, "distance": distance, "mesh_scale": 1.0}
    # MoGe is on the card only for this: both it and a flow model at once
    # is what low-VRAM mode exists to avoid.
    moge = px.load_moge_model(device="cuda", model_name=os.path.join(args.camera, "model.pt"))
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            image.save(f.name)
            return px.get_camera_params_wild_moge(
                f.name,
                moge,
                device="cuda",
                mesh_scale=1.0,
                extend_pixel=0,
                image_resolution=resolution,
            )
    finally:
        moge.cpu()
        del moge
        torch.cuda.empty_cache()


def _run(models, args, task: dict) -> dict:
    import numpy as np
    import o_voxel
    import torch
    from PIL import Image

    px, pipeline = models
    inputs = task_inputs(task)
    try:
        picture = Image.open(io.BytesIO(inputs["image"]))
        picture.load()
    except Exception:
        raise ValueError("'image_b64' is not an image giq can read") from None
    if picture.mode not in ("RGB", "RGBA"):
        picture = picture.convert("RGBA" if "A" in picture.getbands() else "RGB")

    image = pipeline.preprocess_image(picture)
    camera = _camera(px, args, image, inputs["fov"])
    torch.manual_seed(inputs["seed"])
    meshes, (_shape, _texture, grid) = pipeline.run(
        image,
        camera_params=camera,
        seed=inputs["seed"],
        preprocess_image=False,
        return_latent=True,
        pipeline_type=f"{args.resolution}_cascade",
        max_num_tokens=MAX_TOKENS,
    )
    mesh = meshes[0]
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    if inputs["mode"] == "game":
        from giq_3d_child import lowpoly

        glb = lowpoly.export(
            vertices=mesh.vertices.cuda(),
            faces=mesh.faces.cuda(),
            attr_volume=mesh.attrs.cuda(),
            coords=mesh.coords.cuda(),
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=grid,
            aabb=aabb,
            target_faces=inputs["faces"],
            texture_size=inputs["texture_size"],
            close=inputs["close"],
        )
    else:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=grid,
            aabb=aabb,
            decimation_target=inputs["faces"],
            texture_size=inputs["texture_size"],
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )
    glb.apply_transform(np.array(ROTATION, dtype=np.float64))
    data = glb.export(file_type="glb", extension_webp=True)
    del meshes, mesh
    torch.cuda.empty_cache()
    return {
        "id": task.get("id", "unknown"),
        "glb_b64": base64.b64encode(data).decode("ascii"),
        "format": "glb",
        "mode": inputs["mode"],
        "faces": int(len(glb.faces)),
        "vertices": int(len(glb.vertices)),
        "texture_size": inputs["texture_size"],
        "seed": inputs["seed"],
    }


def main() -> None:
    import argparse

    reserve_ipc_stdout()
    parser = argparse.ArgumentParser()
    # --model names the recipe for log lines; the rest is what loads.
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--matting", required=True)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()
    try:
        models = _load(args)
    except Exception as e:  # noqa: BLE001 — report any load failure to parent
        import traceback

        write_startup_error(str(e), traceback.format_exc())
        sys.exit(1)

    def on_run_batch(tasks, params):
        results = []
        for task in tasks:
            try:
                results.append(_run(models, args, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
