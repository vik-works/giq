# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Image-to-mesh child process: Pixal3D or TRELLIS.2, and the export of what
they make.

Invoked by ``MeshAdapter`` via ``<pixal3d python> -u -m
giq_3d_child._mesh_child --model <recipe> --weights <checkpoint root>
--source <Pixal3D checkout> --encoder <DINOv3 folder> --camera <MoGe folder>
--matting <BiRefNet folder> --resolution <1024|1536> --views <single|multi>
[--low-vram]``. Loads the flow models, decoders and the four DINOv3 feature
extractors once: the single-view set or the multi-view set, which are
different checkpoints of the same repository.

Pixal3D is a checkout, not a package: its code is imported from ``--source``.
Three things are put in place before it is: giq's UV rasteriser under the
name ``nvdiffrast.torch`` (the export's texture bake is all Pixal3D asks of
nvdiffrast, whose licence allows research and evaluation only); timm's
DINOv3 file for every feature extractor, its parameters renamed as it loads
(upstream names a mirror on the hub); and BiRefNet's own weights for the
background remover (upstream names RMBG-2.0, which is non-commercial). The
hub is offline, so anything missing fails here rather than being fetched.

``--pipeline trellis2`` loads Microsoft's TRELLIS.2 in Pixal3D's place: the
backbone Pixal3D is built on, from its own checkout and weights, with the
same substitutions. It takes one picture and no camera, and where Pixal3D
keeps the picture's pose it stands the object upright in a frame of its own,
at some cost in likeness.

Per task, single view: the image is cut out of its background unless it
already carries an alpha channel, the camera it was taken with is estimated
(or given, as ``fov``), and the cascade generates the mesh posed the way the
picture was taken. Multi view: four views a quarter turn apart round the
object (front, right, back, left) are cut out and fitted to the rig those
cameras make (see ``rig``), and the mesh comes out in the rig's frame,
upright and squared to the axes. Either way it is exported as GLB by
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

from giq_3d_child.rig import VIEW_NAMES, fit_views  # noqa: E402
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
# The rig's cameras, as Pixal3D ships them with its example views.
RIG_FILE = os.path.join("assets", "mv_images", "example", "transforms.json")


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"'{name}' must be a whole number")
    return int(value)


def _picture(value, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a base64 image")
    try:
        data = base64.b64decode(value.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"{name} is not base64") from None
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{name} is larger than {MAX_IMAGE_BYTES // 2**20} MB")
    return data


def task_inputs(task: dict, multi: bool = False) -> dict:
    """A task's request, checked. Raises ValueError naming the field that is wrong.

    ``multi`` is the recipe's: a multi-view recipe takes ``views_b64``, a
    single-view one ``image_b64``, and neither takes the other's.
    """
    if multi:
        views = task.get("views_b64")
        if not isinstance(views, list) or len(views) != len(VIEW_NAMES):
            raise ValueError(
                f"task needs 'views_b64': {len(VIEW_NAMES)} images of the object, "
                f"in the order {', '.join(VIEW_NAMES)}"
            )
        data = [_picture(v, f"'views_b64' {n}") for v, n in zip(views, VIEW_NAMES, strict=True)]
    else:
        if task.get("image_b64") is None or task.get("image_b64") == "":
            raise ValueError("task needs an 'image_b64' of the object")
        data = _picture(task["image_b64"], "'image_b64'")

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
    if fov is not None and multi:
        raise ValueError("'fov' is a single picture's: the views' cameras are the rig's")
    if fov is not None:
        if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not 0.05 <= fov <= 2.5:
            raise ValueError("'fov' is the camera's horizontal field of view in radians, 0.05-2.5")
        fov = float(fov)
    return {
        "views" if multi else "image": data,
        "mode": mode,
        "faces": faces,
        "texture_size": size,
        "close": close,
        "seed": seed,
        "fov": fov,
    }


def _load_trellis2(args):
    """TRELLIS.2's pipeline, with the same substitutions as Pixal3D's.

    One more: its pipeline names its structure decoder in another repository
    on the hub. That repository is the recipe's `structure` part, and the
    name is pointed at it.
    """
    for path, what in (
        (os.path.join(args.source, "trellis2", "pipelines"), "the TRELLIS.2 checkout"),
        (os.path.join(args.weights, "pipeline.json"), "the weights"),
        (os.path.join(args.encoder, "model.safetensors"), "the DINOv3 encoder"),
        (args.matting, "the background remover"),
        (os.path.join(args.structure or "", "ckpts"), "the structure decoder's repository"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} is missing: {path}")

    from giq_3d_child import uvraster

    uvraster.install()
    sys.path.insert(0, args.source)

    from trellis2 import models
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import rembg

    from giq_3d_child.dinov3 import from_timm

    upstream = rembg.BiRefNet

    class LocalMatting(upstream):
        """BiRefNet from the recipe's folder, whatever the pipeline's config names."""

        def __init__(self, model_name: str | None = None):
            super().__init__(model_name=args.matting)

    class LocalDINOv3:
        """The recipe's encoder, whichever repository the pipeline's config names."""

        @staticmethod
        def from_pretrained(_name: str, *_, **__):
            return from_timm(args.encoder)

    rembg.BiRefNet = LocalMatting
    image_feature_extractor.DINOv3ViTModel = LocalDINOv3
    hub = "microsoft/TRELLIS-image-large/"
    load = models.from_pretrained

    def from_pretrained(path: str, **kwargs):
        if path.startswith(hub):
            path = os.path.join(args.structure, path[len(hub) :])
        return load(path, **kwargs)

    models.from_pretrained = from_pretrained

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.low_vram = args.low_vram
    pipeline.cuda()
    return None, pipeline


def _load(args):
    """Pixal3D's pipeline, with the three substitutions the module docstring names."""
    if args.pipeline == "trellis2":
        return _load_trellis2(args)
    for path, what in (
        (args.source, "the Pixal3D checkout"),
        (args.weights, "the weights"),
        (os.path.join(args.encoder, "model.safetensors"), "the DINOv3 encoder"),
        (os.path.join(args.camera or "", "model.pt"), "the MoGe checkpoint"),
        (args.matting, "the background remover"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} is missing: {path}")
    script = "inference_mv.py" if args.views == "multi" else "inference.py"
    if not os.path.isfile(os.path.join(args.source, script)):
        raise FileNotFoundError(f"{args.source} is not a Pixal3D checkout (no {script})")
    if args.views == "multi" and not os.path.isfile(os.path.join(args.source, RIG_FILE)):
        raise FileNotFoundError(f"{args.source} lacks the rig's cameras ({RIG_FILE})")

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

    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    from giq_3d_child.dinov3 import TimmDINOv3

    image_conditioned_proj.DINOv3ViTModel = TimmDINOv3
    if args.views == "multi":
        import inference_mv as px
    else:
        import inference as px
    for config in px.IMAGE_COND_CONFIGS.values():
        config["model_name"] = args.encoder
    if args.views == "multi":
        pipeline = px.init_pipeline(args.weights, "pipeline_mv.json", low_vram=args.low_vram)
    else:
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


def _open(data: bytes, name: str):
    from PIL import Image

    try:
        picture = Image.open(io.BytesIO(data))
        picture.load()
    except Exception:
        raise ValueError(f"{name} is not an image giq can read") from None
    if picture.mode not in ("RGB", "RGBA"):
        picture = picture.convert("RGBA" if "A" in picture.getbands() else "RGB")
    return picture


def _generate_single(px, pipeline, args, inputs: dict):
    import torch

    image = pipeline.preprocess_image(_open(inputs["image"], "'image_b64'"))
    camera = _camera(px, args, image, inputs["fov"])
    torch.manual_seed(inputs["seed"])
    return pipeline.run(
        image,
        camera_params=camera,
        seed=inputs["seed"],
        preprocess_image=False,
        return_latent=True,
        pipeline_type=f"{args.resolution}_cascade",
        max_num_tokens=MAX_TOKENS,
    )


def _generate_multi(px, pipeline, args, inputs: dict):
    import json
    import shutil
    import tempfile

    import numpy as np
    import torch

    # A view is used as it is when it carries a cut-out of its own, like a
    # single picture is: an alpha channel that is not opaque throughout.
    matte = px.make_rembg(pipeline)
    cutouts = []
    for data, name in zip(inputs["views"], VIEW_NAMES, strict=True):
        picture = _open(data, f"'views_b64' {name}")
        cut = picture.mode == "RGBA" and not np.all(np.asarray(picture)[..., 3] == 255)
        cutouts.append(picture if cut else matte(picture).convert("RGBA"))
    with tempfile.TemporaryDirectory(prefix="giq-pixal3d-views-") as folder:
        rig = os.path.join(folder, "transforms.json")
        shutil.copy(os.path.join(args.source, RIG_FILE), rig)
        with open(rig) as f:
            frames = json.load(f)["frames"]
        for view, frame in zip(fit_views(cutouts), frames, strict=True):
            view.save(os.path.join(folder, frame["file_path"]))
        views = px.load_views(folder, len(VIEW_NAMES), rembg=None)
    torch.manual_seed(inputs["seed"])
    # The sampling Pixal3D's own multi-view script runs with.
    return pipeline.run_mv(
        views,
        seed=inputs["seed"],
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
        return_latent=True,
        pipeline_type=f"{args.resolution}_cascade",
        max_num_tokens=MAX_TOKENS,
    )


def _generate_trellis2(pipeline, args, inputs: dict):
    """One picture to TRELLIS.2's mesh. It needs no camera: it does not keep
    the picture's pose, and stands the object upright in a frame of its own."""
    if inputs["fov"] is not None:
        raise ValueError("'fov' is not TRELLIS.2's: it does not model the picture's camera")
    mesh = pipeline.run(
        _open(inputs["image"], "'image_b64'"),
        seed=inputs["seed"],
        pipeline_type=f"{args.resolution}_cascade",
        max_num_tokens=MAX_TOKENS,
    )[0]
    voxel = mesh.voxel_size
    voxel = float(voxel.flatten()[0]) if hasattr(voxel, "flatten") else float(voxel)
    return mesh, mesh.layout, int(round(1.0 / voxel))


def _run(models, args, task: dict) -> dict:
    import numpy as np
    import o_voxel
    import torch

    px, pipeline = models
    multi = args.views == "multi"
    inputs = task_inputs(task, multi)
    if args.pipeline == "trellis2":
        mesh, layout, grid = _generate_trellis2(pipeline, args, inputs)
        meshes = [mesh]
    else:
        generate = _generate_multi if multi else _generate_single
        meshes, (_shape, _texture, grid) = generate(px, pipeline, args, inputs)
        mesh, layout = meshes[0], pipeline.pbr_attr_layout
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    if inputs["mode"] == "game":
        from giq_3d_child import lowpoly

        glb = lowpoly.export(
            vertices=mesh.vertices.cuda(),
            faces=mesh.faces.cuda(),
            attr_volume=mesh.attrs.cuda(),
            coords=mesh.coords.cuda(),
            attr_layout=layout,
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
            attr_layout=layout,
            grid_size=grid,
            aabb=aabb,
            decimation_target=inputs["faces"],
            texture_size=inputs["texture_size"],
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )
    if args.pipeline == "pixal3d":
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
    # The camera estimator is Pixal3D's; the structure decoder TRELLIS.2's.
    parser.add_argument("--camera", default=None)
    parser.add_argument("--structure", default=None)
    parser.add_argument("--pipeline", choices=("pixal3d", "trellis2"), default="pixal3d")
    parser.add_argument("--matting", required=True)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--views", choices=("single", "multi"), default="single")
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
