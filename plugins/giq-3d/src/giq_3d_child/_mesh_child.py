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

A trellis2 recipe does it in two stages, which a task may take apart.
``texture: false`` stops after the shape: no texture latent, no unwrap, no
bake, and the GLB is geometry only (``full``, the default for a shape, or
the ``game`` cage), returned with the shape latent (``shape_b64``, see
``shape_latent``). A task that carries ``mesh_b64`` is a paint
(``mesh2texture``): the texture latent is sampled for a mesh the client
brings, on the latent it hands back or on its mesh encoded afresh, and
baked onto that mesh's own vertices and faces (see ``paint``). The default
task is the two stages in one, as TRELLIS.2's own ``run`` composes them.

A recipe that carries a segmenter also takes ``parts``: the asset is cut
into the rigid pieces it is made of, and the GLB holds one mesh a piece,
each a game-ready export of its own (see ``_parts``). What moves and how is
not decided here: the result says which pieces touch, and where.

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
# Voxels of surface a game cage may be asked to keep solid whatever is seen
# through: 1 keeps a thin panel, more only thickens it.
MAX_SHELL = 2
# Pixal3D's own cap on the tokens of the high-resolution stage.
MAX_TOKENS = 49152
# The turn Pixal3D gives its export so the asset faces the viewer it was drawn for.
ROTATION = ((-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1))
MAX_IMAGE_BYTES = 40 * 1024 * 1024
# A mesh to paint: what mesh2rig takes, which fits the parent-child pipe.
MAX_MESH_BYTES = 64 * 1024 * 1024
# What a paint is baked at when the task does not say.
PAINT_TEXTURE_SIZE = 2048
# Keys of an image2mesh task that mean nothing to a paint, which keeps the
# mesh's own faces and makes no export of its own.
NOT_PAINTS = (
    "mode",
    "faces",
    "close",
    "shell",
    "parts",
    "part_seeds",
    "texture",
    "fov",
    "views_b64",
)
# How many times a segmentation is drawn when the task does not say, and
# the most it may ask for: each is about 10 s, and past a handful the cuts
# repeat one another.
PART_SEEDS, MAX_PART_SEEDS = 3, 6
# The most pieces an asset is cut into; the smallest beyond it are merged.
MAX_PARTS = 8
# A piece is never asked for fewer faces than this, however small its share.
PART_FACES = 500
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

    texture = task.get("texture", True)
    if not isinstance(texture, bool):
        raise ValueError("'texture' is true or false")
    # A shape is judged by its geometry, so a shape-only task is the
    # generated mesh unless it asks for the cage.
    mode = task.get("mode", "game" if texture else "full")
    if mode not in MODES:
        raise ValueError(f"'mode' must be one of {', '.join(MODES)}, got {mode!r}")
    default, low, high = FACES[mode]
    faces = default if task.get("faces") is None else _integer(task["faces"], "faces")
    if not low <= faces <= high:
        raise ValueError(f"'faces' must be {low}-{high} for a {mode} export, got {faces}")
    size = task.get("texture_size")
    if not texture and size is not None:
        raise ValueError("'texture_size' is a painted export's: a shape has no texture")
    size = TEXTURE_SIZE[mode] if size is None else _integer(size, "texture_size")
    if size not in TEXTURE_SIZES:
        raise ValueError(f"'texture_size' must be one of {TEXTURE_SIZES}, got {size}")
    close = CLOSE if task.get("close") is None else _integer(task["close"], "close")
    if not 0 <= close <= MAX_CLOSE:
        raise ValueError(f"'close' must be 0-{MAX_CLOSE}, got {close}")
    shell = 0 if task.get("shell") is None else _integer(task["shell"], "shell")
    if not 0 <= shell <= MAX_SHELL:
        raise ValueError(f"'shell' must be 0-{MAX_SHELL}, got {shell}")
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
    parts = task.get("parts", False)
    if not isinstance(parts, bool):
        raise ValueError("'parts' is true or false")
    if parts and mode != "game":
        raise ValueError("'parts' is a game export's: each piece is a low-poly cage of its own")
    if parts and not texture:
        raise ValueError("'parts' are painted pieces: a shape-only task takes no 'parts'")
    seeds = task.get("part_seeds")
    seeds = PART_SEEDS if seeds is None else _integer(seeds, "part_seeds")
    if not 1 <= seeds <= MAX_PART_SEEDS:
        raise ValueError(f"'part_seeds' must be 1-{MAX_PART_SEEDS}, got {seeds}")
    return {
        "texture": texture,
        "parts": parts,
        "part_seeds": seeds,
        "views" if multi else "image": data,
        "mode": mode,
        "faces": faces,
        "texture_size": size if texture else None,
        "close": close,
        "shell": shell,
        "seed": seed,
        "fov": fov,
    }


def paint_inputs(task: dict) -> dict:
    """A paint task's request, checked. Raises ValueError naming the field that is wrong.

    The mesh and the shape are only decoded from base64 here; whether they
    are a mesh and a shape this recipe made is the paint's to find out.
    """
    from giq_3d_child import shape_latent

    mesh = task.get("mesh_b64")
    if not isinstance(mesh, str) or not mesh:
        raise ValueError("a paint needs a 'mesh_b64': the mesh to paint, as GLB or OBJ")
    try:
        mesh = base64.b64decode(mesh.split(",", 1)[-1], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("'mesh_b64' is not base64") from None
    if len(mesh) > MAX_MESH_BYTES:
        raise ValueError(f"'mesh_b64' is larger than {MAX_MESH_BYTES // 2**20} MB")
    if task.get("image_b64") is None or task.get("image_b64") == "":
        raise ValueError("a paint needs an 'image_b64': the picture to paint the mesh from")
    image = _picture(task["image_b64"], "'image_b64'")
    for key in NOT_PAINTS:
        if task.get(key) is not None:
            raise ValueError(f"'{key}' is image2mesh's: a paint keeps the mesh it is given")
    shape = task.get("shape_b64")
    shape = None if shape is None else shape_latent.decode(shape)
    size = task.get("texture_size")
    size = PAINT_TEXTURE_SIZE if size is None else _integer(size, "texture_size")
    if size not in TEXTURE_SIZES:
        raise ValueError(f"'texture_size' must be one of {TEXTURE_SIZES}, got {size}")
    seed = 42 if task.get("seed") is None else _integer(task["seed"], "seed")
    if not 0 <= seed < 2**32:
        raise ValueError("'seed' must be 0 to 4294967295")
    unwrap = task.get("unwrap", False)
    if not isinstance(unwrap, bool):
        raise ValueError("'unwrap' is true or false")
    return {
        "mesh": mesh,
        "image": image,
        "shape": shape,
        "texture_size": size,
        "seed": seed,
        "unwrap": unwrap,
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


def _trellis2_shape(pipeline, args, picture, seed: int):
    """TRELLIS.2's first stage: the sparse structure and the shape latent.

    The steps and their order are those of the pipeline's own ``run`` for a
    cascade, so a seed draws the same shape whether the texture follows in
    the same task or not. Returns the picture's 1024 conditioning (the
    texture stage's), the shape latent and the resolution it decodes at,
    which the cascade lowers when the object has more tokens than it takes.
    """
    import torch

    models = pipeline.models
    with torch.no_grad():
        image = pipeline.preprocess_image(picture)
        torch.manual_seed(seed)
        cond_512 = pipeline.get_cond([image], 512)
        cond_1024 = pipeline.get_cond([image], 1024)
        coords = pipeline.sample_sparse_structure(cond_512, 32, 1, {})
        shape, resolution = pipeline.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            models["shape_slat_flow_model_512"],
            models["shape_slat_flow_model_1024"],
            512,
            args.resolution,
            coords,
            {},
            MAX_TOKENS,
        )
    return cond_1024, shape, resolution


def _trellis2_texture(pipeline, cond_1024, shape):
    """TRELLIS.2's second stage: the texture latent, sampled on a shape latent."""
    import torch

    with torch.no_grad():
        return pipeline.sample_tex_slat(
            cond_1024, pipeline.models["tex_slat_flow_model_1024"], shape, {}
        )


def _decoded(pipeline, shape, texture, resolution: int):
    """The two latents decoded: the mesh with its PBR volume, as ``run`` returns it."""
    import torch

    torch.cuda.empty_cache()
    with torch.no_grad():
        return pipeline.decode_latent(shape, texture, resolution)[0]


def _generate_trellis2(pipeline, args, inputs: dict):
    """One picture to TRELLIS.2's mesh: both stages, as its own ``run``
    composes them. It needs no camera: it does not keep the picture's pose,
    and stands the object upright in a frame of its own."""
    if inputs["fov"] is not None:
        raise ValueError("'fov' is not TRELLIS.2's: it does not model the picture's camera")
    cond, shape, resolution = _trellis2_shape(
        pipeline, args, _open(inputs["image"], "'image_b64'"), inputs["seed"]
    )
    texture = _trellis2_texture(pipeline, cond, shape)
    mesh = _decoded(pipeline, shape, texture, resolution)
    return mesh, mesh.layout, resolution


def _decoder_name(args) -> str:
    """What a shape latent is tied to: the shape decoder the recipe loads,
    by its checkpoint's name and size, so new weights refuse old shapes."""
    import json

    with open(os.path.join(args.weights, "pipeline.json")) as f:
        name = json.load(f)["args"]["models"]["shape_slat_decoder"]
    path = os.path.join(args.weights, name + ".safetensors")
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return f"{name}:{size}"


def _shape_only(pipeline, args, task: dict, inputs: dict) -> dict:
    """The shape and nothing after it: geometry, and the latent to paint it from."""
    import numpy as np
    import torch
    import trimesh

    from giq_3d_child import lowpoly, shape_latent

    if inputs["fov"] is not None:
        raise ValueError("'fov' is not TRELLIS.2's: it does not model the picture's camera")
    _cond, shape, resolution = _trellis2_shape(
        pipeline, args, _open(inputs["image"], "'image_b64'"), inputs["seed"]
    )
    with torch.no_grad():
        meshes, _subs = pipeline.decode_shape_slat(shape, resolution)
    mesh = meshes[0]
    # As decode_latent does before an export.
    mesh.fill_holes()
    if inputs["mode"] == "game":
        sv, sf = lowpoly.skin(mesh.vertices.cuda(), inputs["close"], shell=inputs["shell"])
        v, f = lowpoly.simplify(sv, sf, inputs["faces"])
    else:
        v, f = _full_geometry(mesh.vertices, mesh.faces, resolution, inputs["faces"])
    glb = trimesh.Trimesh(vertices=lowpoly.glb_axes(v), faces=np.asarray(f), process=False)
    data = glb.export(file_type="glb")
    assert isinstance(data, bytes)
    packed = shape_latent.pack(
        shape.coords[:, 1:].cpu().numpy(),
        shape.feats.float().cpu().numpy(),
        recipe=args.model,
        decoder=_decoder_name(args),
        resolution=resolution,
        seed=inputs["seed"],
    )
    del meshes, mesh, shape
    torch.cuda.empty_cache()
    return {
        "id": task.get("id", "unknown"),
        "glb_b64": base64.b64encode(data).decode("ascii"),
        "format": "glb",
        "mode": inputs["mode"],
        "texture": False,
        "faces": int(len(glb.faces)),
        "vertices": int(len(glb.vertices)),
        "seed": inputs["seed"],
        "shape_b64": shape_latent.encode(packed),
    }


def _full_geometry(vertices, faces, grid: int, target: int):
    """The stock full export's geometry, without its unwrap and bake: holes
    filled, remeshed on the generator's grid, simplified to ``target``."""
    import cumesh
    import torch

    box = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device="cuda")
    mesh = cumesh.CuMesh()
    mesh.init(vertices.cuda(), faces.cuda())
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()
    bvh = cumesh.cuBVH(vertices, faces)
    band = 1
    mesh.init(
        *cumesh.remeshing.remesh_narrow_band_dc(
            vertices,
            faces,
            center=box.mean(dim=0),
            scale=(grid + 3 * band) / grid * float((box[1] - box[0]).max()),
            resolution=grid,
            band=band,
            project_back=0,
            verbose=False,
            bvh=bvh,
        )
    )
    mesh.simplify(target, verbose=False)
    vertices, faces = mesh.read()
    return vertices.cpu().numpy(), faces.cpu().numpy()


_texturer: list = []


def _texturing(pipeline, args):
    """TRELLIS.2's texturing pipeline on the generator's own models.

    It wants a shape encoder and the texture flow models and decoder the
    generator already holds; only the encoder (0.7 GB) is loaded, on first
    use, and the rest is shared rather than loaded a second time.
    """
    if _texturer:
        return _texturer[0]
    import json

    from trellis2 import models
    from trellis2.pipelines import Trellis2TexturingPipeline

    config = os.path.join(args.weights, "texturing_pipeline.json")
    if not os.path.isfile(config):
        raise FileNotFoundError(f"the texturing pipeline's config is missing: {config}")
    with open(config) as f:
        names = json.load(f)["args"]["models"]
    shared = ("tex_slat_decoder", "tex_slat_flow_model_512", "tex_slat_flow_model_1024")
    texturer = Trellis2TexturingPipeline(
        models={
            "shape_slat_encoder": models.from_pretrained(
                os.path.join(args.weights, names["shape_slat_encoder"])
            ),
            **{name: pipeline.models[name] for name in shared},
        },
        tex_slat_sampler=pipeline.tex_slat_sampler,
        tex_slat_sampler_params=pipeline.tex_slat_sampler_params,
        shape_slat_normalization=pipeline.shape_slat_normalization,
        tex_slat_normalization=pipeline.tex_slat_normalization,
        image_cond_model=pipeline.image_cond_model,
        rembg_model=pipeline.rembg_model,
        low_vram=args.low_vram,
    )
    texturer.cuda()
    _texturer.append(texturer)
    return texturer


# Where a paint's bake runs; giq's suite runs the wrapper on the CPU.
DEVICE = "cuda"
# The PBR channels TRELLIS.2's texture decoder writes, in its order.
PBR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def _paint(models, args, task: dict) -> dict:
    """A mesh the client brings, painted from a picture (see ``paint``).

    Everything but the model is here: the request, the mesh read and
    checked, the bake onto its own vertices and the GLB. The model is
    ``_sampler``'s.
    """
    import numpy as np
    import torch

    from giq_3d_child import paint, shape_latent

    if args.pipeline != "trellis2":
        raise ValueError("painting a mesh is a trellis2 recipe's: Pixal3D has no texturing stage")
    inputs = paint_inputs(task)
    mesh = paint.read(inputs["mesh"])
    shape = None
    if inputs["shape"] is not None:
        # Before the model runs: a stale shape is the client's to make again.
        shape = shape_latent.unpack(inputs["shape"], recipe=args.model, decoder=_decoder_name(args))
    picture = _open(inputs["image"], "'image_b64'")
    frame, sample = _sampler(models, args, mesh, picture, shape, inputs["seed"])
    glb, info = paint.paint(
        mesh,
        frame,
        inputs["texture_size"],
        sample,
        PBR_LAYOUT,
        unwrap=inputs["unwrap"],
        device=DEVICE,
    )
    data = glb.export(file_type="glb", extension_webp=True)
    torch.cuda.empty_cache()
    return {
        "id": task.get("id", "unknown"),
        "glb_b64": base64.b64encode(data).decode("ascii"),
        "format": "glb",
        "faces": int(len(glb.faces)),
        "vertices": int(len(np.asarray(glb.vertices))),
        "texture_size": inputs["texture_size"],
        "seed": inputs["seed"],
        "shape": "given" if shape is not None else "encoded",
        **info,
    }


def _sampler(models, args, mesh, picture, shape, seed: int):
    """The model's half of a paint: the frame the mesh is painted in, and
    what colour lies at a point of it (see ``paint.paint``).

    Given a shape latent, the texture is sampled on it and decoded with the
    generated surface, and each texel reads what lies behind it there, its
    normal included, as the game export bakes. Without one, the mesh itself
    is encoded into a shape latent first (TRELLIS.2's texturing pipeline),
    and each texel reads the texture where it is.
    """
    import torch

    from giq_3d_child import lowpoly, paint

    _px, pipeline = models
    if shape is not None:
        from trellis2.modules.sparse import SparseTensor

        head, coords, feats = shape
        latent = SparseTensor(
            feats=torch.from_numpy(feats).cuda(),
            coords=torch.cat(
                [torch.zeros(len(coords), 1, dtype=torch.int32), torch.from_numpy(coords)], 1
            ).cuda(),
        )
        with torch.no_grad():
            cond = pipeline.get_cond([pipeline.preprocess_image(picture)], 1024)
        torch.manual_seed(seed)
        texture = _trellis2_texture(pipeline, cond, latent)
        full = _decoded(pipeline, latent, texture, head["resolution"])
        grid = head["resolution"]
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device="cuda")
        high_v, high_f = full.vertices.cuda(), full.faces.cuda()
        attrs, coords = full.attrs.cuda(), full.coords.cuda()

        def behind(P, Ns):
            return lowpoly.lookup(high_v, high_f, attrs, coords, grid, aabb, P, Ns)

        return "generator", behind

    texturer = _texturing(pipeline, args)
    canonical = paint.to_model(mesh.vertices, "canonical")
    with torch.no_grad():
        cond = texturer.get_cond([texturer.preprocess_image(picture)], 1024)
        torch.manual_seed(seed)
        encoded = texturer.encode_shape_slat(_Shape(canonical, mesh.faces), resolution=1024)
        texture = texturer.sample_tex_slat(
            cond, texturer.models["tex_slat_flow_model_1024"], encoded, {}
        )
        voxels = texturer.decode_tex_slat(texture)

    from flex_gemm.ops.grid_sample import grid_sample_3d

    def at(P, Ns):
        # The texture voxels sit on the mesh's own surface: read them where
        # each texel is, as upstream's texturing pipeline does.
        values = grid_sample_3d(
            voxels.feats,
            voxels.coords,
            shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
            grid=((P + 0.5) * 1024).reshape(1, -1, 3),
            mode="trilinear",
        )
        return values.reshape(-1, voxels.feats.shape[1]), None

    return "canonical", at


class _Shape:
    """What the texturing pipeline's encoder reads of a mesh: its arrays."""

    def __init__(self, vertices, faces):
        self.vertices, self.faces = vertices, faces


_segmenter: list = []


def _parts(pipeline, args, mesh, layout, grid: int, picture, inputs: dict):
    """The asset as its pieces: a scene of one game-ready mesh a piece, and what touches.

    The segmenter paints the asset one colour a part, once a seed; the
    paintings are laid over one another, so a cut any of them found is a
    cut. The colours go onto the generated mesh at its full detail, and
    each piece is exported on its own from the generator's volume, its
    faces a share of ``faces`` by its share of the surface.
    """
    import numpy as np
    import o_voxel
    import torch
    import trimesh
    from trellis2 import models

    from giq_3d_child import lowpoly, parts
    from giq_3d_child.segment import GRID, Segmenter, condition

    box = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    # The label tidying walks every edge of the mesh: bounded like the stock export.
    mesh.simplify(16777216)
    full = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=layout,
        grid_size=grid,
        aabb=box,
        decimation_target=100000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=False,
    )
    if not _segmenter:
        decoders = {
            "shape": pipeline.models["shape_slat_decoder"],
            "tex": pipeline.models["tex_slat_decoder"],
        }
        _segmenter.append(Segmenter(args.weights, args.segmenter, models, decoders))
    segmenter = _segmenter[0]
    segmenter.to("cuda")
    pipeline.image_cond_model.cuda()
    try:
        laid_out, (centre, scale) = segmenter.lay_out(trimesh.Scene(full))
        cond = condition(pipeline.image_cond_model, pipeline.preprocess_image(picture))
        vertices = mesh.vertices.cpu().numpy().astype(np.float64)
        faces = mesh.faces.cpu().numpy().astype(np.int64)

        def stock_axes(p: np.ndarray) -> np.ndarray:
            return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)

        a, b, c = (vertices[faces[:, i]] for i in range(3))
        area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
        centroids = (stock_axes((a + b + c) / 3) - centre) * scale
        del a, b, c
        labelings = []
        for n in range(inputs["part_seeds"]):
            where, colours = segmenter.paint(laid_out, cond, inputs["seed"] + n)
            groups = parts.colour_groups(colours)
            labelings.append(parts.onto_faces(centroids, (where + 0.5) / GRID - 0.5, groups))
    finally:
        # The decoders are the pipeline's: they go back where low-VRAM mode keeps them.
        segmenter.to("cpu" if args.low_vram else "cuda")
        if args.low_vram:
            pipeline.image_cond_model.cpu()
        torch.cuda.empty_cache()
    labels = parts.tidy(faces, parts.overlay(labelings), MAX_PARTS)
    labels = parts.tidy(faces, parts.smooth(centroids, area, labels), MAX_PARTS)

    scene, pieces = trimesh.Scene(), []
    for part in range(int(labels.max()) + 1):
        used, inverse = np.unique(faces[labels == part], return_inverse=True)
        share = float(area[labels == part].sum() / area.sum())
        target = max(PART_FACES, int(round(inputs["faces"] * share)))
        cage = lowpoly.export(
            vertices=mesh.vertices[torch.from_numpy(used).to(mesh.vertices.device)].cuda(),
            faces=torch.from_numpy(inverse.reshape(-1, 3).astype(np.int32)).cuda(),
            attr_volume=mesh.attrs.cuda(),
            coords=mesh.coords.cuda(),
            attr_layout=layout,
            grid_size=grid,
            aabb=box,
            target_faces=target,
            # A small piece has a small share of the texels to fill, too.
            texture_size=inputs["texture_size"] if share >= 0.2 else inputs["texture_size"] // 2,
            close=min(inputs["close"], 2),
            part=True,
        )
        name = f"part_{part}"
        scene.add_geometry(cage, node_name=name, geom_name=name)
        pieces.append(
            {
                "name": name,
                "faces": int(len(cage.faces)),
                "vertices": int(len(cage.vertices)),
                "share": round(share, 4),
            }
        )
    touching = [
        {
            "parts": c["parts"],
            "points": c["points"],
            "centre": [round(float(x), 5) for x in c["centre"]],
        }
        for c in parts.contacts(stock_axes(vertices), faces, labels)
    ]
    return scene, pieces, touching


def _run(models, args, task: dict) -> dict:
    px, pipeline = models
    multi = args.views == "multi"
    inputs = task_inputs(task, multi)
    if not inputs["texture"]:
        if args.pipeline != "trellis2":
            raise ValueError("'texture': false is a trellis2 recipe's: Pixal3D runs as one stage")
        return _shape_only(pipeline, args, task, inputs)

    import numpy as np
    import o_voxel
    import torch

    if args.pipeline == "trellis2":
        mesh, layout, grid = _generate_trellis2(pipeline, args, inputs)
        meshes = [mesh]
    else:
        generate = _generate_multi if multi else _generate_single
        meshes, (_shape, _texture, grid) = generate(px, pipeline, args, inputs)
        mesh, layout = meshes[0], pipeline.pbr_attr_layout
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    pieces = touching = None
    if inputs["parts"]:
        if not args.segmenter:
            raise ValueError("'parts' needs a recipe that carries a segmenter")
        picture = _open(inputs["image"], "'image_b64'")
        glb, pieces, touching = _parts(pipeline, args, mesh, layout, grid, picture, inputs)
    elif inputs["mode"] == "game":
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
            shell=inputs["shell"],
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
    out = {
        "id": task.get("id", "unknown"),
        "glb_b64": base64.b64encode(data).decode("ascii"),
        "format": "glb",
        "mode": inputs["mode"],
        "faces": sum(p["faces"] for p in pieces) if pieces else int(len(glb.faces)),
        "vertices": sum(p["vertices"] for p in pieces) if pieces else int(len(glb.vertices)),
        "texture_size": inputs["texture_size"],
        "seed": inputs["seed"],
    }
    if pieces:
        out |= {"parts": pieces, "contacts": touching}
    return out


def handler(task: dict):
    """What runs a task. One that brings a mesh is a paint (mesh2texture):
    the job's modality does not reach the child, the task's shape does."""
    return _paint if "mesh_b64" in task else _run


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
    # A part-segmentation checkpoint; with it, a task may ask for `parts`.
    parser.add_argument("--segmenter", default=None)
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
                results.append(handler(task)(models, args, task))
            except Exception as e:  # noqa: BLE001 — per-task error envelope
                results.append({"id": task.get("id", "unknown"), "error": str(e)})
        return results

    run_ipc_child_loop(on_run_batch)


if __name__ == "__main__":
    main()
