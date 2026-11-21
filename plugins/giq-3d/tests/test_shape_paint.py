# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Shape, then paint: the two halves of a trellis2 job, on the CPU.

The model is never run. What is tested is everything around it: which
tasks go where and what they may ask for, the shape artefact a shape-only
job hands out and a paint takes back, and the paint's wrapper, which must
give a mesh back with the vertices, faces and order it went in with.
"""

import base64
import io
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("cv2")

from giq_3d_child import _mesh_child as child  # noqa: E402
from giq_3d_child import paint, shape_latent  # noqa: E402
from giq_3d_child._mesh_child import paint_inputs, task_inputs  # noqa: E402

IMAGE = base64.b64encode(b"\x89PNG not really").decode()


def _png() -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 40, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _character(uv: bool = True, scale: float = 1.7, offset=(0.3, 0.9, -0.2)) -> trimesh.Trimesh:
    """A mesh in a frame of its own: off the origin, not unit-sized, its
    vertices shuffled so order is something a test can see being kept."""
    base = trimesh.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(7)
    order = rng.permutation(len(base.vertices))
    vertices = (base.vertices[order] * [0.4, 1.0, 0.3] * scale + offset).astype(np.float32)
    faces = np.argsort(order)[base.faces]
    visual = None
    if uv:
        visual = trimesh.visual.TextureVisuals(uv=_spherical(vertices))
    return trimesh.Trimesh(vertices, faces, visual=visual, process=False)


def _spherical(p: np.ndarray) -> np.ndarray:
    """A simple atlas in 0..1, round the middle: enough to bake through."""
    p = p - (p.max(0) + p.min(0)) / 2
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    u = (np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(p[:, 2], -1, 1)) / np.pi
    return np.stack([u, v], 1)


def _glb(mesh: trimesh.Trimesh) -> bytes:
    return mesh.export(file_type="glb")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _colour_by_position(seen: list):
    """A stand-in for the model: colour is where the point is in the cube."""

    def sample(P, Ns):
        seen.append(P.clone())
        values = torch.zeros(len(P), 6)
        values[:, 0:3] = P + 0.5
        values[:, 3] = 0.1
        values[:, 4] = 0.8
        values[:, 5] = 1.0
        return values, None

    return sample


# --- routing and validation -------------------------------------------------


def test_a_task_with_a_mesh_is_a_paint_and_any_other_an_image2mesh():
    assert child.handler({"image_b64": IMAGE, "mesh_b64": "Z2xURg=="}) is child._paint
    assert child.handler({"image_b64": IMAGE}) is child._run
    assert child.handler({"image_b64": IMAGE, "texture": False}) is child._run


def test_the_default_task_is_unchanged():
    """Golden: what an existing client's bare task asks for."""
    assert task_inputs({"id": "a", "image_b64": IMAGE}) == {
        "texture": True,
        "parts": False,
        "part_seeds": 3,
        "image": b"\x89PNG not really",
        "mode": "game",
        "faces": 3000,
        "texture_size": 2048,
        "close": 4,
        "shell": 0,
        "seed": 42,
        "fov": None,
    }


def test_a_shape_only_task_is_the_full_mesh_unless_it_asks_for_the_cage():
    shape = task_inputs({"image_b64": IMAGE, "texture": False})
    assert (shape["texture"], shape["mode"], shape["faces"]) == (False, "full", 100000)
    assert shape["texture_size"] is None
    cage = task_inputs({"image_b64": IMAGE, "texture": False, "mode": "game", "faces": 4000})
    assert (cage["mode"], cage["faces"], cage["close"]) == ("game", 4000, 4)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({"texture": "no"}, "'texture' is true or false"),
        ({"texture": False, "texture_size": 2048}, "'texture_size' is a painted export's"),
        ({"texture": False, "mode": "game", "parts": True}, "a shape-only task takes no 'parts'"),
    ],
)
def test_a_wrong_shape_only_task_is_named(task, message):
    with pytest.raises(ValueError, match=message):
        task_inputs({"image_b64": IMAGE, **task})


def test_a_paint_has_its_defaults():
    inputs = paint_inputs({"mesh_b64": _b64(b"glTF...."), "image_b64": IMAGE})
    assert inputs["mesh"] == b"glTF...."
    assert inputs["image"] == b"\x89PNG not really"
    assert (inputs["texture_size"], inputs["seed"], inputs["unwrap"], inputs["shape"]) == (
        2048,
        42,
        False,
        None,
    )


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({"image_b64": IMAGE}, "a paint needs a 'mesh_b64'"),
        ({"mesh_b64": "Z2xURg=="}, "a paint needs an 'image_b64'"),
        ({"mesh_b64": "not base64!", "image_b64": IMAGE}, "'mesh_b64' is not base64"),
        ({"mesh_b64": "Z2xURg==", "image_b64": IMAGE, "faces": 4000}, "'faces' is image2mesh's"),
        ({"mesh_b64": "Z2xURg==", "image_b64": IMAGE, "mode": "game"}, "'mode' is image2mesh's"),
        ({"mesh_b64": "Z2xURg==", "image_b64": IMAGE, "texture_size": 3000}, "'texture_size'"),
        ({"mesh_b64": "Z2xURg==", "image_b64": IMAGE, "unwrap": 1}, "'unwrap' is true or false"),
        ({"mesh_b64": "Z2xURg==", "image_b64": IMAGE, "seed": -2}, "'seed' must be 0"),
        ({"mesh_b64": "Z2xURg==", "image_b64": IMAGE, "shape_b64": "!"}, "'shape_b64' is not"),
    ],
)
def test_a_wrong_paint_is_named(task, message):
    with pytest.raises(ValueError, match=message):
        paint_inputs(task)


def test_pixal3d_neither_stops_at_the_shape_nor_paints():
    args = SimpleNamespace(pipeline="pixal3d", views="single")
    with pytest.raises(ValueError, match="'texture': false is a trellis2 recipe's"):
        child._run((None, None), args, {"image_b64": IMAGE, "texture": False})
    with pytest.raises(ValueError, match="painting a mesh is a trellis2 recipe's"):
        child._paint((None, None), args, {"image_b64": IMAGE, "mesh_b64": "Z2xURg=="})


# --- the shape artefact ----------------------------------------------------


def _latent(tokens: int = 500):
    rng = np.random.default_rng(3)
    coords = rng.integers(0, 64, size=(tokens, 3))
    feats = rng.normal(0, 5, size=(tokens, 32)).astype(np.float32)
    return coords, feats


def test_a_shape_comes_back_as_it_went_out():
    coords, feats = _latent()
    data = shape_latent.pack(
        coords, feats, recipe="trellis2", decoder="dec:1", resolution=1024, seed=9
    )
    text = shape_latent.encode(data)
    head, c, f = shape_latent.unpack(shape_latent.decode(text), recipe="trellis2", decoder="dec:1")
    assert (head["resolution"], head["seed"], head["tokens"]) == (1024, 9, 500)
    assert np.array_equal(c, coords) and np.array_equal(f, feats)


def test_a_full_latent_stays_a_size_a_job_can_carry():
    # The 1536 cascade's cap on its tokens, and a 1024 latent's usual size beyond it.
    coords, feats = _latent(49152)
    data = shape_latent.pack(coords, feats, recipe="r", decoder="d", resolution=1024, seed=1)
    # Random values do not compress; a real latent does a little.
    assert len(data) < 7 * 2**20
    assert len(shape_latent.encode(data)) < shape_latent.MAX_BYTES


@pytest.mark.parametrize(
    ("made", "message"),
    [
        ({"recipe": "trellis2-parts"}, "was made by 'trellis2-parts', not 'trellis2'"),
        ({"decoder": "dec:2"}, "another shape decoder"),
    ],
)
def test_a_shape_from_elsewhere_is_refused_as_stale(made, message):
    coords, feats = _latent()
    fields = {"recipe": "trellis2", "decoder": "dec:1", **made}
    data = shape_latent.pack(coords, feats, resolution=1024, seed=1, **fields)
    with pytest.raises(shape_latent.StaleShape, match=message):
        shape_latent.unpack(data, recipe="trellis2", decoder="dec:1")


def test_a_shape_of_another_format_is_refused_as_stale(monkeypatch):
    coords, feats = _latent()
    monkeypatch.setattr(shape_latent, "VERSION", 0)
    data = shape_latent.pack(coords, feats, recipe="r", decoder="d", resolution=1024, seed=1)
    monkeypatch.undo()
    with pytest.raises(shape_latent.StaleShape, match="format 0"):
        shape_latent.unpack(data, recipe="r", decoder="d")


def test_a_damaged_shape_is_refused():
    coords, feats = _latent()
    data = shape_latent.pack(coords, feats, recipe="r", decoder="d", resolution=1024, seed=1)
    for broken in (data[:-40], data[:-1] + bytes([data[-1] ^ 1]), b"GIQSHAPX" + data[8:], b""):
        with pytest.raises(ValueError):
            shape_latent.unpack(broken, recipe="r", decoder="d")


# --- reading a mesh ----------------------------------------------------------


def test_a_glb_is_read_as_stored():
    mesh = _character()
    read = paint.read(_glb(mesh))
    assert np.array_equal(read.vertices, mesh.vertices)
    assert np.array_equal(read.faces, mesh.faces)
    assert np.allclose(read.uv, mesh.visual.uv, atol=1e-6)


def test_an_obj_is_read():
    mesh = _character(uv=False)
    read = paint.read(mesh.export(file_type="obj").encode())
    assert read.faces.shape == mesh.faces.shape and read.uv is None
    assert np.allclose(read.vertices, mesh.vertices, atol=1e-6)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"not a mesh at all", "must be a GLB or an OBJ"),
        (b"glTF\x02\x00\x00\x00garbage", "is not a GLB giq can read"),
    ],
)
def test_a_bad_mesh_is_named(data, message):
    with pytest.raises(ValueError, match=message):
        paint.read(data)


def test_two_meshes_are_not_one():
    scene = trimesh.Scene([_character(), trimesh.creation.box()])
    with pytest.raises(ValueError, match="holds 2 meshes"):
        paint.read(scene.export(file_type="glb"))


def test_uvs_outside_the_atlas_are_refused():
    mesh = _character()
    mesh.visual.uv = mesh.visual.uv * 2.0
    with pytest.raises(ValueError, match="UVs outside 0..1"):
        paint.read(_glb(mesh))


def test_a_placed_mesh_is_read_where_its_node_puts_it():
    mesh = _character()
    scene = trimesh.Scene()
    scene.add_geometry(mesh, transform=trimesh.transformations.translation_matrix([0, 2, 0]))
    read = paint.read(scene.export(file_type="glb"))
    assert np.allclose(read.vertices, mesh.vertices + [0, 2, 0], atol=1e-6)


# --- frames ------------------------------------------------------------------


def test_the_generator_frame_undoes_the_exports_turn():
    from giq_3d_child import lowpoly

    p = np.random.default_rng(1).uniform(-0.5, 0.5, (50, 3))
    assert np.allclose(paint.generator_axes(lowpoly.glb_axes(torch.from_numpy(p))), p)


def test_the_canonical_frame_is_upstreams_cube():
    v = _character().vertices.astype(np.float64)
    m = paint.to_model(v, "canonical")
    assert m.min() >= -0.5 and m.max() <= 0.5
    assert np.isclose((m.max(0) - m.min(0)).max(), 0.99999)
    # Upstream's preprocess_mesh, as its code reads.
    low, high = v.min(0), v.max(0)
    want = (v - (low + high) / 2) * (0.99999 / (high - low).max())
    want = np.stack([want[:, 0], -want[:, 2], want[:, 1]], 1)
    assert np.allclose(m, want)


# --- the paint wrapper ------------------------------------------------------


def _args(tmp_path, decoder_bytes: int = 100):
    weights = tmp_path / "weights"
    (weights / "ckpts").mkdir(parents=True)
    (weights / "pipeline.json").write_text(
        '{"args": {"models": {"shape_slat_decoder": "ckpts/shape_dec"}}}'
    )
    (weights / "ckpts" / "shape_dec.safetensors").write_bytes(b"\0" * decoder_bytes)
    return SimpleNamespace(pipeline="trellis2", model="trellis2", weights=str(weights))


@pytest.fixture
def on_cpu(monkeypatch):
    monkeypatch.setattr(child, "DEVICE", "cpu")


def _model(frame: str, seen: list, given: list | None = None):
    def sampler(models, args, mesh, picture, shape, seed):
        if given is not None:
            given.append(shape)
        return frame, _colour_by_position(seen)

    return sampler


@pytest.mark.parametrize("frame", ["canonical", "generator"])
def test_a_painted_mesh_keeps_its_vertices_faces_and_uvs(tmp_path, monkeypatch, on_cpu, frame):
    # A shape job's mesh is in the generator's cube already; a client's is anywhere.
    mesh = _character(scale=0.45, offset=(0, 0, 0)) if frame == "generator" else _character()
    seen: list = []
    monkeypatch.setattr(child, "_sampler", _model(frame, seen))
    task = {"id": "p", "mesh_b64": _b64(_glb(mesh)), "image_b64": _png(), "texture_size": 1024}
    result = child._paint((None, None), _args(tmp_path), task)

    out = paint.read(base64.b64decode(result["glb_b64"]))
    assert np.array_equal(out.vertices, mesh.vertices), "vertices, in order, bit for bit"
    assert np.array_equal(out.faces, mesh.faces), "faces, in order"
    assert np.allclose(out.uv, mesh.visual.uv, atol=1e-6)
    assert (result["faces"], result["vertices"]) == (len(mesh.faces), len(mesh.vertices))
    assert (result["uv"], result["shape"], result["texture_size"]) == ("kept", "encoded", 1024)
    assert "vertex_map" not in result

    # The model was asked about points in its own cube, not the client's frame.
    points = seen[0]
    assert float(points.min()) >= -0.5 - 1e-4 and float(points.max()) <= 0.5 + 1e-4
    # And the colours it gave are on the mesh: a texture, not a blank.
    loaded = trimesh.load(io.BytesIO(base64.b64decode(result["glb_b64"])), file_type="glb")
    material = next(iter(loaded.geometry.values())).visual.material
    pixels = np.asarray(material.baseColorTexture.convert("RGB"))
    assert pixels.shape == (1024, 1024, 3) and pixels.std() > 10
    assert material.normalTexture is None, "no detailed surface behind it, no normal map"


def test_a_mesh_without_uvs_is_unwrapped_and_says_which_vertex_is_which(
    tmp_path, monkeypatch, on_cpu
):
    from giq_3d_child import lowpoly

    mesh = _character(uv=False)

    def unwrap(v, f, size):
        # xatlas's answer for a mesh cut along one seam: the first face's
        # corners copied, everything else as it was.
        extra = f[0]
        vmap = np.concatenate([np.arange(len(v)), extra])
        faces = f.copy()
        faces[0] = len(v) + np.arange(3)
        return vmap, faces, _spherical(v[vmap]).astype(np.float32)

    monkeypatch.setattr(lowpoly, "unwrap", unwrap)
    monkeypatch.setattr(child, "_sampler", _model("canonical", []))
    task = {"mesh_b64": _b64(_glb(mesh)), "image_b64": _png(), "texture_size": 1024}
    result = child._paint((None, None), _args(tmp_path), task)

    out = paint.read(base64.b64decode(result["glb_b64"]))
    vmap = np.asarray(result["vertex_map"])
    assert result["uv"] == "unwrapped" and len(vmap) == len(out.vertices)
    assert np.array_equal(out.vertices, mesh.vertices[vmap]), "every vertex a copy of its input"
    assert np.array_equal(vmap[out.faces], mesh.faces), "every face the input's face"


def test_a_paint_handed_its_shape_paints_on_it(tmp_path, monkeypatch, on_cpu):
    args = _args(tmp_path)
    coords, feats = _latent()
    shape = shape_latent.pack(
        coords, feats, recipe="trellis2", decoder=child._decoder_name(args), resolution=1024, seed=5
    )
    given: list = []
    monkeypatch.setattr(child, "_sampler", _model("generator", [], given))
    task = {
        "mesh_b64": _b64(_glb(_character())),
        "image_b64": _png(),
        "shape_b64": shape_latent.encode(shape),
        "texture_size": 1024,
    }
    result = child._paint((None, None), args, task)
    head, c, f = given[0]
    assert head["seed"] == 5 and np.array_equal(c, coords) and np.array_equal(f, feats)
    assert result["shape"] == "given"


def test_a_shape_from_other_weights_is_refused_before_the_model_runs(tmp_path, monkeypatch, on_cpu):
    old = _args(tmp_path / "old", decoder_bytes=100)
    coords, feats = _latent()
    shape = shape_latent.pack(
        coords, feats, recipe="trellis2", decoder=child._decoder_name(old), resolution=1024, seed=5
    )
    ran: list = []
    monkeypatch.setattr(child, "_sampler", _model("generator", ran, ran))
    task = {
        "mesh_b64": _b64(_glb(_character())),
        "image_b64": _png(),
        "shape_b64": shape_latent.encode(shape),
    }
    with pytest.raises(shape_latent.StaleShape, match="another shape decoder"):
        child._paint((None, None), _args(tmp_path / "new", decoder_bytes=200), task)
    assert ran == []


def test_a_paint_of_a_bad_mesh_fails_before_the_model_runs(tmp_path, monkeypatch, on_cpu):
    ran: list = []
    monkeypatch.setattr(child, "_sampler", _model("canonical", ran, ran))
    task = {"mesh_b64": _b64(b"glTF\x02\x00\x00\x00junk"), "image_b64": _png()}
    with pytest.raises(ValueError, match="is not a GLB giq can read"):
        child._paint((None, None), _args(tmp_path), task)
    assert ran == []


def test_a_detailed_surface_behind_bakes_a_normal_map():
    mesh = paint.read(_glb(_character()))

    def sample(P, Ns):
        values = torch.full((len(P), 6), 0.5)
        tilted = torch.nn.functional.normalize(Ns + torch.tensor([0.3, 0.0, 0.0]), dim=1)
        return values, tilted

    layout = child.PBR_LAYOUT
    out, info = paint.paint(mesh, "generator", 1024, sample, layout)
    assert out.visual.material.normalTexture is not None and info == {"uv": "kept"}
    assert np.array_equal(np.asarray(out.vertices), mesh.vertices)


# --- what the parent makes of it --------------------------------------------


def test_a_default_result_has_the_fields_it_had():
    """Golden: an existing client's result keeps its keys and values; the
    new keys are there and empty."""
    from giq_3d.adapter import hydrate

    child_result = {
        "id": "a",
        "glb_b64": "Z2xURg==",
        "format": "glb",
        "mode": "game",
        "faces": 2996,
        "vertices": 3100,
        "texture_size": 2048,
        "seed": 42,
    }
    dumped = hydrate(child_result).model_dump()
    assert {k: dumped[k] for k in child_result} == child_result
    assert {k: v for k, v in dumped.items() if k not in child_result} == {
        "parts": None,
        "contacts": None,
        "texture": None,
        "shape_b64": None,
        "shape": None,
        "uv": None,
        "vertex_map": None,
        "error": None,
    }


def test_shape_and_paint_results_keep_what_they_add():
    from giq_3d.adapter import hydrate

    shape = hydrate({"id": "s", "glb_b64": "Z2xURg==", "texture": False, "shape_b64": "R0lR"})
    assert (shape.texture, shape.shape_b64, shape.texture_size) == (False, "R0lR", None)
    painted = hydrate(
        {
            "id": "p",
            "glb_b64": "Z2xURg==",
            "shape": "encoded",
            "uv": "unwrapped",
            "vertex_map": [0, 1, 1],
        }
    )
    assert (painted.shape, painted.uv, painted.vertex_map) == ("encoded", "unwrapped", [0, 1, 1])


def test_the_trellis2_recipes_serve_both_halves_in_one_instance():
    from giq import plugins
    from giq.registry import get_recipe

    for name in ("trellis2", "trellis2-parts"):
        recipe = get_recipe(name)
        assert list(recipe.modalities) == ["image2mesh", "mesh2texture"]
        # The first modality picks the adapter, so a paint runs in the same child.
        assert recipe.modality == "image2mesh"
    assert plugins.adapter("pixal3d", "mesh2texture") is plugins.adapter("pixal3d", "image2mesh")
    assert list(get_recipe("pixal3d").modalities) == ["image2mesh"]


def test_a_paint_job_is_taken_for_trellis2_and_refused_for_pixal3d():
    from fastapi import HTTPException

    from giq.models import JobRequest
    from giq.services.orchestration import Orchestrator

    task = {"id": "1", "mesh_b64": "Z2xURg==", "image_b64": IMAGE}
    ok = JobRequest(modality="mesh2texture", model="trellis2", tasks=[task])
    Orchestrator._resolve_recipe(ok)
    assert ok.model == "trellis2"
    with pytest.raises(HTTPException, match="pixal3d does not serve mesh2texture"):
        Orchestrator._resolve_recipe(
            JobRequest(modality="mesh2texture", model="pixal3d", tasks=[task])
        )
