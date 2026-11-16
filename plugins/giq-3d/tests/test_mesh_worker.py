# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""MeshAdapter: the child's flags come from the recipe, its results become
MeshResults in the parent.

The child is replaced by an inline script that answers every task with a
canned asset, so this exercises the protocol, hydration and the error
envelope without a GPU or a model.
"""

import os
import sys

import pytest

from giq_3d.adapter import MeshAdapter, MeshConfig, MeshResult, hydrate, source_dir

CHILD_SCRIPT = """
import sys, json
sys.stdout.write(json.dumps({"type": "ready"}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        break
    results = []
    for t in msg["tasks"]:
        if t["id"].startswith("bad"):
            results.append({"id": t["id"], "error": "task needs an 'image_b64' of the object"})
        else:
            results.append({"id": t["id"], "glb_b64": "Z2xURg==", "format": "glb",
                "mode": t.get("mode", "game"), "faces": 2996, "vertices": 3100,
                "texture_size": 2048, "seed": 42})
    sys.stdout.write(json.dumps({"type": "results", "results": results}) + "\\n")
    sys.stdout.flush()
"""


class _StubMeshWorker(MeshAdapter):
    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT]

    def _spawn_env(self) -> dict[str, str]:
        return dict(os.environ)


def _recipe(tmp_path, monkeypatch, params: str = "", parts: str | None = None):
    models = tmp_path / "models"
    for folder in ("Pixal3D", "dinov3", "moge", "birefnet", "trellis1"):
        (models / folder).mkdir(parents=True)
    if parts is None:
        parts = "{encoder: dinov3, camera: moge, matting: birefnet}"
    (tmp_path / "mesh.yaml").write_text(
        "name: mesh-test\n"
        "modalities: [image2mesh]\n"
        "engine: pixal3d\n"
        f"weights: {{path: Pixal3D, format: safetensors, licence: mit, parts: {parts}}}\n"
        f"{params}"
        "vram: {gb: 16.0}\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    monkeypatch.setenv("GIQ_MODELS_DIR", str(models))
    from giq import recipes

    recipes.reload()
    return "mesh-test", models


def test_the_childs_flags_come_from_the_recipe(tmp_path, monkeypatch):
    name, models = _recipe(tmp_path, monkeypatch, "params: {resolution: 1536, low_vram: false}\n")
    monkeypatch.setenv("GIQ_PIXAL3D_SOURCE", "/opt/Pixal3D")
    args = MeshAdapter(MeshConfig(model=name)).child_args()
    flags = dict(zip(args[::2], args[1::2], strict=False))
    assert flags["--model"] == name
    assert flags["--weights"] == str(models / "Pixal3D")
    assert flags["--source"] == "/opt/Pixal3D"
    assert flags["--encoder"] == str(models / "dinov3")
    assert flags["--camera"] == str(models / "moge")
    assert flags["--matting"] == str(models / "birefnet")
    assert flags["--resolution"] == "1536"
    assert flags["--views"] == "single"
    assert "--low-vram" not in args


def test_a_multi_view_recipe_tells_the_child_so(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch, "params: {views: multi}\n")
    args = MeshAdapter(MeshConfig(model=name)).child_args()
    assert args[args.index("--views") + 1] == "multi"


def test_low_vram_is_the_default(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch)
    args = MeshAdapter(MeshConfig(model=name)).child_args()
    assert args[-1] == "--low-vram"
    assert args[args.index("--resolution") + 1] == "1024"


def test_a_recipe_without_its_encoder_is_refused(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch, parts="{camera: moge, matting: birefnet}")
    with pytest.raises(ValueError, match="weights.parts.encoder"):
        MeshAdapter(MeshConfig(model=name)).child_args()


def test_the_source_defaults_to_the_checkout_beside_the_environment(monkeypatch):
    monkeypatch.delenv("GIQ_PIXAL3D_SOURCE", raising=False)
    assert source_dir().endswith(os.path.join("envs", "pixal3d", "Pixal3D"))


def test_the_declared_vram_is_the_recipes(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch)
    assert MeshAdapter(MeshConfig(model=name)).estimated_vram_gb == 16.0


@pytest.mark.asyncio
async def test_results_come_back_as_mesh_results(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch)
    worker = _StubMeshWorker(MeshConfig(model=name))
    await worker.start()
    try:
        good, bad = await worker.run_batch(
            [{"id": "a", "image_b64": "aGk=", "mode": "full"}, {"id": "bad-1"}]
        )
    finally:
        await worker.stop()
    assert isinstance(good, MeshResult)
    assert (good.glb_b64, good.format, good.mode, good.faces) == ("Z2xURg==", "glb", "full", 2996)
    assert good.error is None
    assert bad.glb_b64 is None and "image_b64" in (bad.error or "")


def test_an_error_envelope_stays_an_error():
    result = hydrate({"id": 7, "error": "boom", "faces": "not a number"})
    assert (result.id, result.error, result.faces) == ("7", "boom", None)


def test_a_trellis2_recipe_names_its_own_source_and_parts(tmp_path, monkeypatch):
    name, models = _recipe(
        tmp_path,
        monkeypatch,
        "params: {pipeline: trellis2}\n",
        parts="{encoder: dinov3, matting: birefnet, structure: trellis1}",
    )
    monkeypatch.setenv("GIQ_TRELLIS2_SOURCE", "/opt/TRELLIS.2")
    args = MeshAdapter(MeshConfig(model=name)).child_args()
    flags = dict(zip(args[::2], args[1::2], strict=False))
    assert (flags["--pipeline"], flags["--source"]) == ("trellis2", "/opt/TRELLIS.2")
    assert flags["--structure"] == str(models / "trellis1")
    assert "--camera" not in args, "TRELLIS.2 estimates no camera"


def test_trellis2_without_its_structure_decoder_is_refused(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch, "params: {pipeline: trellis2}\n")
    with pytest.raises(ValueError, match="weights.parts.structure"):
        MeshAdapter(MeshConfig(model=name)).child_args()


def test_trellis2_takes_one_picture():
    from giq_3d.params import Pixal3DParams

    with pytest.raises(ValueError, match="trellis2 takes one picture"):
        Pixal3DParams(pipeline="trellis2", views="multi")


def test_each_pipeline_has_its_own_default_checkout(monkeypatch):
    monkeypatch.delenv("GIQ_TRELLIS2_SOURCE", raising=False)
    assert source_dir("trellis2").endswith(os.path.join("envs", "pixal3d", "TRELLIS.2"))


def test_a_recipe_that_segments_hands_its_segmenter_to_the_child(tmp_path, monkeypatch):
    name, models = _recipe(
        tmp_path,
        monkeypatch,
        "params: {pipeline: trellis2, segment: true}\n",
        parts="{encoder: dinov3, matting: birefnet, structure: trellis1, segmenter: seg.ckpt}",
    )
    args = MeshAdapter(MeshConfig(model=name)).child_args()
    flags = dict(zip(args[::2], args[1::2], strict=False))
    assert flags["--segmenter"] == str(models / "seg.ckpt")


def test_segmenting_needs_its_checkpoint_and_trellis2(tmp_path, monkeypatch):
    from giq_3d.params import Pixal3DParams

    name, _ = _recipe(
        tmp_path,
        monkeypatch,
        "params: {pipeline: trellis2, segment: true}\n",
        parts="{encoder: dinov3, matting: birefnet, structure: trellis1}",
    )
    with pytest.raises(ValueError, match="weights.parts.segmenter"):
        MeshAdapter(MeshConfig(model=name)).child_args()
    with pytest.raises(ValueError, match="`segment` is trellis2's"):
        Pixal3DParams(pipeline="pixal3d", segment=True)


def test_a_parts_result_keeps_its_parts_and_contacts():
    result = hydrate(
        {
            "id": "a",
            "glb_b64": "Z2xURg==",
            "faces": 4000,
            "parts": [{"name": "part_0", "faces": 2300, "vertices": 1500, "share": 0.57}],
            "contacts": [{"parts": [0, 1], "points": 1006, "centre": [0.2, -0.1, 0.0]}],
        }
    )
    assert result.parts[0]["name"] == "part_0" and result.contacts[0]["parts"] == [0, 1]
    assert hydrate({"id": "b", "glb_b64": "Z2xURg=="}).parts is None
