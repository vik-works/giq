# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The mesh2rig child's request checking, which is pure (no SkinTokens, no
GPU), and its start, which refuses a missing checkout or weights before it
imports torch."""

import base64
import sys

import pytest

from giq_rig_child._rig_child import MAX_MESH_BYTES, SAMPLING, SEED, task_inputs

GLB = b"glTF\x02\x00\x00\x00 not really a mesh"
MESH = base64.b64encode(GLB).decode()


def test_a_bare_task_takes_upstreams_defaults():
    inputs = task_inputs({"id": "a", "glb_b64": MESH})
    assert inputs["mesh"] == GLB
    assert inputs["seed"] == SEED
    assert inputs["sampling"] == {name: default for name, (default, _, _) in SAMPLING.items()}
    assert (inputs["keep_skeleton"], inputs["voxel_skin"]) == (False, False)


def test_the_sampler_and_the_options_are_the_tasks_to_set():
    inputs = task_inputs(
        {
            "glb_b64": MESH,
            "seed": 7,
            "num_beams": 4,
            "temperature": 0.8,
            "top_k": 20,
            "keep_skeleton": True,
            "voxel_skin": True,
        }
    )
    assert inputs["seed"] == 7
    assert (inputs["sampling"]["num_beams"], inputs["sampling"]["temperature"]) == (4, 0.8)
    assert inputs["sampling"]["top_k"] == 20
    assert inputs["keep_skeleton"] and inputs["voxel_skin"]


def test_a_data_url_prefix_is_accepted():
    assert task_inputs({"glb_b64": "data:model/gltf-binary;base64," + MESH})["mesh"] == GLB


def test_nulls_are_the_defaults():
    inputs = task_inputs({"glb_b64": MESH, "seed": None, "top_p": None, "voxel_skin": None})
    assert (inputs["seed"], inputs["sampling"]["top_p"], inputs["voxel_skin"]) == (
        SEED,
        0.95,
        False,
    )


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({}, "needs a 'glb_b64'"),
        ({"glb_b64": ""}, "needs a 'glb_b64'"),
        ({"glb_b64": "not base64!"}, "is not base64"),
        ({"glb_b64": base64.b64encode(b"solid ascii stl").decode()}, "is not a GLB"),
        ({"glb_b64": base64.b64encode(b'{"asset": {}}').decode()}, "is not a GLB"),
        ({"glb_b64": MESH, "seed": -1}, "'seed' must be 0 to 4294967295"),
        ({"glb_b64": MESH, "seed": 2**32}, "'seed' must be 0 to 4294967295"),
        ({"glb_b64": MESH, "seed": 1.5}, "'seed' must be a whole number"),
        ({"glb_b64": MESH, "num_beams": 0}, "'num_beams' must be 1-20"),
        ({"glb_b64": MESH, "num_beams": 2.5}, "'num_beams' must be a whole number"),
        ({"glb_b64": MESH, "top_k": True}, "'top_k' must be a whole number"),
        ({"glb_b64": MESH, "temperature": 0}, "'temperature' must be 0.1-2.0"),
        ({"glb_b64": MESH, "top_p": "high"}, "'top_p' must be a number"),
        ({"glb_b64": MESH, "repetition_penalty": 5}, "'repetition_penalty' must be 0.5-3.0"),
        ({"glb_b64": MESH, "keep_skeleton": "yes"}, "'keep_skeleton' must be true or false"),
        ({"glb_b64": MESH, "voxel_skin": 1}, "'voxel_skin' must be true or false"),
    ],
)
def test_a_wrong_field_is_named(task, message):
    with pytest.raises(ValueError, match=message):
        task_inputs(task)


def test_a_mesh_over_the_cap_is_refused(monkeypatch):
    import giq_rig_child._rig_child as child

    monkeypatch.setattr(child, "MAX_MESH_BYTES", len(GLB) - 1)
    with pytest.raises(ValueError, match="larger than"):
        task_inputs({"glb_b64": MESH})
    assert MAX_MESH_BYTES == 64 * 2**20


def test_whole_floats_count_as_whole_numbers():
    inputs = task_inputs({"glb_b64": MESH, "num_beams": 3.0, "seed": 9.0})
    assert (inputs["sampling"]["num_beams"], inputs["seed"]) == (3, 9)


@pytest.mark.asyncio
async def test_a_child_without_its_checkout_says_so_at_start(tmp_path, monkeypatch):
    """The real child, on giq's interpreter: it checks what it was given
    before it imports torch, and reports through the protocol."""
    from giq.adapters._subprocess import SubprocessWorkerStartError
    from giq_rig.adapter import RigAdapter, RigConfig

    models = tmp_path / "models"
    (models / "SkinTokens").mkdir(parents=True)
    (models / "qwen.json").write_text("{}")
    (tmp_path / "rig.yaml").write_text(
        "name: rig-test\nmodalities: [mesh2rig]\nengine: skintokens\n"
        "weights: {path: SkinTokens, parts: {llm: qwen.json}}\nvram: {gb: 5.0}\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    monkeypatch.setenv("GIQ_MODELS_DIR", str(models))
    monkeypatch.setenv("GIQ_SKINTOKENS_SOURCE", str(tmp_path / "no-checkout"))
    from giq import recipes

    recipes.reload()

    class _OnThisInterpreter(RigAdapter):
        def _command(self) -> list[str]:
            return [sys.executable, "-u", "-m", self.child_module, *self.child_args()]

    worker = _OnThisInterpreter(RigConfig(model="rig-test"))
    with pytest.raises(SubprocessWorkerStartError, match="giq prepare skintokens"):
        await worker.start()
    await worker.stop()
