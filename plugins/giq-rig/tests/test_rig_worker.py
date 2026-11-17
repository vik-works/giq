# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""RigAdapter: the child's flags come from the recipe, its results become
RigResults in the parent, and the runner builds, evicts and reloads it like
any on-demand recipe.

The child is replaced by an inline script that answers every task with a
canned rig, so this exercises the protocol, hydration and the error
envelope without a GPU or a model.
"""

import os
import sys

import pytest

from giq_rig.adapter import RigAdapter, RigConfig, RigResult, hydrate, source_dir
from giq_rig.params import CHECKPOINT, SkinTokensParams

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
            results.append({"id": t["id"], "error": "task needs a 'glb_b64'"})
        else:
            results.append({"id": t["id"], "glb_b64": "Z2xURg==", "format": "glb",
                "bones": 23, "seed": t.get("seed", 42)})
    sys.stdout.write(json.dumps({"type": "results", "results": results}) + "\\n")
    sys.stdout.flush()
"""


def _stub_command(self) -> list[str]:
    return [sys.executable, "-u", "-c", CHILD_SCRIPT]


def _stub_env(self) -> dict[str, str]:
    return dict(os.environ)


class _StubRigWorker(RigAdapter):
    _command = _stub_command
    _spawn_env = _stub_env


def _recipe(tmp_path, monkeypatch, params: str = "", parts: str | None = None):
    models = tmp_path / "models"
    (models / "SkinTokens").mkdir(parents=True)
    (models / "qwen").mkdir()
    (models / "qwen" / "config.json").write_text("{}")
    if parts is None:
        parts = "{llm: qwen/config.json}"
    (tmp_path / "rig.yaml").write_text(
        "name: rig-test\n"
        "modalities: [mesh2rig]\n"
        "engine: skintokens\n"
        f"weights: {{path: SkinTokens, licence: mit, parts: {parts}}}\n"
        f"{params}"
        "vram: {gb: 5.0}\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    monkeypatch.setenv("GIQ_MODELS_DIR", str(models))
    from giq import recipes

    recipes.reload()
    return "rig-test", models


def test_the_childs_flags_come_from_the_recipe(tmp_path, monkeypatch):
    name, models = _recipe(tmp_path, monkeypatch)
    monkeypatch.setenv("GIQ_SKINTOKENS_SOURCE", "/opt/SkinTokens")
    args = RigAdapter(RigConfig(model=name)).child_args()
    flags = dict(zip(args[::2], args[1::2], strict=True))
    assert flags == {
        "--model": name,
        "--weights": str(models / "SkinTokens"),
        "--checkpoint": CHECKPOINT,
        "--llm": str(models / "qwen" / "config.json"),
        "--source": "/opt/SkinTokens",
    }


def test_a_recipe_may_name_another_checkpoint_in_its_weights(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch, "params: {checkpoint: experiments/other/last.ckpt}\n")
    args = RigAdapter(RigConfig(model=name)).child_args()
    assert args[args.index("--checkpoint") + 1] == "experiments/other/last.ckpt"


@pytest.mark.parametrize("checkpoint", ["/etc/passwd", "../elsewhere.ckpt", "a/../../b.ckpt", ""])
def test_the_checkpoint_stays_inside_the_weights(checkpoint):
    with pytest.raises(ValueError, match="a path inside weights.path"):
        SkinTokensParams(checkpoint=checkpoint)


def test_a_recipe_without_the_transformers_config_is_refused(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch, parts="{}")
    with pytest.raises(ValueError, match="weights.parts.llm"):
        RigAdapter(RigConfig(model=name)).child_args()


def test_a_recipe_without_weights_is_refused_at_construction(tmp_path, monkeypatch):
    (tmp_path / "rig.yaml").write_text(
        "name: rig-bare\nmodalities: [mesh2rig]\nengine: skintokens\nvram: {gb: 5.0}\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    from giq import recipes

    recipes.reload()
    with pytest.raises(ValueError, match="no weights.path"):
        RigAdapter(RigConfig(model="rig-bare"))


def test_a_part_the_modality_does_not_read_is_refused(tmp_path, monkeypatch):
    from giq.recipes import RecipeError, load_file

    _recipe(tmp_path, monkeypatch, parts="{llm: qwen/config.json, encoder: dinov3}")
    with pytest.raises(RecipeError, match="reads only llm"):
        load_file(tmp_path / "rig.yaml")


def test_a_checkpoint_outside_the_weights_refuses_the_recipe(tmp_path, monkeypatch):
    from giq.recipes import RecipeError, load_file

    _recipe(tmp_path, monkeypatch, "params: {checkpoint: ../../etc/x.ckpt}\n")
    with pytest.raises(RecipeError, match="a path inside weights.path"):
        load_file(tmp_path / "rig.yaml")


def test_the_source_defaults_to_the_checkout_beside_the_environment(monkeypatch):
    monkeypatch.delenv("GIQ_SKINTOKENS_SOURCE", raising=False)
    assert source_dir().endswith(os.path.join("envs", "skintokens", "SkinTokens"))


def test_the_declared_vram_is_the_recipes(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch)
    assert RigAdapter(RigConfig(model=name)).estimated_vram_gb == 5.0


def test_the_child_finds_its_package_and_the_protocol_on_its_path(tmp_path, monkeypatch):
    import giq_child
    import giq_rig_child

    name, _ = _recipe(tmp_path, monkeypatch)
    path = RigAdapter(RigConfig(model=name))._spawn_env()["PYTHONPATH"].split(os.pathsep)
    assert os.path.dirname(os.path.dirname(giq_rig_child.__file__)) in path
    assert os.path.dirname(os.path.dirname(giq_child.__file__)) in path


@pytest.mark.asyncio
async def test_results_come_back_as_rig_results(tmp_path, monkeypatch):
    name, _ = _recipe(tmp_path, monkeypatch)
    worker = _StubRigWorker(RigConfig(model=name))
    await worker.start()
    try:
        good, bad = await worker.run_batch(
            [{"id": "a", "glb_b64": "Z2xURg==", "seed": 7}, {"id": "bad-1"}]
        )
    finally:
        await worker.stop()
    assert isinstance(good, RigResult)
    assert (good.glb_b64, good.format, good.bones, good.seed) == ("Z2xURg==", "glb", 23, 7)
    assert good.error is None
    assert bad.glb_b64 is None and bad.bones is None and "glb_b64" in (bad.error or "")


def test_an_error_envelope_stays_an_error():
    result = hydrate({"id": 7, "error": "boom", "bones": "not a number"})
    assert (result.id, result.error, result.bones) == ("7", "boom", None)


@pytest.mark.asyncio
async def test_the_runner_builds_evicts_and_reloads_it(tmp_path, monkeypatch):
    """A rig recipe is an ordinary on-demand one: the runner builds its
    adapter through the plugin registry, and a stopped child (an eviction)
    starts again for the next job."""
    from giq.queue import JobQueue
    from giq.registry import get_recipe, resident_defaults
    from giq.runner import Runner

    name, _ = _recipe(tmp_path, monkeypatch)
    assert name not in resident_defaults()
    assert get_recipe(name).residency.priority is None, "loads on demand, evictable"
    monkeypatch.setattr(RigAdapter, "_command", _stub_command)
    monkeypatch.setattr(RigAdapter, "_spawn_env", _stub_env)
    worker = Runner(JobQueue())._build_worker(name)
    assert isinstance(worker, RigAdapter)
    for attempt in ("first", "after eviction"):
        await worker.start()
        try:
            (result,) = await worker.run_batch([{"id": attempt, "glb_b64": "Z2xURg=="}])
        finally:
            await worker.stop()
        assert result.bones == 23, attempt
