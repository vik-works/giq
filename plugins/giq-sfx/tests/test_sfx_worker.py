# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""SfxAdapter: the child's flags come from the recipe, its results become
SfxResults in the parent.

The child is replaced by an inline script that answers every task with a
canned sound, so this exercises the protocol, hydration and the error
envelope without a GPU or a model.
"""

import json
import os
import sys

import pytest

from giq_sfx.adapter import (
    CompiledSfxAdapter,
    SfxAdapter,
    SfxConfig,
    SfxResult,
    hydrate,
    sfx_adapter,
)
from giq_sfx.params import COMPILE_READY_TIMEOUT, READY_TIMEOUT

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
            results.append({"id": t["id"], "error": "task needs a 'prompt' describing the sound"})
        else:
            results.append({"id": t["id"], "audio_b64": "ZkxhQw==", "format": "flac",
                "sample_rate": 48000, "duration": 4.0, "seed": 7, "steps": 100, "guidance": 4.0})
    sys.stdout.write(json.dumps({"type": "results", "results": results}) + "\\n")
    sys.stdout.flush()
"""


class _StubSfxWorker(SfxAdapter):
    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT]

    def _spawn_env(self) -> dict[str, str]:
        return dict(os.environ)


def _recipe(tmp_path, monkeypatch, params: str = ""):
    (tmp_path / "sfx.yaml").write_text(
        "name: sfx-test\n"
        "modalities: [text2sfx]\n"
        "engine: moss-sfx\n"
        "weights: {path: OpenMOSS-Team-MOSS-SoundEffect-v2.0, format: safetensors,"
        " licence: apache-2.0}\n"
        f"{params}"
        "vram: {gb: 14.0}\n"
        "max_batch: 4\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    from giq import recipes

    recipes.reload()
    return "sfx-test"


@pytest.fixture()
def sfx_recipe(tmp_path, monkeypatch):
    yield _recipe(tmp_path, monkeypatch)
    monkeypatch.undo()
    from giq import recipes

    recipes.reload()


async def test_run_batch_returns_sfx_results(sfx_recipe):
    worker = _StubSfxWorker(SfxConfig(model=sfx_recipe))
    await worker.start()
    try:
        results = await worker.run_batch([{"id": "a", "prompt": "rain"}, {"id": "bad-1"}])
    finally:
        await worker.stop()

    assert all(isinstance(r, SfxResult) for r in results)
    a, bad = results
    assert (a.format, a.sample_rate, a.duration, a.seed) == ("flac", 48000, 4.0, 7)
    assert (a.steps, a.guidance, a.error) == (100, 4.0, None)
    assert bad.error and bad.audio_b64 is None
    dumped = json.loads(json.dumps([r.model_dump() for r in results]))
    assert dumped[0]["audio_b64"] == "ZkxhQw=="


def test_the_child_gets_the_recipes_defaults(tmp_path, monkeypatch):
    model = _recipe(tmp_path, monkeypatch)
    worker = sfx_adapter(SfxConfig(model=model))
    args = worker.child_args()
    assert args[args.index("--weights") + 1].endswith("OpenMOSS-Team-MOSS-SoundEffect-v2.0")
    assert args[args.index("--steps") + 1] == "100"
    assert args[args.index("--guidance") + 1] == "4.0"
    assert args[args.index("--shift") + 1] == "5.0"
    assert "--compile" not in args
    assert type(worker) is SfxAdapter and worker.ready_timeout == READY_TIMEOUT


def test_a_compiled_recipe_compiles_and_may_take_longer_to_start(tmp_path, monkeypatch):
    model = _recipe(tmp_path, monkeypatch, "params: {steps: 50, guidance: 3, compile: true}\n")
    worker = sfx_adapter(SfxConfig(model=model))
    args = worker.child_args()
    assert "--compile" in args
    assert args[args.index("--steps") + 1] == "50"
    assert isinstance(worker, CompiledSfxAdapter)
    assert worker.ready_timeout == COMPILE_READY_TIMEOUT

    from giq import plugins
    from giq.runner import start_budget

    assert plugins.engine("moss-sfx") is not None
    assert start_budget(model) == COMPILE_READY_TIMEOUT


def test_params_out_of_range_are_refused(tmp_path):
    from giq.recipes import load_file

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nmodalities: [text2sfx]\nengine: moss-sfx\n"
        "weights: {path: x, format: safetensors, licence: apache-2.0}\n"
        "params: {steps: 0}\nvram: {gb: 14.0}\n"
    )
    with pytest.raises(ValueError, match="steps"):
        load_file(bad)


def test_hydrate_error_envelope():
    bad = hydrate({"id": "x", "error": "boom"})
    assert bad.id == "x" and bad.error == "boom"


def test_the_builtin_recipe():
    from giq.registry import get_recipe

    recipe = get_recipe("moss-soundeffect-v2")
    assert recipe is not None and recipe.engine == "moss-sfx"
    assert recipe.modalities == ("text2sfx",)
    assert recipe.params.compile is False  # ty: ignore[unresolved-attribute]
    assert recipe.weights is not None
    assert recipe.weights.source == "hf:OpenMOSS-Team/MOSS-SoundEffect-v2.0"
    assert recipe.weights.revision == "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d"
    assert recipe.measured and recipe.vram_gb == 18.5
