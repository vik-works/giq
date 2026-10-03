# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""DecideAdapter: child results become DecideResults in the parent.

The child is replaced by an inline script that answers every task with a
canned decision, so this exercises hydration and the error envelope without
a GPU or a model. The child's own arithmetic (rotations, calibration) is
imajev's and is tested there.
"""

import json
import os
import sys

import pytest

from giq.models import DecideResult
from giq_imajev.adapter import DecideAdapter, DecideConfig, hydrate

# Answers each task with canned answers, or an error when the task id says so.
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
            results.append({"id": t["id"], "error": "task needs questions"})
        else:
            results.append({"id": t["id"], "answers": {"queue": {"type": "choice",
                "choice": "billing", "probabilities": {"billing": 0.99},
                "confidence": 0.98, "unknown_probability": 0.01, "abstained": False}},
                "tokens_in": 42, "images": 0, "rotations": 1})
    sys.stdout.write(json.dumps({"type": "results", "results": results}) + "\\n")
    sys.stdout.flush()
"""


class _StubDecideWorker(DecideAdapter):
    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT]

    def _spawn_env(self) -> dict[str, str]:
        return dict(os.environ)


@pytest.fixture()
def decide_recipe(tmp_path, monkeypatch):
    """A minimal decide recipe so the adapter's require_path calls resolve."""
    recipe = tmp_path / "decide.imajev-2b.yaml"
    recipe.write_text(
        "name: imajev-2b\n"
        "modalities: [decide]\n"
        "engine: imajev\n"
        "weights:\n"
        "  path: imajev-2b\n"
        "  format: safetensors\n"
        "  licence: apache-2.0\n"
        "  parts:\n"
        "    base: {path: /projects/models/Qwen3.5-2B, format: safetensors, licence: apache-2.0}\n"
        "vram: {gb: 6.0}\n"
        "max_batch: 4\n"
    )
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    from giq import recipes

    recipes.reload()
    yield "imajev-2b"
    monkeypatch.undo()
    from giq import recipes as _recipes

    _recipes.reload()


@pytest.mark.asyncio
async def test_run_batch_returns_decide_results(decide_recipe):
    worker = _StubDecideWorker(DecideConfig(model=decide_recipe))
    await worker.start()
    try:
        results = await worker.run_batch(
            [
                {"id": "a", "state": "Ticket: 'Charged twice.'", "questions": {"q": 1}},
                {"id": "bad-1"},
            ]
        )
    finally:
        await worker.stop()

    assert all(isinstance(r, DecideResult) for r in results)
    a, bad = results
    assert a.answers["queue"].choice == "billing"
    assert a.answers["queue"].unknown_probability == 0.01
    assert (a.tokens_in, a.images, a.rotations) == (42, 0, 1)
    assert a.error is None
    assert bad.error == "task needs questions" and bad.answers == {}
    # The runner stores results via model_dump(); make sure that round-trips.
    dumped = json.loads(json.dumps([r.model_dump() for r in results]))
    assert dumped[0]["answers"]["queue"]["choice"] == "billing"


def test_hydrate_error_envelope():
    bad = hydrate({"id": "x", "error": "boom"})
    assert bad.id == "x" and bad.error == "boom"


def test_operator_recipe_shape(tmp_path):
    from giq.recipes import load_file

    recipe_file = tmp_path / "decide.imajev-2b.yaml"
    recipe_file.write_text(
        "name: imajev-2b\n"
        "modalities: [decide]\n"
        "engine: imajev\n"
        "weights:\n"
        "  path: imajev-2b\n"
        "  source: hf:mohit67890/imajev-2b\n"
        "  format: safetensors\n"
        "  licence: apache-2.0\n"
        "  parts:\n"
        "    base:\n"
        "      path: Qwen-Qwen2.5-2B\n"
        "      source: hf:Qwen/Qwen3.5-2B\n"
        "      revision: 15852e8c16360a2fea060d615a32b45270f8a8fc\n"
        "      format: safetensors\n"
        "      licence: apache-2.0\n"
        "vram: {gb: 6.0}\n"
        "max_batch: 4\n"
    )
    recipe = load_file(recipe_file)
    assert recipe.modalities == ("decide",) and recipe.engine == "imajev"
    assert set(recipe.weights.parts) == {"base"}
