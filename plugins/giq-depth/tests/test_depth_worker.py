# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""DepthAdapter: child results become DepthResults in the parent.

The child is replaced by an inline script that answers every task with a
canned map, so this exercises hydration and the error envelope without a
GPU or a model. The child's own arithmetic (16-bit mapping, colour ramp) is
pure numpy and is tested directly.
"""

import json
import os
import sys

import numpy as np
import pytest

from giq.models import DepthResult
from giq_depth.adapter import DepthAdapter, DepthConfig, hydrate

# Answers each task with a 2x2 map, or an error when the task id says so.
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
            results.append({"id": t["id"], "error": "task needs image_b64"})
        else:
            r = {"id": t["id"], "depth_b64": "iVBORw0=", "width": 2, "height": 2,
                 "depth_min": 0.5, "depth_max": 9.0, "metric": False}
            if t.get("visualize"):
                r["visualization_b64"] = "iVBORw1="
            results.append(r)
    sys.stdout.write(json.dumps({"type": "results", "results": results}) + "\\n")
    sys.stdout.flush()
"""


class _StubDepthWorker(DepthAdapter):
    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", CHILD_SCRIPT]

    def _spawn_env(self) -> dict[str, str]:
        return dict(os.environ)


@pytest.mark.asyncio
async def test_run_batch_returns_depth_results():
    worker = _StubDepthWorker(DepthConfig())
    await worker.start()
    try:
        results = await worker.run_batch(
            [
                {"id": "a", "image_b64": "x"},
                {"id": "b", "image_b64": "x", "visualize": True},
                {"id": "bad-1"},
            ]
        )
    finally:
        await worker.stop()

    assert all(isinstance(r, DepthResult) for r in results)
    a, b, bad = results
    assert a.depth_b64 == "iVBORw0=" and (a.width, a.height) == (2, 2)
    assert (a.depth_min, a.depth_max, a.metric) == (0.5, 9.0, False)
    assert a.visualization_b64 is None and a.error is None
    assert b.visualization_b64 == "iVBORw1="
    assert bad.error == "task needs image_b64" and bad.depth_b64 is None
    # The runner stores results via model_dump(); make sure that round-trips.
    dumped = json.loads(json.dumps([r.model_dump() for r in results]))
    assert dumped[1]["visualization_b64"] == "iVBORw1="


def test_hydrate_keeps_the_error_envelope():
    assert hydrate({"id": "t", "error": "boom"}).error == "boom"
    assert hydrate({"error": "boom"}).id == "unknown"


@pytest.fixture
def second_model(tmp_path, monkeypatch):
    """A second depth recipe with its own snapshot. The public catalog
    carries only Small (Base and Large are non-commercial), so the
    non-default paths are exercised with an operator recipe."""
    from giq.registry import reload_registry

    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    (tmp_path / "large.yaml").write_text(
        "name: depth-test-large\nmodalities: [depth]\nengine: transformers\n"
        "weights: {path: depth-test-large-hf}\nvram: {gb: 4.0}\n"
    )
    reload_registry()
    yield "depth-test-large"
    monkeypatch.undo()
    reload_registry()


def test_estimated_vram_comes_from_the_registry(second_model):
    assert DepthAdapter(DepthConfig()).estimated_vram_gb == 1.5
    assert DepthAdapter(DepthConfig(model=second_model)).estimated_vram_gb == 4.0


def test_unknown_model_fails_at_construction():
    with pytest.raises(ValueError):
        DepthAdapter(DepthConfig(model="midas"))


def test_the_child_gets_the_instance_weights_under_an_overridable_root(monkeypatch, second_model):
    from giq.paths import models_dir

    w = DepthAdapter(DepthConfig(model=second_model))
    assert w._command()[1:] == [
        "-u",
        "-m",
        "giq_depth._depth_child",
        "--model",
        second_model,
        "--weights",
        str(models_dir() / "depth-test-large-hf"),
    ]
    monkeypatch.setenv("GIQ_DEPTH_MODELS_DIR", "/elsewhere")
    w = DepthAdapter(DepthConfig())
    assert w.weights == "/elsewhere/depth-anything-Depth-Anything-V2-Small-hf"


# --- the child's arithmetic ---------------------------------------------------


def test_sixteen_bit_mapping_spans_the_prediction_and_survives_a_flat_map():
    from giq_depth._depth_child import _to_16bit

    depth = np.array([[0.5, 9.0], [4.75, 0.5]], dtype=np.float32)
    map16, lo, hi = _to_16bit(depth)
    assert map16.dtype == np.uint16 and (lo, hi) == (0.5, 9.0)
    assert map16[0, 0] == 0 and map16[0, 1] == 65535 and map16[1, 0] == 32768
    flat, lo, hi = _to_16bit(np.full((2, 2), 3.0, dtype=np.float32))
    assert flat.max() == 0 and lo == hi == 3.0


def test_visualization_paints_near_red_and_far_blue():
    from giq_depth._depth_child import visualize

    depth = np.array([[0.0, 1.0]], dtype=np.float32)  # inverse depth: 1.0 is near
    rgb = visualize(depth)
    assert rgb.shape == (1, 2, 3) and rgb.dtype == np.uint8
    far, near = rgb[0, 0], rgb[0, 1]
    assert tuple(near) == (0x9E, 0x01, 0x42)  # Spectral's red end
    assert tuple(far) == (0x5E, 0x4F, 0xA2)  # Spectral's blue end
    # A metric map (larger is farther) asks for the ramp the other way round.
    assert tuple(visualize(depth, near_is_high=False)[0, 0]) == (0x9E, 0x01, 0x42)
