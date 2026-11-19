# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The child's own logic, without MOSS or torch: what a task asks for, the
layout it loads, the compile switch, and a batch run against a fake
pipeline that records what it was called with."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from giq_sfx_child._sfx_child import (
    REQUIRED,
    SEED_RANGE,
    compile_env,
    missing_files,
    run_tasks,
    task_inputs,
)

DEFAULTS = {"steps": 100, "guidance": 4.0}


def test_a_bare_prompt_takes_the_recipes_defaults():
    got = task_inputs({"prompt": "  a door creaks  "}, DEFAULTS)
    assert got == {
        "prompt": "a door creaks",
        "negative_prompt": "",
        "seconds": 10.0,
        "steps": 100,
        "guidance": 4.0,
        "seed": None,
        "format": "flac",
    }


def test_a_task_overrides_steps_guidance_and_length():
    task = {"prompt": "x", "duration": 2.04, "steps": 30, "guidance": 6, "seed": 9, "format": "WAV"}
    got = task_inputs(task, DEFAULTS)
    assert (got["seconds"], got["steps"], got["guidance"]) == (2.0, 30, 6.0)
    assert (got["seed"], got["format"]) == (9, "wav")


@pytest.mark.parametrize(
    ("task", "field"),
    [
        ({}, "prompt"),
        ({"prompt": "   "}, "prompt"),
        ({"prompt": "x", "negative_prompt": 3}, "negative_prompt"),
        ({"prompt": "x", "duration": 0.2}, "duration"),
        ({"prompt": "x", "duration": 31}, "duration"),
        ({"prompt": "x", "duration": float("nan")}, "duration"),
        ({"prompt": "x", "duration": True}, "duration"),
        ({"prompt": "x", "steps": 0}, "steps"),
        ({"prompt": "x", "steps": 201}, "steps"),
        ({"prompt": "x", "steps": 10.5}, "steps"),
        ({"prompt": "x", "guidance": 0.5}, "guidance"),
        ({"prompt": "x", "guidance": "loud"}, "guidance"),
        ({"prompt": "x", "seed": -1}, "seed"),
        ({"prompt": "x", "seed": 1.5}, "seed"),
        ({"prompt": "x", "format": "ogg"}, "format"),
    ],
)
def test_a_bad_field_is_named(task, field):
    with pytest.raises(ValueError, match=field):
        task_inputs(task, DEFAULTS)


def test_the_longest_sound_is_the_pipelines():
    with pytest.raises(ValueError, match="0.5-20 s"):
        task_inputs({"prompt": "x", "duration": 25}, DEFAULTS, max_seconds=20)


def _layout(tmp_path):
    root = tmp_path / "moss"
    for want in REQUIRED:
        path = root / (want[0] if isinstance(want, tuple) else want)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"w")
    return root


def test_a_complete_layout_lacks_nothing(tmp_path):
    assert missing_files(str(_layout(tmp_path))) == []


def test_either_vae_checkpoint_will_do_and_a_missing_one_is_named(tmp_path):
    root = _layout(tmp_path)
    (root / "vae" / "vae_128d_48k.pth").unlink()
    assert missing_files(str(root)) == [
        "vae/vae_128d_48k.pth or vae/diffusion_pytorch_model.safetensors"
    ]
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"w")
    assert missing_files(str(root)) == []


def test_a_sharded_text_encoder_counts(tmp_path):
    root = _layout(tmp_path)
    (root / "text_encoder" / "model.safetensors").unlink()
    (root / "text_encoder" / "model.safetensors.index.json").write_text("{}")
    assert missing_files(str(root)) == []


def test_uncompiled_turns_dynamo_off():
    assert compile_env(False) == {"TORCHDYNAMO_DISABLE": "1"}


def test_compiled_keeps_its_kernels_in_giqs_cache(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/var/cache/giq")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    assert compile_env(True) == {"TORCHINDUCTOR_CACHE_DIR": "/var/cache/giq/torchinductor"}
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/elsewhere")
    assert compile_env(True) == {}
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR")
    assert compile_env(True) == {}


class _Tensor:
    """What encode needs of a torch tensor, over a numpy array."""

    def __init__(self, array):
        self.array = array

    def __getitem__(self, i):
        return _Tensor(self.array[i])

    def detach(self):
        return self

    def float(self):
        return _Tensor(self.array.astype(np.float32))

    def clamp(self, lo, hi):
        return _Tensor(np.clip(self.array, lo, hi))

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakePipeline:
    """MossSoundEffectPipeline's call shape: (B, C, T) audio, cropped to seconds."""

    sample_rate = 48000
    max_inference_seconds = 30

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, seconds, **kwargs):
        self.calls.append({"prompt": prompt, "seconds": seconds, **kwargs})
        # Past full scale on purpose: the encoder clamps it.
        return _Tensor(np.full((1, 1, int(self.sample_rate * seconds)), 1.5, dtype=np.float32))


ARGS = SimpleNamespace(steps=100, guidance=4.0, shift=5.0)


def test_a_batch_renders_each_task_and_reports_what_it_used():
    sf = pytest.importorskip("soundfile")
    pipe = FakePipeline()
    tasks = [
        {"id": "a", "prompt": "rain on a tin roof", "duration": 2, "seed": 42},
        {"id": "bad"},
        {
            "id": "b",
            "prompt": "thunder",
            "duration": 1,
            "steps": 20,
            "guidance": 1,
            "format": "wav",
        },
    ]
    a, bad, b = run_tasks(pipe, ARGS, tasks)

    assert bad == {"id": "bad", "error": "task needs a 'prompt' describing the sound"}
    assert (a["id"], a["seed"], a["steps"], a["guidance"]) == ("a", 42, 100, 4.0)
    assert (a["format"], a["sample_rate"], a["duration"]) == ("flac", 48000, 2.0)
    data, rate = sf.read(io.BytesIO(base64.b64decode(a["audio_b64"])))
    assert rate == 48000 and data.shape == (96000,) and data.max() <= 1.0

    assert (b["format"], b["steps"], b["guidance"], b["duration"]) == ("wav", 20, 1.0, 1.0)
    assert 0 <= b["seed"] < SEED_RANGE, "a task without a seed reports the one it drew"

    first, second = pipe.calls
    assert first["prompt"] == "rain on a tin roof" and first["seed"] == 42
    assert (first["num_inference_steps"], first["cfg_scale"], first["sigma_shift"]) == (
        100,
        4.0,
        5.0,
    )
    assert first["negative_prompt"] == ""
    assert (second["num_inference_steps"], second["cfg_scale"]) == (20, 1.0)


def test_a_pipeline_failure_is_that_tasks_error():
    class Broken(FakePipeline):
        def __call__(self, prompt, seconds, **kwargs):
            raise RuntimeError("CUDA out of memory")

    (result,) = run_tasks(Broken(), ARGS, [{"id": "x", "prompt": "boom"}])
    assert result == {"id": "x", "error": "CUDA out of memory"}
