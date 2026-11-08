# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The child's own logic, without ACE-Step: what a task asks for, what a
track reports, and the scratch checkpoint root ACE-Step loads from."""

import os

import pytest

from giq_music_child._music_child import REQUIRED, _scratch_root, task_inputs, track_meta


def test_a_bare_prompt_lets_the_planner_choose():
    got = task_inputs({"prompt": " warm lo-fi "}, planner=True)
    assert got["caption"] == "warm lo-fi"
    assert got["plan"] and got["duration"] == -1.0
    assert got["instrumental"] and got["format"] == "flac"


def test_without_a_planner_a_track_gets_a_length():
    got = task_inputs({"prompt": "x", "plan": True}, planner=False)
    assert not got["plan"] and got["duration"] == 30.0


def test_lyrics_make_it_a_song():
    got = task_inputs({"prompt": "x", "lyrics": "[verse]\nla la"}, planner=True)
    assert not got["instrumental"]


@pytest.mark.parametrize(
    ("task", "field"),
    [
        ({}, "prompt"),
        ({"prompt": "x", "duration": 5}, "duration"),
        ({"prompt": "x", "duration": 601}, "duration"),
        ({"prompt": "x", "duration": float("inf")}, "duration"),
        ({"prompt": "x", "duration": True}, "duration"),
        ({"prompt": "x", "bpm": 400}, "bpm"),
        ({"prompt": "x", "seed": -1}, "seed"),
        ({"prompt": "x", "seed": 1.5}, "seed"),
        ({"prompt": "x", "format": "ogg"}, "format"),
        ({"prompt": "x", "time_signature": "5/4"}, "time_signature"),
        ({"prompt": "x", "time_signature": "4/8"}, "time_signature"),
        ({"prompt": "x", "lyrics": 3}, "lyrics"),
    ],
)
def test_a_bad_field_is_named(task, field):
    with pytest.raises(ValueError, match=field):
        task_inputs(task, planner=True)


@pytest.mark.parametrize(("given", "beats"), [("3/4", "3"), ("6/8", "6"), (4, "4"), ("", "")])
def test_time_signatures_become_beats(given, beats):
    assert (
        task_inputs({"prompt": "x", "time_signature": given}, planner=True)["time_signature"]
        == beats
    )


def test_the_meta_is_the_request_where_it_gave_one_and_the_planner_elsewhere():
    inputs = task_inputs({"prompt": "x", "bpm": 120}, planner=True)
    used = {"cot_bpm": "88", "cot_keyscale": "G major", "cot_timesignature": "6", "seed": "4242"}
    meta = track_meta(used, inputs)
    assert meta["bpm"] == 120
    assert (meta["key_scale"], meta["time_signature"], meta["seed"]) == ("G major", "6/8", 4242)
    assert meta["caption"] == "x" and meta["lyrics"] is None


def test_the_planners_rewrite_is_reported():
    inputs = task_inputs({"prompt": "x"}, planner=True)
    meta = track_meta({"cot_caption": "a long caption", "cot_lyrics": "[verse]"}, inputs)
    assert (meta["caption"], meta["lyrics"]) == ("a long caption", "[verse]")


def _weights(tmp_path, dit="acestep-v15-turbo"):
    root = tmp_path / "weights"
    for name in {*REQUIRED, dit}:
        (root / name).mkdir(parents=True, exist_ok=True)
        weight = "diffusion_pytorch_model.safetensors" if name == "vae" else "model.safetensors"
        (root / name / weight).write_bytes(b"w")
    (root / dit / "modeling_acestep_v15_turbo.py").write_text("# model code\n")
    (root / dit / "config.json").write_text("{}")
    return root


def test_the_scratch_root_links_weights_and_copies_the_dits_code(tmp_path, monkeypatch):
    for var in ("ACESTEP_PROJECT_ROOT", "ACESTEP_CHECKPOINTS_DIR", "HF_MODULES_CACHE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    weights = _weights(tmp_path)
    root = _scratch_root(str(weights), "acestep-v15-turbo")

    assert os.path.islink(os.path.join(root, "vae"))
    dit = os.path.join(root, "acestep-v15-turbo")
    assert not os.path.islink(dit)
    assert os.path.islink(os.path.join(dit, "model.safetensors"))
    code = os.path.join(dit, "modeling_acestep_v15_turbo.py")
    assert not os.path.islink(code)
    # ACE-Step's code sync overwrites the copy, never the recipe's weights.
    with open(code, "w") as f:
        f.write("# synced\n")
    assert (
        weights / "acestep-v15-turbo" / "modeling_acestep_v15_turbo.py"
    ).read_text() == "# model code\n"
    assert os.environ["ACESTEP_CHECKPOINTS_DIR"] == root
    assert os.environ["HF_MODULES_CACHE"].startswith(os.path.dirname(root))


def test_a_missing_folder_fails_instead_of_downloading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = _weights(tmp_path)
    (weights / "vae" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="vae holds no weights"):
        _scratch_root(str(weights), "acestep-v15-turbo")
