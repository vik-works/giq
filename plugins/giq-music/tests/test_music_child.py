# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The child's own logic, without ACE-Step: what a task asks for, what a
track reports, and the scratch checkpoint root ACE-Step loads from."""

import base64
import os

import pytest

from giq_music_child._audio import audio_bytes
from giq_music_child._music_child import (
    REQUIRED,
    _scratch_root,
    instruction,
    task_inputs,
    track_meta,
)

WAV = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()
BASE = frozenset({"cover", "lego", "complete"})
# ACE-Step's own templates (acestep.constants.TASK_INSTRUCTIONS), the ones used.
TEMPLATES = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
    "complete_default": "Complete the input track:",
}


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


def test_a_reference_steers_the_sound_and_leaves_the_planner_on():
    got = task_inputs({"prompt": "x", "reference_audio_b64": WAV}, planner=True)
    assert (got["mode"], got["reference"], got["source"]) == ("text2music", WAV, None)
    assert got["plan"] and got["cover_strength"] is None


def test_source_audio_is_covered_unplanned_at_its_own_length():
    got = task_inputs(
        {"prompt": "thrash metal", "source_audio_b64": WAV},
        planner=True,
        modes=frozenset({"cover"}),
    )
    assert (got["mode"], got["cover_strength"]) == ("cover", 0.5)
    assert not got["plan"] and got["duration"] == -1.0
    assert instruction(got, TEMPLATES) == TEMPLATES["cover"]


def test_lego_adds_one_track_over_the_source():
    task = {"prompt": "riffs", "source_audio_b64": WAV, "mode": "lego", "track": "guitar"}
    got = task_inputs(task, planner=True, modes=BASE)
    assert instruction(got, TEMPLATES) == "Generate the GUITAR track based on the audio context:"


def test_complete_names_the_tracks_it_fills_in():
    task = {"prompt": "x", "source_audio_b64": WAV, "mode": "complete"}
    assert instruction(task_inputs(task, True, BASE), TEMPLATES) == TEMPLATES["complete_default"]
    got = task_inputs({**task, "tracks": ["bass", "guitar"]}, True, BASE)
    assert instruction(got, TEMPLATES) == "Complete the input track with BASS | GUITAR:"


@pytest.mark.parametrize(
    ("task", "modes", "match"),
    [
        ({"mode": "cover"}, BASE, "needs a 'source_audio_b64'"),
        ({"source_audio_b64": WAV}, frozenset(), "does no cover"),
        ({"source_audio_b64": WAV, "mode": "lego", "track": "bass"}, {"cover"}, "does no lego"),
        ({"source_audio_b64": WAV, "mode": "remix"}, BASE, "mode must be"),
        ({"source_audio_b64": WAV, "mode": "lego"}, BASE, "needs a 'track'"),
        ({"source_audio_b64": WAV, "mode": "lego", "track": "kazoo"}, BASE, "needs a 'track'"),
        ({"source_audio_b64": WAV, "cover_strength": 1.5}, BASE, "cover_strength"),
        ({"source_audio_b64": WAV, "mode": "lego", "track": "bass", "cover_strength": 0.3},
         BASE, "cover_strength"),
        ({"source_audio_b64": WAV, "track": "bass"}, BASE, "'track' is for"),
        ({"source_audio_b64": WAV, "mode": "complete", "tracks": "bass"}, BASE, "'tracks'"),
        ({"reference_audio_b64": 7}, BASE, "reference_audio_b64"),
    ],
)  # fmt: skip
def test_audio_fields_that_do_not_fit_are_named(task, modes, match):
    with pytest.raises(ValueError, match=match):
        task_inputs({"prompt": "x", **task}, planner=True, modes=frozenset(modes))


@pytest.mark.parametrize(
    ("head", "suffix"),
    [
        (b"RIFF\x00\x00\x00\x00WAVE", ".wav"),
        (b"fLaC\x00", ".flac"),
        (b"OggS\x00", ".ogg"),
        (b"ID3\x04", ".mp3"),
        (b"\xff\xfb\x90", ".mp3"),
    ],
)
def test_task_audio_is_told_by_its_bytes(head, suffix):
    data, got = audio_bytes(base64.b64encode(head).decode(), "source_audio_b64")
    assert (data, got) == (head, suffix)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "base64 audio"),
        ("@@@", "not valid base64"),
        (base64.b64encode(b"%PDF").decode(), "WAV"),
    ],
)
def test_task_audio_that_is_not_audio_is_refused(value, match):
    with pytest.raises(ValueError, match=match):
        audio_bytes(value, "source_audio_b64")


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


def test_a_dit_from_its_own_repository_joins_the_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = _weights(tmp_path)
    xl = tmp_path / "xl-sft"
    xl.mkdir()
    # The XL DiTs are sharded: the index stands for the weights.
    (xl / "model.safetensors.index.json").write_text("{}")
    (xl / "model-00001-of-00004.safetensors").write_bytes(b"w")
    (xl / "configuration_acestep_v15.py").write_text("# code\n")
    root = _scratch_root(str(weights), "acestep-v15-xl-sft", str(xl))

    dit = os.path.join(root, "acestep-v15-xl-sft")
    assert not os.path.islink(dit)
    assert os.path.realpath(os.path.join(dit, "model-00001-of-00004.safetensors")) == str(
        xl / "model-00001-of-00004.safetensors"
    )
    assert not os.path.islink(os.path.join(dit, "configuration_acestep_v15.py"))
    # The turbo DiT ACE-Step checks for is still there, linked.
    assert os.path.islink(os.path.join(root, "acestep-v15-turbo"))


def test_a_dit_repository_without_weights_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = _weights(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="empty holds no weights"):
        _scratch_root(str(weights), "acestep-v15-xl-sft", str(tmp_path / "empty"))
