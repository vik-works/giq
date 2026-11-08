# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The Stable Audio 3 child's own logic, without Stable Audio 3: what a task
asks for, and the model config pointed at the recipe's folder."""

import json

import pytest

from giq_music_child._sa3_child import TEXT_ENCODER, local_config, task_inputs


def test_a_task_without_a_length_gets_the_recipes_default():
    got = task_inputs({"prompt": " rain on a tin roof "}, max_seconds=120, default_seconds=10)
    assert got == {
        "prompt": "rain on a tin roof",
        "negative_prompt": None,
        "duration": 10,
        "seed": None,
        "format": "flac",
    }


def test_a_blank_negative_prompt_is_none():
    got = task_inputs({"prompt": "x", "negative_prompt": "  "}, 120, 30)
    assert got["negative_prompt"] is None


@pytest.mark.parametrize(
    ("task", "field"),
    [
        ({}, "prompt"),
        ({"prompt": "  "}, "prompt"),
        ({"prompt": "x", "duration": 0.5}, "duration"),
        ({"prompt": "x", "duration": 121}, "duration"),
        ({"prompt": "x", "duration": "long"}, "duration"),
        ({"prompt": "x", "negative_prompt": 3}, "negative_prompt"),
        ({"prompt": "x", "seed": -2}, "seed"),
        ({"prompt": "x", "format": "ogg"}, "format"),
    ],
)
def test_a_bad_field_is_named(task, field):
    with pytest.raises(ValueError, match=field):
        task_inputs(task, max_seconds=120, default_seconds=30)


def _folder(tmp_path, conditioning):
    (tmp_path / TEXT_ENCODER).mkdir()
    (tmp_path / TEXT_ENCODER / "model.safetensors").write_bytes(b"w")
    (tmp_path / "model.safetensors").write_bytes(b"w")
    config = {"sample_rate": 44100, "model": {"conditioning": {"configs": conditioning}}}
    (tmp_path / "model_config.json").write_text(json.dumps(config))
    return tmp_path


def test_the_text_encoder_loads_from_the_folder(tmp_path):
    weights = _folder(
        tmp_path,
        [
            {
                "id": "prompt",
                "type": "t5gemma",
                "config": {"repo_id": "stabilityai/x", "subfolder": TEXT_ENCODER},
            },
            {"id": "seconds_total", "type": "number", "config": {"max_val": 380}},
        ],
    )
    config = local_config(str(weights))
    t5, seconds = config["model"]["conditioning"]["configs"]
    assert t5["config"] == {"model_path": str(weights / TEXT_ENCODER)}
    assert seconds["config"] == {"max_val": 380}


def test_a_missing_part_fails_instead_of_downloading(tmp_path):
    weights = _folder(tmp_path, [])
    (weights / TEXT_ENCODER / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match=TEXT_ENCODER):
        local_config(str(weights))


def test_a_config_without_the_text_encoder_is_refused(tmp_path):
    weights = _folder(tmp_path, [{"id": "s", "type": "number", "config": {}}])
    with pytest.raises(RuntimeError, match="T5Gemma"):
        local_config(str(weights))
