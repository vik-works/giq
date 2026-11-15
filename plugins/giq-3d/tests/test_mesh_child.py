# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The image2mesh child's request checking, which is pure: no Pixal3D, no GPU."""

import base64

import pytest

from giq_3d_child._mesh_child import CLOSE, FACES, TEXTURE_SIZE, task_inputs

IMAGE = base64.b64encode(b"\x89PNG not really").decode()


def test_a_bare_task_is_a_game_export_with_the_defaults():
    inputs = task_inputs({"id": "a", "image_b64": IMAGE})
    assert inputs["image"] == b"\x89PNG not really"
    assert inputs["mode"] == "game"
    assert inputs["faces"] == FACES["game"][0]
    assert inputs["texture_size"] == TEXTURE_SIZE["game"]
    assert (inputs["close"], inputs["seed"], inputs["fov"]) == (CLOSE, 42, None)


def test_a_full_export_has_its_own_defaults_and_range():
    inputs = task_inputs({"image_b64": IMAGE, "mode": "full"})
    assert (inputs["faces"], inputs["texture_size"]) == (FACES["full"][0], TEXTURE_SIZE["full"])
    with pytest.raises(ValueError, match="'faces' must be 5000-1000000 for a full export"):
        task_inputs({"image_b64": IMAGE, "mode": "full", "faces": 800})


def test_a_data_url_prefix_is_accepted():
    assert task_inputs({"image_b64": "data:image/png;base64," + IMAGE})["image"].startswith(
        b"\x89PNG"
    )


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({}, "needs an 'image_b64'"),
        ({"image_b64": "not base64!"}, "is not base64"),
        ({"image_b64": IMAGE, "mode": "preview"}, "'mode' must be one of game, full"),
        ({"image_b64": IMAGE, "faces": 50}, "'faces' must be 200-50000 for a game export"),
        ({"image_b64": IMAGE, "faces": 1500.5}, "'faces' must be a whole number"),
        ({"image_b64": IMAGE, "faces": True}, "'faces' must be a whole number"),
        ({"image_b64": IMAGE, "texture_size": 3000}, "'texture_size' must be one of"),
        ({"image_b64": IMAGE, "close": 20}, "'close' must be 0-8"),
        ({"image_b64": IMAGE, "seed": -1}, "'seed' must be 0 to 4294967295"),
        ({"image_b64": IMAGE, "fov": 9}, "'fov' is the camera's horizontal field of view"),
    ],
)
def test_a_wrong_field_is_named(task, message):
    with pytest.raises(ValueError, match=message):
        task_inputs(task)


def test_whole_floats_count_as_whole_numbers():
    assert task_inputs({"image_b64": IMAGE, "faces": 1500.0, "seed": 7.0})["faces"] == 1500
