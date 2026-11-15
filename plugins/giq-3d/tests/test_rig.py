# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Fitting views to the multi-view rig: pure image arithmetic, no model."""

import pytest

Image = pytest.importorskip("PIL.Image")

from giq_3d_child.rig import FILL, SIZE, fit_views, object_box  # noqa: E402


def _view(width: int, height: int, box: tuple[int, int, int, int]):
    """A transparent picture with an opaque block where the object is."""
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    image.paste((200, 60, 20, 255), box)
    return image


def test_the_object_box_is_what_is_opaque():
    assert object_box(_view(100, 80, (10, 20, 60, 70)).getchannel("A")) == (10, 20, 60, 70)


def test_a_view_with_nothing_in_it_is_refused():
    with pytest.raises(ValueError, match="no object in it"):
        object_box(Image.new("RGBA", (10, 10), (0, 0, 0, 0)).getchannel("A"))


def test_views_drawn_at_different_sizes_come_out_at_one_height():
    # A crate 2 wide, 1 deep, 1 high: its long side drawn large, its short
    # side drawn to fill the frame as well, and off-centre.
    long_side = _view(1000, 1000, (100, 300, 900, 700))  # 800 x 400
    short_side = _view(800, 600, (100, 0, 700, 600))  # 600 x 600
    front, side = fit_views([long_side, short_side])
    fb, sb = object_box(front.getchannel("A")), object_box(side.getchannel("A"))
    heights = (fb[3] - fb[1], sb[3] - sb[1])
    assert abs(heights[0] - heights[1]) <= 1, "one object, one height"
    assert fb[2] - fb[0] == FILL, "the widest view fills the frame's share"
    assert abs((sb[2] - sb[0]) - heights[1]) <= 1, "the square side stays square"
    for view, box in ((front, fb), (side, sb)):
        assert view.size == (SIZE, SIZE) and view.mode == "RGBA"
        assert abs((box[0] + box[2]) - SIZE) <= 1 and abs((box[1] + box[3]) - SIZE) <= 1, "centred"


def test_a_tall_object_is_held_by_its_height():
    tall = _view(400, 900, (150, 50, 250, 850))  # 100 x 800
    (view,) = fit_views([tall])
    box = object_box(view.getchannel("A"))
    assert box[3] - box[1] == FILL and box[2] - box[0] < FILL // 4
