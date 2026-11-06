# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""_render_size: a request may pick its size, within the card's budget."""

import pytest

from giq_sdcpp.adapter import MAX_RENDER_PIXELS, _render_size


def test_default_when_nothing_asked():
    assert _render_size({}, None, 800, 480) == (800, 480)


def test_task_wins_over_params():
    assert _render_size(
        {"width": 1024, "height": 768}, {"width": 512, "height": 512}, 800, 480
    ) == (1024, 768)


def test_params_used_when_task_is_silent():
    assert _render_size({}, {"width": 512, "height": 640}, 800, 480) == (512, 640)


def test_snapped_to_multiples_of_16():
    assert _render_size({"width": 1000, "height": 750}, None, 800, 480) == (992, 736)


def test_capped_to_the_pixel_budget_keeping_the_aspect():
    w, h = _render_size({"width": 4096, "height": 2304}, None, 800, 480)
    assert w * h <= MAX_RENDER_PIXELS
    assert w % 16 == 0 and h % 16 == 0
    assert w / h == pytest.approx(16 / 9, rel=0.02)


@pytest.mark.parametrize(
    "bad", [{"width": "wide", "height": 480}, {"width": -5, "height": 480}, {"width": 0}]
)
def test_unusable_sizes_fall_back_to_the_default(bad):
    assert _render_size(bad, None, 800, 480) == (800, 480)


def test_floor_of_256():
    assert _render_size({"width": 64, "height": 64}, None, 800, 480) == (256, 256)
