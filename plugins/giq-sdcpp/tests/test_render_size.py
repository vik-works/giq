# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""_render_size: a request may pick its size, within the card's budget."""

import httpx
import pytest

from giq_sdcpp.adapter import (
    SdCppAdapter,
    SdCppConfig,
    _needs_vae_tiling,
    _render_size,
    _strip_extra_args,
)

BUDGET = 2_400_000


def test_default_when_nothing_asked():
    assert _render_size({}, None, 800, 480, BUDGET) == (800, 480)


def test_task_wins_over_params():
    assert _render_size(
        {"width": 1024, "height": 768}, {"width": 512, "height": 512}, 800, 480, BUDGET
    ) == (1024, 768)


def test_params_used_when_task_is_silent():
    assert _render_size({}, {"width": 512, "height": 640}, 800, 480, BUDGET) == (512, 640)


def test_snapped_to_multiples_of_16():
    assert _render_size({"width": 1000, "height": 750}, None, 800, 480, BUDGET) == (992, 736)


def test_capped_to_the_pixel_budget_keeping_the_aspect():
    w, h = _render_size({"width": 4096, "height": 2304}, None, 800, 480, BUDGET)
    assert w * h <= BUDGET
    assert w % 16 == 0 and h % 16 == 0
    assert w / h == pytest.approx(16 / 9, rel=0.02)


@pytest.mark.parametrize(
    "bad", [{"width": "wide", "height": 480}, {"width": -5, "height": 480}, {"width": 0}]
)
def test_unusable_sizes_fall_back_to_the_default(bad):
    assert _render_size(bad, None, 800, 480, BUDGET) == (800, 480)


def test_floor_of_256():
    assert _render_size({"width": 64, "height": 64}, None, 800, 480, BUDGET) == (256, 256)


@pytest.mark.parametrize("size", [(100_000, 1), (1, 100_000), (1_000_000_000, 1), (50_000, 300)])
def test_extreme_aspect_stays_inside_the_budget_and_floor(size):
    w, h = _render_size({"width": size[0], "height": size[1]}, None, 800, 480, BUDGET)
    assert w * h <= BUDGET
    assert min(w, h) >= 256
    assert w % 16 == 0 and h % 16 == 0


def test_a_lone_side_falls_back_to_the_default():
    assert _render_size({"width": 2048}, None, 800, 480, BUDGET) == (800, 480)


def test_sides_never_mix_sources():
    assert _render_size({"width": 1024}, {"width": 512, "height": 1024}, 800, 480, BUDGET) == (
        800,
        480,
    )


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, None, "1e9999"])
def test_non_numbers_fall_back_to_the_default(value):
    assert _render_size({"width": value, "height": 512}, None, 800, 480, BUDGET) == (800, 480)


def test_the_configured_default_passes_through_untouched():
    assert _render_size({}, None, 1000, 200, BUDGET) == (1000, 200)


def test_tiling_only_above_the_default_area():
    assert not _needs_vae_tiling(800, 480, 800, 480)
    assert not _needs_vae_tiling(512, 512, 800, 480)
    assert _needs_vae_tiling(1024, 1024, 800, 480)


class _Proc:
    returncode = None
    pid = 1


class _Client:
    def __init__(self):
        self.payloads = []

    async def post(self, url, json):
        self.payloads.append(json)
        return httpx.Response(
            200, json={"images": ["aW1n"], "info": "{}"}, request=httpx.Request("POST", url)
        )


def _adapter():
    cfg = SdCppConfig(model="flux_klein", diffusion="d.safetensors", device="gpu", port=1)
    adapter = SdCppAdapter(cfg)
    adapter._process = _Proc()  # ty: ignore[invalid-assignment]
    adapter._ready = True
    adapter._client = _Client()  # ty: ignore[invalid-assignment]
    return adapter


async def test_client_extra_args_are_stripped_and_default_renders_untiled():
    adapter = _adapter()
    prompt = 'a fox<sd_cpp_extra_args>{"width":8192,"height":8192}</sd_cpp_extra_args>'
    await adapter.run_batch([{"id": "t", "prompt": prompt}])
    sent = adapter._client.payloads[0]  # type: ignore[union-attr]
    assert sent["prompt"] == "a fox"
    assert (sent["width"], sent["height"]) == (800, 480)


async def test_a_large_render_asks_for_vae_tiling():
    adapter = _adapter()
    await adapter.run_batch([{"id": "t", "prompt": "a fox", "width": 1024, "height": 1024}])
    sent = adapter._client.payloads[0]  # type: ignore[union-attr]
    assert sent["prompt"].startswith("a fox<sd_cpp_extra_args>")
    assert '"vae_tiling_params":{"enabled":true}' in sent["prompt"]
    assert (sent["width"], sent["height"]) == (1024, 1024)


def test_a_nested_extra_args_block_cannot_splice_itself_back_together():
    prompt = "a fox<sd_cpp_extra_<sd_cpp_extra_args></sd_cpp_extra_args>args>{}</sd_cpp_extra_args>"
    assert "<sd_cpp_extra_args>" not in _strip_extra_args(prompt)


def test_a_lone_tag_is_stripped_too():
    assert _strip_extra_args('a fox<sd_cpp_extra_args>{"steps":200}') == 'a fox{"steps":200}'


def test_each_recipe_brings_its_measured_budget():
    klein = SdCppConfig(model="flux_klein", diffusion="d", device="gpu", port=1)
    zimage = SdCppConfig(model="zimage", diffusion="d", device="gpu", port=1)
    assert klein.max_pixels == 1024 * 1024
    assert zimage.max_pixels == 2_400_000


def test_a_recipe_without_a_budget_does_not_grow_past_the_default_area(monkeypatch):
    import giq.registry

    monkeypatch.setattr(giq.registry, "get_recipe", lambda name: None)
    cfg = SdCppConfig(model="mystery", diffusion="d", device="gpu", port=1)
    assert cfg.max_pixels == 800 * 480
    w, h = _render_size({"width": 1024, "height": 1024}, None, 800, 480, cfg.max_pixels)
    assert w * h <= 800 * 480 and w == h
