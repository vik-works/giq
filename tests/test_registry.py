# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog is the single source of truth for what recipes exist.

These tests exist because the list used to live in six places and drifted:
/capabilities advertised tts/kokoro-82m, a name absent from the VRAM table,
so it inherited the 22GB default and could never load on a 16GB card.
"""

import asyncio

import pytest

from giq.models import Modality
from giq.registry import (
    all_recipes,
    get_recipe,
    lane_width_for,
    recipes_serving,
    reload_registry,
    resident_defaults,
    vram_for,
)
from giq.vram import DEFAULT_VRAM_REQUIREMENT, get_vram_requirement, get_vram_status, margin_for


@pytest.fixture(autouse=True)
def _fresh_registry():
    reload_registry()
    yield
    reload_registry()


def test_every_name_and_alias_means_one_recipe():
    """A recipe's name is what a client sends as `model` (ADR-003)."""
    names = [n for r in all_recipes() for n in (r.name, *r.aliases)]
    assert len(names) == len(set(names))


def test_modalities_are_real():
    """A typo'd modality would silently create an unreachable recipe."""
    for recipe in all_recipes():
        for modality in recipe.modalities:
            Modality(modality)  # raises on an unknown one


def test_vram_lookup_goes_through_the_catalog():
    for recipe in all_recipes():
        assert get_vram_requirement(recipe.name) == recipe.vram_gb


def test_unknown_recipe_falls_back_to_the_default():
    assert get_vram_requirement("no-such-model") == DEFAULT_VRAM_REQUIREMENT


def test_aliases_resolve_to_the_same_recipe():
    for recipe in all_recipes():
        for alias in recipe.aliases:
            assert get_recipe(alias) is recipe


def test_kokoro_82m_alias_is_loadable():
    """The regression: the advertised name must resolve to the real footprint.

    Unaliased it fell through to the 22GB default, and wait_for_vram refuses
    anything larger than the card outright — so /v1/audio/speech failed with
    "VRAM never became available" rather than loading a 0.5GB model.
    """
    total = get_vram_status().total_gb
    req = get_vram_requirement("kokoro-82m")
    assert req == get_vram_requirement("kokoro")
    assert req + margin_for(req) <= total


def test_advertised_models_are_all_recipes_serving_that_modality():
    """Anything /capabilities lists must be schedulable, not just nameable."""
    from giq.api.router import get_capabilities

    caps = asyncio.run(get_capabilities())
    for modality, cap in caps.modalities.items():
        for name in cap.recipes:
            recipe = get_recipe(name)
            assert recipe is not None and recipe.serves(modality), f"{modality}/{name}"


def test_capabilities_covers_every_modality_a_recipe_serves():
    from giq.api.router import get_capabilities

    caps = asyncio.run(get_capabilities())
    served = {m for r in all_recipes() for m in r.modalities}
    assert {str(m) for m in caps.modalities} == served


def test_a_recipe_is_listed_under_every_modality_it_serves():
    """flux_klein renders and edits: one recipe, both image tabs."""
    assert "flux_klein" in [r.name for r in recipes_serving("text2image")]
    assert "flux_klein" in [r.name for r in recipes_serving("image_edit")]
    assert "zimage" not in [r.name for r in recipes_serving("image_edit")]


def test_nothing_is_kept_warm_out_of_the_box():
    """A new install loads nothing until a job asks for it; keeping a model
    warm is the operator's decision, in config.yaml or the dashboard."""
    assert resident_defaults() == []
    assert all(r.residency.priority is None for r in all_recipes())


def test_configured_residents_keep_their_order_and_skip_unknown_names(config_residents):
    config_residents("whisper-large-v3", "llm/gemma-4-12b", "no-such-recipe", "ecapa-tdnn")
    assert resident_defaults() == ["whisper-large-v3", "gemma-4-12b", "ecapa-tdnn"]


def test_runner_residents_match_the_catalog():
    from giq.runner import RESIDENTS_DEFAULT

    assert RESIDENTS_DEFAULT == resident_defaults()


def test_lane_width_falls_back_to_the_modality_default():
    assert lane_width_for("gemma-4-12b") == 4, "llm's registered default"
    assert lane_width_for("unknown", "llm") == 4
    assert lane_width_for("tiny") == 1


def test_vram_for_reads_the_recipe_or_the_default():
    assert vram_for("zimage", default=99.0) == 13.0
    assert vram_for("nothing", default=99.0) == 99.0


def test_config_residents_override_the_builtin_set(monkeypatch):
    class _Cfg:
        residents = ["ecapa-tdnn", "gemma-4-12b"]

    monkeypatch.setattr("giq.config.get_config", lambda: _Cfg())
    assert resident_defaults() == ["ecapa-tdnn", "gemma-4-12b"]


def test_config_residents_written_before_adr_003_still_count(monkeypatch):
    """`llm/gemma-4-12b` was the form before names were unique."""

    class _Cfg:
        residents = ["embed/ecapa-tdnn", "llm/gemma-4-12b"]

    monkeypatch.setattr("giq.config.get_config", lambda: _Cfg())
    assert resident_defaults() == ["ecapa-tdnn", "gemma-4-12b"]


def test_unknown_config_resident_is_ignored_not_fatal(monkeypatch):
    class _Cfg:
        residents = ["does-not-exist", "gemma-4-12b"]

    monkeypatch.setattr("giq.config.get_config", lambda: _Cfg())
    assert resident_defaults() == ["gemma-4-12b"]


def test_estimated_vram_matches_the_gate():
    """The eviction planner and the VRAM gate must read the same number.

    They used to disagree: sd.cpp put zimage at 9GB while the gate wanted 13,
    so a victim's freed VRAM was mis-sized.
    """
    from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig
    from giq_sdcpp.adapter import SdCppAdapter, SdCppConfig

    llm = LlamaCppAdapter(config=LlamaCppConfig(model="gemma-4-12b"))
    assert llm.estimated_vram_gb == get_vram_requirement("gemma-4-12b")

    for model in ("flux_klein", "zimage"):
        worker = SdCppAdapter(config=SdCppConfig(model=model))
        assert worker.estimated_vram_gb == get_vram_requirement(model)
