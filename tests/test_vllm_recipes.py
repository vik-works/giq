# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The vllm engine in the recipe schema: budget, profiles, what the weights can do."""

import pytest
from pydantic import ValidationError

from giq import recipes
from giq.adapters.engine import context_size, engine_for
from giq.recipes.schema import Recipe
from tests._vllm import NAME, doc, make_checkpoint, make_recipe, params_of


def test_refuses_an_instance_without_a_budget():
    """ADR-002 D3: vllm reserves memory up front, so the budget is not optional."""
    with pytest.raises(ValidationError, match="VRAM budget is required"):
        params_of(kv_cache_memory=None)


def test_one_budget_not_two():
    with pytest.raises(ValidationError, match="one budget"):
        params_of(gpu_memory_utilization=0.9)


@pytest.mark.parametrize(
    "given,expected", [("8G", 8 * 2**30), ("512M", 512 * 2**20), (2**31, 2**31)]
)
def test_kv_budget_in_bytes(given, expected):
    assert params_of(kv_cache_memory=given).kv_cache_memory_bytes == expected


def test_a_kv_budget_needs_room_for_something():
    with pytest.raises(ValidationError, match="256M"):
        params_of(kv_cache_memory="64M")


def test_unknown_parameters_are_errors_not_ignored():
    with pytest.raises(ValidationError, match="max_num_seq"):
        params_of(max_num_seq=4)


@pytest.mark.parametrize(
    "bad",
    [
        {"gpu_memory_utilization": 1.5, "kv_cache_memory": None},
        {"kv_cache_dtype": "int4"},
        {"reasoning_parser": "qwen3 --trust-remote-code"},
        {"memory_max": "lots"},
        {"kv_cache_memory": "8 GB"},
        {"speculative": {"method": "eagle", "tokens": 2}},
        {"speculative": {"method": "mtp", "tokens": 0}},
    ],
)
def test_out_of_range_values_are_refused(bad):
    with pytest.raises(ValidationError):
        params_of(**bad)


def test_interactive_profile():
    p = params_of("interactive")
    assert p.speculative is not None and (p.speculative.method, p.speculative.tokens) == ("mtp", 2)
    assert (p.max_num_seqs, p.max_num_batched_tokens, p.kv_cache_dtype) == (4, 8192, "fp8")
    assert p.kv_cache_memory_bytes == 6 * 2**30, "the budget comes from the recipe"


def test_throughput_profile():
    p = params_of("throughput")
    assert p.speculative is None
    assert (p.max_num_seqs, p.kv_cache_dtype) == (32, "fp8")


def test_profiles_leave_the_budget_to_the_instance():
    from giq.recipes.schema import ENGINE_PROFILES

    for name, values in ENGINE_PROFILES["vllm"].items():
        assert not {"kv_cache_memory", "gpu_memory_utilization"} & set(values), name


def test_explicit_params_outrank_the_profile():
    p = params_of("interactive", speculative=None, max_num_seqs=8)
    assert p.speculative is None and p.max_num_seqs == 8


def test_unknown_profile():
    with pytest.raises(ValidationError, match="unknown vllm profile"):
        params_of("fastest")


def test_llama_cpp_still_has_no_profiles():
    with pytest.raises(ValidationError, match="no profiles"):
        Recipe.model_validate(
            {
                "name": "x",
                "worker": "llm",
                "engine": "llama.cpp",
                "weights": {"path": "x.gguf"},
                "profile": "interactive",
                "vram": {"gb": 1.0},
            }
        )


def test_the_vram_figure_is_weights_plus_kv_plus_overhead(tmp_path):
    recipe = make_recipe(make_checkpoint(tmp_path))
    assert recipe.vram.gb == pytest.approx(19.92 + 6 + 3.5)
    assert not recipe.vram.measured


def test_a_given_figure_must_agree_with_its_parts(tmp_path):
    vram = {"gb": 20.0, "weights_gb": 19.92, "overhead_gb": 3.5}
    with pytest.raises(ValidationError, match="disagrees"):
        Recipe.model_validate(doc(make_checkpoint(tmp_path), vram=vram))


def test_a_fraction_budget_needs_a_declared_figure(tmp_path):
    with pytest.raises(ValidationError, match=r"vram\.gb"):
        Recipe.model_validate(
            doc(make_checkpoint(tmp_path), kv_cache_memory=None, gpu_memory_utilization=0.9)
        )
    recipe = Recipe.model_validate(
        doc(
            make_checkpoint(tmp_path),
            vram={"gb": 29.5},
            kv_cache_memory=None,
            gpu_memory_utilization=0.9,
        )
    )
    assert recipe.vram.gb == 29.5


def test_vllm_reads_checkpoint_directories_only(tmp_path):
    bad = doc(make_checkpoint(tmp_path))
    bad["weights"]["format"] = "gguf"
    with pytest.raises(ValidationError, match="safetensors or modelopt"):
        Recipe.model_validate(bad)


def test_lanes_are_derived_not_set(tmp_path):
    bad = {**doc(make_checkpoint(tmp_path)), "lane_width": 4}
    with pytest.raises(ValidationError, match="derived from params.max_num_seqs"):
        Recipe.model_validate(bad)


def test_request_defaults_are_sampling_fields_only(tmp_path):
    ok = {**doc(make_checkpoint(tmp_path)), "request_defaults": {"top_k": 20}}
    assert Recipe.model_validate(ok).request_defaults == {"top_k": 20}
    bad = {**doc(make_checkpoint(tmp_path)), "request_defaults": {"dry_multiplier": 0.8}}
    with pytest.raises(ValidationError, match="not vllm sampling fields"):
        Recipe.model_validate(bad)


def test_mtp_is_refused_at_load_for_weights_without_the_head(tmp_path):
    weights = make_checkpoint(tmp_path, mtp=None)
    with pytest.raises(ValidationError, match="MTP head"):
        make_recipe(weights, "interactive")
    make_recipe(weights, "throughput")


def test_mtp_is_accepted_when_the_weights_are_not_here():
    """The loader can only check what is on this machine; start() checks again."""
    params_of("interactive")


# --- the built-in recipes --------------------------------------------------------


def test_the_builtin_instances():
    builtin = {recipe.name: recipe for recipe, _ in recipes.builtin().values()}
    throughput, chat = builtin[NAME], builtin[f"{NAME}-chat"]
    assert throughput.engine == chat.engine == "vllm"
    assert throughput.params.speculative is None and throughput.params.max_num_seqs == 16
    assert chat.params.speculative is not None and chat.params.speculative.tokens == 3
    assert chat.params.max_num_seqs == 4
    # 8192 ran a 32 GB card out of memory under concurrent long prompts.
    assert chat.params.max_num_batched_tokens == 4096
    for recipe in (throughput, chat):
        assert recipe.residency.priority is None, "vllm starts take minutes: the operator pins"
        assert recipe.params.kv_cache_memory_bytes is not None, "a KV size, not a card fraction"
        # Loadable on a 32 GB card through the gate (2 GB margin on 31.8 GiB).
        assert recipe.vram.gb <= 29.8


def test_the_catalog_lists_them_with_derived_lanes():
    from giq.registry import get_recipe

    recipe = get_recipe(NAME)
    assert recipe is not None and recipe.engine == "vllm"
    assert recipe.lanes == 16, "max_num_seqs, not the llm default of 4"
    assert get_recipe(f"{NAME}-chat").lanes == 4
    assert engine_for(NAME) == "vllm" and engine_for("qwen3.8-27b") == "llama.cpp"
    assert context_size(NAME) == 131072
