# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The built-in recipe files reproduce the catalog they replaced.

`_catalog_baseline` is the registry and the per-model llm tables exactly as
the hand-written code defined them. Every field of every (worker, model)
entry and every table entry — including which models are *absent* from a
table, since absence means "take the default" — must come out of the YAML
unchanged. A recipe serving several modalities (ADR-003) reproduces one
entry per modality: flux_klein still renders in batches of 8 and edits in
batches of 4.
"""

import pytest

from giq import recipes
from giq.adapters import llama_cpp
from giq.recipes.schema import Recipe
from giq_vllm.params import VllmParams
from tests._catalog_baseline import BUILTIN_SPECS, LLM_DEFAULTS, LLM_TABLES


def _builtin_instances():
    return [recipe for recipe, _ in recipes.builtin().values()]


def _flat(recipe: Recipe, modality: str) -> dict:
    """A recipe, as the old catalog wrote one (worker, model) entry of it."""
    params = recipe.params
    return {
        "aliases": recipe.aliases,
        "backend": recipe.engine,
        "detail": recipe.detail,
        "engine": recipe.engine if modality in ("text2image", "image_edit") else None,
        "label": recipe.label,
        "lane_width": params.max_num_seqs if isinstance(params, VllmParams) else recipe.lane_width,
        "max_batch": recipe.max_batch_for(modality),
        "measured": recipe.measured,
        "mmproj": recipe.mmproj,
        "model": recipe.name,
        "resident_priority": recipe.residency.priority,
        "vision": recipe.vision,
        "voices": recipe.voices,
        "vram_gb": recipe.vram_gb,
        "worker": modality,
    }


def _builtin_specs() -> dict[tuple[str, str], dict]:
    return {
        (modality, recipe.name): _flat(recipe, modality)
        for recipe in _builtin_instances()
        for modality in recipe.modalities
    }


def test_every_builtin_file_is_valid():
    """A broken built-in raises here rather than at a customer's startup."""
    assert recipes.builtin()


def test_the_same_models_exist():
    assert set(_builtin_specs()) == {(d["worker"], d["model"]) for d in BUILTIN_SPECS}


@pytest.mark.parametrize("expected", BUILTIN_SPECS, ids=lambda d: f"{d['worker']}/{d['model']}")
def test_every_spec_field_is_reproduced(expected):
    actual = _builtin_specs()[(expected["worker"], expected["model"])]
    assert actual == expected


@pytest.mark.parametrize("table", sorted(LLM_TABLES))
def test_every_llm_table_is_reproduced(table):
    assert llama_cpp.tables_of(_builtin_instances())[table] == LLM_TABLES[table]


@pytest.mark.parametrize("name", sorted(LLM_DEFAULTS))
def test_the_engine_defaults_did_not_move(name):
    """The tables are sparse over these; a changed default would change every
    model that does not set the parameter, without touching a single file."""
    assert getattr(llama_cpp, name) == LLM_DEFAULTS[name]


# --- operator recipes reach the consumers -----------------------------------


@pytest.fixture
def operator_dir(tmp_path, monkeypatch):
    """An operator recipes directory, with the catalog rebuilt around it."""
    from giq.registry import reload_registry

    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path))
    yield tmp_path
    monkeypatch.undo()
    reload_registry()


def test_an_operator_instance_is_served_like_a_builtin(operator_dir):
    from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig
    from giq.registry import get_recipe, reload_registry

    (operator_dir / "private.yaml").write_text(
        "name: private-ft\nmodalities: [llm]\nengine: llama.cpp\n"
        "weights: {path: /srv/models/private-ft.gguf}\n"
        "params: {ctx_size: 32768, reasoning: 'on'}\n"
        "vram: {gb: 12.0}\n"
    )
    reload_registry()

    assert get_recipe("private-ft").vram_gb == 12.0
    cmd = LlamaCppAdapter(LlamaCppConfig(model="private-ft")).build_command()
    assert cmd[cmd.index("-m") + 1] == "/srv/models/private-ft.gguf"
    assert cmd[cmd.index("-c") + 1] == "32768"


def test_an_operator_override_replaces_the_builtin_everywhere(operator_dir):
    """Parameters the override leaves out fall back to the engine defaults —
    they are not inherited from the built-in it replaces."""
    from giq.registry import get_recipe, reload_registry

    (operator_dir / "gemma.yaml").write_text(
        "name: gemma-4-12b\nmodalities: [llm]\nengine: llama.cpp\n"
        "weights: {path: elsewhere/gemma.gguf}\nvram: {gb: 7.5, measured: true}\n"
    )
    reload_registry()

    assert get_recipe("gemma-4-12b").vram_gb == 7.5
    assert get_recipe("gemma-4-12b").residency.priority is None
    assert llama_cpp.MODEL_PATHS["gemma-4-12b"] == "elsewhere/gemma.gguf"
    assert "gemma-4-12b" not in llama_cpp.MODEL_PARALLEL
    assert "gemma-4-12b" not in llama_cpp.MODEL_CTX_SIZE


def test_the_tables_return_to_the_builtins_when_the_override_goes(operator_dir):
    from giq.registry import reload_registry

    path = operator_dir / "gemma.yaml"
    path.write_text(
        "name: gemma-4-12b\nmodalities: [llm]\nengine: llama.cpp\n"
        "weights: {path: elsewhere/gemma.gguf}\nvram: {gb: 7.5}\n"
    )
    reload_registry()
    path.unlink()
    reload_registry()

    for table, expected in LLM_TABLES.items():
        assert getattr(llama_cpp, table) == expected


def test_a_broken_operator_file_leaves_the_catalog_serving(operator_dir):
    from giq.registry import get_recipe, reload_registry

    (operator_dir / "gemma.yaml").write_text("name: gemma-4-12b\nmodalities: [llm]\nctx: 1\n")
    reload_registry()

    assert get_recipe("gemma-4-12b").vram_gb == 9.5
    assert llama_cpp.MODEL_PARALLEL["gemma-4-12b"] == 4


async def test_storage_reports_the_operator_files_and_what_was_left_out(operator_dir, monkeypatch):
    """A broken file is logged and left out; /storage is where the dashboard
    learns of it, so an operator does not have to read the journal."""
    from httpx import ASGITransport, AsyncClient

    from giq.api import stats_api
    from giq.main import app
    from giq.registry import reload_registry

    (operator_dir / "gemma.yaml").write_text(
        "name: gemma-4-12b\nmodalities: [llm]\nengine: llama.cpp\n"
        "weights: {path: elsewhere/gemma.gguf}\nvram: {gb: 7.5}\n"
    )
    (operator_dir / "private.yaml").write_text(
        "name: private-ft\nmodalities: [llm]\nengine: llama.cpp\n"
        "weights: {path: private.gguf}\nvram: {gb: 12.0}\n"
    )
    (operator_dir / "broken.yaml").write_text("name: broken\nmodalities: [llm]\nctx: 1\n")
    reload_registry()
    monkeypatch.setattr(stats_api, "disk_report", lambda: [])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as c:
        block = (await c.get("/storage")).json()["recipes"]

    assert block["dir"] == str(operator_dir)
    assert [(f["name"], f["replaces_builtin"]) for f in block["files"]] == [
        ("gemma-4-12b", True),
        ("private-ft", False),
    ]
    assert block["overrides"] == ["gemma-4-12b"]
    (error,) = block["errors"]
    assert error["file"] == str(operator_dir / "broken.yaml")
    assert "ctx" in error["message"] and str(operator_dir) not in error["message"]
