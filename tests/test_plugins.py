# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The plugin registry (ADR-004): built-ins through the same contract a
third party uses, a plugin's modality served end to end, and refusals that
leave everything else standing."""

from pathlib import Path

import pytest

from giq import plugins, recipes
from giq.models import JobRequest
from giq.plugin import API_VERSION, AdapterContext, Engine, Modality, Plugin
from giq.registry import get_recipe, reload_registry

BUILTIN_MODALITIES = {
    "llm",
    "text2image",
    "image_edit",
    "tts",
    "stt",
    "audio",
    "embed",
    "ocr",
    "depth",
}


class EchoAdapter:
    """What a plugin's adapter looks like to the runner."""

    def __init__(self, ctx: AdapterContext):
        self.ctx = ctx


def echo_plugin(tmp_path: Path, **overrides) -> Plugin:
    recipes_dir = tmp_path / "echo-recipes"
    recipes_dir.mkdir(exist_ok=True)
    (recipes_dir / "echo-small.yaml").write_text(
        "name: echo-small\nmodalities: [echo]\nengine: echo-engine\nvram: {gb: 0.5}\n"
    )
    fields = {
        "name": "giq-echo",
        "api_version": API_VERSION,
        "version": "1.0",
        "engines": (Engine("echo-engine"),),
        "modalities": (Modality("echo", label="Echo", icon="waveform", lane_width=3),),
        "adapters": {("echo-engine", "echo"): EchoAdapter},
        "recipes": recipes_dir,
    }
    return Plugin(**{**fields, **overrides})


@pytest.fixture
def installed(monkeypatch):
    """Install plugins for one test: call with (name, dist, plugin or error) rows."""

    def install(*rows):
        monkeypatch.setattr(plugins, "_installed", lambda: list(rows))
        plugins.reset()
        recipes.builtin.cache_clear()
        reload_registry()

    yield install
    monkeypatch.undo()
    plugins.reset()
    recipes.builtin.cache_clear()
    reload_registry()


def test_the_builtins_register_through_the_contract():
    assert set(plugins.modalities()) == BUILTIN_MODALITIES
    assert {"llama.cpp", "vllm", "sd.cpp", "transformers"} <= set(plugins.engines())
    assert plugins.engines_for("llm") == {"llama.cpp", "vllm"}
    assert plugins.engines_for("ocr") == {"transformers"}
    assert plugins.engine("sdcpp").name == "sd.cpp", "old spellings resolve"
    loaded = {s.name for s in plugins.status() if s.loaded}
    assert {"giq", "giq-vllm", "giq-sdcpp", "giq-speech", "giq-ocr", "giq-depth"} <= loaded


def test_a_plugins_modality_is_served_end_to_end(installed, tmp_path):
    installed(("echo", "giq-echo", echo_plugin(tmp_path)))

    recipe = get_recipe("echo-small")
    assert recipe is not None and recipe.lanes == 3, "its recipe loads, with its lane width"
    assert JobRequest(modality="echo", model="echo-small").modality == "echo"

    from giq.queue import JobQueue
    from giq.runner import Runner

    adapter = Runner(JobQueue())._build_worker("echo-small")
    assert isinstance(adapter, EchoAdapter) and adapter.ctx.recipe == "echo-small"
    status = next(s for s in plugins.status() if s.name == "giq-echo")
    assert status.loaded and status.source == "giq-echo" and status.modalities == ("echo",)


def test_an_unknown_modality_is_refused_at_the_door():
    with pytest.raises(ValueError, match="unknown modality 'echo'"):
        JobRequest(modality="echo", model="x")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"api_version": API_VERSION + 1}, "plugin API"),
        ({"engines": (Engine("llama.cpp"),)}, "engine name 'llama.cpp' is already registered"),
        ({"engines": (Engine("echo-engine", aliases=("sdcpp",)),)}, "'sdcpp' is already"),
        ({"modalities": (Modality("ocr"),)}, "modality 'ocr' is already registered"),
        (
            {"adapters": {("echo-engine", "nothing"): EchoAdapter}},
            "names modality 'nothing'",
        ),
        ({"adapters": {("llama.cpp", "llm"): EchoAdapter}}, "already registered"),
    ],
)
def test_a_plugin_that_cannot_join_is_refused_whole(installed, tmp_path, overrides, reason):
    installed(("echo", "giq-echo", echo_plugin(tmp_path, **overrides)))

    status = next(s for s in plugins.status() if s.name == "giq-echo")
    assert not status.loaded and reason in (status.reason or "")
    assert "echo" not in plugins.modalities(), "nothing of it was registered"
    assert get_recipe("echo-small") is None, "nor its recipes"
    assert set(plugins.modalities()) == BUILTIN_MODALITIES, "the others still stand"


def test_an_entry_point_that_does_not_load_is_reported(installed):
    installed(
        ("broken", "giq-broken", ImportError("no module named torch_but_newer")),
        ("odd", "giq-odd", object()),
    )
    by_name = {s.name: s for s in plugins.status()}
    assert not by_name["broken"].loaded and "does not load" in by_name["broken"].reason
    assert not by_name["odd"].loaded and "not a giq.plugin.Plugin" in by_name["odd"].reason
    assert set(plugins.modalities()) == BUILTIN_MODALITIES


def test_a_plugin_recipe_cannot_take_a_builtin_name(installed, tmp_path):
    plugin = echo_plugin(tmp_path)
    (plugin.recipes / "gemma.yaml").write_text(
        "name: gemma-4-12b\nmodalities: [echo]\nengine: echo-engine\nvram: {gb: 1.0}\n"
    )
    installed(("echo", "giq-echo", plugin))
    assert get_recipe("gemma-4-12b").engine == "llama.cpp", "the built-in keeps its name"
    assert get_recipe("echo-small") is not None, "the plugin's other recipes still load"
