# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The plugin registry (ADR-004): built-ins through the same contract a
third party uses, a plugin's modality served end to end, and refusals that
leave everything else standing."""

from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI

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


# --- routes (ADR-004 D5: plugins add routes, never take one over) -------------

echo_router = APIRouter()


@echo_router.get("/echo")
async def echo() -> dict:
    return {"echo": True}


squatter = APIRouter()


@squatter.get("/status")
async def not_the_status() -> dict:
    return {}


def served(app: FastAPI) -> dict[str, dict]:
    """Path -> its operations, from what the app actually routes."""
    return app.openapi()["paths"]


def core_app() -> FastAPI:
    app = FastAPI()
    plugins.mount(app)
    return app


def test_a_plugins_routes_are_mounted_after_cores(installed, tmp_path):
    installed(("echo", "giq-echo", echo_plugin(tmp_path, routers=(f"{__name__}:echo_router",))))
    paths = list(served(core_app()))
    assert "/echo" in paths and paths.index("/status") < paths.index("/echo")
    assert "echo" in plugins.modalities(), "mounting refused nothing"


@pytest.mark.parametrize(
    ("router", "reason"),
    [
        (f"{__name__}:squatter", "would take over routes already served: GET /status"),
        ("giq_no_such_module:router", "its routes do not import"),
    ],
)
def test_a_plugin_whose_routes_cannot_join_is_refused_whole(installed, tmp_path, router, reason):
    installed(("echo", "giq-echo", echo_plugin(tmp_path, routers=(router,))))
    app = core_app()

    status = next(s for s in plugins.status() if s.name == "giq-echo")
    assert not status.loaded and reason in (status.reason or "")
    assert "echo" not in plugins.modalities(), "its modality went with its routes"
    assert "not_the_status" not in served(app)["/status"]["get"]["operationId"]


# --- what the HTTP API says about plugins ------------------------------------


class FakeOrchestrator:
    """Completes every job at once with one canned result."""

    def __init__(self, result: dict):
        self.result = result
        self.submitted: list[JobRequest] = []

    async def submit_job(self, request: JobRequest) -> tuple[str, int]:
        self.submitted.append(request)
        return "j1", 0

    async def wait_for_job(self, job_id: str, timeout: float | None = None):
        from giq.queue import Job

        job = Job(job_id=job_id, request=self.submitted[-1])
        job.results = [self.result]
        return job


@pytest.fixture
def api():
    """The full app, without its lifespan, with a fake orchestrator."""
    from fastapi.testclient import TestClient

    from giq.api.dependencies import get_orchestrator
    from giq.main import app

    orch = FakeOrchestrator({"embedding": [1.0], "dim": 192})
    app.dependency_overrides[get_orchestrator] = lambda: orch
    try:
        yield TestClient(app, base_url="http://localhost"), orch
    finally:
        app.dependency_overrides.pop(get_orchestrator, None)


def test_status_lists_the_plugins(api):
    import giq.queue
    import giq.runner
    from giq.queue import JobQueue
    from giq.runner import Runner

    giq.queue._queue = JobQueue()
    giq.runner._runner = Runner(giq.queue._queue)
    client, _ = api
    listed = {p["name"]: p for p in client.get("/status").json()["plugins"]}
    assert listed["giq"]["loaded"] and listed["giq"]["source"] == "builtin"
    assert "llm" in listed["giq"]["modalities"]
    assert listed["giq-ocr"]["engines"] == []


def test_a_smoke_test_runs_its_modalitys_canned_job(api):
    client, orch = api
    r = client.post("/test/embed", params={"recipe": "ecapa-tdnn"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["recipe"] == "ecapa-tdnn" and body["job_id"] == "j1"
    assert body["dim"] == 192, "the modality's own summary of the result"
    sent = orch.submitted[-1]
    assert sent.modality == "embed" and sent.model == "ecapa-tdnn" and sent.tasks


def test_a_smoke_test_that_evicts_must_be_confirmed(api):
    client, orch = api
    r = client.post("/test/text2image", params={"recipe": "x"})
    assert r.status_code == 400 and "confirm=true" in r.json()["detail"]
    assert not orch.submitted


def test_a_modality_without_a_smoke_test_is_a_404(api):
    client, _ = api
    r = client.post("/test/tts")
    assert r.status_code == 404 and "embed" in r.json()["detail"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [("whisper-1", "whisper-large-v3"), ("whisper-large-v3", "whisper-large-v3")],
)
def test_the_audio_routes_run_the_recipe_a_request_names(model, expected):
    from giq.api.audio_api import _recipe_for

    assert _recipe_for(model, "audio", "whisper-large-v3") == expected


def test_a_recipe_that_does_not_serve_the_route_is_a_400():
    from fastapi import HTTPException

    from giq.api.audio_api import _recipe_for

    with pytest.raises(HTTPException, match="does not serve tts"):
        _recipe_for("gemma-4-12b", "tts", "kokoro")


# --- the curated index (ADR-005 D5) ------------------------------------------


def test_the_index_matches_what_the_curated_plugins_register():
    """The index is written by hand; it must say what the plugins do."""
    registered = {s.name: s for s in plugins.status()}
    for entry in plugins.curated():
        state = registered[entry["name"]]
        assert set(entry["engines"]) <= set(state.engines), entry["name"]
        assert set(entry["modalities"]) == set(state.modalities), entry["name"]


def test_every_builtin_recipe_is_brought_by_exactly_one_curated_plugin():
    listed = [name for entry in plugins.curated() for name in entry["recipes"]]
    builtin = {recipe.name for recipe, _ in recipes.builtin().values()}
    assert sorted(listed) == sorted(builtin), "each recipe once, and none missing"
    for entry in plugins.curated():
        modalities = set(entry["modalities"])
        for name in entry["recipes"]:
            recipe = get_recipe(name)
            assert recipe is not None
            assert recipe.engine in entry["engines"] or modalities & set(recipe.modalities), (
                f"{name} is not {entry['name']}'s"
            )


def test_a_curated_plugin_not_installed_carries_its_install_command(monkeypatch):
    present = [s for s in plugins.status() if s.name != "giq-depth"]
    monkeypatch.setattr(plugins, "status", lambda: present)
    depth = next(e for e in plugins.catalog() if e["name"] == "giq-depth")
    assert not depth["installed"] and depth["status"] is None
    assert depth["install"].startswith("uv pip install --python ")
    assert depth["install"].endswith(" giq-depth")
    core = next(e for e in plugins.catalog() if e["name"] == "giq")
    assert core["installed"] and core["install"] is None
