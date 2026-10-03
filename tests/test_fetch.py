# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Fetching a recipe's weights through the service (ADR-005 D4).

The child that talks to the Hub is replaced by a few lines of Python that
write, sleep or fail: what is under test is the queue, progress, cancelling
and the refusals around it.
"""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from giq import fetch
from giq.availability import Check
from giq.plan import Plan, Transfer
from giq.registry import reload_registry

DEPTH = (
    "name: {name}\nmodalities: [depth]\nengine: transformers\nweights: {weights}\nvram: {{gb: 1}}\n"
)


@pytest.fixture
def models(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "shared").mkdir(parents=True)
    (models / "shared" / "config.json").write_text("{}")
    (models / "own").mkdir()
    (models / "own" / "config.json").write_text("{}")
    operator = tmp_path / "recipes"
    operator.mkdir()
    for name, weights in (
        ("d-gone", "{path: gone, source: 'hf:org/gone'}"),
        ("d-one", "{path: own, source: 'hf:org/own'}"),
        ("d-two", "{path: shared, source: 'hf:org/shared'}"),
        ("d-three", "{path: shared, source: 'hf:org/shared'}"),
    ):
        (operator / f"{name}.yaml").write_text(DEPTH.format(name=name, weights=weights))
    monkeypatch.setenv("GIQ_MODELS_DIR", str(models))
    monkeypatch.setenv("GIQ_DEPTH_MODELS_DIR", str(models))
    monkeypatch.setenv("GIQ_RECIPES_DIR", str(operator))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr("giq.gpus.get_gpus", lambda *a, **k: [])
    reload_registry()
    yield models
    monkeypatch.undo()
    reload_registry()


def _child(monkeypatch, script: str) -> None:
    """Every transfer runs ``script`` with DEST set to its destination."""

    def command(move: Transfer, report: bool = False) -> list[str]:
        return [sys.executable, "-c", f"DEST = {str(move.dest)!r}\n{script}"]

    monkeypatch.setattr(fetch, "child_command", command)


def _plan(models: Path, name: str = "d-gone") -> Plan:
    move = Transfer(None, "org/gone", None, None, models / "gone", files=[("w", 10)])
    return Plan(name, "fetchable", [], [move], service_can_fetch=True)


async def _settled(download: fetch.Download) -> fetch.Download:
    for _ in range(200):
        if not download.active:
            return download
        await asyncio.sleep(0.05)
    raise AssertionError(f"still {download.state}")


@pytest.mark.asyncio
async def test_a_fetch_runs_its_transfers_and_the_recipe_becomes_ready(models, monkeypatch):
    _child(
        monkeypatch,
        "import os\nos.makedirs(DEST)\nopen(os.path.join(DEST, 'w'), 'w').write('x' * 10)",
    )
    downloads = fetch.Downloads()
    download = await _settled(downloads.start(_plan(models)))
    assert download.state == "done", download.error
    assert download.bytes_done == download.bytes_total == 10
    assert (models / "gone" / "w").exists()
    assert downloads.start(_plan(models)) is not download, "a finished fetch is not reused"


@pytest.mark.asyncio
async def test_asking_twice_joins_the_running_fetch(models, monkeypatch):
    _child(monkeypatch, "import time\ntime.sleep(5)")
    downloads = fetch.Downloads()
    first = downloads.start(_plan(models))
    assert downloads.start(_plan(models)) is first
    downloads.cancel(first.id)
    await _settled(first)


@pytest.mark.asyncio
async def test_a_cancelled_fetch_ends_its_child(models, monkeypatch):
    _child(monkeypatch, "import time\ntime.sleep(30)")
    downloads = fetch.Downloads()
    download = downloads.start(_plan(models))
    while download.state != "running" or download._proc is None:
        await asyncio.sleep(0.02)
    downloads.cancel(download.id)
    assert (await _settled(download)).state == "cancelled"


@pytest.mark.asyncio
async def test_a_failing_child_says_why(models, monkeypatch):
    _child(
        monkeypatch,
        "import json, sys\nprint(json.dumps({'error': 'GatedRepoError: no access'}), "
        "file=sys.stderr)\nsys.exit(1)",
    )
    download = await _settled(fetch.Downloads().start(_plan(models)))
    assert download.state == "failed" and "GatedRepoError: no access" in (download.error or "")


@pytest.mark.asyncio
async def test_a_child_that_reports_success_without_the_files_fails(models, monkeypatch):
    _child(monkeypatch, "pass")
    download = await _settled(fetch.Downloads().start(_plan(models)))
    assert download.state == "failed" and "not every file" in (download.error or "")


# --- the routes ---------------------------------------------------------------


@pytest.fixture
async def client(models, monkeypatch):
    from giq.api.recipes_api import router

    monkeypatch.setattr(fetch, "_downloads", fetch.Downloads())
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        yield ac


def _planned(monkeypatch, result: Plan) -> None:
    monkeypatch.setattr("giq.plan.plan", lambda name: result)


@pytest.mark.asyncio
async def test_a_recipe_already_here_is_not_fetched(client, monkeypatch):
    _planned(monkeypatch, Plan("d-one", "ready", [], [], True))
    r = await client.post("/recipes/d-one/fetch")
    assert r.status_code == 409 and "already" in r.json()["detail"]


@pytest.mark.asyncio
async def test_a_plan_that_fails_refuses_the_fetch(client, monkeypatch, models):
    failing = Check("access", "fail", "org/gone is gated")
    _planned(monkeypatch, Plan("d-gone", "fetchable", [failing], _plan(models).transfers, True))
    r = await client.post("/recipes/d-gone/fetch")
    assert r.status_code == 409
    assert r.json()["detail"]["checks"] == [failing.to_dict()]


@pytest.mark.asyncio
async def test_a_read_only_model_store_hands_the_operator_the_command(client, monkeypatch, models):
    _planned(monkeypatch, Plan("d-gone", "fetchable", [], _plan(models).transfers, False))
    r = await client.post("/recipes/d-gone/fetch")
    assert r.status_code == 409 and r.json()["detail"]["command"] == "giq add d-gone"


@pytest.mark.asyncio
async def test_a_fetch_is_started_and_listed(client, monkeypatch, models):
    _child(monkeypatch, "import time\ntime.sleep(30)")
    _planned(monkeypatch, _plan(models))
    r = await client.post("/recipes/d-gone/fetch")
    assert r.status_code == 202 and r.json()["recipe"] == "d-gone"
    listed = (await client.get("/downloads")).json()["downloads"]
    assert [d["id"] for d in listed] == [r.json()["id"]]
    cancelled = await client.delete(f"/downloads/{r.json()['id']}")
    assert cancelled.status_code == 200
    await _settled(fetch.get_downloads().get(r.json()["id"]))
    assert (await client.get("/downloads/nope")).status_code == 404


@pytest.mark.asyncio
async def test_removing_a_recipe_keeps_what_another_recipe_loads(client, monkeypatch, models):
    from giq import runner

    monkeypatch.setattr(runner, "get_runner", lambda: type("R", (), {"loaded_keys": set})())
    monkeypatch.setattr("giq.api.recipes_api.get_runner", runner.get_runner)
    own = (await client.delete("/recipes/d-one/weights")).json()
    assert own["deleted"] == [str(models / "own")] and not (models / "own").exists()
    shared = (await client.delete("/recipes/d-two/weights")).json()
    assert shared["deleted"] == [] and shared["kept"][0]["used_by"] == ["d-three"]
    assert (models / "shared").exists()


@pytest.mark.asyncio
async def test_progress_is_what_the_child_reports(models, monkeypatch):
    _child(
        monkeypatch,
        "import time\nprint('{\"bytes\": 7}', flush=True)\n"
        "print('noise', flush=True)\ntime.sleep(30)",
    )
    downloads = fetch.Downloads()
    download = downloads.start(_plan(models))
    for _ in range(100):
        if download.bytes_done == 7:
            break
        await asyncio.sleep(0.05)
    assert download.bytes_done == 7, "a line that is not progress is ignored"
    downloads.cancel(download.id)
    await _settled(download)


def test_a_fetch_may_download_where_the_service_is_offline(monkeypatch):
    """The unit sets HF_HUB_OFFLINE for the model children; a fetch is asked for."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert "HF_HUB_OFFLINE" not in fetch._child_env()
