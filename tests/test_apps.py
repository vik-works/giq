# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Apps (ADR-006): a name and a key, so usage can be counted per app.

The two promises under test: an app sends nothing a plain OpenAI client
would not (its key, where keys go), and a caller without one is still served,
counted as anonymous, so nothing that works today breaks.
"""

import os
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from giq import apps
from giq.api import access
from giq.apps import ANONYMOUS, AppError, AppStore
from giq.config import get_config
from giq.queue import Job


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh apps file, installed as the one the middleware asks."""
    s = AppStore(tmp_path / "apps.yaml")
    monkeypatch.setattr(apps, "_store", s)
    return s


def probe_app() -> FastAPI:
    """The access rules around one route that reports the request's app:
    what every real route and the orchestrator see."""
    app = FastAPI()

    @app.get("/who")
    async def who() -> dict:
        return {"app": apps.current_app.get()}

    access.install(app)
    return app


async def who(headers=None, params=None):
    async with AsyncClient(
        transport=ASGITransport(app=probe_app()), base_url="http://localhost:8084"
    ) as c:
        return await c.get("/who", headers=headers or {}, params=params or {})


# --- the store --------------------------------------------------------------


def test_a_key_is_shown_once_and_only_its_hash_is_kept(store):
    app, key = store.create("kompost", "asset pipeline")

    assert key.startswith("giq-") and len(key) > 40
    text = store.path.read_text()
    assert key not in text
    assert apps.hash_key(key) in text
    assert oct(os.stat(store.path).st_mode & 0o777) == "0o600"
    assert store.resolve(key) == "kompost"
    assert app.public() == {
        "name": "kompost",
        "description": "asset pipeline",
        "created": app.created,
        "revoked": None,
    }


@pytest.mark.parametrize("name", ["anonymous", "unknown", "Upper", "-dash", "a b", "x" * 64])
def test_reserved_and_malformed_names_are_refused(store, name):
    with pytest.raises(AppError):
        store.create(name)


def test_a_name_stays_taken_after_revoke(store):
    store.create("bot")
    store.revoke("bot")
    with pytest.raises(AppError):
        store.create("bot")


def test_a_revoked_key_no_longer_resolves(store):
    _, key = store.create("bot")
    store.revoke("bot")
    assert store.resolve(key) is None


def test_a_hand_edit_is_read_without_a_restart(store):
    _, key = store.create("bot")
    assert store.resolve(key) == "bot"

    text = store.path.read_text().replace("bot:", "renamed:")
    store.path.write_text(text)
    # mtime resolution can be coarse; make the change visible
    os.utime(store.path, (time.time() + 5, time.time() + 5))

    assert store.resolve(key) == "renamed"


def test_a_broken_file_keeps_the_previous_list(store):
    _, key = store.create("bot")
    assert store.resolve(key) == "bot"

    store.path.write_text("apps: [unclosed\n")
    os.utime(store.path, (time.time() + 5, time.time() + 5))

    assert store.resolve(key) == "bot"


def test_no_file_means_no_apps(store):
    assert store.entries() == []
    assert store.resolve("giq-whatever") is None


# --- the request ------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_key_is_served_as_anonymous(store):
    r = await who()
    assert r.status_code == 200
    assert r.json() == {"app": ANONYMOUS}
    assert r.headers["x-giq-app"] == ANONYMOUS


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["bearer", "header", "query"])
async def test_the_key_is_read_where_keys_already_go(store, where):
    _, key = store.create("kompost")
    headers, params = {}, {}
    if where == "bearer":
        headers["authorization"] = f"Bearer {key}"  # what the OpenAI SDKs send
    elif where == "header":
        headers["x-giq-token"] = key
    else:
        params["token"] = key  # firmware that cannot set a header

    r = await who(headers, params)

    assert r.json() == {"app": "kompost"}
    assert r.headers["x-giq-app"] == "kompost"


@pytest.mark.asyncio
async def test_a_mistyped_key_is_served_but_says_so(store):
    store.create("kompost")
    r = await who({"authorization": "Bearer giq-not-the-key"})
    assert r.status_code == 200
    assert r.headers["x-giq-app"] == ANONYMOUS


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(get_config().access, "token", "s3cret", raising=False)
    return "s3cret"


@pytest.mark.asyncio
async def test_with_a_token_an_app_key_admits_on_its_own(store, token):
    _, key = store.create("kompost")
    r = await who({"authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert r.json() == {"app": "kompost"}


@pytest.mark.asyncio
async def test_with_a_token_the_shared_secret_counts_as_anonymous(store, token):
    r = await who({"authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == {"app": ANONYMOUS}


@pytest.mark.asyncio
async def test_with_a_token_unknown_and_revoked_keys_are_refused(store, token):
    _, key = store.create("kompost")
    store.revoke("kompost")
    for bad in (key, "giq-not-the-key"):
        r = await who({"authorization": f"Bearer {bad}"})
        assert r.status_code == 401


# --- the job ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_job_stamps_the_requests_app(monkeypatch):
    """The orchestrator takes the app from the request's context, so every
    route is attributed without knowing apps exist."""
    from giq.models import JobRequest
    from giq.services.orchestration import Orchestrator

    added: list[Job] = []

    class FakeQueue:
        async def add(self, job):
            added.append(job)
            return 0

    orch = Orchestrator.__new__(Orchestrator)
    orch.queue = FakeQueue()
    monkeypatch.setattr(Orchestrator, "_reject_if_paused", staticmethod(lambda: None))
    monkeypatch.setattr(Orchestrator, "_reject_if_disabled", staticmethod(lambda r: None))
    request = JobRequest(modality="llm", model="not-a-recipe", tasks=[{"id": "t", "user": "hi"}])

    token = apps.current_app.set("kompost")
    try:
        await orch.submit_job(request)
    finally:
        apps.current_app.reset(token)
    await orch.submit_job(request)

    assert [j.app for j in added] == ["kompost", ANONYMOUS]


# --- the stats --------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_by_app_keeps_history_from_before_apps_apart(tmp_path, monkeypatch):
    """A job recorded before apps existed is unknown, not anonymous: it was
    never attributed, which is a different statement."""
    import giq.stats as stats_mod
    from giq.api.stats_api import stats_usage
    from giq.stats import StatsRecorder

    monkeypatch.setattr(stats_mod, "_recorder", StatsRecorder(db_path=tmp_path / "a.db"))
    rec = stats_mod.get_stats()
    now = time.time()
    for job_id, app in (("j1", "kompost"), ("j2", "kompost"), ("j3", ANONYMOUS), ("j4", None)):
        rec._record_job_sync(
            (now, job_id, "llm", "gemma-4-12b", "completed", 5, 100, 1, None, 10, 2, None, app)
        )

    d = await stats_usage(app=None, period="week", gpu=None, since=None, until=None)
    only = await stats_usage(app="kompost", period="week", gpu=None, since=None, until=None)
    unknown = await stats_usage(app="unknown", period="week", gpu=None, since=None, until=None)
    rec.close()

    by_app = {row["app"]: row for row in d["apps"]}
    assert by_app["kompost"]["jobs"] == 2
    assert by_app["kompost"]["run_ms"] == 200
    assert by_app[ANONYMOUS]["jobs"] == 1
    assert by_app[None]["jobs"] == 1
    assert only["totals"]["jobs"] == 2
    assert unknown["totals"]["jobs"] == 1
