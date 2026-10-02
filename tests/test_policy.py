# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Per-model residency policy: pinned / auto / off.

The invariant worth protecting: `off` means no load path at all — not merely
"no new submissions" — and unpinning actually unloads instead of being undone
by the residents loop on its next tick.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from giq.models import JobRequest, JobStatus, Modality
from giq.policy import AUTO, OFF, PINNED, reset_policy_store
from giq.queue import Job, JobQueue
from giq.runner import RESIDENT, Instance, ResidentDemoted, Runner

RESIDENTS = [
    "gemma-4-12b",
    "whisper-large-v3",
    "ecapa-tdnn",
]


@pytest.fixture
def store():
    """A policy store with a clean DB-backed override table."""
    from giq.stats import get_stats

    stats = get_stats()
    for name in list(stats.load_policies()):
        stats.delete_policy(name)
    store = reset_policy_store()
    store.load()
    yield store
    for name in list(stats.load_policies()):
        stats.delete_policy(name)
    reset_policy_store()


@pytest.fixture
def sixteen_gb_card(monkeypatch):
    """The card size forced rather than read from the host, so "overcommits"
    means the same thing on every machine."""
    from giq.vram import VRAMStatus

    monkeypatch.setattr(
        "giq.vram.get_vram_status",
        lambda *args, **kwargs: VRAMStatus(used_gb=0.0, total_gb=16.0, free_gb=16.0),
    )


@pytest.fixture
def queue():
    return JobQueue()


def _prime_resident(runner: Runner, key, width: int = 1):
    worker = AsyncMock()
    worker.is_ready = True
    worker.estimated_vram_gb = 4.0
    # On the card this model actually binds to — a resident sitting on the
    # wrong card is torn down and reloaded, which is not what these tests are
    # about (see test_binding for that path).
    res = Instance(worker, key, runner._device_for(key), residency=RESIDENT, width=width)
    runner._residents[key] = res
    return res


# --- defaults ---------------------------------------------------------------


def test_registry_residents_default_to_pinned(store):
    assert store.policy_for("gemma-4-12b") == PINNED
    assert store.policy_for("whisper-large-v3") == PINNED


def test_everything_else_defaults_to_auto(store):
    assert store.policy_for("flux_klein") == AUTO
    assert store.policy_for("kokoro") == AUTO


def test_unregistered_model_is_auto_not_off(store):
    """An unknown model must not be silently blocked."""
    assert store.policy_for("no-such-model") == AUTO


# --- persistence ------------------------------------------------------------


def test_override_round_trips_through_the_db(store):
    store.set("flux_klein", PINNED, reason="demo")
    reloaded = reset_policy_store()
    reloaded.load()
    record = reloaded.record_for("flux_klein")
    assert record.policy == PINNED
    assert record.source == "override"
    assert record.reason == "demo"


def test_setting_a_model_back_to_its_default_clears_the_override(store):
    """Storing "same as default" would freeze a later default change in place."""
    store.set("gemma-4-12b", AUTO)
    assert store.record_for("gemma-4-12b").source == "override"
    store.set("gemma-4-12b", PINNED)
    assert store.record_for("gemma-4-12b").source == "default"

    from giq.stats import get_stats

    assert "gemma-4-12b" not in get_stats().load_policies()


def test_clear_reverts_to_default(store):
    store.set("gemma-4-12b", OFF)
    store.clear("gemma-4-12b")
    assert store.policy_for("gemma-4-12b") == PINNED


def test_unknown_model_cannot_be_set(store):
    with pytest.raises(ValueError, match="not a recipe"):
        store.set("no-such-model", PINNED)


def test_invalid_policy_is_rejected(store):
    with pytest.raises(ValueError, match="unknown policy"):
        store.set("gemma-4-12b", "sometimes")


def test_stale_override_for_a_delisted_model_is_dropped_on_load(store):
    """A model removed from the registry must not resurrect as a resident."""
    from giq.stats import get_stats

    get_stats().save_policy("model-that-was-deleted", PINNED, None)
    fresh = reset_policy_store()
    fresh.load()
    assert "model-that-was-deleted" not in fresh.residents()


# --- the resident set -------------------------------------------------------


def test_pinning_adds_to_the_resident_set(store):
    assert "kokoro" not in store.residents()
    store.set("kokoro", PINNED)
    assert "kokoro" in store.residents()


def test_unpinning_removes_from_the_resident_set(store):
    store.set("whisper-large-v3", AUTO)
    assert "whisper-large-v3" not in store.residents()


def test_registry_residents_keep_their_priority_order(store):
    """Declared order encodes which model matters most when VRAM is tight."""
    store.set("kokoro", PINNED)
    residents = store.residents()
    assert residents[:3] == [
        "gemma-4-12b",
        "whisper-large-v3",
        "ecapa-tdnn",
    ]
    assert residents[3] == "kokoro"  # operator pins follow


def test_off_is_not_resident(store):
    store.set("gemma-4-12b", OFF)
    assert "gemma-4-12b" not in store.residents()
    assert store.is_off("gemma-4-12b")


# --- the VRAM budget --------------------------------------------------------


def test_default_pinned_set_fits(store):
    fits, projected, total, _device = store.pinned_fit()
    assert fits, f"the shipped resident set must fit: {projected} of {total}"


def test_pinning_a_huge_model_overcommits(store, sixteen_gb_card):
    fits, projected, total, _device = store.pinned_fit(extra="zimage")
    assert not fits
    assert projected > total


def test_only_one_llm_can_be_pinned_per_card(store):
    """Two llama-servers on one card would fight over its internal port."""
    assert store.resident_llm() == "gemma-4-12b"
    assert store.resident_llm(exclude="gemma-4-12b") is None
    # Scoped to the card gemma actually lands on...
    where = store.effective_device("gemma-4-12b")
    assert store.resident_llm(device=where) == "gemma-4-12b"
    # ...and silent about any other card, which is what lets a second LLM be
    # pinned elsewhere.
    assert store.resident_llm(device="GPU-nothing-here") is None


# --- enforcement: submission ------------------------------------------------


@pytest.mark.asyncio
async def test_submit_is_refused_for_a_disabled_model(store, monkeypatch):
    from giq.services.orchestration import Orchestrator

    store.set("gemma-4-12b", OFF, reason="freeing the card")
    orch = Orchestrator()
    request = JobRequest(modality=Modality.llm, model="gemma-4-12b", tasks=[{"id": "t1"}])
    with pytest.raises(HTTPException) as excinfo:
        await orch.submit_job(request)
    assert excinfo.value.status_code == 503
    assert "switched off" in excinfo.value.detail
    assert "freeing the card" in excinfo.value.detail
    assert excinfo.value.headers["X-Giq-Model-Policy"] == "off"


@pytest.mark.asyncio
async def test_submit_is_allowed_once_re_enabled(store):
    from giq.services.orchestration import Orchestrator

    store.set("flux_klein", OFF)
    store.set("flux_klein", AUTO)
    orch = Orchestrator()
    job_id, _pos = await orch.submit_job(
        JobRequest(modality=Modality.text2image, model="flux_klein", tasks=[{"id": "t1"}])
    )
    assert job_id
    await orch.queue.remove(job_id)


# --- enforcement: the load paths -------------------------------------------


@pytest.mark.asyncio
async def test_disabled_model_cannot_load_via_the_batch_path(store, queue):
    """Submission is the usual gate, but `off` must block loading outright."""
    runner = Runner(queue, use_policy=True)
    store.set("flux_klein", OFF)
    with pytest.raises(RuntimeError, match="disabled"):
        await runner._ensure_worker("flux_klein")


@pytest.mark.asyncio
async def test_disabled_resident_is_not_reloaded(store, queue):
    runner = Runner(queue, use_policy=True)
    store.set("gemma-4-12b", OFF)
    await runner._load_resident("gemma-4-12b")
    assert "gemma-4-12b" not in runner._residents


@pytest.mark.asyncio
async def test_runner_resident_set_follows_policy_live(store, queue):
    runner = Runner(queue, use_policy=True)
    assert "whisper-large-v3" in runner._resident_keys
    store.set("whisper-large-v3", AUTO)
    assert "whisper-large-v3" not in runner._resident_keys


@pytest.mark.asyncio
async def test_static_residents_ignore_pinning_but_not_off(store, queue):
    """use_policy governs the resident set only.

    `off` is a hard statement that a model must not load; it would be a trap
    for that to depend on how the runner was constructed. Caught in live
    verification: /llm/endpoint reported "loading" for a switched-off model
    because that runner had use_policy=False.
    """
    runner = Runner(queue, residents=RESIDENTS)
    store.set("whisper-large-v3", OFF)
    # the static list still drives residency...
    assert "whisper-large-v3" in runner._resident_keys
    # ...but off is honoured everywhere
    assert runner.is_disabled("whisper-large-v3")
    with pytest.raises(RuntimeError, match="disabled"):
        await runner._ensure_worker("whisper-large-v3")


# --- unpinning actually unloads ---------------------------------------------


@pytest.mark.asyncio
async def test_unpinning_unloads_the_resident(store, queue):
    """The point of the whole design: the loop must not reload it right back."""
    runner = Runner(queue, use_policy=True)
    key = "whisper-large-v3"
    res = _prime_resident(runner, key)

    store.set("whisper-large-v3", AUTO)
    await runner._release_demoted_residents()

    res.adapter.stop.assert_awaited()
    assert key not in runner._residents


@pytest.mark.asyncio
async def test_pinned_residents_are_left_alone(store, queue):
    runner = Runner(queue, use_policy=True)
    key = "whisper-large-v3"
    res = _prime_resident(runner, key)

    await runner._release_demoted_residents()

    res.adapter.stop.assert_not_awaited()
    assert key in runner._residents


@pytest.mark.asyncio
async def test_unload_waits_for_in_flight_lane_jobs(store, queue):
    """Teardown drains the lane, so nothing dies mid-generation."""
    runner = Runner(queue, use_policy=True)
    key = "whisper-large-v3"
    res = _prime_resident(runner, key)
    await res.lane.acquire()  # simulate a job in flight
    res.active_count = 1

    store.set("whisper-large-v3", AUTO)
    task = asyncio.create_task(runner._release_demoted_residents())
    await asyncio.sleep(0.05)
    assert key in runner._residents  # still waiting on the lane
    res.adapter.stop.assert_not_awaited()

    res.active_count = 0
    res.lane.release()
    await asyncio.wait_for(task, timeout=2)
    assert key not in runner._residents


# --- a lane job whose model is unpinned mid-flight --------------------------


@pytest.mark.asyncio
async def test_lane_job_fails_fast_when_unpinned(store, queue):
    """Without this it would wait out RESIDENT_WAIT_TIMEOUT_SECONDS (840s)."""
    runner = Runner(queue, use_policy=True)
    store.set("whisper-large-v3", AUTO)
    with pytest.raises(ResidentDemoted):
        await runner._wait_resident_ready("whisper-large-v3")


@pytest.mark.asyncio
async def test_demoted_lane_job_is_requeued_not_failed(store, queue):
    runner = Runner(queue, use_policy=True)
    key = "whisper-large-v3"
    job = Job(
        job_id="j1",
        request=JobRequest(modality=Modality.audio, model="whisper-large-v3", tasks=[{"id": "t"}]),
    )
    job.status = JobStatus.running
    await queue.add(job)
    store.set("whisper-large-v3", AUTO)

    await runner._process_resident_job(job, key)

    assert job.status == JobStatus.pending  # picked up again by the batch path
    assert job.error is None
    assert job.completed_at is None


# --- the control endpoints --------------------------------------------------


@pytest.fixture
async def client(store, queue):
    from httpx import ASGITransport, AsyncClient

    import giq.queue
    import giq.runner
    from giq.main import app

    giq.queue._queue = queue
    giq.runner._runner = Runner(queue, use_policy=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        yield ac


@pytest.mark.asyncio
async def test_policy_round_trip_over_http(client, store):
    # kokoro is 0.5GB, so it fits alongside the default resident set; pinning
    # a 9GB image model on top of it would (correctly) be refused as overcommit.
    r = await client.put(
        "/recipes/kokoro/residency", json={"policy": "pinned", "reason": "demo day"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    residency = body["recipe"]["residency"]
    assert (residency["policy"], residency["source"], residency["reason"]) == (
        "pinned",
        "override",
        "demo day",
    )
    # The resident set itself, not the per-card budgets: a machine without a
    # GPU (a CI runner) has no cards to list it under.
    assert "kokoro" in (await client.get("/recipes")).json()["pinned"]
    assert all(c["pinned_gb"] <= c["total_gb"] for c in body["cards"])

    r = await client.delete("/recipes/kokoro/residency")
    assert r.status_code == 200
    residency = r.json()["recipe"]["residency"]
    assert (residency["policy"], residency["source"]) == ("auto", "default")
    assert "kokoro" not in (await client.get("/recipes")).json()["pinned"]


@pytest.mark.asyncio
async def test_pinning_an_image_model_overcommits_a_small_card(client, store, monkeypatch):
    """A pinned set that cannot coexist is refused, not retried forever.

    The card size is forced rather than read from the host. This test used to
    assert "on this machine", and stopped meaning anything the day a second,
    bigger card went in — the same ambiguity that made every VRAM figure in
    the service need a device attached to it.
    """
    from giq.vram import VRAMStatus

    monkeypatch.setattr(
        "giq.vram.get_vram_status",
        lambda *args, **kwargs: VRAMStatus(used_gb=0.0, total_gb=16.0, free_gb=16.0),
    )
    r = await client.put("/recipes/flux_klein/residency", json={"policy": "pinned"})
    assert r.status_code == 409
    # gemma 9.5 + whisper 4.0 + ecapa 0.6 + klein 8.0 + 0.5 headroom = 22.6
    assert "22.6GB of 16.0GB" in r.json()["detail"]


@pytest.mark.asyncio
async def test_overcommitting_the_card_is_refused(client, store, sixteen_gb_card):
    """The footgun guard: an unsatisfiable pinned set thrashes reloads forever."""
    r = await client.put("/recipes/zimage/residency", json={"policy": "pinned"})
    assert r.status_code == 409
    assert "force=true" in r.json()["detail"]
    assert store.policy_for("zimage") == AUTO  # nothing was written


@pytest.mark.asyncio
async def test_overcommit_can_be_forced_but_warns(client, store, sixteen_gb_card):
    r = await client.put("/recipes/zimage/residency", json={"policy": "pinned", "force": True})
    assert r.status_code == 200
    assert any("cannot all load" in w for w in r.json()["warnings"])
    assert store.policy_for("zimage") == PINNED


@pytest.mark.asyncio
async def test_pinning_a_second_llm_is_refused(client, store):
    r = await client.put("/recipes/llama-3.2-3b/residency", json={"policy": "pinned"})
    assert r.status_code == 409
    assert "One resident LLM per card" in r.json()["detail"]
    assert store.policy_for("llama-3.2-3b") == AUTO


@pytest.mark.asyncio
async def test_swapping_the_pinned_llm_works(client, store):
    """Unpin then pin: the natural way to change which LLM is resident."""
    assert (
        await client.put("/recipes/gemma-4-12b/residency", json={"policy": "auto"})
    ).status_code == 200
    r = await client.put("/recipes/llama-3.2-3b/residency", json={"policy": "pinned"})
    assert r.status_code == 200
    assert store.resident_llm() == "llama-3.2-3b"


@pytest.mark.asyncio
async def test_catalog_reports_policy(client, store):
    store.set("kokoro", OFF, reason="noisy")
    r = await client.get("/recipes")
    assert r.status_code == 200
    body = r.json()
    entry = next(m for m in body["recipes"] if m["name"] == "kokoro")
    assert (entry["residency"]["policy"], entry["residency"]["reason"]) == ("off", "noisy")
    assert all(c["pinned_fits"] for c in body["cards"])


@pytest.mark.asyncio
async def test_llm_endpoint_reports_disabled(client, store):
    store.set("gemma-4-12b", OFF)
    r = await client.get("/llm/endpoint")
    assert r.status_code == 503
    assert r.json()["detail"]["state"] == "disabled"


# --- the loop has to be there before the first pin ---------------------------


@pytest.mark.asyncio
async def test_residents_loop_runs_when_nothing_is_pinned_at_boot(store, queue, monkeypatch):
    """A pin is useless if the loop that acts on it was never started.

    The loop used to be started only for a non-empty resident set. On a host
    where every registry resident had been set to on-demand, giq booted
    without it: pinning a model afterwards showed it in the resident lanes
    and sent its jobs down the resident lane, where they waited out a load
    nothing would ever do.
    """
    import giq.runner

    for name in RESIDENTS:
        store.set(name, AUTO)
    runner = Runner(queue, use_policy=True)
    assert runner._resident_keys == []

    loaded: list[tuple[Modality, str]] = []
    monkeypatch.setattr(giq.runner, "RESIDENT_TICK_SECONDS", 0.01)
    monkeypatch.setattr(giq.runner, "RESIDENT_RELOAD_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(runner, "_load_resident", AsyncMock(side_effect=loaded.append))

    await runner.start()
    try:
        assert runner._resident_task is not None
        store.set("whisper-large-v3", PINNED)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if loaded:
                break
        assert "whisper-large-v3" in loaded
    finally:
        await runner.stop()


# --- ADR-003: recipes, not (worker, model) pairs ---------------------------------


def test_a_job_is_named_by_its_recipe_whatever_alias_the_client_sent():
    """Policy, the VRAM gate and the runner key on the recipe's own name, so
    the alias is resolved once, at submit."""
    from giq.services.orchestration import Orchestrator

    request = JobRequest(modality=Modality.tts, model="kokoro-82m", tasks=[{"id": "t"}])
    Orchestrator._resolve_recipe(request)
    assert request.model == "kokoro"


def test_a_modality_the_recipe_does_not_serve_is_refused_at_submit():
    from giq.services.orchestration import Orchestrator

    request = JobRequest(modality=Modality.image_edit, model="zimage", tasks=[{"id": "t"}])
    with pytest.raises(HTTPException) as exc:
        Orchestrator._resolve_recipe(request)
    assert exc.value.status_code == 400 and "does not serve image_edit" in exc.value.detail
    # flux_klein serves both image modalities from one recipe.
    edit = JobRequest(modality=Modality.image_edit, model="flux_klein", tasks=[{"id": "t"}])
    Orchestrator._resolve_recipe(edit)


def test_old_policy_rows_become_recipe_rows_once(tmp_path):
    """Rows keyed (worker, model) move to the recipe name; the two flux_klein
    rows are one recipe now, and the newer intent wins while a binding
    survives from whichever row had one."""
    import sqlite3

    from giq.stats import StatsRecorder

    db = tmp_path / "stats.db"
    old = sqlite3.connect(db)
    old.execute(
        "CREATE TABLE model_policy (worker TEXT NOT NULL, model TEXT NOT NULL, policy TEXT "
        "NOT NULL, reason TEXT, updated_at REAL NOT NULL, device TEXT, "
        "PRIMARY KEY (worker, model))"
    )
    old.executemany(
        "INSERT INTO model_policy VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("text2image", "flux_klein", "pinned", "renders", 1.0, "GPU-small"),
            ("image_edit", "flux_klein", "off", "edits are broken", 2.0, None),
            ("llm", "gemma-4-12b", "auto", None, 3.0, None),
        ],
    )
    old.commit()
    old.close()

    stats = StatsRecorder(db)
    policies = stats.load_policies()
    assert policies["flux_klein"] == ("off", "edits are broken", 2.0)
    assert policies["gemma-4-12b"][0] == "auto"
    assert stats.load_devices() == {"flux_klein": "GPU-small"}
    # Once: a later restart does not re-merge over what the operator changed since.
    stats.save_policy("flux_klein", "pinned", None)
    again = StatsRecorder(db)
    assert again.load_policies()["flux_klein"][0] == "pinned"


def test_a_row_saved_under_a_renamed_recipes_old_name_still_counts(store):
    """The stt recipes were renamed (tiny -> faster-whisper-tiny) and the old
    names kept as aliases; an override saved under the old one is the same
    recipe's intent."""
    from giq.stats import get_stats

    get_stats().save_policy("tiny", OFF, "kept off")
    store.load()
    assert store.policy_for("faster-whisper-tiny") == OFF
    assert store.policy_for("tiny") == OFF
