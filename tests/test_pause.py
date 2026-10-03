# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for pause/resume: unload everything, refuse work, come back cleanly."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from giq.models import JobRequest, JobStatus, Modality
from giq.queue import Job, JobQueue
from giq.runner import Runner

RESIDENTS = [
    "gemma-4-12b",
    "whisper-large-v3",
    "ecapa-tdnn",
]


@pytest.fixture
def queue():
    return JobQueue()


def make_job(job_id: str, model: str = "test-model") -> Job:
    return Job(
        job_id=job_id,
        request=JobRequest(
            modality=Modality.llm, model=model, tasks=[{"id": "t1", "user": "Hello"}]
        ),
    )


def _prime_resident(runner: Runner, key, width: int = 1):
    from giq.runner import RESIDENT, Instance

    worker = AsyncMock()
    worker.is_ready = True
    worker.estimated_vram_gb = 4.0
    res = Instance(worker, key, residency=RESIDENT, width=width)
    runner._residents[key] = res
    return res


@pytest.mark.asyncio
async def test_pause_unloads_residents_and_batch_worker(queue: JobQueue):
    runner = Runner(queue, residents=RESIDENTS)
    batch = AsyncMock()
    from giq.runner import Instance

    device = runner._device_for("zimage")
    runner._slots[device] = Instance(batch, "zimage", device)
    residents = [_prime_resident(runner, key) for key in RESIDENTS]

    state = await runner.pause()

    assert state["paused"] is True
    assert state["warnings"] == []
    assert runner.is_paused
    assert not runner._residents
    assert runner.active_modality is None
    batch.stop.assert_awaited()
    for res in residents:
        res.adapter.stop.assert_awaited()


@pytest.mark.asyncio
async def test_pause_fails_queued_jobs(queue: JobQueue):
    """Nothing is left waiting behind the pause — queued jobs get an answer."""
    runner = Runner(queue, residents=RESIDENTS)
    job = make_job("j1")
    await queue.add(job)

    await runner.pause()

    assert job.status == JobStatus.failed
    assert "paused" in job.error


@pytest.mark.asyncio
async def test_pause_closes_the_streams_of_queued_jobs(queue: JobQueue):
    """A streaming caller is not polling the job: it waits on the stream's end
    sentinel, which only the run paths send. A job failed before it ever ran
    never reaches them, so pause has to close the stream itself."""
    from giq.queue import JobStream

    runner = Runner(queue, residents=RESIDENTS)
    job = make_job("j1")
    job.stream = JobStream()
    await queue.add(job)

    await runner.pause()

    assert job.status == JobStatus.failed
    assert job.stream.queue.get_nowait() is None


@pytest.mark.asyncio
async def test_pause_is_idempotent(queue: JobQueue):
    runner = Runner(queue)
    await runner.pause(reason="first")
    state = await runner.pause(reason="second")
    assert state["reason"] == "first"  # the original pause stands


@pytest.mark.asyncio
async def test_graceful_pause_waits_for_in_flight_job(queue: JobQueue, monkeypatch):
    """A graceful pause lets a running job finish before tearing workers down."""
    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    res = _prime_resident(runner, RESIDENTS[2])
    monkeypatch.setattr(runner_mod, "PAUSE_DRAIN_TIMEOUT_SECONDS", 5.0)
    res.active_count = 1  # in flight

    task = asyncio.create_task(runner.pause())
    await asyncio.sleep(0.2)
    assert runner.is_paused  # gate is up immediately...
    assert RESIDENTS[2] in runner._residents  # ...but the worker still serves

    res.active_count = 0
    state = await asyncio.wait_for(task, timeout=5.0)

    assert state["drained"] is True
    assert not runner._residents
    res.adapter.stop.assert_awaited()


@pytest.mark.asyncio
async def test_forced_pause_skips_the_drain(queue: JobQueue, monkeypatch):
    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    res = _prime_resident(runner, RESIDENTS[2])
    res.active_count = 1  # in flight — a graceful pause would wait it out
    monkeypatch.setattr(runner_mod, "PAUSE_DRAIN_TIMEOUT_SECONDS", 30.0)

    state = await asyncio.wait_for(runner.pause(force=True), timeout=5.0)

    assert state["forced"] is True
    assert not runner._residents
    res.adapter.stop.assert_awaited()


@pytest.mark.asyncio
async def test_drain_timeout_unloads_anyway_and_warns(queue: JobQueue, monkeypatch):
    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    res = _prime_resident(runner, RESIDENTS[2])
    res.active_count = 1  # never finishes
    monkeypatch.setattr(runner_mod, "PAUSE_DRAIN_TIMEOUT_SECONDS", 0.6)

    state = await asyncio.wait_for(runner.pause(), timeout=10.0)

    assert state["drained"] is False
    assert any("still running" in w for w in state["warnings"])
    assert not runner._residents


@pytest.mark.asyncio
async def test_unload_failure_is_reported_not_raised(queue: JobQueue):
    """An unkillable worker leaves giq paused with a warning, not a 500."""
    runner = Runner(queue, residents=RESIDENTS)
    res = _prime_resident(runner, RESIDENTS[0])
    res.adapter.stop.side_effect = RuntimeError("CUDA-stuck")

    state = await runner.pause()

    assert state["paused"] is True
    assert any("still loaded" in w for w in state["warnings"])
    assert RESIDENTS[0] in runner._residents  # kept for a later retry


@pytest.mark.asyncio
async def test_paused_runner_does_not_dispatch(queue: JobQueue):
    runner = Runner(queue, residents=RESIDENTS)
    await runner.pause()
    await runner.start()
    try:
        await queue.add(make_job("j1", model="gemma-4-12b"))
        await asyncio.sleep(0.5)
        pending = await queue.get_pending()
        assert len(pending) == 1  # still pending: no lane task, no load attempt
        assert not runner._resident_jobs
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_paused_runner_refuses_to_load(queue: JobQueue):
    """A load already in flight when the pause lands must not re-claim the card."""
    runner = Runner(queue, residents=RESIDENTS)
    await runner.pause()

    with pytest.raises(RuntimeError, match="paused"):
        await runner._ensure_worker("llama-3.2-3b")
    assert runner.active_modality is None

    await runner._load_resident(RESIDENTS[0])
    assert not runner._residents


@pytest.mark.asyncio
async def test_resume_clears_state(queue: JobQueue):
    runner = Runner(queue, residents=RESIDENTS)
    await runner.pause(reason="gaming")
    state = await runner.resume()

    assert state["paused"] is False
    assert state["reason"] is None
    assert not runner.is_paused


@pytest.mark.asyncio
async def test_resume_when_not_paused_is_a_noop(queue: JobQueue):
    runner = Runner(queue)
    assert (await runner.resume())["paused"] is False


@pytest.mark.asyncio
async def test_control_endpoints_round_trip(queue: JobQueue, monkeypatch):
    """/control/pause → /status → /control/resume over the real ASGI app."""
    import httpx

    import giq.runner as runner_mod
    from giq.main import app

    runner = Runner(queue, residents=RESIDENTS)
    _prime_resident(runner, RESIDENTS[0])
    monkeypatch.setattr(runner_mod, "_runner", runner)

    transport = httpx.ASGITransport(app=app)  # no lifespan: nothing touches the GPU
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        r = await client.post("/control/pause", json={"reason": "gaming"})
        assert r.status_code == 200, r.text
        assert r.json()["paused"] is True
        assert not runner._residents

        status = (await client.get("/status")).json()
        assert status["state"] == "paused"
        assert status["paused"] is True
        assert status["pause_reason"] == "gaming"

        # Submissions are refused with a retryable 503 while paused.
        r = await client.post(
            "/run",
            json={"worker": "llm", "model": "gemma-4-12b", "tasks": [{"id": "t", "user": "hi"}]},
        )
        assert r.status_code == 503
        assert r.headers["Retry-After"]
        assert "paused" in r.json()["detail"]

        assert (await client.get("/llm/endpoint", params={"model": "gemma-4-12b"})).json()[
            "detail"
        ]["state"] == "paused"

        r = await client.post("/control/resume")
        assert r.status_code == 200
        assert r.json()["paused"] is False
        assert (await client.get("/status")).json()["paused"] is False


@pytest.mark.asyncio
async def test_submit_is_refused_while_paused(queue: JobQueue, monkeypatch):
    """The API edge fails fast (503) instead of queueing behind the pause."""
    import giq.runner as runner_mod
    from giq.services.orchestration import Orchestrator

    runner = Runner(queue, residents=RESIDENTS)
    monkeypatch.setattr(runner_mod, "_runner", runner)
    orch = Orchestrator()
    orch.queue = queue

    job_id, _ = await orch.submit_job(
        JobRequest(modality=Modality.llm, model="gemma-4-12b", tasks=[{"id": "t", "user": "hi"}])
    )
    assert job_id

    await runner.pause(reason="gaming")
    with pytest.raises(HTTPException) as exc:
        await orch.submit_job(
            JobRequest(
                modality=Modality.llm, model="gemma-4-12b", tasks=[{"id": "t", "user": "hi"}]
            )
        )
    assert exc.value.status_code == 503
    assert "gaming" in exc.value.detail
    assert exc.value.headers["Retry-After"]

    await runner.resume()
    job_id, _ = await orch.submit_job(
        JobRequest(modality=Modality.llm, model="gemma-4-12b", tasks=[{"id": "t", "user": "hi"}])
    )
    assert job_id
