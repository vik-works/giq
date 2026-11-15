# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""An on-demand instance takes as many jobs at once as its lane is wide."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import giq.runner as runner_mod
from giq.models import JobRequest, JobStatus, Modality
from giq.queue import Job, JobQueue
from giq.runner import Runner


class FakeAdapter:
    """An engine that holds every job until released, and counts them."""

    estimated_vram_gb = 1.0
    pid = None

    def __init__(self, model: str, log: list[str]):
        self.model = model
        self.log = log
        self.is_ready = False
        self.is_running = False
        self.release = asyncio.Event()
        self.running = 0
        self.peak = 0
        self.fail: set[str] = set()

    async def start(self) -> None:
        self.is_ready = self.is_running = True
        self.log.append(f"start {self.model}")

    async def stop(self) -> None:
        self.is_ready = self.is_running = False
        self.log.append(f"stop {self.model}")

    async def run_batch(self, tasks, params=None):
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await self.release.wait()
            if tasks[0]["id"] in self.fail:
                raise RuntimeError("the engine rejected this request")
            return [{"id": t["id"], "text": "ok", "error": None} for t in tasks]
        finally:
            self.running -= 1


def make_job(job_id: str, model: str) -> Job:
    return Job(
        job_id=job_id,
        request=JobRequest(modality=Modality.llm, model=model, tasks=[{"id": job_id}]),
    )


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch):
    """A runner whose loads build FakeAdapters, ``widths`` lanes wide."""
    queue = JobQueue()
    runner = Runner(queue)
    log: list[str] = []
    adapters: dict[str, FakeAdapter] = {}
    widths: dict[str, int] = {}

    def build(model, model_path=None):
        adapters[model] = FakeAdapter(model, log)
        return adapters[model]

    monkeypatch.setattr(runner, "_build_worker", build)
    monkeypatch.setattr(runner, "_device_for", lambda model: None)
    monkeypatch.setattr(runner_mod, "_lane_width", lambda model, adapter: widths.get(model, 1))
    monkeypatch.setattr(runner_mod, "wait_for_vram", AsyncMock(return_value=True))
    monkeypatch.setattr(runner_mod, "get_free_vram", lambda *a: 24.0)
    return runner, queue, adapters, widths, log


async def settle(predicate, timeout: float = 3.0) -> None:
    """Let the dispatch loop run until ``predicate`` holds."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "the runner never got there"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_burst_fills_the_lanes_of_an_on_demand_instance(rig):
    runner, queue, adapters, widths, _log = rig
    widths["wide"] = 4
    jobs = [make_job(f"j{i}", "wide") for i in range(6)]
    for job in jobs:
        await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "wide" in adapters and adapters["wide"].running == 4)
        await asyncio.sleep(0.1)
        # Four on the engine, the other two still queued in order behind them.
        assert adapters["wide"].running == 4
        assert [j.status for j in jobs[4:]].count(JobStatus.completed) == 0
        (inst,) = runner.instances()
        assert (inst.residency, inst.width, inst.active_count) == ("on_demand", 4, 4)

        adapters["wide"].release.set()
        await settle(lambda: all(j.status == JobStatus.completed for j in jobs))
        assert adapters["wide"].peak == 4
        assert inst.active_count == 0
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_an_instance_one_lane_wide_still_runs_one_job_at_a_time(rig):
    runner, queue, adapters, _widths, _log = rig
    jobs = [make_job(f"j{i}", "narrow") for i in range(3)]
    for job in jobs:
        await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "narrow" in adapters and adapters["narrow"].running == 1)
        await asyncio.sleep(0.1)
        assert adapters["narrow"].running == 1
        adapters["narrow"].release.set()
        await settle(lambda: all(j.status == JobStatus.completed for j in jobs))
        assert adapters["narrow"].peak == 1
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_a_job_for_another_recipe_waits_out_the_jobs_in_flight(rig):
    """The card changes hands in queue order, and never under a running job."""
    runner, queue, adapters, widths, log = rig
    widths["wide"] = 4
    burst = [make_job(f"j{i}", "wide") for i in range(3)]
    other = make_job("other", "narrow")
    late = make_job("late", "wide")
    for job in [*burst, other, late]:
        await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "wide" in adapters and adapters["wide"].running == 3)
        await asyncio.sleep(0.1)
        # A lane is free, but the job behind `other` does not jump the queue.
        assert adapters["wide"].running == 3
        assert late.status == JobStatus.pending
        assert log == ["start wide"]

        first = adapters["wide"]
        first.release.set()
        await settle(lambda: "narrow" in adapters and adapters["narrow"].running == 1)
        assert all(j.status == JobStatus.completed for j in burst)
        assert log == ["start wide", "stop wide", "start narrow"]

        adapters["narrow"].release.set()
        await settle(lambda: adapters["wide"] is not first)
        adapters["wide"].release.set()
        await settle(lambda: late.status == JobStatus.completed)
        assert other.status == JobStatus.completed
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_one_failed_job_leaves_its_siblings_running(rig):
    runner, queue, adapters, widths, log = rig
    widths["wide"] = 4
    jobs = [make_job(f"j{i}", "wide") for i in range(3)]
    for job in jobs:
        await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "wide" in adapters and adapters["wide"].running == 3)
        adapter = adapters["wide"]
        adapter.fail.add("j0")
        adapter.release.set()
        await settle(lambda: all(j.status != JobStatus.running for j in jobs))
        assert [j.status for j in jobs] == [
            JobStatus.failed,
            JobStatus.completed,
            JobStatus.completed,
        ]
        assert "rejected" in (jobs[0].error or "")
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_a_job_that_fails_alone_unloads_the_instance(rig):
    """Nothing else is running on it, so the next job gets a clean start."""
    runner, queue, adapters, widths, log = rig
    widths["wide"] = 4
    job = make_job("j0", "wide")
    await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "wide" in adapters and adapters["wide"].running == 1)
        adapters["wide"].fail.add("j0")
        adapters["wide"].release.set()
        await settle(lambda: log == ["start wide", "stop wide"])
        assert job.status == JobStatus.failed
        assert runner.instances() == []
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_a_forced_pause_fails_the_jobs_on_the_lanes(rig):
    runner, queue, adapters, widths, _log = rig
    widths["wide"] = 4
    jobs = [make_job(f"j{i}", "wide") for i in range(2)]
    for job in jobs:
        await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "wide" in adapters and adapters["wide"].running == 2)
        await runner.pause(force=True)
        assert [j.status for j in jobs] == [JobStatus.failed, JobStatus.failed]
        assert all("paused" in (j.error or "") for j in jobs)
        assert runner.instances() == []
        assert not runner._on_demand_jobs
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_the_warm_timeout_leaves_an_instance_with_jobs_on_its_lanes(rig):
    """A queued job for another recipe must not unload it under them."""
    runner, queue, adapters, widths, log = rig
    widths["wide"] = 4
    job = make_job("j0", "wide")
    await queue.add(job)

    await runner.start()
    try:
        await settle(lambda: "wide" in adapters and adapters["wide"].running == 1)
        runner._paused = True  # hold the dispatch loop, so only the check acts
        await queue.add(make_job("other", "narrow"))
        await runner._warm_timeout_check()
        assert log == ["start wide"]
    finally:
        runner._paused = False
        adapters["wide"].release.set()
        await runner.stop()
