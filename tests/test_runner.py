# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for job runner."""

from unittest.mock import AsyncMock

import pytest

from giq.models import JobRequest, JobStatus, Modality
from giq.queue import Job, JobQueue
from giq.runner import Runner


@pytest.fixture
def queue():
    """Create a fresh queue."""
    return JobQueue()


@pytest.fixture
def runner(queue: JobQueue):
    """Create a runner with fresh queue."""
    return Runner(queue)


def make_job(job_id: str, model: str = "test-model") -> Job:
    """Helper to create a test job."""
    return Job(
        job_id=job_id,
        request=JobRequest(
            modality=Modality.llm,
            model=model,
            tasks=[{"id": "t1", "user": "Hello"}],
        ),
    )


def test_runner_initial_state(runner: Runner):
    """Test runner starts with no active worker."""
    assert runner.active_modality is None
    assert runner.active_recipe is None


@pytest.mark.asyncio
async def test_runner_start_stop(runner: Runner):
    """Test runner can start and stop."""
    await runner.start()
    assert runner._running is True

    await runner.stop()
    assert runner._running is False


@pytest.mark.asyncio
async def test_runner_processes_job(queue: JobQueue, runner: Runner):
    """Test runner picks up job from queue (without actual worker)."""
    job = make_job("job1")
    await queue.add(job)

    # Job should be pending
    assert job.status == JobStatus.pending

    # Note: We can't test full processing without mocking llama-server
    # This just verifies the queue integration works
    pending = await queue.get_pending()
    assert len(pending) == 1


async def _prime_warm_worker(runner: Runner, modality: Modality, model: str):
    """Set runner state as if a job for (modality, model) just finished."""
    from giq.runner import Instance

    device = runner._device_for(model)
    worker = AsyncMock()  # _unload_worker calls .stop()
    runner._slots[device] = Instance(worker, model, device)
    runner._processing_job = False


@pytest.mark.asyncio
async def test_warm_timeout_keeps_worker_for_matching_model(queue: JobQueue, runner: Runner):
    """Same (modality, model) pending: stay warm (don't eager-unload)."""
    await _prime_warm_worker(runner, Modality.llm, "gemma-3-27b-it-qat")
    await queue.add(make_job("j1", model="gemma-3-27b-it-qat"))

    # Drop into the eager-check branch only (skip the 120s sleep by cancelling
    # the task immediately after the branch decides).
    check = runner._warm_timeout_check()
    # Await up to the sleep: the eager branch is synchronous until asyncio.sleep.
    import asyncio

    try:
        await asyncio.wait_for(check, timeout=0.5)
    except TimeoutError:
        # Expected — we hit the 120s sleep, which means eager-unload did NOT fire.
        pass

    assert runner._slots  # still loaded
    assert runner.active_recipe == "gemma-3-27b-it-qat"


@pytest.mark.asyncio
async def test_warm_timeout_evicts_for_different_model(queue: JobQueue, runner: Runner):
    """Same modality, different model pending: eager-unload."""
    await _prime_warm_worker(runner, Modality.llm, "gemma-3-27b-it-qat")
    await queue.add(make_job("j1", model="qwen-coder-30b"))

    await runner._warm_timeout_check()

    assert runner._slots == {}


@pytest.mark.asyncio
async def test_warm_timeout_evicts_for_different_worker_type(queue: JobQueue, runner: Runner):
    """Different modality pending: eager-unload."""
    await _prime_warm_worker(runner, Modality.llm, "gemma-3-27b-it-qat")
    job = Job(
        job_id="img1",
        request=JobRequest(
            modality=Modality.text2image,
            model="zimage",
            tasks=[{"id": "t1", "prompt": "a cat"}],
        ),
    )
    await queue.add(job)

    await runner._warm_timeout_check()

    assert runner._slots == {}


@pytest.mark.asyncio
async def test_warm_timeout_keeps_a_recipe_warm_for_its_other_modality(
    queue: JobQueue, runner: Runner
):
    """flux_klein renders and edits from one sd-server (ADR-003): an edit
    queued behind a render is served by the loaded process, not a reload."""
    import asyncio

    await _prime_warm_worker(runner, Modality.text2image, "flux_klein")
    job = Job(
        job_id="edit1",
        request=JobRequest(
            modality=Modality.image_edit,
            model="flux_klein",
            tasks=[{"id": "t1", "prompt": "a hat", "image_b64": "x"}],
        ),
    )
    await queue.add(job)

    try:
        await asyncio.wait_for(runner._warm_timeout_check(), timeout=0.5)
    except TimeoutError:
        pass  # reached the warm sleep: no eager unload

    assert runner.active_recipe == "flux_klein"


RESIDENTS = [
    "gemma-4-12b",
    "whisper-large-v3",
    "ecapa-tdnn",
]


def _prime_resident(runner: Runner, key, vram_gb: float, width: int = 1):
    """Install a fake ready resident."""
    from giq.runner import RESIDENT, Instance

    worker = AsyncMock()
    worker.is_ready = True
    worker.estimated_vram_gb = vram_gb
    # Same card the model would really load on, so eviction (which only
    # considers residents sharing the incoming model's card) sees them.
    res = Instance(worker, key, runner._device_for(key), residency=RESIDENT, width=width)
    res.last_active = 0.0  # long quiet — eviction grace already satisfied
    runner._residents[key] = res
    return res


def test_resident_key_detection(queue: JobQueue):
    runner = Runner(queue, residents=RESIDENTS)
    assert runner.is_resident_key("gemma-4-12b")
    assert not runner.is_resident_key("flux_klein")
    assert not runner.is_resident_key("qwen3.6-27b")


@pytest.mark.asyncio
async def test_eviction_picks_minimal_single_victim(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
):
    """The smallest single resident that covers the deficit is evicted alone."""
    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    _prime_resident(runner, RESIDENTS[0], vram_gb=9.5)  # gemma
    audio = _prime_resident(runner, RESIDENTS[1], vram_gb=4.0)
    _prime_resident(runner, RESIDENTS[2], vram_gb=0.6)

    # 2.4GB free; llama-3.2-3b needs 3+1.5=4.5 → deficit 2.1 → audio alone
    # covers it (smallest single ≥ deficit); embed and gemma keep serving.
    monkeypatch.setattr(runner_mod, "get_free_vram", lambda *a: 2.4)

    await runner._evict_residents_for("llama-3.2-3b")

    assert RESIDENTS[0] in runner._residents  # gemma survives
    assert RESIDENTS[1] not in runner._residents  # audio evicted
    assert RESIDENTS[2] in runner._residents  # embed survives
    audio.adapter.stop.assert_awaited()


@pytest.mark.asyncio
async def test_eviction_flux_takes_gemma_only(queue: JobQueue, monkeypatch: pytest.MonkeyPatch):
    """flux via sdcpp (9GB) evicts gemma alone; audio residents survive."""
    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    gemma = _prime_resident(runner, RESIDENTS[0], vram_gb=9.5)
    _prime_resident(runner, RESIDENTS[1], vram_gb=4.0)
    _prime_resident(runner, RESIDENTS[2], vram_gb=0.6)

    # 2.5GB free; flux_klein needs 9+2=11 → deficit 8.5 → gemma (9.5) alone.
    monkeypatch.setattr(runner_mod, "get_free_vram", lambda *a: 2.5)

    await runner._evict_residents_for("flux_klein")

    assert RESIDENTS[0] not in runner._residents  # gemma evicted
    assert RESIDENTS[1] in runner._residents  # audio survives
    assert RESIDENTS[2] in runner._residents  # embed survives
    gemma.adapter.stop.assert_awaited()


@pytest.mark.asyncio
async def test_eviction_cumulative_fallback_takes_everything(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
):
    """When no single resident covers the deficit (zimage, 13+2GB into 0.5 free),
    cumulative cheapest-first displaces the whole set."""
    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    for key, gb in zip(RESIDENTS, (9.5, 4.0, 0.6), strict=True):
        _prime_resident(runner, key, vram_gb=gb)
    monkeypatch.setattr(runner_mod, "get_free_vram", lambda *a: 0.5)

    await runner._evict_residents_for("zimage")

    assert not runner._residents


@pytest.mark.asyncio
async def test_eviction_noop_for_resident_job_key(queue: JobQueue):
    """Jobs for resident models never trigger eviction (they use lanes)."""
    runner = Runner(queue, residents=RESIDENTS)
    _prime_resident(runner, RESIDENTS[0], vram_gb=9.5)
    # No monkeypatched VRAM: must return before ever reading free VRAM.
    await runner._evict_residents_for(RESIDENTS[0])
    assert RESIDENTS[0] in runner._residents


@pytest.mark.asyncio
async def test_resident_dispatch_claims_job_before_yielding(queue: JobQueue):
    """Regression (an OOM incident): the dispatch loop must mark a resident job
    running synchronously — otherwise get_next() returns the same pending job
    in a tight loop, spawning unbounded tasks and starving the event loop."""
    import asyncio

    runner = Runner(queue, residents=RESIDENTS)
    job = make_job("aud1", model="gemma-4-12b")
    await queue.add(job)

    await runner.start()
    try:
        await asyncio.sleep(0.3)
        assert job.status == JobStatus.running
        # Exactly one lane task, not thousands.
        assert len(runner._resident_jobs) == 1
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_eviction_waits_for_in_flight_lane(queue: JobQueue, monkeypatch: pytest.MonkeyPatch):
    """Eviction drains an active lane instead of stopping the worker mid-job."""
    import asyncio

    import giq.runner as runner_mod

    runner = Runner(queue, residents=RESIDENTS)
    embed = _prime_resident(runner, RESIDENTS[2], vram_gb=0.6, width=1)
    monkeypatch.setattr(runner_mod, "get_free_vram", lambda *a: 15.0)
    monkeypatch.setattr(runner_mod, "EVICT_DEFER_CAP_SECONDS", 2.0)

    # Simulate an in-flight embed holding the lane.
    await embed.lane.acquire()
    embed.active_count = 1

    evict = asyncio.create_task(runner._evict_residents_for("other-embed"))
    await asyncio.sleep(0.1)
    assert RESIDENTS[2] in runner._residents  # not evicted while in flight

    embed.active_count = 0
    embed.last_active = 0.0
    embed.lane.release()
    await asyncio.wait_for(evict, timeout=10.0)
    assert RESIDENTS[2] not in runner._residents


@pytest.mark.asyncio
async def test_an_on_demand_child_job_with_dict_results_completes(
    queue: JobQueue, monkeypatch: pytest.MonkeyPatch
):
    """The audio and voiceprint children return plain dicts. Once nothing was
    kept warm by default they ran on demand, where the runner called
    model_dump() on every result and failed each transcription with a 500."""
    runner = Runner(queue)

    class ChildAdapter:
        async def run_batch(self, tasks, params=None):
            return [{"id": t["id"], "text": "hello", "error": None} for t in tasks]

    monkeypatch.setattr(runner, "_evict_residents_for", AsyncMock())
    monkeypatch.setattr(runner, "_ensure_worker", AsyncMock(return_value=ChildAdapter()))
    job = Job(
        job_id="audio1",
        request=JobRequest(modality=Modality.audio, model="whisper-large-v3", tasks=[{"id": "t1"}]),
    )
    await queue.add(job)
    await runner._process_job(job)

    assert job.status == JobStatus.completed, job.error
    assert job.results == [{"id": "t1", "text": "hello", "error": None}]
