# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""How long a model may take to start, and how long a caller waits for it.

A start is not a job's run: it has its own budget, per engine and per
recipe, and its own failure. These pin the three places that used to get
it wrong — llama.cpp's fixed 30 s, a failed start reported as a job timeout,
and the API's flat 120 s wait that a cold vllm start could not meet.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from giq import runner
from giq.adapters import llama_cpp
from giq.adapters.engine import StartError
from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig
from giq.models import JobRequest, JobStatus, Modality
from giq.queue import Job


def _llama(tmp_path, **over) -> LlamaCppAdapter:
    log = tmp_path / "llama.log"
    log.write_text("load_model: failed to open GGUF\n")
    config = LlamaCppConfig(
        model="gemma-4-12b", model_path=str(tmp_path / "m.gguf"), log_path=str(log), **over
    )
    return LlamaCppAdapter(config=config)


def _answering(status: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda r: httpx.Response(status))
    )


@pytest.mark.asyncio
async def test_llama_start_that_dies_fails_at_once_with_its_log(tmp_path):
    worker = _llama(tmp_path)
    worker._process = SimpleNamespace(returncode=1)  # type: ignore[assignment]
    worker._client = _answering(503)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(StartError, match="exited with 1") as exc:
        await worker._wait_for_ready(timeout=60, poll=0.01)
    assert loop.time() - started < 1, "a dead server must not be waited out"
    assert "failed to open GGUF" in str(exc.value)


@pytest.mark.asyncio
async def test_llama_start_past_its_budget_is_a_start_error_not_a_timeout(tmp_path):
    worker = _llama(tmp_path)
    worker._process = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    worker._client = _answering(503)
    with pytest.raises(StartError, match="did not become ready") as exc:
        await worker._wait_for_ready(timeout=0.05, poll=0.01)
    # The runner reads TimeoutError as "the job ran too long".
    assert not isinstance(exc.value, TimeoutError)


@pytest.mark.asyncio
async def test_llama_ready_once_health_answers(tmp_path):
    worker = _llama(tmp_path)
    worker._process = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    worker._client = _answering(200)
    await worker._wait_for_ready(timeout=5, poll=0.01)


def test_llama_ready_timeout_is_the_engine_default_unless_the_instance_sets_one(
    tmp_path, monkeypatch
):
    assert _llama(tmp_path).config.ready_timeout == llama_cpp.DEFAULT_READY_TIMEOUT
    monkeypatch.setitem(llama_cpp.MODEL_READY_TIMEOUT, "gemma-4-12b", 900.0)
    assert _llama(tmp_path).config.ready_timeout == 900.0
    assert runner.start_budget("gemma-4-12b") == 900.0


def test_vllm_start_budget_is_the_instance_ready_timeout():
    from giq_vllm.adapter import recipe_for

    recipe = recipe_for("qwen3.8-27b-nvfp4")
    assert recipe is not None
    budget = runner.start_budget("qwen3.8-27b-nvfp4")
    assert budget == recipe.params.ready_timeout
    # The measured cold start the API's old flat wait could not cover.
    assert budget > 192


def test_wait_budget_covers_a_start_and_the_run():
    job = Job(
        job_id="j",
        request=JobRequest(
            modality=Modality.llm,
            model="qwen3.8-27b-nvfp4",
            chat_request={"messages": [], "max_tokens": 100},
        ),
    )
    expected = runner.start_budget("qwen3.8-27b-nvfp4") + runner._job_timeout(Modality.llm, job)
    assert runner.wait_budget(job) == expected
    assert runner.wait_budget(job) > 120


@pytest.mark.asyncio
async def test_wait_for_job_defaults_to_the_jobs_own_budget(monkeypatch):
    """No timeout given means the job's wait budget, not a constant."""
    from fastapi import HTTPException

    from giq.services.orchestration import Orchestrator

    job = Job(job_id="j", request=JobRequest(modality=Modality.llm, model="m", chat_request={}))
    job.status = JobStatus.running
    seen = []

    def budget(j):
        seen.append(j.job_id)
        return 0.05

    monkeypatch.setattr(runner, "wait_budget", budget)
    orch = Orchestrator.__new__(Orchestrator)
    orch.queue = SimpleNamespace(get=lambda _id: _found(job))  # type: ignore[assignment]

    with pytest.raises(HTTPException) as exc:
        await orch.wait_for_job("j")
    assert exc.value.status_code == 504
    assert seen == ["j"]


async def _found(job):
    return job
