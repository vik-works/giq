# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import uuid

from fastapi import HTTPException

from giq.apps import current_app
from giq.models import JobRequest, JobStatus
from giq.queue import Job, JobStream, get_queue

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.queue = get_queue()

    @staticmethod
    def _reject_if_paused() -> None:
        """503 every submission while the runner is paused.

        The single chokepoint for job entry, so this covers /run, the OpenAI
        endpoints and the dashboard self-tests alike. Failing fast (rather than
        queueing) is deliberate: sync callers wait up to ~880s, and a pause can
        last hours — clients should back off and retry, not block.
        """
        from giq.runner import get_runner

        runner = get_runner()
        if not runner.is_paused:
            return
        state = runner.pause_state
        detail = "giq is paused — all models unloaded to free VRAM"
        if state["since"]:
            detail += f" (since {state['since']})"
        if state["reason"]:
            detail += f": {state['reason']}"
        raise HTTPException(
            status_code=503,
            detail=detail,
            headers={"Retry-After": "60", "X-Giq-Paused": "1"},
        )

    @staticmethod
    def _resolve_recipe(request: JobRequest) -> None:
        """Name the job by its recipe, and refuse a modality the recipe does not serve.

        Clients may send an alias; everything after this point — policy, the
        VRAM gate, the runner's residents and slots, the stats — keys on the
        recipe's own name, so the alias is resolved once, here (ADR-003). An
        unknown name passes through: the catalog is not a whitelist, and the
        VRAM gate's default makes an unknown model fail loudly on its own.
        """
        from giq.registry import get_recipe

        recipe = get_recipe(request.model)
        if recipe is None:
            return
        if not recipe.serves(request.modality):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{recipe.name} does not serve {request.modality} "
                    f"(it serves: {', '.join(recipe.modalities)})"
                ),
            )
        request.model = recipe.name

    @staticmethod
    def _reject_if_disabled(request: JobRequest) -> None:
        """503 submissions for a model the operator has switched off.

        Refuse rather than queue, for the same reason pause does: `off` is a
        deliberate state that can last days, and a sync caller waiting ~880s
        for a model that will never load is worse than a fast, clear error.
        """
        from giq.policy import get_policy_store

        model = request.model
        if not get_policy_store().is_off(model):
            return
        record = get_policy_store().record_for(model)
        detail = (
            f"{model} is switched off — it will not load until it is set back to pinned or auto"
        )
        if record.reason:
            detail += f": {record.reason}"
        raise HTTPException(
            status_code=503,
            detail=detail,
            headers={"Retry-After": "300", "X-Giq-Model-Policy": "off"},
        )

    async def submit_job(self, request: JobRequest) -> tuple[str, int]:
        self._reject_if_paused()
        self._resolve_recipe(request)
        self._reject_if_disabled(request)
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, request=request, app=current_app.get())
        position = await self.queue.add(job)
        return job_id, position

    async def submit_streaming_job(self, request: JobRequest) -> tuple[str, int, JobStream]:
        """Submit a job whose output is consumed live rather than at the end.

        Deliberately the same entry as submit_job — pause checks, the off-policy
        check, the queue, eviction and accounting all still apply. The only
        difference is that the job carries a channel the worker writes into as
        it generates. Streaming must not become a side door around the
        scheduler.
        """
        self._reject_if_paused()
        self._resolve_recipe(request)
        self._reject_if_disabled(request)
        job_id = str(uuid.uuid4())[:8]
        stream = JobStream()
        job = Job(job_id=job_id, request=request, stream=stream, app=current_app.get())
        position = await self.queue.add(job)
        return job_id, position, stream

    async def wait_for_job(self, job_id: str, timeout: float | None = None) -> Job:
        """Wait for job completion, polling queue.

        ``timeout`` None waits as long as the job may take: a start of its
        model plus its run (``giq.runner.wait_budget``), so a cold start is
        not cut off by a constant chosen for warm ones.
        """
        start = asyncio.get_event_loop().time()

        while True:
            job = await self.queue.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job disappeared")
            if timeout is None:
                from giq.runner import wait_budget

                timeout = wait_budget(job)

            if job.status == JobStatus.completed:
                return job
            if job.status == JobStatus.failed:
                error = job.error or "Unknown error"
                if job.results:
                    # Results may be dicts (runner stores model_dump output).
                    errors = [
                        (r.get("error") if isinstance(r, dict) else r.error) for r in job.results
                    ]
                    errors = [e for e in errors if e]
                    if errors:
                        error = "; ".join(errors)
                raise HTTPException(status_code=500, detail=f"Job failed: {error}")

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                # Remove zombie job from queue so it doesn't block warm timeout
                if job.status == JobStatus.pending:
                    await self.queue.remove(job_id)
                raise HTTPException(status_code=504, detail="Job timed out")

            await asyncio.sleep(0.1)

    async def cancel_job(self, job_id: str) -> tuple[bool, str]:
        job = await self.queue.get(job_id)
        if not job:
            return False, "Job not found"

        from giq.models import JobStatus

        if job.status == JobStatus.pending:
            await self.queue.remove(job_id)
            return True, "Cancelled"

        # A running streaming job can be stopped for real: the worker checks
        # the flag between chunks and drops the connection to llama-server,
        # which stops generating. Everything else has to run to completion.
        if job.stream is not None and job.status == JobStatus.running:
            job.stream.cancel()
            return True, "Cancelling"

        return False, f"Job is {job.status} (cannot cancel)"
