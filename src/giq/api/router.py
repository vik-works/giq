# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from giq import __version__, plugins
from giq.api.access import posture as access_posture
from giq.api.dependencies import get_orchestrator
from giq.gpus import attribute_vram, get_gpus, resolve_device, selected_device
from giq.models import (
    Capabilities,
    JobRequest,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    ModalityCapability,
    PauseRequest,
    PauseResponse,
    ServiceState,
    ServiceStatus,
)
from giq.queue import get_queue
from giq.registry import get_recipe, recipes_serving
from giq.runner import get_runner
from giq.services.orchestration import Orchestrator
from giq.vram import (
    device_for_recipe,
    get_free_vram,
    get_vram_requirement,
    get_vram_status,
    margin_for,
    reserve_for,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Ceiling for ?wait=true: an image batch ahead in the queue can hold a job
# for many minutes. Kept under 900s, a common client-side poll budget.
SYNC_WAIT_TIMEOUT_SECONDS = 880.0

# The routers are imported once, as the service starts; uptime counts from
# here. Monotonic, so a clock change does not make the service younger.
_STARTED = time.monotonic()


@router.post("/run")
async def run_job(
    request: JobRequest,
    wait: bool = False,
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict:
    """Submit a new job for execution.

    With ``?wait=true`` the call blocks until the job completes and returns
    the full status payload (no client-side polling — the low-latency path
    for live audio). Client disconnect cancels a still-pending job so stale
    work doesn't pile up behind an image batch.
    """
    job_id, position = await orch.submit_job(request)
    logger.info(
        f"Job {job_id} submitted: {request.modality}/{request.model}, "
        f"{len(request.tasks) if request.tasks else 0} tasks"
    )
    if not wait:
        return JobResponse(job_id=job_id, position=position).model_dump()

    try:
        job = await orch.wait_for_job(job_id, timeout=SYNC_WAIT_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        await orch.cancel_job(job_id)
        raise
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        modality=job.request.modality,
        model=job.request.model,
        results=job.results if job.results else None,
        duration_ms=job.duration_ms,
    ).model_dump()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str, orch: Orchestrator = Depends(get_orchestrator)
) -> JobStatusResponse:
    """Get job status and results."""
    queue = get_queue()
    job = await queue.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        modality=job.request.modality,
        model=job.request.model,
        results=job.results if job.results else None,
        duration_ms=job.duration_ms,
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, orch: Orchestrator = Depends(get_orchestrator)) -> dict:
    """Cancel a pending job."""
    cancelled, reason = await orch.cancel_job(job_id)
    if not cancelled:
        if "not found" in reason.lower():
            raise HTTPException(status_code=404, detail=reason)
        return {"cancelled": False, "reason": reason}
    return {"cancelled": True}


@router.get("/gpus")
async def list_gpus() -> dict:
    """Per-GPU telemetry: identity (stable UUID), VRAM, temp, power, throttle.

    ``selected`` marks the default card. Each card also splits its used VRAM
    into ``vram_giq_gb`` (models giq is holding, with a per-process
    ``giq`` breakdown) and ``vram_other_gb`` (everything else on the card —
    the desktop, another CUDA app, a game). "Is this card full because of me?"
    is a different question from "is it full", and only the first one tells
    an operator whether giq can do anything about it.
    """
    gpus = await asyncio.to_thread(get_gpus)
    device = await asyncio.to_thread(selected_device)
    chosen = device.uuid if device else None
    owned = get_runner().owned_pids
    attribution = await asyncio.to_thread(attribute_vram, owned, gpus)
    return {
        "selected": chosen,
        "gpus": [
            {
                **g.to_dict(),
                "selected": g.uuid == chosen,
                "vram_giq_gb": attribution.get(g.uuid, {}).get("giq_gb", 0.0),
                "vram_other_gb": attribution.get(g.uuid, {}).get(
                    "other_gb", round(g.vram_used_gb, 2)
                ),
                "giq": attribution.get(g.uuid, {}).get("giq", []),
            }
            for g in gpus
        ],
    }


@router.get("/instances")
async def list_instances() -> dict:
    """Every recipe running on a card right now (ADR-003): what, where, how.

    ``residency`` says why it is up — ``resident`` (kept loaded by policy, on
    its own lane) or ``on_demand`` (loaded for a job, unloaded when idle).
    ``state`` is the adapter's: ``starting`` while the process is up but not
    answering, ``ready`` once it does. ``port`` is the loopback server a
    llama.cpp, vllm or sd.cpp instance listens on; in-process and child
    adapters have none.
    """
    runner = get_runner()
    out = []
    for inst in runner.instances():
        recipe = get_recipe(inst.recipe)
        gpu = resolve_device(inst.device) if inst.device else None
        out.append(
            {
                "id": inst.id,
                "recipe": inst.recipe,
                "modalities": list(recipe.modalities) if recipe else [],
                "engine": recipe.engine if recipe else None,
                "residency": inst.residency,
                "state": inst.state,
                "device": inst.device,
                "device_index": gpu.index if gpu else None,
                "device_name": gpu.name if gpu else None,
                "port": inst.port,
                "pid": getattr(inst.adapter, "pid", None),
                "lanes": inst.width,
                "in_flight": inst.active_count,
                "vram_gb": getattr(inst.adapter, "estimated_vram_gb", None),
                "started_at": inst.started_at,
            }
        )
    return {"instances": out}


@router.post("/control/pause", response_model=PauseResponse)
async def pause_serving(request: PauseRequest | None = None) -> PauseResponse:
    """Suspend serving and unload every model, freeing the GPU.

    Graceful by default: in-flight jobs and live llama-server sessions get up
    to a minute to finish before the teardown. ``{"force": true}`` skips that
    wait for when the VRAM is needed right now. While paused, job submission
    returns 503 — nothing queues up behind the pause. Idempotent.
    """
    req = request or PauseRequest()
    result = await get_runner().pause(force=req.force, reason=req.reason)
    # nvidia-smi can lag CUDA teardown by a second or two; the dashboard's
    # /status poll shows the settled figure moments later.
    vram = await asyncio.to_thread(get_vram_status)
    return PauseResponse(**result, vram_free_gb=vram.free_gb, vram_total_gb=vram.total_gb)


@router.post("/control/resume", response_model=PauseResponse)
async def resume_serving() -> PauseResponse:
    """Resume serving. Residents reload within a few seconds. Idempotent."""
    result = await get_runner().resume()
    vram = await asyncio.to_thread(get_vram_status)
    return PauseResponse(**result, vram_free_gb=vram.free_gb, vram_total_gb=vram.total_gb)


@router.get("/status", response_model=ServiceStatus)
async def get_service_status() -> ServiceStatus:
    """Get current service status."""
    queue = get_queue()
    runner = get_runner()
    vram = get_vram_status()
    device = selected_device()
    cards = await asyncio.to_thread(get_gpus)
    attribution = await asyncio.to_thread(attribute_vram, runner.owned_pids, cards)
    mine = attribution.get(device.uuid, {}) if device else {}

    all_jobs = await queue.get_all()
    pending = [j for j in all_jobs if j.status == JobStatus.pending]
    running = [j for j in all_jobs if j.status == JobStatus.running]

    pending_ids = [j.job_id for j in pending]
    running_ids = [j.job_id for j in running]

    pause = runner.pause_state

    gpus = [
        {
            "uuid": g.uuid,
            "index": g.index,
            "name": g.name,
            "selected": device is not None and g.uuid == device.uuid,
            "vram_used_gb": g.vram_used_gb,
            "vram_total_gb": g.vram_total_gb,
            "vram_free_gb": g.vram_free_gb,
            "vram_giq_gb": attribution.get(g.uuid, {}).get("giq_gb"),
            "vram_other_gb": attribution.get(g.uuid, {}).get("other_gb"),
        }
        for g in cards
    ]

    # Blocked is a fact about a job and its card, not about the default card:
    # a job bound to the second card waits on that card's VRAM however empty
    # the first is, and a job for a model already loaded waits on nothing.
    vram_ok = True
    vram_message = None
    vram_blocked_gpu = None
    if not pause["paused"]:
        loaded = runner.loaded_keys()
        for job in pending:
            worker, model = str(job.request.modality), job.request.model
            if model in loaded:
                continue
            on = device_for_recipe(model)
            need = get_vram_requirement(model)
            need += margin_for(need) + reserve_for(on)
            free = get_free_vram(on)
            if free >= need:
                continue
            card = resolve_device(on)
            vram_ok = False
            vram_blocked_gpu = card.uuid if card else None
            where = f"GPU {card.index} ({card.name})" if card else "the GPU"
            vram_message = (
                f"{worker}/{model} needs {need:.1f}GB on {where}, {free:.1f}GB free — "
                "waiting for residents there to be evicted"
            )
            break

    state = ServiceState.idle
    state_message = None

    if pause["paused"]:
        state = ServiceState.paused
        state_message = "Serving paused — instances stopped, GPUs free; requests get 503"
        if pause["reason"]:
            state_message += f" ({pause['reason']})"
    elif running:
        state = ServiceState.running
        job = running[0]
        state_message = f"Processing {job.request.modality}/{job.request.model}"
    elif runner.active_modality:
        state = ServiceState.ready
        state_message = f"{runner.active_recipe} ({runner.active_modality}) loaded, waiting"
    elif pending:
        if vram_ok:
            state = ServiceState.ready
            state_message = f"{len(pending)} jobs queued, will process shortly"
        else:
            state = ServiceState.blocked
            state_message = f"{len(pending)} jobs waiting for VRAM: {vram_message}"
    else:
        state = ServiceState.idle
        state_message = "No jobs, nothing loaded"

    return ServiceStatus(
        state=state,
        state_message=state_message,
        active_modality=runner.active_modality,
        active_recipe=runner.active_recipe,
        active=runner.active_slots,
        gpu=({"uuid": device.uuid, "index": device.index, "name": device.name} if device else None),
        vram_used_gb=vram.used_gb,
        vram_total_gb=vram.total_gb,
        vram_free_gb=vram.free_gb,
        vram_giq_gb=mine.get("giq_gb"),
        vram_other_gb=mine.get("other_gb"),
        vram_ok=vram_ok,
        vram_message=vram_message,
        vram_blocked_gpu=vram_blocked_gpu,
        gpus=gpus,
        queue_depth=len(pending_ids),
        jobs_pending=pending_ids,
        jobs_running=running_ids,
        paused=pause["paused"],
        paused_since=pause["since"],
        pause_reason=pause["reason"],
        access=access_posture(),
        version=__version__,
        uptime_s=round(time.monotonic() - _STARTED, 1),
        plugins=[asdict(p) for p in plugins.status()],
    )


@router.get("/llm/endpoint")
async def get_llm_endpoint(model: str | None = None) -> dict:
    """Discovery endpoint for LLM clients: where is the llama-server?

    200 {model, base_url} when the model is resident and ready. 503 with a
    state hint while it's loading, evicted for other work, paused, or not yet
    loaded — clients retry; the runner's always-on loop restores residency once
    the queue drains, so this endpoint never triggers loads itself.
    """
    runner = get_runner()
    if model is None:
        # The first LLM kept warm: the one a client asking without a name
        # most plausibly means. Nothing kept warm, nothing to point at.
        from giq.registry import resident_defaults

        model = next(
            (n for n in resident_defaults() if (r := get_recipe(n)) and r.serves("llm")), None
        )
        if model is None:
            raise HTTPException(
                status_code=400, detail="no LLM is kept warm; name one with ?model="
            )
    # Per model, not "any ready LLM": with one llama-server per card, the
    # first ready one can easily be a different model on a different port.
    base_url = runner.llm_base_url_for(model)
    if base_url and not runner.is_paused:
        return {"model": model, "base_url": base_url}
    # A switched-off model is not "loading" — say so, or clients keep probing
    # a server that will never come back.
    if runner.is_disabled(model):
        raise HTTPException(
            status_code=503,
            detail={"state": "disabled", "model": model},
            headers={"Retry-After": "300"},
        )
    state = "paused" if runner.is_paused else "evicted" if runner.active_modality else "loading"
    raise HTTPException(
        status_code=503,
        detail={"state": state, "model": model},
        headers={"Retry-After": "60"} if runner.is_paused else None,
    )


@router.get("/engines")
async def list_engines(refresh: bool = False) -> dict:
    """The runtimes giq executes models with, and which build each one is.

    giq could describe every model file it knew about and nothing about the
    binaries running them — so a deleted llama.cpp source tree could go
    unnoticed for months while its binary kept serving. `version`
    is whatever the engine says about itself; a `present: false` entry is a
    declared engine whose binary is missing, which fails loudly at load.
    """
    from giq.engines import probe_all

    return {"engines": await asyncio.to_thread(probe_all, refresh)}


@router.get("/capabilities", response_model=Capabilities)
async def get_capabilities() -> Capabilities:
    """What this service can run, generated from the recipes.

    Hand-maintained before: it had drifted to omit the audio, embed and stt
    modalities entirely, omit flux_klein under both image workers (the two
    models actually in production since the sd.cpp move), and advertise a tts
    model name that was absent from the VRAM table and therefore unloadable.
    Generating it means a model is discoverable exactly when it is runnable.
    """
    from giq import plugins

    modalities: dict[str, ModalityCapability] = {}
    for modality, spec in plugins.modalities().items():
        for recipe in recipes_serving(modality):
            batch = recipe.max_batch_for(modality)
            cap = modalities.get(modality)
            if cap is None:
                modalities[modality] = ModalityCapability(
                    engine=recipe.engine,
                    recipes=[recipe.name],
                    max_batch=batch,
                    voices=list(recipe.voices) or None,
                    label=spec.label,
                    icon=spec.icon,
                )
                continue
            cap.recipes.append(recipe.name)
            # One modality can span engines (llm runs on llama.cpp or vllm),
            # so report every one in play.
            if recipe.engine not in cap.engine:
                cap.engine = f"{cap.engine}, {recipe.engine}"
            if batch is not None:
                cap.max_batch = max(cap.max_batch or 0, batch)
            if recipe.voices:
                cap.voices = sorted({*(cap.voices or []), *recipe.voices})

    return Capabilities(
        modalities=modalities,
        constraints={
            "max_concurrent_heavy": 1,
            "vram_total_gb": round(get_vram_status().total_gb),
        },
    )
