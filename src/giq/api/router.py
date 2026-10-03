# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartParser

from giq import __version__
from giq.api.access import posture as access_posture
from giq.api.dependencies import get_orchestrator
from giq.gpus import attribute_vram, get_gpus, resolve_device, selected_device
from giq.models import (
    Capabilities,
    JobRequest,
    JobResponse,
    JobStatus,
    JobStatusResponse,
    Modality,
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
    )


@router.get("/llm/endpoint")
async def get_llm_endpoint(model: str = "gemma-4-12b") -> dict:
    """Discovery endpoint for LLM clients: where is the llama-server?

    200 {model, base_url} when the model is resident and ready. 503 with a
    state hint while it's loading, evicted for other work, paused, or not yet
    loaded — clients retry; the runner's always-on loop restores residency once
    the queue drains, so this endpoint never triggers loads itself.
    """
    runner = get_runner()
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


# --- OCR ---------------------------------------------------------------------

# One document is one job, and a dozen-page pass runs minutes at ~80 tok/s.
OCR_WAIT_TIMEOUT_SECONDS = 3600.0


def upload_limit() -> int:
    """Largest body accepted by ``/ocr`` and ``/depth``, in bytes. The hard
    ceiling is the parent→child pipe: a task is one JSON line, base64 inflates
    by a third, and the line limit is 128 MiB — so 64 MB is the most a PDF or
    an image can be and still fit. The env name predates ``/depth``."""
    return int(os.environ.get("GIQ_OCR_MAX_UPLOAD_MB", "64")) * 1024 * 1024


class _InMemoryMultipart(MultiPartParser):
    """Starlette's parser spools a file part over 1 MB to a temp file on disk.

    A document service holds the document in memory: the body has already
    been read under ``upload_limit()``, so a spool threshold above it
    means nothing ever rolls over to ``/tmp`` — a plain ext4 volume here,
    where a deleted file is still a forensic artifact."""

    def __init__(self, headers: Headers, body: bytes) -> None:
        async def one_chunk():
            yield body

        super().__init__(headers, one_chunk())
        self.spool_max_size = len(body) + 1


async def _read_body_capped(request: Request, limit: int) -> bytes:
    """The whole body, in memory, or 413 before the cap is exceeded."""
    declared = request.headers.get("content-length", "")
    too_big = HTTPException(
        status_code=413,
        detail=f"document exceeds the {limit // (1024 * 1024)} MB limit "
        "(GIQ_OCR_MAX_UPLOAD_MB on the server)",
    )
    if declared.isdigit() and int(declared) > limit:
        raise too_big
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_big
        chunks.append(chunk)
    return b"".join(chunks)


async def _document_from(request: Request, body: bytes) -> bytes:
    """The PDF bytes: the body itself, or the ``file`` part of a multipart body."""
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        return body
    form = await _InMemoryMultipart(request.headers, body).parse()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="multipart body needs a 'file' part")
    data = await upload.read()
    await upload.close()
    return data


def parse_pages(spec: str | None) -> list[int] | None:
    """``"1-3,7"`` → ``[1, 2, 3, 7]``; None or empty means every page."""
    if not spec or not spec.strip():
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            a = int(lo)
            b = int(hi) if hi else a
        except ValueError:
            raise HTTPException(status_code=400, detail=f"bad page spec {part!r}") from None
        if a < 1 or b < a:
            raise HTTPException(status_code=400, detail=f"bad page range {part!r}")
        pages.extend(range(a, b + 1))
    return pages


@router.post(
    "/ocr",
    response_model=None,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                },
            },
        }
    },
)
async def ocr_document(
    request: Request,
    model: str = Query(default="unlimited-ocr", description="unlimited-ocr | glm-ocr"),
    dpi: int = Query(default=200, ge=50, le=400),
    pages: str | None = Query(default=None, description="e.g. 1-3,7; default all"),
    raw: bool = Query(default=False, description="add the model's tagged text"),
    strip: bool = Query(default=True, description="drop headers, footers, page numbers"),
    merge: bool = Query(default=True, description="re-join tables/paragraphs across pages"),
    response_format: str = Query(default="json", pattern="^(json|html)$"),
    orch: Orchestrator = Depends(get_orchestrator),
) -> Response | dict:
    """One PDF in, one document out: headers, footers and page numbers
    stripped, tables and paragraphs re-joined across page breaks, as HTML.

    Send the PDF as the request body (``Content-Type: application/pdf``) or
    as the ``file`` part of a multipart form; options are query parameters
    either way. ``model`` picks the engine: ``unlimited-ocr`` (one pass over
    many pages) or ``glm-ocr`` (layout detection, then each region read with
    the prompt for its kind — the stronger choice for tables). The document
    is held in memory only, never spooled to disk, and refused with 413 above
    the server's size limit.

    A convenience over ``POST /run`` with ``worker: "ocr"`` — same job, same
    queue, same stats row — for a consumer that has a file and wants a
    document, not a job id to poll. Page images go through ``/run`` with
    ``images_b64``. ``response_format=html`` returns the fragment itself.
    """
    if not ((recipe := get_recipe(model)) and recipe.serves(Modality.ocr)):
        known = [r.name for r in recipes_serving(Modality.ocr)]
        raise HTTPException(status_code=400, detail=f"unknown OCR model {model!r}; one of {known}")
    limit = upload_limit()
    data = await _document_from(request, await _read_body_capped(request, limit))
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="not a PDF")
    task = {
        "id": "ocr-0",
        "pdf_b64": base64.b64encode(data).decode(),
        "dpi": dpi,
        "raw": raw,
        "strip": strip,
        "merge": merge,
    }
    if page_list := parse_pages(pages):
        task["pages"] = page_list
    job_id, _ = await orch.submit_job(JobRequest(modality=Modality.ocr, model=model, tasks=[task]))
    logger.info(f"ocr job {job_id}: {model}, {len(data)} bytes, dpi={dpi}")
    try:
        job = await orch.wait_for_job(job_id, timeout=OCR_WAIT_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        # Client gone: do not spend minutes of GPU on a document nobody
        # will collect.
        await orch.cancel_job(job_id)
        raise
    if job.status != JobStatus.completed or not job.results:
        raise HTTPException(status_code=500, detail=job.error or "OCR failed")
    result = job.results[0]
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"OCR failed: {result['error']}")
    if response_format == "html":
        return HTMLResponse(result.get("html", ""))
    keys = ("html", "pages", "blocks", "tokens_in", "tokens_out", "truncated")
    out: dict = {k: result.get(k) for k in keys}
    out["job_id"] = job_id
    if raw:
        out["raw"] = result.get("raw")
    return out


# --- Depth -------------------------------------------------------------------

# What the body may be. Anything else is refused before it costs a job.
_IMAGE_MAGIC = ((b"\x89PNG", "png"), (b"\xff\xd8", "jpeg"), (b"RIFF", "webp"))


def _is_image(data: bytes) -> bool:
    for magic, kind in _IMAGE_MAGIC:
        if data.startswith(magic):
            return kind != "webp" or data[8:12] == b"WEBP"
    return False


@router.post(
    "/depth",
    response_model=None,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "image/png": {"schema": {"type": "string", "format": "binary"}},
                "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                "image/webp": {"schema": {"type": "string", "format": "binary"}},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                },
            },
        }
    },
)
async def depth_map(
    request: Request,
    model: str = Query(default="depth-anything-v2-small", description="depth-anything-v2-small"),
    visualize: bool = Query(default=False, description="add a colour-mapped PNG (near red)"),
    response_format: str = Query(default="json", pattern="^(json|png|visualization)$"),
    orch: Orchestrator = Depends(get_orchestrator),
) -> Response | dict:
    """One image in, one depth map out, at the image's own resolution.

    Send a PNG, JPEG or WebP as the request body or as the ``file`` part of a
    multipart form. The map is a 16-bit PNG whose 0..65535 spans
    ``depth_min``..``depth_max`` of the model's prediction — for Depth
    Anything V2 that is relative inverse depth, larger nearer, no unit (see
    ``DepthResult``). ``response_format=png`` returns that PNG itself;
    ``visualization`` returns the colour-mapped one, near red and far blue.

    A convenience over ``POST /run`` with ``worker: "depth"`` — same job,
    same queue, same stats row — for a consumer that has an image and wants
    a map, not a job id to poll.
    """
    if not ((recipe := get_recipe(model)) and recipe.serves(Modality.depth)):
        known = [r.name for r in recipes_serving(Modality.depth)]
        raise HTTPException(
            status_code=400, detail=f"unknown depth model {model!r}; one of {known}"
        )
    data = await _document_from(request, await _read_body_capped(request, upload_limit()))
    if not _is_image(data):
        raise HTTPException(status_code=400, detail="not a PNG, JPEG or WebP image")
    want_vis = visualize or response_format == "visualization"
    task = {"id": "depth-0", "image_b64": base64.b64encode(data).decode(), "visualize": want_vis}
    job_id, _ = await orch.submit_job(
        JobRequest(modality=Modality.depth, model=model, tasks=[task])
    )
    logger.info(f"depth job {job_id}: {model}, {len(data)} bytes")
    try:
        job = await orch.wait_for_job(job_id, timeout=SYNC_WAIT_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        await orch.cancel_job(job_id)
        raise
    if job.status != JobStatus.completed or not job.results:
        raise HTTPException(status_code=500, detail=job.error or "depth estimation failed")
    result = job.results[0]
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"depth estimation failed: {result['error']}")
    if response_format == "png":
        return Response(base64.b64decode(result["depth_b64"]), media_type="image/png")
    if response_format == "visualization":
        return Response(base64.b64decode(result["visualization_b64"]), media_type="image/png")
    keys = ("depth_b64", "width", "height", "depth_min", "depth_max", "metric")
    out: dict = {k: result.get(k) for k in keys}
    if want_vis:
        out["visualization_b64"] = result.get("visualization_b64")
    out["job_id"] = job_id
    return out
