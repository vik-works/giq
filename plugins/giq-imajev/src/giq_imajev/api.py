# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""POST /decide: typed photo+record decisions. The decide modality's route,
registered by its plugin (ADR-004)."""

import asyncio
import base64
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from giq.api.dependencies import get_orchestrator
from giq.api.router import SYNC_WAIT_TIMEOUT_SECONDS
from giq.api.uploads import InMemoryMultipart, read_body_capped, upload_limit
from giq.models import JobRequest, JobStatus
from giq.registry import get_recipe, recipes_serving
from giq.services.orchestration import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Decide ------------------------------------------------------------------

# A decide batch is up to max_batch tasks of up to 8 questions at up to 4
# rotations: seconds, plus whatever the queue holds ahead of it.
DECIDE_WAIT_TIMEOUT_SECONDS = SYNC_WAIT_TIMEOUT_SECONDS


async def _decide_body(request: Request) -> tuple[dict, list[bytes]]:
    """The decide payload: JSON directly, or multipart `request` + `image0..1`.

    Returns the payload and the images as raw bytes either way, so the task
    below can base64 them uniformly.
    """
    body = await read_body_capped(request, upload_limit())
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await InMemoryMultipart(request.headers, body).parse()
        raw = form.get("request")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if hasattr(raw, "read"):
            upload = raw
            raw = (await upload.read()).decode("utf-8", "replace")
            await upload.close()
        if not raw or not isinstance(raw, str):
            raise HTTPException(status_code=400, detail="multipart body needs a 'request' part")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="'request' part is not valid JSON"
            ) from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="'request' part must be a JSON object")
        images = []
        for key in ("image0", "image1", "image", "files"):
            for upload in form.getlist(key):
                if hasattr(upload, "read"):
                    images.append(await upload.read())
                    await upload.close()
        return payload, images
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError:
        raise HTTPException(
            status_code=400, detail="send JSON or multipart with a 'request' part"
        ) from None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="send JSON or multipart with a 'request' part"
        )
    images = []
    for index in (0, 1):
        blob = payload.pop(f"image{index}_b64", None)
        if blob is None:
            continue
        try:
            images.append(base64.b64decode(blob, validate=True))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"image{index}_b64 is not valid base64"
            ) from None
    return payload, images


@router.post("/decide")
async def decide(
    request: Request,
    model: str = Query(default="imajev-2b", description="decide recipe to run"),
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict:
    """Typed decisions over a record plus 0-2 images: Jev's contract, with photos.

    Send `{"state": {...}, "questions": {"queue": {"type": "choice", ...}}}`,
    as JSON (images as `image0_b64`/`image1_b64`) or as multipart `request`
    with `image0..1` parts.
    A convenience over ``POST /run`` with ``modality: "decide"`` — same job,
    same queue, same stats row — for a consumer that has a decision and wants
    answers, not a job id to poll.
    """
    if not ((recipe := get_recipe(model)) and recipe.serves("decide")):
        known = [r.name for r in recipes_serving("decide")]
        raise HTTPException(
            status_code=400, detail=f"unknown decide model {model!r}; one of {known}"
        )
    payload, images = await _decide_body(request)
    if not isinstance(payload.get("questions"), dict) or not payload["questions"]:
        raise HTTPException(status_code=400, detail="payload needs a non-empty 'questions' object")
    if len(images) > 2:
        raise HTTPException(status_code=400, detail="at most 2 images per decide request")
    task = {
        "id": "decide-0",
        "state": payload.get("state", {}),
        "questions": payload["questions"],
        "images_b64": [base64.b64encode(data).decode() for data in images],
        "rotations": payload.get("rotations", 1),
        "calibration": payload.get("calibration"),
    }
    job_id, _ = await orch.submit_job(JobRequest(modality="decide", model=model, tasks=[task]))
    logger.info(f"decide job {job_id}: {model}, {len(images)} images")
    try:
        job = await orch.wait_for_job(job_id, timeout=DECIDE_WAIT_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        await orch.cancel_job(job_id)
        raise
    if job.status != JobStatus.completed or not job.results:
        raise HTTPException(status_code=500, detail=job.error or "decision failed")
    result = job.results[0]
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"decision failed: {result['error']}")
    out: dict = {k: result.get(k) for k in ("answers", "tokens_in", "images", "rotations")}
    out["job_id"] = job_id
    return out
