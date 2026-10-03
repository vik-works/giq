# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""POST /depth: one image in, one 16-bit depth map out. The depth modality's
route, registered by its plugin (ADR-004)."""

import asyncio
import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from giq.api.dependencies import get_orchestrator
from giq.api.router import SYNC_WAIT_TIMEOUT_SECONDS
from giq.api.uploads import file_from, read_body_capped, upload_limit
from giq.models import (
    JobRequest,
    JobStatus,
    Modality,
)
from giq.registry import get_recipe, recipes_serving
from giq.services.orchestration import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


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
    data = await file_from(request, await read_body_capped(request, upload_limit()))
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
