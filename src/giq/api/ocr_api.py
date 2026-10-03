# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""POST /ocr: one PDF in, one HTML document out. The ocr modality's route,
registered by its plugin (ADR-004)."""

import asyncio
import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse

from giq.api.dependencies import get_orchestrator
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


# --- OCR ---------------------------------------------------------------------

# One document is one job, and a dozen-page pass runs minutes at ~80 tok/s.
OCR_WAIT_TIMEOUT_SECONDS = 3600.0


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
    data = await file_from(request, await read_body_capped(request, limit))
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
