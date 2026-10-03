# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Request bodies held in memory under a size cap, for the routes that take
a document or an image (the OCR and depth plugins'): read once, never
spooled to disk."""

import os

from fastapi import HTTPException, Request
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartParser


def upload_limit() -> int:
    """Largest body accepted by ``/ocr`` and ``/depth``, in bytes. The hard
    ceiling is the parent→child pipe: a task is one JSON line, base64 inflates
    by a third, and the line limit is 128 MiB — so 64 MB is the most a PDF or
    an image can be and still fit. The env name predates ``/depth``."""
    return int(os.environ.get("GIQ_OCR_MAX_UPLOAD_MB", "64")) * 1024 * 1024


class InMemoryMultipart(MultiPartParser):
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


async def read_body_capped(request: Request, limit: int) -> bytes:
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


async def file_from(request: Request, body: bytes) -> bytes:
    """The uploaded bytes: the body itself, or the ``file`` part of a multipart body."""
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        return body
    form = await InMemoryMultipart(request.headers, body).parse()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="multipart body needs a 'file' part")
    data = await upload.read()
    await upload.close()
    return data
