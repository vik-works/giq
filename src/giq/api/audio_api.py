# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The OpenAI-compatible audio routes: speech, transcription, voiceprints.
The speech plugin's routes (ADR-004); each takes the recipe a request
names, defaulting to the one it served before it could be chosen."""

import asyncio
import base64
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from giq.api.dependencies import get_audio_cache, get_orchestrator
from giq.models import JobRequest
from giq.queue import Job
from giq.registry import get_recipe
from giq.services.audio_cache import AudioCache
from giq.services.orchestration import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


class TTSRequest(BaseModel):
    """OpenAI TTS request format."""

    model: str = "tts-1"
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="alloy", description="Voice to use")
    language: str | None = Field(default=None, description="Language")
    response_format: str = Field(default="wav", description="Audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class RenderResponse(BaseModel):
    """Response for audio render endpoint."""

    url: str
    file_id: str
    size_bytes: int
    expires_in: int


def _recipe_for(model: str, modality: str, default: str) -> str:
    """The recipe a request's ``model`` names, or ``default``.

    OpenAI clients send OpenAI's own names (``tts-1``, ``whisper-1``), which
    are no recipe of ours; those, like any name the catalog does not know,
    get the default. A recipe that exists but does not serve this modality
    is a mistake worth a 400 rather than a silent substitute.
    """
    recipe = get_recipe(model)
    if recipe is None:
        return default
    if not recipe.serves(modality):
        raise HTTPException(status_code=400, detail=f"recipe {model!r} does not serve {modality}")
    return recipe.name


async def _run_tts_job(orch: Orchestrator, request: "TTSRequest", timeout: float) -> bytes:
    """Submit a TTS job, wait for it, and return decoded audio bytes."""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required")

    job_id, _ = await orch.submit_job(
        JobRequest(
            modality="tts",
            model=_recipe_for(request.model, "tts", "kokoro"),
            tasks=[
                {
                    "id": "tts-0",
                    "text": request.input,
                    "voice": request.voice,
                    "language": request.language,
                }
            ],
        )
    )
    logger.info(f"TTS job {job_id}: {len(request.input)} chars, voice={request.voice}")
    completed_job = await orch.wait_for_job(job_id, timeout=timeout)

    if not completed_job.results:
        raise HTTPException(status_code=500, detail="No audio generated")

    result = completed_job.results[0]
    if isinstance(result, dict):
        audio_b64 = result.get("output", "")
        error = result.get("error")
    else:
        audio_b64 = result.output
        error = result.error

    if error:
        raise HTTPException(status_code=500, detail=f"TTS failed: {error}")
    if not audio_b64:
        raise HTTPException(status_code=500, detail="No audio generated")
    return base64.b64decode(audio_b64)


@router.post("/audio/speech")
async def create_speech(
    request: TTSRequest, orch: Orchestrator = Depends(get_orchestrator)
) -> Response:
    """Generate speech from text (OpenAI-compatible endpoint)."""
    audio_bytes = await _run_tts_job(orch, request, timeout=60.0)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@router.get("/audio/speech")
async def create_speech_get(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query(default="alloy"),
    model: str = Query(default="kokoro", description="TTS recipe"),
    language: str = Query(default=None),
    orch: Orchestrator = Depends(get_orchestrator),
) -> Response:
    """Generate speech via GET (for ESP32 streaming)."""
    request = TTSRequest(input=text, voice=voice, model=model, language=language)
    return await create_speech(request, orch=orch)


@router.post("/audio/render", response_model=RenderResponse)
async def render_speech(
    request: TTSRequest,
    ttl: int = Query(default=300, ge=60, le=3600, description="Cache TTL in seconds"),
    orch: Orchestrator = Depends(get_orchestrator),
    cache: AudioCache = Depends(get_audio_cache),
) -> RenderResponse:
    """Pre-render TTS and return a streaming URL (for long narrations)."""
    await cache.cleanup()
    # Scale timeout with text length: ~50 chars/sec floor.
    timeout = max(60.0, len(request.input) / 50)
    audio_bytes = await _run_tts_job(orch, request, timeout=timeout)

    file_id = str(uuid.uuid4())[:8]
    cache.set(file_id, audio_bytes, expires_in=ttl)
    logger.info(f"Audio cache: stored {file_id} ({len(audio_bytes)} bytes, expires in {ttl}s)")
    return RenderResponse(
        url=f"/v1/audio/files/{file_id}.wav",
        file_id=file_id,
        size_bytes=len(audio_bytes),
        expires_in=ttl,
    )


@router.get("/audio/files/{file_id}.wav")
async def get_audio_file(file_id: str, cache: AudioCache = Depends(get_audio_cache)) -> Response:
    """Stream pre-rendered audio file (used by ESP32 for long narrations)."""
    info = cache.get(file_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Audio not found or expired")
    logger.info(f"Audio cache: serving {file_id} ({len(info['data'])} bytes)")
    return Response(
        content=info["data"],
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{file_id}.wav"',
            "Cache-Control": "no-cache",
        },
    )


@router.delete("/audio/files/{file_id}")
async def delete_audio_file(file_id: str, cache: AudioCache = Depends(get_audio_cache)) -> dict:
    """Delete pre-rendered audio file (optional cleanup)."""
    if file_id not in cache:
        raise HTTPException(status_code=404, detail="Audio not found")
    cache.delete(file_id)
    return {"status": "deleted", "file_id": file_id}


# Transcription/embedding jobs can queue behind a multi-minute image batch;
# the sync wait must absorb queue time + model runtime. Stay under 900s, a
# common client-side budget for batch transcription. Live callers time out client-side much
# earlier; their disconnect cancels the pending job (see _wait_cancelling).
AUDIO_WAIT_TIMEOUT_SECONDS = 880.0


async def _wait_cancelling(orch: Orchestrator, job_id: str) -> "Job":
    """wait_for_job, but client disconnect cancels a still-pending job.

    Live audio chunks arrive every ~5s; during an image batch each caller
    gives up in 30-60s. Without cancellation every abandoned chunk stays
    queued and gets pointlessly transcribed after the batch drains.
    """
    try:
        return await orch.wait_for_job(job_id, timeout=AUDIO_WAIT_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        await orch.cancel_job(job_id)
        raise


@router.post("/audio/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(...),  # noqa: B008 (FastAPI dependency-injection idiom)
    model: str = Form(default="whisper-1"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    diarize: bool = Query(default=True),
    orch: Orchestrator = Depends(get_orchestrator),
) -> Response | dict:
    """Transcribe (+diarize) audio.

    Runs on the recipe ``model`` names, whisper-large-v3 (faster-whisper +
    pyannote) for any name that is not a recipe. response_format: json
    (default), verbose_json (per-segment language + words), text.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    audio_b64 = base64.b64encode(audio_bytes).decode()
    job_id, _ = await orch.submit_job(
        JobRequest(
            modality="audio",
            model=_recipe_for(model, "audio", "whisper-large-v3"),
            tasks=[
                {
                    "id": "asr-0",
                    "audio_b64": audio_b64,
                    "language": language,
                    "diarize": diarize,
                }
            ],
        )
    )
    logger.info(f"audio job {job_id}: {len(audio_bytes)} bytes (diarize={diarize})")
    completed_job = await _wait_cancelling(orch, job_id)

    if not completed_job.results:
        raise HTTPException(status_code=500, detail="No transcription generated")
    result = completed_job.results[0]
    if result.get("error"):
        raise HTTPException(status_code=500, detail=f"Transcription failed: {result['error']}")

    if response_format == "text":
        return PlainTextResponse(result.get("text", ""))
    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": result.get("language"),
            "duration": result.get("duration"),
            "text": result.get("text", ""),
            "speakers": result.get("speakers", []),
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "language": result.get("language"),
                    "speaker": s.get("speaker"),
                    "words": s.get("words", []),
                }
                for s in result.get("segments", [])
            ],
        }
    return {
        "text": result.get("text", ""),
        "language": result.get("language"),
        "duration": result.get("duration"),
        "speakers": result.get("speakers", []),
        "segments": [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "speaker": s.get("speaker"),
            }
            for s in result.get("segments", [])
        ],
    }


@router.post("/audio/embeddings")
async def create_audio_embedding(
    file: UploadFile = File(...),  # noqa: B008 (FastAPI dependency-injection idiom)
    model: str = Form(default="ecapa-tdnn"),
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict:
    """Speaker voiceprint: clip → L2-normalized vector."""
    recipe = _recipe_for(model, "embed", "ecapa-tdnn")
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    audio_b64 = base64.b64encode(audio_bytes).decode()
    job_id, _ = await orch.submit_job(
        JobRequest(
            modality="embed",
            model=recipe,
            tasks=[{"id": "emb-0", "audio_b64": audio_b64}],
        )
    )
    completed_job = await _wait_cancelling(orch, job_id)

    if not completed_job.results:
        raise HTTPException(status_code=500, detail="No embedding generated")
    result = completed_job.results[0]
    if result.get("error"):
        # Undecodable clip is a client error: 400.
        raise HTTPException(status_code=400, detail=result["error"])
    return {
        "embedding": result["embedding"],
        "dim": result["dim"],
        "model": recipe,
        "normalized": True,
    }
