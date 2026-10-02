# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for giq (GPU Inference Queue)."""

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class Modality(StrEnum):
    """The kind of job: it picks the endpoint and the task shape (ADR-003).

    Not the engine adapter that runs it, and not the recipe: a recipe serves
    one or more modalities.
    """

    llm = "llm"
    text2image = "text2image"
    image_edit = "image_edit"
    tts = "tts"
    stt = "stt"
    # Resident audio stack: faster-whisper ASR +
    # pyannote diarization, and ECAPA speaker embeddings, each in its own
    # child process under giq's own venv.
    audio = "audio"
    embed = "embed"
    # Document parsing (baidu/Unlimited-OCR in a child process): PDF or page
    # images in, layout-tagged text and HTML out. See giq.ocrdoc.
    ocr = "ocr"
    # Monocular depth (Depth Anything V2 in a child process): one RGB image
    # in, a 16-bit depth map at the input resolution out.
    depth = "depth"


class JobStatus(StrEnum):
    """Job execution status."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# --- LLM Tasks ---


class LLMTask(BaseModel):
    """LLM completion task."""

    id: str
    system: str | None = None
    user: str
    params: dict[str, Any] | None = None


# Backward compatibility alias
Task = LLMTask


# --- Image Tasks ---


class Text2ImageTask(BaseModel):
    """Text-to-image generation task."""

    id: str
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None  # Optional seed for reproducibility


class ImageEditTask(BaseModel):
    """Image edit/style transfer task."""

    id: str
    reference_image_b64: str  # Base64 encoded reference image
    instruction: str
    negative_prompt: str | None = None


# --- Job Request (unified) ---


class JobRequest(BaseModel):
    """Request to submit a new job.

    Task format depends on the modality:
    - llm: tasks are LLMTask dicts
    - text2image: tasks are Text2ImageTask dicts
    - image_edit: tasks are ImageEditTask dicts
    - ocr: {id, pdf_b64 | images_b64[], dpi?, pages?, raw?, strip?, merge?}
    - depth: {id, image_b64, visualize?}
    """

    # `worker` is the name before ADR-003, still accepted on input so existing
    # clients keep working; responses and logs say `modality`.
    modality: Modality = Field(validation_alias=AliasChoices("modality", "worker"))
    model: str
    model_path: str | None = None  # Override default model path
    params: dict[str, Any] | None = None  # Worker-level params
    tasks: list[dict[str, Any]] = Field(default_factory=list)  # Task dicts (validated per modality)
    chat_request: dict[str, Any] | None = (
        None  # Raw OpenAI chat completion body (bypasses run_batch)
    )


class JobResponse(BaseModel):
    """Response after submitting a job."""

    job_id: str
    position: int


# --- Job Results ---


class LLMResult(BaseModel):
    """Result of LLM task."""

    id: str
    output: str
    tokens: int | None = None  # total (kept for old clients; = in + out)
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None


class ImageResult(BaseModel):
    """Result of image generation task."""

    id: str
    image_b64: str | None = None  # Base64 encoded PNG
    seed: int | None = None
    error: str | None = None


class OCRResult(BaseModel):
    """Result of an OCR task: the document, not the page images."""

    id: str
    html: str = ""  # fragment: furniture stripped, page breaks merged
    pages: int = 0
    blocks: list[dict[str, Any]] = Field(default_factory=list)  # label, bbox, page, content
    raw: str | None = None  # the model's tagged text, only when the task asked
    tokens_in: int | None = None  # prompt tokens: ~257 per page plus the prompt
    tokens_out: int | None = None
    truncated: bool = False  # a pass hit the context ceiling; output may be short
    error: str | None = None


class DepthResult(BaseModel):
    """Result of a depth task: one map per image, at the image's resolution.

    ``depth_b64`` is a 16-bit grayscale PNG. Its 0..65535 is the prediction's
    ``depth_min``..``depth_max`` mapped linearly, so a consumer that wants the
    model's own values recovers them from the three fields. What those values
    mean is the model's affair: Depth Anything V2 predicts *relative inverse*
    depth (larger is nearer, no unit), so ``metric`` is False and the map is
    a disparity map. A metric checkpoint would set it True and mean metres.
    """

    id: str
    depth_b64: str | None = None
    width: int = 0
    height: int = 0
    depth_min: float | None = None
    depth_max: float | None = None
    metric: bool = False
    # 8-bit colour-mapped PNG (near red, far blue), only when the task asked.
    visualization_b64: str | None = None
    error: str | None = None


# Backward compatibility - JobResult was the old name for LLMResult
JobResult = LLMResult


class JobStatusResponse(BaseModel):
    """Full job status with results."""

    job_id: str
    status: JobStatus
    modality: Modality
    model: str
    results: list[dict[str, Any]] | None = None  # Polymorphic results
    duration_ms: int | None = None


class ServiceState(StrEnum):
    """Overall service state."""

    idle = "idle"  # No work, nothing loaded
    ready = "ready"  # An on-demand instance up, waiting for work
    running = "running"  # Actively processing a job
    blocked = "blocked"  # Jobs queued but can't run (VRAM)
    paused = "paused"  # Serving suspended by an operator; GPU handed back
    error = "error"  # Something went wrong


class ServiceStatus(BaseModel):
    """Current service status."""

    # Overall state
    state: ServiceState
    state_message: str | None = None  # Human-readable explanation

    # Active worker info. The scalars report one loaded worker for
    # back-compat; `active` lists every card's slot with its binding.
    active_modality: Modality | None = None
    active_recipe: str | None = None
    active: list[dict[str, Any]] = []

    # VRAM status. These scalars describe ONE card — the default device,
    # named in `gpu` below — and stay for the clients that read them. On a
    # multi-card rig the machine total is not this, and reading it as such is
    # how the two numbers drifted apart in the first place; `gpus` below has
    # every card.
    gpu: dict[str, Any] | None = None
    vram_used_gb: float
    vram_total_gb: float
    vram_free_gb: float
    # How the used figure splits: models giq is holding vs everything else on
    # the card (desktop, other CUDA apps). None when it can't be measured.
    vram_giq_gb: float | None = None
    vram_other_gb: float | None = None
    # False while a pending job waits for room on its own card — judged per
    # job and per card, since a job only ever waits on the card it is bound
    # to; `vram_blocked_gpu` names that card (a `gpus` uuid).
    vram_ok: bool
    vram_message: str | None = None  # Explanation if not OK
    vram_blocked_gpu: str | None = None
    # Every card, in the terms of the one-card fields above (`selected` marks
    # that one): uuid, index, name, vram_used/total/free_gb, vram_giq_gb,
    # vram_other_gb. /gpus adds temperature, power and the per-process split.
    gpus: list[dict[str, Any]] = []

    # Queue info
    queue_depth: int
    jobs_pending: list[str]
    jobs_running: list[str] = []

    # Pause (operator handed the GPU back; submissions get 503)
    # How reachable giq is and whether a token guards it — surfaced so the
    # dashboard can say so on screen, not just in the boot log nobody reads.
    access: dict | None = None
    paused: bool = False
    paused_since: str | None = None
    pause_reason: str | None = None

    # Which build is answering and since when, so the dashboard can say so
    # without a second endpoint.
    version: str = ""
    uptime_s: float = 0.0


class PauseRequest(BaseModel):
    """Body for POST /control/pause."""

    force: bool = False  # skip the drain wait and unload now
    reason: str | None = None


class PauseResponse(BaseModel):
    """Result of a pause/resume control call."""

    paused: bool
    since: str | None = None
    reason: str | None = None
    forced: bool = False
    drained: bool = True  # False if in-flight work outlasted the drain window
    warnings: list[str] = []
    vram_free_gb: float
    vram_total_gb: float


class ModalityCapability(BaseModel):
    """What giq serves for one modality."""

    # Every engine in play for it, comma-separated (ocr runs on two).
    engine: str
    recipes: list[str]
    max_batch: int | None = None
    voices: list[str] | None = None  # TTS only


class Capabilities(BaseModel):
    """Full service capabilities."""

    modalities: dict[Modality, ModalityCapability]
    constraints: dict[str, Any]
