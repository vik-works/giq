# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for giq (GPU Inference Queue)."""

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator


class Modality(StrEnum):
    """The names of the modalities giq ships (ADR-003), for code that means one.

    The kind of job: it picks the endpoint and the task shape. Not the
    engine adapter that runs it, and not the recipe: a recipe serves one or
    more modalities. This is not the set giq accepts: that is the plugin
    registry's (ADR-004), which a plugin extends with modalities of its own.
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
    # images in, layout-tagged text and HTML out. See giq_ocr.ocrdoc.
    ocr = "ocr"
    # Monocular depth (Depth Anything V2 in a child process): one RGB image
    # in, a 16-bit depth map at the input resolution out.
    depth = "depth"
    # Typed decisions (imajev in a child process on its own interpreter):
    # a record plus 0-2 images and 1-8 typed questions in, calibrated
    # probabilities with an explicit unknown out. See DecideResult.
    decide = "decide"


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
    - decide: {id, state, questions, images_b64[]?, rotations?, calibration?}
    """

    # `worker` is the name before ADR-003, still accepted on input so existing
    # clients keep working; responses and logs say `modality`.
    modality: str = Field(validation_alias=AliasChoices("modality", "worker"))
    model: str
    model_path: str | None = None  # Override default model path
    params: dict[str, Any] | None = None  # Worker-level params
    tasks: list[dict[str, Any]] = Field(default_factory=list)  # Task dicts (validated per modality)
    chat_request: dict[str, Any] | None = (
        None  # Raw OpenAI chat completion body (bypasses run_batch)
    )

    @field_validator("modality")
    @classmethod
    def _registered(cls, v: str) -> str:
        from giq import plugins

        known = plugins.modalities()
        if v not in known:
            raise ValueError(
                f"unknown modality {v!r} (known: {', '.join(sorted(known))}; a plugin adds others)"
            )
        return v


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


class DecideAnswer(BaseModel):
    """One typed answer: Jev's shape plus the trained unknown."""

    type: str
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    labels: list[str] | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] | None = None
    threshold: float | None = None
    confidence: float | None = None
    unknown_probability: float = 0.0
    unknown_probabilities: dict[str, float] | None = None
    abstained: bool = False
    calibration_version: str | None = None


class DecideResult(BaseModel):
    """Result of a decide task: the model's typed answers, one per question."""

    id: str
    answers: dict[str, DecideAnswer] = Field(default_factory=dict)
    tokens_in: int | None = None
    images: int | None = None
    rotations: int | None = None
    error: str | None = None


# Backward compatibility - JobResult was the old name for LLMResult
JobResult = LLMResult


class JobStatusResponse(BaseModel):
    """Full job status with results."""

    job_id: str
    status: JobStatus
    modality: str
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
    active_modality: str | None = None
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
    # Every plugin found (ADR-004): what it registered, or why it was refused.
    plugins: list[dict[str, Any]] = []


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


class AvailableRecipe(BaseModel):
    """A recipe this machine could have for a modality, and what it takes."""

    name: str
    # fetchable | manual (ADR-005); unfit recipes are not offered.
    availability: str
    # The one line that matters: what stands between it and running.
    verdict: str


class ModalityCapability(BaseModel):
    """What giq serves for one modality (ADR-005 D7)."""

    # Every engine of the ready recipes, comma-separated (llm runs on two).
    engine: str
    # The recipes that run here now, preferred first: kept warm first, then
    # on demand by name. What a client may pass as `model`.
    recipes: list[str]
    # The recipe a request naming none runs on: the first of `recipes`.
    default: str | None = None
    # What it could have: recipes whose weights are not here yet.
    available: list[AvailableRecipe] = []
    max_batch: int | None = None
    voices: list[str] | None = None  # TTS only
    # How the dashboard names and draws it, as its plugin registered it.
    label: str = ""
    icon: str = ""


class Capabilities(BaseModel):
    """Full service capabilities."""

    modalities: dict[str, ModalityCapability]
    constraints: dict[str, Any]
    # Plugins' dashboard UIs: each manifest, with the `base` URL its files
    # are served under (ADR-004 D6).
    ui: list[dict[str, Any]] = []
