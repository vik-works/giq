# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Job runner - processes jobs from queue using adapters."""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from time import monotonic
from typing import Any, Protocol

from giq.adapters.engine import ServedLLM, context_size
from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig
from giq.adapters.stt import SttAdapter
from giq.models import ImageResult, JobRequest, JobStatus, LLMResult, Modality
from giq.paths import inflight_log
from giq.privacy import safe_extra
from giq.queue import Job, JobQueue, get_queue
from giq.recipes.schema import VllmParams
from giq.registry import get_recipe, lane_width_for
from giq.vram import get_free_vram, wait_for_vram

logger = logging.getLogger(__name__)

# O_SYNC'd in-flight job log for post-mortem after hard power cuts.
# Each job writes a "start" record and an "end" record; after a crash, any
# "start" without a matching "end" = the job that was running at the cut.
#
# It records the SHAPE of a request and never its content: a local inference
# service should not keep every prompt, system message and chat body in
# plaintext. Identifying which job was in flight needs a job id, a model and a
# size; it never needs the text. See _request_shape.
#
# GIQ_INFLIGHT_LOG, default <state dir>/inflight.log (next to stats.db, which
# backfills its job history from this file on first start). Resolved on first
# write rather than at import; tests pin it by setting this.
_INFLIGHT_LOG_PATH: str | None = None
_inflight_fd: int | None = None


def _get_inflight_fd() -> int:
    global _inflight_fd
    if _inflight_fd is None:
        path = _INFLIGHT_LOG_PATH or str(inflight_log())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _inflight_fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_SYNC,
            0o644,
        )
    return _inflight_fd


def _measure(obj: Any, seen: dict[str, int]) -> None:
    """Walk a request counting characters and images, keeping no text.

    Deliberately additive: an unfamiliar field contributes to a character
    count and nothing else, so a future field carrying user text cannot start
    leaking just because nobody remembered to add it to a redaction list.
    """
    if isinstance(obj, str):
        seen["chars"] += len(obj)
    elif isinstance(obj, dict):
        if obj.get("type") in ("image_url", "input_audio") or any(
            k in obj for k in ("reference_image_b64", "image_b64", "audio_b64", "pdf_b64")
        ):
            seen["blobs"] += 1
        if isinstance(obj.get("images_b64"), list):
            seen["blobs"] += len(obj["images_b64"])
        for v in obj.values():
            _measure(v, seen)
    elif isinstance(obj, list):
        for v in obj:
            _measure(v, seen)


def _request_shape(request: Any) -> dict[str, Any]:
    """How big a request was, never what it said.

    Enough to identify a job in a post-mortem — which model, how many tasks,
    roughly how much input — with nothing in it that belongs to whoever asked.
    """
    seen = {"chars": 0, "blobs": 0}
    _measure(request.tasks, seen)
    _measure(request.chat_request, seen)
    shape: dict[str, Any] = {"input_chars": seen["chars"]}
    if seen["blobs"]:
        shape["attachments"] = seen["blobs"]
    if request.tasks:
        shape["tasks"] = len(request.tasks)
    if request.chat_request:
        shape["messages"] = len(request.chat_request.get("messages") or [])
        for k in ("max_tokens", "temperature", "top_p", "stream"):
            if request.chat_request.get(k) is not None:
                shape[k] = request.chat_request[k]
        if request.chat_request.get("tools"):
            shape["tools"] = len(request.chat_request["tools"])
    # Numbers and flags only: a param whose value is a string could be text
    # someone typed, and none of the ones worth logging are strings.
    if request.params:
        for k, v in request.params.items():
            if isinstance(v, bool | int | float):
                shape[k] = v
    return shape


def _log_inflight(event: str, job: Job, **extra: Any) -> None:
    try:
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            "job_id": job.job_id,
            "modality": str(job.request.modality),
            "recipe": job.request.model,
        }
        # Everything above is giq's own vocabulary. Everything below came from
        # a caller, so it goes through the same check the shape implies: a
        # string may be written unless it quotes the request.
        record.update(safe_extra(extra, job.request))
        if event == "start":
            record["shape"] = _request_shape(job.request)
        line = (json.dumps(record, default=str, ensure_ascii=False) + "\n").encode("utf-8")
        os.write(_get_inflight_fd(), line)
    except Exception as e:
        logger.warning(f"inflight log write failed: {e}")


# How long to keep a model warm after last job
WARM_TIMEOUT_SECONDS = 120

# Max time a single job can run before being killed
JOB_TIMEOUT_SECONDS = 300

# Per-adapter-type overrides. Audio: diarizing an hours-long recording takes
# minutes of GPU time; give it a full batch budget, just under 15 minutes.
JOB_TIMEOUT_OVERRIDES: dict[Modality, float] = {
    Modality.audio: 870.0,
    # A long PDF is several passes of a few minutes each (~80 tok/s, up to
    # 32k tokens a pass). Matches OcrAdapter.run_batch_timeout.
    Modality.ocr: 3600.0,
}


# Slowest generation rate we plan for, tokens/sec. Measured on qwen3.8-27b
# (a 27B Q6_K on an RTX 5090): 59-61 tok/s across four prompts, flat.
# 20 leaves room for a bigger model on the smaller card plus a cold load.
FLOOR_TOKENS_PER_SECOND = 20.0


def _job_timeout(modality: Modality, job: "Job | None" = None) -> float:
    """How long a job may run before it is killed and its adapter unloaded.

    The flat 300s was fine while answers were a few hundred tokens. It is not
    fine for a thinking model: measured, real reasoning prompts
    spend 10k-21k tokens and 3-6 minutes, and one of them exceeded 300s while
    generating perfectly good output. A budget the caller explicitly asked for
    should not be cut off by a constant chosen before anyone asked for it, so
    the ceiling follows the requested token budget at a pessimistic rate.
    """
    base = JOB_TIMEOUT_OVERRIDES.get(modality, JOB_TIMEOUT_SECONDS)
    if job is None:
        return base
    request = job.request
    budget = None
    if request.chat_request:
        budget = request.chat_request.get("max_tokens")
    if budget is None and request.params:
        budget = request.params.get("max_tokens")
    if not isinstance(budget, int | float) or budget <= 0:
        # No explicit budget means "up to the model's context", so the ceiling
        # follows the context rather than the flat base — otherwise omitting
        # max_tokens would give a long answer *less* time than asking for one.
        budget = _context_budget(request)
        if budget is None:
            return base
    return max(base, float(budget) / FLOOR_TOKENS_PER_SECOND)


# Seconds a adapter without an engine-specific start budget may take to load:
# the in-process and child adapters (whisper, kokoro, the OCR and depth
# children) load in seconds to a minute.
DEFAULT_START_BUDGET_SECONDS = 300.0


def start_budget(model: str) -> float:
    """How long recipe ``model`` may take to start, from its engine or recipe.

    A start is not part of a job's run time — ``_job_timeout`` only begins
    once the adapter is up — but a caller waiting on the job sits through it.
    vllm needs minutes (192 s cold on an RTX 5090, CUDA graph capture
    included), llama.cpp seconds to minutes depending on the disk; each
    engine says so, and a recipe can say more.
    """
    recipe = get_recipe(model)
    if recipe is None:
        return DEFAULT_START_BUDGET_SECONDS
    if isinstance(recipe.params, VllmParams):
        return float(recipe.params.ready_timeout)
    if recipe.engine == "llama.cpp":
        from giq.adapters.llama_cpp import DEFAULT_READY_TIMEOUT, MODEL_READY_TIMEOUT

        return float(MODEL_READY_TIMEOUT.get(recipe.name, DEFAULT_READY_TIMEOUT))
    if recipe.engine == "sd.cpp":
        from giq.adapters.sdcpp import READY_TIMEOUT_SECONDS

        return READY_TIMEOUT_SECONDS
    return DEFAULT_START_BUDGET_SECONDS


def wait_budget(job: Job) -> float:
    """How long a caller should wait for ``job``: a start of its model, then its run.

    The flat 120 s the API used to wait could not cover a cold vllm start
    alone, so a non-streaming request that triggered one got a 504 while the
    job went on loading and generating for no one. The start is counted even
    when the model is already up: a resident can be evicted while the job
    waits in the queue, and an over-generous wait costs a caller nothing that
    a too-short one doesn't cost more.
    """
    modality = Modality(job.request.modality)
    return start_budget(job.request.model) + _job_timeout(modality, job)


def _context_budget(request: JobRequest) -> int | None:
    """The most an LLM job could generate: its model's context window.

    An upper bound, not a prediction — the prompt occupies part of that
    context, so the completion is always shorter. Only LLMs have one.
    """
    if request.modality != Modality.llm:
        return None
    return context_size(request.model)


# Max wait for VRAM to drop after unload before falling back to indefinite wait.
# CUDA context teardown can lag behind process exit; nvidia-smi may still report
# memory as used for a second or two after SIGKILL.
POST_UNLOAD_VRAM_GRACE_SECONDS = 10.0

# Bounded VRAM wait for owned adapter loads. giq controls the VRAM, so if it
# can't be freed in this window something is wrong — fail the job loudly
# instead of wedging the queue.
VRAM_WAIT_CAP_SECONDS = 180.0


# --- Resident set --------------------------------------------------------------
# Models giq keeps loaded whenever nothing else claims the VRAM, in reload-
# priority order. Jobs for resident models run on per-resident lanes (each
# lane serial, lanes concurrent with each other — an embed never queues
# behind a chat generation). Jobs for any other ("sleepy") model evict
# residents cheapest-first until the requirement fits, and the resident loop
# reloads them once the queue drains. Passed via get_runner()/constructor;
# tests construct Runner without it and see the legacy batch-only behavior.
def _residents_default() -> list[str]:
    """The configured resident set, in reload-priority order.

    Sourced from giq.registry (config.yaml `residents:` overriding the
    built-in set) rather than a constant here, so the machine-specific choice
    lives in config and every consumer — scheduler, catalog, dashboard —
    reads the same list.
    """
    from giq.registry import resident_defaults

    return resident_defaults()


RESIDENTS_DEFAULT = _residents_default()

# Contiguous quiet time (no llama-server slots active, no resident lane
# in-flight or recent) required before evicting residents — protects a live
# session from a mid-generation SIGTERM.
EVICT_IDLE_GRACE_SECONDS = 30.0

# Hard cap on eviction deferral so queued image jobs can never starve.
EVICT_DEFER_CAP_SECONDS = 600.0

# The queue must be free of sleepy-model jobs this long before residents
# reload — prevents reload thrash between back-to-back image jobs. Pending
# resident jobs skip the grace: they are waiting on the reload.
RESIDENT_RELOAD_GRACE_SECONDS = 15.0

RESIDENT_TICK_SECONDS = 5.0

# How long a queued resident job waits for its adapter to come back (evicted
# for an image batch) before failing. Kept under the 900s a typical polling
# client waits, so the failure is attributable to giq rather than a timeout.
RESIDENT_WAIT_TIMEOUT_SECONDS = 840.0

# --- Pause (hand the card back) ----------------------------------------------
# A paused runner serves nothing and holds no VRAM: every adapter is unloaded,
# residents stop reloading, and job submission is refused at the API edge
# (503, see Orchestrator.submit_job). A graceful pause first waits this long
# for in-flight work — batch job, resident lanes, live llama-server slots — to
# finish; a forced pause skips the wait and tears everything down now.
PAUSE_DRAIN_TIMEOUT_SECONDS = 60.0


class _AllDevices:
    """Sentinel for "every card" in _unload_worker.

    Not None: None already means "the slot on a machine with no visible GPU",
    which is a real, distinct slot key.
    """


_ALL_DEVICES = _AllDevices()


def _declared_size(res: "Instance") -> float:
    return res.adapter.estimated_vram_gb


def _choose_victims(
    loaded: list[tuple[str, "Instance"]],
    deficit: float,
    size_of: Callable[["Instance"], float] = _declared_size,
) -> list[tuple[str, "Instance"]]:
    """Pick the least disruptive set of residents to evict for ``deficit`` GB.

    Fewest residents first, then least VRAM freed. Evicting a resident is not
    a cost proportional to its size — it is an outage of that capability — so
    the count is what matters, and among equal counts we prefer freeing the
    least (leaving the most still loaded).

    This replaces cheapest-first accumulation, which was actively backwards
    on a packed card: "cheapest" selects the small, always-on models first
    (ecapa 0.6GB, whisper 4GB) and so a deficit gemma could nearly cover alone
    took all three residents down. Concretely, at 0.3GB free a flux render
    evicted ecapa + whisper + gemma where {gemma, ecapa} covers it.

    Exhaustive over subsets: the resident set is a handful of models by
    construction, and 2^n at n<=12 is nothing next to a model load. Above
    that, degrade to largest-first accumulation rather than hang.
    """
    from itertools import combinations

    if not loaded:
        return []
    if len(loaded) > 12:
        victims, projected = [], 0.0
        for key, res in sorted(loaded, key=lambda kv: -size_of(kv[1])):
            victims.append((key, res))
            projected += size_of(res)
            if projected >= deficit:
                break
        return victims

    best: list | None = None
    best_rank: tuple[int, float] | None = None
    for size in range(1, len(loaded) + 1):
        for combo in combinations(loaded, size):
            freed = sum(size_of(res) for _key, res in combo)
            if freed < deficit:
                continue
            rank = (size, freed)
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, list(combo)
        if best is not None:
            break  # no larger set can beat a smaller one on count
    # Nothing covers the deficit even by evicting everything: evict all and let
    # the caller's VRAM wait fail loudly rather than silently under-freeing.
    return best if best is not None else list(loaded)


class ResidentDemoted(Exception):
    """A lane job's model stopped being resident while the job was waiting."""


class Adapter(Protocol):
    """What the runner needs from an engine adapter: start it, stop it, run on it."""

    @property
    def is_running(self) -> bool: ...

    @property
    def is_ready(self) -> bool: ...

    @property
    def estimated_vram_gb(self) -> float: ...

    @property
    def pid(self) -> int | None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def run_batch(
        self, tasks: list[dict], params: dict | None = None
    ) -> list[LLMResult] | list[ImageResult]: ...


# Lane widths: how many jobs may run concurrently on a resident's adapter.
# llm matches llama-server's 4 slots; embed takes 2; audio is GPU-heavy and
# serial. The figures live in registry.DEFAULT_LANE_WIDTH.


def _lane_width(model: str, adapter: Any) -> int:
    """How many jobs a resident runs at once.

    An engine that schedules requests itself says how many it takes (ADR-002,
    D8): handing vllm fewer than its max_num_seqs wastes the batch it was
    sized for, handing it more only queues them inside the server where giq
    cannot see or cancel them. Everything else keeps the registry's width.
    """
    if isinstance(adapter, ServedLLM) and adapter.lanes_from_engine:
        return max(1, adapter.concurrency().max_parallel)
    return lane_width_for(model)


RESIDENT = "resident"
ON_DEMAND = "on_demand"


class Instance:
    """A recipe running on a card (ADR-003): its adapter, where, and why.

    The runner indexes instances two ways, because it schedules them two
    ways. A *resident* is kept loaded by policy and has a lane — a semaphore
    of ``width`` concurrent jobs, so an embed never queues behind a chat
    generation; the runner finds residents by recipe. An *on-demand* instance
    is loaded for a job and unloaded when idle, one per card, and dispatch to
    it is serialized; the runner finds it by card. ``GET /instances`` is both.
    """

    def __init__(
        self,
        adapter: Adapter,
        recipe: str,
        device: str | None = None,
        *,
        residency: str = ON_DEMAND,
        width: int = 1,
    ):
        self.adapter = adapter
        self.recipe = recipe
        # UUID of the card this instance occupies. Recorded at load time
        # rather than looked up on demand: eviction has to reason about where
        # the VRAM actually is, and a rebind while it is loaded must not
        # retarget an already-running process.
        self.device = device
        self.residency = residency
        self.width = width
        self.lane = asyncio.Semaphore(width)
        self.active_count = 0
        self.last_active = monotonic()
        self.started_at = time.time()

    @property
    def id(self) -> str:
        """``recipe@card``: unique while one recipe runs at most once per card."""
        return f"{self.recipe}@{self.device or 'default'}"

    @property
    def key(self) -> str:
        return self.recipe

    @property
    def model(self) -> str:
        """The recipe name, as the job API calls it."""
        return self.recipe

    @property
    def modality(self) -> Modality:
        """The recipe's first modality, as the status scalars report an instance."""
        recipe = get_recipe(self.recipe)
        return Modality(recipe.modality) if recipe is not None else Modality.llm

    @property
    def state(self) -> str:
        """``ready``, ``starting`` (process up, not answering yet), or ``stopped``."""
        if getattr(self.adapter, "is_ready", False):
            return "ready"
        if getattr(self.adapter, "is_running", False):
            return "starting"
        return "stopped"

    @property
    def port(self) -> int | None:
        """The loopback port of a server engine; None for in-process and child adapters."""
        port = getattr(getattr(self.adapter, "config", None), "port", None)
        return port if isinstance(port, int) else None

    async def drain(self) -> None:
        """Acquire the full lane (waits out in-flight jobs). Callers must
        pair with release_all()."""
        for _ in range(self.width):
            await self.lane.acquire()

    def release_all(self) -> None:
        for _ in range(self.width):
            self.lane.release()

    @property
    def in_flight(self) -> bool:
        return self.active_count > 0


def _as_dicts(results: list) -> list[dict]:
    """Batch results as stored on a job: dicts, whichever an adapter returned.

    The child-process adapters (audio, voiceprints, depth) pass the child's
    dicts through; the others return result models. The resident path
    accepted both and the on-demand path only models — harmless while the
    audio stack was kept warm by default, a 500 on every transcription
    once nothing was.
    """
    return [r if isinstance(r, dict) else r.model_dump() for r in results]


class Runner:
    """Processes jobs from queue, manages adapter lifecycle."""

    def __init__(
        self,
        queue: JobQueue | None = None,
        residents: list[str] | None = None,
        use_policy: bool = False,
    ):
        # `JobQueue` is falsy when empty (defines __len__), so `queue or …`
        # silently swaps a freshly-passed queue for the global one.
        self._queue = queue if queue is not None else get_queue()
        # Sleepy adapters by GPU UUID (None when no card is visible at all).
        self._slots: dict[str | None, Instance] = {}
        self._running = False
        self._processing_job = False  # True while actively processing a job
        self._task: asyncio.Task | None = None
        self._warm_task: asyncio.Task | None = None
        # Two modes. use_policy=True (production) makes the resident set live:
        # it is whatever the policy store currently pins, re-read every tick,
        # so an operator can pin or unpin without a restart. A static list is
        # the legacy path and what tests construct.
        self._use_policy = use_policy
        self._static_residents = list(residents or [])
        self._residents: dict[str, Instance] = {}
        self._resident_task: asyncio.Task | None = None
        self._resident_jobs: set[asyncio.Task] = set()
        self._queue_empty_since: float | None = None
        self._last_job_time: datetime | None = None
        self._paused = False
        self._paused_since: datetime | None = None
        self._pause_reason: str | None = None
        # Serializes adapter lifecycle between _ensure_worker, warm-timeout
        # unloads, shutdown, resident load/evict, and timeout/failure unloads.
        self._worker_lock = asyncio.Lock()

    @property
    def _resident_keys(self) -> list[str]:
        """Models that should be resident right now, in reload-priority order."""
        if not self._use_policy:
            return self._static_residents
        from giq.policy import get_policy_store

        return get_policy_store().residents()

    def _policy_for(self, model: str) -> str:
        """This model's policy.

        Consulted regardless of ``use_policy``: that flag decides whether the
        *resident set* is live, but `off` is a hard statement that the model
        must not load, and it would be a trap for that to depend on how the
        runner happened to be constructed. With no override set this returns
        the registry default, so a static-residents runner behaves as before.
        """
        from giq.policy import get_policy_store

        return get_policy_store().policy_for(model)

    def is_disabled(self, model: str) -> bool:
        """True when policy forbids this recipe from loading at all."""
        return self._policy_for(model) == "off"

    def _device_for(self, model: str) -> str | None:
        """UUID of the card this recipe is bound to (None = no GPU visible)."""
        from giq.policy import device_of

        gpu = device_of(model)
        return gpu.uuid if gpu else None

    def _slot_for(self, model: str) -> Instance | None:
        """The loaded sleepy adapter for this model, if it is loaded."""
        slot = self._slots.get(self._device_for(model))
        if slot and slot.key == model:
            return slot
        return None

    @property
    def active_modality(self) -> Modality | None:
        """A currently loaded on-demand instance's (first) modality.

        Scalar for back-compat (/status, the stats sampler). With one slot per
        card there can be more than one; ``active_slots`` has them all.
        """
        slot = next(iter(self._slots.values()), None)
        return slot.modality if slot else None

    @property
    def active_recipe(self) -> str | None:
        """The recipe of the instance ``active_modality`` reports."""
        slot = next(iter(self._slots.values()), None)
        return slot.model if slot else None

    @property
    def active_slots(self) -> list[dict[str, Any]]:
        """Every loaded on-demand instance, with the card it sits on."""
        return [
            {
                "device": slot.device,
                "modality": str(slot.modality),
                "recipe": slot.model,
                "ready": bool(getattr(slot.adapter, "is_ready", False)),
            }
            for slot in self._slots.values()
        ]

    @property
    def resident_models(self) -> dict[str, bool]:
        """{recipe: ready} for configured residents (for /status)."""
        out = {}
        for key in self._resident_keys:
            res = self._residents.get(key)
            out[key] = bool(res and res.adapter.is_ready)
        return out

    def loaded_keys(self) -> set[str]:
        """The recipe of every ready adapter, resident or on a card's slot.

        A job for one of these needs no VRAM to start, which is what tells a
        job that is waiting its turn from one that is waiting for room.
        """
        keys = {k for k, res in self._residents.items() if getattr(res.adapter, "is_ready", False)}
        keys |= {s.model for s in self._slots.values() if getattr(s.adapter, "is_ready", False)}
        return keys

    def instances(self) -> list[Instance]:
        """Every instance giq holds, resident or on demand, by id."""
        return sorted([*self._residents.values(), *self._slots.values()], key=lambda i: i.id)

    @property
    def owned_pids(self) -> dict[int, str]:
        """{pid: recipe} for every process giq is holding VRAM through.

        Includes giq's own pid: STT is the one adapter that still loads in this
        process, so when it is up, part of the card is ours through no child
        at all. Workers with no pid (nothing spawned yet) are skipped rather
        than guessed at.
        """
        owned: dict[int, str] = {}
        for key, res in self._residents.items():
            pid = getattr(res.adapter, "pid", None)
            if pid:
                owned[pid] = key
        for slot in self._slots.values():
            pid = getattr(slot.adapter, "pid", None)
            if pid:
                owned[pid] = f"{slot.model}"
            elif isinstance(slot.adapter, SttAdapter):
                owned[os.getpid()] = f"{slot.model}"
        return owned

    @property
    def resident_devices(self) -> dict[str, str | None]:
        """{recipe: GPU UUID} for residents that are currently loaded.

        Where they *are*, not where they belong — during a rebind those differ
        for a tick, and what a caller sizing a card wants is the truth.
        """
        return {key: res.device for key, res in self._residents.items()}

    @property
    def llm_base_url(self) -> str | None:
        """Base URL of any ready llama-server. Prefer ``llm_base_url_for``."""
        return self._llm_base_url(None)

    def llm_base_url_for(self, model: str) -> str | None:
        """Base URL of the llama-server running ``model``, if it is ready.

        Model-specific because two cards can each hold a resident LLM on their
        own port. "Any ready LLM" was a safe answer while only one could
        exist; now it can hand a client the wrong model's server.
        """
        return self._llm_base_url(model)

    def _llm_base_url(self, model: str | None) -> str | None:
        for key, res in self._residents.items():
            if model is not None and key != model:
                continue
            if isinstance(res.adapter, ServedLLM) and res.adapter.is_ready:
                return res.adapter.base_url
        for slot in self._slots.values():
            if model is not None and slot.model != model:
                continue
            if isinstance(slot.adapter, ServedLLM) and slot.adapter.is_ready:
                return slot.adapter.base_url
        return None

    def is_resident_key(self, model: str) -> bool:
        return model in self._resident_keys

    # --- Pause ---------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """True while serving is suspended and the card is being left alone."""
        return self._paused

    @property
    def pause_state(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "since": self._paused_since.isoformat() if self._paused_since else None,
            "reason": self._pause_reason,
        }

    async def pause(self, force: bool = False, reason: str | None = None) -> dict[str, Any]:
        """Stop serving and unload every model, handing the GPU back.

        Graceful (the default) waits up to ``PAUSE_DRAIN_TIMEOUT_SECONDS`` for
        in-flight work to finish before unloading; ``force`` skips that wait and
        tears the adapters down immediately, failing whatever was running. Jobs
        still queued are failed either way — the API refuses new ones while
        paused, so nothing accumulates behind the pause.

        Idempotent: pausing an already-paused runner just re-reports the state.
        """
        if self._paused:
            return {**self.pause_state, "forced": False, "drained": True, "warnings": []}

        # Set first: this gates the dispatch loop and the residents loop, so
        # nothing new starts while we are draining what's already running.
        self._paused = True
        self._paused_since = datetime.now()
        self._pause_reason = reason
        logger.info(f"Pausing giq ({'forced' if force else 'graceful'}), reason={reason!r}")

        warnings: list[str] = []
        drained = True
        if not force:
            drained = await self._await_quiet(PAUSE_DRAIN_TIMEOUT_SECONDS)
            if not drained:
                warnings.append(
                    f"in-flight work was still running after {PAUSE_DRAIN_TIMEOUT_SECONDS:.0f}s "
                    "— unloaded anyway"
                )

        failed = await self._fail_pending_jobs("giq is paused — job never started")
        if failed:
            warnings.append(f"failed {failed} queued job(s)")

        # Lane tasks that are still waiting for (or holding) a resident have no
        # adapter to come back to; cancel them rather than let them wait out
        # RESIDENT_WAIT_TIMEOUT_SECONDS.
        for task in list(self._resident_jobs):
            task.cancel()
        if self._resident_jobs:
            await asyncio.gather(*self._resident_jobs, return_exceptions=True)

        async with self._worker_lock:
            try:
                await self._unload_worker()
            except Exception as e:
                logger.error(f"Failed to unload batch adapter(s) for pause: {e}")
                warnings.append(f"batch adapter still loaded: {e}")
            for key in list(self._residents):
                res = self._residents[key]
                try:
                    await res.adapter.stop()
                    self._residents.pop(key, None)
                except Exception as e:
                    logger.error(f"Failed to stop resident {key} for pause: {e}")
                    warnings.append(f"{key} still loaded: {e}")

        from giq.stats import get_stats

        await get_stats().record_event("pause", reason or ("forced" if force else "graceful"))
        logger.info(f"giq paused ({get_free_vram():.1f}GB free)")
        return {**self.pause_state, "forced": force, "drained": drained, "warnings": warnings}

    async def resume(self) -> dict[str, Any]:
        """Resume serving. Residents reload on the next residents-loop tick."""
        if not self._paused:
            return self.pause_state
        self._paused = False
        self._paused_since = None
        self._pause_reason = None
        # Restart the reload grace from now rather than from whenever the queue
        # last went quiet, so resume doesn't race a job arriving in the same tick.
        self._queue_empty_since = None
        await self._cleanup_finished_jobs()

        from giq.stats import get_stats

        await get_stats().record_event("resume")
        logger.info("giq resumed — residents reload shortly")
        return self.pause_state

    async def _await_quiet(self, timeout: float) -> bool:
        """Wait until nothing is running. False if ``timeout`` expired first.

        Busy = the batch slot is processing, a resident lane has a job in
        flight, or a resident llama-server reports active slots (direct clients
        bypass the lanes entirely — see _defer_while_busy).
        """
        started = monotonic()
        while True:
            busy = self._processing_job or any(r.in_flight for r in self._residents.values())
            if not busy:
                for res in self._residents.values():
                    if isinstance(res.adapter, ServedLLM) and await res.adapter.active_slot_count():
                        busy = True
                        break
            if not busy:
                return True
            if monotonic() - started >= timeout:
                logger.warning(f"pause: still busy after {timeout:.0f}s, unloading anyway")
                return False
            await asyncio.sleep(0.5)

    async def _fail_pending_jobs(self, error: str) -> int:
        """Fail every queued-but-unstarted job so waiters get an answer now.

        A streaming waiter is not polling the job: it is blocked on the stream's
        end sentinel, which only the run paths send. A job failed before it ever
        ran never reaches them, so the sentinel has to come from here or the
        client hangs on an open SSE response until it gives up.
        """
        count = 0
        for job in await self._queue.get_pending():
            job.status = JobStatus.failed
            job.error = error
            job.completed_at = datetime.now()
            if job.stream is not None:
                await job.stream.close()
            count += 1
        return count

    async def start(self) -> None:
        """Start the runner loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        # Under policy the pinned set is live: an operator can pin a model at
        # any time, so the loop has to exist even when nothing is pinned at
        # boot. Starting it only for a non-empty set means a machine whose
        # every registry resident is set to on-demand boots without the loop
        # at all — pinning then shows the model in the resident lanes and
        # routes its jobs to the lane, where they wait on a load that nothing
        # would ever perform. A static list cannot
        # change, so an empty one there still means there is nothing to run.
        if self._use_policy or self._static_residents:
            self._resident_task = asyncio.create_task(self._residents_loop())
        logger.info("Runner started")

    async def stop(self) -> None:
        """Stop the runner and unload all adapters."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._warm_task:
            self._warm_task.cancel()
        if self._resident_task:
            self._resident_task.cancel()
            try:
                await self._resident_task
            except asyncio.CancelledError:
                pass
        for task in list(self._resident_jobs):
            task.cancel()
        if self._resident_jobs:
            await asyncio.gather(*self._resident_jobs, return_exceptions=True)
        async with self._worker_lock:
            await self._unload_worker()
            for key in list(self._residents):
                res = self._residents.pop(key)
                try:
                    await res.adapter.stop()
                except Exception as e:
                    logger.error(f"Failed to stop resident {key}: {e}")
        logger.info("Runner stopped")

    async def _run_loop(self) -> None:
        """Main processing loop.

        Jobs for resident models fan out to per-resident lane tasks (concurrent
        across residents, width-bounded within one); everything else runs
        serially in the batch slot with resident eviction as needed.
        """
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(0.2)
                    continue
                job = await self._queue.get_next()
                if job:
                    key = job.request.model
                    if key in self._resident_keys:
                        # Claim the job SYNCHRONOUSLY before create_task: the
                        # spawned task may not run for a while, and get_next()
                        # can complete without yielding — leaving the job
                        # pending here would respawn it in a tight loop that
                        # starves the event loop (it has OOM-killed giq).
                        job.status = JobStatus.running
                        job.started_at = datetime.now()
                        task = asyncio.create_task(self._process_resident_job(job, key))
                        self._resident_jobs.add(task)
                        task.add_done_callback(self._resident_jobs.discard)
                        # Explicit yield so lane tasks start promptly even if
                        # the queue never suspends this loop.
                        await asyncio.sleep(0)
                    else:
                        await self._process_job(job)
                else:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Runner loop error: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: Job) -> None:
        """Process a single job."""
        logger.info(f"Processing job {job.job_id}: {job.request.modality}/{job.request.model}")
        _log_inflight("start", job)

        job.status = JobStatus.running
        job.started_at = datetime.now()
        self._processing_job = True

        device = self._device_for(job.request.model)
        try:
            # Sleepy-model job: free enough resident VRAM first, on THIS card
            # (waits for in-flight resident sessions, capped so this job can't
            # starve). Residents on other cards are not candidates — evicting
            # them would free memory the load cannot use.
            await self._evict_residents_for(job.request.model)

            # Ensure correct adapter is loaded
            adapter = await self._ensure_worker(job.request.model, job.request.model_path)

            # Run the job with a timeout to prevent infinite hangs
            timeout = _job_timeout(job.request.modality, job)
            if adapter:
                if job.request.chat_request is not None:
                    # Raw chat completion passthrough (tool-calling path)
                    if not isinstance(adapter, ServedLLM):
                        raise RuntimeError("chat_request requires LLM worker")
                    if job.stream is not None:
                        raw_result = await asyncio.wait_for(
                            adapter.chat_completion_stream(job.request.chat_request, job.stream),
                            timeout=timeout,
                        )
                    else:
                        raw_result = await asyncio.wait_for(
                            adapter.chat_completion(job.request.chat_request),
                            timeout=timeout,
                        )
                    # Store raw dict as single result
                    job.results = [raw_result]
                    job.status = JobStatus.completed
                else:
                    results = await asyncio.wait_for(
                        adapter.run_batch(
                            job.request.tasks,
                            job.request.params,
                        ),
                        timeout=timeout,
                    )
                    job.results = _as_dicts(results)
                    job.status = JobStatus.completed
            else:
                job.status = JobStatus.failed
                job.error = f"Failed to load adapter: {job.request.modality}"

        except TimeoutError:
            limit = _job_timeout(job.request.modality, job)
            logger.error(f"Job {job.job_id} timed out after {limit:.0f}s, unloading worker")
            job.status = JobStatus.failed
            job.error = f"Job timed out after {limit:.0f}s"
            # llama-server may still be generating after HTTP cancel; tear it
            # down so it doesn't keep burning VRAM + CPU in the background.
            try:
                async with self._worker_lock:
                    await self._unload_worker(device)
            except Exception as unload_err:
                logger.error(f"Failed to unload timed-out adapter: {unload_err}")

        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            job.status = JobStatus.failed
            job.error = str(e)
            # A adapter that errored mid-generation may be wedged. Unload so
            # the next job gets a clean slate rather than inheriting the mess.
            try:
                async with self._worker_lock:
                    await self._unload_worker(device)
            except Exception as unload_err:
                logger.error(f"Failed to unload failed adapter: {unload_err}")

        finally:
            self._processing_job = False
            # The reader is blocked on the sentinel; it has to arrive whether
            # the job succeeded, failed, timed out or never loaded a adapter.
            if job.stream is not None:
                await job.stream.close()

        job.completed_at = datetime.now()
        self._last_job_time = datetime.now()

        from giq.stats import get_stats

        await get_stats().record_job(job)

        # Clean up finished jobs older than 5 minutes
        await self._cleanup_finished_jobs()

        # Schedule warm timeout check
        self._schedule_warm_timeout()

        logger.info(
            f"Job {job.job_id} {job.status}: "
            f"{len(job.results) if job.results else 0} results, "
            f"{job.duration_ms}ms"
        )
        _log_inflight(
            "end",
            job,
            status=str(job.status),
            duration_ms=job.duration_ms,
            results=len(job.results) if job.results else 0,
            error=job.error,
        )

    # --- Resident machinery -------------------------------------------------

    def _build_worker(self, model: str, model_path: str | None = None):
        """Construct (but don't start) the adapter that runs recipe ``model``.

        Chosen by the recipe's engine and its first modality: a recipe serving
        several (flux_klein) runs them all in one process, so any of them
        picks the same adapter. Every adapter is handed the card its recipe is
        bound to. That is what makes a binding real: it decides
        CUDA_VISIBLE_DEVICES for the child, and for the server backends the
        port as well, so two cards can each run their own llama-server or
        sd-server.
        """
        recipe = get_recipe(model)
        if recipe is None:
            raise ValueError(f"unknown recipe {model!r}")
        model = recipe.name
        modality = Modality(recipe.modality)
        device = self._device_for(model)
        if recipe.engine == "vllm":
            from giq.adapters.vllm import VllmAdapter, VllmConfig

            if model_path:
                # A path override names a GGUF for llama-server; a vllm
                # model's weights are part of its declaration.
                logger.warning(f"llm/{model}: model_path ignored, vllm serves its declared weights")
            return VllmAdapter(config=VllmConfig(model=model, device=device))
        if modality == Modality.llm:
            return LlamaCppAdapter(
                config=LlamaCppConfig(model=model, model_path=model_path, device=device)
            )
        if modality == Modality.audio:
            from giq.adapters.audio import AudioAdapter, AudioConfig

            return AudioAdapter(config=AudioConfig(model=model), device=device)
        if modality == Modality.embed:
            from giq.adapters.audio import EmbedAdapter, EmbedConfig

            return EmbedAdapter(config=EmbedConfig(model=model), device=device)
        if modality in (Modality.text2image, Modality.image_edit):
            # sd.cpp is the one image runtime; the recipe schema admits no other.
            from giq.adapters.sdcpp import SdCppAdapter, SdCppConfig

            return SdCppAdapter(config=SdCppConfig(model=model, device=device))
        if modality == Modality.tts:
            from giq.adapters.tts import TtsAdapter, TtsConfig

            return TtsAdapter(config=TtsConfig(model=model), device=device)
        if modality == Modality.stt:
            from giq.adapters.stt import SttAdapter, SttConfig

            return SttAdapter(config=SttConfig(model=model, gpu_device=device))
        if modality == Modality.ocr:
            from giq.adapters.ocr import OcrAdapter, OcrConfig

            return OcrAdapter(config=OcrConfig(model=model), device=device)
        if modality == Modality.depth:
            from giq.adapters.depth import DepthAdapter, DepthConfig

            return DepthAdapter(config=DepthConfig(model=model), device=device)
        raise ValueError(f"no adapter for {model}'s modality {modality}")

    async def _process_resident_job(self, job: Job, key: str) -> None:
        """Run one job on a resident's lane (concurrent with other lanes)."""
        # status/started_at are set by the dispatch loop before this task is
        # created (see _run_loop's claim-before-create_task comment).
        logger.info(f"Resident job {job.job_id}: {key}")
        _log_inflight("start", job)
        try:
            while True:
                res = await self._wait_resident_ready(key)
                await res.lane.acquire()
                if res.adapter.is_ready:
                    break
                # Evicted between our ready-check and lane entry — the
                # residents loop will bring up a fresh Instance; wait for it.
                res.lane.release()
            try:
                res.active_count += 1
                res.last_active = monotonic()
                try:
                    timeout = _job_timeout(job.request.modality, job)
                    if job.request.chat_request is not None:
                        if not isinstance(res.adapter, ServedLLM):
                            raise RuntimeError("chat_request requires LLM worker")
                        if job.stream is not None:
                            raw_result = await asyncio.wait_for(
                                res.adapter.chat_completion_stream(
                                    job.request.chat_request, job.stream
                                ),
                                timeout=timeout,
                            )
                        else:
                            raw_result = await asyncio.wait_for(
                                res.adapter.chat_completion(job.request.chat_request),
                                timeout=timeout,
                            )
                        job.results = [raw_result]
                    else:
                        results = await asyncio.wait_for(
                            res.adapter.run_batch(job.request.tasks, job.request.params),
                            timeout=timeout,
                        )
                        job.results = _as_dicts(results)
                    job.status = JobStatus.completed
                finally:
                    res.active_count -= 1
                    res.last_active = monotonic()
            finally:
                res.lane.release()
        except ResidentDemoted as e:
            # Not a failure: the model simply stopped being resident. Put the
            # job back on the queue and the dispatch loop will route it to the
            # batch path, which can still load it on demand (unless policy
            # also turned it off, in which case the batch path fails it with
            # that reason instead of a misleading timeout).
            logger.info(f"Job {job.job_id} requeued to the batch path: {e}")
            job.status = JobStatus.pending
            job.started_at = None
            _log_inflight("end", job, status="requeued", reason=str(e))
            return
        except TimeoutError:
            job.status = JobStatus.failed
            job.error = f"Job timed out after {_job_timeout(job.request.modality, job):.0f}s"
        except asyncio.CancelledError:
            job.status = JobStatus.failed
            job.error = "Cancelled (giq paused)" if self._paused else "Cancelled (runner shutdown)"
            raise
        except Exception as e:
            logger.error(f"Resident job {job.job_id} failed: {e}")
            job.status = JobStatus.failed
            job.error = str(e)
        finally:
            # A requeued job (status back to pending) is not finished: no
            # completion timestamp and no stats row, or the batch path's later
            # run would double-count it. Guarded rather than returned early —
            # a return here would swallow the re-raised CancelledError.
            # A requeued streaming job is about to run again on the batch
            # path and will stream from there — closing here would end the
            # client's response before the answer exists.
            if job.status != JobStatus.pending:
                if job.stream is not None:
                    await job.stream.close()
                job.completed_at = datetime.now()
                self._last_job_time = datetime.now()
                _log_inflight(
                    "end",
                    job,
                    status=str(job.status),
                    duration_ms=job.duration_ms,
                    results=len(job.results) if job.results else 0,
                    error=job.error,
                )
                from giq.stats import get_stats

                await get_stats().record_job(job)

    async def _wait_resident_ready(self, key: str) -> Instance:
        """Wait for a resident to be loaded and ready (it may be evicted for
        an image batch; the residents loop reloads it once the batch drains)."""
        started = monotonic()
        while True:
            res = self._residents.get(key)
            if res and res.adapter.is_ready:
                return res
            # Unpinned mid-flight: nothing will ever load this on the lane, so
            # fail now instead of waiting out the full timeout. The job is not
            # lost — the caller retries it through the batch path.
            if key not in self._resident_keys:
                raise ResidentDemoted(f"{key} is no longer resident")
            if monotonic() - started >= RESIDENT_WAIT_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"resident {key} unavailable after "
                    f"{RESIDENT_WAIT_TIMEOUT_SECONDS:.0f}s (evicted for batch work?)"
                )
            await asyncio.sleep(0.5)

    async def _residents_loop(self) -> None:
        """Keep residents loaded while nothing else claims the VRAM.

        Single reload path: API handlers and lane tasks never load adapters.
        Reload fires when no sleepy-model jobs are pending/running and either
        the queue has been quiet for the reload grace or resident jobs are
        actively waiting (they skip the grace — they need the model now).
        """
        while self._running:
            try:
                await asyncio.sleep(RESIDENT_TICK_SECONDS)
                if self._paused:
                    self._queue_empty_since = None
                    continue
                if self._processing_job:
                    self._queue_empty_since = None
                    continue
                await self._release_demoted_residents()
                pending = await self._queue.get_pending()
                sleepy = [j for j in pending if j.request.model not in self._resident_keys]
                if sleepy:
                    self._queue_empty_since = None
                    continue
                resident_waiting = bool(pending) or any(True for _ in self._resident_jobs)
                now = monotonic()
                if self._queue_empty_since is None:
                    self._queue_empty_since = now
                    if not resident_waiting:
                        continue
                if (
                    not resident_waiting
                    and now - self._queue_empty_since < RESIDENT_RELOAD_GRACE_SECONDS
                ):
                    continue
                for key in self._resident_keys:
                    res = self._residents.get(key)
                    if res and res.adapter.is_ready:
                        continue
                    # Re-check before EACH load: a sleepy job that arrived
                    # mid-sequence would otherwise fight our loads for VRAM
                    # (its eviction pass vs. our next resident — starved a
                    # zimage job for 180s at startup).
                    if self._processing_job:
                        break
                    pending = await self._queue.get_pending()
                    if any(j.request.model not in self._resident_keys for j in pending):
                        break
                    await self._load_resident(key)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"residents tick failed: {e}")

    async def _release_demoted_residents(self) -> None:
        """Unload residents that should no longer be where they are.

        Two cases, one mechanism. A model the operator has unpinned (or turned
        off) must go: the residents loop is the single reload path, so a model
        that is merely stopped comes straight back on the next tick, and it
        stops coming back only once it leaves the pinned set. A model that has
        been *rebound* must also go, because a running process cannot be moved
        between cards — unloading it here is what lets the next tick bring it
        up on the card it now belongs to.

        Teardown goes through the same drain + lock path as eviction so
        in-flight lane jobs finish first.
        """
        wanted = set(self._resident_keys)
        demoted: list[tuple[str, str]] = []
        for key, res in self._residents.items():
            if key not in wanted:
                demoted.append((key, f"policy is now {self._policy_for(key)}"))
            elif res.device != self._device_for(key):
                demoted.append((key, "rebound to another card"))
        if not demoted:
            return

        from giq.stats import get_stats

        for key, why in demoted:
            res = self._residents.get(key)
            if res is None:
                continue
            await res.drain()  # waits out in-flight lane jobs
            try:
                async with self._worker_lock:
                    if self._residents.get(key) is not res:
                        continue
                    logger.info(f"unloading {key} — {why}")
                    await res.adapter.stop()
                    self._residents.pop(key, None)
                    await get_stats().record_event("unload", f"{key} ({why})")
            except Exception as e:
                logger.error(f"Failed to unload demoted resident {key}: {e}")
            finally:
                res.release_all()

    async def _load_resident(self, key: str) -> None:
        model = key
        async with self._worker_lock:
            # Re-check under the lock: a pause may have landed while this tick
            # was waiting for VRAM or for the lock, and loading now would put
            # the card straight back under load.
            if self._paused:
                return
            if self.is_disabled(key):
                return
            res = self._residents.get(key)
            if res and res.adapter.is_ready:
                return
            if res:
                # Loaded-but-dead (child crashed): reap before respawning.
                try:
                    await res.adapter.stop()
                except Exception as e:
                    logger.error(f"Failed to reap dead resident {key}: {e}")
                    return
                self._residents.pop(key, None)
            # Residents get a reduced margin: the set is validated to coexist,
            # and the full spike margin can wedge the reload forever when free
            # VRAM sits just inside the margin window (see wait_for_vram).
            device = self._device_for(model)
            vram_ok = await wait_for_vram(
                model,
                timeout=POST_UNLOAD_VRAM_GRACE_SECONDS,
                margin_gb=0.5,
                device=device,
            )
            if not vram_ok:
                logger.info(f"resident {model}: VRAM not free yet, retry next tick")
                return
            logger.info(f"resident: loading {model}")
            adapter = self._build_worker(model)
            width = _lane_width(model, adapter)
            resident = Instance(adapter, model, device, residency=RESIDENT, width=width)
            self._residents[key] = resident
            try:
                await adapter.start()
            except BaseException:
                if not getattr(adapter, "is_running", False):
                    self._residents.pop(key, None)
                raise
            from giq.stats import get_stats

            await get_stats().record_event("reload", model)

    async def _evict_residents_for(self, model: str) -> None:
        """Free enough resident VRAM on this model's card for it to load.

        Victims are drawn only from residents on that same card: freeing VRAM
        elsewhere buys the load nothing, and taking a resident down for it
        would be a pure outage. This is the payoff of binding — gemma pinned
        to the big card is simply not a candidate when a render needs room on
        the small one.

        Within the card, ``_choose_victims`` picks the fewest residents that
        cover the deficit. Before stopping anything, wait for contiguous quiet
        (no llama slots active, no lane in-flight) up to the deferral cap so
        live sessions aren't SIGTERMed mid-request.
        """
        from giq.vram import get_vram_requirement, margin_for, reserve_for

        if not self._residents or model in self._resident_keys:
            return

        device = self._device_for(model)
        base = get_vram_requirement(model)
        required = base + margin_for(base) + reserve_for(device)
        free = get_free_vram(device)
        if free >= required:
            return

        on_card = [(key, res) for key, res in self._residents.items() if res.device == device]
        if not on_card:
            return

        deficit = required - free
        size_of = await self._victim_sizer(device)
        loaded = sorted(on_card, key=lambda kv: size_of(kv[1]))
        victims = _choose_victims(loaded, deficit, size_of)

        await self._defer_while_busy(victims, model)

        from giq.stats import get_stats

        for key, res in victims:
            await res.drain()  # waits out in-flight lane jobs
            try:
                async with self._worker_lock:
                    if self._residents.get(key) is not res:
                        continue
                    logger.info(f"evicting resident {key} for {model}")
                    await res.adapter.stop()
                    self._residents.pop(key, None)
                    await get_stats().record_event("evict", f"{key} for {model}")
            finally:
                res.release_all()

    async def _victim_sizer(self, device: str | None = None) -> Callable[[Instance], float]:
        """How much VRAM each resident would actually give back if evicted.

        Deliberately *not* the same number the VRAM gate uses. The declared
        figure in the registry is an upper bound — its job is to answer "is
        there room to load this?", so erring high is correct and it carries
        padding on purpose. Sizing victims with it means planning against
        padding: gemma is declared 9.5GB and holds 8.88, so the planner
        believed evicting it freed half a gigabyte that does not exist, and
        could pick a victim set that leaves the load short.

        Measured per-process via nvidia-smi where a adapter owns a child.
        Workers running in giq's own process have no distinguishable share, so
        they keep the declared figure.
        """
        from giq.gpus import compute_app_memory

        pid_memory = await asyncio.to_thread(compute_app_memory, device)
        if not pid_memory:
            return _declared_size

        def size_of(res: Instance) -> float:
            pid = getattr(res.adapter, "pid", None)
            measured = pid_memory.get(pid) if pid is not None else None
            # A just-started child may not have allocated yet; trust the
            # declared figure rather than plan around a near-zero reading.
            if measured is None or measured <= 0.05:
                return res.adapter.estimated_vram_gb
            return measured

        return size_of

    async def _defer_while_busy(
        self,
        victims: list[tuple[str, Instance]],
        model: str,
    ) -> None:
        """Wait until every victim has been quiet for EVICT_IDLE_GRACE_SECONDS.

        Quiet = no lane job in flight, no active llama-server slots, and
        last activity older than the grace (that last check IS the contiguous-
        quiet window, since last_active updates on every lane entry/exit).
        Capped at EVICT_DEFER_CAP_SECONDS so queued jobs can't starve.
        """
        started = monotonic()
        while True:
            busy = False
            for _key, res in victims:
                if res.in_flight:
                    busy = True
                    break
                if isinstance(res.adapter, ServedLLM):
                    slots = await res.adapter.active_slot_count()
                    if slots:
                        res.last_active = monotonic()  # direct clients bypass lanes
                        busy = True
                        break
                if monotonic() - res.last_active < EVICT_IDLE_GRACE_SECONDS:
                    busy = True
                    break
            if not busy:
                return
            if monotonic() - started >= EVICT_DEFER_CAP_SECONDS:
                logger.warning(
                    f"Evicting busy residents for {model}: "
                    f"deferral cap ({EVICT_DEFER_CAP_SECONDS:.0f}s) reached"
                )
                return
            await asyncio.sleep(2.0)

    async def _ensure_worker(
        self,
        model: str,
        model_path: str | None = None,
    ) -> Adapter | None:
        """Ensure this model's adapter is loaded on its card. Returns it.

        Only the slot on that card is disturbed: a adapter loaded on another
        card keeps running, because it is not in the way. Serialized via
        ``self._worker_lock`` so that warm-timeout unloads and timeout/failure
        unloads can't race with loads.
        """
        async with self._worker_lock:
            # A forced pause can land between job dispatch and this load; fail
            # the job rather than re-claim the card the operator just took back.
            if self._paused:
                raise RuntimeError("giq is paused — no adapters may load")
            # Policy is enforced at submit, but re-check here: a model can be
            # turned off between dispatch and load, and `off` must mean no
            # load path at all, not just no new submissions.
            if self.is_disabled(model):
                raise RuntimeError(f"{model} is disabled (policy: off)")

            device = self._device_for(model)
            # Check if we already have the right adapter AND it's ready
            slot = self._slots.get(device)
            if slot and slot.key == model and slot.adapter.is_ready:
                return slot.adapter

            # Unload whatever else holds this card's slot (free VRAM first)
            await self._unload_worker(device)

            # Bounded wait for VRAM to actually drop after unload/eviction.
            # CUDA context teardown lags process exit, so nvidia-smi may still
            # show memory used for a second or two. If the grace window
            # expires, keep waiting up to VRAM_WAIT_CAP_SECONDS, then fail
            # the job loudly rather than wedge the queue.
            vram_ok = await wait_for_vram(
                model,
                timeout=POST_UNLOAD_VRAM_GRACE_SECONDS,
                poll_interval=0.5,
                device=device,
            )
            if not vram_ok:
                logger.warning(
                    f"VRAM still blocked {POST_UNLOAD_VRAM_GRACE_SECONDS}s after unload, "
                    f"waiting up to {VRAM_WAIT_CAP_SECONDS:.0f}s for {model}"
                )
                vram_ok = await wait_for_vram(model, timeout=VRAM_WAIT_CAP_SECONDS, device=device)
                if not vram_ok:
                    raise RuntimeError(
                        f"VRAM never became available for {model} "
                        f"within {VRAM_WAIT_CAP_SECONDS:.0f}s"
                    )

            logger.info(f"VRAM available ({get_free_vram(device):.1f}GB free), loading {model}")

            adapter = self._build_worker(model, model_path)

            # Publish the adapter BEFORE start() so that _unload_worker can reap
            # it if start() raises (including CancelledError during load).
            self._slots[device] = Instance(adapter, model, device, residency=ON_DEMAND)
            try:
                await adapter.start()
            except BaseException:
                # start() already attempts its own stop(). If the process is
                # still alive (SIGKILL-stuck CUDA), keep the reference so a
                # later _unload_worker can retry. Otherwise clear the slot.
                if not getattr(adapter, "is_running", False):
                    self._slots.pop(device, None)
                raise
            return adapter

    async def _unload_worker(self, device: str | None | _AllDevices = _ALL_DEVICES) -> None:
        """Unload the sleepy adapter on one card, or (by default) on all.

        Raises RuntimeError if a adapter process cannot be killed (e.g. stuck
        in CUDA driver). In that case its slot is NOT cleared — the caller
        must handle the zombie state.
        """
        devices: list[str | None] = (
            list(self._slots) if isinstance(device, _AllDevices) else [device]
        )
        for dev in devices:
            slot = self._slots.get(dev)
            if slot is None:
                continue
            await slot.adapter.stop()  # raises RuntimeError if unkillable
            self._slots.pop(dev, None)

    async def _cleanup_finished_jobs(self, max_age_seconds: float = 300) -> None:
        """Remove completed/failed jobs older than max_age to prevent memory leak."""
        now = datetime.now()
        all_jobs = await self._queue.get_all()
        removed = 0
        for job in all_jobs:
            if job.status in (JobStatus.completed, JobStatus.failed) and job.completed_at:
                age = (now - job.completed_at).total_seconds()
                if age > max_age_seconds:
                    await self._queue.remove(job.job_id)
                    removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} finished jobs")

    def _schedule_warm_timeout(self) -> None:
        """Schedule check to unload adapter after timeout."""
        if self._warm_task:
            self._warm_task.cancel()
        self._warm_task = asyncio.create_task(self._warm_timeout_check())

    def _contested_slots(self, pending: list[Job]) -> list[Instance]:
        """Loaded slots a pending job needs for a *different* model.

        Keep-warm key is the recipe: a pending job for it is served by the
        loaded adapter, whichever of the recipe's modalities it asks for, so it
        stays warm. A job that wants
        the same card for something else is what makes unloading urgent — and
        a job destined for another card is not, which is the whole point of
        having a slot per card.
        """
        contested = []
        for slot in self._slots.values():
            for job in pending:
                key = job.request.model
                if key == slot.key:
                    continue
                if self._device_for(key) == slot.device:
                    contested.append(slot)
                    break
        return contested

    async def _warm_timeout_check(self) -> None:
        """Unload idle sleepy adapters, or evict one a queued job is waiting on.

        A pending job for a different model on the same card triggers an
        immediate unload of that card's slot so the next load starts sooner;
        otherwise slots are unloaded once the warm timeout has passed with
        nothing queued for them.
        """
        # Never unload while actively processing a job
        if self._processing_job:
            return

        pending = await self._queue.get_pending()
        if pending and self._slots:
            contested = self._contested_slots(pending)
            if contested:
                async with self._worker_lock:
                    # Re-validate state under lock: run-loop may have already
                    # switched adapters while we were waiting to acquire.
                    if self._processing_job:
                        return
                    pending = await self._queue.get_pending()
                    for slot in self._contested_slots(pending):
                        logger.info(
                            f"Unloading {slot.model} immediately: "
                            "a queued job needs its card for another model"
                        )
                        await self._unload_worker(slot.device)
                return

        # No other adapters waiting - apply warm timeout
        await asyncio.sleep(WARM_TIMEOUT_SECONDS)

        # Re-check after sleep - might be processing now
        if self._processing_job:
            return

        if self._last_job_time:
            elapsed = (datetime.now() - self._last_job_time).total_seconds()
            if elapsed >= WARM_TIMEOUT_SECONDS:
                async with self._worker_lock:
                    # Re-validate under lock; run-loop may have claimed a job.
                    if self._processing_job:
                        return
                    if not self._last_job_time:
                        return
                    elapsed = (datetime.now() - self._last_job_time).total_seconds()
                    if elapsed < WARM_TIMEOUT_SECONDS:
                        return
                    pending = await self._queue.get_pending()
                    if pending:
                        return
                    logger.info("Warm timeout reached, unloading adapter(s)")
                    await self._unload_worker()


# Global runner recipe
_runner: Runner | None = None


def get_runner(
    residents: list[str] | None = None,
    use_policy: bool = False,
) -> Runner:
    """Get or create global runner.

    Both arguments only apply on first construction (the service lifespan
    passes ``use_policy=True``); later calls return the existing recipe.
    ``residents`` pins a static set instead, which is the legacy/test path.
    """
    global _runner
    if _runner is None:
        _runner = Runner(residents=residents, use_policy=use_policy)
    return _runner
