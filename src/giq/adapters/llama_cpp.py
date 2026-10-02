# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""LLM worker using llama.cpp server."""

import asyncio
import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import httpx

from giq import recipes
from giq.adapters.engine import Concurrency, ServedLLM, StartError, open_engine_log
from giq.gpus import device_env, device_port, server_port
from giq.loopguard import LoopGuard
from giq.models import JobResult
from giq.paths import model_path as resolve_path
from giq.paths import state_dir
from giq.recipes.schema import LlamaCppParams, Recipe
from giq.registry import vram_for

if TYPE_CHECKING:
    from giq.queue import JobStream

logger = logging.getLogger(__name__)


def weights_installed(model: str) -> bool:
    """Are this model's weights actually on disk?

    A registry entry is a declaration, not a guarantee: five models were
    registered with paths whose GGUFs had been deleted, leaving empty
    directories behind. giq happily advertised them and llama-server would
    have failed on the first request. Registered means "giq knows how to run
    this"; installed means "the file is here".
    """
    from giq.adapters.engine import engine_for

    if engine_for(model) == "vllm":
        from giq.adapters.vllm import weights_installed as vllm_weights_installed

        return vllm_weights_installed(model)
    path = resolved_model_path(model)
    return bool(path) and Path(path).exists()


def resolved_model_path(model: str) -> str | None:
    """The absolute GGUF path for a registered model, or None."""
    raw = MODEL_PATHS.get(model)
    return resolve_path(raw) if raw else None


# --- per-model settings, from the recipe files -----------------------------
#
# Every llm recipe (giq/recipes/llm.*.yaml, and the operator's files in
# giq.paths.recipes_dir()) sets some llama-server parameters; the MODEL_*
# tables in this module hold exactly what the files set, keyed by model name.
# A model absent from a table takes the DEFAULT_* beside it. The tables are
# refilled in place when the recipes are reloaded, so a module that imported
# one keeps a live reference. Why a model runs with the values it does is
# written in its recipe file; what stays here are the defaults and the
# engine knowledge that holds for every model.


def tables_of(declared: Iterable[Recipe]) -> dict[str, dict]:
    """The MODEL_* tables, as a set of recipes declares them.

    Sparse like the tables: a model appears in one only when its file sets
    that parameter, so everything unset keeps taking the DEFAULT_*.
    """
    llm = [i for i in declared if i.serves("llm") and isinstance(i.params, LlamaCppParams)]

    def given(param: str) -> dict:
        return {i.name: i.params.given()[param] for i in llm if param in i.params.given()}

    return {
        "MODEL_PATHS": {i.name: i.weights.path for i in llm if i.weights and i.weights.path},
        "MODEL_CTX_SIZE": given("ctx_size"),
        "MODEL_CACHE_TYPE_K": given("cache_type_k"),
        "MODEL_CACHE_TYPE_V": given("cache_type_v"),
        "MODEL_REASONING": given("reasoning"),
        "MODEL_REASONING_BUDGET": given("reasoning_budget"),
        "MODEL_REQUEST_DEFAULTS": {
            i.name: dict(i.request_defaults) for i in llm if i.request_defaults
        },
        "MODEL_SPEC_TYPE": given("spec_type"),
        "MODEL_PARALLEL": given("parallel"),
        "MODEL_LOOP_GUARD": given("loop_guard"),
        "MODEL_ALIAS": given("alias"),
        "MODEL_READY_TIMEOUT": given("ready_timeout"),
    }


# weights.path: relative to giq.paths.models_dir() (GIQ_MODELS_DIR, default
# ~/models); a `~` or absolute entry is used as written.
MODEL_PATHS: dict[str, str] = {}

# giq owns every llama-server it talks to. There is deliberately no setting that
# routes prompts to an external endpoint: a switch that silently sends every
# prompt to a host giq does not run has no place in a privacy-first service.

# The llama-server binary comes from giq.engines — one declared build for every
# model, never a per-model path or a PATH lookup that could split them.

# params.parallel: slots (-np). >1 implies --cont-batching. The slots share
# ctx_size — 4 slots at 128k is 32k per slot, and 4 at 8k would be 2k per slot,
# silently rejecting longer prompts.
DEFAULT_PARALLEL = 1
MODEL_PARALLEL: dict[str, int] = {}

# params.alias: --alias, the model id served via /v1/models.
MODEL_ALIAS: dict[str, str] = {}

# Internal port for llama-servers giq spawns itself. lifecycle._kill_stale_servers
# frees this port with fuser -k at startup, so nothing else may use it.
INTERNAL_LLM_PORT = 8086


DEFAULT_CTX_SIZE = 131072  # 128k, q4_0 KV cache

# params.ctx_size: the context llama-server allocates, shared by its slots.
MODEL_CTX_SIZE: dict[str, int] = {}

DEFAULT_REASONING = "off"  # llama-server --reasoning value

# params.reasoning: "on" exposes the model's native thinking channel; "off"
# suppresses it. "template" omits the --reasoning flag entirely and lets the
# model's chat template decide, which leaves callers a per-request opt-out via
# chat_template_kwargs.
MODEL_REASONING: dict[str, str] = {}

# KV cache quantization (llama-server --cache-type-k/--cache-type-v).
# q4_0 = aggressive 0.5 B/elem, saves VRAM, loses precision at long context.
# q8_0 = 1 B/elem, near-f16 quality, ~2× the KV memory.
# Defaults stay at q4_0 so high-context configs (256k) still fit; a recipe
# raises both to q8_0 when there is a quality reason.
# Ceiling on the *thought*, separate from max_tokens' ceiling on the whole
# response. When the budget is spent llama.cpp injects an end-of-thinking tag
# (optionally preceded by --reasoning-budget-message), so the model stops
# deliberating and writes its answer *from the thinking it has already done* —
# nothing is discarded. It is a deadline, not a truncation.
#
# What it prevents: measured, a prompt with no clean resolution ran
# the full 32,768 tokens, 102k characters of thought, in 573s, and returned
# finish_reason "length" with an empty answer. A budget would have turned that
# into an answer built on 24k tokens of deliberation.
#
# Deliberately empty anyway. This is a launch flag, so one number has to serve
# a 512-token request and a 65k-token one alike, and the same cap that rescues
# a thought going nowhere truncates one that was getting somewhere — the 102k
# think above read as fluent and on-track, not as looping. With finish_reason
# reported to the caller, hitting the ceiling is legible rather than silent,
# which is the cheaper half of the problem solved. A recipe sets
# params.reasoning_budget when that model demonstrably needs the deadline.
#
# (A per-request budget does exist, under a key that is easy to get wrong: the
# body field is `reasoning_budget_tokens` — `server-common.cpp:1354`, alias
# `thinking_budget_tokens`, -1 meaning "fall back to this flag". Any other key
# is silently ignored. The per-request form is what `request_defaults` in the
# recipe files use, which answers the objection above: one number no longer
# has to serve a 512-token request and a 65k one.)
DEFAULT_REASONING_BUDGET: int | None = None
MODEL_REASONING_BUDGET: dict[str, int] = {}

# K and V must be the SAME type. Benchmarked on Qwen3.8-27B, Q6_K
# (head_dim 256), prompt-processing throughput, generation unaffected:
#
#   q8_0 / q8_0     2949 tok/s     fused attention kernel
#   q4_0 / q4_0     2948 tok/s     fused attention kernel
#   q8_0 / f16       382 tok/s     7.7x slower
#   q8_0 / q4_0      235 tok/s     12.6x slower
#   q8_0 / iq4_nl     76 tok/s     38x slower
#
# Any mixed pair falls off the fused path onto something that ran at 0% GPU
# utilisation for minutes. So "keep K precise and economise on V" — the usual
# advice, since K errors compound through the softmax and V errors average out
# — is not available here at any context length. It is symmetric or nothing.
#
# The trap is that mixed pairs *look* cheaper: llama-fit-params reports a flat
# 505 MiB compute buffer for them against 1360 MiB for a symmetric pair at
# 256k. That is not a cheaper kernel, it is the absence of one.
DEFAULT_CACHE_TYPE_K = "q4_0"
DEFAULT_CACHE_TYPE_V = "q4_0"

# params.cache_type_k / cache_type_v: set together and equal (the schema
# refuses a mixed pair).
MODEL_CACHE_TYPE_K: dict[str, str] = {}
MODEL_CACHE_TYPE_V: dict[str, str] = {}


# --- what giq puts in the request body, under whatever the caller sent -------
#
# On its own giq sends three sampler fields and no others: temperature, top_p,
# max_tokens (openai_compat.py, and `complete` below). Everything else runs on
# llama.cpp's defaults, and two of those defaults matter here — `repeat_penalty
# 1.00` and `dry_multiplier 0.00` both mean *disabled*. Without the defaults
# below there is no repetition control anywhere in the stack, which is the
# configuration a thinking model needs to loop freely. A recipe's
# `request_defaults` supply the missing control.
#
# DRY ("Don't Repeat Yourself") is the right instrument for the observed
# failure. `repeat_penalty` and `frequency_penalty` penalise *tokens* wherever
# they appear, which damages ordinary long-form prose — every "the" is a
# repetition. DRY penalises only the *continuation of a sequence already seen*,
# exponentially in the length of the match, so a fresh sentence pays nothing
# and the fifth re-emission of a thirty-token run pays enormously. The looping
# fragments share a ~30-token verbatim run with one slot varying, which is
# exactly the shape it is built to catch. The Qwen recipes set it, with the
# per-field reasoning in llm.qwen3.8-27b.yaml.
#
# Two traps in llama.cpp, both verified against its source:
#
#   * `dry_penalty_last_n` defaults to 64 tokens — shorter than a single one of
#     those fragments, so the default is inert here even once DRY is on.
#   * `-1` does NOT mean "the whole context" the way it does in the sampler's
#     documentation elsewhere. `llama-sampler.cpp:3640` clamps it with
#     `std::max(n, 0)` and line 3645 treats 0 as disabled — so passing -1
#     silently turns DRY back off. It has to be an explicit number.
#
# Defaults, not overrides: `request_defaults` puts these *under* the caller's
# body, so anything a client actually sends still wins. One consequence worth
# stating — `temperature` deliberately does not appear in them. openai_compat
# gives it a default of 0.7 and therefore always sends it, so an entry there
# would never apply. Qwen's own guidance for thinking mode is 0.6; changing
# that means changing the API default, which is a decision about every model.
DEFAULT_REQUEST_DEFAULTS: dict = {}
MODEL_REQUEST_DEFAULTS: dict[str, dict] = {}


# --- ending a thought that is going nowhere ----------------------------------
#
# llama.cpp exposes a control plane for an in-flight completion:
# `POST /v1/chat/completions/control` with {"id", "action": "reasoning_end"}
# forces the end-of-thinking tag on the live slot *mid generation*
# (`server-context.cpp:2409` — "act on the live slot mid generation, never
# defer"), so the model stops deliberating and writes its answer from the
# thinking it has already done. No second request, no re-prefill, no KV
# discarded — which is what makes acting on a heuristic cheap enough to be
# worth doing at all.
#
# It only works if the completion was armed for it: `reasoning_control: true`
# on the original request, or the call comes back
# `{"success": false, "message": "reasoning control not enabled..."}`. Arming
# is free — with no budget set the sampler counts to INT_MAX
# (`sampling.cpp:317`) and never fires on its own.
CONTROL_ENDPOINT = "/v1/chat/completions/control"

# Prepended to the forced end-of-thinking tag, so the model sees a reason for
# the interruption rather than an abrupt cut (`reasoning_budget_message`,
# `server-schema.cpp:415`). It serves both triggers — the guard below and an
# exhausted `reasoning_budget_tokens` — so it is worded for either. It is also
# the one part of the intervention a reader can see: it lands in the thought,
# just before it ends.
REASONING_STOP_MESSAGE = (
    "\n\nI have enough to answer now, and further deliberation is going in "
    "circles rather than making progress. I will stop here and write the best "
    "answer I can from the reasoning above.\n\n"
)

# The loop guard watches the thinking channel on every model by default. It is
# not a behaviour change in the way a token cap is: it cannot fire before 6,000
# characters of thought, it needs two agreeing verdicts, and when it does fire
# the answer still gets written. A recipe sets params.loop_guard: false if
# its model turns out to think in a shape the measure reads wrong.
DEFAULT_LOOP_GUARD = True
MODEL_LOOP_GUARD: dict[str, bool] = {}


# --- speculative decoding -----------------------------------------------------
#
# llama.cpp's `--spec-type`. Speculative decoding drafts several tokens cheaply
# and then has the real model verify them in one pass, accepting only those it
# would have produced itself — so it buys speed without trading quality, which
# is what makes it the right lever here and NVFP4 the wrong one (4.5-bit weights
# against Q6_K's 6.56 is a quality trade however fast it runs).
#
# Qwen3.8-27B ships its own multi-token-prediction head — `nextn_predict_layers=1`
# in the GGUF, the `blk.64.nextn.*` tensors llama-server otherwise reports as
# "unused tensor ... ignoring" on every load. `draft-mtp` puts them to work; no
# separate draft model is loaded.
#
# Measured on Qwen3.8-27B, Q6_K, at 196608 ctx, q8_0 KV, one code-generation
# prompt:
#
#   profile                          tok/s   VRAM
#   baseline                          60.5   29476 MiB
#   draft-mtp, mmproj loaded          75.3   30412 MiB
#   draft-mtp, no mmproj             132.7   30156 MiB
#
# The middle row is why a vision model with an MTP head is registered as two
# entries rather than one flag. Loading the vision projector roughly halves the
# speedup — at identical context, so it is not a memory-pressure effect, and
# dropping mmproj gives back only 256 MiB. Where vision and the speedup do not
# coexist in one process, they are served as two profiles over one file: the
# vision recipe keeps mmproj and runs no --spec-type, a `-fast` recipe
# points weights.path at the same GGUF, carries no mmproj, and runs
# `draft-mtp`. They share a card, so asking for one evicts the other through
# the ordinary warm-swap path. (The penalty did not reproduce in a later A/B at 96k on a
# newer build; tests/test_streaming.py::SPEC_TYPE_WITH_VISION_ALLOWED is where
# a profile that pairs the two anyway has to show its numbers.)
#
# Losslessness is by construction, not by observation: at temperature 0 the
# text-only draft-mtp output matched the baseline for 775 of 802 characters and
# then took a different branch of an edge-case list. That is the expected
# signature of float non-associativity — verifying k tokens in one batch
# reorders reductions and can flip a near-tie argmax — and `ngram-mod`
# reproduced the baseline byte-for-byte on the same prompt. It is NOT proof, and
# is recorded as such.
#
# ngram-* variants are also available and cost nothing, but predict only tokens
# already present in the context, so they returned the baseline rate on the
# from-scratch prompt above. They are worth revisiting for summarise/edit/
# refactor work over a long document.
#
# No built-in recipe runs a speculative profile: the registered qwen3.8-27b
# build has not been measured under draft-mtp, and a profile goes in with its
# own measurement, not by analogy. params.spec_type sets it.
DEFAULT_SPEC_TYPE: str | None = None
MODEL_SPEC_TYPE: dict[str, str] = {}

# params.ready_timeout: seconds a llama-server start may take. Mapping a GGUF
# from page cache takes seconds and from an NVMe a 22 GB one takes about ten;
# a spinning disk or a network share takes minutes. The wait used to be a
# fixed 30 s, which a cold load on slow storage could not meet. A start that
# fails outright is caught when the process exits, not at the deadline, so
# a generous ceiling costs nothing in the common failure.
DEFAULT_READY_TIMEOUT = 300.0
MODEL_READY_TIMEOUT: dict[str, float] = {}

_TABLES: dict[str, dict] = {
    "MODEL_PATHS": MODEL_PATHS,
    "MODEL_CTX_SIZE": MODEL_CTX_SIZE,
    "MODEL_CACHE_TYPE_K": MODEL_CACHE_TYPE_K,
    "MODEL_CACHE_TYPE_V": MODEL_CACHE_TYPE_V,
    "MODEL_REASONING": MODEL_REASONING,
    "MODEL_REASONING_BUDGET": MODEL_REASONING_BUDGET,
    "MODEL_REQUEST_DEFAULTS": MODEL_REQUEST_DEFAULTS,
    "MODEL_SPEC_TYPE": MODEL_SPEC_TYPE,
    "MODEL_PARALLEL": MODEL_PARALLEL,
    "MODEL_LOOP_GUARD": MODEL_LOOP_GUARD,
    "MODEL_ALIAS": MODEL_ALIAS,
    "MODEL_READY_TIMEOUT": MODEL_READY_TIMEOUT,
}


def _refill(snapshot: recipes.Snapshot) -> None:
    """Replace every table's contents with what ``snapshot`` declares."""
    for name, values in tables_of(snapshot.recipes.values()).items():
        table = _TABLES[name]
        table.clear()
        table.update(values)


recipes.subscribe(_refill)


# --- how long giq waits on llama-server ---------------------------------------
#
# httpx's `timeout` is per-read, not per-response. For a streamed answer that is
# the gap between chunks — but the *first* chunk only exists once the prompt has
# been processed, and for a non-streamed answer the single read covers the whole
# generation. A flat 120s was fine when contexts were small. It is not fine at
# 256k: measured, prefill runs ~1400 tok/s over 32k chunks and gets
# slower as context grows (pp2048 benches at 2949 tok/s, so the rate roughly
# halves by 32k and keeps falling). A full-context prompt needs minutes before
# its first token exists, and would have died at 120s with nothing to show.
#
# So it scales with the model's own context, at a rate deliberately more
# pessimistic than runner.FLOOR_TOKENS_PER_SECOND. That ordering is the point:
# runner._job_timeout must always fire first, because it cancels *with cleanup*
# — it unloads a worker that may be wedged — whereas an httpx read timeout
# firing first would pre-empt that with a bare exception and leave the model
# holding VRAM.
HTTP_READ_FLOOR_SECONDS = 300.0
HTTP_READ_TOKENS_PER_SECOND = 15.0  # runner plans for 20; this must be slower
HTTP_CONNECT_SECONDS = 10.0  # llama-server is on loopback; this is generous


@dataclass
class LlamaCppConfig:
    """Configuration for LLM worker."""

    model: str
    model_path: str | None = None
    # None = pick from MODEL_CTX_SIZE/DEFAULT_CTX_SIZE in __post_init__.
    ctx_size: int | None = None
    # None = pick from MODEL_REASONING/DEFAULT_REASONING in __post_init__.
    reasoning: str | None = None
    # None = pick from MODEL_CACHE_TYPE_K/V or DEFAULT_CACHE_TYPE_K/V.
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    # None = giq.engines' llama.cpp build, and MODEL_PARALLEL/MODEL_ALIAS.
    binary: str | None = None
    parallel: int | None = None
    alias: str | None = None
    # Seconds a start may take; None = MODEL_READY_TIMEOUT/DEFAULT_READY_TIMEOUT.
    ready_timeout: float | None = None
    # Where llama-server's own output goes. A start that fails is otherwise
    # silent: its reason is on stderr.
    log_path: str | None = None
    n_gpu_layers: int = -1  # All layers on GPU
    host: str = "127.0.0.1"
    # GPU UUID this server runs on. None = the model's binding, or giq's
    # selected card. It picks the port too: two cards can each hold a resident
    # LLM, and they cannot both bind 8086.
    device: str | None = None
    port: int | None = None
    # llama.cpp's vision projector. None = text-only, or take the registry's.
    mmproj: str | None = None
    # Token ceiling on thinking; None leaves it unrestricted.
    reasoning_budget: int | None = None
    # llama.cpp --spec-type. None = no speculative decoding.
    spec_type: str | None = None

    def __post_init__(self):
        if self.device is None:
            from giq.vram import device_for_recipe

            self.device = device_for_recipe(self.model)
        if self.port is None:
            self.port = device_port(INTERNAL_LLM_PORT, self.device)
        if self.model_path is None:
            self.model_path = MODEL_PATHS.get(self.model)
        if self.model_path:
            self.model_path = resolve_path(self.model_path)
        if self.ctx_size is None:
            self.ctx_size = MODEL_CTX_SIZE.get(self.model, DEFAULT_CTX_SIZE)
        if self.reasoning is None:
            self.reasoning = MODEL_REASONING.get(self.model, DEFAULT_REASONING)
        if self.cache_type_k is None:
            self.cache_type_k = MODEL_CACHE_TYPE_K.get(self.model, DEFAULT_CACHE_TYPE_K)
        if self.cache_type_v is None:
            self.cache_type_v = MODEL_CACHE_TYPE_V.get(self.model, DEFAULT_CACHE_TYPE_V)
        if self.binary is None:
            from giq.engines import binary_for

            self.binary = binary_for("llama.cpp")
        if self.reasoning_budget is None:
            self.reasoning_budget = MODEL_REASONING_BUDGET.get(self.model, DEFAULT_REASONING_BUDGET)
        if self.spec_type is None:
            self.spec_type = MODEL_SPEC_TYPE.get(self.model, DEFAULT_SPEC_TYPE)
        if self.mmproj is None:
            from giq.registry import get_recipe

            recipe = get_recipe(self.model)
            self.mmproj = recipe.mmproj if recipe else None
        if self.mmproj:
            self.mmproj = resolve_path(self.mmproj)
        if self.parallel is None:
            self.parallel = MODEL_PARALLEL.get(self.model, DEFAULT_PARALLEL)
        if self.alias is None:
            self.alias = MODEL_ALIAS.get(self.model)
        if self.ready_timeout is None:
            self.ready_timeout = MODEL_READY_TIMEOUT.get(self.model, DEFAULT_READY_TIMEOUT)
        if self.log_path is None:
            self.log_path = str(state_dir() / "logs" / f"llama-{self.model}.log")


@dataclass
class LlamaCppAdapter(ServedLLM):
    """LLM worker wrapping llama.cpp server."""

    engine: ClassVar[str] = "llama.cpp"

    config: LlamaCppConfig
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _ready: bool = field(default=False, repr=False)

    @property
    def is_running(self) -> bool:
        """Check if worker process is running."""
        return self._process is not None and self._process.returncode is None

    @property
    def is_ready(self) -> bool:
        """Check if worker is ready to accept requests."""
        return self.is_running and self._ready

    @property
    def pid(self) -> int | None:
        """PID of the child holding this model's VRAM, for measuring what
        evicting it would actually free. None when there is no child."""
        return self._process.pid if self._process else None

    @property
    def estimated_vram_gb(self) -> float:
        """Estimated VRAM usage for this model."""
        return vram_for(self.config.model, default=15.0)

    @property
    def base_url(self) -> str:
        """Base URL for llama-server API. Always a server giq spawned itself."""
        return f"http://{self.config.host}:{self.config.port}"

    def concurrency(self) -> Concurrency:
        """D8 as llama.cpp means it: -np slots, each with an even share of
        the context. Reported, not yet used for dispatch — see
        ServedLLM.lanes_from_engine."""
        parallel = max(1, self.config.parallel or 1)
        ctx = self.config.ctx_size or DEFAULT_CTX_SIZE
        return Concurrency(
            max_parallel=parallel, per_request_context=ctx // parallel, shared_kv=False
        )

    def build_command(self) -> list[str]:
        """The argv for llama-server. Separate from start() so the flags a
        model runs with can be asserted without spawning anything — several of
        them (reasoning, its budget, the KV types) change what the model does,
        not just how fast it does it."""
        cmd = [
            self.config.binary,
            "-m",
            self.config.model_path,
            "-c",
            str(self.config.ctx_size),
            "-ngl",
            str(self.config.n_gpu_layers),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--cache-type-k",
            self.config.cache_type_k,
            "--cache-type-v",
            self.config.cache_type_v,
            "--log-disable",  # Reduce noise
            "-np",
            str(self.config.parallel),
            "--flash-attn",
            "on",  # Smaller compute buffers during long prompts
            "--jinja",  # Use model's native chat template; enables OpenAI tool-call parsing
            "--slots",  # /slots endpoint — the runner's always-on idleness signal
        ]
        if self.config.reasoning != "template":
            cmd += ["--reasoning", self.config.reasoning]
        if self.config.parallel > 1:
            cmd += ["--cont-batching"]
        if self.config.alias:
            cmd += ["--alias", self.config.alias]
        if self.config.reasoning_budget:
            cmd += ["--reasoning-budget", str(self.config.reasoning_budget)]
        if self.config.spec_type:
            cmd += ["--spec-type", self.config.spec_type]
        if self.config.mmproj:
            # Without this the weights load and serve text; image parts in a
            # request are accepted and then quietly ignored by the model.
            cmd += ["--mmproj", self.config.mmproj]

        return cmd

    def request_defaults(self) -> dict:
        """Body fields giq supplies for this model when the caller does not.

        Kept separate from the three call sites for the same reason
        build_command is separate from start: several of these change what the
        model does rather than how fast it does it, so the set a model runs
        with has to be assertable without spawning anything.
        """
        return dict(MODEL_REQUEST_DEFAULTS.get(self.config.model, DEFAULT_REQUEST_DEFAULTS))

    def http_timeout(self) -> httpx.Timeout:
        """Read budget for one call to llama-server, sized off this model's
        context. Callers that know better pass their own (the /slots poll wants
        2s, the reasoning-end control 10s); this is the one for generation."""
        read = max(
            HTTP_READ_FLOOR_SECONDS,
            (self.config.ctx_size or DEFAULT_CTX_SIZE) / HTTP_READ_TOKENS_PER_SECOND,
        )
        return httpx.Timeout(read, connect=HTTP_CONNECT_SECONDS)

    async def _force_reasoning_end(self, cmpl_id: str | None) -> bool:
        """End the thinking block of a completion that is still generating.

        Awaited inline rather than fired off as a task. The server processes a
        control task between decode batches and answers in milliseconds, and
        the pause it costs the read loop is invisible — llama-server keeps
        generating into the response buffer meanwhile. A background task would
        buy nothing and could outlive the request it belongs to.

        Never raises: this is a best-effort intervention on top of a generation
        that is already in trouble, and failing to improve it must not also
        destroy it. The budget in MODEL_REQUEST_DEFAULTS is the fallback.
        """
        if not cmpl_id or not self._client:
            return False
        try:
            resp = await self._client.post(
                CONTROL_ENDPOINT,
                json={"id": cmpl_id, "action": "reasoning_end"},
                timeout=10.0,
            )
        except httpx.HTTPError as e:
            logger.warning(f"reasoning_end control call failed: {e}")
            return False

        if resp.status_code >= 400:
            logger.warning(f"reasoning_end rejected: HTTP {resp.status_code}")
            return False
        try:
            body = resp.json()
        except ValueError:
            logger.warning("reasoning_end returned an undecodable body")
            return False
        if not body.get("success"):
            # The interesting failures both live here: not armed for control,
            # or a completion that has already left its thinking block.
            logger.warning(f"reasoning_end declined: {body.get('message')}")
            return False
        return True

    async def start(self) -> None:
        """Start the llama-server process."""
        if self.is_running:
            logger.info(f"LLM worker already running: {self.config.model}")
            return

        if not self.config.model_path:
            raise ValueError(f"No model path for: {self.config.model}")

        from giq.engines import require_binary

        require_binary("llama.cpp")

        await self._claim_port()
        cmd = self.build_command()

        logger.info(f"Starting LLM worker: {self.config.model}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Pinned to giq's card. llama.cpp's default -sm layer would otherwise
        # spread layers and KV across every visible GPU, putting part of the
        # model on a card the VRAM gate isn't even reading.
        log_path = Path(self.config.log_path or os.devnull)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open_engine_log(log_path) as log:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=device_env(self.config.device),
            )

        # Wait for server to be ready
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.http_timeout())
        try:
            await self._wait_for_ready(self.config.ready_timeout or DEFAULT_READY_TIMEOUT)
            self._ready = True
        except BaseException:
            # BaseException so CancelledError also reaps the subprocess
            # (shutdown mid-load would otherwise orphan llama-server).
            await self.stop()
            raise
        logger.info(f"LLM worker ready: {self.config.model}")

    async def active_slot_count(self) -> int | None:
        """Slots currently generating, or None when /slots is unavailable.

        None (endpoint disabled/unreachable — e.g. an external server started
        without --slots) is treated by callers as "idle": the always-on
        deferral cap backstops a wrong guess.
        """
        if not self._client:
            return None
        try:
            resp = await self._client.get("/slots", timeout=2.0)
            if resp.status_code != 200:
                return None
            slots = resp.json()
            return sum(
                1
                for s in slots
                if s.get("is_processing") or (isinstance(s.get("state"), int) and s["state"] != 0)
            )
        except (httpx.HTTPError, ValueError):
            return None

    async def _claim_port(self) -> None:
        """Move off the card's llama port when another llama-server holds it."""
        if self.config.port == device_port(INTERNAL_LLM_PORT, self.config.device):
            self.config.port = await asyncio.to_thread(
                server_port, self.config.port, self.config.device
            )

    def _log_tail(self, lines: int = 20) -> str:
        try:
            with open(self.config.log_path or os.devnull, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return ""

    async def _wait_for_ready(self, timeout: float, poll: float = 0.5) -> None:
        """Wait until /health answers, or fail with llama-server's last words.

        A process that exits during the start — a build that cannot read the
        model, a port already taken — fails at once rather than at the
        deadline, which is what lets the deadline be generous.
        """
        assert self._client is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise StartError(
                    f"llama-server exited with {self._process.returncode} while starting "
                    f"{self.config.model}:\n{self._log_tail()}"
                )
            try:
                response = await self._client.get("/health", timeout=5.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll)
        raise StartError(
            f"llama-server did not become ready within {timeout:.0f}s for "
            f"{self.config.model}:\n{self._log_tail()}"
        )

    async def stop(self) -> None:
        """Stop the llama-server process."""
        try:
            if self._client:
                await self._client.aclose()
        except Exception:
            logger.warning("Failed to close HTTP client")
        finally:
            self._client = None

        if self._process:
            pid = self._process.pid
            logger.info(f"Stopping LLM worker: {self.config.model} (pid={pid})")
            if self._process.returncode is not None:
                # Already exited (crashed, or killed externally) — nothing to
                # unload. Treating this as "unkillable" used to poison the
                # runner slot forever.
                logger.info(f"llama-server pid={pid} already exited, nothing to stop")
                self._process = None
                self._ready = False
                return
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=10.0)
            except (TimeoutError, ProcessLookupError):
                logger.warning(f"SIGTERM failed for pid={pid}, sending SIGKILL")
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except (TimeoutError, ProcessLookupError):
                    # Process survived SIGKILL (stuck in CUDA D state).
                    # Do NOT clear self._process — caller must know it's leaked.
                    logger.error(
                        f"llama-server pid={pid} survived SIGKILL (likely stuck in CUDA driver). "
                        f"VRAM is still held by this process."
                    )
                    raise RuntimeError(
                        f"Cannot unload model: llama-server pid={pid} is unkillable. "
                        f"Manual intervention required (GPU reset or reboot)."
                    ) from None
            self._process = None
            self._ready = False

    async def complete(
        self, system: str | None, user: str, params: dict | None = None
    ) -> tuple[str, dict]:
        """Run a single completion. Returns (text, usage) — usage is
        llama-server's token accounting ({prompt,completion,total}_tokens).

        Recognised giq-specific keys in `params` (translated, not forwarded as-is
        to llama-server):
          enable_thinking: bool — per-request override of the model's native
            thinking channel. Lets one-shot callers (e.g. structured-output
            pipelines) skip reasoning when --reasoning is on globally; the
            entire generation budget then goes to the actual answer.
        """
        if not self._ready or not self._client:
            raise RuntimeError("Worker not ready")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # No max_tokens: llama.cpp's n_predict then defaults to -1, i.e.
        # generate until EOS or the context is full. That is the right ceiling
        # — the model's own — and it is what callers get unless they ask for
        # less. A 2048 default here used to silently re-impose the small cap
        # even after the API layer stopped sending one.
        request_params: dict = {
            **self.request_defaults(),
            "messages": messages,
            "temperature": 0.7,
        }
        if params:
            params = dict(params)  # don't mutate caller's dict
            enable_thinking = params.pop("enable_thinking", None)
            if enable_thinking is not None:
                # Routed via the model's chat template (gemma-4, qwen3 expose
                # `enable_thinking` as a jinja var). Overrides --reasoning for
                # this single request.
                ctk = dict(params.pop("chat_template_kwargs", {}) or {})
                ctk["enable_thinking"] = bool(enable_thinking)
                request_params["chat_template_kwargs"] = ctk
            request_params.update(params)

        response = await self._client.post(
            "/v1/chat/completions",
            json=request_params,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"] or "", data.get("usage") or {}

    async def chat_completion(self, request_body: dict) -> dict:
        """Forward full OpenAI chat completion request to llama-server.

        Supports tools/function calling when llama-server is started with --jinja.
        """
        if not self._ready or not self._client:
            raise RuntimeError("Worker not ready")

        response = await self._client.post(
            "/v1/chat/completions",
            json={**self.request_defaults(), **request_body},
        )
        response.raise_for_status()
        return response.json()

    async def chat_completion_stream(self, request_body: dict, stream: "JobStream") -> dict:
        """Stream a chat completion from llama-server, chunk by chunk.

        Pushes each SSE chunk into `stream` as it arrives and returns the
        assembled answer in the ordinary non-streaming shape, so the job's
        stored result and token accounting are identical either way.

        Two things this buys beyond a live cursor. The httpx client is
        configured with a 120s timeout, which for a streaming response is the
        gap *between* chunks rather than the total — tokens arrive every ~17ms,
        so a nine-minute thought no longer trips it, while a genuinely wedged
        server still does. And a client that goes away sets `cancelled`, which
        breaks the loop and closes the connection; llama-server sees the
        disconnect and stops generating instead of finishing a thought nobody
        is waiting for.

        Third: this is the only place in giq that sees a thought *while it is
        being thought*, which is the only moment a loop can still be acted on.
        The non-streaming paths await a finished response, so for them the
        reasoning budget is the whole answer.
        """
        if not self._ready or not self._client:
            raise RuntimeError("Worker not ready")

        # Per-request opt-out, ahead of the per-model default. Popped rather
        # than forwarded: llama-server copies unrecognised body keys into its
        # own params, and giq's names are not its vocabulary.
        request_body = dict(request_body)
        want_guard = request_body.pop("giq_loop_guard", None)
        if want_guard is None:
            want_guard = MODEL_LOOP_GUARD.get(self.config.model, DEFAULT_LOOP_GUARD)
        guard = LoopGuard() if want_guard else None

        body = {
            **self.request_defaults(),
            **request_body,
            "stream": True,
            # Streaming responses carry no usage unless asked; without this the
            # job would be recorded with no tokens and the stats would quietly
            # under-count every streamed answer.
            "stream_options": {"include_usage": True},
        }
        if guard is not None:
            # Arming, not acting: this only creates the sampler that
            # _force_reasoning_end can later address. With no budget set it
            # counts to INT_MAX and never fires by itself.
            body["reasoning_control"] = True
            body.setdefault("reasoning_budget_message", REASONING_STOP_MESSAGE)

        content: list[str] = []
        reasoning: list[str] = []
        usage: dict = {}
        finish_reason: str | None = None
        head: dict = {}
        tool_calls: list[dict] = []
        acted: bool = False
        intervened: bool = False

        async with self._client.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code >= 400:
                # A streamed response has no body until read; without this the
                # error surfaces as ResponseNotRead instead of the real cause.
                await response.aread()
                response.raise_for_status()

            async for line in response.aiter_lines():
                if stream.is_cancelled:
                    logger.info(f"Stream cancelled by client: {self.config.model}")
                    break
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning(f"Undecodable SSE chunk from llama-server: {payload[:120]}")
                    continue

                if not head:
                    head = {k: chunk.get(k) for k in ("id", "created", "model")}
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    # Thinking arrives on its own channel, so a client can show
                    # the thought and the answer as different things. It is also
                    # the only channel a loop shows up on before the answer that
                    # never comes, which is what makes watching it here worth
                    # the arithmetic.
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                        # Kept fed after it trips, so the evidence it reports
                        # measures the whole thought and not just the part
                        # before the verdict.
                        if guard is not None and guard.feed(delta["reasoning_content"]):
                            if not acted:
                                acted = True
                                logger.warning(
                                    f"thought is looping on {self.config.model}, "
                                    f"ending it: {guard.evidence}"
                                )
                                intervened = await self._force_reasoning_end(head.get("id"))
                                if not intervened:
                                    # Deliberately no break. A guard that cannot
                                    # act has still only offered an opinion, and
                                    # tearing down the generation over it would
                                    # turn a recoverable answer into no answer at
                                    # all. The reasoning budget backstops it.
                                    logger.warning(
                                        "could not end the thought; "
                                        "leaving it to the reasoning budget"
                                    )
                    if delta.get("tool_calls"):
                        tool_calls.extend(delta["tool_calls"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                await stream.put(chunk)

        message: dict = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = tool_calls

        result: dict = {
            "id": head.get("id") or f"chatcmpl-{self.config.model}",
            "object": "chat.completion",
            "created": head.get("created") or int(time.time()),
            "model": head.get("model") or self.config.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": (
                        "cancelled" if stream.is_cancelled else (finish_reason or "stop")
                    ),
                }
            ],
            "usage": usage,
        }
        if guard is not None and guard.tripped:
            # Namespaced rather than spent on finish_reason, which stays the
            # server's own word. After a successful intervention the response
            # really did end normally and carries a real answer — calling that
            # anything else would be a lie told to every consumer of the stats
            # to make one event easier to count. This is giq's record of what
            # giq did, and it rides the stored result into the job history.
            #
            # Nothing is injected into the SSE stream for it either. The client
            # is reading llama-server's chunks verbatim, and a synthetic one
            # would be a non-standard field in an OpenAI-shaped response. What a
            # reader sees instead is REASONING_STOP_MESSAGE, which arrives on
            # the thinking channel in the ordinary way.
            result["giq_loop_guard"] = {"intervened": intervened, **guard.evidence}
        return result

    async def run_batch(self, tasks: list[dict], params: dict | None = None) -> list[JobResult]:
        """Run a batch of tasks sequentially."""
        if not self._ready:
            raise RuntimeError("Worker not ready")

        results: list[JobResult] = []

        for task_dict in tasks:
            # Handle both dict and Pydantic model inputs
            task_id = task_dict.get("id", "unknown")
            task_system = task_dict.get("system")
            task_user = task_dict.get("user", "")
            task_params_override = task_dict.get("params") or {}

            merged_params = {**(params or {}), **task_params_override}
            try:
                output, usage = await self.complete(
                    system=task_system,
                    user=task_user,
                    params=merged_params if merged_params else None,
                )
                results.append(
                    JobResult(
                        id=task_id,
                        output=output,
                        tokens=usage.get("total_tokens"),
                        tokens_in=usage.get("prompt_tokens"),
                        tokens_out=usage.get("completion_tokens"),
                    )
                )
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                results.append(JobResult(id=task_id, output="", error=str(e)))

        return results
