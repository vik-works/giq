# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The vllm engine: ``vllm serve`` as a second LLM engine next to llama.cpp.

What it is for. Vendor checkpoints — NVIDIA's ModelOpt NVFP4 and FP8 — run
natively on vllm and not on llama.cpp, and vllm's paged KV cache with
continuous batching serves many users from one context pool where
llama-server hands out fixed slots. Measured on an RTX 5090, NVIDIA's NVFP4
checkpoint of Qwen3.8-27B at 256 generated tokens per request:

    parallel   vllm, fp8 KV, budget 0.93      llama.cpp Q6_K through giq (-np 1)
       1        72 tok/s, TTFT 0.1 s           56 tok/s
       4       242 tok/s total                 56 tok/s total, TTFT median 9.5 s
      16       937 tok/s total, TTFT 0.2 s     56 tok/s total, TTFT median 37 s

What it costs. A cold start of minutes rather than seconds (192 s the first
time, 64 s of it CUDA graph capture; 82 s warm), and a process that claims
its whole VRAM budget up front. So a vllm model is meant to be resident, and
it must be given a budget (ADR-002, D3), because vllm otherwise takes 90 % of
the card whatever giq's gate believes. The budget is preferably a KV cache
size in bytes (``kv_cache_memory``): a fraction of the card means 29 GB on a
32 GB card and 89 GB on a 96 GB one, starving every model that is meant to
share it. giq's VRAM figure for the model is then weights + KV + a measured
overhead, the same on any card.

How it runs. The engine has its own interpreter (``envs/vllm``: torch 2.13 on
CUDA 13, a step ahead of giq's venv) and is spawned as ``vllm serve`` on a
loopback port, which giq proxies exactly as it proxies llama-server. Three
things about the process are not optional:

* **A memory ceiling.** FlashInfer JIT-compiles kernels on first use, one
  ``cicc`` per ninja job at 4-10 GB of RAM each. Uncapped, one such start
  exhausted a 61 GB desktop and took the session down with it. The server
  runs in a transient systemd scope with ``MemoryMax`` and no swap, so the
  kernel's OOM killer can only ever pick the engine; ``MAX_JOBS=2`` keeps a
  compile inside it. ``giq prepare vllm`` builds the big GEMM kernels once,
  ahead of time, so a start never has to.
* **The CUDA 13.0 toolchain from its own env.** ``CUDA_HOME`` points at the
  env's ``nvidia/cu13`` wheels, pinned to the runtime's minor version (see
  envs/vllm/pyproject.toml): the system nvcc is older than the headers, and a
  newer one is rejected by them.
* **Nothing phones home.** vllm reports usage statistics by default; giq is
  a local service and turns that off, and runs the hub offline so a
  missing file fails instead of downloading.

Parameters are declared once in the recipe schema (``VllmParams``, D2),
recipes pick a profile and override values (D9), and the checkpoint is
inspected for what it can do before anything starts: speculative decoding
with MTP needs the head in the weights.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from giq.adapters.engine import Concurrency, ServedLLM, StartError, open_engine_log
from giq.gpus import compute_capability, device_env, device_port, resolve_device, server_port
from giq.models import JobResult
from giq.paths import cache_dir, model_path, state_dir
from giq.recipes.schema import Recipe
from giq.registry import vram_for
from giq_vllm.params import VllmParams, mtp_layers, read_hf_config

if TYPE_CHECKING:
    from giq.queue import JobStream

logger = logging.getLogger(__name__)

ENGINE = "vllm"

# Internal port for vllm servers giq spawns, one per card like llama-server's
# 8086 (device_port adds the card's stride). 8088 sits between the llama and
# sd series of every card and collides with neither.
INTERNAL_VLLM_PORT = 8088

# SIGTERM makes vllm shut its engine core down and release the card; that
# takes a few seconds, more with a request in flight.
STOP_GRACE_SECONDS = 30.0
KILL_WAIT_SECONDS = 10.0

# The RAM ceiling on the engine process: 2 compile jobs at up to ~15 GB each
# plus the server itself. A machine with less RAM lowers it per recipe.
DEFAULT_MEMORY_MAX = "40G"

# FlashInfer's JIT parallelism, inside the ceiling above.
JIT_MAX_JOBS = 2


class VLLMConfigError(ValueError):
    """A recipe the vllm adapter refuses: bad parameters or weights that
    cannot do what the parameters ask."""


# --- the recipe ---------------------------------------------------------------
#
# Parameters, profiles and their validation live in the recipe schema
# (giq.recipes.schema.VllmParams, ENGINE_PROFILES["vllm"]): a recipe file
# is checked when it is loaded, not when the server starts. What stays here
# is turning a validated recipe into a running server.


def concurrency_of(params: VllmParams) -> Concurrency:
    """D8: max_num_seqs is a ceiling over one paged KV pool that every request
    shares, so each of them may use the whole window."""
    return Concurrency(
        max_parallel=params.max_num_seqs,
        per_request_context=params.max_model_len,
        shared_kv=True,
    )


def recipe_for(model: str) -> Recipe | None:
    """The vllm recipe clients reach as ``model`` (a name or an alias)."""
    from giq.registry import get_recipe

    recipe = get_recipe(model)
    return recipe if recipe is not None and recipe.engine == ENGINE else None


def weights_dir(recipe: Recipe) -> str:
    assert recipe.weights is not None and recipe.weights.path is not None
    return model_path(recipe.weights.path)


def weights_installed(model: str) -> bool:
    recipe = recipe_for(model)
    return recipe is not None and (Path(weights_dir(recipe)) / "config.json").is_file()


def check_checkpoint(weights: str | Path, params: VllmParams) -> dict[str, Any]:
    """Refuse at start what would otherwise fail minutes into it; returns the
    checkpoint's config.json.

    vllm reads Hugging Face directories (safetensors, ModelOpt), not GGUF
    files (D4 ``accepts``). Speculative MTP needs the head in the weights:
    without one vllm fails after loading them. The recipe loader checks the
    same when the weights are present at load; this is the check that always
    runs.
    """
    path = Path(weights)
    if path.suffix == ".gguf" or path.is_file():
        raise VLLMConfigError(f"{weights}: vllm serves a checkpoint directory, not a file")
    hf_config = read_hf_config(path)
    if hf_config is None:
        raise VLLMConfigError(f"{weights}: no readable config.json — not a Hugging Face checkpoint")
    spec = params.speculative
    if spec is not None and spec.method == "mtp" and mtp_layers(hf_config) == 0:
        raise VLLMConfigError(
            f"{weights}: speculative mtp asked for, but the checkpoint's config.json "
            "declares no MTP head"
        )
    return hf_config


# --- how to start it safely (D9) ---------------------------------------------------


def flashinfer_arch(capability: str) -> str:
    """FLASHINFER_CUDA_ARCH_LIST for one card, the way FlashInfer itself
    would spell it: 12.0 builds the family target (12.0f), other Blackwell
    and Hopper parts their arch-specific one (10.0a, 12.1a, 9.0a)."""
    major_text, _, minor_text = capability.strip().partition(".")
    major, minor = int(major_text), int(minor_text or 0)
    if major == 12 and minor == 0:
        return "12.0f"
    if major >= 9:
        return f"{major}.{minor}a"
    return f"{major}.{minor}"


def env_root(python: str | None = None) -> Path:
    """The vllm env's prefix (``envs/vllm/.venv``), from its interpreter.

    Not ``resolve()``d: the venv's python is a symlink to the uv-managed
    interpreter, and the prefix that matters is the venv's own.
    """
    if python is None:
        from giq.engines import binary_for

        python = binary_for(ENGINE)
    return Path(python).parent.parent


def cuda_home(root: Path) -> Path:
    """The CUDA 13 toolkit the env's nvidia wheels provide (nvcc, headers, cicc)."""
    found = sorted(root.glob("lib/python3*/site-packages/nvidia/cu13"))
    return found[0] if found else root / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"


def engine_env(device: str | None, python: str | None = None) -> dict[str, str]:
    """The process environment every vllm process giq starts shares — the
    server and ``giq prepare vllm`` alike, so the kernels prepare builds are
    the ones the server finds."""
    root = env_root(python)
    home = cuda_home(root)
    env = device_env(device)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_HOME"] = str(home)
    env["PATH"] = os.pathsep.join([str(root / "bin"), str(home / "bin"), env.get("PATH", "")])
    env["HF_HUB_OFFLINE"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["MAX_JOBS"] = str(JIT_MAX_JOBS)
    capability = compute_capability(device)
    if capability:
        env["FLASHINFER_CUDA_ARCH_LIST"] = flashinfer_arch(capability)
    else:
        logger.warning("vllm: compute capability unknown; FlashInfer will probe every visible card")
    cache = cache_dir()
    if cache is not None:
        # FlashInfer keys its cache off a workspace base (default $HOME, then
        # .cache/flashinfer under it), not XDG_CACHE_HOME; vllm's own
        # compile cache does follow XDG. Both land under giq's cache dir, a
        # set variable still winning.
        env.setdefault("FLASHINFER_WORKSPACE_BASE", str(cache))
        env.setdefault("VLLM_CACHE_ROOT", str(cache / "vllm"))
        # vllm keeps a config directory under ~/.config, which a hardened
        # unit makes read-only.
        env.setdefault("VLLM_CONFIG_ROOT", str(cache / "vllm" / "config"))
    return env


_user_systemd: bool | None = None


def user_systemd_available() -> bool:
    """Can this process create transient user scopes? Asked once.

    A desktop session has a user manager; a system service's user usually
    does not (no lingering, no session bus), and systemd-run then fails.
    """
    global _user_systemd
    if _user_systemd is None:
        ok = False
        if shutil.which("systemd-run") and shutil.which("systemctl"):
            try:
                ok = (
                    subprocess.run(
                        ["systemctl", "--user", "show-environment"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    ).returncode
                    == 0
                )
            except (OSError, subprocess.SubprocessError):
                ok = False
        _user_systemd = ok
    return _user_systemd


def scope_unit(port: int) -> str:
    """The transient unit a server runs in: one per port, so a stale one
    left by a crashed giq is found and stopped by name."""
    return f"giq-vllm-{port}"


def memory_cap_prefix(unit: str, memory_max: str | None) -> list[str]:
    """argv prefix that runs a command under a RAM ceiling, or [] without one.

    A transient scope is the only ceiling that works here: RLIMIT_AS caps
    virtual address space, and a CUDA process maps far more of that than it
    ever touches, so it cannot even initialise under a limit that means
    anything. Without a user systemd there is no ceiling and giq says so.
    """
    if not memory_max:
        return []
    if not user_systemd_available():
        logger.warning(
            f"vllm: no user systemd to cap RAM at {memory_max}; running uncapped — a kernel "
            "compile on first use can exhaust the machine's memory (run `giq prepare vllm` first, "
            "or set MemoryMax on giq's own unit)"
        )
        return []
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        f"--unit={unit}",
        "-p",
        f"MemoryMax={memory_max}",
        "-p",
        "MemorySwapMax=0",
        "--",
    ]


def stop_scope(unit: str) -> None:
    """Stop a scope and everything still in it. Harmless when there is none."""
    if not user_systemd_available():
        return
    for args in (["stop", f"{unit}.scope"], ["reset-failed", f"{unit}.scope"]):
        try:
            subprocess.run(
                ["systemctl", "--user", *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=STOP_GRACE_SECONDS + KILL_WAIT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"vllm: systemctl {args[0]} {unit}.scope failed: {e}")


def _metric(text: str, name: str) -> float:
    """Sum of a Prometheus gauge across its label sets."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and line[len(name) : len(name) + 1] in ("{", " "):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return total


# Fields giq or its llama.cpp path puts in a request body that are not vllm's
# vocabulary. vllm ignores unknown keys with a warning per request, and the
# DRY sampler it lacks would read as if it were applied.
_LLAMA_ONLY_KEYS = (
    "giq_loop_guard",
    "reasoning_budget_tokens",
    "thinking_budget_tokens",
    "reasoning_control",
    "reasoning_budget_message",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
)


def _normalise_reasoning(message: dict) -> None:
    """vllm names the thinking channel ``reasoning``; giq's API and its
    llama.cpp path say ``reasoning_content``. One name for every engine, so a
    client never has to know which one answered."""
    if "reasoning" in message:
        value = message.pop("reasoning")
        if value is not None and not message.get("reasoning_content"):
            message["reasoning_content"] = value


@dataclass
class VllmConfig:
    """How one vllm model is started. Everything is derived from its recipe."""

    model: str
    recipe: Recipe | None = None
    device: str | None = None
    port: int | None = None
    host: str = "127.0.0.1"
    python: str | None = None
    # Where the server's own output goes: a failed start is otherwise
    # invisible, and minutes long.
    log_path: str | None = None
    # Whether the checkpoint carries a vision tower; None = read it from
    # config.json.
    multimodal: bool | None = None
    # Seconds a start may take; None = the recipe's ready_timeout. The
    # warm-up in `giq prepare vllm --recipe` waits out a first start's
    # compiles with a longer one.
    ready_timeout: float | None = None

    def __post_init__(self):
        if self.recipe is None:
            self.recipe = recipe_for(self.model)
        if self.recipe is None:
            raise VLLMConfigError(f"{self.model}: no vllm recipe of that name")
        # The name vllm serves is the recipe's, even when a client asked by
        # an alias.
        self.model = self.recipe.name
        if self.device is None:
            from giq.vram import device_for_recipe

            self.device = device_for_recipe(self.model)
        if self.port is None:
            self.port = device_port(INTERNAL_VLLM_PORT, self.device)
        if self.python is None:
            from giq.engines import binary_for

            self.python = binary_for(ENGINE)
        if self.log_path is None:
            self.log_path = str(state_dir() / "logs" / f"vllm-{self.model}.log")
        if self.multimodal is None:
            hf_config = read_hf_config(self.weights) or {}
            self.multimodal = "vision_config" in hf_config

    @property
    def params(self) -> VllmParams:
        assert self.recipe is not None
        params = self.recipe.params
        assert isinstance(params, VllmParams)
        return params

    @property
    def weights(self) -> str:
        assert self.recipe is not None
        return weights_dir(self.recipe)


@dataclass
class VllmAdapter(ServedLLM):
    """An LLM served by ``vllm serve`` in a RAM-capped scope on one card."""

    engine: ClassVar[str] = ENGINE
    lanes_from_engine: ClassVar[bool] = True

    config: VllmConfig
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _ready: bool = field(default=False, repr=False)
    _scoped: bool = field(default=False, repr=False)

    # --- identity -------------------------------------------------------------

    @property
    def recipe(self) -> Recipe:
        assert self.config.recipe is not None
        return self.config.recipe

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def is_ready(self) -> bool:
        return self.is_running and self._ready

    @property
    def pid(self) -> int | None:
        """The API server's pid. The VRAM is held by its engine-core child,
        so a per-process reading of this pid finds nothing and the eviction
        planner uses the declared figure — which, for an engine that claims a
        fixed budget, is the right one anyway."""
        return self._process.pid if self._process else None

    @property
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=self.recipe.vram.gb)

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def scope(self) -> str:
        assert self.config.port is not None
        return scope_unit(self.config.port)

    def concurrency(self) -> Concurrency:
        return concurrency_of(self.config.params)

    def http_timeout(self) -> httpx.Timeout:
        """Same rule as llama-server's: the read budget follows the context,
        slower than the runner's own plan so its timeout fires first."""
        from giq.adapters.llama_cpp import (
            HTTP_CONNECT_SECONDS,
            HTTP_READ_FLOOR_SECONDS,
            HTTP_READ_TOKENS_PER_SECOND,
        )

        read = max(
            HTTP_READ_FLOOR_SECONDS,
            self.config.params.max_model_len / HTTP_READ_TOKENS_PER_SECOND,
        )
        return httpx.Timeout(read, connect=HTTP_CONNECT_SECONDS)

    # --- the command line -------------------------------------------------------

    @property
    def executable(self) -> str:
        """The env's ``vllm`` console script, next to its interpreter."""
        assert self.config.python is not None
        return str(Path(self.config.python).parent / "vllm")

    def startup_utilization(self, card_total_gb: float | None) -> float | None:
        """The ``--gpu-memory-utilization`` that goes with a KV budget in bytes.

        vllm ignores the fraction for sizing once it has a byte budget, but
        still refuses to start unless free memory covers ``fraction × card``
        — at its default of 0.9, a card another model already shares would
        never qualify. The model's own VRAM figure over the card's size makes
        that check mean what it should: is there room for this model.
        """
        p = self.config.params
        if p.gpu_memory_utilization is not None:
            return p.gpu_memory_utilization
        if not card_total_gb:
            return None
        return round(min(0.98, max(0.01, self.recipe.vram.gb / card_total_gb)), 3)

    def build_command(self, card_total_gb: float | None = None) -> list[str]:
        """argv for ``vllm serve``, from validated parameters only (D2, D4)."""
        p = self.config.params
        cmd = [
            self.executable,
            "serve",
            self.config.weights,
            "--served-model-name",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--max-model-len",
            str(p.max_model_len),
            "--kv-cache-dtype",
            p.kv_cache_dtype,
            "--max-num-seqs",
            str(p.max_num_seqs),
            # One line per request otherwise, with nothing giq would read.
            "--disable-uvicorn-access-log",
        ]
        if p.kv_cache_memory_bytes is not None:
            cmd += ["--kv-cache-memory-bytes", str(p.kv_cache_memory_bytes)]
        utilization = self.startup_utilization(card_total_gb)
        if utilization is not None:
            cmd += ["--gpu-memory-utilization", f"{utilization:g}"]
        if p.max_num_batched_tokens is not None:
            cmd += ["--max-num-batched-tokens", str(p.max_num_batched_tokens)]
        if p.speculative is not None:
            spec = {"method": p.speculative.method, "num_speculative_tokens": p.speculative.tokens}
            cmd += ["--speculative-config", json.dumps(spec)]
        if p.enforce_eager:
            cmd += ["--enforce-eager"]
        if self.config.multimodal and "vision" not in self.recipe.capabilities:
            # A multimodal checkpoint served as text: the vision tower is not
            # loaded, and its memory profile does not eat into the budget.
            cmd += ["--language-model-only"]
        if p.reasoning_parser:
            cmd += ["--reasoning-parser", p.reasoning_parser]
        if p.tool_call_parser:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", p.tool_call_parser]
        if p.chat_template_file:
            # A path (vllm reads the file); resolved like weights.path.
            # A literal template string is read verbatim instead, so a value
            # without jinja markers only errors at start, not here.
            cmd += ["--chat-template", model_path(p.chat_template_file)]
        # Constrained decoding: name the grammar backend and forbid the free
        # inter-token whitespace that otherwise lets a large json_schema diverge
        # (schema.StructuredOutputs). vllm takes the whole config as one JSON arg.
        so = p.structured_outputs
        cmd += [
            "--structured-outputs-config",
            json.dumps(
                {
                    "backend": so.backend,
                    "disable_any_whitespace": so.disable_any_whitespace,
                }
            ),
        ]
        return cmd

    def build_env(self) -> dict[str, str]:
        return engine_env(self.config.device, self.config.python)

    def request_defaults(self) -> dict:
        return dict(self.recipe.request_defaults)

    def _body(self, request_body: dict) -> dict:
        body = {**self.request_defaults(), **request_body}
        for key in _LLAMA_ONLY_KEYS:
            body.pop(key, None)
        return body

    # --- lifecycle --------------------------------------------------------------

    def _log_tail(self, lines: int = 20) -> str:
        try:
            text = Path(self.config.log_path or "").read_text(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    async def start(self) -> None:
        if self.is_running:
            logger.info(f"vllm worker already running: {self.config.model}")
            return

        params = self.config.params
        if params.gpu_memory_utilization is None and params.kv_cache_memory_bytes is None:
            raise VLLMConfigError(
                f"{self.config.model}: no VRAM budget"
            )  # validate_params refuses it
        check_checkpoint(self.config.weights, params)

        from giq.engines import require_binary

        require_binary(ENGINE)
        if not os.path.exists(self.executable):
            raise FileNotFoundError(
                f"{self.executable} not found — run `uv sync` in envs/vllm (docs/engines.md)"
            )

        gpu = await asyncio.to_thread(resolve_device, self.config.device)
        # Before the scope: its name follows the port, and a second vllm on
        # this card must neither bind the first one's port nor stop its scope
        # as if it were left over from a crashed run.
        if self.config.port == device_port(INTERNAL_VLLM_PORT, self.config.device):
            self.config.port = await asyncio.to_thread(
                server_port, self.config.port, self.config.device
            )
        cmd = self.build_command(gpu.vram_total_gb if gpu is not None else None)
        env = self.build_env()
        await asyncio.to_thread(stop_scope, self.scope)  # a stale one from a crashed run
        prefix = await asyncio.to_thread(memory_cap_prefix, self.scope, params.memory_max)
        self._scoped = bool(prefix)

        log_path = Path(self.config.log_path or os.devnull)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Starting vllm: {self.config.model} on port {self.config.port}"
            f"{f' (RAM cap {params.memory_max})' if prefix else ''}, log {log_path}"
        )
        logger.debug(f"Command: {' '.join(prefix + cmd)}")
        with open_engine_log(log_path) as log:
            # A session of its own: vllm forks its engine core, and stop()
            # signals the whole group so no child is left holding the card.
            self._process = await asyncio.create_subprocess_exec(
                *prefix,
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.http_timeout())
        started = time.monotonic()
        try:
            await self._wait_for_ready(self.config.ready_timeout or params.ready_timeout)
            self._ready = True
        except BaseException:
            await self.stop()
            raise
        logger.info(f"vllm ready: {self.config.model} after {time.monotonic() - started:.0f}s")

    async def _wait_for_ready(self, timeout: float, poll: float = 1.0) -> None:
        """/health answers once the engine is up; /v1/models naming this model
        is the proof the API layer serves it. A process that exits during the
        start fails at once, with its last words, instead of after the timeout."""
        assert self._client is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise StartError(
                    f"vllm exited with {self._process.returncode} while starting "
                    f"{self.config.model}:\n{self._log_tail()}"
                )
            try:
                health = await self._client.get("/health", timeout=5.0)
                if health.status_code == 200:
                    models = await self._client.get("/v1/models", timeout=5.0)
                    if models.status_code == 200:
                        ids = {m.get("id") for m in models.json().get("data", [])}
                        if self.config.model in ids:
                            return
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(poll)
        raise StartError(
            f"vllm did not become ready within {timeout:.0f}s for {self.config.model}:\n"
            f"{self._log_tail()}"
        )

    def _signal_group(self, sig: int) -> bool:
        """Signal the server's whole process group. False once it is empty."""
        assert self._process is not None
        try:
            os.killpg(self._process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    async def _wait_group_gone(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._signal_group(0):
                return True
            await asyncio.sleep(0.2)
        return not self._signal_group(0)

    async def stop(self) -> None:
        """SIGTERM the group, then SIGKILL, then the scope — the card is only
        free once the engine core is gone, not just the API server."""
        try:
            if self._client:
                await self._client.aclose()
        except Exception:
            logger.warning("Failed to close vllm HTTP client")
        finally:
            self._client = None

        process = self._process
        if process is None:
            return
        pid = process.pid
        logger.info(f"Stopping vllm: {self.config.model} (pid={pid})")
        if process.returncode is None:
            self._signal_group(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=STOP_GRACE_SECONDS)
            except TimeoutError:
                logger.warning(f"vllm pid={pid} ignored SIGTERM, sending SIGKILL")
                self._signal_group(signal.SIGKILL)
                try:
                    await asyncio.wait_for(process.wait(), timeout=KILL_WAIT_SECONDS)
                except TimeoutError:
                    logger.error(f"vllm pid={pid} survived SIGKILL (likely stuck in the driver)")
                    raise RuntimeError(
                        f"Cannot unload model: vllm pid={pid} is unkillable. "
                        f"Manual intervention required (GPU reset or reboot)."
                    ) from None
        # The API server is gone; its engine core may still be finishing.
        if not await self._wait_group_gone(KILL_WAIT_SECONDS):
            logger.warning(f"vllm pid={pid}: children outlived the server, killing them")
            self._signal_group(signal.SIGKILL)
            await self._wait_group_gone(KILL_WAIT_SECONDS)
        if self._scoped:
            await asyncio.to_thread(stop_scope, self.scope)
        self._process = None
        self._ready = False
        self._scoped = False

    # --- serving ----------------------------------------------------------------

    async def active_slot_count(self) -> int | None:
        """Requests running or waiting, from vllm's Prometheus metrics."""
        if not self._client:
            return None
        try:
            resp = await self._client.get("/metrics", timeout=2.0)
            if resp.status_code != 200:
                return None
            text = resp.text
        except httpx.HTTPError:
            return None
        busy = _metric(text, "vllm:num_requests_running") + _metric(
            text, "vllm:num_requests_waiting"
        )
        return int(busy)

    async def chat_completion(self, request_body: dict) -> dict:
        if not self._ready or not self._client:
            raise RuntimeError("Worker not ready")
        response = await self._client.post("/v1/chat/completions", json=self._body(request_body))
        if response.status_code >= 400:
            raise RuntimeError(f"vllm HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        for choice in data.get("choices") or []:
            if isinstance(choice.get("message"), dict):
                _normalise_reasoning(choice["message"])
        return data

    async def chat_completion_stream(self, request_body: dict, stream: JobStream) -> dict:
        """Relay vllm's SSE chunks, normalised to giq's shape, and return the
        assembled answer as a non-streamed response — same contract as
        LlamaCppAdapter, so the job's stored result and token accounting match.

        The loop guard is llama.cpp's: it ends a looping thought through
        llama-server's control endpoint, which vllm has no counterpart for.
        A caller's giq_loop_guard is accepted and has no effect here.
        """
        if not self._ready or not self._client:
            raise RuntimeError("Worker not ready")
        body = {
            **self._body(request_body),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        content: list[str] = []
        reasoning: list[str] = []
        usage: dict = {}
        finish_reason: str | None = None
        head: dict = {}
        tool_calls: list[dict] = []

        async with self._client.stream("POST", "/v1/chat/completions", json=body) as response:
            if response.status_code >= 400:
                await response.aread()
                raise RuntimeError(f"vllm HTTP {response.status_code}: {response.text[:500]}")
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
                    logger.warning(f"Undecodable SSE chunk from vllm: {payload[:120]}")
                    continue
                if not head:
                    head = {k: chunk.get(k) for k in ("id", "created", "model")}
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    _normalise_reasoning(delta)
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
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
        return {
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

    async def complete(
        self, system: str | None, user: str, params: dict | None = None
    ) -> tuple[str, dict]:
        """One completion from a system and user message; (text, usage)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body: dict = {"messages": messages, "temperature": 0.7}
        if params:
            params = dict(params)
            enable_thinking = params.pop("enable_thinking", None)
            if enable_thinking is not None:
                ctk = dict(params.pop("chat_template_kwargs", {}) or {})
                ctk["enable_thinking"] = bool(enable_thinking)
                body["chat_template_kwargs"] = ctk
            body.update(params)
        data = await self.chat_completion(body)
        return data["choices"][0]["message"].get("content") or "", data.get("usage") or {}

    async def run_batch(self, tasks: list[dict], params: dict | None = None) -> list[JobResult]:
        if not self._ready:
            raise RuntimeError("Worker not ready")
        results: list[JobResult] = []
        for task in tasks:
            task_id = task.get("id", "unknown")
            merged = {**(params or {}), **(task.get("params") or {})}
            try:
                output, usage = await self.complete(
                    system=task.get("system"), user=task.get("user", ""), params=merged or None
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


# --- giq prepare vllm -----------------------------------------------------------------
#
# FlashInfer's GEMM modules for NVFP4 and FP8 are the expensive compiles —
# minutes each and 20+ GB of RAM at two jobs — and vllm builds them on first
# use, inside whatever request or start happens to come first. Preparing them
# once, under the same ceiling, keeps that out of serving. The generator names
# are FlashInfer's (flashinfer.jit.gemm.core).

PREPARE_MODULES: dict[int, tuple[str, ...]] = {
    12: ("gen_gemm_sm120_module", "gen_gemm_sm120_module_cutlass_fp4"),
    10: ("gen_gemm_sm100_module", "gen_gemm_sm100_module_cutlass_fp4"),
}


def prepare_modules(capability: str) -> tuple[str, ...]:
    major, _, minor = capability.partition(".")
    if major == "10" and minor.startswith("3"):
        # B300 has its own FP4 GEMM; the groupwise module is shared with sm_100.
        return ("gen_gemm_sm100_module", "gen_gemm_sm103_module_cutlass_fp4")
    modules = PREPARE_MODULES.get(int(major))
    if modules is None:
        raise VLLMConfigError(
            f"no FlashInfer kernels to prepare for compute capability {capability} "
            f"(known: {', '.join(f'{m}.x' for m in PREPARE_MODULES)})"
        )
    return modules


# Runs inside the env's interpreter. Builds with ninja directly rather than
# through JitSpec.build, which captures ninja's output until the end: a
# twenty-minute compile should say where it is. Always through ninja, never
# "skip if the library exists": the build files carry the env's absolute
# paths, so kernels built from another checkout's env are stale here and
# vllm would rebuild them on its first start — ninja knows, a file test does
# not.
_PREPARE_SCRIPT = r"""
import os, subprocess, sys, time
from flashinfer.jit.gemm import core
for gen in sys.argv[1:]:
    spec = getattr(core, gen)()
    lib = spec.get_library_path()
    before = lib.stat().st_mtime if lib.exists() else None
    print(f"{spec.name}: checking ({gen}) ...", flush=True)
    t0 = time.monotonic()
    spec.write_ninja()
    jobs = os.environ.get("MAX_JOBS", "2")
    subprocess.run(
        ["ninja", "-C", str(spec.build_dir), "-f", str(spec.ninja_path), "-j", jobs],
        check=True,
    )
    after = lib.stat().st_mtime if lib.exists() else None
    if before is not None and after == before:
        print(f"{spec.name}: up to date", flush=True)
    else:
        print(f"{spec.name}: built in {time.monotonic() - t0:.0f}s", flush=True)
"""


def prepare_command(
    capability: str, memory_max: str | None = DEFAULT_MEMORY_MAX, python: str | None = None
) -> list[str]:
    from giq.engines import binary_for

    python = python or binary_for(ENGINE)
    prefix = memory_cap_prefix("giq-vllm-prepare", memory_max)
    return [*prefix, python, "-c", _PREPARE_SCRIPT, *prepare_modules(capability)]


def prepare_targets(device: str | None = None) -> dict[str, str]:
    """{compute capability: UUID of one card with it} to build for.

    One card when named; otherwise every card, deduplicated by capability —
    kernels are built per architecture, not per card, so two identical cards
    (or a 5090 next to an RTX PRO 6000, both 12.0) need one build.
    """
    from giq.gpus import get_gpus

    if device is not None:
        gpu = resolve_device(device)
        if gpu is None:
            raise VLLMConfigError(f"no GPU matches {device!r}")
        cards = [gpu]
    else:
        cards = sorted(get_gpus(), key=lambda g: g.index)
        if not cards:
            raise VLLMConfigError("no GPU visible (nvidia-smi)")
    targets: dict[str, str] = {}
    for gpu in cards:
        capability = compute_capability(gpu.uuid)
        if capability is None:
            raise VLLMConfigError(f"cannot read GPU {gpu.index}'s compute capability (nvidia-smi)")
        targets.setdefault(capability, gpu.uuid)
    return targets


def prepare(device: str | None = None, memory_max: str | None = DEFAULT_MEMORY_MAX) -> int:
    """Build FlashInfer's GEMM kernels for one card, or for every distinct
    architecture among the cards. Returns the first non-zero exit code."""
    from giq.engines import require_binary

    require_binary(ENGINE)
    for capability, uuid in prepare_targets(device).items():
        env = engine_env(uuid)
        cmd = prepare_command(capability, memory_max)
        arch = env.get("FLASHINFER_CUDA_ARCH_LIST")
        print(
            f"preparing FlashInfer kernels for compute capability {capability} ({arch}), "
            f"{JIT_MAX_JOBS} jobs, RAM cap {memory_max if cmd[0] == 'systemd-run' else 'none'}",
            flush=True,
        )
        code = subprocess.run(cmd, env=env).returncode
        if code != 0:
            return code
    return 0


# A first start compiles what no prepare step can list ahead: FlashInfer's
# attention and sampling modules for this model's shapes, torch.compile's
# graphs, CUDA graph capture. Measured on an RTX 5090 with a warm kernel
# cache: 82 s; from nothing it is tens of minutes at two compile jobs.
WARMUP_TIMEOUT_SECONDS = 3600.0


async def warm_up(model: str, device: str | None = None) -> float:
    """Start a recipe once, with time for every first-use compile, and stop
    it. Returns the seconds the start took. Nothing is served meanwhile."""
    worker = VllmAdapter(
        VllmConfig(model=model, device=device, ready_timeout=WARMUP_TIMEOUT_SECONDS)
    )
    started = time.monotonic()
    try:
        await worker.start()
        return time.monotonic() - started
    finally:
        await worker.stop()


def cli_prepare(argv: list[str]) -> int:
    """``giq prepare vllm …``, registered with the vllm engine."""
    parser = argparse.ArgumentParser(
        prog="giq prepare vllm",
        description="Compile FlashInfer's GEMM kernels for the card's architecture inside a "
        "RAM-capped scope, so no start or request ever has to. Already-built kernels are "
        "skipped.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Card to build for, by index or UUID (default: every card, once per "
        "distinct compute capability)",
    )
    parser.add_argument(
        "--recipe",
        # The name before ADR-003, kept so existing deploy scripts still run.
        "--instance",
        dest="recipe",
        action="append",
        default=[],
        metavar="NAME",
        help="also start this vllm recipe once and stop it, so the compiles its first "
        "start needs (attention kernels, torch.compile, CUDA graphs) are done now; uses "
        "the GPU for a few minutes. Repeatable",
    )
    parser.add_argument(
        "--memory-max",
        default=None,
        help="RAM ceiling for the build (systemd MemoryMax; default 40G — two compile "
        "jobs peak near 15 GB each). `none` when the caller already runs this in a "
        "capped scope, as the installer does",
    )
    args = parser.parse_args(argv)
    try:
        memory_max = args.memory_max or DEFAULT_MEMORY_MAX
        code = prepare(args.gpu, None if memory_max.lower() == "none" else memory_max)
    except (VLLMConfigError, FileNotFoundError) as e:
        print(f"giq prepare vllm: {e}", file=sys.stderr)
        return 2
    if code != 0:
        print(f"giq prepare vllm: build failed (exit {code})", file=sys.stderr)
        return code
    for name in args.recipe:
        import asyncio

        from giq.gpus import resolve_device

        gpu = resolve_device(args.gpu) if args.gpu is not None else None
        print(f"warming up {name} (a full start, then stop) ...", flush=True)
        try:
            took = asyncio.run(warm_up(name, gpu.uuid if gpu else None))
        except Exception as e:
            print(f"giq prepare vllm: warm-up of {name} failed: {e}", file=sys.stderr)
            return 1
        print(f"{name}: started in {took:.0f}s; the next start reuses its caches", flush=True)
    return 0
