# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess-isolated worker base class.

Torch's `torch.cuda.empty_cache()` returns blocks to PyTorch's allocator but
does not destroy the CUDA context — the context lives for the lifetime of the
Python process. Measured: after `del pipeline; gc.collect();
torch.cuda.empty_cache()` in an in-process text2image worker, the giq main
process still appeared in `nvidia-smi --query-compute-apps` holding ~622 MiB.
That retained context shifts the physical VRAM layout for the next
llama-server load, which was strongly correlated with hard PSU/VRM trips under
sustained large-model inference on the machine it was measured on.

This base class spawns an `execve`'d Python child per worker. Killing the
child destroys its CUDA context cleanly. The pattern mirrors
`giq.adapters.llama_cpp.LlamaCppAdapter`, which has always run llama-server as a subprocess
and thus never exhibited this retention.

Wire protocol (one JSON object per line, UTF-8, `\\n`-terminated; the child
side is the standalone package ``giq_child``):

  Child → Parent:
    {"type": "ready"}
    {"type": "results", "results": [...]}
    {"type": "error",   "message": "...", "traceback": "..."}

  Parent → Child:
    {"type": "run_batch", "tasks": [...], "params": {...}|null}
    {"type": "shutdown"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import deque
from typing import Any, ClassVar

from giq.gpus import device_env

logger = logging.getLogger(__name__)


# --- Protocol knobs -------------------------------------------------------

READY_TIMEOUT_SECONDS = 300.0  # model load can take ~30s; allow headroom
RUN_BATCH_TIMEOUT_SECONDS = 600.0  # image gen a few seconds; video could be long
SHUTDOWN_GRACE_SECONDS = 2.0  # after shutdown message, before SIGTERM
SIGTERM_TIMEOUT_SECONDS = 10.0  # match llm.py
SIGKILL_TIMEOUT_SECONDS = 10.0
STDERR_RING_MAX_LINES = 100  # kept for crash diagnostics
# StreamReader line-buffer cap. Default 64 KiB is too small: a single image
# batch serialized as JSON can easily exceed that (base64 PNG ≈ 1.5-3 MB per
# image, up to max_batch=8 → ~24 MB). 128 MiB covers realistic payloads with
# headroom without pinning significant RSS (buffer grows on demand).
PIPE_LINE_LIMIT = 128 * 1024 * 1024


# --- Exceptions -----------------------------------------------------------


class SubprocessWorkerError(RuntimeError):
    """Base class for subprocess worker errors."""


class SubprocessWorkerStartError(SubprocessWorkerError):
    """Worker subprocess failed to reach ready state."""


class SubprocessWorkerDied(SubprocessWorkerError):
    """Worker subprocess exited unexpectedly mid-operation."""

    def __init__(self, returncode: int | None, stderr_tail: str = ""):
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        msg = f"subprocess died (returncode={returncode})"
        if stderr_tail:
            msg += f"\n--- stderr tail ---\n{stderr_tail}"
        super().__init__(msg)


# --- Parent-side: SubprocessAdapter base ----------------------------------


class SubprocessAdapter:
    """Base class for workers that run in a child process for CUDA isolation.

    Subclasses must define:
      - ``child_module`` (ClassVar[str]): dotted module path for ``python -m``
      - ``estimated_vram_gb`` (property)

    Subclasses may override ``child_args()`` to pass CLI arguments to the child
    (e.g. ``["--model", self.config.model]``). The worker's public surface
    (``start``, ``stop``, ``run_batch``, ``is_running``, ``is_ready``) satisfies
    the ``giq.runner.Worker`` Protocol without any runner-side changes.
    """

    child_module: ClassVar[str] = ""
    # Registry worker type, for resolving which card this model is bound to.
    modality: ClassVar[str] = ""
    # Per-class override for run_batch's IPC wait (diarization of a long
    # recording can exceed the 600s default).
    run_batch_timeout: ClassVar[float] = RUN_BATCH_TIMEOUT_SECONDS

    def __init__(self, config: Any, device: str | None = None) -> None:
        self.config = config
        # GPU UUID the child is pinned to. None resolves to the model's
        # binding at spawn time (see _spawn_env) rather than here, so a
        # rebind between construction and start is honoured.
        self.device = device
        self._process: asyncio.subprocess.Process | None = None
        self._ready: bool = False
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_RING_MAX_LINES)
        # Serialize stdin writes so concurrent run_batch + stop don't interleave.
        # Lazy-init: asyncio.Lock() can be created outside an event loop in 3.10+,
        # but we defer to keep constructor pure-sync.
        self._write_lock: asyncio.Lock | None = None

    # --- Subclass hooks ---------------------------------------------------

    def child_args(self) -> list[str]:
        """Return additional CLI args to pass after ``python -m <child_module>``."""
        return []

    def _spawn_env(self) -> dict[str, str]:
        """Child environment, with CUDA_VISIBLE_DEVICES pinned to one card."""
        device = self.device
        if device is None:
            from giq.vram import device_for_recipe

            device = device_for_recipe(getattr(self.config, "model", ""))
        return device_env(device)

    def _command(self) -> list[str]:
        """Build the full subprocess argv. Tests may override."""
        if not self.child_module:
            raise NotImplementedError(f"{type(self).__name__} must set child_module")
        return [sys.executable, "-u", "-m", self.child_module, *self.child_args()]

    # --- Protocol properties ---------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def is_ready(self) -> bool:
        return self.is_running and self._ready

    @property
    def pid(self) -> int | None:
        """PID of the child holding this model's VRAM, for measuring what
        evicting it would actually free. None when there is no child."""
        return self._process.pid if self._process else None

    @property
    def estimated_vram_gb(self) -> float:  # pragma: no cover - subclass contract
        raise NotImplementedError

    @property
    def _log_prefix(self) -> str:
        tag = self.child_module.rsplit(".", 1)[-1] or "subprocess"
        pid = self._process.pid if self._process else "?"
        return f"[{tag} pid={pid}]"

    # --- Lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if self.is_running:
            logger.info(f"Worker already running: {self.child_module}")
            return

        self._write_lock = asyncio.Lock()
        cmd = self._command()
        logger.info(f"Spawning {self.child_module}: {' '.join(cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
            limit=PIPE_LINE_LIMIT,
            # torch children default to cuda:0 of whatever they can see; this
            # makes that the card this model is bound to.
            env=self._spawn_env(),
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        try:
            await self._wait_for_ready()
            self._ready = True
        except BaseException:
            # BaseException so CancelledError (shutdown mid-load) also reaps.
            await self.stop()
            raise

        logger.info(f"{self._log_prefix} ready")

    async def _wait_for_ready(self) -> None:
        try:
            msg = await asyncio.wait_for(self._read_message(), timeout=READY_TIMEOUT_SECONDS)
        except TimeoutError:
            raise SubprocessWorkerStartError(
                f"{self._log_prefix} did not signal ready within {READY_TIMEOUT_SECONDS}s"
            ) from None
        except SubprocessWorkerDied as e:
            # Convert EOF-during-startup into a clearer startup-specific error.
            raise SubprocessWorkerStartError(
                f"{self._log_prefix} exited before signaling ready "
                f"(returncode={e.returncode}).\n"
                f"--- stderr tail ---\n{e.stderr_tail}"
            ) from e

        mtype = msg.get("type")
        if mtype == "ready":
            return
        if mtype == "error":
            tail = "\n".join(self._stderr_tail)
            raise SubprocessWorkerStartError(
                f"{self._log_prefix} reported error before ready: "
                f"{msg.get('message')}\n"
                f"child traceback:\n{msg.get('traceback', '')}\n"
                f"--- stderr tail ---\n{tail}"
            )
        raise SubprocessWorkerStartError(f"{self._log_prefix} unexpected first message: {msg!r}")

    async def _read_message(self) -> dict[str, Any]:
        """Read one JSON line from child stdout.

        Raises ``SubprocessWorkerDied`` on EOF (child exited before writing).
        Raises ``SubprocessWorkerError`` on malformed JSON.
        """
        assert self._process and self._process.stdout
        line = await self._process.stdout.readline()
        if not line:
            rc = await self._process.wait()
            tail = "\n".join(self._stderr_tail)
            raise SubprocessWorkerDied(rc, tail)
        try:
            return json.loads(line.decode("utf-8").strip())
        except json.JSONDecodeError as e:
            raise SubprocessWorkerError(
                f"{self._log_prefix} malformed JSON from child: {line!r}: {e}"
            ) from e

    async def _write_message(self, msg: dict[str, Any]) -> None:
        """Write one JSON line to child stdin."""
        assert self._process and self._process.stdin and self._write_lock
        data = json.dumps(msg).encode("utf-8") + b"\n"
        async with self._write_lock:
            try:
                self._process.stdin.write(data)
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                raise SubprocessWorkerDied(
                    self._process.returncode, "\n".join(self._stderr_tail)
                ) from e

    async def _drain_stderr(self) -> None:
        """Background task: read child stderr, forward to logger, keep tail."""
        assert self._process and self._process.stderr
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                self._stderr_tail.append(text)
                logger.debug(f"{self._log_prefix} stderr: {text}")
        except asyncio.CancelledError:
            return

    # --- Work dispatch ---------------------------------------------------

    async def run_batch(
        self, tasks: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Send a batch to the child and await results.

        Returns the raw list of result dicts from the child — runner converts
        to Pydantic models via ``[model(**r) for r in result_dicts]`` as
        needed, matching the existing in-process worker pattern.
        """
        if not self.is_ready:
            raise RuntimeError(f"{self._log_prefix} worker not ready")

        await self._write_message({"type": "run_batch", "tasks": tasks, "params": params})
        try:
            msg = await asyncio.wait_for(self._read_message(), timeout=self.run_batch_timeout)
        except TimeoutError:
            raise SubprocessWorkerError(
                f"{self._log_prefix} did not return results within {self.run_batch_timeout}s"
            ) from None

        mtype = msg.get("type")
        if mtype == "results":
            return msg["results"]
        if mtype == "error":
            raise SubprocessWorkerError(
                f"{self._log_prefix} batch error: {msg.get('message')}\n{msg.get('traceback', '')}"
            )
        raise SubprocessWorkerError(
            f"{self._log_prefix} unexpected message during run_batch: {msg!r}"
        )

    # --- Teardown --------------------------------------------------------

    async def stop(self) -> None:
        """Shutdown msg → SIGTERM → SIGKILL → RuntimeError, same as llm.py."""
        if self._process is None:
            return

        pid = self._process.pid
        logger.info(f"{self._log_prefix} stopping")

        # 1. Graceful shutdown message via stdin.
        if self._process.stdin and not self._process.stdin.is_closing():
            try:
                await self._write_message({"type": "shutdown"})
            except Exception:
                pass  # best-effort; child may already be dead
            try:
                self._process.stdin.close()
            except Exception:
                pass

        try:
            await asyncio.wait_for(self._process.wait(), timeout=SHUTDOWN_GRACE_SECONDS)
        except (TimeoutError, ProcessLookupError):
            pass

        # 2. SIGTERM escalation — only if still alive.
        if self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=SIGTERM_TIMEOUT_SECONDS)
            except (TimeoutError, ProcessLookupError):
                logger.warning(f"{self._log_prefix} SIGTERM failed, sending SIGKILL")
                # 3. SIGKILL.
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=SIGKILL_TIMEOUT_SECONDS)
                except (TimeoutError, ProcessLookupError):
                    # Process survived SIGKILL (stuck in CUDA D state).
                    # Do NOT clear self._process — caller needs to know it leaked.
                    logger.error(
                        f"{self._log_prefix} survived SIGKILL "
                        f"(likely stuck in CUDA driver). VRAM still held."
                    )
                    raise RuntimeError(
                        f"Cannot unload {self.child_module} pid={pid}: "
                        f"subprocess is unkillable. Manual intervention "
                        f"required (GPU reset or reboot)."
                    ) from None

        # 4. Reap stderr drain task.
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stderr_task = None

        self._process = None
        self._ready = False
