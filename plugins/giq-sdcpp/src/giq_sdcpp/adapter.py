# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Image generation via stable-diffusion.cpp's sd-server.

Spawns sd-server as a child process — the same operational pattern as
llama-server: pinned binary, own port, health-poll, SIGTERM to evict. The
dependency is one MIT binary that loads in seconds, and with --offload-to-cpu
it peaks at ~8.5GB VRAM for flux_klein — image jobs only need the resident LLM
evicted; the audio residents keep serving through renders.

Uses the A1111-compatible /sdapi/v1/txt2img surface (structured fields, no
prompt-embedded extras).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field

import httpx

from giq.adapters.engine import StartError
from giq.gpus import device_env, device_port, server_port
from giq.models import ImageResult
from giq.provenance import stamp_results
from giq.registry import vram_for

logger = logging.getLogger(__name__)

# Declared in giq.engines (GIQ_SDCPP_BINARY still overrides).

# Internal port for sd-server. Swept by lifecycle._kill_stale_servers.
INTERNAL_SD_PORT = 8087


# sd-server loads all weights at startup; from page cache that's seconds,
# from cold disk considerably longer.
READY_TIMEOUT_SECONDS = 300.0
RENDER_TIMEOUT_SECONDS = 600.0


@dataclass
class SdCppConfig:
    """Paths come from the model's recipe file (``giq_sdcpp.files.image_files``)."""

    model: str
    diffusion: str = ""
    text_encoder: str = ""
    vae: str = ""
    width: int = 800
    height: int = 480
    steps: int = 8
    cfg_scale: float = 1.0
    sampler_name: str = "Euler"
    host: str = "127.0.0.1"
    # GPU UUID this render server runs on (None = the model's binding), and
    # the port that follows from it — one sd-server per card is possible now.
    device: str | None = None
    port: int | None = None

    def __post_init__(self):
        if self.device is None:
            from giq.vram import device_for_recipe

            self.device = device_for_recipe(self.model)
        if self.port is None:
            self.port = device_port(INTERNAL_SD_PORT, self.device)
        if not self.diffusion:
            from giq.config import get_config
            from giq_sdcpp.files import image_files

            cfg = image_files(self.model)
            self.diffusion = cfg.diffusion
            self.text_encoder = cfg.text_encoder
            self.vae = cfg.vae
            gen = get_config().image_generation
            self.width = gen.width
            self.height = gen.height
            self.steps = gen.steps
            self.cfg_scale = gen.cfg


@dataclass
class SdCppAdapter:
    """Image worker wrapping sd-server (stable-diffusion.cpp)."""

    config: SdCppConfig
    _process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _ready: bool = field(default=False, repr=False)
    _stderr_task: asyncio.Task | None = field(default=None, repr=False)
    _stderr_tail: deque = field(default_factory=lambda: deque(maxlen=40), repr=False)

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
    def estimated_vram_gb(self) -> float:
        return vram_for(self.config.model, default=12.0)

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    async def start(self) -> None:
        if self.is_running:
            logger.info(f"sd-server already running: {self.config.model}")
            return

        from giq.engines import require_binary

        # A second render model on this card cannot take the first one's port.
        if self.config.port == device_port(INTERNAL_SD_PORT, self.config.device):
            self.config.port = await asyncio.to_thread(
                server_port, self.config.port, self.config.device
            )
        cmd = [
            require_binary("sd.cpp"),
            "--diffusion-model",
            self.config.diffusion,
            "--vae",
            self.config.vae,
            "--llm",
            self.config.text_encoder,
            "--listen-ip",
            self.config.host,
            "--listen-port",
            str(self.config.port),
            "--cfg-scale",
            str(self.config.cfg_scale),
            "--diffusion-fa",
            # Weights stage through RAM per pipeline phase: peak VRAM drops
            # from ~15.4GB to ~8.5GB, and the 800x480 render cost is seconds.
            "--offload-to-cpu",
        ]
        logger.info(f"Starting sd-server: {self.config.model}")
        logger.debug(f"Command: {' '.join(cmd)}")

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=device_env(self.config.device),  # sd.cpp's cuda0 = this model's card
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=RENDER_TIMEOUT_SECONDS)
        try:
            await self._wait_for_ready()
            self._ready = True
        except BaseException:
            # CancelledError included — never orphan a loading sd-server.
            await self.stop()
            raise
        logger.info(f"sd-server ready: {self.config.model}")

    async def _drain_stderr(self) -> None:
        """Keep a tail of sd-server stderr for error reporting (an early
        DEVNULL here turned a VRAM-collision 500 into a blind debugging
        round — never again)."""
        assert self._process and self._process.stderr
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug(f"[sd-server] {text}")
        except asyncio.CancelledError:
            return

    async def _wait_for_ready(self) -> None:
        deadline = asyncio.get_event_loop().time() + READY_TIMEOUT_SECONDS
        while asyncio.get_event_loop().time() < deadline:
            if self._process and self._process.returncode is not None:
                raise StartError(
                    f"sd-server exited during startup (code {self._process.returncode})"
                )
            try:
                resp = await self._client.get("/sdapi/v1/options", timeout=3.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
        raise StartError(f"sd-server not ready after {READY_TIMEOUT_SECONDS:.0f}s")

    async def stop(self) -> None:
        try:
            if self._client:
                await self._client.aclose()
        except Exception:
            logger.warning("Failed to close sd-server HTTP client")
        finally:
            self._client = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._process:
            pid = self._process.pid
            if self._process.returncode is not None:
                self._process = None
                self._ready = False
                return
            logger.info(f"Stopping sd-server: {self.config.model} (pid={pid})")
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=10.0)
            except (TimeoutError, ProcessLookupError):
                logger.warning(f"SIGTERM failed for sd-server pid={pid}, sending SIGKILL")
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=10.0)
                except (TimeoutError, ProcessLookupError):
                    logger.error(f"sd-server pid={pid} survived SIGKILL (stuck in CUDA driver).")
                    raise RuntimeError(
                        f"Cannot unload model: sd-server pid={pid} is unkillable. "
                        f"Manual intervention required (GPU reset or reboot)."
                    ) from None
            self._process = None
            self._ready = False

    async def run_batch(self, tasks: list[dict], params: dict | None = None) -> list[ImageResult]:
        if not self.is_ready or not self._client:
            raise RuntimeError("sd-server worker not ready")

        results: list[ImageResult] = []
        for task in tasks:
            task_id = task.get("id", "unknown")
            try:
                # Edit tasks (image_edit worker type) carry a reference image
                # and an instruction; klein consumes references kontext-style
                # via txt2img extra_images.
                prompt = task.get("prompt") or task.get("instruction")
                if not prompt:
                    raise ValueError("task has neither prompt nor instruction")
                payload = {
                    "prompt": prompt,
                    "negative_prompt": task.get("negative_prompt") or "",
                    "width": self.config.width,
                    "height": self.config.height,
                    "steps": self.config.steps,
                    "cfg_scale": self.config.cfg_scale,
                    "sampler_name": self.config.sampler_name,
                    "seed": task.get("seed", -1) if task.get("seed") is not None else -1,
                    "batch_size": 1,
                }
                if task.get("reference_image_b64"):
                    payload["extra_images"] = [task["reference_image_b64"]]
                resp = await self._client.post("/sdapi/v1/txt2img", json=payload)
                resp.raise_for_status()
                data = resp.json()
                images = data.get("images") or []
                if not images:
                    raise RuntimeError("sd-server returned no images")
                seed = payload["seed"]
                try:
                    info = json.loads(data.get("info") or "{}")
                    seed = info.get("seed", seed)
                except (ValueError, TypeError):
                    pass
                results.append(ImageResult(id=task_id, image_b64=images[0], seed=seed))
            except Exception as e:
                tail = " | ".join(list(self._stderr_tail)[-5:])
                logger.error(f"sd-server task {task_id} failed: {e}; stderr tail: {tail}")
                results.append(
                    ImageResult(id=task_id, error=f"{e}" + (f" [{tail}]" if tail else ""))
                )
        return stamp_results(results, self.config.model)
