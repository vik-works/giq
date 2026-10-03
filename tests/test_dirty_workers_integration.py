# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for subprocess-isolated dirty workers.

These tests require a real NVIDIA GPU (the whole point is to verify CUDA
context isolation via subprocess). Skipped automatically on hosts without
``nvidia-smi``.

The key regression test is ``test_text2image_process_not_in_nvidia_smi_after_stop`` —
that's the entire reason for the refactor. Before this refactor, running a
text2image job would leave the giq main Python process holding ~622 MiB of
CUDA memory (visible in ``nvidia-smi --query-compute-apps``) because
``torch.cuda.empty_cache()`` doesn't destroy the CUDA context. After the
refactor the generation happens in a child process and its context dies
cleanly with the subprocess.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess

import pytest

from giq.vram import get_free_vram
from giq_sdcpp.adapter import SdCppAdapter, SdCppConfig
from giq_speech.tts import TtsAdapter, TtsConfig


def _has_nvidia_gpu() -> bool:
    """True if ``nvidia-smi`` is usable."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(not _has_nvidia_gpu(), reason="requires NVIDIA GPU")


def _query_nvidia_smi_compute_apps_pids() -> set[int]:
    """Return set of PIDs currently holding a CUDA context."""
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return {int(line.strip()) for line in r.stdout.strip().splitlines() if line.strip().isdigit()}


def _query_free_vram_mib() -> int:
    """Return free VRAM on GPU 0 in MiB."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return int(r.stdout.strip().splitlines()[0])


async def test_text2image_zimage_full_lifecycle():
    """Sanity: worker starts, produces an image, stops cleanly."""
    worker = SdCppAdapter(SdCppConfig(model="zimage"))
    await worker.start()
    try:
        assert worker.is_ready
        results = await worker.run_batch(
            [{"id": "t1", "prompt": "a tiny red cube on a blue table"}]
        )
        assert len(results) == 1
        r = results[0]
        assert r.id == "t1"
        assert r.error is None, f"generation errored: {r.error}"
        assert r.image_b64, "expected non-empty image_b64"
        # Verify it's a valid PNG
        png_bytes = base64.b64decode(r.image_b64)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "expected PNG magic"
    finally:
        await worker.stop()
    assert not worker.is_running


async def test_text2image_process_not_in_nvidia_smi_after_stop():
    """REGRESSION: the giq main PID must not hold CUDA memory after stop.

    Before this refactor the in-process torch path left ~622 MiB held by the
    parent Python process (measured on an RTX 5090). That retained context
    shifted VRAM layout for subsequent llama-server loads and was strongly
    correlated with PSU/VRM trip crashes.
    """
    parent_pid = os.getpid()
    # Sanity: before any CUDA work, parent should not appear.
    assert parent_pid not in _query_nvidia_smi_compute_apps_pids()

    worker = SdCppAdapter(SdCppConfig(model="zimage"))
    await worker.start()
    try:
        await worker.run_batch([{"id": "t", "prompt": "a potted plant by a window"}])
    finally:
        await worker.stop()

    # Give the CUDA driver a beat to finalize reclaim after child exit.
    await asyncio.sleep(1.5)

    holders = _query_nvidia_smi_compute_apps_pids()
    assert parent_pid not in holders, (
        f"giq main PID {parent_pid} is STILL holding CUDA memory after stop. "
        f"This is the bug this refactor exists to fix. "
        f"nvidia-smi compute-apps PIDs: {holders}"
    )


async def test_text2image_vram_fully_returned_after_stop():
    """After stop, free VRAM should return within 0.5 GB of the cold baseline.

    Tighter than the compute-apps check: catches any CUDA-using library
    (not necessarily torch) sneaking into the parent process.
    """
    # Cold baseline.
    await asyncio.sleep(0.5)  # let any prior work settle
    cold_free_mib = _query_free_vram_mib()

    worker = SdCppAdapter(SdCppConfig(model="zimage"))
    await worker.start()
    try:
        await worker.run_batch([{"id": "t", "prompt": "a mountain landscape at dusk"}])
    finally:
        await worker.stop()

    await asyncio.sleep(1.5)  # CUDA reclaim settle
    post_free_mib = _query_free_vram_mib()
    delta_mib = cold_free_mib - post_free_mib

    assert delta_mib < 512, (
        f"VRAM not returned after stop: cold={cold_free_mib} MiB, "
        f"post={post_free_mib} MiB, delta={delta_mib} MiB "
        f"(>512 MiB = likely retained CUDA context in parent)"
    )


async def test_kokoro_process_not_in_nvidia_smi_after_stop():
    """Kokoro was the last CUDA worker running inside giq's own process.

    Observed live: the runner logged "Kokoro TTS stopped" and
    ``active_worker`` went None, while ``nvidia-smi`` still showed the giq
    main PID holding 968 MiB — and kept showing it until giq restarted. On a
    16GB card with the resident set loaded that was the difference between the
    next render working and OOMing.
    """
    parent_pid = os.getpid()
    assert parent_pid not in _query_nvidia_smi_compute_apps_pids()

    worker = TtsAdapter(TtsConfig())
    await worker.start()
    assert worker.pid is not None, "kokoro must own a child process"
    child_pid = worker.pid
    try:
        results = await worker.run_batch(
            [{"id": "t", "text": "The lighthouse keeper recorded the weather.", "voice": "alloy"}]
        )
        assert results[0].error is None, f"TTS errored: {results[0].error}"
        assert results[0].output, "expected non-empty audio output"
        # Decodes as a real WAV, i.e. the IPC round trip survived the move.
        audio = base64.b64decode(results[0].output)
        assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    finally:
        await worker.stop()

    await asyncio.sleep(1.5)
    holders = _query_nvidia_smi_compute_apps_pids()
    assert parent_pid not in holders, (
        f"giq main PID {parent_pid} is STILL holding CUDA memory after kokoro "
        f"stop — the in-process leak is back. compute-apps PIDs: {holders}"
    )
    assert child_pid not in holders, f"kokoro child {child_pid} outlived stop()"


async def test_kokoro_vram_fully_returned_after_stop():
    """The number that actually mattered: ~1GB back, not merely a dead PID."""
    cold_free_mib = _query_free_vram_mib()

    worker = TtsAdapter(TtsConfig())
    await worker.start()
    try:
        await worker.run_batch([{"id": "t", "text": "Hello there.", "voice": "alloy"}])
    finally:
        await worker.stop()

    await asyncio.sleep(1.5)
    post_free_mib = _query_free_vram_mib()
    leaked = cold_free_mib - post_free_mib
    assert leaked < 256, (
        f"{leaked} MiB not returned after kokoro stop "
        f"(cold {cold_free_mib} MiB, after {post_free_mib} MiB)"
    )


async def test_sequencing_text2image_then_llm_sees_cold_vram():
    """Crash-2 precondition test: after text2image unload, ``get_free_vram()``
    — the function giq's runner uses to decide whether a next worker can load
    — must report essentially the cold baseline. If it reports less, we'd be
    back in the "load llama.cpp around retained image-model memory" crash path.
    """
    # Cold baseline from the same function the runner queries.
    await asyncio.sleep(0.5)
    cold_free_gb = get_free_vram()

    worker = SdCppAdapter(SdCppConfig(model="zimage"))
    await worker.start()
    try:
        await worker.run_batch([{"id": "x", "prompt": "a waterfall"}])
    finally:
        await worker.stop()

    # Simulate what Runner does between workers (runner.py uses wait_for_vram
    # with a POST_UNLOAD_VRAM_GRACE_SECONDS=10s grace).
    await asyncio.sleep(1.5)

    post_free_gb = get_free_vram()
    delta_gb = cold_free_gb - post_free_gb
    assert delta_gb < 0.5, (
        f"runner would now load llama.cpp into fragmented VRAM: "
        f"cold_free={cold_free_gb:.2f} GB, post_unload_free={post_free_gb:.2f} GB, "
        f"delta={delta_gb:.2f} GB (>0.5 GB = the crash-2 precondition is back)"
    )
