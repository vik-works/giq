# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from contextlib import asynccontextmanager

from giq.adapters.llama_cpp import INTERNAL_LLM_PORT
from giq.api.dependencies import get_audio_cache
from giq.runner import get_runner

logger = logging.getLogger(__name__)

# Reaps expired entries from the AudioCache singleton. ESP32 streaming puts
# rendered audio in this cache with a per-file TTL; without a sweeper the dict
# would grow forever since get() only evicts on access.
AUDIO_CACHE_CLEANUP_INTERVAL = 60


async def _audio_cache_cleanup_loop():
    cache = get_audio_cache()
    while True:
        try:
            await asyncio.sleep(AUDIO_CACHE_CLEANUP_INTERVAL)
            await cache.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"audio cache cleanup tick failed: {e}")


async def _vram_sampler_loop(runner):
    """Periodic VRAM + per-GPU telemetry samples for the dashboard timelines."""
    from giq.gpus import get_gpus
    from giq.stats import VRAM_SAMPLE_INTERVAL_SECONDS, get_stats
    from giq.vram import get_vram_status

    stats = get_stats()
    while True:
        try:
            await asyncio.sleep(VRAM_SAMPLE_INTERVAL_SECONDS)
            vram = await asyncio.to_thread(get_vram_status)
            ready = sum(1 for ok in runner.resident_models.values() if ok)
            active = runner.active_modality.value if runner.active_modality else None
            await stats.sample_vram(vram.used_gb, vram.free_gb, active, ready)
            gpus = await asyncio.to_thread(get_gpus)
            await stats.note_gpus(gpus)  # catches a card swapped in mid-run
            await stats.sample_gpus(gpus)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"vram sampler tick failed: {e}")


async def _kill_stale_servers(port: int = INTERNAL_LLM_PORT):
    """Kill any engine server lingering from a prior run.

    fuser only catches processes already bound to the port; pkill also catches
    processes still mid-model-load that haven't called bind() yet (but are
    already holding VRAM).

    Every pattern is qualified by one of giq's own internal ports, so the sweep
    can only reach servers giq is responsible for. There used to be a bare
    ``^llama-server `` pattern as well, for bare-invoked servers — but giq
    passes --port to every server it starts, so the port-qualified patterns
    already cover all of them, and the bare one only added reach over other
    people's processes: a hand-started llama-server, or another tool's, killed
    for the crime of sharing a program name.

    A block of ports per card since bindings landed (giq.gpus.server_port):
    a run that ended with servers on both cards leaves servers on both, and a
    second server of an engine on one card sits on a spare port of that
    card's block. Every engine's registered pattern is swept over every block
    port, or those would be left holding VRAM with nothing tracking them.
    """
    from giq import plugins
    from giq.gpus import device_port, device_ports, get_gpus

    gpus = [None, *get_gpus()]
    blocks = {p for gpu in gpus for p in device_ports(gpu)}
    # What a server engine left behind is the engine's to say (ADR-004):
    # each registers its process pattern, and any further cleanup.
    patterns = []
    for engine in plugins.engines().values():
        if engine.sweep is not None:
            await asyncio.to_thread(engine.sweep, blocks)
        if engine.stale_pattern:
            patterns += [engine.stale_pattern.format(port=p) for p in sorted(blocks)]
    # SIGTERM everything matching llama-server
    for pattern in patterns:
        proc = await asyncio.create_subprocess_exec(
            "pkill",
            "-TERM",
            "-f",
            pattern,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    # Grace period for clean shutdown before SIGKILL
    await asyncio.sleep(2.0)

    for pattern in patterns:
        proc = await asyncio.create_subprocess_exec(
            "pkill",
            "-KILL",
            "-f",
            pattern,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    # Belt-and-braces: free llama's own ports too (catches anything else
    # holding them). Not the whole block: fuser kills whatever owns a port,
    # and the spares are only giq's while a giq server sits on one — the
    # engine-qualified patterns above are what sweep those.
    for p in sorted({port} | {device_port(INTERNAL_LLM_PORT, gpu) for gpu in gpus}):
        proc = await asyncio.create_subprocess_exec(
            "fuser",
            "-k",
            f"{p}/tcp",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()


@asynccontextmanager
async def lifespan(app):
    """Manage runner lifecycle."""
    # Before any worker imports huggingface_hub, which reads HF_HOME once at
    # import: in-process workers, children and the storage catalog must all
    # agree on one cache (see paths.cache_env).
    from giq.paths import apply_cache_env

    if exported := apply_cache_env():
        logger.info("cache locations: %s", ", ".join(f"{k}={v}" for k, v in exported.items()))
    await _kill_stale_servers()
    from giq.stats import get_stats

    stats = get_stats()
    # Record the installed cards before init(): the era backfill attributes
    # post-swap jobs to the current card, which it can only do once that card
    # is in gpu_eras. This also arms the per-job GPU stamp before the first job,
    # rather than waiting a full sampler interval.
    from giq.gpus import get_gpus

    await stats.note_gpus(await asyncio.to_thread(get_gpus))
    await stats.init()  # creates schema; backfills from inflight log on first run
    await stats.record_event("start")
    # Policy overrides must load before the runner starts, or the residents
    # loop would spend its first ticks loading models the operator unpinned.
    from giq.policy import get_policy_store

    await asyncio.to_thread(get_policy_store().load)
    runner = get_runner(use_policy=True)
    await runner.start()
    cleanup_task = asyncio.create_task(_audio_cache_cleanup_loop())
    sampler_task = asyncio.create_task(_vram_sampler_loop(runner))
    logger.info("giq started")
    try:
        yield
    finally:
        for task in (cleanup_task, sampler_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await runner.stop()
        await stats.record_event("stop")
        await asyncio.to_thread(stats.close)
        logger.info("giq stopped")
