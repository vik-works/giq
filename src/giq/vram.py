# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""VRAM monitoring and gating for giq workers.

Every figure here describes *one card* — the device ``giq.gpus`` selects, or
the one a caller names. It used to describe "the GPU" by reading the first
row nvidia-smi printed, which was correct exactly as long as there was one
row. On a second card that silently became "GPU 0's memory, whatever giq is
actually running on", while the eviction planner's per-process measurements
covered both cards at once.
"""

import asyncio
import logging
from dataclasses import dataclass

from giq.registry import get_recipe

logger = logging.getLogger(__name__)

# Telemetry staleness the gate tolerates. The gate polls at 0.5s while waiting
# for a killed worker's VRAM to come back, so this must stay well under that;
# it exists only to collapse the burst of reads a single scheduling decision
# makes (status + gate + planner) into one nvidia-smi call.
GATE_MAX_AGE_SECONDS = 0.25


@dataclass
class VRAMStatus:
    """Current VRAM status."""

    used_gb: float
    total_gb: float
    free_gb: float

    @property
    def utilization(self) -> float:
        """VRAM utilization percentage."""
        return (self.used_gb / self.total_gb * 100) if self.total_gb > 0 else 0


# Default if not specified
DEFAULT_VRAM_REQUIREMENT = 22.0

# Minimum free VRAM to consider "safe" (GB)
VRAM_SAFETY_MARGIN = 2.0


def margin_for(required_gb: float) -> float:
    """Safety margin scaled to the model.

    The flat 2GB margin exists to absorb compute-buffer spikes on big loads;
    demanding 2.6GB free to load a 0.6GB embedder would make the packed
    resident set (gemma+whisper+ecapa ≈ 14GB of 15.9) unschedulable. Small
    models get a proportional margin, floored at 0.5GB.
    """
    return min(VRAM_SAFETY_MARGIN, max(0.5, required_gb * 0.5))


def margin_of(name: str, required_gb: float | None = None) -> float:
    """The safety margin for loading recipe ``name``: its engine's declared
    one, else the scaled default for its VRAM figure."""
    from giq import plugins
    from giq.registry import get_recipe

    recipe = get_recipe(name)
    engine = plugins.engine(recipe.engine) if recipe is not None else None
    if engine is not None and engine.vram_margin is not None:
        return engine.vram_margin
    return margin_for(required_gb if required_gb is not None else get_vram_requirement(name))


def get_vram_requirement(name: str) -> float:
    """VRAM recipe ``name`` needs, from its recipe file.

    An unknown name falls back to DEFAULT_VRAM_REQUIREMENT, which exceeds a
    16GB card once the margin is added — so an unknown model fails fast
    rather than loading against an unknown footprint. That is deliberate, but
    it means "forgot the recipe" and "genuinely too big" look identical;
    give a model you intend to run a recipe file (see giq.registry).
    """
    recipe = get_recipe(name)
    return recipe.vram_gb if recipe else DEFAULT_VRAM_REQUIREMENT


def get_vram_status(device: str | int | None = None) -> VRAMStatus:
    """Current VRAM on one card: the selected device, or ``device``.

    Reads through ``gpus.get_gpus``, so the gate, the /gpus endpoint and the
    eviction planner all see the same sampling of the same card instead of
    each running their own nvidia-smi against a different notion of "the GPU".

    Returns:
        VRAMStatus with used_gb, total_gb, free_gb
    """
    try:
        from giq.gpus import get_gpus, resolve_device, selected_device

        get_gpus(GATE_MAX_AGE_SECONDS)  # refresh the shared telemetry cache
        gpu = resolve_device(device) if device is not None else selected_device()
        if gpu is not None:
            return VRAMStatus(
                used_gb=gpu.vram_used_gb,
                total_gb=gpu.vram_total_gb,
                free_gb=gpu.vram_free_gb,
            )
        logger.warning("No GPU visible for VRAM status; using the fallback figure")
    except Exception as e:
        logger.warning(f"Failed to get VRAM status: {e}")

    # Fallback: assume a 16GB card, 0 used. Deliberately small — a gate that
    # guesses high admits models that then OOM.
    return VRAMStatus(used_gb=0.0, total_gb=16.0, free_gb=16.0)


def get_free_vram(device: str | int | None = None) -> float:
    """Get free VRAM in GB on the selected (or named) card."""
    return get_vram_status(device).free_gb


def reserve_for(device: str | int | None) -> float:
    """VRAM to leave unclaimed on a card, from config ``gpu.reserve``.

    For cards giq shares with something it does not schedule — above all the
    one driving the desktop, where a browser can take another gigabyte at any
    moment. Free VRAM already nets out what the desktop holds *now*; this is
    headroom for what it grabs next, which a model load would otherwise lose a
    race to.
    """
    if device is None:
        return 0.0
    try:
        from giq.config import get_config
        from giq.gpus import resolve_device

        reserve = get_config().gpu.reserve
        if not reserve:
            return 0.0
        gpu = resolve_device(device)
        if gpu is None:
            return 0.0
        for name, gb in reserve.items():
            named = resolve_device(name)
            if named is not None and named.uuid == gpu.uuid:
                return float(gb)
    except Exception as e:
        logger.warning(f"gpu.reserve unreadable, assuming none: {e}")
    return 0.0


def device_for_recipe(name: str) -> str | None:
    """UUID of the card recipe ``name`` loads on. None when no GPU is visible."""
    try:
        from giq.policy import device_of

        gpu = device_of(name)
        return gpu.uuid if gpu else None
    except Exception as e:  # a broken policy store must not blind the gate
        logger.warning(f"device lookup failed for {name}: {e}")
        from giq.gpus import selected_device

        gpu = selected_device()
        return gpu.uuid if gpu else None


def can_load(name: str, device: str | int | None = None) -> tuple[bool, str]:
    """Check if we have enough VRAM to load recipe ``name`` on its bound card.

    Returns:
        Tuple of (can_load, reason)
    """
    if device is None:
        device = device_for_recipe(name)
    required = get_vram_requirement(name)
    status = get_vram_status(device)
    reserve = reserve_for(device)
    margin = margin_of(name, required)
    needed = required + margin + reserve

    if needed > status.total_gb:
        return False, (
            f"Model can never fit: {needed:.1f}GB needed > {status.total_gb:.1f}GB total "
            f"({required:.1f}GB model + {margin:.1f}GB margin"
            + (f" + {reserve:.1f}GB reserved)" if reserve else ")")
        )
    if status.free_gb >= needed:
        return True, f"OK: {status.free_gb:.1f}GB free >= {needed:.1f}GB needed"
    else:
        msg = (
            f"Insufficient VRAM: {status.free_gb:.1f}GB free < {needed:.1f}GB needed "
            f"({required:.1f}GB model + {margin:.1f}GB margin"
            + (f" + {reserve:.1f}GB reserved)" if reserve else ")")
        )
        return False, msg


async def wait_for_vram(
    name: str,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
    margin_gb: float | None = None,
    device: str | int | None = None,
) -> bool:
    """Wait until enough VRAM is available on this recipe's card.

    Args:
        name: Recipe name
        timeout: Max seconds to wait (0 = no timeout)
        poll_interval: Seconds between checks
        margin_gb: Override the scaled safety margin. Resident reloads pass a
            small fixed margin: the resident set is validated to coexist, so
            demanding the full spike margin can deadlock — free VRAM lands in
            [required, required+margin) with nothing left to evict, and the
            loop retries forever (a resident model can sit unreloadable for
            an hour, blocked by 70MB).
        device: Card to wait on; None resolves the model's binding.

    Returns:
        True if VRAM became available, False if timeout
    """
    if device is None:
        device = device_for_recipe(name)
    required = get_vram_requirement(name)
    needed = (
        required
        + (margin_gb if margin_gb is not None else margin_of(name, required))
        + reserve_for(device)
    )
    elapsed = 0.0

    # Fail fast when the requirement exceeds the card entirely — waiting can
    # never succeed (without this, a job asking for a 24GB model on a 16GB
    # card hangs until its timeout). With bindings this is the
    # loud failure for "bound a 22GB model to the 16GB card".
    total = get_vram_status(device).total_gb
    if needed > total:
        logger.error(
            f"VRAM requirement impossible for {name} on its card: "
            f"{needed:.1f}GB needed > {total:.1f}GB total — failing fast"
        )
        return False

    while timeout == 0 or elapsed < timeout:
        free = get_free_vram(device)
        if free >= needed:
            return True

        logger.info(
            f"Waiting for VRAM: {free:.1f}GB free, need {needed:.1f}GB (waited {elapsed:.0f}s)"
        )
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    return False
