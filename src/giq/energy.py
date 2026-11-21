# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Energy and GPU time per job, split fairly between jobs sharing a card (ADR-006 D5).

The source is each card's cumulative energy counter (NVML's
``nvmlDeviceGetTotalEnergyConsumption``, millijoules since the driver loaded,
Volta and later). Measured on an RTX 5090 and an RTX 5060 Ti: one read takes
about 1.5 ms. Where the counter is missing (an older card, NVML unavailable)
the meter integrates the card's power draw between reads instead, and every
figure it produces that way is marked as an estimate.

A card is read whenever a job starts or ends on it and on every sampler tick.
The energy between two reads is split evenly between the jobs running in
that interval, and so is the time, which makes GPU-seconds a share of the
card rather than a wall clock (that is ``run_ms``). Splitting by tokens was
rejected: tokens mean nothing for an image, audio or depth job. Energy drawn
with no job running, by recipes kept resident and idle, belongs to nobody;
it is collected per card as *idle* and reported on its own line, never
spread over the apps.

Loading and unloading an instance is housekeeping: the time between an
adapter's ``start()`` or ``stop()`` being called and returning is a share of
its own on the card, like a job's, and what it drew is collected per card as
*load*, on another line of its own. It is not charged to the job whose
request caused the load: a loaded instance serves every job after it, often
from other apps, and which job happened to arrive first says nothing about
who the load was for.

Every read and the bookkeeping after it run together under one lock in a
worker thread: the event loop never waits on the driver, and two reads of a
card are applied in the order they were taken, so an interval is never
counted backwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Reading:
    energy_mj: int | None = None  # the card's cumulative counter
    power_w: float | None = None  # instantaneous draw, for the estimate


class NvmlReader:
    """Reads cards through NVML; falls back to nvidia-smi's power column."""

    def __init__(self) -> None:
        self._nvml = None
        self._tried = False
        self._handles: dict[str, object] = {}

    def _init(self):
        if not self._tried:
            self._tried = True
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
            except Exception as e:  # noqa: BLE001 — no driver, no library, no card
                logger.info(f"energy: NVML unavailable, estimating from power samples: {e}")
        return self._nvml

    def cards(self) -> list[str]:
        nvml = self._init()
        if nvml is None:
            from giq.gpus import get_gpus

            return [g.uuid for g in get_gpus()]
        try:
            return [
                nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(i))
                for i in range(nvml.nvmlDeviceGetCount())
            ]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"energy: listing cards failed: {e}")
            return []

    def read(self, uuid: str) -> Reading:
        nvml = self._init()
        if nvml is None:
            from giq.gpus import get_gpus

            for g in get_gpus():
                if g.uuid == uuid:
                    return Reading(power_w=g.power_draw_w)
            return Reading()
        reading = Reading()
        try:
            handle = self._handles.get(uuid)
            if handle is None:
                handle = self._handles[uuid] = nvml.nvmlDeviceGetHandleByUUID(uuid)
            try:
                reading.energy_mj = int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            except Exception:  # noqa: BLE001 — pre-Volta: no counter, power still works
                pass
            try:
                reading.power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"energy: reading {uuid} failed: {e}")
        return reading


@dataclass
class _Share:
    joules: float = 0.0
    seconds: float = 0.0
    measured: bool = True


@dataclass
class _Card:
    last_t: float | None = None
    last_mj: int | None = None
    last_power: float | None = None
    active: dict[object, _Share] = field(default_factory=dict)
    idle_j: float = 0.0
    idle_measured: bool = True
    load_j: float = 0.0
    load_measured: bool = True


@dataclass
class Idle:
    """A card's energy since the last tick that no job drew."""

    gpu_uuid: str
    wh: float  # nothing running
    measured: bool  # both lines are from the counter
    load_wh: float = 0.0  # loading and unloading instances


class EnergyMeter:
    def __init__(self, reader=None, clock=time.monotonic) -> None:
        self._reader = reader or NvmlReader()
        self._clock = clock
        self._lock = threading.Lock()
        self._cards: dict[str, _Card] = {}
        # Jobs whose end was applied before their begin: a job cancelled
        # while its begin was still reading in a worker thread. The begin,
        # when it lands, must not leave a share behind that never ends and
        # takes half the card from every later job.
        self._ended_early: set[object] = set()
        # Housekeeping shares get keys of their own, which can never equal a
        # job's id().
        self._housekeeping_keys = itertools.count()

    # --- the bookkeeping (under the lock) ------------------------------------

    def _advance(self, uuid: str) -> _Card:
        card = self._cards.setdefault(uuid, _Card())
        reading = self._reader.read(uuid)
        now = self._clock()
        if card.last_t is not None:
            dt = max(0.0, now - card.last_t)
            if (
                reading.energy_mj is not None
                and card.last_mj is not None
                and reading.energy_mj >= card.last_mj
            ):
                joules, measured = (reading.energy_mj - card.last_mj) / 1000.0, True
            else:
                # No counter, or it went backwards (a driver reload resets
                # it): the mean of the two power readings over the interval.
                powers = [p for p in (card.last_power, reading.power_w) if p is not None]
                joules = (sum(powers) / len(powers)) * dt if powers else 0.0
                measured = False
            if card.active:
                n = len(card.active)
                for share in card.active.values():
                    share.joules += joules / n
                    share.seconds += dt / n
                    share.measured = share.measured and measured
            else:
                card.idle_j += joules
                card.idle_measured = card.idle_measured and measured
        card.last_t = now
        card.last_mj = reading.energy_mj
        card.last_power = reading.power_w
        return card

    def _begin_sync(self, key: object, uuid: str) -> None:
        with self._lock:
            if key in self._ended_early:
                self._ended_early.discard(key)
                return
            self._advance(uuid).active[key] = _Share()

    def _end_sync(self, key: object, uuid: str, housekeeping: bool = False) -> _Share | None:
        with self._lock:
            card = self._advance(uuid)
            share = card.active.pop(key, None)
            if share is None:
                self._ended_early.add(key)
            elif housekeeping:
                card.load_j += share.joules
                card.load_measured = card.load_measured and share.measured
            return share

    def _tick_sync(self) -> list[Idle]:
        with self._lock:
            for uuid in self._reader.cards():
                self._advance(uuid)
            out = []
            for uuid, card in self._cards.items():
                if card.idle_j > 0 or card.load_j > 0:
                    out.append(
                        Idle(
                            uuid,
                            card.idle_j / 3600.0,
                            card.idle_measured and card.load_measured,
                            card.load_j / 3600.0,
                        )
                    )
                card.idle_j = card.load_j = 0.0
                card.idle_measured = card.load_measured = True
            return out

    # --- the async face: never raises, never blocks the loop -----------------

    async def begin(self, job, uuid: str | None) -> None:
        """A job starts running on card ``uuid``."""
        if not uuid:
            return
        job.gpu_uuid = uuid
        try:
            await asyncio.to_thread(self._begin_sync, id(job), uuid)
        except Exception as e:  # noqa: BLE001 — telemetry must not break jobs
            logger.warning(f"energy: begin failed: {e}")

    async def end(self, job) -> None:
        """The job stopped running; add its share to the job.

        Added, not set: a resident job demoted mid-flight runs again on the
        batch path, and both runs are its.
        """
        uuid = getattr(job, "gpu_uuid", None)
        if not uuid:
            return
        try:
            share = await asyncio.to_thread(self._end_sync, id(job), uuid)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"energy: end failed: {e}")
            return
        if share is None:
            return
        job.gpu_ms = (job.gpu_ms or 0) + int(round(share.seconds * 1000))
        job.energy_wh = (job.energy_wh or 0.0) + share.joules / 3600.0
        job.energy_measured = share.measured and job.energy_measured is not False

    @contextlib.asynccontextmanager
    async def housekeeping(self, uuid: str | None):
        """Load or unload an instance on card ``uuid`` inside this block.

        The begin sits inside the ``try`` for the same reason as a job's: a
        cancellation during its read still runs the end, and the end that
        overtakes it leaves no share behind.
        """
        if not uuid:
            yield
            return
        key = ("housekeeping", next(self._housekeeping_keys))
        try:
            try:
                await asyncio.to_thread(self._begin_sync, key, uuid)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"energy: housekeeping begin failed: {e}")
            yield
        finally:
            try:
                await asyncio.to_thread(self._end_sync, key, uuid, True)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"energy: housekeeping end failed: {e}")

    async def tick(self) -> list[Idle]:
        """Read every card; return the idle energy since the last tick."""
        try:
            return await asyncio.to_thread(self._tick_sync)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"energy: tick failed: {e}")
            return []


_meter: EnergyMeter | None = None


def get_meter() -> EnergyMeter:
    global _meter
    if _meter is None:
        _meter = EnergyMeter()
    return _meter
