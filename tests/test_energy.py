# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Energy per job (ADR-006 D5): the card's counter, split by overlap.

A fake card stands in for NVML: its counter and the clock move only when the
test moves them, so every share can be stated exactly.
"""

import time
from types import SimpleNamespace

import pytest

from giq.energy import EnergyMeter, Reading

CARD = "GPU-test-0"


class FakeCard:
    def __init__(self, counter=True):
        self.mj = 1_000_000
        self.power_w = 100.0
        self.counter = counter
        self.t = 0.0

    def run(self, seconds, watts):
        """The card draws ``watts`` for ``seconds``."""
        self.t += seconds
        self.mj += int(watts * seconds * 1000)
        self.power_w = watts

    # the reader interface
    def cards(self):
        return [CARD]

    def read(self, uuid):
        return Reading(energy_mj=self.mj if self.counter else None, power_w=self.power_w)


def meter_for(card):
    return EnergyMeter(reader=card, clock=lambda: card.t)


def job():
    return SimpleNamespace(gpu_uuid=None, gpu_ms=None, energy_wh=None, energy_measured=None)


@pytest.mark.asyncio
async def test_a_job_alone_gets_all_of_the_card():
    card = FakeCard()
    m = meter_for(card)
    j = job()

    await m.begin(j, CARD)
    card.run(36, 100)  # 3600 J
    await m.end(j)

    assert j.gpu_uuid == CARD
    assert j.energy_wh == pytest.approx(1.0)
    assert j.gpu_ms == 36_000
    assert j.energy_measured is True


@pytest.mark.asyncio
async def test_overlapping_jobs_split_the_overlap_evenly():
    card = FakeCard()
    m = meter_for(card)
    a, b = job(), job()

    await m.begin(a, CARD)
    card.run(10, 360)  # 3600 J, a alone
    await m.begin(b, CARD)
    card.run(10, 720)  # 7200 J, shared
    await m.end(a)
    card.run(10, 360)  # 3600 J, b alone
    await m.end(b)

    assert a.energy_wh == pytest.approx(1.0 + 1.0)
    assert b.energy_wh == pytest.approx(1.0 + 1.0)
    assert a.gpu_ms == b.gpu_ms == 15_000
    # nothing is lost or counted twice
    assert a.energy_wh + b.energy_wh == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_energy_with_no_job_running_is_idle():
    card = FakeCard()
    m = meter_for(card)
    await m.tick()  # first read: the baseline
    card.run(36, 50)  # 1800 J idle
    j = job()
    await m.begin(j, CARD)
    card.run(36, 100)
    await m.end(j)
    card.run(36, 50)  # 1800 J idle

    idle = await m.tick()

    assert [(i.gpu_uuid, round(i.wh, 6), i.measured) for i in idle] == [(CARD, 1.0, True)]
    assert j.energy_wh == pytest.approx(1.0)
    assert await m.tick() == []  # drained


@pytest.mark.asyncio
async def test_without_a_counter_the_power_draw_is_integrated_and_marked():
    card = FakeCard(counter=False)
    m = meter_for(card)
    j = job()

    card.power_w = 100.0
    await m.begin(j, CARD)
    card.run(36, 100)
    await m.end(j)

    assert j.energy_wh == pytest.approx(1.0)
    assert j.energy_measured is False


@pytest.mark.asyncio
async def test_a_counter_that_went_backwards_falls_back_to_power():
    """A driver reload resets the counter; the interval is estimated, not
    counted as a huge negative or a huge positive."""
    card = FakeCard()
    m = meter_for(card)
    j = job()
    await m.begin(j, CARD)
    card.run(36, 100)
    card.mj = 5  # reset
    await m.end(j)

    assert j.energy_wh == pytest.approx(1.0)
    assert j.energy_measured is False


@pytest.mark.asyncio
async def test_a_job_that_runs_twice_adds_both_runs():
    """A resident demoted mid-flight runs again on the batch path."""
    card = FakeCard()
    m = meter_for(card)
    j = job()
    for _ in range(2):
        await m.begin(j, CARD)
        card.run(36, 100)
        await m.end(j)

    assert j.energy_wh == pytest.approx(2.0)
    assert j.gpu_ms == 72_000


def test_an_end_that_overtakes_its_begin_leaves_no_ghost():
    """A job cancelled while its begin still reads: the end lands first.
    The late begin must not leave a share that splits the card forever."""
    card = FakeCard()
    m = meter_for(card)
    m._end_sync(1, CARD)
    m._begin_sync(1, CARD)
    assert m._cards[CARD].active == {}
    assert m._ended_early == set()


@pytest.mark.asyncio
async def test_no_card_no_figures():
    m = meter_for(FakeCard())
    j = job()
    await m.begin(j, None)
    await m.end(j)
    assert (j.gpu_uuid, j.energy_wh, j.gpu_ms) == (None, None, None)


@pytest.mark.asyncio
async def test_a_failing_reader_never_breaks_the_job():
    class Broken:
        def cards(self):
            raise RuntimeError("driver gone")

        def read(self, uuid):
            raise RuntimeError("driver gone")

    m = EnergyMeter(reader=Broken())
    j = job()
    await m.begin(j, CARD)
    await m.end(j)
    assert await m.tick() == []
    assert j.energy_wh is None


# --- what the stats make of it ------------------------------------------------


@pytest.mark.asyncio
async def test_usage_reports_attributed_and_idle_energy(tmp_path, monkeypatch):
    import giq.stats as stats_mod
    from giq.api.stats_api import stats_usage
    from giq.energy import Idle
    from giq.stats import StatsRecorder

    monkeypatch.setattr(stats_mod, "_recorder", StatsRecorder(db_path=tmp_path / "e.db"))
    rec = stats_mod.get_stats()
    now = time.time()
    for job_id, app, wh, measured in (("j1", "kompost", 2.0, 1), ("j2", "anonymous", 1.0, 0)):
        rec._record_job_sync(
            {
                "ts": now,
                "job_id": job_id,
                "modality": "llm",
                "recipe": "gemma-4-12b",
                "status": "completed",
                "run_ms": 1000,
                "tasks": 1,
                "gpu_uuid": CARD,
                "app": app,
                "gpu_ms": 1000,
                "energy_wh": wh,
                "energy_measured": measured,
            }
        )
    await rec.record_idle([Idle(CARD, 0.5, True)])
    await rec.record_idle([Idle(CARD, 0.25, True)])  # same hour: added up

    everything = await stats_usage(app=None, period="day", gpu=None, since=None, until=None)
    one_app = await stats_usage(app="kompost", period="day", gpu=None, since=None, until=None)
    rec.close()

    e = everything["energy"]
    assert e["attributed_wh"] == pytest.approx(3.0)
    assert e["idle_wh"] == pytest.approx(0.75)
    assert e["estimated"] is True  # j2 was an estimate
    assert e["cards"] == [{"gpu_uuid": CARD, "attributed_wh": 3.0, "idle_wh": 0.75}]
    by_app = {r["app"]: r for r in everything["apps"]}
    assert by_app["kompost"]["energy_wh"] == 2.0
    assert by_app["kompost"]["energy_estimated"] is False
    assert everything["totals"]["energy_wh"] == pytest.approx(3.0)
    # idle belongs to nobody, so one app's view has none
    assert one_app["energy"]["idle_wh"] is None
    assert one_app["energy"]["attributed_wh"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_a_job_records_the_card_it_ran_on(tmp_path):
    from datetime import datetime

    from giq.models import JobRequest
    from giq.queue import Job
    from giq.stats import StatsRecorder

    rec = StatsRecorder(db_path=tmp_path / "c.db")
    rec._primary_gpu = "GPU-primary"
    request = JobRequest(modality="llm", model="m", tasks=[{"id": "t", "user": "hi"}])
    ran = Job(job_id="a", request=request, gpu_uuid=CARD, energy_wh=0.5, energy_measured=True)
    never = Job(job_id="b", request=request)
    for j in (ran, never):
        j.started_at = j.completed_at = datetime.now()
        await rec.record_job(j)

    rows = rec.query("SELECT job_id, gpu_uuid, energy_wh, energy_measured FROM jobs ORDER BY id")
    rec.close()
    assert rows == [("a", CARD, 0.5, 1), ("b", "GPU-primary", None, None)]
