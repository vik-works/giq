# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Which residents get evicted to make room.

Evicting a resident is an outage of that capability, not a cost proportional
to its size — so the victim *count* is what to minimise, then the VRAM freed.
Cheapest-first accumulation got this backwards on a packed card: "cheapest"
picks the small always-on models (ecapa, whisper) first, so a deficit gemma
could nearly cover alone took the whole resident set down.
"""

from unittest.mock import MagicMock

import pytest

from giq.registry import get_recipe, reload_registry
from giq.runner import _choose_victims
from giq.vram import get_vram_requirement, margin_for

GEMMA = "gemma-4-12b"
WHISPER = "whisper-large-v3"
ECAPA = "ecapa-tdnn"


def _loaded(**sizes):
    """Residents as the runner holds them, sorted smallest first."""
    out = []
    for key, gb in sizes.items():
        res = MagicMock()
        res.adapter.estimated_vram_gb = gb
        out.append(((key[0], key[1]), res))
    return sorted(out, key=lambda kv: kv[1].adapter.estimated_vram_gb)


def _keys(victims):
    return {key for key, _res in victims}


@pytest.fixture(autouse=True)
def _registry():
    reload_registry()
    yield
    reload_registry()


def standard_set():
    out = []
    for key, gb in ((GEMMA, 9.5), (WHISPER, 4.0), (ECAPA, 0.6)):
        res = MagicMock()
        res.adapter.estimated_vram_gb = gb
        out.append((key, res))
    return sorted(out, key=lambda kv: kv[1].adapter.estimated_vram_gb)


def test_one_resident_covering_the_deficit_is_evicted_alone():
    assert _keys(_choose_victims(standard_set(), deficit=8.0)) == {GEMMA}


def test_the_smallest_sufficient_resident_wins():
    """Not the largest: freeing less leaves more still serving."""
    assert _keys(_choose_victims(standard_set(), deficit=3.0)) == {WHISPER}
    assert _keys(_choose_victims(standard_set(), deficit=0.5)) == {ECAPA}


def test_pairs_beat_evicting_everything():
    """The regression: at a 9.7GB deficit, {gemma, ecapa} covers it.

    Cheapest-first accumulated ecapa -> whisper -> gemma and took all three.
    """
    victims = _keys(_choose_victims(standard_set(), deficit=9.7))
    assert victims == {GEMMA, ECAPA}
    assert WHISPER not in victims


def test_fewest_victims_beats_least_vram_freed():
    """{gemma, ecapa} frees 10.1 and {whisper, ecapa} only 4.6 — but a
    single-model answer, if one exists, beats both."""
    assert len(_choose_victims(standard_set(), deficit=9.4)) == 1


def test_ties_on_count_prefer_freeing_less():
    """At equal victim count, take the pair that frees least.

    A 10.0GB deficit needs two: {gemma, ecapa} frees 10.1 and {gemma, whisper}
    frees 13.5. Both are two outages; the first keeps whisper serving.
    """
    assert _keys(_choose_victims(standard_set(), deficit=10.0)) == {GEMMA, ECAPA}


def test_one_big_victim_beats_two_small_ones():
    """A 4.5GB deficit is covered by gemma alone or by whisper+ecapa.

    One outage beats two, even though it frees more VRAM than needed — the
    cost of eviction is the capability going away, not the bytes.
    """
    assert _keys(_choose_victims(standard_set(), deficit=4.5)) == {GEMMA}


def test_impossible_deficit_evicts_everything():
    """Nothing covers it: evict all and let the VRAM wait fail loudly rather
    than silently under-freeing and hanging."""
    assert len(_choose_victims(standard_set(), deficit=99.0)) == 3


def test_no_residents_is_no_victims():
    assert _choose_victims([], deficit=5.0) == []


def test_a_flux_render_costs_whisper_only_on_a_nearly_full_card():
    """The user-visible property, at every realistic free-VRAM level.

    flux_klein's declared footprint decides this. Declared to cover a 2.4 MP
    edit (10.0 plus margin against an 11.8 peak), gemma, with ecapa at worst,
    covers the deficit from 2GB free; below that whisper goes down with them,
    the price of full-size renders.
    """
    base = get_vram_requirement("flux_klein")
    required = base + margin_for(base)
    for free in (2.0, 2.5, 3.0, 4.0):
        victims = _keys(_choose_victims(standard_set(), deficit=required - free))
        assert WHISPER not in victims, f"whisper evicted at {free}GB free"
    for free in (0.0, 1.0, 1.5):
        victims = _keys(_choose_victims(standard_set(), deficit=required - free))
        assert WHISPER in victims, f"whisper spared at {free}GB free"


def test_flux_klein_is_declared_at_its_measured_footprint():
    """Guards the number the property above depends on."""
    recipe = get_recipe("flux_klein")
    assert recipe is not None and recipe.modalities == ("text2image", "image_edit")
    assert recipe.vram_gb == 10.0, "measured 11.8GB peak for a 2.4 MP edit, inside it plus margin"
    assert recipe.measured is True


def test_zimage_still_needs_the_whole_card():
    """Not everything can be helped: 13GB + margin of a 15.93GB card."""
    base = get_vram_requirement("zimage")
    required = base + margin_for(base)
    victims = _keys(_choose_victims(standard_set(), deficit=required - 1.0))
    assert victims == {GEMMA, WHISPER, ECAPA}


# --- gate size vs victim size ------------------------------------------------


def sized(**by_key):
    """Residents whose *declared* size differs from what they actually hold."""
    out = []
    for key, (declared, pid) in by_key.items():
        res = MagicMock()
        res.adapter.estimated_vram_gb = declared
        res.adapter.pid = pid
        out.append((key, res))
    return out


def test_victim_size_can_differ_from_the_declared_gate_size():
    """The planner must size victims by what they hold, not what they declare.

    gemma declares 9.5GB and holds 8.88. Planning with 9.5 believes evicting
    it frees half a gigabyte that does not exist.
    """
    loaded = [(GEMMA, MagicMock()), (WHISPER, MagicMock())]
    loaded[0][1].adapter.estimated_vram_gb = 9.5
    loaded[1][1].adapter.estimated_vram_gb = 4.0
    measured = {id(loaded[0][1]): 8.88, id(loaded[1][1]): 3.76}
    size_of = lambda res: measured[id(res)]  # noqa: E731

    # A 9.0GB deficit: gemma covers it on the declared figure but not on the
    # real one, so the honest answer needs a second victim.
    assert _keys(_choose_victims(loaded, 9.0)) == {GEMMA}
    assert _keys(_choose_victims(loaded, 9.0, size_of)) == {GEMMA, WHISPER}


def test_declared_size_is_the_default():
    loaded = [(GEMMA, MagicMock())]
    loaded[0][1].adapter.estimated_vram_gb = 9.5
    assert _choose_victims(loaded, 9.0) == loaded


@pytest.mark.asyncio
async def test_sizer_prefers_measured_then_falls_back(monkeypatch):
    from giq.queue import JobQueue
    from giq.runner import RESIDENT, Instance, Runner

    runner = Runner(JobQueue())
    child = MagicMock()
    child.estimated_vram_gb, child.pid = 9.5, 4242
    inproc = MagicMock()
    inproc.estimated_vram_gb, inproc.pid = 0.5, None
    fresh = MagicMock()
    fresh.estimated_vram_gb, fresh.pid = 4.0, 9999  # allocated nothing yet

    monkeypatch.setattr("giq.gpus.compute_app_memory", lambda *a: {4242: 8.88, 9999: 0.01})
    size_of = await runner._victim_sizer()

    assert size_of(Instance(child, "r", residency=RESIDENT)) == 8.88  # measured wins
    assert size_of(Instance(inproc, "r", residency=RESIDENT)) == 0.5  # no pid -> declared
    assert size_of(Instance(fresh, "r", residency=RESIDENT)) == 4.0  # near-zero reading -> declared


@pytest.mark.asyncio
async def test_sizer_falls_back_wholesale_when_nvidia_smi_fails(monkeypatch):
    from giq.queue import JobQueue
    from giq.runner import RESIDENT, Instance, Runner, _declared_size

    runner = Runner(JobQueue())
    monkeypatch.setattr("giq.gpus.compute_app_memory", lambda *a: {})
    assert await runner._victim_sizer() is _declared_size
    worker = MagicMock()
    worker.estimated_vram_gb, worker.pid = 7.0, 1
    assert _declared_size(Instance(worker, "r", residency=RESIDENT)) == 7.0


def test_kokoro_is_declared_at_its_measured_runtime_footprint():
    """It held 968 MiB and was declared 0.5GB, so the gate let it load into
    space it did not fit in and it OOM'd."""
    spec = get_recipe("kokoro")
    assert spec.vram_gb >= 0.95
