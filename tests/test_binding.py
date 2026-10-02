# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Per-model device bindings: where a model loads, and what follows from it.

A binding is one field, but it changes four answers: which card the VRAM gate
measures, which residents may be evicted to make room, which port the model's
server binds, and which environment its child process is spawned with. These
cover all four, plus the precedence of the binding itself.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from giq import gpus
from giq.models import JobRequest
from giq.policy import reset_policy_store
from giq.queue import JobQueue
from giq.runner import RESIDENT, Instance, Runner

BIG = "GPU-8f6adead-beef-0000-0000-c0ffee000001"
SMALL = "GPU-8f6adead-beef-0000-0000-c0ffee000002"

TWO_CARDS = (
    f"{BIG}, 0, NVIDIA GeForce RTX 5090, 32607, 4921, 30, 9.00, 500.00, 0, 0, 0x0\n"
    f"{SMALL}, 1, NVIDIA GeForce RTX 5060 Ti, 16311, 811, 36, 9.00, 180.00, 3, 0, 0x0\n"
)


class FakeResult:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


@pytest.fixture
def two_cards(monkeypatch):
    """Two visible cards and a config the test controls."""
    gpus._cache = None
    gpus.reset_selected_device()

    def configure(*, device=None, bind=None, reserve=None, residents=()):
        monkeypatch.setattr(
            "giq.config.get_config",
            lambda: SimpleNamespace(
                gpu=SimpleNamespace(device=device, bind=bind or {}, reserve=reserve or {}),
                residents=list(residents),
            ),
        )
        return patch("giq.gpus.subprocess.run", return_value=FakeResult(TWO_CARDS))

    yield configure
    gpus._cache = None
    gpus.reset_selected_device()


@pytest.fixture
def store():
    """A policy store with a clean DB-backed override table."""
    from giq.stats import get_stats

    stats = get_stats()
    for name in list(stats.load_policies()):
        stats.delete_policy(name)
    for name in list(stats.load_devices()):
        stats.save_device(name, None)
    s = reset_policy_store()
    s.load()
    yield s
    for name in list(stats.load_devices()):
        stats.save_device(name, None)
    for name in list(stats.load_policies()):
        stats.delete_policy(name)
    reset_policy_store()


# --- resolution and precedence ----------------------------------------------


def test_unbound_models_land_on_the_selected_card(two_cards, store):
    with two_cards():
        assert store.device_for("gemma-4-12b") is None
        assert store.effective_device("gemma-4-12b") == BIG  # biggest


def test_config_bind_places_a_model(two_cards, store):
    with two_cards(bind={"flux_klein": "1"}):
        device, source = store.device_record("flux_klein")
    assert device == SMALL
    assert source == "config"


def test_an_override_outranks_config(two_cards, store):
    with two_cards(bind={"flux_klein": "1"}):
        store.set_device("flux_klein", "0")
        device, source = store.device_record("flux_klein")
    assert device == BIG
    assert source == "override"


def test_binding_is_stored_as_a_uuid_not_an_index(two_cards, store):
    """An index is this boot's enumeration order; a binding outlives that."""
    from giq.stats import get_stats

    with two_cards():
        store.set_device("kokoro", "1")
    assert get_stats().load_devices()["kokoro"] == SMALL


def test_binding_survives_a_reload(two_cards, store):
    with two_cards():
        store.set_device("kokoro", "1")
        reloaded = reset_policy_store()
        reloaded.load()
        assert reloaded.device_for("kokoro") == SMALL


def test_binding_and_residency_are_independent(two_cards, store):
    """Unpinning must not unbind, and binding must not pin."""
    from giq.policy import AUTO, PINNED

    with two_cards():
        store.set_device("kokoro", "1")
        assert store.policy_for("kokoro") == AUTO  # binding didn't pin it

        store.set("kokoro", PINNED)
        assert store.device_for("kokoro") == SMALL  # pinning didn't unbind

        store.clear("kokoro")
        assert store.policy_for("kokoro") == AUTO
        assert store.device_for("kokoro") == SMALL  # still bound


def test_a_binding_to_a_missing_card_falls_back_loudly(two_cards, store, caplog):
    with two_cards():
        store.set_device("kokoro", "1")
    # Card 1 is gone on the next boot.
    gpus._cache = None
    one_card = f"{BIG}, 0, NVIDIA GeForce RTX 5090, 32607, 4921, 30, 9.00, 500.00, 0, 0, 0x0\n"
    with caplog.at_level("WARNING"):
        with patch("giq.gpus.subprocess.run", return_value=FakeResult(one_card)):
            device, source = store.device_record("kokoro")
    assert (device, source) == (None, "default")
    assert "not present" in caplog.text


def test_binding_an_unknown_card_is_refused(two_cards, store):
    with two_cards():
        with pytest.raises(ValueError, match="matches no GPU"):
            store.set_device("kokoro", "7")


# --- what the binding changes ------------------------------------------------


def test_the_vram_gate_measures_the_bound_card(two_cards, store):
    """gemma-4-31b-it (22GB) fits the 5090 and can never fit the 5060 Ti.

    Same model, same question, opposite answers — which is the whole reason
    the gate had to learn which card it was talking about.
    """
    from giq.vram import can_load

    with two_cards():
        ok_big, _ = can_load("gemma-4-31b-it")
        store.set_device("gemma-4-31b-it", "1")
        ok_small, why = can_load("gemma-4-31b-it")
    assert ok_big is True
    assert ok_small is False
    assert "can never fit" in why and "15.9GB total" in why


def test_a_reserve_is_deducted_on_the_card_that_declares_it(two_cards, store):
    from giq.vram import can_load

    with two_cards(bind={"flux_klein": "1"}, reserve={"1": 6.0}):
        # klein needs 8 + 2 margin = 10; 15.1 free covers it, 15.1 - 6 does not.
        ok, why = can_load("flux_klein")
    assert ok is False
    assert "6.0GB reserved" in why


@pytest.mark.asyncio
async def test_eviction_only_considers_residents_on_the_target_card(two_cards, store):
    """The payoff: a render on the small card cannot cost you the big card's LLM."""
    runner = Runner(JobQueue())

    def resident(key, gb, device):
        worker = AsyncMock()
        worker.is_ready = True
        worker.estimated_vram_gb = gb
        worker.pid = None
        res = Instance(worker, key, device, residency=RESIDENT)
        res.last_active = 0.0
        runner._residents[key] = res
        return res

    with two_cards(bind={"zimage": "1"}):
        gemma = resident("gemma-4-12b", 9.5, BIG)
        kokoro = resident("kokoro", 1.0, SMALL)
        # zimage wants 13 + 2 = 15 on the small card, which has 15.1 free —
        # but only 0.5 once we pretend the desktop grew.
        with patch("giq.runner.get_free_vram", lambda *a: 0.5):
            with patch("giq.gpus.compute_app_memory", lambda *a: {}):
                await runner._evict_residents_for("zimage")

    assert "gemma-4-12b" in runner._residents  # untouched
    gemma.adapter.stop.assert_not_awaited()
    assert "kokoro" not in runner._residents  # its card, its cost
    kokoro.adapter.stop.assert_awaited()


def test_each_card_gets_its_own_server_port(two_cards, store):
    """Two llama-servers cannot both bind 8086."""
    from giq.adapters.llama_cpp import INTERNAL_LLM_PORT
    from giq.adapters.sdcpp import INTERNAL_SD_PORT
    from giq.gpus import device_port

    with two_cards():
        assert device_port(INTERNAL_LLM_PORT, BIG) == 8086  # unchanged for card 0
        assert device_port(INTERNAL_LLM_PORT, SMALL) == 8096
        assert device_port(INTERNAL_SD_PORT, BIG) == 8087
        assert device_port(INTERNAL_SD_PORT, SMALL) == 8097
        # The two series never collide, whichever cards are in play.
        llm = {device_port(INTERNAL_LLM_PORT, d) for d in (BIG, SMALL)}
        sd = {device_port(INTERNAL_SD_PORT, d) for d in (BIG, SMALL)}
        assert not (llm & sd)


def test_a_bound_llm_config_takes_its_cards_port(two_cards, store):
    from giq.adapters.llama_cpp import LlamaCppConfig

    with two_cards():
        store.set_device("gemma-4-12b", "1")
        config = LlamaCppConfig(model="gemma-4-12b")
    assert config.device == SMALL
    assert config.port == 8096


def test_the_child_environment_names_the_bound_card(two_cards, store):
    with two_cards(bind={"kokoro": "1"}):
        from giq.adapters.tts import TtsAdapter, TtsConfig

        worker = TtsAdapter(TtsConfig(model="kokoro"))
        env = worker._spawn_env()
    assert env["CUDA_VISIBLE_DEVICES"] == SMALL


@pytest.mark.asyncio
async def test_a_slot_per_card_survives_the_other_cards_load(two_cards, store):
    """Loading on one card no longer unloads the worker on the other."""
    from giq.runner import Instance

    runner = Runner(JobQueue())
    with two_cards(bind={"flux_klein": "1"}):
        other = AsyncMock()
        runner._slots[BIG] = Instance(other, "llama-3.2-3b", BIG)

        built = AsyncMock()
        built.is_ready = True
        built.is_running = True
        with patch.object(runner, "_build_worker", return_value=built):
            with patch("giq.runner.wait_for_vram", AsyncMock(return_value=True)):
                worker = await runner._ensure_worker("flux_klein")

    assert worker is built
    assert runner._slots[SMALL].model == "flux_klein"
    assert runner._slots[BIG].model == "llama-3.2-3b"  # still loaded
    other.stop.assert_not_awaited()


def test_the_pinned_budget_is_per_card(two_cards, store):
    """gemma+whisper+ecapa on the big card leaves the small one nearly empty."""
    from giq.policy import PINNED
    from tests.conftest import CLASSIC_RESIDENTS

    with two_cards(residents=CLASSIC_RESIDENTS):
        store.set("kokoro", PINNED)
        store.set_device("kokoro", "1")
        big = store.pinned_vram_gb(device=BIG)
        small = store.pinned_vram_gb(device=SMALL)
        fits, projected, total, device = store.pinned_fit(extra="kokoro")
    assert big == pytest.approx(14.1)  # gemma 9.5 + whisper 4.0 + ecapa 0.6
    assert small == pytest.approx(1.0)  # kokoro alone
    assert (fits, device) == (True, SMALL)
    assert total == pytest.approx(15.93, abs=0.01)
    assert projected == pytest.approx(1.5)  # kokoro + headroom


@pytest.mark.asyncio
async def test_rebinding_a_loaded_resident_moves_it(two_cards, store):
    """A running process cannot change cards, so it has to be unloaded first.

    Without this, binding a pinned model would quietly do nothing until the
    next eviction — the one case where the operator most expects it to act.
    """
    from giq.policy import PINNED

    runner = Runner(JobQueue(), use_policy=True)
    with two_cards():
        store.set("kokoro", PINNED)
        worker = AsyncMock()
        worker.is_ready = True
        runner._residents["kokoro"] = Instance(worker, "kokoro", BIG, residency=RESIDENT)

        await runner._release_demoted_residents()
        assert "kokoro" in runner._residents  # still where it belongs

        store.set_device("kokoro", "1")
        await runner._release_demoted_residents()

    assert "kokoro" not in runner._residents
    worker.stop.assert_awaited()  # the residents loop reloads it on card 1


# --- engines -----------------------------------------------------------------
# giq could describe every model file and nothing about the binaries running
# them, which is how a destroyed llama.cpp source tree served for four months.


def test_engine_binary_comes_from_config_not_path(monkeypatch, tmp_path):
    """The bare-name PATH lookup is gone: one declared path per engine."""
    from giq import engines

    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\necho 'version: 1 (deadbeef)'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(
        "giq.config.get_config",
        lambda: SimpleNamespace(engines={"llama.cpp": str(fake)}, gpu=SimpleNamespace()),
    )
    engines.reload_engines()
    try:
        assert engines.binary_for("llama.cpp") == str(fake)
        assert engines.require_binary("llama.cpp") == str(fake)
        assert engines.probe("llama.cpp", refresh=True)["version"] == "version: 1 (deadbeef)"
    finally:
        engines.reload_engines()


def test_a_missing_engine_fails_loudly_at_spawn_not_at_config(monkeypatch):
    """Config building happens in tests and on machines with no engines; the
    complaint belongs at the moment something is actually being run."""
    from giq import engines

    monkeypatch.setattr(
        "giq.config.get_config",
        lambda: SimpleNamespace(engines={"llama.cpp": "/nope/llama-server"}, gpu=SimpleNamespace()),
    )
    engines.reload_engines()
    try:
        assert engines.binary_for("llama.cpp") == "/nope/llama-server"  # no raise
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            engines.require_binary("llama.cpp")
        assert engines.probe("llama.cpp", refresh=True)["present"] is False
    finally:
        engines.reload_engines()


def test_undeclared_engine_is_an_error():
    from giq import engines

    with pytest.raises(ValueError, match="unknown engine"):
        engines.binary_for("tensorrt")


def test_every_backend_maps_to_a_runtime():
    """A model whose backend has no engine would report a runtime of its own
    label, which reads as an engine that does not exist."""
    from giq.engines import ENGINE_OF_BACKEND
    from giq.registry import all_recipes

    unmapped = {r.engine for r in all_recipes()} - set(ENGINE_OF_BACKEND)
    assert not unmapped, f"backends with no declared runtime: {unmapped}"


# --- vision ------------------------------------------------------------------


def test_multimodal_content_is_detected_not_dropped():
    """extract_text_content keeps only type == "text"; before this check an
    image part was discarded in silence and the model answered about nothing."""
    from giq.api.openai_compat import extract_text_content, has_non_text_parts, is_multimodal

    image = [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    assert has_non_text_parts(image) is True
    assert has_non_text_parts([{"type": "text", "text": "hi"}]) is False
    assert has_non_text_parts("plain string") is False
    assert extract_text_content(image) == "what is this?"  # the lossy path, still there

    class M:
        def __init__(self, content):
            self.content = content

    assert is_multimodal([M("hi"), M(image)]) is True
    assert is_multimodal([M("hi"), M([{"type": "text", "text": "x"}])]) is False


def test_vision_models_declare_a_projector_requirement():
    """A vision model without --mmproj loads and serves text, accepting images
    and ignoring them — the exact silent failure this feature exists to end."""
    from giq.registry import all_recipes

    seers = [r for r in all_recipes() if r.vision]
    assert seers, "expected at least one vision-capable recipe"
    for recipe in seers:
        assert recipe.modalities == ("llm",), "vision is a capability of an LLM, not a modality"


# --- a second server of one engine on a card ---------------------------------------


def _taken(monkeypatch, *ports: int) -> None:
    """Ports some live server holds, as giq.gpus._bindable sees them."""
    monkeypatch.setattr(gpus, "_bindable", lambda port, host="127.0.0.1": port not in ports)


def test_a_server_keeps_its_engines_port_while_it_is_free(two_cards, monkeypatch):
    _taken(monkeypatch)
    with two_cards():
        assert gpus.server_port(8096, SMALL) == 8096


def test_a_second_server_on_a_card_takes_a_spare_port_of_that_card(two_cards, monkeypatch):
    """A pinned model and an on-demand one of the same engine used to share
    the card's port: the second could not bind it (and a second vllm stopped
    the first's scope, which is named after the port)."""
    with two_cards():
        _taken(monkeypatch, 8096)
        assert gpus.server_port(8096, SMALL) == 8105
        _taken(monkeypatch, 8096, 8105)
        assert gpus.server_port(8096, SMALL) == 8104
        # Never a port of another card's block.
        _taken(monkeypatch, *range(8096, 8106))
        with pytest.raises(RuntimeError, match="no free internal port"):
            gpus.server_port(8096, SMALL)


@pytest.mark.asyncio
async def test_llama_moves_off_a_held_port_at_start(two_cards, store, monkeypatch):
    from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig

    with two_cards():
        store.set_device("gemma-4-12b", "1")
        worker = LlamaCppAdapter(config=LlamaCppConfig(model="gemma-4-12b"))
        _taken(monkeypatch, 8096)
        await worker._claim_port()
    assert worker.config.port == 8105
    assert worker.base_url.endswith(":8105")


@pytest.mark.asyncio
async def test_the_stale_sweep_covers_every_port_of_every_card(two_cards, monkeypatch):
    from giq.adapters import vllm
    from giq.core import lifecycle

    calls: list[tuple[str, ...]] = []

    class _Done:
        async def wait(self):
            return 0

    async def spawn(*argv, **_kw):
        calls.append(argv)
        return _Done()

    async def no_sleep(_s):
        return None

    stopped: list[str] = []
    monkeypatch.setattr(lifecycle.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(lifecycle.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(vllm, "stop_scope", stopped.append)
    with two_cards():
        await lifecycle._kill_stale_servers()

    patterns = {argv[3] for argv in calls if argv[0] == "pkill"}
    for p in (*range(8086, 8096), *range(8096, 8106)):
        assert rf"llama-server .*--port {p}\b" in patterns
        assert rf"vllm serve .*--port {p}\b" in patterns
        assert f"giq-vllm-{p}" in stopped
    # fuser kills whatever holds a port, so it stays off the spares.
    fused = {argv[2] for argv in calls if argv[0] == "fuser"}
    assert fused == {"8086/tcp", "8096/tcp"}


# --- /status on a machine with two cards -----------------------------------------


async def _status(monkeypatch, *jobs, loaded=()):
    from giq.api import router as api
    from giq.queue import Job
    from giq.registry import get_recipe

    queued = [
        Job(
            job_id=f"j{i}",
            request=JobRequest(modality=get_recipe(m).modality, model=m, chat_request={}),
        )
        for i, m in enumerate(jobs)
    ]
    queue = SimpleNamespace(get_all=AsyncMock(return_value=queued))
    runner = SimpleNamespace(
        owned_pids={},
        pause_state={"paused": False, "since": None, "reason": None},
        active_modality=None,
        active_recipe=None,
        active_slots=[],
        loaded_keys=lambda: set(loaded),
    )
    monkeypatch.setattr(api, "get_queue", lambda: queue)
    monkeypatch.setattr(api, "get_runner", lambda: runner)
    monkeypatch.setattr(api, "attribute_vram", lambda owned, gpus=None: {})
    monkeypatch.setattr(api, "access_posture", lambda: {})
    return await api.get_service_status()


@pytest.mark.asyncio
async def test_status_reports_every_card(two_cards, monkeypatch):
    with two_cards():
        status = await _status(monkeypatch)
    assert [(g["uuid"], g["selected"]) for g in status.gpus] == [(BIG, True), (SMALL, False)]
    small = status.gpus[1]
    assert small["vram_total_gb"] == pytest.approx(16311 / 1024, abs=0.01)
    # The one-card fields still describe the default card.
    assert status.gpu["uuid"] == BIG
    assert status.vram_total_gb == pytest.approx(32607 / 1024, abs=0.01)
    assert status.vram_ok


@pytest.mark.asyncio
async def test_a_job_waits_on_its_own_cards_vram(two_cards, store, monkeypatch):
    """22 GB bound to a 16 GB card is blocked there, however empty the
    default card is; the default card's free figure would say otherwise."""
    with two_cards():
        store.set_device("qwen3.8-27b", "1")
        status = await _status(monkeypatch, "qwen3.8-27b")
    assert not status.vram_ok
    assert status.vram_blocked_gpu == SMALL
    assert "GPU 1" in status.vram_message
    assert status.vram_free_gb > 22  # the default card has the room — the wrong card
    assert status.state == "blocked"


@pytest.mark.asyncio
async def test_a_job_that_fits_its_card_or_is_loaded_is_not_blocked(two_cards, store, monkeypatch):
    with two_cards():
        store.set_device("gemma-4-12b", "1")
        fits = await _status(monkeypatch, "gemma-4-12b")
        store.set_device("qwen3.8-27b", "1")
        loaded = await _status(monkeypatch, "qwen3.8-27b", loaded={"qwen3.8-27b"})
    assert fits.vram_ok and fits.vram_blocked_gpu is None
    assert loaded.vram_ok


def test_a_bind_key_written_before_adr_003_still_binds(two_cards, store):
    """`text2image/flux_klein` was the key form before names were unique."""
    from giq.policy import get_policy_store

    with two_cards(bind={"text2image/flux_klein": "1"}):
        assert get_policy_store().device_record("flux_klein") == (SMALL, "config")
