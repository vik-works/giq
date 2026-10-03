# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The vllm engine adapter, without a GPU: command line, environment, checkpoint
checks at start, runner selection, SSE normalisation, readiness and stop."""

import asyncio
import json
import signal

import httpx
import pytest
from pydantic import ValidationError

from giq.adapters.engine import ServedLLM, StartError
from giq.models import Modality
from giq.queue import JobQueue, JobStream
from giq.recipes.schema import Recipe
from giq_vllm import adapter as vllm
from giq_vllm.adapter import (
    VllmAdapter,
    VllmConfig,
    VLLMConfigError,
    check_checkpoint,
    flashinfer_arch,
)
from vllm_fixtures import NAME, doc, make_checkpoint, make_recipe, params_of


def worker_for(recipe: Recipe, **kw) -> VllmAdapter:
    cfg = VllmConfig(
        model=recipe.name,
        recipe=recipe,
        device="GPU-test",
        port=8088,
        python="/opt/giq/envs/vllm/.venv/bin/python",
        log_path=kw.pop("log_path", "/nonexistent/vllm.log"),
        **kw,
    )
    return VllmAdapter(config=cfg)


@pytest.fixture
def operator_dir(tmp_path, monkeypatch):
    """An operator recipes directory, the catalog rebuilt around it."""
    from giq.registry import reload_registry

    monkeypatch.setenv("GIQ_RECIPES_DIR", str(tmp_path / "recipes"))
    (tmp_path / "recipes").mkdir()
    yield tmp_path / "recipes"
    monkeypatch.undo()
    reload_registry()


# --- checkpoint checks at start ---------------------------------------------------


@pytest.mark.parametrize("nested", [True, False])
def test_mtp_head_found_top_level_or_nested(tmp_path, nested):
    weights = make_checkpoint(tmp_path, mtp=1, nested=nested)
    check_checkpoint(weights, make_recipe(weights, "interactive").params)


def test_start_check_refuses_a_gguf(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    with pytest.raises(VLLMConfigError, match="not a file"):
        check_checkpoint(gguf, params_of())


def test_start_check_refuses_a_directory_without_config(tmp_path):
    with pytest.raises(VLLMConfigError, match="config.json"):
        check_checkpoint(tmp_path, params_of())


@pytest.mark.asyncio
async def test_start_checks_the_weights_before_spawning(tmp_path, monkeypatch):
    weights = make_checkpoint(tmp_path, mtp=1)
    worker = worker_for(make_recipe(weights, "interactive"))
    (weights / "config.json").write_text(json.dumps({"text_config": {}}))  # head gone since load

    async def no_spawn(*a, **kw):
        raise AssertionError("spawned a server for weights that cannot run it")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_spawn)
    with pytest.raises(VLLMConfigError, match="MTP"):
        await worker.start()


@pytest.mark.asyncio
async def test_a_second_vllm_on_a_card_moves_port_before_touching_a_scope(tmp_path, monkeypatch):
    """The scope is named after the port, and start() stops a scope of that
    name as left over from a crashed run. With the card's port held by a
    running vllm, the second must move first — or it stops the first."""
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)), log_path=str(tmp_path / "v.log"))
    held = worker.config.port
    stopped: list[str] = []
    spawned: list[tuple] = []

    class Spawned(Exception):
        pass

    async def spawn(*argv, **kw):
        spawned.append(argv)
        raise Spawned

    monkeypatch.setattr(vllm, "device_port", lambda base, device=None: held)
    monkeypatch.setattr(vllm, "server_port", lambda port, device=None: 8095)
    monkeypatch.setattr(vllm, "stop_scope", stopped.append)
    monkeypatch.setattr(vllm, "memory_cap_prefix", lambda unit, cap: [])
    monkeypatch.setattr(vllm, "resolve_device", lambda device: None)
    monkeypatch.setattr("giq.engines.require_binary", lambda engine: "/bin/true")
    monkeypatch.setattr(vllm.os.path, "exists", lambda path: True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    with pytest.raises(Spawned):
        await worker.start()
    assert stopped and all(unit == "giq-vllm-8095" for unit in stopped)
    assert f"giq-vllm-{held}" not in stopped
    assert argv_value(list(spawned[0]), "--port") == "8095"


# --- the command line ----------------------------------------------------------------


def argv_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_command_from_the_throughput_profile(tmp_path):
    weights = make_checkpoint(tmp_path)
    recipe = Recipe.model_validate(
        doc(
            weights,
            "throughput",
            vram={"gb": 29.5},
            kv_cache_memory=None,
            gpu_memory_utilization=0.93,
            reasoning_parser="qwen3",
            tool_call_parser="qwen3_coder",
        )
    )
    cmd = worker_for(recipe).build_command(card_total_gb=31.84)

    assert cmd[:3] == ["/opt/giq/envs/vllm/.venv/bin/vllm", "serve", str(weights)]
    assert argv_value(cmd, "--served-model-name") == NAME
    assert argv_value(cmd, "--host") == "127.0.0.1"
    assert argv_value(cmd, "--port") == "8088"
    assert argv_value(cmd, "--gpu-memory-utilization") == "0.93"
    assert argv_value(cmd, "--max-model-len") == "131072"
    assert argv_value(cmd, "--kv-cache-dtype") == "fp8"
    assert argv_value(cmd, "--max-num-seqs") == "32"
    assert "--kv-cache-memory-bytes" not in cmd
    assert argv_value(cmd, "--reasoning-parser") == "qwen3"
    assert argv_value(cmd, "--tool-call-parser") == "qwen3_coder"
    assert "--enable-auto-tool-choice" in cmd
    assert json.loads(argv_value(cmd, "--structured-outputs-config")) == {
        "backend": "xgrammar",
        "disable_any_whitespace": True,
    }, "a vllm instance forbids free whitespace by default so a large schema converges"
    for absent in ("--speculative-config", "--enforce-eager", "--max-num-batched-tokens"):
        assert absent not in cmd
    assert "--language-model-only" not in cmd, "vision stays on by default"


def test_a_recipe_can_restore_vllm_default_structured_output(tmp_path):
    """The convergent setting is giq's default, not a lock-in: a recipe that
    wants vllm's own permissive grammar (any backend, free whitespace) says so,
    and the adapter spells exactly that onto the command line."""
    recipe = Recipe.model_validate(
        doc(
            make_checkpoint(tmp_path),
            "throughput",
            vram={"gb": 29.5},
            kv_cache_memory=None,
            gpu_memory_utilization=0.93,
            structured_outputs={"backend": "auto", "disable_any_whitespace": False},
        )
    )
    cmd = worker_for(recipe).build_command(card_total_gb=31.84)
    assert json.loads(argv_value(cmd, "--structured-outputs-config")) == {
        "backend": "auto",
        "disable_any_whitespace": False,
    }


def test_disable_any_whitespace_needs_a_grammar_backend(tmp_path):
    """vllm rejects disable_any_whitespace with the auto backend minutes into a
    start; the recipe refuses to load the pair instead."""
    with pytest.raises(ValidationError):
        Recipe.model_validate(
            doc(
                make_checkpoint(tmp_path),
                "throughput",
                vram={"gb": 29.5},
                kv_cache_memory=None,
                gpu_memory_utilization=0.93,
                structured_outputs={"backend": "auto", "disable_any_whitespace": True},
            )
        )


def test_a_kv_budget_sizes_the_start_check_to_the_model(tmp_path):
    """With a byte budget vllm still refuses to start unless free memory covers
    gpu_memory_utilization x card, 0.9 by default — a shared card never would.
    giq passes the model's own figure over the card's size instead."""
    recipe = Recipe.model_validate(
        doc(
            make_checkpoint(tmp_path),
            "interactive",
            vram={"weights_gb": 20.73, "overhead_gb": 3.0},
            kv_cache_memory="8G",
        )
    )
    assert recipe.vram.gb == pytest.approx(31.73), "weights + KV + overhead"

    worker = worker_for(recipe)
    on_pro6000 = worker.build_command(card_total_gb=95.59)
    assert argv_value(on_pro6000, "--kv-cache-memory-bytes") == str(8 * 2**30)
    assert float(argv_value(on_pro6000, "--gpu-memory-utilization")) == pytest.approx(0.332)
    unknown_card = worker.build_command(card_total_gb=None)
    assert "--gpu-memory-utilization" not in unknown_card


def test_command_from_the_interactive_profile(tmp_path):
    weights = make_checkpoint(tmp_path)
    text_only = {**doc(weights, "interactive", enforce_eager=True), "capabilities": ["chat"]}
    cmd = worker_for(Recipe.model_validate(text_only)).build_command()

    assert json.loads(argv_value(cmd, "--speculative-config")) == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    assert argv_value(cmd, "--max-num-seqs") == "4"
    assert argv_value(cmd, "--max-num-batched-tokens") == "8192"
    assert "--enforce-eager" in cmd
    assert "--language-model-only" in cmd, "a multimodal checkpoint served as text"
    assert "--enable-auto-tool-choice" not in cmd


# --- the environment -----------------------------------------------------------------


@pytest.mark.parametrize(
    "cap,arch",
    [
        ("12.0", "12.0f"),
        ("12.1", "12.1a"),
        ("10.0", "10.0a"),
        ("10.3", "10.3a"),
        ("9.0", "9.0a"),
        ("8.9", "8.9"),
    ],
)
def test_flashinfer_arch(cap, arch):
    assert flashinfer_arch(cap) == arch


def test_environment(tmp_path, monkeypatch):
    root = tmp_path / "envs" / "vllm" / ".venv"
    cu13 = root / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    cu13.mkdir(parents=True)
    python = str(root / "bin" / "python")
    seen = {}

    def fake_device_env(device):
        seen["device"] = device
        return {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "GPU-test", "HF_HOME": "/c/hf"}

    monkeypatch.setattr(vllm, "device_env", fake_device_env)
    monkeypatch.setattr(vllm, "compute_capability", lambda device: "12.0")
    monkeypatch.setattr(vllm, "cache_dir", lambda: tmp_path / "cache")

    env = vllm.engine_env("GPU-test", python)

    assert seen["device"] == "GPU-test"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_HOME"] == str(cu13)
    assert env["PATH"].split(":")[:3] == [str(root / "bin"), str(cu13 / "bin"), "/usr/bin"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["VLLM_NO_USAGE_STATS"] == "1", "a local service does not phone home"
    assert env["FLASHINFER_CUDA_ARCH_LIST"] == "12.0f"
    assert env["MAX_JOBS"] == "2"
    assert env["HF_HOME"] == "/c/hf", "giq's cache variables come through device_env"
    assert env["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path / "cache")
    assert env["VLLM_CACHE_ROOT"] == str(tmp_path / "cache" / "vllm")
    assert env["VLLM_CONFIG_ROOT"].startswith(str(tmp_path / "cache")), "~/.config is read-only"


def test_no_cache_dir_leaves_library_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm, "device_env", lambda device: {"PATH": "/usr/bin"})
    monkeypatch.setattr(vllm, "compute_capability", lambda device: None)
    monkeypatch.setattr(vllm, "cache_dir", lambda: None)
    env = vllm.engine_env(None, str(tmp_path / "bin" / "python"))
    assert "FLASHINFER_WORKSPACE_BASE" not in env
    assert "FLASHINFER_CUDA_ARCH_LIST" not in env


def test_memory_ceiling_is_a_scope(monkeypatch):
    monkeypatch.setattr(vllm, "user_systemd_available", lambda: True)
    prefix = vllm.memory_cap_prefix("giq-vllm-8088", "40G")
    assert prefix[:3] == ["systemd-run", "--user", "--scope"]
    assert "--unit=giq-vllm-8088" in prefix
    assert "MemoryMax=40G" in prefix and "MemorySwapMax=0" in prefix
    assert prefix[-1] == "--"


def test_no_user_systemd_runs_uncapped_and_says_so(monkeypatch, caplog):
    monkeypatch.setattr(vllm, "user_systemd_available", lambda: False)
    with caplog.at_level("WARNING"):
        assert vllm.memory_cap_prefix("giq-vllm-8088", "40G") == []
    assert "uncapped" in caplog.text


def test_a_null_ceiling_means_none(monkeypatch):
    monkeypatch.setattr(vllm, "user_systemd_available", lambda: True)
    assert vllm.memory_cap_prefix("u", None) == []


# --- registration and the runner --------------------------------------------------------


def test_an_operator_instance_on_vllm(operator_dir, tmp_path):
    from giq.adapters.llama_cpp import MODEL_PATHS, weights_installed
    from giq.registry import get_recipe, reload_registry

    weights = make_checkpoint(tmp_path)
    (operator_dir / "big.yaml").write_text(
        "name: qwen-big\nmodalities: [llm]\nengine: vllm\nprofile: throughput\n"
        f"weights: {{path: {weights}, format: modelopt}}\n"
        "params: {kv_cache_memory: 24G, max_model_len: 131072, max_num_seqs: 64}\n"
        "vram: {weights_gb: 19.92, overhead_gb: 3.5}\n"
    )
    reload_registry()
    spec = get_recipe("qwen-big")
    assert spec.vram_gb == pytest.approx(47.42) and spec.lanes == 64
    assert weights_installed("qwen-big")
    assert "qwen-big" not in MODEL_PATHS, "llama.cpp's tables stay llama.cpp's"
    worker = VllmAdapter(VllmConfig(model="qwen-big", device="GPU-test", port=8088))
    assert worker.config.weights == str(weights)


def test_runner_builds_a_vllm_worker_for_engine_vllm(operator_dir, tmp_path, monkeypatch):
    from giq.adapters.engine import context_size, engine_for
    from giq.adapters.llama_cpp import LlamaCppAdapter
    from giq.registry import reload_registry
    from giq.runner import Runner, _lane_width

    weights = make_checkpoint(tmp_path)
    (operator_dir / "served.yaml").write_text(
        "name: served\nmodalities: [llm]\nengine: vllm\nprofile: throughput\n"
        f"weights: {{path: {weights}, format: modelopt}}\n"
        "params: {kv_cache_memory: 6G, max_model_len: 65536}\n"
        "vram: {weights_gb: 19.92, overhead_gb: 3.5}\n"
    )
    reload_registry()
    runner = Runner(JobQueue())
    monkeypatch.setattr(runner, "_device_for", lambda model: "GPU-test")

    worker = runner._build_worker("served")
    assert isinstance(worker, VllmAdapter) and isinstance(worker, ServedLLM)
    assert worker.config.device == "GPU-test"
    assert _lane_width("served", worker) == 32
    assert engine_for("served") == "vllm" and context_size("served") == 65536

    llama = runner._build_worker("qwen3.8-27b")
    assert isinstance(llama, LlamaCppAdapter) and isinstance(llama, ServedLLM)
    assert _lane_width("qwen3.8-27b", llama) == 4, "llama.cpp lanes unchanged"
    assert engine_for("qwen3.8-27b") == "llama.cpp"


def test_lanes_follow_max_num_seqs(tmp_path):
    from giq.runner import _lane_width

    worker = worker_for(make_recipe(make_checkpoint(tmp_path), "interactive"))
    assert _lane_width(NAME, worker) == 4
    c = worker.concurrency()
    assert (c.max_parallel, c.per_request_context, c.shared_kv) == (4, 131072, True)


def test_http_timeout_lets_the_job_timeout_fire_first(tmp_path):
    from giq.models import JobRequest
    from giq.queue import Job
    from giq.runner import _job_timeout

    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    job = Job(job_id="j", request=JobRequest(modality="llm", model=NAME, chat_request={}))
    assert _job_timeout(Modality.llm, job) < worker.http_timeout().read


# --- serving: recorded vllm chunks ------------------------------------------------------
#
# The shape vllm 0.30 streams for a thinking model with the qwen3 reasoning
# parser: the role first, the thought on `reasoning`, the answer on
# `content`, a finish_reason, then a usage-only chunk with no choices.


def chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> dict:
    out = {
        "id": "chatcmpl-9f2",
        "object": "chat.completion.chunk",
        "created": 1790000000,
        "model": NAME,
        "choices": []
        if delta is None
        else [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
    }
    if usage is not None:
        out["usage"] = usage
    return out


VLLM_STREAM = [
    chunk({"role": "assistant", "content": ""}),
    chunk({"reasoning": "17 times 23"}),
    chunk({"reasoning": " is 391."}),
    chunk({"content": "The answer "}),
    chunk({"content": "is 391."}),
    chunk({"content": ""}, finish="stop"),
    chunk(usage={"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}),
]


def sse(chunks: list[dict]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def serving_worker(tmp_path, handler) -> VllmAdapter:
    served = {**doc(make_checkpoint(tmp_path)), "request_defaults": {"top_k": 20, "min_p": 0.0}}
    worker = worker_for(Recipe.model_validate(served))
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker._ready = True

    class _Alive:
        returncode = None
        pid = 4242

    worker._process = _Alive()  # type: ignore[assignment]
    return worker


@pytest.mark.asyncio
async def test_stream_is_normalised_to_reasoning_content(tmp_path):
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, content=sse(VLLM_STREAM))

    worker = serving_worker(tmp_path, handler)
    stream = JobStream()
    body = {
        "messages": [{"role": "user", "content": "17*23?"}],
        "temperature": 0.7,
        "giq_loop_guard": True,
        "reasoning_budget_tokens": 4096,
    }
    result = await worker.chat_completion_stream(body, stream)

    relayed = []
    while not stream.queue.empty():
        relayed.append(stream.queue.get_nowait())
    deltas = [c["choices"][0]["delta"] for c in relayed if c["choices"]]
    assert all("reasoning" not in d for d in deltas), "vllm's field name must not leak"
    assert [d["reasoning_content"] for d in deltas if "reasoning_content" in d] == [
        "17 times 23",
        " is 391.",
    ]

    message = result["choices"][0]["message"]
    assert message["content"] == "The answer is 391."
    assert message["reasoning_content"] == "17 times 23 is 391."
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 9
    assert result["id"] == "chatcmpl-9f2" and result["model"] == NAME

    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    assert sent["top_k"] == 20 and sent["min_p"] == 0.0, "recipe request defaults apply"
    assert "giq_loop_guard" not in sent and "reasoning_budget_tokens" not in sent


@pytest.mark.asyncio
async def test_the_caller_outranks_request_defaults(tmp_path):
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, content=sse(VLLM_STREAM))

    worker = serving_worker(tmp_path, handler)
    await worker.chat_completion_stream({"messages": [], "top_k": 5}, JobStream())
    assert sent["top_k"] == 5


@pytest.mark.asyncio
async def test_non_streamed_answer_is_normalised(tmp_path):
    answer = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "391", "reasoning": "17*23"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    worker = serving_worker(tmp_path, lambda request: httpx.Response(200, json=answer))
    data = await worker.chat_completion({"messages": []})
    message = data["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "391", "reasoning_content": "17*23"}

    results = await worker.run_batch([{"id": "t1", "user": "17*23?"}])
    assert results[0].output == "391" and results[0].tokens_out == 3


@pytest.mark.asyncio
async def test_an_error_carries_vllms_message(tmp_path):
    worker = serving_worker(
        tmp_path, lambda request: httpx.Response(400, json={"message": "max_tokens too large"})
    )
    with pytest.raises(RuntimeError, match="max_tokens too large"):
        await worker.chat_completion_stream({"messages": []}, JobStream())


@pytest.mark.asyncio
async def test_active_requests_from_metrics(tmp_path):
    metrics = (
        "# HELP vllm:num_requests_running Number of requests in model execution batches.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        f'vllm:num_requests_running{{engine="0",model_name="{NAME}"}} 3.0\n'
        f'vllm:num_requests_waiting{{engine="0",model_name="{NAME}"}} 2.0\n'
        f'vllm:num_requests_waiting_by_reason{{engine="0",reason="capacity"}} 2.0\n'
    )
    worker = serving_worker(tmp_path, lambda request: httpx.Response(200, text=metrics))
    assert await worker.active_slot_count() == 5


# --- readiness and stop ------------------------------------------------------------------


class FakeProcess:
    """An asyncio process whose group dies on the signal that is set to kill it."""

    def __init__(self, dies_on: int | None = signal.SIGTERM):
        self.pid = 4242
        self.returncode: int | None = None
        self.dies_on = dies_on
        self._exited = asyncio.Event()

    def die(self, code: int) -> None:
        self.returncode = code
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode or 0


@pytest.fixture
def fast_stop(monkeypatch):
    monkeypatch.setattr(vllm, "STOP_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(vllm, "KILL_WAIT_SECONDS", 0.2)


def install_group(monkeypatch, process: FakeProcess) -> list[int]:
    signals: list[int] = []
    group_alive = {"yes": True}

    def killpg(pgid, sig):
        assert pgid == process.pid
        if sig != 0:
            signals.append(sig)
        if not group_alive["yes"]:
            raise ProcessLookupError
        if sig == process.dies_on or sig == signal.SIGKILL:
            process.die(-sig)
            group_alive["yes"] = False

    monkeypatch.setattr(vllm.os, "killpg", killpg)
    return signals


@pytest.mark.asyncio
async def test_stop_terminates_the_group_and_the_scope(tmp_path, monkeypatch, fast_stop):
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    process = FakeProcess(dies_on=signal.SIGTERM)
    worker._process = process  # type: ignore[assignment]
    worker._ready = True
    worker._scoped = True
    signals = install_group(monkeypatch, process)
    stopped = []
    monkeypatch.setattr(vllm, "stop_scope", stopped.append)

    await worker.stop()

    assert signals == [signal.SIGTERM]
    assert stopped == ["giq-vllm-8088"]
    assert worker._process is None and not worker.is_ready


@pytest.mark.asyncio
async def test_stop_escalates_to_sigkill(tmp_path, monkeypatch, fast_stop):
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    process = FakeProcess(dies_on=None)
    worker._process = process  # type: ignore[assignment]
    signals = install_group(monkeypatch, process)
    monkeypatch.setattr(vllm, "stop_scope", lambda unit: None)

    await worker.stop()

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert worker._process is None


@pytest.mark.asyncio
async def test_an_unkillable_server_keeps_its_reference(tmp_path, monkeypatch, fast_stop):
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    process = FakeProcess(dies_on=None)
    worker._process = process  # type: ignore[assignment]
    monkeypatch.setattr(vllm.os, "killpg", lambda pgid, sig: None)  # nothing ever dies

    with pytest.raises(RuntimeError, match="unkillable"):
        await worker.stop()
    assert worker._process is process, "the runner must know the VRAM is still held"


@pytest.mark.asyncio
async def test_stop_of_an_exited_server_only_cleans_up(tmp_path, monkeypatch, fast_stop):
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    process = FakeProcess()
    process.die(1)
    worker._process = process  # type: ignore[assignment]
    signals = install_group(monkeypatch, process)
    monkeypatch.setattr(
        vllm.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(ProcessLookupError())
    )

    await worker.stop()
    assert signals == [] and worker._process is None


@pytest.mark.asyncio
async def test_a_start_that_dies_fails_at_once_with_the_log(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("loading\nValueError: Free memory on device is less than desired\n")
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)), log_path=str(log))
    process = FakeProcess()
    process.die(1)
    worker._process = process  # type: ignore[assignment]
    worker._client = httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(lambda r: httpx.Response(503)),
    )
    with pytest.raises(RuntimeError, match="Free memory on device"):
        await worker._wait_for_ready(timeout=5, poll=0.01)


@pytest.mark.asyncio
async def test_ready_needs_health_and_the_model_listed(tmp_path):
    calls = {"health": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            calls["health"] += 1
            return httpx.Response(200 if calls["health"] > 2 else 503)
        return httpx.Response(200, json={"data": [{"id": NAME}]})

    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    worker._process = FakeProcess()  # type: ignore[assignment]
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    await worker._wait_for_ready(timeout=5, poll=0.01)
    assert calls["health"] == 3


@pytest.mark.asyncio
async def test_ready_times_out(tmp_path):
    worker = worker_for(make_recipe(make_checkpoint(tmp_path)))
    worker._process = FakeProcess()  # type: ignore[assignment]
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    # A start that outlasts its budget is a failed start, not a job timeout.
    with pytest.raises(StartError, match="did not become ready"):
        await worker._wait_for_ready(timeout=0.05, poll=0.01)


# --- giq prepare vllm ----------------------------------------------------------------------


def test_prepare_builds_the_gemm_modules_for_the_arch(monkeypatch):
    monkeypatch.setattr(vllm, "user_systemd_available", lambda: True)
    cmd = vllm.prepare_command("12.0", "40G", python="/env/bin/python")
    assert cmd[0] == "systemd-run" and "MemoryMax=40G" in cmd
    assert cmd[-2:] == ["gen_gemm_sm120_module", "gen_gemm_sm120_module_cutlass_fp4"]
    assert vllm.prepare_modules("10.0") == (
        "gen_gemm_sm100_module",
        "gen_gemm_sm100_module_cutlass_fp4",
    )
    assert vllm.prepare_modules("10.3")[1] == "gen_gemm_sm103_module_cutlass_fp4"
    with pytest.raises(VLLMConfigError):
        vllm.prepare_modules("8.9")


def test_prepare_builds_once_per_architecture(monkeypatch):
    from giq.gpus import GpuTelemetry

    cards = [
        GpuTelemetry("GPU-a", 0, "RTX PRO 6000", 95.6, 0.0),
        GpuTelemetry("GPU-b", 1, "RTX PRO 6000", 95.6, 0.0),
        GpuTelemetry("GPU-c", 2, "B200", 179.0, 0.0),
    ]
    caps = {"GPU-a": "12.0", "GPU-b": "12.0", "GPU-c": "10.0"}
    monkeypatch.setattr("giq.gpus.get_gpus", lambda *a, **k: cards)
    monkeypatch.setattr(vllm, "compute_capability", lambda uuid: caps[uuid])
    assert vllm.prepare_targets() == {"12.0": "GPU-a", "10.0": "GPU-c"}
    monkeypatch.setattr(vllm, "resolve_device", lambda d: cards[1])
    assert vllm.prepare_targets("1") == {"12.0": "GPU-b"}


@pytest.mark.asyncio
async def test_warm_up_starts_with_time_to_compile_and_always_stops(tmp_path, monkeypatch):
    calls = []

    async def start(self):
        calls.append(("start", self.config.ready_timeout))
        raise TimeoutError("still compiling")

    async def stop(self):
        calls.append(("stop", None))

    monkeypatch.setattr(vllm, "recipe_for", lambda model: make_recipe(make_checkpoint(tmp_path)))
    monkeypatch.setattr(vllm.VllmAdapter, "start", start)
    monkeypatch.setattr(vllm.VllmAdapter, "stop", stop)
    with pytest.raises(TimeoutError):
        await vllm.warm_up(NAME, device="GPU-test")
    assert calls == [("start", vllm.WARMUP_TIMEOUT_SECONDS), ("stop", None)]


def test_a_restart_keeps_the_previous_run_log(tmp_path):
    """giq restarts a crashed resident on its own; the new run must not wipe
    the stack trace that says why the old one died."""
    from giq.adapters.engine import open_engine_log

    log = tmp_path / "vllm-m.log"
    log.write_text("EngineCore: torch.OutOfMemoryError\n")
    with open_engine_log(log) as f:
        f.write("second run\n")
    assert log.read_text() == "second run\n"
    assert (tmp_path / "vllm-m.log.1").read_text() == "EngineCore: torch.OutOfMemoryError\n"

    # An empty log (a run that wrote nothing) does not displace the one kept.
    log.write_text("")
    with open_engine_log(log) as f:
        f.write("third run\n")
    assert (tmp_path / "vllm-m.log.1").read_text() == "EngineCore: torch.OutOfMemoryError\n"
