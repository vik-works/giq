# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Streaming chat completions: tokens reaching the client while the job runs.

Before this, `stream: true` was answered by waiting for the whole job and then
emitting the finished text as a single chunk — a shape that looked like
streaming and delivered none of its benefits. It mattered more than cosmetics:
a long answer has to survive the browser, the proxy and the job timeout, and a
response that arrives all at once at minute six survives none of them.

The rule these tests defend is that streaming must not be a side door around
the scheduler. A streaming job is still a queued job: it waits its turn, it can
trigger an eviction, it is counted. Only the delivery changes.
"""

import asyncio
import json

import httpx
import pytest

from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig
from giq.models import JobRequest, JobStatus, Modality
from giq.queue import STREAM_BUFFER_CHUNKS, Job, JobStream
from giq.registry import get_recipe
from giq.runner import FLOOR_TOKENS_PER_SECOND, JOB_TIMEOUT_SECONDS, _job_timeout


def sse(chunks: list[dict], done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


def delta(**kw) -> dict:
    return {"id": "c1", "created": 1, "model": "m", "choices": [{"index": 0, "delta": kw}]}


def worker_with(response_bytes: bytes, capture: dict | None = None) -> LlamaCppAdapter:
    """An LlamaCppAdapter whose llama-server is a mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.update(json.loads(request.content))
        return httpx.Response(200, content=response_bytes)

    worker = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b"))
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker._ready = True
    return worker


# --- the channel -------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_sentinel_arrives_even_when_the_buffer_is_full():
    """The reader blocks until the sentinel; it must never be the thing dropped.

    A full buffer means a slow reader, which is exactly when the end matters —
    dropping the sentinel instead would hang the request forever.
    """
    stream = JobStream()
    for i in range(STREAM_BUFFER_CHUNKS):
        stream.queue.put_nowait({"i": i})
    assert stream.queue.full()

    await stream.close()

    drained = []
    while not stream.queue.empty():
        drained.append(stream.queue.get_nowait())
    assert drained[-1] is None, "sentinel must be the last thing in the buffer"


@pytest.mark.asyncio
async def test_a_cancelled_stream_drops_chunks_instead_of_blocking():
    """Once the client is gone the worker must not park on a full buffer."""
    stream = JobStream()
    for i in range(STREAM_BUFFER_CHUNKS):
        stream.queue.put_nowait({"i": i})
    stream.cancel()

    await asyncio.wait_for(stream.put({"late": True}), timeout=1.0)


# --- the worker --------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_separates_thinking_from_the_answer():
    """`reasoning_content` is its own delta channel, and must stay separate.

    Concatenating the two would paste the model's scratchpad onto the front of
    its answer, which is worse than showing neither.
    """
    body = sse(
        [
            delta(role="assistant"),
            delta(reasoning_content="let me "),
            delta(reasoning_content="think"),
            delta(content="The answer "),
            delta(content="is 391."),
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5}},
        ]
    )
    worker = worker_with(body)
    stream = JobStream()

    result = await worker.chat_completion_stream({"messages": []}, stream)

    message = result["choices"][0]["message"]
    assert message["content"] == "The answer is 391."
    assert message["reasoning_content"] == "let me think"
    assert result["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_every_chunk_reaches_the_reader_as_it_arrives():
    body = sse([delta(content="a"), delta(content="b"), delta(content="c")])
    worker = worker_with(body)
    stream = JobStream()

    await worker.chat_completion_stream({"messages": []}, stream)

    got = []
    while not stream.queue.empty():
        got.append(stream.queue.get_nowait())
    contents = [c["choices"][0]["delta"].get("content") for c in got if c]
    assert contents == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_usage_is_requested_or_streamed_answers_go_uncounted():
    """llama-server omits usage from a stream unless asked, and the stats read
    it off the stored result — without this every streamed job records zero
    tokens and the accounting silently drifts."""
    capture: dict = {}
    worker = worker_with(sse([delta(content="hi")]), capture)

    await worker.chat_completion_stream({"messages": []}, JobStream())

    assert capture["stream"] is True
    assert capture["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_cancelling_stops_the_generation_mid_answer():
    """A client that goes away must not leave the model generating for minutes."""
    stream = JobStream()
    stream.cancel()
    worker = worker_with(sse([delta(content=str(i)) for i in range(50)]))

    result = await worker.chat_completion_stream({"messages": []}, stream)

    assert result["choices"][0]["finish_reason"] == "cancelled"
    assert result["choices"][0]["message"]["content"] == ""


@pytest.mark.asyncio
async def test_a_server_error_surfaces_rather_than_hiding_as_an_empty_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"model exploded")

    worker = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b"))
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker._ready = True

    with pytest.raises(httpx.HTTPStatusError):
        await worker.chat_completion_stream({"messages": []}, JobStream())


# --- the timeout -------------------------------------------------------------


def _job(max_tokens: int | None) -> Job:
    chat = {"messages": []}
    if max_tokens is not None:
        chat["max_tokens"] = max_tokens
    return Job(
        job_id="j1",
        request=JobRequest(modality=Modality.llm, model="qwen3.8-27b", chat_request=chat),
    )


def test_a_big_token_budget_gets_the_time_to_spend_it():
    """Measured on an RTX 5090: real reasoning prompts ran 343s and 356s, past the
    flat 300s ceiling, while generating perfectly good output."""
    assert _job_timeout(Modality.llm, _job(32768)) == pytest.approx(32768 / FLOOR_TOKENS_PER_SECOND)


def test_a_small_budget_still_gets_the_old_floor():
    """Scaling down would make short jobs *more* fragile than before."""
    assert _job_timeout(Modality.llm, _job(512)) == JOB_TIMEOUT_SECONDS


def test_no_budget_gets_the_models_whole_context():
    """Omitting max_tokens means "up to the context", so the ceiling has to
    follow the context — otherwise not asking for a budget would give a long
    answer *less* time than asking for one."""
    from giq.adapters.llama_cpp import MODEL_CTX_SIZE

    ctx = MODEL_CTX_SIZE["qwen3.8-27b"]
    assert _job_timeout(Modality.llm, _job(None)) == pytest.approx(ctx / FLOOR_TOKENS_PER_SECOND)


def test_an_explicit_budget_still_wins_over_the_context():
    """A caller asking for less gets less time, not the full-context ceiling."""
    assert _job_timeout(Modality.llm, _job(4096)) == pytest.approx(
        max(JOB_TIMEOUT_SECONDS, 4096 / FLOOR_TOKENS_PER_SECOND)
    )


def test_timeout_without_a_job_is_unchanged():
    assert _job_timeout(Modality.llm) == JOB_TIMEOUT_SECONDS


def test_audio_keeps_its_own_override():
    job = Job(job_id="j2", request=JobRequest(modality=Modality.audio, model="whisperx"))
    assert _job_timeout(Modality.audio, job) == 870.0


# --- the endpoint relay ------------------------------------------------------


class FakeQueue:
    def __init__(self, job):
        self._job = job

    async def get(self, job_id):
        return self._job


class FakeOrch:
    def __init__(self, job):
        self.queue = FakeQueue(job)
        self.cancelled = []

    async def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        return True, "Cancelled"


async def collect(orch, job_id, stream):
    from giq.api.openai_compat import _relay_stream

    return [part async for part in _relay_stream(orch, job_id, stream, "qwen3.8-27b")]


@pytest.mark.asyncio
async def test_relay_emits_chunks_then_done():
    job = _job(512)
    job.status = JobStatus.completed
    stream = JobStream()
    stream.queue.put_nowait(delta(content="hello"))
    await stream.close()

    parts = await collect(FakeOrch(job), "j1", stream)

    assert "hello" in parts[0]
    assert parts[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_a_failure_after_the_first_chunk_is_reported_in_the_stream():
    """Headers are already sent, so a failure cannot become an HTTP status. If
    it did not ride the stream the client would see a clean truncation and
    treat half an answer as the whole one."""
    job = _job(512)
    job.status = JobStatus.failed
    job.error = "Job timed out after 300s"
    stream = JobStream()
    stream.queue.put_nowait(delta(content="partial"))
    await stream.close()

    parts = await collect(FakeOrch(job), "j1", stream)

    assert any("Job timed out" in p for p in parts)
    assert parts[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_relay_cancels_the_job_when_the_client_walks_away():
    """FastAPI closes the generator on disconnect; that has to reach the GPU."""
    job = _job(512)
    job.status = JobStatus.running
    stream = JobStream()
    stream.queue.put_nowait(delta(content="one"))
    orch = FakeOrch(job)

    from giq.api.openai_compat import _relay_stream

    gen = _relay_stream(orch, "j1", stream, "qwen3.8-27b")
    await gen.__anext__()
    await gen.aclose()

    assert stream.is_cancelled
    assert orch.cancelled == ["j1"]


@pytest.mark.asyncio
async def test_a_queued_job_sends_keepalives_instead_of_going_silent(monkeypatch):
    """Nothing is generated while a job waits its turn or a 26GB model loads.
    Browsers and reverse proxies drop a connection that goes quiet, so the
    silence has to be filled or long waits die before the first token."""
    from giq.api import openai_compat

    monkeypatch.setattr(openai_compat, "SSE_KEEPALIVE_SECONDS", 0.01)

    job = _job(512)
    job.status = JobStatus.completed
    stream = JobStream()

    async def late_answer():
        await asyncio.sleep(0.05)
        await stream.put(delta(content="finally"))
        await stream.close()

    task = asyncio.create_task(late_answer())
    parts = await collect(FakeOrch(job), "j1", stream)
    await task

    assert ": keepalive\n\n" in parts
    assert any("finally" in p for p in parts)


# --- the thinking ceiling ----------------------------------------------------


def test_thinking_is_unrestricted_unless_a_model_asks_for_a_deadline():
    """No blanket cap. A launch flag has to serve every request size at once,
    and the same ceiling that rescues a thought going nowhere truncates one
    that was getting somewhere — so it is opt-in per model, not a default."""
    from giq.adapters.llama_cpp import MODEL_REASONING_BUDGET

    assert MODEL_REASONING_BUDGET == {}
    assert LlamaCppConfig(model="qwen3.8-27b").reasoning_budget is None

    cmd = LlamaCppAdapter(
        LlamaCppConfig(model="qwen3.8-27b", model_path="/tmp/x.gguf")
    ).build_command()
    assert "--reasoning-budget" not in cmd


def test_a_declared_budget_reaches_the_command_line(monkeypatch):
    """The launch flag still works, and is still the floor for a model that
    should never think without one. (The claim that used to be here — that a
    per-request budget is "silently ignored, launch flag only" — was wrong
    about the cause: the body field is `reasoning_budget_tokens`, and it is
    used now. See test_the_reasoning_budget_reaches_the_body.) Live proof the
    flag works: same prompt and model, unbudgeted
    gave 4,993 chars of thought and an empty answer at finish_reason=length;
    with --reasoning-budget 200 it gave ~920 chars of thought and a real
    1,105-char answer at finish_reason=stop."""
    from giq.adapters import llama_cpp

    monkeypatch.setitem(llama_cpp.MODEL_REASONING_BUDGET, "qwen3.8-27b", 24576)

    cmd = LlamaCppAdapter(
        LlamaCppConfig(model="qwen3.8-27b", model_path="/tmp/x.gguf")
    ).build_command()

    assert cmd[cmd.index("--reasoning-budget") + 1] == "24576"
    # A budget means nothing unless reasoning is actually on.
    assert cmd[cmd.index("--reasoning") + 1] == "on"


# --- repetition control ------------------------------------------------------
#
# Before this, giq sent three sampler fields and no others, so llama.cpp's
# defaults decided the rest — and two of those defaults mean "off":
# repeat_penalty 1.00 and dry_multiplier 0.00. There was no repetition control
# anywhere in the stack.


def test_dry_is_configured_for_the_models_that_looped():
    """DRY rather than repeat_penalty: the latter penalises tokens wherever they
    appear, which taxes every "the" in ordinary prose. DRY penalises only the
    continuation of a run already seen, which is the observed failure exactly."""
    defaults = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b")).request_defaults()

    assert defaults["dry_multiplier"] > 0
    assert defaults["dry_allowed_length"] > 2, "2 penalises any repeated pair, which is normal"


def test_dry_penalty_last_n_is_a_real_number_and_not_minus_one():
    """The trap. Elsewhere -1 means "the whole context"; in this build
    llama-sampler.cpp:3640 clamps it with max(n, 0) and line 3645 treats 0 as
    disabled, so -1 silently switches DRY back off. The default of 64 is no
    better here — it is shorter than one of the repeating fragments."""
    for model in ("qwen3.8-27b",):
        last_n = LlamaCppAdapter(LlamaCppConfig(model=model)).request_defaults()[
            "dry_penalty_last_n"
        ]

        assert last_n > 64, f"{model}: must outrun the repeating fragment"
        assert last_n > 0, f"{model}: -1 or 0 disables DRY in this build"


def test_the_reasoning_budget_reaches_the_body():
    """The per-request form of the deadline, and the only one that reaches the
    non-streaming paths — they await a finished response, so no in-flight
    detector can help them. Zero would mean "skip thinking entirely"."""
    defaults = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b")).request_defaults()

    assert defaults["reasoning_budget_tokens"] > 0


@pytest.mark.asyncio
async def test_model_defaults_reach_llama_server_but_lose_to_the_caller():
    """Defaults, not overrides. A client that has tuned its own sampling must
    not have giq quietly overrule it."""
    capture: dict = {}
    worker = worker_with(sse([delta(content="hi")]), capture)

    await worker.chat_completion_stream({"messages": [], "dry_multiplier": 0.0}, JobStream())

    assert capture["dry_penalty_last_n"] > 0, "giq's default should be there"
    assert capture["dry_multiplier"] == 0.0, "the caller's value should win"


# --- the loop guard ----------------------------------------------------------


def looping_reasoning_sse(fragments: int = 60) -> bytes:
    """A stream whose thinking channel loops, then answers."""
    from tests.test_loopguard import looping_thought

    thought = looping_thought(fragments)
    chunks = [delta(reasoning_content=thought[i : i + 40]) for i in range(0, len(thought), 40)]
    chunks.append(delta(content="Here is the answer."))
    return sse(chunks)


def guarded_worker(response_bytes: bytes, control_result: dict | None = None):
    """A worker whose llama-server answers the completion and the control call,
    recording every control request it received."""
    control_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/control"):
            control_calls.append(json.loads(request.content))
            return httpx.Response(200, json=control_result or {"success": True})
        return httpx.Response(200, content=response_bytes)

    worker = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b"))
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker._ready = True
    return worker, control_calls


@pytest.mark.asyncio
async def test_a_streamed_completion_is_armed_for_control():
    """reasoning_control has to be on the original request or the control call
    is refused — and arming costs nothing, because with no budget set the
    sampler counts to INT_MAX and never fires by itself."""
    capture: dict = {}
    worker = worker_with(sse([delta(content="hi")]), capture)

    await worker.chat_completion_stream({"messages": []}, JobStream())

    assert capture["reasoning_control"] is True
    assert capture["reasoning_budget_message"], "the model should be told why it was stopped"


@pytest.mark.asyncio
async def test_a_looping_thought_is_ended_through_the_control_endpoint():
    """The whole point: act on the live slot mid-generation, so the model writes
    its answer from the thinking it has already done. No second request, no
    re-prefill, nothing in the KV cache thrown away."""
    worker, control_calls = guarded_worker(looping_reasoning_sse())

    result = await worker.chat_completion_stream({"messages": []}, JobStream())

    assert control_calls == [{"id": "c1", "action": "reasoning_end"}]
    assert result["giq_loop_guard"]["intervened"] is True
    assert result["choices"][0]["message"]["content"] == "Here is the answer."


async def with_reader(worker: LlamaCppAdapter, body: dict) -> dict:
    """Run a stream with something draining it, the way the endpoint does. A
    thought long enough to be worth testing is longer than the 512-chunk buffer,
    and an unread buffer is supposed to park the worker — that is the stall
    protection working, not a thing to test around."""
    stream = JobStream()

    async def drain():
        while await stream.queue.get() is not None:
            pass

    reader = asyncio.create_task(drain())
    try:
        return await worker.chat_completion_stream(body, stream)
    finally:
        reader.cancel()


@pytest.mark.asyncio
async def test_the_thought_is_only_ended_once():
    """The guard latches, so it keeps reporting a loop on every later delta.
    Acting on each of them would be hundreds of control calls."""
    worker, control_calls = guarded_worker(looping_reasoning_sse(fragments=200))

    await with_reader(worker, {"messages": []})

    assert len(control_calls) == 1


@pytest.mark.asyncio
async def test_a_refused_control_call_does_not_destroy_the_answer():
    """A guard that cannot act has still only offered an opinion. Tearing the
    generation down over it would turn a recoverable answer into no answer —
    which is the failure the guard exists to prevent, arrived at differently."""
    worker, control_calls = guarded_worker(
        looping_reasoning_sse(),
        control_result={"success": False, "message": "reasoning control not enabled"},
    )

    result = await worker.chat_completion_stream({"messages": []}, JobStream())

    assert control_calls, "it should still have tried"
    assert result["giq_loop_guard"]["intervened"] is False
    assert result["choices"][0]["message"]["content"] == "Here is the answer."
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_a_control_endpoint_that_errors_is_survivable():
    """Best-effort on top of a generation already in trouble: failing to improve
    it must not also destroy it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/control"):
            return httpx.Response(500, content=b"boom")
        return httpx.Response(200, content=looping_reasoning_sse())

    worker = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b"))
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker._ready = True

    result = await worker.chat_completion_stream({"messages": []}, JobStream())

    assert result["choices"][0]["message"]["content"] == "Here is the answer."
    assert result["giq_loop_guard"]["intervened"] is False


@pytest.mark.asyncio
async def test_finish_reason_stays_the_servers_own_word():
    """After a successful intervention the response really did end normally and
    carries a real answer. Renaming that to make one event easier to count would
    be a lie told to every consumer of the stats."""
    worker, _ = guarded_worker(looping_reasoning_sse())

    result = await worker.chat_completion_stream({"messages": []}, JobStream())

    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["giq_loop_guard"]["thought_chars"] > 0


@pytest.mark.asyncio
async def test_an_ordinary_answer_carries_no_loop_record():
    worker, control_calls = guarded_worker(sse([delta(content="short and done")]))

    result = await worker.chat_completion_stream({"messages": []}, JobStream())

    assert "giq_loop_guard" not in result
    assert control_calls == []


@pytest.mark.asyncio
async def test_the_guard_can_be_switched_off_per_request():
    """For a caller that wants the whole thought however long it circles. The
    flag is giq's, not llama-server's, so it must not be forwarded — the server
    copies unrecognised keys into its own params."""
    capture: dict = {}
    control_calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/control"):
            control_calls.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True})
        capture.update(json.loads(request.content))
        return httpx.Response(200, content=looping_reasoning_sse())

    worker = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b"))
    worker._client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker._ready = True

    result = await worker.chat_completion_stream(
        {"messages": [], "giq_loop_guard": False}, JobStream()
    )

    assert control_calls == []
    assert "giq_loop_guard" not in capture, "giq's own field must not reach llama-server"
    assert "reasoning_control" not in capture, "nothing to arm if nothing will act"
    assert "giq_loop_guard" not in result


@pytest.mark.asyncio
async def test_the_callers_request_body_is_not_mutated():
    """The body belongs to the job and is read again by the stats layer; popping
    giq's own key out of the caller's dict would edit it under them."""
    body = {"messages": [], "giq_loop_guard": False}
    worker, _ = guarded_worker(sse([delta(content="hi")]))

    await worker.chat_completion_stream(body, JobStream())

    assert body == {"messages": [], "giq_loop_guard": False}


# --- context and the timeouts that have to follow it -------------------------


def test_qwen38_runs_the_context_that_was_measured_to_fit():
    """128k, not the native 256k. qwen35 is an SSM hybrid: 65 layers with
    full_attention_interval=4, so only 16 hold a KV cache at 34 KiB/token (q8_0)
    and the other 49 cost a fixed ~150 MiB. 256k of q8_0 KV is 8.5 GiB, which
    does not fit beside 20.5 GiB of weights on a 31.8 GiB card."""
    from giq.adapters.llama_cpp import MODEL_CTX_SIZE

    assert MODEL_CTX_SIZE["qwen3.8-27b"] == 131072


def test_k_and_v_cache_types_match_or_the_fast_kernel_is_lost():
    """Benchmarked on head_dim 256: q8_0/q8_0 and q4_0/q4_0 both run
    at ~2949 tok/s prefill, while every mixed pair falls off the fused attention
    path -- q8_0/q4_0 to 235 tok/s, q8_0/iq4_nl to 76. So the usual "keep K
    precise, economise on V" is not available on this model at any context, and
    a mixed pair here would be a 12x prefill regression, not a saving."""
    from giq.adapters.llama_cpp import DEFAULT_CACHE_TYPE_K as DK
    from giq.adapters.llama_cpp import DEFAULT_CACHE_TYPE_V as DV
    from giq.adapters.llama_cpp import MODEL_CACHE_TYPE_K, MODEL_CACHE_TYPE_V

    for model in set(MODEL_CACHE_TYPE_K) | set(MODEL_CACHE_TYPE_V):
        k = MODEL_CACHE_TYPE_K.get(model, DK)
        v = MODEL_CACHE_TYPE_V.get(model, DV)
        assert k == v, f"{model}: mixed KV cache types ({k}/{v}) lose the fused kernel"
    assert DK == DV


def test_the_http_read_budget_follows_the_context():
    """A flat 120s was survivable at 8k of context and fatal at 192k. httpx's
    timeout is per-read: for a stream it is the gap between chunks, but the
    first chunk only exists once the prompt is processed, and prefill measured
    ~1400 tok/s at 32k and falls from there. A full-context prompt needs minutes
    before its first token."""
    big = LlamaCppAdapter(LlamaCppConfig(model="qwen3.8-27b")).http_timeout()
    small = LlamaCppAdapter(LlamaCppConfig(model="gemma-4-12b", ctx_size=8192)).http_timeout()

    assert big.read > 120.0
    assert big.read > small.read, "a bigger context must get a longer budget"
    assert small.connect == 10.0, "connecting to loopback stays short"


def test_the_job_timeout_still_fires_before_the_http_one():
    """Ordering is the point. runner._job_timeout cancels *with cleanup* -- it
    unloads a worker that may be wedged -- while an httpx read timeout would
    pre-empt that with a bare exception and leave the model holding VRAM."""
    from giq.models import JobRequest, Modality
    from giq.queue import Job

    model = "qwen3.8-27b"
    job = Job(job_id="j", request=JobRequest(modality="llm", model=model, chat_request={}))
    job_limit = _job_timeout(Modality.llm, job)
    http_limit = LlamaCppAdapter(LlamaCppConfig(model=model)).http_timeout().read

    assert job_limit < http_limit, "the runner must give up first, with cleanup"


# --- two profiles over one file ----------------------------------------------
#
# A vision model with an MTP head can be served as two registry entries over
# one GGUF: the vision entry keeps mmproj, the `-fast` entry drops it and runs
# --spec-type draft-mtp (see MODEL_SPEC_TYPE in adapters/llama_cpp.py for the
# measurement behind it). No built-in entry uses the pattern, so these tests
# exercise it through a pair injected into the registry and llm tables, and the
# invariants also run over the real tables so a pair added later is held to
# them.


@pytest.fixture
def mtp_pair(monkeypatch):
    """A vision profile and its text-only speculative sibling, one GGUF."""
    from giq.adapters import llama_cpp
    from tests._recipes import with_recipes

    base = get_recipe("qwen3.8-27b")
    assert base is not None
    vision, fast = "pair-vision", "pair-vision-fast"
    with_recipes(
        monkeypatch,
        base.model_copy(update={"name": vision, "label": ""}),
        base.model_copy(
            update={
                "name": fast,
                "label": "",
                "capabilities": tuple(c for c in base.capabilities if c != "vision"),
                "weights": base.weights.model_copy(update={"parts": {}}),
            }
        ),
    )
    for name in (vision, fast):
        monkeypatch.setitem(llama_cpp.MODEL_PATHS, name, "pair/model-Q6_K.gguf")
        monkeypatch.setitem(llama_cpp.MODEL_CTX_SIZE, name, 196608)
        monkeypatch.setitem(llama_cpp.MODEL_CACHE_TYPE_K, name, "q8_0")
        monkeypatch.setitem(llama_cpp.MODEL_CACHE_TYPE_V, name, "q8_0")
    monkeypatch.setitem(llama_cpp.MODEL_SPEC_TYPE, fast, "draft-mtp")
    return vision, fast


def _profile_pairs() -> list[tuple[str, str]]:
    """(vision, fast): a speculative entry and a plain one over the same GGUF."""
    from giq.adapters.llama_cpp import MODEL_PATHS, MODEL_SPEC_TYPE

    return [
        (other, fast)
        for fast in MODEL_SPEC_TYPE
        for other, path in MODEL_PATHS.items()
        if other != fast and other not in MODEL_SPEC_TYPE and path == MODEL_PATHS.get(fast)
    ]


def _profile_drift(vision: str, fast: str) -> list[str]:
    """What differs between two profiles besides vision and speculation."""
    from giq.adapters import llama_cpp

    tables = {
        "ctx": (llama_cpp.MODEL_CTX_SIZE, llama_cpp.DEFAULT_CTX_SIZE),
        "cache_k": (llama_cpp.MODEL_CACHE_TYPE_K, llama_cpp.DEFAULT_CACHE_TYPE_K),
        "cache_v": (llama_cpp.MODEL_CACHE_TYPE_V, llama_cpp.DEFAULT_CACHE_TYPE_V),
    }
    return [
        name
        for name, (table, default) in tables.items()
        if table.get(vision, default) != table.get(fast, default)
    ]


def test_profiles_over_one_file_differ_only_in_vision_and_speculation():
    """Anything else drifting apart makes them two models rather than two
    profiles -- and the fast one must be text-only, or it loses the speedup it
    exists for."""
    for vision, fast in _profile_pairs():
        assert _profile_drift(vision, fast) == [], f"{vision} / {fast} drifted apart"
        assert not get_recipe(fast).mmproj, f"{fast} must be text-only"


def test_the_pair_is_recognised_and_consistent(mtp_pair):
    assert mtp_pair in _profile_pairs()
    assert _profile_drift(*mtp_pair) == []


def test_a_profile_that_drifts_is_caught(mtp_pair, monkeypatch):
    from giq.adapters import llama_cpp

    vision, fast = mtp_pair
    monkeypatch.setitem(llama_cpp.MODEL_CTX_SIZE, fast, 131072)

    assert _profile_drift(vision, fast) == ["ctx"]


def test_the_speculative_flag_reaches_the_command_line(mtp_pair):
    """It only exists if it is in the argv -- and it changes what the model
    does per token, not just how fast, so it is asserted like --reasoning is."""
    vision_name, fast_name = mtp_pair
    fast = LlamaCppAdapter(LlamaCppConfig(model=fast_name)).build_command()
    vision = LlamaCppAdapter(LlamaCppConfig(model=vision_name)).build_command()

    assert fast[fast.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-type" not in vision
    assert "--mmproj" not in fast, "the fast profile is text-only by construction"
    assert "--mmproj" in vision
    assert fast[fast.index("-m") + 1] == vision[vision.index("-m") + 1], "one file on disk"


# Pairing vision with --spec-type used to be forbidden outright. The rule came
# from a measurement on Qwen3.8-27B, Q6_K, at 196608 ctx: draft-mtp measured
# 132.7 tok/s without the projector and 75.3 with it, so a profile carrying both
# quietly lost half the gain for 256 MiB of savings it did not need.
#
# Re-measured at 98304 ctx on a newer llama.cpp build, as a direct A/B -- two
# registry entries identical but for mmproj, same prompt, same context,
# temperature 0, 1200 completion tokens counted from the usage block rather
# than estimated from character counts:
#
#   with mmproj      119.4, 117.9, 120.5 tok/s
#   without mmproj   117.9, 117.0, 115.5, 118.2 tok/s
#
# The penalty is gone -- the two are equal within run-to-run noise. Whether the
# earlier rows were confounded or the rebuild fixed it is not established,
# and the A/B was at 96k, not 192k, so the earlier rows are NOT hereby declared
# wrong. What is established is that at 96k on that build the conflict does not
# reproduce, which is enough to let one profile carry both.
#
# So the invariant is "not by accident" rather than "never": a model that pairs
# them must be listed here, with the measurement that justifies it.
SPEC_TYPE_WITH_VISION_ALLOWED: set[str] = set()


def _unexempted_vision_speedup(allowed: set[str]) -> list[str]:
    """Models that pair mmproj with --spec-type without being allowlisted."""
    from giq.adapters.llama_cpp import MODEL_SPEC_TYPE

    offenders = []
    for model in MODEL_SPEC_TYPE:
        spec = get_recipe(model)
        assert spec is not None, f"{model} runs with --spec-type but has no recipe"
        if model not in allowed and (spec.mmproj or spec.vision):
            offenders.append(model)
    return offenders


def test_vision_and_the_speedup_are_not_paired_by_accident():
    """A profile may carry both only if it is listed above with its evidence.
    Re-measure the two against each other and add it with the numbers, or drop
    one of them."""
    assert _unexempted_vision_speedup(SPEC_TYPE_WITH_VISION_ALLOWED) == []


def test_an_unlisted_vision_profile_with_the_speedup_is_caught(mtp_pair, monkeypatch):
    from giq.adapters import llama_cpp

    vision, _ = mtp_pair
    monkeypatch.setitem(llama_cpp.MODEL_SPEC_TYPE, vision, "draft-mtp")

    assert _unexempted_vision_speedup(set()) == [vision]
    assert _unexempted_vision_speedup({vision}) == []


def test_the_vision_speedup_exception_list_is_honest():
    """Every name on the allowlist must actually pair them. A stale entry would
    silently re-open the hole the test above exists to close."""
    from giq.adapters.llama_cpp import MODEL_SPEC_TYPE

    for model in SPEC_TYPE_WITH_VISION_ALLOWED:
        assert model in MODEL_SPEC_TYPE, f"{model} is exempted but runs no --spec-type"
        spec = get_recipe(model)
        assert spec is not None and spec.mmproj, (
            f"{model} is exempted but carries no mmproj -- remove it from the list"
        )
