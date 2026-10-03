# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

# OpenAI-compatible API router - consolidated
import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from giq.api.dependencies import get_orchestrator
from giq.models import JobRequest
from giq.recipes.schema import Recipe
from giq.registry import get_recipe, recipes_serving, resident_defaults
from giq.services.orchestration import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None
    name: str | None = None


def has_non_text_parts(content: str | list | None) -> bool:
    """True when a message carries something other than text — an image, say.

    `extract_text_content` keeps type == "text" blocks and drops the rest, so
    before this check an image_url part was discarded in silence and the model
    answered about nothing at all.
    """
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") not in (None, "text") for block in content
    )


def is_multimodal(messages: list) -> bool:
    return any(has_non_text_parts(m.content) for m in messages)


def format_messages(messages: list) -> list[dict]:
    """The message list as the engine wants it, kept whole: every turn, the
    tool results and any image parts."""
    formatted: list[dict] = []
    for m in messages:
        msg: dict = {"role": m.role}
        content_text = extract_text_content(m.content)
        if m.role == "assistant" and m.tool_calls:
            msg["content"] = content_text or None
            msg["tool_calls"] = m.tool_calls
        elif m.role == "tool":
            msg["content"] = content_text
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
        else:
            # Verbatim when it carries images: llama-server speaks the
            # OpenAI content-part shape natively once --mmproj is loaded.
            msg["content"] = m.content if has_non_text_parts(m.content) else content_text
        formatted.append(msg)
    return formatted


# How long the relay waits with nothing to send before emitting an SSE comment.
# A queued job or a cold model load produces no tokens for tens of seconds, and
# browsers and reverse proxies drop a connection that goes quiet. The comment
# is protocol-legal filler that every SSE client ignores.
SSE_KEEPALIVE_SECONDS = 10.0


async def _relay_stream(orch: Orchestrator, job_id: str, stream, model: str):
    """Turn a running job's chunks into an SSE response body.

    The job is an ordinary queued job — it waited its turn, it may have
    triggered an eviction, it is counted in the stats. This only changes when
    the caller sees the output: as it is produced, rather than at the end.
    """
    from giq.models import JobStatus

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(stream.queue.get(), timeout=SSE_KEEPALIVE_SECONDS)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if chunk is None:
                break
            yield f"data: {json.dumps(chunk)}\n\n"

        # The sentinel says the job ended, not that it succeeded. A failure
        # after headers are sent cannot become an HTTP status, so it has to
        # ride the stream or the client would see a clean, silent truncation.
        job = await orch.queue.get(job_id)
        if job is not None and job.status == JobStatus.failed:
            error = {"error": {"message": job.error or "Job failed", "type": "giq_job_failed"}}
            yield f"data: {json.dumps(error)}\n\n"

        yield "data: [DONE]\n\n"
    finally:
        # Client gone, or we are done: either way stop generating. Draining
        # what is left unblocks a worker parked on a full buffer instead of
        # leaving it to time out.
        stream.cancel()
        while True:
            try:
                stream.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await orch.cancel_job(job_id)


def extract_text_content(content: str | list | None) -> str:
    """Extract text from message content (handles string or array of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
        return "\n".join(texts)
    return str(content)


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    model: str = Field(..., description="Model to use")
    messages: list[ChatMessage] = Field(..., description="Messages")
    # Unset means "as much as the model's context allows", which is what
    # OpenAI means by omitting it and what llama.cpp does with no n_predict.
    # It used to default to 2048 — a ceiling on thinking *and* answer, so on a
    # reasoning model the thought ate the budget and the answer came back
    # empty with finish_reason "length". 2048 was 1.5% of qwen3.8-27b's 131k
    # context. A caller that wants a smaller ceiling still says so.
    #
    # `max_completion_tokens` is the current OpenAI spelling (`max_tokens` is
    # deprecated) and is what LibreChat and the modern SDKs send. With
    # extra="ignore" it was being dropped in silence and the 2048 applied
    # anyway, so a client asking for a big budget got a small one.
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
        description="Max tokens to generate; unset = up to the model's context",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Stream response")
    tools: list[dict] | None = Field(default=None, description="Tool definitions")
    tool_choice: str | dict | None = Field(default=None, description="Tool choice mode")
    # Structured output. Without these, extra="ignore" dropped them in silence
    # and the engine only ever saw a free-form request, so a client asking for
    # schema-constrained JSON got best-effort prose. Whichever one is set is
    # forwarded to the engine verbatim in every request branch below (the
    # workers relay the body as-is); setting both is a 400, see the validator:
    # * `response_format` — the OpenAI-native spec, {"type":"json_object"} or
    #   {"type":"json_schema","json_schema":{...}}. Honoured by both llama.cpp
    #   (GBNF) and vllm.
    # * `structured_outputs` — vllm's native constrained-decoding knob
    #   ({"json": <schema>} / regex / choice / grammar; xgrammar/guidance).
    #   vllm enforces it at decode time; llama-server ignores the unknown key.
    response_format: dict | None = Field(
        default=None, description="OpenAI structured-output spec (json_object/json_schema)"
    )
    structured_outputs: dict | None = Field(
        default=None, description="vllm native structured outputs (json/regex/choice/grammar)"
    )

    @model_validator(mode="after")
    def _one_structured_output_knob(self) -> "ChatCompletionRequest":
        """Refuse both structured-output knobs at once.

        vllm folds `response_format` into its own structured-output constraints
        and then rejects the merged set as mutually exclusive, so forwarding
        both turns a plausible-looking request into an engine-side 4xx the
        caller cannot read. Saying so here keeps the failure at the API edge
        and keeps the two knobs independent everywhere else.
        """
        if self.response_format is not None and self.structured_outputs is not None:
            raise ValueError(
                "response_format and structured_outputs are mutually exclusive: "
                "vllm merges them into one constraint set and rejects the result. "
                "Send response_format for portable requests, structured_outputs "
                "for vllm-native regex/choice/grammar."
            )
        return self


def _apply_structured_output(target: dict, request: "ChatCompletionRequest") -> None:
    """Copy structured-output fields onto an engine request, if set.

    The one place the two knobs are threaded through, called from each of the
    three request branches (streaming, tool/multimodal, plain) so a client gets
    the same enforcement regardless of which path its request takes. The
    workers relay the body to the engine verbatim, so a key placed here reaches
    llama-server / vllm unchanged.
    """
    if request.response_format is not None:
        target["response_format"] = request.response_format
    if request.structured_outputs is not None:
        target["structured_outputs"] = request.structured_outputs


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    raw_request: Request, orch: Orchestrator = Depends(get_orchestrator)
) -> dict | StreamingResponse:
    """Generate chat completion (OpenAI-compatible endpoint)."""
    body = await raw_request.json()
    try:
        request = ChatCompletionRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model = request.model
    messages = request.messages

    if "/" in model:
        model = model.split("/", 1)[1]

    # Images reach the engine only on a recipe that can see them; anywhere else
    # they would be dropped without a word.
    if is_multimodal(messages):
        recipe = get_recipe(model)
        if recipe is None or not recipe.vision:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{model} has no vision support, and the images in this request would "
                    "be dropped silently. Use a model whose catalog entry says vision."
                ),
            )
    # Streaming: one path for every kind of request, because it needs the same
    # thing the tool path needs — the message list intact — and because a
    # streamed answer is the only way a long one arrives at all. A nine-minute
    # thought exceeds the browser, proxy and job timeouts a single response has
    # to survive; chunks reset all three.
    if request.stream:
        stream_request: dict = {
            "messages": format_messages(messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens is not None:
            stream_request["max_tokens"] = request.max_tokens
        if request.tools:
            stream_request["tools"] = request.tools
            if request.tool_choice is not None:
                stream_request["tool_choice"] = request.tool_choice
        ctk = body.get("chat_template_kwargs")
        if ctk:
            stream_request["chat_template_kwargs"] = ctk
        _apply_structured_output(stream_request, request)
        # Deadline on the *thought*, separate from max_tokens' ceiling on the
        # whole response; llama.cpp closes the thinking block when it is spent
        # rather than truncating, so the answer is written from the reasoning
        # already done. MODEL_REQUEST_DEFAULTS carries a per-model default and
        # this overrides it for one request — which is the point: an agent loop
        # wants a short leash per tool call, and the same client wants none on
        # a hard question. `thinking_budget_tokens` is llama.cpp's own alias.
        for key in ("reasoning_budget_tokens", "thinking_budget_tokens"):
            if isinstance(body.get(key), int):
                stream_request["reasoning_budget_tokens"] = body[key]
                break
        # The loop guard is on by default; a caller that wants the whole thought
        # however long it circles says so here. The worker pops it — it is
        # giq's field, not llama-server's.
        if isinstance(body.get("giq_loop_guard"), bool):
            stream_request["giq_loop_guard"] = body["giq_loop_guard"]

        job_id, _, stream = await orch.submit_streaming_job(
            JobRequest(modality="llm", model=model, chat_request=stream_request)
        )
        return StreamingResponse(
            _relay_stream(orch, job_id, stream, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # nginx buffers proxied responses by default, which would hold
                # every chunk until the answer finished and undo the point.
                "X-Accel-Buffering": "no",
            },
        )

    # Not streaming: the request goes to the engine whole, as streaming does,
    # and its answer comes back as the engine wrote it. That is what keeps a
    # conversation's earlier turns, the thought (`reasoning_content`) and the
    # engine's finish_reason — "length" when max_tokens ran out mid-thought,
    # which a client cannot tell from a finished answer if it reads "stop".
    formatted_messages = format_messages(messages)

    llm_request: dict = {
        "messages": formatted_messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
    }
    if request.tools:
        llm_request["tools"] = request.tools
    if request.max_tokens is not None:
        llm_request["max_tokens"] = request.max_tokens
    if request.tool_choice is not None:
        llm_request["tool_choice"] = request.tool_choice
    # Parity with the streaming branch above. These were honoured when
    # streaming and silently dropped when not, which is the worst shape for
    # a knob to have: a client that turns thinking off gets it turned off
    # or not depending on a flag it set for an unrelated reason.
    ctk = body.get("chat_template_kwargs")
    if ctk:
        llm_request["chat_template_kwargs"] = ctk
    _apply_structured_output(llm_request, request)
    for key in ("reasoning_budget_tokens", "thinking_budget_tokens"):
        if isinstance(body.get(key), int):
            llm_request["reasoning_budget_tokens"] = body[key]
            break
    # giq_loop_guard is deliberately NOT forwarded here. It is giq's own
    # field, popped by the worker's streaming loop; the non-streaming
    # worker forwards its request body to llama-server verbatim, so an
    # unknown key would reach llama-server instead of being consumed.

    job_id, _ = await orch.submit_job(
        JobRequest(modality="llm", model=model, chat_request=llm_request)
    )
    completed_job = await orch.wait_for_job(job_id)

    if not completed_job.results:
        raise HTTPException(status_code=500, detail="No response generated")

    result = completed_job.results[0]
    # The engine names the model by whatever it was started as; the client
    # asked for a recipe, and the answer says which.
    if isinstance(result, dict):
        result["model"] = request.model
    return result


def _advertised_llm_recipes() -> list[Recipe]:
    """The LLM recipes an OpenAI client should be offered, in display order.

    To a client a recipe is a model: its name is the `id` a client sends
    back as `model` (ADR-003).

    Installed is the only filter: if the weights are on disk, the model is
    offered. No audit gate, no staging step, no opinion about how good or how
    aligned a model is — this is a homelab, and a model that is here is a
    model you can pick. The one thing it will not do is advertise a model
    whose GGUF is missing, which is not curation but honesty: five registered
    models had had their weights deleted and were being offered anyway.

    Residents first in reload-priority order, then everything else by name,
    so a client that treats `data[0]` as its default gets the model already
    loaded rather than one that forces an eviction.

    Chat clients get chat models: `/capabilities` enumerates every modality.
    """
    from giq.adapters.llama_cpp import weights_installed

    residents = resident_defaults()
    offered = [r for r in recipes_serving("llm") if weights_installed(r.name)]
    return sorted(
        offered,
        key=lambda r: (0, residents.index(r.name), "") if r.name in residents else (1, 0, r.name),
    )


@router.get("/models")
async def list_models() -> dict:
    """List available models (OpenAI-compatible endpoint).

    Generated from the recipes, for the same reason `/capabilities` is: a
    hand-written list drifts. With a literal here, a model could be
    registered and servable while every OpenAI client was told it did not
    exist, because adding a model meant remembering to edit this list too.
    """
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": recipe.name, "object": "model", "created": now, "owned_by": "giq"}
            for recipe in _advertised_llm_recipes()
        ],
    }
