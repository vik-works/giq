# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for giq API endpoints."""

import base64

import pytest
from httpx import ASGITransport, AsyncClient

from giq.main import app
from giq.models import JobRequest, JobStatus, Modality
from giq.queue import JobQueue
from giq.runner import Runner


@pytest.fixture
async def client():
    """Create async test client with fresh queue/runner."""
    # Reset global state for each test
    import giq.queue
    import giq.runner

    giq.queue._queue = JobQueue()
    giq.runner._runner = Runner(giq.queue._queue)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_run_job(client: AsyncClient):
    """Test submitting a job."""
    response = await client.post(
        "/run",
        json={
            "worker": "llm",
            "model": "gemma-3-27b-it-qat",
            "tasks": [
                {"id": "t1", "system": "You are helpful.", "user": "Hello!"},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "position" in data
    assert isinstance(data["position"], int)


@pytest.mark.asyncio
async def test_run_job_multiple_tasks(client: AsyncClient):
    """Test submitting a job with multiple tasks."""
    response = await client.post(
        "/run",
        json={
            "worker": "llm",
            "model": "gemma-3-27b-it-qat",
            "params": {"temperature": 0.7},
            "tasks": [
                {"id": "t1", "user": "First task"},
                {"id": "t2", "user": "Second task"},
                {"id": "t3", "user": "Third task"},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_get_job_status(client: AsyncClient):
    """Test getting job status."""
    # First submit a job
    submit_response = await client.post(
        "/run",
        json={
            "worker": "llm",
            "model": "test-model",
            "tasks": [{"id": "t1", "user": "Test"}],
        },
    )
    job_id = submit_response.json()["job_id"]

    # Then get its status
    response = await client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == JobStatus.pending
    assert data["modality"] == Modality.llm
    assert data["model"] == "test-model"


@pytest.mark.asyncio
async def test_get_job_not_found(client: AsyncClient):
    """Test getting non-existent job."""
    response = await client.get("/jobs/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_job(client: AsyncClient):
    """Test cancelling a job."""
    # Submit a job
    submit_response = await client.post(
        "/run",
        json={
            "worker": "llm",
            "model": "test-model",
            "tasks": [{"id": "t1", "user": "Test"}],
        },
    )
    job_id = submit_response.json()["job_id"]

    # Cancel it
    response = await client.delete(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["cancelled"] is True

    # Verify it's gone
    get_response = await client.get(f"/jobs/{job_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_job_not_found(client: AsyncClient):
    """Test cancelling non-existent job."""
    response = await client.delete("/jobs/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_service_status(client: AsyncClient):
    """Test getting service status."""
    response = await client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "vram_used_gb" in data
    assert "vram_total_gb" in data
    assert "queue_depth" in data
    assert "jobs_pending" in data
    assert isinstance(data["jobs_pending"], list)


@pytest.mark.asyncio
async def test_capabilities(client: AsyncClient):
    """Test getting capabilities."""
    response = await client.get("/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "modalities" in data
    assert "constraints" in data
    # Every modality giq serves, not just the original four — the
    # hand-written payload used to omit audio, embed and stt.
    for worker in (
        "llm",
        "text2image",
        "image_edit",
        "tts",
        "audio",
        "embed",
        "stt",
        "ocr",
        "depth",
    ):
        assert worker in data["modalities"], f"{worker} missing from /capabilities"
    # Two LLM engines since vllm joined llama.cpp, reported like the image runtimes.
    assert "llama.cpp" in data["modalities"]["llm"]["engine"]
    assert "vllm" in data["modalities"]["llm"]["engine"]
    # flux_klein is the production model and was absent from the old
    # hand-written list.
    assert data["modalities"]["text2image"]["engine"] == "sd.cpp"
    assert "flux_klein" in data["modalities"]["text2image"]["recipes"]
    assert "flux_klein" in data["modalities"]["image_edit"]["recipes"]
    assert "recipes" in data["modalities"]["llm"]


def test_the_token_budget_defaults_to_the_models_context():
    """Unset means "as much as the context allows", and the modern OpenAI
    spelling has to land on the same field.

    It used to default to 2048 — a ceiling on thinking *and* answer, so a
    reasoning model spent it on the thought and returned an empty answer with
    finish_reason "length". And `max_completion_tokens`, which is what current
    OpenAI clients send, was dropped by extra="ignore" so a client asking for
    a big budget silently got 2048.
    """
    from giq.api.openai_compat import ChatCompletionRequest

    msgs = [{"role": "user", "content": "hi"}]

    assert ChatCompletionRequest(model="m", messages=msgs).max_tokens is None
    # Both spellings reach the same field.
    assert (
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": msgs, "max_completion_tokens": 40000}
        ).max_tokens
        == 40000
    )
    assert (
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": msgs, "max_tokens": 4096}
        ).max_tokens
        == 4096
    )


@pytest.mark.asyncio
async def test_v1_models_advertises_every_installed_model(client: AsyncClient):
    """Installed is the whole rule: on disk means offered.

    Two failures, opposite directions, from the same hand-written list of ids.
    It omitted qwen3.8-27b, which was registered and servable — and it
    advertised five models whose GGUFs had been deleted, so a client could
    pick one and llama-server would fail on it. Generated from
    the recipes now and filtered only on whether the weights exist.
    """
    from giq.adapters.llama_cpp import weights_installed
    from giq.registry import all_recipes, recipes_serving, resident_defaults

    response = await client.get("/v1/models")
    assert response.status_code == 200
    advertised = [m["id"] for m in response.json()["data"]]

    installed = {r.name for r in recipes_serving("llm") if weights_installed(r.name)}
    assert set(advertised) == installed, (
        f"missing {sorted(installed - set(advertised))}, "
        f"phantom {sorted(set(advertised) - installed)}"
    )

    # No audit gate: nothing is withheld for being unaudited or unpopular.
    # If it is on disk it is on the list, whatever we think of it.
    for name in installed:
        assert name in advertised, f"{name} is installed but withheld"

    # Chat clients get chat models; /capabilities enumerates every modality.
    others = {r.name for r in all_recipes() if not r.serves("llm")}
    assert not (set(advertised) & others)

    # Residents lead, so a client defaulting to data[0] gets the loaded model
    # rather than one that forces an eviction.
    residents = [name for name in resident_defaults() if name in installed]
    if residents:
        assert advertised[0] in residents


@pytest.mark.asyncio
async def test_run_text2image_job(client: AsyncClient):
    """Test submitting a text2image generation job."""
    response = await client.post(
        "/run",
        json={
            "worker": "text2image",
            "model": "zimage",
            "tasks": [
                {"id": "img1", "prompt": "A beautiful sunset over mountains"},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_run_image_edit_job(client: AsyncClient):
    """Test submitting an image edit job."""
    response = await client.post(
        "/run",
        json={
            "worker": "image_edit",
            "model": "flux_klein",
            "tasks": [
                {
                    "id": "edit1",
                    "reference_image_b64": "dGVzdA==",  # "test" in base64
                    "instruction": "Make it look like a painting",
                },
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


@pytest.mark.asyncio
async def test_run_tts_job(client: AsyncClient):
    """Test submitting a TTS job."""
    response = await client.post(
        "/run",
        json={
            "worker": "tts",
            "model": "kokoro-82m",
            "params": {"voice": "af_heart"},
            "tasks": [
                {"id": "tts1", "user": "Hello, this is a test."},
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


# --- /ocr --------------------------------------------------------------------


def _completed_ocr_job(job_id: str, result: dict):
    from giq.queue import Job

    job = Job(
        job_id=job_id,
        request=JobRequest(modality=Modality.ocr, model="unlimited-ocr", tasks=[]),
    )
    job.status = JobStatus.completed
    job.results = [result]
    return job


_OCR_RESULT = {
    "id": "ocr-0",
    "html": '<p class="text" data-page="1">hello</p>',
    "pages": 1,
    "blocks": [{"page": 1, "label": "text", "content": "hello"}],
    "raw": "<PAGE>\n<|det|>text [1, 1, 2, 2]<|/det|>hello\n",
    "tokens_in": 260,
    "tokens_out": 3,
    "truncated": False,
    "error": None,
}


@pytest.fixture
def ocr_completes(monkeypatch):
    """wait_for_job answers with a canned result and records the submitted task."""
    from giq.services.orchestration import Orchestrator

    seen: dict = {}
    real_submit = Orchestrator.submit_job

    async def submit(self, request):
        seen["task"] = request.tasks[0]
        seen["model"] = request.model
        return await real_submit(self, request)

    async def wait(self, job_id, timeout=None):
        seen["timeout"] = timeout
        return _completed_ocr_job(job_id, _OCR_RESULT)

    monkeypatch.setattr(Orchestrator, "submit_job", submit)
    monkeypatch.setattr(Orchestrator, "wait_for_job", wait)
    return seen


PDF = b"%PDF-1.4 not really a document"


@pytest.mark.asyncio
async def test_ocr_takes_the_pdf_as_the_body(client: AsyncClient, ocr_completes):
    """One PDF in, one document out — the job goes through the queue like any
    other, and the answer is the assembled HTML, not a job id."""
    r = await client.post("/ocr", content=PDF, headers={"content-type": "application/pdf"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["html"].startswith("<p") and body["pages"] == 1 and body["job_id"]
    assert body["tokens_in"] == 260 and body["tokens_out"] == 3
    assert "raw" not in body
    assert ocr_completes["timeout"] >= 600
    task = ocr_completes["task"]
    assert task["dpi"] == 200 and task["strip"] and task["merge"] and "pages" not in task
    assert ocr_completes["model"] == "unlimited-ocr"


@pytest.mark.asyncio
async def test_ocr_model_is_the_consumers_choice(client: AsyncClient, ocr_completes):
    r = await client.post(
        "/ocr",
        params={"model": "glm-ocr"},
        content=PDF,
        headers={"content-type": "application/pdf"},
    )
    assert r.status_code == 200, r.text
    assert ocr_completes["model"] == "glm-ocr"
    r = await client.post(
        "/ocr",
        params={"model": "tesseract"},
        content=PDF,
        headers={"content-type": "application/pdf"},
    )
    assert r.status_code == 400 and "glm-ocr" in r.json()["detail"]


@pytest.mark.asyncio
async def test_ocr_takes_a_multipart_file_too(client: AsyncClient, ocr_completes):
    r = await client.post("/ocr", files={"file": ("doc.pdf", PDF, "application/pdf")})
    assert r.status_code == 200, r.text
    assert r.json()["html"].startswith("<p")
    # A multipart body without the file part is a client error, not a crash.
    r = await client.post("/ocr", files={"other": ("x.pdf", PDF, "application/pdf")})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_ocr_options_are_query_parameters(client: AsyncClient, ocr_completes):
    r = await client.post(
        "/ocr",
        params={"dpi": 150, "pages": "1-2,5", "raw": "true", "strip": "false"},
        content=PDF,
        headers={"content-type": "application/pdf"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["raw"].startswith("<PAGE>")
    task = ocr_completes["task"]
    assert task["dpi"] == 150 and task["pages"] == [1, 2, 5] and task["strip"] is False

    r = await client.post(
        "/ocr",
        params={"response_format": "html"},
        content=PDF,
        headers={"content-type": "application/pdf"},
    )
    assert r.headers["content-type"].startswith("text/html") and r.text.startswith("<p")


@pytest.mark.asyncio
async def test_ocr_refuses_a_document_over_the_cap(client: AsyncClient, monkeypatch):
    """The cap is checked before the body is buffered, and again while it
    streams, so a lying Content-Length does not get around it."""
    monkeypatch.setenv("GIQ_OCR_MAX_UPLOAD_MB", "1")
    big = b"%PDF" + b"x" * (1024 * 1024 + 1)
    r = await client.post("/ocr", content=big, headers={"content-type": "application/pdf"})
    assert r.status_code == 413
    r = await client.post("/ocr", files={"file": ("d.pdf", big, "application/pdf")})
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_ocr_rejects_what_is_not_a_pdf(client: AsyncClient):
    r = await client.post("/ocr", content=b"hello", headers={"content-type": "text/plain"})
    assert r.status_code == 400
    r = await client.post("/ocr", params={"pages": "3-1"}, content=PDF)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_ocr_endpoint_reports_a_failed_task(client: AsyncClient, monkeypatch):
    from giq.services.orchestration import Orchestrator

    async def fake_wait(self, job_id, timeout=None):
        return _completed_ocr_job(job_id, {"id": "ocr-0", "error": "no pages to parse"})

    monkeypatch.setattr(Orchestrator, "wait_for_job", fake_wait)
    r = await client.post("/ocr", content=PDF, headers={"content-type": "application/pdf"})
    assert r.status_code == 500 and "no pages" in r.json()["detail"]


@pytest.mark.asyncio
async def test_ocr_runs_through_the_generic_job_path(client: AsyncClient):
    """The worker is a first-class job type: /run accepts it and /capabilities
    advertises it, so a consumer on another node needs nothing special."""
    r = await client.post(
        "/run",
        json={"worker": "ocr", "model": "unlimited-ocr", "tasks": [{"id": "t", "pdf_b64": "AAAA"}]},
    )
    assert r.status_code == 200 and r.json()["job_id"]
    caps = (await client.get("/capabilities")).json()
    assert sorted(caps["modalities"]["ocr"]["recipes"]) == ["glm-ocr", "unlimited-ocr"]


def test_parse_pages():
    from giq.api.router import parse_pages

    assert parse_pages(None) is None and parse_pages(" ") is None
    assert parse_pages("1-3,7") == [1, 2, 3, 7]
    assert parse_pages("4") == [4]


# --- /depth ------------------------------------------------------------------

_DEPTH_RESULT = {
    "id": "depth-0",
    "depth_b64": "iVBORw0=",
    "width": 2,
    "height": 2,
    "depth_min": 0.5,
    "depth_max": 9.0,
    "metric": False,
    "visualization_b64": "iVBORw1=",
    "error": None,
}


@pytest.fixture
def depth_completes(monkeypatch):
    """wait_for_job answers with a canned map and records the submitted task."""
    from giq.queue import Job
    from giq.services.orchestration import Orchestrator

    seen: dict = {}
    real_submit = Orchestrator.submit_job

    async def submit(self, request):
        seen["task"] = request.tasks[0]
        seen["model"] = request.model
        seen["worker"] = request.modality
        return await real_submit(self, request)

    async def wait(self, job_id, timeout=None):
        seen["timeout"] = timeout
        job = Job(
            job_id=job_id,
            request=JobRequest(modality=Modality.depth, model="depth-anything-v2-small", tasks=[]),
        )
        job.status = JobStatus.completed
        job.results = [_DEPTH_RESULT]
        return job

    monkeypatch.setattr(Orchestrator, "submit_job", submit)
    monkeypatch.setattr(Orchestrator, "wait_for_job", wait)
    return seen


PNG = b"\x89PNG\r\n\x1a\n not really an image"


@pytest.mark.asyncio
async def test_depth_takes_the_image_as_the_body(client: AsyncClient, depth_completes):
    r = await client.post("/depth", content=PNG, headers={"content-type": "image/png"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["depth_b64"] == "iVBORw0=" and (body["width"], body["height"]) == (2, 2)
    assert body["depth_min"] == 0.5 and body["metric"] is False and body["job_id"]
    assert "visualization_b64" not in body  # only on request
    assert depth_completes["worker"] == Modality.depth
    assert depth_completes["model"] == "depth-anything-v2-small"
    task = depth_completes["task"]
    assert task["image_b64"] and task["visualize"] is False


@pytest.mark.asyncio
async def test_depth_visualization_is_opt_in(client: AsyncClient, depth_completes):
    r = await client.post(
        "/depth", params={"visualize": "true"}, content=PNG, headers={"content-type": "image/png"}
    )
    assert r.json()["visualization_b64"] == "iVBORw1="
    assert depth_completes["task"]["visualize"] is True
    # The image formats hand back the PNG bytes themselves.
    r = await client.post(
        "/depth",
        params={"response_format": "png"},
        content=PNG,
        headers={"content-type": "image/png"},
    )
    assert r.headers["content-type"] == "image/png"
    assert r.content == base64.b64decode("iVBORw0=")
    r = await client.post(
        "/depth",
        params={"response_format": "visualization"},
        content=PNG,
        headers={"content-type": "image/png"},
    )
    assert r.status_code == 200 and depth_completes["task"]["visualize"] is True


@pytest.mark.asyncio
async def test_depth_model_is_the_consumers_choice(
    client: AsyncClient, depth_completes, monkeypatch
):
    # The public registry has one depth model; a second one added for the
    # test shows the query parameter reaches the job rather than the default.
    from giq.registry import get_recipe
    from tests._recipes import with_recipes

    small = get_recipe("depth-anything-v2-small")
    assert small is not None
    with_recipes(monkeypatch, small.model_copy(update={"name": "depth-test-large"}))
    r = await client.post(
        "/depth",
        params={"model": "depth-test-large"},
        content=PNG,
        headers={"content-type": "image/png"},
    )
    assert r.status_code == 200 and depth_completes["model"] == "depth-test-large"
    r = await client.post(
        "/depth", params={"model": "midas"}, content=PNG, headers={"content-type": "image/png"}
    )
    assert r.status_code == 400 and "depth-anything-v2-small" in r.json()["detail"]


@pytest.mark.asyncio
async def test_depth_refuses_what_is_not_an_image(client: AsyncClient, depth_completes):
    r = await client.post("/depth", content=b"%PDF-1.4", headers={"content-type": "image/png"})
    assert r.status_code == 400 and "task" not in depth_completes
    r = await client.post(
        "/depth", files={"file": ("a.jpg", b"\xff\xd8\xff\xe0 jpeg", "image/jpeg")}
    )
    assert r.status_code == 200
    r = await client.post(
        "/depth", files={"file": ("a.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp")}
    )
    assert r.status_code == 200
    r = await client.post(
        "/depth", files={"file": ("a.webp", b"RIFF\x00\x00\x00\x00WAVEfmt ", "image/webp")}
    )
    assert r.status_code == 400


# --- structured output forwarding --------------------------------------------
#
# giq used to drop `response_format` and `structured_outputs` at the API edge
# (extra="ignore"), so a client asking for schema-constrained JSON got
# free-form text. These assert both knobs are declared on the request model,
# forwarded unchanged to the engine request in every branch, and refused when
# a caller sends the pair vllm cannot satisfy.

_CAR_SCHEMA = {
    "type": "object",
    "properties": {"brand": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["brand", "year"],
    "additionalProperties": False,
}


def test_structured_output_fields_survive_parsing():
    """response_format and structured_outputs are first-class, not dropped."""
    from giq.api.openai_compat import ChatCompletionRequest

    msgs = [{"role": "user", "content": "hi"}]
    native = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": msgs, "response_format": {"type": "json_object"}}
    )
    assert native.response_format == {"type": "json_object"}
    assert native.structured_outputs is None
    vllm = ChatCompletionRequest.model_validate(
        {"model": "m", "messages": msgs, "structured_outputs": {"json": _CAR_SCHEMA}}
    )
    assert vllm.structured_outputs == {"json": _CAR_SCHEMA}
    assert vllm.response_format is None
    # Absent by default — a plain request stays plain.
    plain = ChatCompletionRequest.model_validate({"model": "m", "messages": msgs})
    assert plain.response_format is None and plain.structured_outputs is None


def test_structured_output_knobs_are_mutually_exclusive():
    """Both at once is a caller error, not something to pass to the engine.

    vllm merges response_format into its structured-output constraints and
    rejects the merged pair, so forwarding both would surface as an opaque
    engine failure long after the request was accepted.
    """
    from pydantic import ValidationError

    from giq.api.openai_compat import ChatCompletionRequest

    with pytest.raises(ValidationError, match="mutually exclusive"):
        ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
                "structured_outputs": {"json": _CAR_SCHEMA},
            }
        )


@pytest.fixture
def chat_completes(monkeypatch):
    """Capture the JobRequest a /v1/chat/completions call submits."""
    from giq.queue import Job
    from giq.services.orchestration import Orchestrator

    seen: dict = {}

    async def submit(self, request):
        seen["request"] = request
        return "job-xyz", 0

    async def wait(self, job_id, timeout=None):
        job = Job(
            job_id=job_id,
            request=JobRequest(modality=Modality.llm, model="m", tasks=[]),
        )
        job.status = JobStatus.completed
        job.results = [
            {
                "model": "whatever-the-engine-was-started-as",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "", "reasoning_content": "hmm"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ]
        return job

    monkeypatch.setattr(Orchestrator, "submit_job", submit)
    monkeypatch.setattr(Orchestrator, "wait_for_job", wait)
    return seen


@pytest.mark.asyncio
async def test_plain_path_forwards_structured_output(client: AsyncClient, chat_completes):
    """A plain request (no tools, not streamed) threads either knob through.

    This is the shape an extraction pipeline sends: one system+user message,
    no tools. The key must reach the engine in the request it is sent.
    """
    base = {
        "model": "qwen3.8-27b-nvfp4-chat",
        "messages": [
            {"role": "system", "content": "Extract car info."},
            {"role": "user", "content": "Mazda MX-5, 1990."},
        ],
    }
    r = await client.post(
        "/v1/chat/completions", json={**base, "response_format": {"type": "json_object"}}
    )
    assert r.status_code == 200, r.text
    params = chat_completes["request"].chat_request
    assert params["response_format"] == {"type": "json_object"}
    assert "structured_outputs" not in params

    r = await client.post(
        "/v1/chat/completions", json={**base, "structured_outputs": {"json": _CAR_SCHEMA}}
    )
    assert r.status_code == 200, r.text
    params = chat_completes["request"].chat_request
    assert params["structured_outputs"] == {"json": _CAR_SCHEMA}
    assert "response_format" not in params


@pytest.mark.asyncio
async def test_tool_path_forwards_structured_output(client: AsyncClient, chat_completes):
    """The tool path threads either knob into chat_request too."""
    base = {
        "model": "qwen3.8-27b-nvfp4-chat",
        "messages": [{"role": "user", "content": "pick a tool"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }
        ],
    }
    r = await client.post(
        "/v1/chat/completions", json={**base, "response_format": {"type": "json_object"}}
    )
    assert r.status_code == 200, r.text
    chat_request = chat_completes["request"].chat_request
    assert chat_request["response_format"] == {"type": "json_object"}
    assert "structured_outputs" not in chat_request

    r = await client.post(
        "/v1/chat/completions", json={**base, "structured_outputs": {"json": _CAR_SCHEMA}}
    )
    assert r.status_code == 200, r.text
    chat_request = chat_completes["request"].chat_request
    assert chat_request["structured_outputs"] == {"json": _CAR_SCHEMA}
    assert "response_format" not in chat_request


@pytest.mark.asyncio
async def test_stream_path_forwards_structured_output(client: AsyncClient, monkeypatch):
    """Streaming uses chat_request too and must carry the decoding constraint."""
    from giq.queue import JobStream
    from giq.services.orchestration import Orchestrator

    seen = {}

    async def submit(self, request):
        seen["request"] = request
        stream = JobStream()
        stream.queue.put_nowait(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": "stop"}],
            }
        )
        await stream.close()
        return "job-xyz", 0, stream

    monkeypatch.setattr(Orchestrator, "submit_streaming_job", submit)
    base = {
        "model": "qwen3.8-27b-nvfp4-chat",
        "messages": [{"role": "user", "content": "extract"}],
        "stream": True,
    }
    r = await client.post(
        "/v1/chat/completions", json={**base, "response_format": {"type": "json_object"}}
    )
    assert r.status_code == 200, r.text
    assert seen["request"].chat_request["response_format"] == {"type": "json_object"}

    r = await client.post(
        "/v1/chat/completions", json={**base, "structured_outputs": {"json": _CAR_SCHEMA}}
    )
    assert r.status_code == 200, r.text
    chat_request = seen["request"].chat_request
    assert chat_request["structured_outputs"] == {"json": _CAR_SCHEMA}
    assert "response_format" not in chat_request


@pytest.mark.asyncio
async def test_both_structured_output_knobs_are_rejected(client: AsyncClient, chat_completes):
    """The conflict is reported as a 400 instead of reaching the engine."""
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.8-27b-nvfp4-chat",
            "messages": [{"role": "user", "content": "extract"}],
            "response_format": {"type": "json_object"},
            "structured_outputs": {"json": _CAR_SCHEMA},
        },
    )
    assert r.status_code == 400
    assert "mutually exclusive" in r.json()["detail"]
    assert "request" not in chat_completes


@pytest.mark.asyncio
async def test_a_plain_request_goes_to_the_engine_whole(client: AsyncClient, chat_completes):
    """Every turn of the conversation reaches the engine, and its answer comes
    back as written: an answer cut off mid-thought says "length", not "stop"."""
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello Ada."},
        {"role": "user", "content": "What is my name?"},
    ]
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "gemma-4-12b", "messages": messages, "max_tokens": 20},
    )
    assert r.status_code == 200, r.text
    sent = chat_completes["request"].chat_request
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]
    assert sent["max_tokens"] == 20 and "tools" not in sent

    body = r.json()
    assert body["model"] == "gemma-4-12b"
    (choice,) = body["choices"]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["reasoning_content"] == "hmm"
