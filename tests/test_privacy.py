# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Prompts must not be written anywhere. Only that a job happened, and its size.

giq used to write the full request — every prompt, system message and chat body
— in plaintext to the in-flight diagnostics log, 155MB of it. The intent was
post-mortem after a hard power cut ("which job was running when the machine
died"), which needs a job id, a model and a size, and never needed the text.

These tests exist because that kind of leak is invisible in normal operation:
nothing fails, nothing looks wrong, the file just quietly accumulates what
people typed. So the check is not "is the current field list safe" but "does a
canary string survive anywhere in what we write" — a test that keeps working
when someone adds a field nobody thought about.
"""

import asyncio
import json

import pytest

from giq.models import JobRequest, Modality
from giq.queue import Job
from giq.runner import _log_inflight, _request_shape

# Distinctive enough that a substring check cannot pass by accident.
CANARY = "zqx-private-prompt-canary-8f3a"


def chat_job(**chat) -> Job:
    body = {"messages": [{"role": "user", "content": CANARY}], "max_tokens": 2048}
    body.update(chat)
    return Job(
        job_id="j1",
        request=JobRequest(modality=Modality.llm, model="qwen3.8-27b", chat_request=body),
    )


def task_job() -> Job:
    return Job(
        job_id="j2",
        request=JobRequest(
            modality=Modality.llm,
            model="qwen3.8-27b",
            tasks=[{"id": "t0", "system": f"system {CANARY}", "user": f"user {CANARY}"}],
            params={"max_tokens": 512, "temperature": 0.7},
        ),
    )


@pytest.fixture
def written(tmp_path, monkeypatch):
    """Capture what _log_inflight actually writes to disk."""
    import giq.runner as runner

    path = tmp_path / "inflight.log"
    monkeypatch.setattr(runner, "_INFLIGHT_LOG_PATH", str(path))
    monkeypatch.setattr(runner, "_inflight_fd", None)
    yield path


def test_a_chat_prompt_never_reaches_the_log(written):
    _log_inflight("start", chat_job())
    text = written.read_text()

    assert CANARY not in text, "the prompt was written to the in-flight log"
    assert '"job_id": "j1"' in text, "the job itself must still be identifiable"


def test_task_prompts_never_reach_the_log(written):
    _log_inflight("start", task_job())

    assert CANARY not in written.read_text()


def test_an_image_is_counted_not_stored(written):
    """A data: URI is the user's picture. Its size is diagnostic; it is not."""
    job = chat_job(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CANARY},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
    )
    _log_inflight("start", job)
    text = written.read_text()

    assert CANARY not in text
    assert "base64" not in text and "AAAA" not in text
    assert json.loads(text)["shape"]["attachments"] == 1


def test_an_unknown_field_carrying_text_still_does_not_leak(written):
    """The guarantee has to survive someone adding a field years from now, so
    measurement is additive rather than a list of things to redact."""
    _log_inflight("start", chat_job(some_future_field={"nested": [CANARY]}))

    assert CANARY not in written.read_text()


def test_the_end_record_carries_no_output(written):
    """The answer is as private as the question."""
    job = chat_job()
    job.results = [{"choices": [{"message": {"content": CANARY}}]}]
    _log_inflight("end", job, status="completed", results=len(job.results))

    assert CANARY not in written.read_text()


def test_shape_still_answers_the_question_the_log_exists_for():
    """Post-mortem needs: which job, which model, how big. All present."""
    shape = _request_shape(chat_job().request)

    assert shape["messages"] == 1
    assert shape["max_tokens"] == 2048
    assert shape["input_chars"] >= len(CANARY)


def test_string_params_are_dropped_rather_than_logged():
    """A string parameter could be text someone typed; none worth logging are."""
    job = task_job()
    job.request.params["negative_prompt"] = CANARY

    shape = _request_shape(job.request)

    assert CANARY not in json.dumps(shape)
    assert shape["temperature"] == 0.7


def test_the_stats_database_has_nowhere_to_put_a_prompt():
    """Defence in depth: even a careless caller cannot persist text here."""
    import sqlite3
    import tempfile
    from pathlib import Path

    from giq.stats import StatsRecorder

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "s.db"
        StatsRecorder(db).init_sync()
        cols = {row[1] for row in sqlite3.connect(db).execute("PRAGMA table_info(jobs)")}

    assert not cols & {"prompt", "messages", "chat_request", "input", "output", "text"}


# --- the free-text field the shape does not cover -------------------------
#
# _request_shape measures, so it is safe as fields are added. Errors bypass it
# entirely: _log_inflight writes whatever keyword arguments it is handed, and
# stats.db.jobs.error keeps that string forever. Most errors are CUDA messages
# and belong there — but some exceptions quote their input, and nothing about
# the string says which kind it is.


def test_an_error_that_quotes_the_prompt_is_withheld(written):
    """A validation error carries the value that failed; an engine that logs
    the prompt puts it in the stderr tail giq attaches to the failure. Either
    way the request comes back out through the error."""
    job = chat_job()
    _log_inflight("end", job, status="failed", error=f"render failed on input: {CANARY}")

    text = written.read_text()
    assert CANARY not in text
    assert "withheld" in text


def test_an_ordinary_error_is_still_recorded(written):
    """The log has to stay useful. Withholding every error would be safe and
    would also throw away the reason the file exists."""
    job = chat_job()
    _log_inflight("end", job, status="failed", error="No CUDA GPUs are available")

    assert "No CUDA GPUs are available" in written.read_text()


def test_a_scheduler_message_naming_the_model_is_not_mistaken_for_a_quote(written):
    """Model and worker names are giq's vocabulary, not the caller's. If they
    counted as request text, the eviction messages would all disappear."""
    _log_inflight("end", chat_job(), status="requeued", reason="qwen3.8-27b is no longer resident")

    assert "no longer resident" in written.read_text()


def test_a_future_keyword_carrying_the_prompt_cannot_reopen_the_hole(written):
    """The same additive promise the shape makes: someone attaching a new
    field years from now does not get to leak through it."""
    _log_inflight("end", chat_job(), status="failed", some_future_note=f"context: {CANARY}")

    assert CANARY not in written.read_text()


def test_the_database_never_stores_an_error_that_quotes_the_prompt():
    """stats.db keeps errors forever, so this is the copy that matters most."""
    import sqlite3
    import tempfile
    from datetime import datetime
    from pathlib import Path

    from giq.models import JobStatus
    from giq.stats import StatsRecorder

    job = chat_job()
    job.status = JobStatus.failed
    job.error = f"failed on input: {CANARY}"
    job.completed_at = datetime.now()

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "s.db"
        rec = StatsRecorder(db)
        rec.init_sync()
        asyncio.run(rec.record_job(job))
        rows = list(sqlite3.connect(db).execute("SELECT error FROM jobs"))

    assert rows, "the job should still be recorded"
    assert CANARY not in str(rows)


# --- prompts must not leave the machine either ----------------------------
#
# giq once let `model: "codex"` in /v1/chat/completions shelled out to the
# Codex CLI with --full-auto, handing the caller's prompt to OpenAI and letting
# it run shell commands here. It bypassed the orchestrator entirely — no pause
# check, no queue, no stats row, no in-flight log — so a request that left the
# machine was also a request giq had no record of. It is gone; these keep it
# gone, and any successor with it.


def test_the_api_layer_never_spawns_a_process():
    """Only workers run programs, and only through the queue. An API handler
    that shells out is a request escaping the scheduler and the accounting —
    which is how the prompt got off the machine last time."""
    import pathlib

    api = pathlib.Path(__file__).resolve().parents[1] / "src" / "giq" / "api"
    offenders = [p.name for p in api.glob("*.py") if "subprocess" in p.read_text()]

    assert not offenders, f"API modules spawning processes: {offenders}"


@pytest.mark.asyncio
async def test_only_locally_served_models_are_advertised(tmp_path, monkeypatch):
    """/v1/models is where a caller learns what giq will run. Everything on it
    has to be something giq serves itself."""
    from giq.adapters.llama_cpp import MODEL_PATHS
    from giq.api.openai_compat import list_models
    from giq.engines import reload_engines

    # Only what runs here is listed: weights on disk and an engine to run
    # them, both faked under a temporary directory.
    gguf = tmp_path / MODEL_PATHS["gemma-4-12b"]
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    server = tmp_path / "llama-server"
    server.write_text("")
    monkeypatch.setenv("GIQ_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("GIQ_LLAMA_BINARY", str(server))
    reload_engines()
    try:
        data = (await list_models())["data"]
    finally:
        monkeypatch.delenv("GIQ_LLAMA_BINARY")
        reload_engines()

    assert data, "the model list should not be empty"
    assert {m["owned_by"] for m in data} == {"giq"}


def test_an_ocr_document_is_counted_not_stored(written):
    """A PDF or a page image is the user's document. Its count is diagnostic;
    its bytes are not — and neither is any text a client put next to it."""
    job = Job(
        job_id="j-ocr",
        request=JobRequest(
            modality=Modality.ocr,
            model="unlimited-ocr",
            tasks=[
                {"id": "t0", "pdf_b64": CANARY, "dpi": 200},
                {"id": "t1", "images_b64": [CANARY, CANARY]},
            ],
        ),
    )
    _log_inflight("start", job)
    text = written.read_text()

    assert CANARY not in text
    assert json.loads(text)["shape"]["attachments"] == 3
