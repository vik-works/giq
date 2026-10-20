# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""POST /decide: the body becomes one decide task, and what it may not carry is
refused before it reaches the queue. The app runs without its lifespan, on a
fake orchestrator."""

import base64
import json

import pytest

from giq.models import JobRequest, JobStatus

QUESTIONS = {"queue": {"type": "choice", "instructions": "Route.", "criteria": {"a": None}}}
ANSWER = {"answers": {"queue": {"type": "choice", "choice": "a"}}, "tokens_in": 9, "images": 0}


class FakeOrchestrator:
    """Completes every job at once with one canned decision."""

    def __init__(self):
        self.submitted: list[JobRequest] = []

    async def submit_job(self, request: JobRequest) -> tuple[str, int]:
        self.submitted.append(request)
        return "j1", 0

    async def wait_for_job(self, job_id: str, timeout: float | None = None):
        from giq.queue import Job

        job = Job(job_id=job_id, request=self.submitted[-1])
        job.status = JobStatus.completed
        job.results = [{"id": "decide-0", **ANSWER}]
        return job


@pytest.fixture
def api():
    from fastapi.testclient import TestClient

    from giq.api.dependencies import get_orchestrator
    from giq.main import app

    orch = FakeOrchestrator()
    app.dependency_overrides[get_orchestrator] = lambda: orch
    try:
        yield TestClient(app, base_url="http://localhost"), orch
    finally:
        app.dependency_overrides.pop(get_orchestrator, None)


def test_a_json_request_is_one_decide_task(api):
    client, orch = api
    r = client.post("/decide", json={"state": "Ticket.", "questions": QUESTIONS, "rotations": 4})
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["queue"]["choice"] == "a" and r.json()["job_id"] == "j1"
    (task,) = orch.submitted[-1].tasks
    assert orch.submitted[-1].modality == "decide"
    assert (task["state"], task["rotations"], task["images_b64"]) == ("Ticket.", 4, [])


def test_multipart_images_travel_as_base64(api):
    client, orch = api
    r = client.post(
        "/decide",
        data={"request": json.dumps({"questions": QUESTIONS})},
        files=[("image0", ("a.png", b"\x89PNG one")), ("image1", ("b.png", b"\x89PNG two"))],
    )
    assert r.status_code == 200, r.text
    (task,) = orch.submitted[-1].tasks
    assert [base64.b64decode(b) for b in task["images_b64"]] == [b"\x89PNG one", b"\x89PNG two"]


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"questions": QUESTIONS, "rotations": 0}, "rotations"),
        ({"questions": QUESTIONS, "rotations": 50}, "rotations"),
        ({"questions": QUESTIONS, "rotations": True}, "rotations"),
        ({"questions": {}}, "questions"),
        ({"questions": QUESTIONS, "image0_b64": "not base64!"}, "base64"),
    ],
)
def test_what_a_request_may_not_carry_is_refused(api, body, detail):
    client, orch = api
    r = client.post("/decide", json=body)
    assert r.status_code == 400 and detail in r.json()["detail"]
    assert not orch.submitted


def test_an_unknown_model_is_refused(api):
    client, _ = api
    r = client.post("/decide", params={"model": "nope"}, json={"questions": QUESTIONS})
    assert r.status_code == 400 and "imajev-2b" in r.json()["detail"]
