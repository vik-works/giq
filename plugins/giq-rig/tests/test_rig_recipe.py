# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The built-in skintokens recipe, and the mesh2rig modality reached through
the job API like any other: the plugin's registrations, nothing special in
core."""

import pytest
from httpx import ASGITransport, AsyncClient

from giq import plugins
from giq.models import JobRequest
from giq.registry import get_recipe, resident_defaults


def test_the_builtin_recipe_is_complete():
    recipe = get_recipe("skintokens")
    assert recipe is not None
    assert (recipe.engine, recipe.modalities) == ("skintokens", ("mesh2rig",))
    assert recipe.weights.source == "hf:VAST-AI/SkinTokens" and recipe.weights.revision
    assert recipe.weights.licence == "mit"
    llm = recipe.weights.parts["llm"]
    assert llm.source == "hf:Qwen/Qwen3-0.6B/config.json", "the config alone, not Qwen's weights"
    assert llm.revision and llm.licence == "apache-2.0"
    assert recipe.vram.measured and recipe.vram.gb == 4.5


def test_it_loads_on_demand():
    assert "skintokens" not in resident_defaults()
    assert get_recipe("skintokens").residency.priority is None


def test_the_modality_is_the_plugins():
    modality = plugins.modalities()["mesh2rig"]
    assert modality.parts == frozenset({"llm"})
    assert modality.payload_keys == ("glb_b64",), "the mesh is counted as a blob, never logged"
    assert plugins.engine("skintokens").prepare is not None
    assert JobRequest(modality="mesh2rig", model="skintokens").modality == "mesh2rig"


@pytest.fixture
async def client():
    import giq.queue
    import giq.runner
    from giq.main import app
    from giq.queue import JobQueue
    from giq.runner import Runner

    giq.queue._queue = JobQueue()
    giq.runner._runner = Runner(giq.queue._queue)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        yield ac


@pytest.mark.asyncio
async def test_a_rig_job_is_queued_through_the_job_api(client):
    response = await client.post(
        "/run",
        json={
            "modality": "mesh2rig",
            "model": "skintokens",
            "tasks": [{"id": "1", "glb_b64": "Z2xURg==", "seed": 3}],
        },
    )
    assert response.status_code == 200, response.text
    job = (await client.get(f"/jobs/{response.json()['job_id']}")).json()
    assert (job["modality"], job["model"]) == ("mesh2rig", "skintokens")


@pytest.mark.asyncio
async def test_the_recipe_serves_only_its_modality(client):
    response = await client.post(
        "/run",
        json={"modality": "image2mesh", "model": "skintokens", "tasks": [{"id": "1"}]},
    )
    assert response.status_code == 400
    assert "skintokens does not serve image2mesh" in response.text
