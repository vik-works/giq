# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test config."""

import os
import tempfile

import pytest

# The suite checks the built-in catalog and giq's defaults, so whatever
# configures *this* giq in the shell that runs it is dropped, not merely
# defaulted: an operator's GIQ_HOME, config, recipes directory, residents or
# token would change what is under test, and did — with an operator's config
# and recipes in the environment, a dozen policy, runner and registry tests
# failed against a main that was fine. What only locates things on the
# machine (binaries, model directories, interpreters) stays, for the GPU
# integration tests.
for name in (
    "GIQ_HOME",
    "GIQ_CONFIG",
    "GIQ_RECIPES_DIR",
    "GIQ_INSTANCES_DIR",
    "GIQ_DATA_DIR",
    "GIQ_CACHE_DIR",
    "GIQ_GPU_DEVICE",
    "GIQ_TOKEN",
    "GIQ_HOST",
    "GIQ_PORT",
    "GIQ_BOUND_HOST",
):
    os.environ.pop(name, None)

# Keep test telemetry out of the production stats DB. Set before any giq
# import can construct the StatsRecorder singleton — unit tests exercising
# eviction/jobs write through it.
os.environ["GIQ_STATS_DB"] = os.path.join(tempfile.mkdtemp(prefix="giq-test-stats-"), "stats.db")
# The in-flight log would otherwise land in the checkout's data/ directory.
os.environ["GIQ_INFLIGHT_LOG"] = os.path.join(
    tempfile.mkdtemp(prefix="giq-test-inflight-"), "inflight.log"
)
# Operator recipe files would otherwise come from ~/.config/giq/recipes and
# change the catalog under test; the built-ins are what the suite checks.
os.environ["GIQ_RECIPES_DIR"] = tempfile.mkdtemp(prefix="giq-test-recipes-")

# A typical keep-warm set: an LLM, the audio stack and voiceprints, 14.1 GB.
# Nothing is kept warm out of the box, so tests of residency, pinning and the
# card budget declare the set they exercise, as an operator does.
CLASSIC_RESIDENTS = ("gemma-4-12b", "whisper-large-v3", "ecapa-tdnn")


@pytest.fixture
def config_residents(monkeypatch):
    """Set config.yaml's ``residents:`` for one test; call with recipe names."""
    from giq.config import get_config

    def declare(*names: str) -> None:
        monkeypatch.setattr(get_config(), "residents", list(names))

    return declare
