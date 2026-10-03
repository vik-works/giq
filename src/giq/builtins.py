# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq's own registrations, through the same contract a plugin uses (ADR-004).

Core registers the `llm` modality, the llama.cpp engine and the
`transformers` engine name. Everything else is a plugin: the curated ones
are packages of this repository's workspace (plugins/), installed together
by `giq-defaults` (D7).

`transformers` is core's although core runs no transformers model: OCR and
depth both name it, and two plugins registering one engine would clash. It
declares only that a recipe runs in giq's own interpreter. The library
itself is a dependency of the plugins that use it.

Every adapter import is deferred to the factory, so building the registry
imports no engine and no ML library.
"""

from __future__ import annotations

from typing import Any

from giq.paths import engine_binary
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Modality, Plugin, SmokeTest

# --- core: llm, llama.cpp, transformers ----------------------------------------


def _llamacpp_start_budget(recipe: Any) -> float:
    from giq.adapters.llama_cpp import DEFAULT_READY_TIMEOUT, MODEL_READY_TIMEOUT

    return float(MODEL_READY_TIMEOUT.get(recipe.name, DEFAULT_READY_TIMEOUT))


def _llamacpp(ctx: AdapterContext):
    from giq.adapters.llama_cpp import LlamaCppAdapter, LlamaCppConfig

    return LlamaCppAdapter(
        config=LlamaCppConfig(model=ctx.recipe, model_path=ctx.model_path, device=ctx.device)
    )


def _llm_smoke() -> SmokeTest:
    return SmokeTest(
        tasks=[
            {
                "id": "smoke-llm",
                "user": "Reply with exactly: PONG",
                "params": {"max_tokens": 8, "enable_thinking": False},
            }
        ],
        timeout=120.0,
        summary=lambda result: {"output": (result.get("output") or "")[:200]},
    )


def _core() -> Plugin:
    from giq.recipes.schema import LlamaCppParams, llamacpp_consistent

    return Plugin(
        name="giq",
        api_version=API_VERSION,
        engines=(
            Engine(
                "llama.cpp",
                Binary(
                    lambda: engine_binary("llama.cpp", "llama-server"),
                    env="GIQ_LLAMA_BINARY",
                    detail="llama-server, one process per model on its card's port",
                ),
                params=LlamaCppParams,
                request_defaults=None,
                validate=llamacpp_consistent,
                start_budget=_llamacpp_start_budget,
                # \b: --port 8089 must not also match --port 80891.
                stale_pattern=r"llama-server .*--port {port}\b",
            ),
            Engine("transformers", aliases=("transformers-4.57",)),
        ),
        modalities=(
            # llama-server's 4 slots: a client fanning out a few chat calls
            # at once gets them served in parallel.
            Modality(
                "llm",
                label="LLM",
                icon="chat",
                lane_width=4,
                # llama.cpp's vision projector.
                parts=frozenset({"mmproj"}),
                smoke_test=_llm_smoke,
            ),
        ),
        adapters={("llama.cpp", "llm"): _llamacpp},
        routers=(
            "giq.api.router:router",
            "giq.api.openai_compat:router",
            "giq.api.openai_responses:router",
            "giq.api.stats_api:router",
            "giq.api.recipes_api:router",
        ),
    )


class _Plugins:
    """Built lazily: the declarations import the recipe schema, which itself
    looks names up in the registry."""

    def __iter__(self):
        return iter((_core(),))


PLUGINS = _Plugins()
