# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""giq-vllm: the vllm engine for LLMs (ADR-004).

vllm serves Hugging Face checkpoints with continuous batching and MTP
speculation, from its own interpreter (envs/vllm): it pins its torch,
transformers and fastapi, so it runs as a separate server process, like
llama-server. This package registers the engine, its recipe parameters,
`giq prepare vllm` and its built-in recipes; nothing of vllm itself is
imported into giq's interpreter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from giq.paths import env_python
from giq.plugin import API_VERSION, AdapterContext, Binary, Engine, Plugin

__version__ = "0.6.1"


def _vllm(ctx: AdapterContext):
    import logging

    from giq_vllm.adapter import VllmAdapter, VllmConfig

    if ctx.model_path:
        # A path override names a GGUF for llama-server; a vllm model's
        # weights are part of its declaration.
        logging.getLogger("giq.runner").warning(
            f"llm/{ctx.recipe}: model_path ignored, vllm serves its declared weights"
        )
    return VllmAdapter(config=VllmConfig(model=ctx.recipe, device=ctx.device))


def _vllm_sweep(ports: set[int]) -> None:
    # A vllm server runs in a transient systemd scope, which outlives a giq
    # that died without stopping it; the scope takes its engine core down
    # with it, which a pattern on the API server's command line would miss.
    from giq_vllm.adapter import scope_unit, stop_scope

    for port in sorted(ports):
        stop_scope(scope_unit(port))


def _vllm_prepare(argv: list[str]) -> int:
    from giq_vllm.adapter import cli_prepare

    return cli_prepare(argv)


def _vllm_check(recipe: Any, capability: str) -> str | None:
    # vllm's own floor: its kernels need Volta (compute capability 7.0) or newer.
    try:
        major, minor = (int(x) for x in capability.split(".")[:2])
    except ValueError:
        return None
    if (major, minor) < (7, 0):
        return f"vllm needs compute capability 7.0 or newer; this card has {capability}"
    return None


def _vllm_plugin() -> Plugin:
    from giq_vllm.params import (
        ENGINE_PROFILES,
        VLLM_REQUEST_DEFAULTS,
        VllmParams,
        derived_vram,
        vllm_consistent,
    )

    return Plugin(
        name="giq-vllm",
        api_version=API_VERSION,
        version=__version__,
        engines=(
            Engine(
                "vllm",
                # `vllm serve` runs from this env's console script, next to the
                # interpreter; the interpreter is what is declared and probed,
                # because asking the script for its version imports torch and
                # takes longer than the probe allows.
                Binary(
                    lambda: env_python("vllm"),
                    env="GIQ_VLLM_PYTHON",
                    version_args=(
                        "-c",
                        "import importlib.metadata as m; "
                        "print('vllm', m.version('vllm'), 'torch', m.version('torch'), "
                        "'flashinfer', m.version('flashinfer-python'))",
                    ),
                    detail="vllm serve, one process per model on its card's port (envs/vllm)",
                ),
                params=VllmParams,
                profiles=ENGINE_PROFILES["vllm"],
                request_defaults=VLLM_REQUEST_DEFAULTS,
                validate=vllm_consistent,
                derive_vram=derived_vram,
                # max_num_seqs is how many requests vllm really runs at once.
                lanes=lambda params: params.max_num_seqs,
                start_budget=lambda recipe: float(recipe.params.ready_timeout),
                stale_pattern=r"vllm serve .*--port {port}\b",
                sweep=_vllm_sweep,
                prepare=_vllm_prepare,
                check=_vllm_check,
                # vllm reserves its weights, the KV budget and its buffers at
                # start and never grows past them; the recipe's figure is all
                # of it, so a load needs little on top (the same 0.5 GB a
                # resident reload gets). The scaled 2 GB kept a 29.7 GB recipe
                # from ever loading on demand on a 32 GB card.
                vram_margin=0.5,
                # The context it serves is the recipe's max_model_len.
                context=lambda params: params.max_model_len,
                # No server-level switch: the chat template decides, and a caller
                # turns thinking off per request with chat_template_kwargs.
                reasoning=lambda recipe: "template",
            ),
        ),
        adapters={("vllm", "llm"): _vllm},
        recipes=Path(__file__).parent / "recipes",
    )


plugin = _vllm_plugin()
