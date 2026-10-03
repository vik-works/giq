# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""What the runner and the OpenAI layer need from an LLM served over HTTP.

llama-server and ``vllm serve`` are different engines behind one contract: a
process giq spawns on one card, an OpenAI-compatible endpoint on a loopback
port, and a way to ask how busy it is. The runner used to test for
``LlamaCppAdapter`` by class wherever it meant "a chat server"; it now tests for
this base, so a second engine is an adapter rather than a second set of
branches through the scheduler (ADR-002, D4).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import httpx

    from giq.models import JobResult
    from giq.queue import JobStream


class StartError(RuntimeError):
    """A server giq spawned did not come up: it exited, or outlasted its start budget.

    Deliberately not a ``TimeoutError``. The runner reads ``TimeoutError`` as
    a *job* that ran past its limit and words the failure that way, so a
    llama-server that died at spawn was reported as "Job timed out after
    300s" thirty seconds in. A start that fails says what failed.
    """


def open_engine_log(path: Path) -> IO[str]:
    """The log a server's new run writes to, with the previous run kept as ``.1``.

    A start truncated the log, and the start that matters most is the one
    after a crash: giq restarts a resident engine on its own, and the new run
    wiped the stack trace that said why the old one died. One previous run is
    kept rather than all of them, so a crash-looping engine cannot fill the
    disk.
    """
    if path != Path(os.devnull) and path.exists() and path.stat().st_size:
        path.replace(path.with_name(path.name + ".1"))
    return open(path, "w", encoding="utf-8")


@dataclass(frozen=True)
class Concurrency:
    """How many requests an engine serves at once, derived from its parameters (D8).

    ``per_request_context`` is the most one request can hold: llama.cpp
    without a unified KV splits its context evenly across slots, vllm's
    paged KV lets any one request use the whole window. ``shared_kv`` says
    which of the two the figure means.
    """

    max_parallel: int
    per_request_context: int
    shared_kv: bool


class ServedLLM(ABC):
    """An LLM engine giq runs as a child process and talks to over HTTP."""

    # The engine name, as giq.engines declares it.
    engine: ClassVar[str]

    # Whether the runner sizes this model's resident lane from the engine's
    # own concurrency (D8) instead of the registry's lane width. vllm does:
    # its max_num_seqs is the number of requests the server really runs at
    # once. llama.cpp does not yet — its lane width and its -np still come
    # from two tables, and aligning them changes dispatch for every built-in
    # model, which belongs with the move to recipe files.
    lanes_from_engine: ClassVar[bool] = False

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @property
    @abstractmethod
    def pid(self) -> int | None: ...

    @property
    @abstractmethod
    def estimated_vram_gb(self) -> float: ...

    @property
    @abstractmethod
    def base_url(self) -> str: ...

    @abstractmethod
    def concurrency(self) -> Concurrency: ...

    @abstractmethod
    def http_timeout(self) -> httpx.Timeout: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def active_slot_count(self) -> int | None:
        """Requests the server is working on right now; None when unknown."""

    @abstractmethod
    async def chat_completion(self, request_body: dict) -> dict: ...

    @abstractmethod
    async def chat_completion_stream(self, request_body: dict, stream: JobStream) -> dict: ...

    @abstractmethod
    async def run_batch(self, tasks: list[dict], params: dict | None = None) -> list[JobResult]: ...


def engine_for(model: str) -> str:
    """The engine that serves an LLM recipe: its own, llama.cpp when unknown."""
    from giq.registry import get_recipe

    recipe = get_recipe(model)
    return recipe.engine if recipe is not None else "llama.cpp"


def context_size(model: str) -> int:
    """The context window an LLM is served with, whichever engine runs it."""
    from giq import plugins
    from giq.registry import get_recipe

    recipe = get_recipe(model)
    engine = plugins.engine(recipe.engine) if recipe is not None else None
    if recipe is not None and engine is not None and engine.context is not None:
        served = engine.context(recipe.params)
        if served is not None:
            return served
    from giq.adapters.llama_cpp import DEFAULT_CTX_SIZE, MODEL_CTX_SIZE

    return MODEL_CTX_SIZE.get(model, DEFAULT_CTX_SIZE)
