# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""The child side of giq's subprocess protocol, on the standard library alone.

A model that runs in a child process speaks one JSON object per line
(UTF-8, ``\\n``-terminated) with the giq process that spawned it:

  Child → Parent:
    {"type": "ready"}
    {"type": "results", "results": [...]}
    {"type": "error",   "message": "...", "traceback": "..."}

  Parent → Child:
    {"type": "run_batch", "tasks": [...], "params": {...}|null}
    {"type": "shutdown"}

It is its own top-level package, not part of ``giq``, so that a plugin's
child on an interpreter of its own (ADR-004 D4) needs nothing of giq but
this: install ``giq-child`` there, never giq's source tree on PYTHONPATH.
Nothing here may import beyond the standard library.

A child's entry point::

    from giq_child import reserve_ipc_stdout, run_ipc_child_loop, write_startup_error

    def main():
        reserve_ipc_stdout()            # first, before any noisy import
        try:
            model = load()
        except Exception as e:
            write_startup_error(str(e), traceback.format_exc())
            sys.exit(1)
        run_ipc_child_loop(lambda tasks, params: [run(model, t) for t in tasks])
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any

# --- Child-side IPC stdout reservation -----------------------------------
#
# Problem: some imports (notably transformers and other libraries' startup banners) print
# chatter to stdout during pipeline load. On the child that corrupts the JSON
# IPC channel — the parent reads garbage before the ``{"type":"ready"}`` line.
#
# Fix: each child entry point calls ``reserve_ipc_stdout()`` as its first
# action, BEFORE any noisy imports. That duplicates the real stdout fd for
# JSON writes and re-points ``sys.stdout`` at stderr. After this, any
# ``print()`` or library chatter lands in the parent's stderr drain (which
# just logs at DEBUG), while JSON frames go to the preserved pipe.
#
# This global lives in the child process only; the parent never calls
# ``reserve_ipc_stdout()`` so its ``_IPC_STDOUT_FD`` stays None.

_IPC_STDOUT_FD: int | None = None


def reserve_ipc_stdout() -> None:
    """Capture real stdout fd for IPC; redirect ``sys.stdout`` to stderr.

    Call this as the **first line** of a child entry point's ``main()``,
    before any imports that may print (torch, transformers, etc.).
    Safe to call more than once — idempotent.
    """
    global _IPC_STDOUT_FD
    if _IPC_STDOUT_FD is None:
        _IPC_STDOUT_FD = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _write_ipc_line(msg: dict[str, Any]) -> None:
    """Write a single JSON line to the preserved IPC stdout.

    Falls back to regular sys.stdout if ``reserve_ipc_stdout`` was not called
    (useful for tests that use the child helper without redirection).
    """
    data = json.dumps(msg).encode("utf-8") + b"\n"
    if _IPC_STDOUT_FD is not None:
        os.write(_IPC_STDOUT_FD, data)
    else:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


# --- Child-side helper ---------------------------------------------------


def run_ipc_child_loop(
    on_run_batch: Callable[[list[dict[str, Any]], dict[str, Any] | None], list[dict[str, Any]]],
) -> None:
    """Synchronous IPC loop for child processes.

    Call this from a child entry point **after** the model/pipeline is loaded.
    Writes ``{"type":"ready"}`` to signal the parent, then reads JSON lines
    from stdin and dispatches ``run_batch`` to ``on_run_batch``.

    ``on_run_batch(tasks, params)`` returns a ``list[dict]`` (typically
    ``[result.model_dump() for result in results]``). Any exception is caught
    and serialized as a batch-level error envelope.

    Returns (i.e. the child exits) on ``{"type":"shutdown"}`` or stdin EOF.
    """
    import traceback

    # Signal ready to parent.
    _write_ipc_line({"type": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _write_ipc_line({"type": "error", "message": f"bad JSON from parent: {line!r}"})
            continue

        mtype = msg.get("type")
        if mtype == "shutdown":
            return
        if mtype == "run_batch":
            try:
                results = on_run_batch(msg.get("tasks", []), msg.get("params"))
                _write_ipc_line({"type": "results", "results": results})
            except Exception as e:
                _write_ipc_line(
                    {
                        "type": "error",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
        else:
            _write_ipc_line({"type": "error", "message": f"unknown message type: {mtype!r}"})


def write_startup_error(message: str, tb: str = "") -> None:
    """Write an ``error`` envelope to IPC stdout before ``run_ipc_child_loop``.

    Use in a child's top-level ``try/except`` that wraps model loading — if
    loading fails, the parent's ``_wait_for_ready`` sees this and raises
    ``SubprocessWorkerStartError`` with the structured info.
    """
    try:
        _write_ipc_line({"type": "error", "message": message, "traceback": tb})
    except Exception:
        pass
