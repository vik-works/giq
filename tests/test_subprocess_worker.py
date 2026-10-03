# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SubprocessAdapter base class.

No GPU required. Uses tiny inline Python scripts as stub children, invoked via
``sys.executable -c <script>``. The behaviors exercised are IPC framing,
startup-readiness detection, teardown escalation, and stderr capture.
"""

import asyncio
import logging
import sys

import pytest

from giq.adapters._subprocess import (
    SHUTDOWN_GRACE_SECONDS,
    SubprocessAdapter,
    SubprocessWorkerDied,
    SubprocessWorkerError,
    SubprocessWorkerStartError,
)

# ---- Test helper: stub worker that runs an inline script --------------------


class _StubWorker(SubprocessAdapter):
    """Run arbitrary Python via ``python -u -c <script>`` instead of ``-m``."""

    def __init__(self, script: str):
        super().__init__(config=None)
        self._script = script

    def _command(self) -> list[str]:
        return [sys.executable, "-u", "-c", self._script]

    @property
    def estimated_vram_gb(self) -> float:
        return 0.0


# ---- Inline child scripts ---------------------------------------------------
#
# Note on escaping: these strings are parsed by the OUTER Python (this test
# file), then the resulting string is executed as source by the INNER Python
# (the subprocess). "\\n" in the outer source becomes "\n" in the inner source,
# which the inner Python parses to a literal newline character.


SCRIPT_READY_ECHO = """
import sys, json
sys.stdout.write(json.dumps({"type":"ready"})+"\\n")
sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        break
    if msg.get("type") == "run_batch":
        sys.stdout.write(json.dumps({"type":"results","results":msg["tasks"]})+"\\n")
        sys.stdout.flush()
"""


SCRIPT_ERROR_BEFORE_READY = """
import sys, json
sys.stdout.write(json.dumps({"type":"error","message":"nope","traceback":"stk"})+"\\n")
sys.stdout.flush()
sys.exit(1)
"""


SCRIPT_EXIT_IMMEDIATELY = """
import sys
sys.stderr.write("boom\\n")
sys.exit(2)
"""


SCRIPT_ERROR_ON_RUN_BATCH = """
import sys, json
sys.stdout.write(json.dumps({"type":"ready"})+"\\n")
sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        break
    if msg.get("type") == "run_batch":
        sys.stdout.write(json.dumps({"type":"error","message":"runtime","traceback":"tb"})+"\\n")
        sys.stdout.flush()
"""


SCRIPT_DIE_MIDBATCH = """
import sys, json
sys.stdout.write(json.dumps({"type":"ready"})+"\\n")
sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "run_batch":
        sys.exit(7)
"""


SCRIPT_IGNORE_SIGTERM = """
import sys, json, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
sys.stdout.write(json.dumps({"type":"ready"})+"\\n")
sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        time.sleep(300)
"""


SCRIPT_SLOW_READY = """
import time
time.sleep(60)
"""


SCRIPT_STDERR_NOISY = """
import sys, json
sys.stderr.write("line1\\nline2\\nline3\\n")
sys.stderr.flush()
sys.stdout.write(json.dumps({"type":"ready"})+"\\n")
sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("type") == "shutdown":
        break
"""


# ---- Tests ------------------------------------------------------------------


async def test_start_waits_for_ready():
    w = _StubWorker(SCRIPT_READY_ECHO)
    await w.start()
    try:
        assert w.is_running is True
        assert w.is_ready is True
    finally:
        await w.stop()
    assert w.is_running is False
    assert w.is_ready is False


async def test_start_error_envelope_before_ready():
    w = _StubWorker(SCRIPT_ERROR_BEFORE_READY)
    with pytest.raises(SubprocessWorkerStartError) as excinfo:
        await w.start()
    assert "nope" in str(excinfo.value)
    assert w.is_running is False


async def test_start_exit_before_ready():
    w = _StubWorker(SCRIPT_EXIT_IMMEDIATELY)
    with pytest.raises(SubprocessWorkerStartError) as excinfo:
        await w.start()
    msg = str(excinfo.value).lower()
    assert "exited" in msg or "returncode" in msg


async def test_run_batch_roundtrip():
    w = _StubWorker(SCRIPT_READY_ECHO)
    await w.start()
    try:
        tasks = [{"id": "a", "val": 1}, {"id": "b", "val": 2}]
        results = await w.run_batch(tasks, params={"x": 9})
        assert results == tasks
    finally:
        await w.stop()


async def test_run_batch_error_envelope():
    w = _StubWorker(SCRIPT_ERROR_ON_RUN_BATCH)
    await w.start()
    try:
        with pytest.raises(SubprocessWorkerError) as excinfo:
            await w.run_batch([{"id": "t"}])
        assert "runtime" in str(excinfo.value)
    finally:
        await w.stop()


async def test_run_batch_child_dies_midbatch():
    w = _StubWorker(SCRIPT_DIE_MIDBATCH)
    await w.start()
    try:
        with pytest.raises(SubprocessWorkerDied) as excinfo:
            await w.run_batch([{"id": "t"}])
        assert excinfo.value.returncode == 7
    finally:
        await w.stop()


async def test_run_batch_before_start_raises():
    w = _StubWorker(SCRIPT_READY_ECHO)
    with pytest.raises(RuntimeError, match="not ready"):
        await w.run_batch([])


async def test_stop_graceful_is_fast():
    w = _StubWorker(SCRIPT_READY_ECHO)
    await w.start()
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await w.stop()
    elapsed = loop.time() - t0
    assert elapsed < SHUTDOWN_GRACE_SECONDS + 1.0, (
        f"graceful stop took {elapsed:.2f}s, expected < {SHUTDOWN_GRACE_SECONDS + 1.0}s"
    )
    assert w.is_running is False


async def test_stop_sigterm_escalates_to_sigkill(caplog):
    w = _StubWorker(SCRIPT_IGNORE_SIGTERM)
    await w.start()
    with caplog.at_level(logging.WARNING, logger="giq.adapters._subprocess"):
        await w.stop()
    assert any("SIGKILL" in r.message for r in caplog.records), (
        f"Expected SIGKILL escalation warning in logs, got: {[r.message for r in caplog.records]}"
    )
    assert w.is_running is False


async def test_stderr_forwarded_to_logger(caplog):
    w = _StubWorker(SCRIPT_STDERR_NOISY)
    with caplog.at_level(logging.DEBUG, logger="giq.adapters._subprocess"):
        await w.start()
        await asyncio.sleep(0.2)  # give drain task a moment
        await w.stop()
    stderr_msgs = [r.message for r in caplog.records if "stderr" in r.message]
    assert any("line1" in m for m in stderr_msgs), (
        f"Expected stderr lines in logger, got: {stderr_msgs}"
    )


async def test_cancellation_during_start_reaps_child():
    w = _StubWorker(SCRIPT_SLOW_READY)
    task = asyncio.create_task(w.start())
    await asyncio.sleep(0.3)  # let subprocess spawn + begin its sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # After cancellation, start()'s except-BaseException should have reaped.
    assert w.is_running is False
    assert w._process is None


async def test_estimated_vram_gb_raises_on_base():
    class NoVramWorker(SubprocessAdapter):
        child_module = "x"

    w = NoVramWorker(config=None)
    with pytest.raises(NotImplementedError):
        _ = w.estimated_vram_gb


async def test_command_not_set_raises():
    class EmptyWorker(SubprocessAdapter):
        pass  # no child_module set

    w = EmptyWorker(config=None)
    with pytest.raises(NotImplementedError, match="child_module"):
        await w.start()


async def test_ipc_child_loop_integration():
    """End-to-end: parent spawns child that uses run_ipc_child_loop helper."""
    script = (
        "import sys\n"
        # run_ipc_child_loop lives in giq_child, giq's stdlib-only child side
        "sys.path.insert(0, " + repr(str(_repo_src_path())) + ")\n"
        "from giq_child import run_ipc_child_loop\n"
        "def echo(tasks, params):\n"
        "    return [{'id': t.get('id'), 'echoed': True} for t in tasks]\n"
        "run_ipc_child_loop(echo)\n"
    )
    w = _StubWorker(script)
    await w.start()
    try:
        results = await w.run_batch([{"id": "x"}, {"id": "y"}])
        assert results == [
            {"id": "x", "echoed": True},
            {"id": "y", "echoed": True},
        ]
    finally:
        await w.stop()


def _repo_src_path():
    """Return absolute path to src/ so inline-script children can import giq."""
    import pathlib

    # tests/test_subprocess_worker.py → repo/
    here = pathlib.Path(__file__).resolve().parent.parent
    return here / "src"


def test_the_child_side_needs_nothing_but_the_standard_library(tmp_path):
    """A plugin's child on its own interpreter installs giq_child and nothing
    of giq (ADR-004 D4). Imported with no site-packages at all (-S) and
    nothing else on the path, it still loads, and pulls in no giq module."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import giq_child

    alone = tmp_path / "path"
    shutil.copytree(Path(giq_child.__file__).parent, alone / "giq_child")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import giq_child; "
        "print(sorted(m for m in sys.modules if m.startswith('giq')))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(alone)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "['giq_child']"
