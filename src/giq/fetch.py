# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Fetching a recipe's weights (ADR-005 D4).

A fetch carries out the transfers its plan listed, one child process each
(:mod:`giq._fetch_child`): the Hub library downloads there and reports its
byte counters as JSON lines, and cancelling ends the child. The partial files
stay, so the next fetch of the same recipe resumes.

Fetches use no GPU, so they never touch the job queue. They run one at a
time, in the order they were asked for: two downloads only share the same
bandwidth, and one at a time keeps a checkpoint two recipes share from
being fetched twice. Each transfer checks again whether its files arrived
meanwhile and skips them if so.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from giq.plan import Plan, Transfer

logger = logging.getLogger(__name__)

State = Literal["queued", "running", "done", "failed", "cancelled"]


@dataclass
class Download:
    """One recipe's fetch, as /downloads reports it."""

    id: str
    recipe: str
    transfers: list[Transfer]
    state: State = "queued"
    bytes_total: int = 0
    bytes_done: int = 0
    error: str | None = None
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    # The repository being fetched now.
    current: str | None = None
    _proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _cancelled: bool = field(default=False, repr=False)

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recipe": self.recipe,
            "state": self.state,
            "bytes_total": self.bytes_total,
            "bytes_done": self.bytes_done,
            "current": self.current,
            "error": self.error,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _arrived(move: Transfer) -> bool:
    if move.dest is not None:
        return move.dest.exists()
    return False


def _child_env() -> dict[str, str]:
    from giq.paths import cache_env

    # The same HF home the children load from, or the files land where
    # nobody looks for them. HF_HUB_OFFLINE keeps the model children from
    # downloading behind giq's back (the systemd unit sets it); a fetch is
    # the one download someone asked for, so it is lifted here only.
    env: dict[str, str] = {**os.environ, **cache_env()}
    env.pop("HF_HUB_OFFLINE", None)
    return env


def child_command(move: Transfer, report: bool = False) -> list[str]:
    """The child for one transfer; with ``report`` it prints its progress
    as JSON lines instead of drawing the Hub library's bars."""
    spec = {
        "repo": move.repo,
        "revision": move.revision,
        "file": move.file,
        "dest": str(move.dest) if move.dest else None,
        "report": report,
    }
    return [sys.executable, "-m", "giq._fetch_child", json.dumps(spec)]


def _child_error(stderr: bytes) -> str:
    for line in reversed(stderr.decode(errors="replace").splitlines()):
        try:
            return str(json.loads(line)["error"])
        except (ValueError, KeyError, TypeError):
            continue
    tail = stderr.decode(errors="replace").strip().splitlines()
    return tail[-1] if tail else "the fetch process failed"


class Downloads:
    """Every fetch this service ran or runs, and the one worker running them."""

    def __init__(self) -> None:
        self._all: dict[str, Download] = {}
        self._queue: asyncio.Queue[Download] | None = None
        self._worker: asyncio.Task | None = None

    def all(self) -> list[Download]:
        return sorted(self._all.values(), key=lambda d: d.queued_at, reverse=True)

    def get(self, download_id: str) -> Download | None:
        return self._all.get(download_id)

    def start(self, plan: Plan) -> Download:
        """Queue ``plan``'s transfers; the recipe's running fetch if it has one."""
        for d in self._all.values():
            if d.recipe == plan.recipe and d.active:
                return d
        download = Download(
            id=uuid.uuid4().hex[:12],
            recipe=plan.recipe,
            transfers=plan.transfers,
            bytes_total=plan.download_bytes,
        )
        self._all[download.id] = download
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_all(), name="giq-downloads")
        self._queue.put_nowait(download)
        logger.info(f"fetch {download.id}: {plan.recipe}, {plan.download_bytes / 1e9:.1f} GB")
        return download

    def cancel(self, download_id: str) -> Download | None:
        download = self._all.get(download_id)
        if download is None or not download.active:
            return download
        download._cancelled = True
        if download._proc is not None and download._proc.returncode is None:
            download._proc.kill()
        if download.state == "queued":
            download.state = "cancelled"
            download.finished_at = time.time()
        return download

    async def _run_all(self) -> None:
        assert self._queue is not None
        while True:
            download = await self._queue.get()
            if download.state != "queued":
                continue
            try:
                await self._run(download)
            except Exception as e:  # one fetch failing must not stop the queue
                logger.exception(f"fetch {download.id} ({download.recipe}) failed")
                download.state, download.error = "failed", str(e)
                download.finished_at = time.time()

    async def _run(self, download: Download) -> None:
        from giq.storage import missing

        download.state, download.started_at = "running", time.time()
        done = 0
        for move in download.transfers:
            if download._cancelled:
                break
            if _arrived(move):
                done += move.bytes
                continue
            download.current = move.repo
            ok = await self._transfer(download, move, done)
            if not ok:
                break
            done = max(done + move.bytes, download.bytes_done)
            download.bytes_done = done
        download.current = None
        download.finished_at = time.time()
        if download._cancelled:
            download.state = "cancelled"
            logger.info(f"fetch {download.id} ({download.recipe}) cancelled")
            return
        if download.state == "failed":
            return
        gone = await asyncio.to_thread(missing, download.recipe)
        if gone:
            download.state = "failed"
            download.error = "fetched, but not every file is there: " + ", ".join(
                str(loc.path or loc.repo) for loc in gone
            )
            return
        download.state = "done"
        download.bytes_done = max(download.bytes_done, download.bytes_total)
        logger.info(f"fetch {download.id} ({download.recipe}) done")

    async def _transfer(self, download: Download, move: Transfer, before: int) -> bool:
        proc = await asyncio.create_subprocess_exec(
            *child_command(move, report=True),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(),
        )
        download._proc = proc
        assert proc.stdout is not None and proc.stderr is not None
        stderr = asyncio.create_task(proc.stderr.read())
        async for line in proc.stdout:
            try:
                received = int(json.loads(line)["bytes"])
            except (ValueError, KeyError, TypeError):
                continue
            download.bytes_done = before + received
        await proc.wait()
        download._proc = None
        if proc.returncode == 0 or download._cancelled:
            return proc.returncode == 0
        download.state = "failed"
        download.error = f"{move.repo}: {_child_error(await stderr)}"
        logger.error(f"fetch {download.id} ({download.recipe}) failed: {download.error}")
        return False


_downloads: Downloads | None = None


def get_downloads() -> Downloads:
    global _downloads
    if _downloads is None:
        _downloads = Downloads()
    return _downloads
