# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Job queue management."""

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from giq.models import JobRequest, JobStatus

logger = logging.getLogger(__name__)

# Chunks a streaming job may buffer before the worker blocks. A slow reader
# should throttle generation, not accumulate the whole answer in RAM — but the
# buffer has to be deep enough that ordinary network jitter never stalls the
# GPU. 512 chunks is a few seconds of tokens at ~60/s.
STREAM_BUFFER_CHUNKS = 512

# How long a full buffer may stay full before we conclude nobody is reading.
# Generous: a browser tab throttled in the background still drains eventually,
# and cancelling a nine-minute thought by mistake is the worse error.
STREAM_STALL_SECONDS = 60.0


@dataclass
class JobStream:
    """A live channel from a running job back to the client that asked for it.

    Streaming has to reach the client *while* the job runs, but a job is
    normally a submit-then-poll affair: the API waits on wait_for_job and the
    result appears at the end. Rather than let streaming callers bypass the
    queue — which is the single chokepoint where eviction, VRAM gating and
    accounting happen — the job still goes through the queue and carries this
    channel with it. The worker pushes chunks in as they arrive; the endpoint
    pushes them out as SSE.

    `cancelled` closes the loop the other way: when the client disconnects, the
    endpoint sets it and the worker stops generating. Without it an abandoned
    tab would hold the model for the full length of a thought.
    """

    queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=STREAM_BUFFER_CHUNKS)
    )
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    async def put(self, chunk: dict[str, Any]) -> None:
        """Hand one chunk to the reader, blocking only if it has fallen behind.

        A bounded buffer means a slow reader throttles generation, which is
        what we want. A *vanished* reader is different: it would block the
        worker forever and hold the model hostage, so a buffer that stays full
        for this long is treated as the reader being gone and cancels the job.
        """
        if self.cancelled.is_set():
            return
        try:
            self.queue.put_nowait(chunk)
            return
        except asyncio.QueueFull:
            pass
        try:
            await asyncio.wait_for(self.queue.put(chunk), timeout=STREAM_STALL_SECONDS)
        except TimeoutError:
            self.cancel()

    async def close(self) -> None:
        """Sentinel: no more chunks. Always sent, success or failure.

        Never blocks and never gives up — the reader is waiting on this to know
        the answer ended, so if the buffer is full we drop a chunk to make room
        for it. Losing a token beats leaving a request hanging.
        """
        try:
            self.queue.put_nowait(None)
            return
        except asyncio.QueueFull:
            pass
        try:
            self.queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - racing reader drained it
            pass
        try:
            self.queue.put_nowait(None)
        except asyncio.QueueFull:  # pragma: no cover - reader refilled it
            logger.warning("stream buffer full; reader may hang waiting for the end sentinel")

    def cancel(self) -> None:
        self.cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()


@dataclass
class Job:
    """A job in the queue."""

    job_id: str
    request: JobRequest
    status: JobStatus = JobStatus.pending
    # As stored, whatever the adapter returned: dicts (a result model's
    # model_dump(), a child's own dict, or the engine's raw chat completion).
    results: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    # Set only for streaming requests; None means the ordinary batch path.
    stream: JobStream | None = None

    @property
    def duration_ms(self) -> int | None:
        """Job duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class JobQueue:
    """Thread-safe job queue."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = asyncio.Lock()

    async def add(self, job: Job) -> int:
        """Add a job to the queue, return position."""
        async with self._lock:
            self._jobs[job.job_id] = job
            return len(self._jobs) - 1

    async def get(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def remove(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    async def get_pending(self) -> list[Job]:
        """Get all pending jobs in order."""
        return [j for j in self._jobs.values() if j.status == JobStatus.pending]

    async def get_all(self) -> list[Job]:
        """Get all jobs."""
        return list(self._jobs.values())

    async def get_next(self) -> Job | None:
        """Get next pending job."""
        for job in self._jobs.values():
            if job.status == JobStatus.pending:
                return job
        return None

    def __len__(self) -> int:
        return len(self._jobs)


# Global queue recipe
_queue = JobQueue()


def get_queue() -> JobQueue:
    """Get the global job queue."""
    return _queue
