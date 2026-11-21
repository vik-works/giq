# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Usage statistics: persistent record of jobs, GPU events, and VRAM samples.

SQLite (WAL) at GIQ_STATS_DB (default: <state dir>/stats.db, see giq.paths). All writes
hop off the event loop via asyncio.to_thread — a blocked loop OOM-killed giq
once already (see runner dispatch-claim comment); never risk it for telemetry.

On first init (empty jobs table) the recorder backfills from the inflight
log (GIQ_INFLIGHT_LOG, default <state dir>/inflight.log), which has one
start/end pair per job — the dashboard is born with whatever history exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from giq.paths import inflight_log, stats_db
from giq.privacy import safe_detail

logger = logging.getLogger(__name__)


def _sum_tokens(results: list[dict] | None) -> tuple[int | None, int | None]:
    """Aggregate token usage across a job's results.

    Handles both result shapes: task-path LLMResult dumps (tokens_in/tokens_out)
    and the tool-call path's raw OpenAI response ({"usage": {...}}).
    """
    tin = tout = None
    for r in results or []:
        if not isinstance(r, dict):
            continue
        usage = r.get("usage") if isinstance(r.get("usage"), dict) else {}
        i = r.get("tokens_in", usage.get("prompt_tokens"))
        o = r.get("tokens_out", usage.get("completion_tokens"))
        if i is not None:
            tin = (tin or 0) + i
        if o is not None:
            tout = (tout or 0) + o
    return tin, tout


VRAM_SAMPLE_INTERVAL_SECONDS = 30.0
VRAM_RETENTION_DAYS = 14

# --- GPU eras --------------------------------------------------------------
# gpu_samples is high-volume and purged after VRAM_RETENTION_DAYS; gpu_eras is
# the permanent record of *which cards this machine has had*, so a swap stays
# answerable years later. Without it the card identity vanishes with the
# samples.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,              -- completion time (unix epoch)
    job_id TEXT,
    modality TEXT NOT NULL,        -- the kind of job (ADR-003; `worker` before)
    recipe TEXT NOT NULL,          -- the recipe's name (`model` before)
    status TEXT NOT NULL,          -- completed | failed
    queue_wait_ms INTEGER,
    run_ms INTEGER,
    tasks INTEGER,
    error TEXT,
    tokens_in INTEGER,
    tokens_out INTEGER,
    gpu_uuid TEXT,
    app TEXT                       -- ADR-006: the app it is counted under; NULL = before apps
);
-- NOTE: no index on gpu_uuid here. executescript runs before the ALTER TABLE
-- migration below, and on a pre-gpu_uuid database CREATE TABLE IF NOT EXISTS
-- is a no-op — so indexing that column here fails with "no such column".
-- It is created after the migration instead.
CREATE INDEX IF NOT EXISTS idx_jobs_ts ON jobs(ts);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,            -- evict | reload | oom | start | stop
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS vram_samples (
    ts REAL PRIMARY KEY,
    used_gb REAL NOT NULL,
    free_gb REAL NOT NULL,
    active_worker TEXT,
    residents_ready INTEGER
);

-- Permanent (never purged): one row per card this machine has ever had.
CREATE TABLE IF NOT EXISTS gpu_eras (
    gpu_uuid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    total_gb REAL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL
);

-- Operator overrides of a model's residency policy. Absent row = the
-- registry default. Never purged: this is intent, not telemetry.
CREATE TABLE IF NOT EXISTS model_policy (
    worker TEXT NOT NULL,
    model TEXT NOT NULL,
    policy TEXT NOT NULL,          -- pinned | auto | off
    reason TEXT,
    updated_at REAL NOT NULL,
    device TEXT,                   -- GPU UUID this model is bound to; NULL = unbound
    PRIMARY KEY (worker, model)
);

-- A recipe's residency policy and card binding, keyed by recipe name
-- (ADR-003). Absent row = the recipe's default. Never purged: this is intent,
-- not telemetry. model_policy above is its predecessor, keyed by
-- (worker, model); it is read once into this table and then left alone, so a
-- rollback to an older giq finds its own rows where it left them.
CREATE TABLE IF NOT EXISTS recipe_policy (
    recipe TEXT PRIMARY KEY,
    policy TEXT NOT NULL,          -- pinned | auto | off
    reason TEXT,
    updated_at REAL NOT NULL,
    device TEXT                    -- GPU UUID this recipe is bound to; NULL = unbound
);

CREATE TABLE IF NOT EXISTS gpu_samples (
    ts REAL NOT NULL,
    gpu_uuid TEXT NOT NULL,
    used_gb REAL NOT NULL,
    total_gb REAL NOT NULL,
    temp_c REAL,
    power_w REAL,
    util_pct REAL,
    PRIMARY KEY (ts, gpu_uuid)
);
"""


def _migrate_policies(conn: sqlite3.Connection) -> None:
    """Copy model_policy into recipe_policy once (ADR-003).

    Rows were keyed (worker, model); a recipe's name is the old model name.
    Two old rows can become one recipe (text2image/flux_klein and
    image_edit/flux_klein are one recipe now): the newer intent wins, and a
    binding survives from whichever row had one.
    """
    if conn.execute("SELECT COUNT(*) FROM recipe_policy").fetchone()[0]:
        return
    rows = conn.execute(
        "SELECT model, policy, reason, updated_at, device FROM model_policy ORDER BY updated_at"
    ).fetchall()
    merged: dict[str, list] = {}
    for model, policy, reason, ts, device in rows:
        prior = merged.get(model)
        merged[model] = [policy, reason, ts, device or (prior[3] if prior else None)]
    conn.executemany(
        "INSERT INTO recipe_policy (recipe, policy, reason, updated_at, device) "
        "VALUES (?, ?, ?, ?, ?)",
        [(name, *values) for name, values in merged.items()],
    )


class StatsRecorder:
    """Synchronous SQLite core; async wrappers hop via to_thread."""

    def __init__(self, db_path: Path | None = None):
        # Resolved at construction, not import — tests point GIQ_STATS_DB at
        # a tmp file in conftest.py; without this, unit tests that touch
        # eviction polluted the production DB and its row count suppressed
        # the one-shot inflight-log backfill.
        self._db_path = db_path or stats_db()
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        # UUID of the lowest-index card, refreshed by the telemetry sampler and
        # stamped onto each finished job. This is an *era* marker — which card
        # was installed while a job ran — and NOT where the job executed. Since
        # models can be bound per device (see policy.device_for), those are
        # different questions; per-job device attribution is not recorded yet.
        self._primary_gpu: str | None = None

    # --- lifecycle (sync, called via to_thread) ---------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            # Migrate pre-token DBs (production DB predates the usage view;
            # CREATE IF NOT EXISTS won't touch an existing jobs table).
            cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            for col in ("tokens_in", "tokens_out"):
                if col not in cols:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} INTEGER")
            if "gpu_uuid" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN gpu_uuid TEXT")
            # Rows from before ADR-006 stay NULL: they were never attributed,
            # which is not the same as anonymous.
            if "app" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN app TEXT")
            # Device bindings were added after the policy table, and existing
            # DBs have rows in it, so add the column rather than relying on
            # CREATE IF NOT EXISTS.
            policy_cols = {row[1] for row in conn.execute("PRAGMA table_info(model_policy)")}
            if "device" not in policy_cols:
                conn.execute("ALTER TABLE model_policy ADD COLUMN device TEXT")
            # ADR-003 renamed the job's worker and model: the kind of job is
            # its modality, the model its recipe. Renamed in place so history
            # stays, before any index on the new names is made — executescript
            # above ran against whatever the file already had.
            if "worker" in cols:
                conn.execute("ALTER TABLE jobs RENAME COLUMN worker TO modality")
            if "model" in cols:
                conn.execute("ALTER TABLE jobs RENAME COLUMN model TO recipe")
            conn.execute("DROP INDEX IF EXISTS idx_jobs_worker_ts")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_modality_ts ON jobs(modality, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_gpu_ts ON jobs(gpu_uuid, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_app_ts ON jobs(app, ts)")
            _migrate_policies(conn)
            conn.commit()
            self._conn = conn
        return self._conn

    def init_sync(self) -> None:
        with self._lock:
            conn = self._connect()
            (count,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        if count == 0:
            self._backfill()

    # --- GPU eras ----------------------------------------------------------

    def _note_gpus_sync(self, rows: list[tuple[str, str, float]]) -> None:
        """Upsert the permanent era record and refresh the primary-GPU stamp."""
        now = time.time()
        with self._lock:
            conn = self._connect()
            for uuid, name, total in rows:
                conn.execute(
                    "INSERT INTO gpu_eras (gpu_uuid, name, total_gb, first_seen, last_seen)"
                    " VALUES (?,?,?,?,?)"
                    " ON CONFLICT(gpu_uuid) DO UPDATE SET last_seen = excluded.last_seen,"
                    " name = excluded.name, total_gb = excluded.total_gb",
                    (uuid, name, total, now, now),
                )
            conn.commit()
        if rows:
            self._primary_gpu = rows[0][0]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _backfill(self) -> None:
        """Import historical 'end' records from the inflight diagnostics log."""
        # Written by runner._log_inflight; read here once, for the backfill.
        log_path = inflight_log()
        if not log_path.exists():
            return
        rows = []
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    # Cheap prefilter: start lines carry full prompts (156MB
                    # file); only end lines become rows.
                    if '"event": "end"' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        ts = datetime.fromisoformat(rec["ts"]).timestamp()
                    except (KeyError, ValueError):
                        continue
                    status = str(rec.get("status", "")).split(".")[-1] or "unknown"
                    rows.append(
                        (
                            ts,
                            rec.get("job_id"),
                            rec.get("modality") or rec.get("worker", "unknown"),
                            rec.get("recipe") or rec.get("model", "unknown"),
                            status,
                            None,  # queue wait unknown historically
                            rec.get("duration_ms"),
                            rec.get("results"),
                            rec.get("error"),
                        )
                    )
        except OSError as e:
            logger.warning(f"stats backfill failed reading {log_path}: {e}")
            return
        if not rows:
            return
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "INSERT INTO jobs (ts, job_id, modality, recipe, status, queue_wait_ms,"
                " run_ms, tasks, error) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        logger.info(f"stats: backfilled {len(rows)} historical jobs from {log_path.name}")

    # --- sync write cores --------------------------------------------------

    def _record_job_sync(self, row: tuple) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO jobs (ts, job_id, modality, recipe, status, queue_wait_ms,"
                " run_ms, tasks, error, tokens_in, tokens_out, gpu_uuid, app)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            conn.commit()

    def _record_event_sync(self, kind: str, detail: str | None) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO events (ts, kind, detail) VALUES (?,?,?)",
                (time.time(), kind, detail),
            )
            conn.commit()

    def _record_vram_sync(self, used: float, free: float, active: str | None, ready: int) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO vram_samples (ts, used_gb, free_gb, active_worker,"
                " residents_ready) VALUES (?,?,?,?,?)",
                (time.time(), used, free, active, ready),
            )
            conn.execute(
                "DELETE FROM vram_samples WHERE ts < ?",
                (time.time() - VRAM_RETENTION_DAYS * 86400,),
            )
            conn.commit()

    def _record_gpus_sync(self, rows: list[tuple]) -> None:
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "INSERT OR REPLACE INTO gpu_samples (ts, gpu_uuid, used_gb, total_gb,"
                " temp_c, power_w, util_pct) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            conn.execute(
                "DELETE FROM gpu_samples WHERE ts < ?",
                (time.time() - VRAM_RETENTION_DAYS * 86400,),
            )
            conn.commit()

    # --- model policy (sync; the runner reads these on its tick) ----------

    def load_policies(self) -> dict[str, tuple[str, str | None, float]]:
        """recipe -> (policy, reason, updated_at) for every override."""
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT recipe, policy, reason, updated_at FROM recipe_policy")
                .fetchall()
            )
        return {name: (p, r, ts) for name, p, r, ts in rows}

    def load_devices(self) -> dict[str, str]:
        """recipe -> bound GPU UUID, for every recipe that has one.

        Separate from ``load_policies`` because the two are independent: a
        recipe can be bound to a card without its residency being overridden,
        and unpinning must not silently unbind it.
        """
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT recipe, device FROM recipe_policy WHERE device IS NOT NULL")
                .fetchall()
            )
        return dict(rows)

    def save_policy(self, recipe: str, policy: str, reason: str | None) -> float:
        """Upsert one override. Returns the stored timestamp.

        Leaves ``device`` alone — an existing binding survives a policy change.
        """
        ts = time.time()
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO recipe_policy (recipe, policy, reason, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(recipe) DO UPDATE SET "
                "policy=excluded.policy, reason=excluded.reason, updated_at=excluded.updated_at",
                (recipe, policy, reason, ts),
            )
            conn.commit()
        return ts

    def save_device(self, recipe: str, device: str | None) -> float:
        """Bind a recipe to a GPU UUID, or clear the binding with None.

        Upserts against the same row as the residency policy, defaulting the
        policy column to the recipe's own default when it has no override —
        binding a card must not accidentally pin or unpin anything.
        """
        from giq.policy import PolicyStore

        ts = time.time()
        default_policy = PolicyStore.default_for(recipe)
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO recipe_policy (recipe, policy, reason, updated_at, device) "
                "VALUES (?, ?, NULL, ?, ?) ON CONFLICT(recipe) DO UPDATE SET "
                "device=excluded.device, updated_at=excluded.updated_at",
                (recipe, default_policy, ts, device),
            )
            conn.commit()
        return ts

    def delete_policy(self, recipe: str) -> bool:
        """Drop a residency override, keeping any device binding on the row."""
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "DELETE FROM recipe_policy WHERE recipe = ? AND device IS NULL", (recipe,)
            )
            if not cur.rowcount:
                # Row survives for its binding; reset the policy half of it.
                from giq.policy import PolicyStore

                cur = conn.execute(
                    "UPDATE recipe_policy SET policy = ?, reason = NULL WHERE recipe = ?",
                    (PolicyStore.default_for(recipe), recipe),
                )
            conn.commit()
        return cur.rowcount > 0

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Read-only query (sync; call via to_thread from handlers)."""
        with self._lock:
            conn = self._connect()
            return conn.execute(sql, params).fetchall()

    # --- async wrappers ----------------------------------------------------

    async def init(self) -> None:
        await asyncio.to_thread(self.init_sync)

    async def record_job(self, job) -> None:
        """Persist a finished Job (never raises — telemetry must not break jobs)."""
        try:
            queue_wait_ms = None
            if job.started_at and job.created_at:
                queue_wait_ms = int((job.started_at - job.created_at).total_seconds() * 1000)
            tokens_in, tokens_out = _sum_tokens(job.results)
            row = (
                (job.completed_at or datetime.now()).timestamp(),
                job.job_id,
                str(job.request.modality),
                job.request.model,
                str(job.status).split(".")[-1],
                queue_wait_ms,
                job.duration_ms,
                len(job.request.tasks) if job.request.tasks else 1,
                # jobs.error is free text and outlives everything else here, so
                # it gets the same check as the in-flight log: an error may not
                # quote the request that produced it.
                safe_detail(job.error, job.request),
                tokens_in,
                tokens_out,
                self._primary_gpu,
                getattr(job, "app", None),
            )
            await asyncio.to_thread(self._record_job_sync, row)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"stats record_job failed: {e}")

    async def record_event(self, kind: str, detail: str | None = None) -> None:
        try:
            await asyncio.to_thread(self._record_event_sync, kind, detail)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"stats record_event failed: {e}")

    async def sample_vram(self, used: float, free: float, active: str | None, ready: int) -> None:
        try:
            await asyncio.to_thread(self._record_vram_sync, used, free, active, ready)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"stats sample_vram failed: {e}")

    async def note_gpus(self, gpus: list) -> None:
        """Update the permanent era record from live telemetry (GpuTelemetry)."""
        try:
            rows = [(g.uuid, g.name, round(g.vram_total_gb, 2)) for g in gpus]
            if rows:
                await asyncio.to_thread(self._note_gpus_sync, rows)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"stats note_gpus failed: {e}")

    async def sample_gpus(self, gpus: list) -> None:
        """Persist one telemetry row per GPU (GpuTelemetry objects)."""
        try:
            now = time.time()
            rows = [
                (
                    now,
                    g.uuid,
                    round(g.vram_used_gb, 2),
                    round(g.vram_total_gb, 2),
                    g.temperature_c,
                    g.power_draw_w,
                    g.utilization_pct,
                )
                for g in gpus
            ]
            if rows:
                await asyncio.to_thread(self._record_gpus_sync, rows)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"stats sample_gpus failed: {e}")

    async def fetch(self, sql: str, params: tuple = ()) -> list[tuple]:
        return await asyncio.to_thread(self.query, sql, params)


_recorder: StatsRecorder | None = None


def get_stats() -> StatsRecorder:
    global _recorder
    if _recorder is None:
        _recorder = StatsRecorder()
    return _recorder
