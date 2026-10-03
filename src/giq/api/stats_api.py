# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Dashboard-facing API: usage stats, model catalog, and quick self-tests."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from giq import paths as giq_paths
from giq import recipes
from giq.api.dependencies import get_orchestrator
from giq.gpus import get_gpus
from giq.models import JobRequest
from giq.policy import get_policy_store
from giq.runner import get_runner
from giq.services.orchestration import Orchestrator
from giq.stats import get_stats
from giq.storage import (
    StorageError,
    delete_weights,
    disk_report,
    weights_report,
)
from giq.vram import get_vram_status

logger = logging.getLogger(__name__)

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
# The React dashboard, built by `make ui` (frontend/ → vite build). Not in
# git: the wheel carries it as a build artifact, a checkout builds it.
_UI_DIR = _STATIC_DIR / "ui"

_UI_MISSING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>giq</title></head>
<body style="font-family: system-ui, sans-serif; padding: 2rem; line-height: 1.5">
<h1 style="font-weight: 500">UI not built</h1>
<p>The dashboard is a separate build step. Run <code>make ui</code> in the giq
checkout (it needs Node.js 22.12+ and npm), then reload this page. A server
without one installs a prebuilt dashboard instead:
<code>deploy/install-debian.sh --ui-only --ui-tarball giq-ui-&lt;version&gt;.tar.gz</code>
(made by <code>make ui-dist</code>), as docs/deployment.md describes.</p>
<p>The API is up regardless: <a href="/status">/status</a>, <a href="/docs">/docs</a>.</p>
</body></html>
"""


class _UiAssets(StaticFiles):
    """The built dashboard's hashed assets.

    Mounted at import, whether or not the UI has been built yet: without the
    directory every request is a plain 404 (StaticFiles would otherwise fail
    its startup check and turn each one into a 500), and a `make ui` run
    while giq is up is served without a restart.
    """

    async def check_config(self) -> None:
        if Path(str(self.directory)).is_dir():
            await super().check_config()


router.mount(
    "/dash/assets",
    _UiAssets(directory=_UI_DIR / "assets", check_dir=False),
    name="dash-assets",
)


@router.get("/dash", include_in_schema=False)
@router.get("/dash/", include_in_schema=False)
async def dashboard() -> Response:
    """The giq dashboard: the built React app, or a note saying how to build it."""
    index = _UI_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(_UI_MISSING, status_code=503)
    # The page names its hashed assets, so it must never be cached stale:
    # the assets themselves are immutable and cache fine.
    return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/sandbox", include_in_schema=False)
async def sandbox() -> RedirectResponse:
    """The sandbox is part of the dashboard; keep old links alive."""
    return RedirectResponse("/dash#/sandbox")


# --- stats -----------------------------------------------------------------


@router.get("/stats/summary")
async def stats_summary(hours: float = Query(default=24, gt=0, le=24 * 365)) -> dict:
    """Per recipe aggregates over the window, with the modality each ran as."""
    since = time.time() - hours * 3600
    rows = await get_stats().fetch(
        """
        SELECT modality, recipe, COUNT(*),
               SUM(CASE WHEN status != 'completed' THEN 1 ELSE 0 END),
               AVG(run_ms), MAX(run_ms), AVG(queue_wait_ms), SUM(tasks)
        FROM jobs WHERE ts >= ? GROUP BY modality, recipe ORDER BY COUNT(*) DESC
        """,
        (since,),
    )
    evictions = await get_stats().fetch(
        "SELECT COUNT(*) FROM events WHERE ts >= ? AND kind = 'evict'", (since,)
    )
    return {
        "hours": hours,
        "evictions": evictions[0][0] if evictions else 0,
        "recipes": [
            {
                "modality": w,
                "recipe": m,
                "jobs": n,
                "failed": failed,
                "avg_run_ms": round(avg_run) if avg_run is not None else None,
                "max_run_ms": max_run,
                "avg_queue_ms": round(avg_q) if avg_q is not None else None,
                "tasks": tasks,
            }
            for (w, m, n, failed, avg_run, max_run, avg_q, tasks) in rows
        ],
    }


_USAGE_PERIODS: dict[str, tuple[float | None, str]] = {
    # period -> (window seconds before now | None = local midnight / all-time,
    #            strftime bucket format, local time)
    "day": (None, "%Y-%m-%d %H:00"),
    "week": (7 * 86400, "%Y-%m-%d"),
    "month": (30 * 86400, "%Y-%m-%d"),
    "all": (0, "%Y-%m"),
}


def _parse_bound(value: str | None) -> float | None:
    """Accept a unix epoch or an ISO date/datetime as a range bound."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Bad time bound {value!r}: use epoch seconds or ISO 8601"
        ) from e


@router.get("/stats/gpus/eras")
async def gpu_eras() -> dict:
    """Every card this machine has had, with per-era usage rollups.

    Survives gpu_samples' retention window — this is the permanent record of
    what hardware ran what.
    """
    rows = await get_stats().fetch(
        """
        SELECT e.gpu_uuid, e.name, e.total_gb, e.first_seen, e.last_seen,
               COUNT(j.id),
               SUM(CASE WHEN j.status != 'completed' THEN 1 ELSE 0 END),
               SUM(j.tokens_in), SUM(j.tokens_out), AVG(j.run_ms),
               MIN(j.ts), MAX(j.ts)
        FROM gpu_eras e LEFT JOIN jobs j ON j.gpu_uuid = e.gpu_uuid
        GROUP BY e.gpu_uuid ORDER BY e.first_seen
        """
    )
    return {
        "eras": [
            {
                "uuid": uuid,
                "name": name,
                "total_gb": total,
                "first_seen": first,
                "last_seen": last,
                "jobs": jobs,
                "failed": failed or 0,
                "tokens_in": tin,
                "tokens_out": tout,
                "avg_run_ms": round(avg) if avg is not None else None,
                "first_job": fj,
                "last_job": lj,
            }
            for (
                uuid,
                name,
                total,
                first,
                last,
                jobs,
                failed,
                tin,
                tout,
                avg,
                fj,
                lj,
            ) in rows
        ]
    }


@router.get("/stats/usage")
async def stats_usage(
    period: str = Query(default="week", pattern="^(day|week|month|all)$"),
    gpu: str | None = Query(default=None, description="Restrict to one GPU uuid"),
    since: str | None = Query(default=None, description="Range start: epoch or ISO 8601"),
    until: str | None = Query(default=None, description="Range end: epoch or ISO 8601"),
) -> dict:
    """Token/call rollups per model, plus a bucketed series for the chart.

    Windows: day = since local midnight (hourly buckets), week = 7d, month = 30d
    (daily buckets), all = everything (monthly buckets). An explicit
    since/until pair overrides `period` for the range but still uses its
    bucket size; `gpu` restricts every figure to one card.
    """
    window, bucket_fmt = _USAGE_PERIODS[period]
    lo, hi = _parse_bound(since), _parse_bound(until)
    if lo is None:
        if period == "day":
            lo = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        elif window:
            lo = time.time() - window
        else:
            lo = 0.0
    if hi is not None and hi <= lo:
        raise HTTPException(status_code=400, detail="`until` must be after `since`")

    where = "ts >= ?"
    params: list = [lo]
    if hi is not None:
        where += " AND ts < ?"
        params.append(hi)
    if gpu:
        where += " AND gpu_uuid = ?"
        params.append(gpu)

    models = await get_stats().fetch(
        f"""
        SELECT modality, recipe, COUNT(*),
               SUM(CASE WHEN status != 'completed' THEN 1 ELSE 0 END),
               SUM(tasks), SUM(tokens_in), SUM(tokens_out), MAX(ts)
        FROM jobs WHERE {where} GROUP BY modality, recipe
        ORDER BY COALESCE(SUM(tokens_in),0) + COALESCE(SUM(tokens_out),0) DESC, COUNT(*) DESC
        """,
        tuple(params),
    )
    series = await get_stats().fetch(
        f"""
        SELECT strftime('{bucket_fmt}', ts, 'unixepoch', 'localtime') AS b,
               modality, recipe, COUNT(*), SUM(tokens_in), SUM(tokens_out)
        FROM jobs WHERE {where} GROUP BY b, modality, recipe ORDER BY b
        """,
        tuple(params),
    )
    return {
        "period": period,
        "gpu": gpu,
        "since": lo,
        "until": hi,
        "totals": {
            "jobs": sum(r[2] for r in models),
            "failed": sum(r[3] or 0 for r in models),
            "tokens_in": sum(r[5] or 0 for r in models),
            "tokens_out": sum(r[6] or 0 for r in models),
        },
        "recipes": [
            {
                "modality": w,
                "recipe": m,
                "jobs": n,
                "failed": failed or 0,
                "tasks": tasks,
                "tokens_in": tin,
                "tokens_out": tout,
                "last_ts": last,
            }
            for (w, m, n, failed, tasks, tin, tout, last) in models
        ],
        "series": [
            {"b": b, "modality": w, "recipe": m, "jobs": n, "tokens_in": tin, "tokens_out": tout}
            for (b, w, m, n, tin, tout) in series
        ],
    }


@router.get("/stats/timeline")
async def stats_timeline(
    hours: float = Query(default=24, gt=0, le=24 * 365),
    bucket_s: int = Query(default=3600, ge=60),
) -> dict:
    """Bucketed job counts per modality (stacked-bar source)."""
    since = time.time() - hours * 3600
    rows = await get_stats().fetch(
        """
        SELECT CAST(ts / ? AS INTEGER) * ? AS bucket, modality, COUNT(*),
               SUM(CASE WHEN status != 'completed' THEN 1 ELSE 0 END), AVG(run_ms)
        FROM jobs WHERE ts >= ? GROUP BY bucket, modality ORDER BY bucket
        """,
        (bucket_s, bucket_s, since),
    )
    return {
        "bucket_s": bucket_s,
        "points": [
            {
                "t": b,
                "modality": w,
                "jobs": n,
                "failed": f,
                "avg_run_ms": round(avg) if avg is not None else None,
            }
            for (b, w, n, f, avg) in rows
        ],
    }


@router.get("/stats/vram")
async def stats_vram(hours: float = Query(default=6, gt=0, le=24 * 14)) -> dict:
    since = time.time() - hours * 3600
    rows = await get_stats().fetch(
        "SELECT ts, used_gb, free_gb, active_worker, residents_ready FROM vram_samples"
        " WHERE ts >= ? ORDER BY ts",
        (since,),
    )
    return {
        "total_gb": get_vram_status().total_gb,
        "samples": [
            {"t": t, "used": round(u, 2), "active": a, "ready": r} for (t, u, _f, a, r) in rows
        ],
    }


@router.get("/stats/gpus")
async def stats_gpus(hours: float = Query(default=6, gt=0, le=24 * 14)) -> dict:
    """Per-GPU telemetry history (temp/power/util/vram), grouped by UUID."""
    import asyncio

    since = time.time() - hours * 3600
    rows = await get_stats().fetch(
        "SELECT ts, gpu_uuid, used_gb, total_gb, temp_c, power_w, util_pct"
        " FROM gpu_samples WHERE ts >= ? ORDER BY ts",
        (since,),
    )
    live = {g.uuid: g for g in await asyncio.to_thread(get_gpus)}
    series: dict[str, dict] = {}
    for ts, uuid, used, total, temp, power, util in rows:
        entry = series.setdefault(
            uuid,
            {
                "uuid": uuid,
                "index": live[uuid].index if uuid in live else None,
                "name": live[uuid].name if uuid in live else uuid,
                "total_gb": total,
                "power_limit": live[uuid].power_limit_w if uuid in live else None,
                "samples": [],
            },
        )
        entry["samples"].append({"t": ts, "used": used, "temp": temp, "power": power, "util": util})
    ordered = sorted(series.values(), key=lambda s: (s["index"] is None, s["index"]))
    return {"hours": hours, "gpus": ordered}


@router.get("/stats/events")
async def stats_events(
    hours: float = Query(default=48, gt=0, le=24 * 30), limit: int = 200
) -> list:
    since = time.time() - hours * 3600
    rows = await get_stats().fetch(
        "SELECT ts, kind, detail FROM events WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
        (since, limit),
    )
    return [{"t": t, "kind": k, "detail": d} for (t, k, d) in rows]


@router.get("/stats/jobs")
async def stats_jobs(
    limit: int = Query(default=50, le=500),
    gpu: str | None = Query(default=None, description="Restrict to one GPU uuid"),
) -> list:
    where = "WHERE gpu_uuid = ?" if gpu else ""
    params = (gpu, limit) if gpu else (limit,)
    rows = await get_stats().fetch(
        "SELECT ts, job_id, modality, recipe, status, queue_wait_ms, run_ms, tasks, error,"
        f" tokens_in, tokens_out FROM jobs {where} ORDER BY ts DESC LIMIT ?",
        params,
    )
    return [
        {
            "t": t,
            "job_id": j,
            "modality": w,
            "recipe": m,
            "status": s,
            "queue_ms": q,
            "run_ms": r,
            "tasks": n,
            "error": e,
            "tokens_in": tin,
            "tokens_out": tout,
        }
        for (t, j, w, m, s, q, r, n, e, tin, tout) in rows
    ]


# --- storage ---------------------------------------------------------------


@router.get("/storage")
async def storage() -> dict:
    """The disk per mount, the data directories giq resolved (``paths``), and
    the operator's recipe files (``recipes``): the directory, the files
    serving, the built-ins they replace, and the files left out with the
    reason. What is on those disks, checkpoint by checkpoint, is ``/weights``.

    The directories sit here rather than on /status because this endpoint
    already answers with absolute paths under the same access rules: it tells
    the operator nothing about the host that it did not before, and "which
    config and which state dir is this recipe using" is a storage question.
    """
    return {
        "paths": giq_paths.resolved(),
        "recipes": recipes.describe(recipes.current()),
        "disks": await asyncio.to_thread(disk_report),
    }


@router.get("/weights")
async def list_weights() -> dict:
    """Every checkpoint the recipes name, once each (ADR-003).

    Where it is (a path, or an HF repository in the cache), what it is
    (format, source, revision, licence), which recipes load it, and whether
    it is on disk and how big. One checkpoint serving two recipes is one
    entry here; ``GET /storage`` reports per mount.
    """
    return {"weights": await asyncio.to_thread(weights_report)}


@router.delete("/weights/{weights_id}")
async def delete_weights_by_id(weights_id: str) -> dict:
    """Remove one checkpoint from disk. The recipes that used it stay, uninstalled.

    Refused (409) while a recipe that uses it is resident or loaded.
    """
    busy = set(get_policy_store().residents()) | get_runner().loaded_keys()
    try:
        result = await asyncio.to_thread(delete_weights, weights_id, busy=busy)
    except StorageError as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from e
    if result["deleted"]:
        await get_stats().record_event(
            "delete_weights",
            f"{weights_id} ({', '.join(result['recipes'])}) freed {result['freed_bytes']} bytes",
        )
        logger.info("deleted weights %s: %s", weights_id, result["deleted"])
    return result


# --- quick tests -----------------------------------------------------------


@router.post("/test/{modality}")
async def smoke_test(
    modality: str,
    recipe: str | None = None,
    confirm: bool = False,
    orch: Orchestrator = Depends(get_orchestrator),
) -> dict:
    """One canned end-to-end job for ``modality``, as its plugin defines it.

    Runs on ``recipe`` when given, else on the first installed recipe that
    serves the modality. A test that evicts the resident set for minutes
    (image generation) requires ``?confirm=true``.
    """
    from giq import plugins
    from giq.registry import recipes_serving
    from giq.storage import installed

    spec = plugins.modality(modality)
    if spec is None or spec.smoke_test is None:
        known = sorted(m for m, s in plugins.modalities().items() if s.smoke_test)
        raise HTTPException(
            status_code=404,
            detail=f"no smoke test for {modality!r} (modalities with one: {', '.join(known)})",
        )
    test = spec.smoke_test()
    if test.evicts and not confirm:
        raise HTTPException(
            status_code=400,
            detail=f"the {modality} test evicts the resident set for minutes; pass ?confirm=true",
        )
    if recipe is None:
        candidates = [r.name for r in recipes_serving(modality) if installed(r.name)]
        if not candidates:
            raise HTTPException(status_code=404, detail=f"no installed recipe serves {modality!r}")
        recipe = candidates[0]
    started = time.monotonic()
    job_id, _ = await orch.submit_job(JobRequest(modality=modality, model=recipe, tasks=test.tasks))
    job = await orch.wait_for_job(job_id, timeout=test.timeout)
    result = job.results[0] if job.results else {}
    return {
        "job_id": job_id,
        "recipe": recipe,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "ok": job.error is None and bool(job.results),
        **test.summary(result),
    }
