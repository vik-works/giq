# SPDX-FileCopyrightText: 2026 vikworks UG (haftungsbeschränkt)
#
# SPDX-License-Identifier: Apache-2.0

"""Token accounting and GPU-era attribution in the stats recorder."""

import sqlite3
from pathlib import Path

from giq.stats import StatsRecorder, _sum_tokens


def test_sum_tokens_llmresult_shape():
    results = [
        {"id": "t1", "output": "a", "tokens_in": 100, "tokens_out": 20},
        {"id": "t2", "output": "b", "tokens_in": 50, "tokens_out": 5},
    ]
    assert _sum_tokens(results) == (150, 25)


def test_sum_tokens_openai_tool_path_shape():
    results = [
        {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        }
    ]
    assert _sum_tokens(results) == (42, 7)


def test_sum_tokens_absent_stays_null():
    # Non-LLM workers (audio/image) report no tokens — must stay NULL, not 0,
    # so the dashboard can distinguish "no accounting" from "zero tokens".
    assert _sum_tokens([{"id": "t1", "text": "transcript"}]) == (None, None)
    assert _sum_tokens(None) == (None, None)


def test_migration_adds_token_columns(tmp_path: Path):
    """A pre-token production DB gains tokens_in/tokens_out on connect."""
    db = tmp_path / "stats.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY, ts REAL NOT NULL, job_id TEXT,"
        " worker TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,"
        " queue_wait_ms INTEGER, run_ms INTEGER, tasks INTEGER, error TEXT)"
    )
    conn.execute(
        "INSERT INTO jobs (ts, worker, model, status) VALUES (1.0, 'llm', 'gemma', 'completed')"
    )
    conn.commit()
    conn.close()

    rec = StatsRecorder(db_path=db)
    rec._record_job_sync(
        (2.0, "j1", "llm", "gemma", "completed", 5, 100, 1, None, 30, 10, None, None)
    )
    rows = rec.query("SELECT tokens_in, tokens_out FROM jobs ORDER BY ts")
    # ADR-003: the old worker/model columns are the modality and recipe now,
    # renamed in place with their history.
    history = rec.query("SELECT modality, recipe FROM jobs ORDER BY ts")
    rec.close()
    assert rows == [(None, None), (30, 10)]
    assert history == [("llm", "gemma"), ("llm", "gemma")]


async def test_stats_usage_endpoint(tmp_path: Path, monkeypatch):
    import time

    import giq.stats as stats_mod
    from giq.api.stats_api import stats_usage

    monkeypatch.setattr(stats_mod, "_recorder", StatsRecorder(db_path=tmp_path / "u.db"))
    rec = stats_mod.get_stats()
    now = time.time()
    rec._record_job_sync(
        (now, "j1", "llm", "gemma-4-12b", "completed", 5, 100, 1, None, 200, 50, None, None)
    )
    rec._record_job_sync(
        (now, "j2", "llm", "gemma-4-12b", "failed", 5, 100, 1, "boom", None, None, None, None)
    )
    rec._record_job_sync(
        (
            now,
            "j3",
            "audio",
            "whisper-large-v3",
            "completed",
            5,
            100,
            1,
            None,
            None,
            None,
            None,
            None,
        )
    )

    d = await stats_usage(app=None, period="week", gpu=None, since=None, until=None)
    rec.close()

    assert d["totals"] == {"jobs": 3, "failed": 1, "tokens_in": 200, "tokens_out": 50}
    by_model = {m["recipe"]: m for m in d["recipes"]}
    assert by_model["gemma-4-12b"]["tokens_in"] == 200
    assert by_model["gemma-4-12b"]["jobs"] == 2
    assert by_model["whisper-large-v3"]["tokens_in"] is None
    # token-ranked ordering puts gemma first
    assert d["recipes"][0]["recipe"] == "gemma-4-12b"
    assert len(d["series"]) == 2  # one daily bucket per (worker, model)


# --- GPU era attribution ---------------------------------------------------


def _rec(rec, ts, uuid=None, worker="llm", model="m", status="completed"):
    rec._record_job_sync((ts, "j", worker, model, status, 0, 100, 1, None, 10, 5, uuid, None))


def test_note_gpus_keeps_first_seen_and_tracks_primary(tmp_path: Path):
    rec = StatsRecorder(db_path=tmp_path / "n.db")
    rec._note_gpus_sync([("GPU-a", "card A", 16.0)])
    first = rec.query("SELECT first_seen FROM gpu_eras WHERE gpu_uuid='GPU-a'")[0][0]
    rec._note_gpus_sync([("GPU-a", "card A", 16.0)])
    row = rec.query("SELECT first_seen, last_seen FROM gpu_eras WHERE gpu_uuid='GPU-a'")[0]
    assert row[0] == first  # first_seen is never overwritten
    assert row[1] >= first  # last_seen advances
    assert rec._primary_gpu == "GPU-a"
    rec.close()


async def test_usage_gpu_filter_and_explicit_range(tmp_path: Path, monkeypatch):
    import giq.stats as stats_mod
    from giq.api.stats_api import gpu_eras, stats_usage

    monkeypatch.setattr(stats_mod, "_recorder", StatsRecorder(db_path=tmp_path / "f.db"))
    rec = stats_mod.get_stats()
    base = 1783000000
    _rec(rec, base, uuid="GPU-a", model="old-model")
    _rec(rec, base + 86400, uuid="GPU-b", model="new-model")
    rec._note_gpus_sync([("GPU-b", "card B", 16.0)])

    everything = await stats_usage(app=None, period="all", gpu=None, since=None, until=None)
    assert everything["totals"]["jobs"] == 2

    only_a = await stats_usage(app=None, period="all", gpu="GPU-a", since=None, until=None)
    assert only_a["totals"]["jobs"] == 1
    assert only_a["recipes"][0]["recipe"] == "old-model"
    assert only_a["gpu"] == "GPU-a"

    windowed = await stats_usage(
        app=None, period="all", gpu=None, since=str(base + 100), until=str(base + 200000)
    )
    assert windowed["totals"]["jobs"] == 1
    assert windowed["recipes"][0]["recipe"] == "new-model"

    iso = await stats_usage(app=None, period="all", gpu=None, since="2026-01-01", until=None)
    assert iso["totals"]["jobs"] == 2

    eras = await gpu_eras()
    by_uuid = {e["uuid"]: e for e in eras["eras"]}
    assert by_uuid["GPU-b"]["jobs"] == 1
    rec.close()


async def test_usage_rejects_bad_bounds(tmp_path: Path, monkeypatch):
    from fastapi import HTTPException

    import giq.stats as stats_mod
    from giq.api.stats_api import stats_usage

    monkeypatch.setattr(stats_mod, "_recorder", StatsRecorder(db_path=tmp_path / "b.db"))
    for kwargs in (
        {"since": "not-a-date", "until": None},
        {"since": "1783000000", "until": "1782000000"},
    ):
        try:
            await stats_usage(app=None, period="all", gpu=None, **kwargs)
        except HTTPException as e:
            assert e.status_code == 400
        else:
            raise AssertionError(f"expected 400 for {kwargs}")
    stats_mod.get_stats().close()
