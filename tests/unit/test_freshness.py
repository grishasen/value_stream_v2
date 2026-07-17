"""Run freshness metadata ordering tests."""

from __future__ import annotations

import datetime as dt

import pytest

from valuestream.engine.ledger import insert_run, start_run
from valuestream.ui.freshness import latest_run, recent_runs


@pytest.mark.unit
def test_running_run_is_latest_even_while_finished_at_is_null(tmp_path) -> None:
    earlier = dt.datetime(2026, 7, 16, 10, tzinfo=dt.UTC)
    insert_run(
        tmp_path,
        run_id="11111111-1111-4111-8111-111111111111",
        workspace="test",
        source_id="ih",
        config_hash="hash",
        started_at=earlier,
        finished_at=earlier + dt.timedelta(minutes=5),
        status="ok",
        rows_in=10,
        rows_kept=10,
        chunks_total=1,
        chunks_ok=1,
        chunks_failed=0,
    )
    running_id = "22222222-2222-4222-8222-222222222222"
    start_run(
        tmp_path,
        run_id=running_id,
        workspace="test",
        source_id="ih",
        config_hash="hash",
        started_at=earlier + dt.timedelta(minutes=10),
        chunks_total=2,
    )

    latest = latest_run(tmp_path, source_id="ih")
    recent = recent_runs(tmp_path)

    assert str(latest["id"]) == running_id
    assert latest["status"] == "running"
    assert latest["finished_at"] is None
    assert str(recent[0, "id"]) == running_id
