"""DuckDB aggregate-view publication tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest

from valuestream.engine.ledger import insert_chunk, insert_run
from valuestream.store.duckdb_views import _refresh_successful_chunks_table


def _insert_terminal_chunk(
    workspace: Path,
    *,
    run_id: str,
    chunk_id: str,
    finished_at: dt.datetime,
    config_hash: str,
    run_status: str = "ok",
    rows_kept: int = 0,
) -> None:
    started_at = finished_at - dt.timedelta(minutes=1)
    insert_run(
        workspace,
        run_id=run_id,
        workspace="test",
        source_id="ih",
        config_hash=config_hash,
        started_at=started_at,
        finished_at=finished_at,
        status=run_status,
        rows_in=rows_kept,
        rows_kept=rows_kept,
        chunks_total=2 if run_status == "failed" else 1,
        chunks_ok=1,
        chunks_failed=int(run_status == "failed"),
    )
    insert_chunk(
        workspace,
        source_id="ih",
        chunk_id=chunk_id,
        files=[],
        rows_in=rows_kept,
        rows_kept=rows_kept,
        started_at=started_at,
        finished_at=finished_at,
        status="ok",
        error=None,
        pipeline_run_id=run_id,
    )


@pytest.mark.unit
def test_successful_chunks_table_keeps_only_latest_successful_terminal_attempt(
    tmp_path: Path,
) -> None:
    old_run_id = "11111111-1111-4111-8111-111111111111"
    new_run_id = "22222222-2222-4222-8222-222222222222"
    chunk_id = "2026-07-01"
    for run_id, finished_at, run_status, rows_kept in (
        (old_run_id, dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC), "ok", 1),
        (new_run_id, dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC), "failed", 0),
    ):
        _insert_terminal_chunk(
            tmp_path,
            run_id=run_id,
            chunk_id=chunk_id,
            finished_at=finished_at,
            config_hash="source-hash",
            run_status=run_status,
            rows_kept=rows_kept,
        )

    with duckdb.connect() as conn:
        _refresh_successful_chunks_table(conn, tmp_path)
        rows = conn.execute(
            "SELECT pipeline_run_id, chunk_id, source_id FROM successful_chunks"
        ).fetchall()

    assert rows == [(new_run_id, chunk_id, "ih")]


@pytest.mark.unit
def test_successful_chunks_table_does_not_reauthorize_an_older_source_contract(
    tmp_path: Path,
) -> None:
    chunk_id = "2026-07-01"
    contract_a_run_id = "33333333-3333-4333-8333-333333333333"
    contract_b_run_id = "44444444-4444-4444-8444-444444444444"
    _insert_terminal_chunk(
        tmp_path,
        run_id=contract_a_run_id,
        chunk_id=chunk_id,
        finished_at=dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC),
        config_hash="source-contract-a",
        rows_kept=1,
    )
    _insert_terminal_chunk(
        tmp_path,
        run_id=contract_b_run_id,
        chunk_id=chunk_id,
        finished_at=dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC),
        config_hash="source-contract-b",
        rows_kept=1,
    )

    with duckdb.connect() as conn:
        _refresh_successful_chunks_table(conn, tmp_path)
        rows = conn.execute(
            "SELECT pipeline_run_id, chunk_id, source_id "
            "FROM successful_chunks ORDER BY pipeline_run_id"
        ).fetchall()

    assert rows == [(contract_b_run_id, chunk_id, "ih")]
