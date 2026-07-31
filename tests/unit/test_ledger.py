"""DuckDB ledger helper tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pytest

from valuestream.engine.ledger import (
    _attach_runs_db,
    chunk_done,
    done_chunk_ids,
    ensure,
    insert_chunk,
    insert_run,
    source_run_lock,
    successful_chunk_keys,
)
from valuestream.store.meta import meta_dir


def _insert_terminal_chunk(
    workspace: Path,
    *,
    run_id: str,
    chunk_id: str,
    finished_at: dt.datetime,
    run_status: str = "ok",
    chunk_status: str = "ok",
    rows_kept: int = 0,
    config_hash: str = "source-hash",
) -> None:
    started_at = finished_at - dt.timedelta(minutes=1)
    failed_parent = run_status == "failed"
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
        chunks_total=2 if failed_parent else 1,
        chunks_ok=int(chunk_status == "ok"),
        chunks_failed=int(chunk_status != "ok") + int(failed_parent),
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
        status=chunk_status,
        error="replacement failed" if chunk_status == "failed" else None,
        pipeline_run_id=run_id,
    )


@pytest.mark.unit
def test_attach_runs_db_is_idempotent(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    ensure(workspace)

    with duckdb.connect(str(meta_dir(workspace) / "chunks.duckdb"), read_only=True) as conn:
        _attach_runs_db(conn, meta_dir(workspace) / "pipeline_runs.duckdb")
        _attach_runs_db(conn, meta_dir(workspace) / "pipeline_runs.duckdb")

        rows = conn.execute("SELECT COUNT(*) FROM runs_db.pipeline_runs").fetchone()

    assert rows == (0,)


@pytest.mark.unit
def test_successful_chunk_keys_does_not_reinitialize_existing_meta_dbs(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    ensure(workspace)

    with duckdb.connect(str(meta_dir(workspace) / "pipeline_runs.duckdb"), read_only=True):
        assert successful_chunk_keys(workspace, source_id="ih") == set()


@pytest.mark.unit
def test_successful_chunk_keys_supersedes_nonempty_attempt_with_empty_success(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    old_run_id = "11111111-1111-4111-8111-111111111111"
    new_run_id = "22222222-2222-4222-8222-222222222222"
    _insert_terminal_chunk(
        workspace,
        run_id=old_run_id,
        chunk_id="2026-07-01",
        finished_at=dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC),
        rows_kept=1,
    )
    _insert_terminal_chunk(
        workspace,
        run_id=new_run_id,
        chunk_id="2026-07-01",
        finished_at=dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC),
        rows_kept=0,
    )

    assert successful_chunk_keys(workspace, source_id="ih") == {(new_run_id, "2026-07-01")}


@pytest.mark.unit
def test_successful_chunk_keys_includes_success_from_terminal_failed_parent(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_id = "33333333-3333-4333-8333-333333333333"
    _insert_terminal_chunk(
        workspace,
        run_id=run_id,
        chunk_id="2026-07-01",
        finished_at=dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC),
        run_status="failed",
        rows_kept=1,
    )

    assert successful_chunk_keys(workspace, source_id="ih") == {(run_id, "2026-07-01")}


@pytest.mark.unit
def test_successful_chunk_keys_keeps_previous_success_after_failed_replacement(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    old_run_id = "44444444-4444-4444-8444-444444444444"
    failed_run_id = "55555555-5555-4555-8555-555555555555"
    _insert_terminal_chunk(
        workspace,
        run_id=old_run_id,
        chunk_id="2026-07-01",
        finished_at=dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC),
        rows_kept=1,
    )
    _insert_terminal_chunk(
        workspace,
        run_id=failed_run_id,
        chunk_id="2026-07-01",
        finished_at=dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC),
        run_status="failed",
        chunk_status="failed",
    )

    assert successful_chunk_keys(workspace, source_id="ih") == {(old_run_id, "2026-07-01")}


@pytest.mark.unit
def test_config_rollback_reprocesses_before_restoring_visibility(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    chunk_id = "2026-07-01"
    contract_a_run_id = "66666666-6666-4666-8666-666666666666"
    contract_b_run_id = "77777777-7777-4777-8777-777777777777"
    rollback_run_id = "88888888-8888-4888-8888-888888888888"
    _insert_terminal_chunk(
        workspace,
        run_id=contract_a_run_id,
        chunk_id=chunk_id,
        finished_at=dt.datetime(2026, 7, 1, 1, tzinfo=dt.UTC),
        rows_kept=1,
        config_hash="source-contract-a",
    )
    _insert_terminal_chunk(
        workspace,
        run_id=contract_b_run_id,
        chunk_id=chunk_id,
        finished_at=dt.datetime(2026, 7, 2, 1, tzinfo=dt.UTC),
        rows_kept=1,
        config_hash="source-contract-b",
    )

    assert not chunk_done(
        workspace,
        source_id="ih",
        chunk_id=chunk_id,
        files=[],
        config_hash="source-contract-a",
    )
    assert (
        done_chunk_ids(
            workspace,
            source_id="ih",
            config_hash="source-contract-a",
        )
        == set()
    )
    assert successful_chunk_keys(workspace, source_id="ih") == {(contract_b_run_id, chunk_id)}

    _insert_terminal_chunk(
        workspace,
        run_id=rollback_run_id,
        chunk_id=chunk_id,
        finished_at=dt.datetime(2026, 7, 3, 1, tzinfo=dt.UTC),
        rows_kept=1,
        config_hash="source-contract-a",
    )

    assert chunk_done(
        workspace,
        source_id="ih",
        chunk_id=chunk_id,
        files=[],
        config_hash="source-contract-a",
    )
    assert done_chunk_ids(
        workspace,
        source_id="ih",
        config_hash="source-contract-a",
    ) == {chunk_id}
    assert successful_chunk_keys(workspace, source_id="ih") == {(rollback_run_id, chunk_id)}


@pytest.mark.unit
def test_source_run_lock_rejects_a_concurrent_run(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    with (
        source_run_lock(workspace, "ih"),
        pytest.raises(RuntimeError, match="already has an ingestion run"),
        source_run_lock(workspace, "ih"),
    ):
        pass
