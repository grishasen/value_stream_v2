"""DuckDB ledger helper tests."""

from __future__ import annotations

import duckdb
import pytest

from valuestream.engine.ledger import (
    _attach_runs_db,
    ensure,
    source_run_lock,
    successful_chunk_keys,
)
from valuestream.store.meta import meta_dir


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
def test_source_run_lock_rejects_a_concurrent_run(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    with (
        source_run_lock(workspace, "ih"),
        pytest.raises(RuntimeError, match="already has an ingestion run"),
        source_run_lock(workspace, "ih"),
    ):
        pass
