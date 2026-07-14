"""Metadata-DDL tests: init_meta_dbs creates the four databases idempotently."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from valuestream.store.meta import init_meta_dbs, list_tables, meta_dir

_EXPECTED: dict[str, set[str]] = {
    "chunks.duckdb": {"chunks"},
    "pipeline_runs.duckdb": {"pipeline_runs"},
    "config_versions.duckdb": {"config_versions"},
    "lineage.duckdb": {"lineage"},
}


@pytest.mark.unit
class TestInit:
    def test_creates_meta_dir(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        init_meta_dbs(ws)
        assert meta_dir(ws).is_dir()

    def test_creates_all_four_dbs(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        paths = init_meta_dbs(ws)
        assert set(paths.keys()) == {"chunks", "pipeline_runs", "config_versions", "lineage"}
        for name, set_of_tables in _EXPECTED.items():
            db_path = meta_dir(ws) / name
            assert db_path.is_file()
            assert set(list_tables(db_path)) == set_of_tables

    def test_chunks_schema_matches_spec(self, tmp_path: Path) -> None:
        # REPLACEMENT_DESIGN.md §6.4 — required column names and primary key.
        ws = tmp_path / "ws"
        init_meta_dbs(ws)
        with duckdb.connect(str(meta_dir(ws) / "chunks.duckdb"), read_only=True) as conn:
            cols = {
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = 'chunks'"
                ).fetchall()
            }
        assert cols == {
            "source_id",
            "chunk_id",
            "files",
            "file_hash",
            "rows_in",
            "rows_kept",
            "started_at",
            "finished_at",
            "status",
            "error",
            "pipeline_run_id",
        }

    def test_pipeline_runs_schema(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        init_meta_dbs(ws)
        with duckdb.connect(str(meta_dir(ws) / "pipeline_runs.duckdb"), read_only=True) as conn:
            cols = {
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'pipeline_runs'"
                ).fetchall()
            }
        assert cols == {
            "id",
            "workspace",
            "source_id",
            "config_hash",
            "started_at",
            "finished_at",
            "status",
            "rows_in",
            "rows_kept",
            "chunks_total",
            "chunks_ok",
            "chunks_failed",
        }

    def test_config_versions_schema(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        init_meta_dbs(ws)
        with duckdb.connect(str(meta_dir(ws) / "config_versions.duckdb"), read_only=True) as conn:
            cols = {
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'config_versions'"
                ).fetchall()
            }
        assert cols == {"config_hash", "yaml", "introduced_at"}

    def test_lineage_schema(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        init_meta_dbs(ws)
        with duckdb.connect(str(meta_dir(ws) / "lineage.duckdb"), read_only=True) as conn:
            cols = {
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'lineage'"
                ).fetchall()
            }
        assert cols == {
            "pipeline_run_id",
            "chunk_id",
            "source_id",
            "processor_id",
            "grain",
            "period",
            "partial_path",
            "config_hash",
            "rows",
            "created_at",
        }


# ---------------------------------------------------------------------------
# Idempotency: running twice is a no-op.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIdempotent:
    def test_re_run_does_not_error(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        init_meta_dbs(ws)
        # Second call must succeed without throwing.
        init_meta_dbs(ws)

    def test_re_run_preserves_existing_data(self, tmp_path: Path) -> None:
        """A row inserted between two init calls must still be present."""
        ws = tmp_path / "ws"
        init_meta_dbs(ws)

        # Insert a sentinel row into pipeline_runs.
        with duckdb.connect(str(meta_dir(ws) / "pipeline_runs.duckdb")) as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (id, workspace, status) VALUES (?, ?, ?)",
                ("00000000-0000-0000-0000-000000000001", "demo", "ok"),
            )

        # Re-init.
        init_meta_dbs(ws)

        with duckdb.connect(str(meta_dir(ws) / "pipeline_runs.duckdb"), read_only=True) as conn:
            count = conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0]
        assert count == 1

    def test_returns_same_paths_on_re_run(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        first = init_meta_dbs(ws)
        second = init_meta_dbs(ws)
        assert first == second
