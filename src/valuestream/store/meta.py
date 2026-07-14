"""Metadata DDL for the four ``<workspace>/meta/*.duckdb`` files.

Schemas come from ``REPLACEMENT_DESIGN.md`` §6.4 (chunks, pipeline_runs,
config_versions) and ``docs/ARCHITECTURE.md`` §6.1 (lineage). Phase 0
ships only the schemas and an idempotent ``init_meta_dbs`` — no writers,
no readers. The ingestion engine in Phase 1 will use these tables to
record runs and chunks.

Each metadata table has one DuckDB file under ``<workspace>/meta/``,
matching the on-disk layout in ``ARCHITECTURE.md`` §9. The files are
single-writer (the ingestion engine) but multi-reader (the query layer).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# REPLACEMENT_DESIGN.md §6.4 — chunks ledger.
_CHUNKS_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    source_id        VARCHAR NOT NULL,
    chunk_id         VARCHAR NOT NULL,
    files            JSON NOT NULL,
    file_hash        VARCHAR NOT NULL,
    rows_in          BIGINT,
    rows_kept        BIGINT,
    started_at       TIMESTAMP,
    finished_at      TIMESTAMP,
    status           VARCHAR,
    error            VARCHAR,
    pipeline_run_id  UUID NOT NULL,
    PRIMARY KEY (source_id, chunk_id, pipeline_run_id)
);
"""

# REPLACEMENT_DESIGN.md §6.4 — run-level metadata.
_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id             UUID PRIMARY KEY,
    workspace      VARCHAR,
    source_id      VARCHAR,
    config_hash    VARCHAR,
    started_at     TIMESTAMP,
    finished_at    TIMESTAMP,
    status         VARCHAR,
    rows_in        BIGINT,
    rows_kept      BIGINT,
    chunks_total   INTEGER,
    chunks_ok      INTEGER,
    chunks_failed  INTEGER
);
"""

# REPLACEMENT_DESIGN.md §6.4 — config history.
_CONFIG_VERSIONS_DDL = """
CREATE TABLE IF NOT EXISTS config_versions (
    config_hash    VARCHAR PRIMARY KEY,
    yaml           VARCHAR,
    introduced_at  TIMESTAMP
);
"""

# ARCHITECTURE.md §6.1 — row-level lineage. The columns derive from
# REPLACEMENT_DESIGN.md §6.3 provenance and ARCHITECTURE.md §19's
# "lineage emitter" cross-cut: every aggregate partial maps to the
# (run, chunk, config, source, processor, grain, period) tuple that
# produced it.
_LINEAGE_DDL = """
CREATE TABLE IF NOT EXISTS lineage (
    pipeline_run_id  UUID NOT NULL,
    chunk_id         VARCHAR NOT NULL,
    source_id        VARCHAR NOT NULL,
    processor_id     VARCHAR NOT NULL,
    grain            VARCHAR NOT NULL,
    period           VARCHAR NOT NULL,
    partial_path     VARCHAR NOT NULL,
    config_hash      VARCHAR NOT NULL,
    rows             BIGINT,
    created_at       TIMESTAMP,
    PRIMARY KEY (pipeline_run_id, chunk_id, processor_id, grain, period)
);
"""


_META_DBS: tuple[tuple[str, str], ...] = (
    ("chunks.duckdb", _CHUNKS_DDL),
    ("pipeline_runs.duckdb", _RUNS_DDL),
    ("config_versions.duckdb", _CONFIG_VERSIONS_DDL),
    ("lineage.duckdb", _LINEAGE_DDL),
)
META_DB_FILENAMES: tuple[str, ...] = tuple(name for name, _ in _META_DBS)


def meta_dir(workspace_path: str | Path) -> Path:
    """Return ``<workspace_path>/meta/``."""
    return Path(workspace_path) / "meta"


def init_meta_dbs(workspace_path: str | Path) -> dict[str, Path]:
    """Create the four metadata DuckDB files with their schemas.

    Idempotent: running twice on the same workspace is a no-op (every DDL
    statement uses ``CREATE TABLE IF NOT EXISTS``). Returns a mapping of
    table name to the on-disk path.
    """
    md = meta_dir(workspace_path)
    md.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for filename, ddl in _META_DBS:
        path = md / filename
        with duckdb.connect(str(path)) as conn:
            conn.execute(ddl)
        # Map table name (filename minus suffix) → path for the caller.
        paths[Path(filename).stem] = path
    return paths


def list_tables(db_path: str | Path) -> list[str]:
    """Return the sorted list of user table names in ``db_path``."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
    return [r[0] for r in rows]


__all__ = ["META_DB_FILENAMES", "init_meta_dbs", "list_tables", "meta_dir"]
