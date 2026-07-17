"""Governed read-only SQL over DuckDB aggregate views and metric exports.

The SQL surface is a deliberate escape hatch for questions the structured
intent cannot express (joins across metrics, window functions, arbitrary SQL
aggregation). It keeps the core invariants: only persisted aggregates are
reachable, connections are read-only, statements must be a single SELECT,
filesystem/catalog functions are denied, sketch state blobs are masked, and
row counts are capped.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from valuestream.config import model
from valuestream.store.duckdb_views import refresh_aggregate_views, views_db_path
from valuestream.store.meta import meta_dir
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ROW_CAP = 500
MAX_ROW_CAP = 2000
DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_SQL_LENGTH = 8000

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DENIED_KEYWORDS = frozenset(
    {
        "alter",
        "attach",
        "begin",
        "call",
        "checkpoint",
        "commit",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "export",
        "force",
        "grant",
        "import",
        "insert",
        "install",
        "load",
        "merge",
        "pragma",
        "reset",
        "revoke",
        "rollback",
        "set",
        "transaction",
        "truncate",
        "update",
        "use",
        "vacuum",
    }
)
_DENIED_FUNCTIONS = frozenset(
    {
        "getenv",
        "glob",
        "parquet_scan",
        "read_blob",
        "read_csv",
        "read_csv_auto",
        "read_json",
        "read_json_auto",
        "read_json_objects",
        "read_ndjson",
        "read_ndjson_auto",
        "read_parquet",
        "read_text",
        "sniff_csv",
    }
)


@dataclass(frozen=True)
class SqlTable:
    """One queryable governed table or view."""

    name: str
    kind: str
    columns: list[tuple[str, str]]


@dataclass(frozen=True)
class SqlQueryResult:
    """Result of one governed SQL query."""

    sql: str
    rows: pl.DataFrame
    row_count: int
    truncated: bool
    masked_columns: list[str]


def validate_sql(sql: str) -> str:
    """Validate one read-only SELECT statement and return the trimmed SQL."""

    trimmed = str(sql or "").strip()
    if not trimmed:
        raise ValueError("SQL statement is empty")
    if len(trimmed) > _MAX_SQL_LENGTH:
        raise ValueError(f"SQL statement is too long (>{_MAX_SQL_LENGTH} characters)")
    if "--" in trimmed or "/*" in trimmed:
        raise ValueError("SQL comments are not allowed in governed queries")
    trimmed = trimmed.rstrip(";").strip()
    if ";" in trimmed:
        raise ValueError("only one SQL statement is allowed")
    first_word = _WORD_RE.search(trimmed)
    if first_word is None or first_word.group(0).lower() not in {"select", "with"}:
        raise ValueError("governed SQL must be a single SELECT (or WITH ... SELECT) statement")
    words = {word.lower() for word in _WORD_RE.findall(trimmed)}
    denied = sorted(words & (_DENIED_KEYWORDS | _DENIED_FUNCTIONS))
    if denied:
        raise ValueError(
            "governed SQL rejected; the following keywords/functions are not allowed: "
            + ", ".join(denied)
        )
    return trimmed


def run_sql_query(
    workspace_path: str | Path,
    sql: str,
    *,
    catalog: model.Catalog | None = None,
    limit: int = DEFAULT_ROW_CAP,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> SqlQueryResult:
    """Execute one governed SELECT over the workspace's DuckDB aggregates."""

    trimmed = validate_sql(sql)
    cap = max(1, min(int(limit), MAX_ROW_CAP))
    wrapped = f"SELECT * FROM ({trimmed}) AS governed_query LIMIT {cap + 1}"
    query_id = hashlib.sha256(trimmed.encode("utf-8")).hexdigest()[:12]
    statement_kind = _WORD_RE.search(trimmed)
    logger.info(
        "Running governed SQL: query_id=%s statement=%s sql_length=%s cap=%s",
        query_id,
        statement_kind.group(0).lower() if statement_kind is not None else "unknown",
        len(trimmed),
        cap,
    )
    with _GovernedConnection(workspace_path, catalog) as conn:
        timer = threading.Timer(max(1.0, float(timeout_seconds)), conn.interrupt)
        timer.start()
        try:
            frame = conn.execute(wrapped).pl()
        except duckdb.InterruptException as exc:
            raise TimeoutError(
                f"governed SQL query exceeded {timeout_seconds:.0f}s and was interrupted"
            ) from exc
        finally:
            timer.cancel()
    truncated = frame.height > cap
    if truncated:
        frame = frame.head(cap)
    masked = [name for name, dtype in frame.schema.items() if dtype == pl.Binary]
    if masked:
        frame = frame.drop(masked)
    logger.info(
        "Governed SQL completed: query_id=%s rows=%s column_count=%s truncated=%s "
        "masked_column_count=%s",
        query_id,
        frame.height,
        frame.width,
        truncated,
        len(masked),
    )
    return SqlQueryResult(
        sql=trimmed,
        rows=frame,
        row_count=frame.height,
        truncated=truncated,
        masked_columns=masked,
    )


def list_sql_tables(
    workspace_path: str | Path,
    catalog: model.Catalog | None = None,
) -> list[SqlTable]:
    """Return governed tables/views with their non-masked columns."""

    tables: list[SqlTable] = []
    with _GovernedConnection(workspace_path, catalog) as conn:
        entries = conn.execute(
            """
            SELECT database_name, view_name AS table_name, 'view' AS kind
            FROM duckdb_views() WHERE NOT internal
            UNION ALL
            SELECT database_name, table_name, 'table' AS kind
            FROM duckdb_tables() WHERE NOT internal
            ORDER BY database_name, table_name
            """
        ).fetchall()
        for database_name, table_name, kind in entries:
            if database_name in ("memory", "system", "temp"):
                continue
            qualified = f'{database_name}."{table_name}"'
            try:
                described = conn.execute(f"DESCRIBE SELECT * FROM {qualified} LIMIT 0").fetchall()
            except duckdb.Error as exc:
                error_type = type(exc).__name__
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", error_type):
                    error_type = "DatabaseError"
                logger.error("Governed table description failed: error_type=%s", error_type)
                continue
            columns = [
                (str(name), str(dtype))
                for name, dtype, *_ in described
                if str(dtype).upper() != "BLOB"
            ]
            tables.append(SqlTable(name=qualified, kind=str(kind), columns=columns))
    return tables


def sql_schema_summary(
    workspace_path: str | Path,
    catalog: model.Catalog | None = None,
) -> str:
    """Return a compact text schema listing for prompts."""

    try:
        tables = list_sql_tables(workspace_path, catalog)
    except FileNotFoundError:
        tables = []
    if not tables:
        return "No governed SQL tables are available; run ingestion first."
    lines = []
    for table in tables:
        columns = ", ".join(f"{name} {dtype}" for name, dtype in table.columns)
        lines.append(f"- {table.name} ({table.kind}): {columns}")
    return "\n".join(lines)


class _GovernedConnection:
    """Context manager yielding a read-only connection over governed DuckDB files."""

    def __init__(self, workspace_path: str | Path, catalog: model.Catalog | None) -> None:
        self._workspace = Path(workspace_path)
        self._catalog = catalog
        self._conn: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> duckdb.DuckDBPyConnection:
        views_path = views_db_path(self._workspace)
        if self._catalog is not None:
            refresh_aggregate_views(self._workspace, self._catalog)
        # The views database is the primary catalog so its view bodies resolve
        # their own `successful_chunks` table without qualification.
        attached = 0
        if views_path.exists():
            conn = duckdb.connect(str(views_path), read_only=True)
            attached += 1
        else:
            conn = duckdb.connect(":memory:")
        try:
            for export_path in sorted(meta_dir(self._workspace).glob("metric_export_*.duckdb")):
                alias = _export_alias(export_path)
                conn.execute(f"ATTACH '{_sql_string(str(export_path))}' AS {alias} (READ_ONLY)")
                attached += 1
            if not attached:
                raise FileNotFoundError(
                    "no governed DuckDB aggregates found; run ingestion "
                    "(and optionally `valuestream export-duckdb`) first"
                )
            _restrict_external_access(conn, self._workspace)
        except Exception:
            conn.close()
            raise
        self._conn = conn
        return conn

    def __exit__(self, *exc_info: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _export_alias(path: Path) -> str:
    stem = path.stem.removeprefix("metric_export_")
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_") or "export"
    return f"metrics_{safe}"


def _restrict_external_access(
    conn: duckdb.DuckDBPyConnection,
    workspace: Path,
) -> None:
    """Allow only canonical aggregate files needed by existing governed views.

    DuckDB accepts file paths directly in table position (for example
    ``SELECT * FROM 'events.parquet'``), so a text denylist is not a security
    boundary.  The connection-level external-access switch is the boundary:
    pre-created aggregate views may read only their exact glob patterns and
    matching Parquet files, while source files and every other filesystem path
    remain inaccessible.
    """

    aggregate_root = workspace / "aggregates"
    allowed: list[Path] = []
    if aggregate_root.exists():
        allowed.extend(path.resolve() for path in aggregate_root.glob("**/*.parquet"))
        allowed.extend(
            (directory.resolve() / "**" / "*.parquet")
            for directory in aggregate_root.glob("*/*/*")
            if directory.is_dir()
        )
    allowed_paths = sorted({str(path) for path in allowed})
    quoted = ", ".join(f"'{_sql_string(path)}'" for path in allowed_paths)
    conn.execute(f"SET allowed_paths = [{quoted}]")
    conn.execute("SET autoinstall_known_extensions = false")
    conn.execute("SET autoload_known_extensions = false")
    conn.execute("SET allow_community_extensions = false")
    conn.execute("SET enable_external_access = false")
    conn.execute("SET lock_configuration = true")


def _sql_string(value: str) -> str:
    return value.replace("'", "''")


__all__ = [
    "DEFAULT_ROW_CAP",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_ROW_CAP",
    "SqlQueryResult",
    "SqlTable",
    "list_sql_tables",
    "run_sql_query",
    "sql_schema_summary",
    "validate_sql",
]
