"""Shared Streamlit page context and UI metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import streamlit as st

from valuestream.config import model
from valuestream.config.canonical import catalog_config_hash
from valuestream.config.loader import load
from valuestream.config.validate import CatalogValidationResult, validate_catalog
from valuestream.store.meta import meta_dir
from valuestream.ui.builder import MINIMUM_CATALOG_FILES, ensure_minimum_workspace
from valuestream.ui.freshness import latest_run, recent_runs

_CatalogSignature = tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ValueStreamContext:
    """Data shared by every Streamlit page."""

    workspace: Path
    catalog: model.Catalog
    validation: CatalogValidationResult
    catalog_hash: str


def load_context(workspace: str | Path) -> ValueStreamContext:
    """Load the active workspace and catalog validation context.

    Parsing, semantic validation, and hashing are cached per catalog file
    signature so ordinary Streamlit reruns skip the expensive work; any
    edit to a catalog YAML file changes the signature and reloads.
    """
    resolved = Path(workspace).resolve()
    signature = _catalog_signature(resolved)
    if any(size == 0 for _, _, size in signature):
        ensure_minimum_workspace(resolved)
        signature = _catalog_signature(resolved)
    return _load_context_cached(str(resolved), signature)


@st.cache_data(show_spinner=False, max_entries=8)
def _load_context_cached(
    workspace_str: str,
    signature: _CatalogSignature,
) -> ValueStreamContext:
    del signature  # cache key only
    resolved = Path(workspace_str)
    catalog = load(resolved)
    validation = validate_catalog(catalog)
    return ValueStreamContext(
        workspace=resolved,
        catalog=catalog,
        validation=validation,
        catalog_hash=catalog_config_hash(catalog)[:12],
    )


def _catalog_signature(workspace: Path) -> _CatalogSignature:
    out: list[tuple[str, int, int]] = []
    for name in MINIMUM_CATALOG_FILES:
        path = workspace / "catalog" / name
        try:
            stat = path.stat()
            out.append((name, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            out.append((name, 0, 0))
    return tuple(out)


def source_by_id(ctx: ValueStreamContext) -> dict[str, model.Source]:
    """Return sources keyed by id."""
    return {source.id: source for source in ctx.catalog.pipelines.sources}


def processors_for_source(ctx: ValueStreamContext, source_id: str) -> list[model.Processor]:
    """Return processors bound to a source."""
    return [
        processor
        for processor in ctx.catalog.processors.processors
        if processor.source == source_id
    ]


def metrics_for_processor(ctx: ValueStreamContext, processor_id: str) -> list[str]:
    """Return metric names backed by a processor."""
    return [
        name
        for name, metric in ctx.catalog.metrics.metrics.items()
        if metric.source == processor_id
    ]


def metrics_for_source(ctx: ValueStreamContext, source_id: str) -> list[str]:
    """Return metric names backed by processors for a source."""
    processor_ids = {processor.id for processor in processors_for_source(ctx, source_id)}
    return [
        name
        for name, metric in ctx.catalog.metrics.metrics.items()
        if metric.source in processor_ids
    ]


def source_root(ctx: ValueStreamContext, source: model.Source) -> Path:
    """Resolve a source reader root relative to the workspace."""
    extra = dict(source.reader.model_extra or {})
    raw = extra.get("root") or extra.get("base_dir") or "."
    root = Path(str(raw))
    if not root.is_absolute():
        root = ctx.workspace / root
    return root


def source_last_run(ctx: ValueStreamContext, source_id: str) -> dict[str, Any]:
    """Return latest pipeline run metadata for a source."""
    return latest_run(ctx.workspace, source_id=source_id)


def recent_runs_frame(ctx: ValueStreamContext, *, limit: int = 50) -> pl.DataFrame:
    """Return recent runs for the active workspace."""
    return recent_runs(ctx.workspace, limit=limit)


def chunks_for_run(ctx: ValueStreamContext, run_id: str) -> pl.DataFrame:
    """Return chunk ledger rows for a pipeline run."""
    db_path = meta_dir(ctx.workspace) / "chunks.duckdb"
    if not db_path.exists():
        return pl.DataFrame()
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(
            """
            SELECT source_id, chunk_id, status, rows_in, rows_kept, started_at,
                   finished_at, error, files
            FROM chunks
            WHERE CAST(pipeline_run_id AS VARCHAR) = ?
            ORDER BY started_at DESC
            """,
            (str(run_id),),
        ).pl()


def catalog_counts(ctx: ValueStreamContext) -> dict[str, int]:
    """Return top-level catalog object counts."""
    return {
        "Sources": len(ctx.catalog.pipelines.sources),
        "Processors": len(ctx.catalog.processors.processors),
        "Metrics": len(ctx.catalog.metrics.metrics),
        "Dashboards": len(ctx.catalog.dashboards.dashboards),
    }


__all__ = [
    "ValueStreamContext",
    "catalog_counts",
    "chunks_for_run",
    "load_context",
    "metrics_for_processor",
    "metrics_for_source",
    "processors_for_source",
    "recent_runs_frame",
    "source_by_id",
    "source_last_run",
    "source_root",
]
