"""Dashboard query helpers for the Streamlit app."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.processors import grain_levels
from valuestream.query import query_metric
from valuestream.reporting.tiles import (
    _canonical_column,
    _humanize_identifier,
    _processor_for_metric,
    available_filter_columns_for_page,
    available_filter_columns_for_tile,
    filter_columns_for_tile,
    grain_for_tile,
    group_by_for_tile,
    tile_to_dict,
)
from valuestream.reporting.tiles import query_tile as _query_tile
from valuestream.store.meta import meta_dir
from valuestream.store.parquet import aggregate_dir
from valuestream.ui.freshness import Freshness, metric_freshness


@dataclass(frozen=True)
class FilterCapability:
    """Aggregate-backed coverage and authored UI settings for one page filter."""

    field: str
    label: str
    display: str
    scope: str
    control: str
    supported_tile_ids: tuple[str, ...]
    unsupported_tile_ids: tuple[str, ...]
    explicit: bool = False

    @property
    def applies_to_all(self) -> bool:
        return not self.unsupported_tile_ids


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_query_metric(
    workspace_path: str,
    cache_signature: str,
    metric_name: str,
    group_by: tuple[str, ...],
    filters_json: str,
    grain: str,
    start: str | None,
    end: str | None,
    compare: str | None,
    include_state_columns: bool,
    include_quantile_suite: bool,
    include_curve_columns: bool,
) -> pl.DataFrame:
    del cache_signature
    return query_metric(
        workspace_path,
        metric_name,
        group_by=list(group_by),
        filters=json.loads(filters_json),
        grain=grain,
        start=start,
        end=end,
        compare=compare,
        include_state_columns=include_state_columns,
        include_quantile_suite=include_quantile_suite,
        include_curve_columns=include_curve_columns,
    )


def query_tile(
    workspace_path: str | Path,
    catalog: model.Catalog,
    tile: model.Tile,
    *,
    filters: Mapping[str, Any] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pl.DataFrame:
    """Query a dashboard tile through the Streamlit-cached metric query."""

    def cached_query(
        workspace: str | Path,
        metric_name: str,
        **options: Any,
    ) -> pl.DataFrame:
        return query_metric_cached(workspace, catalog, metric_name, **options)

    return _query_tile(
        workspace_path,
        catalog,
        tile,
        filters=filters,
        start=start,
        end=end,
        query_fn=cached_query,
    )


def query_metric_cached(
    workspace_path: str | Path,
    catalog: model.Catalog,
    metric_name: str,
    *,
    group_by: list[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    grain: str = "summary",
    start: dt.date | None = None,
    end: dt.date | None = None,
    compare: str | None = None,
    include_state_columns: bool = False,
    include_quantile_suite: bool = False,
    include_curve_columns: bool = False,
) -> pl.DataFrame:
    """Run a bounded Streamlit-cached metric query with aggregate invalidation."""

    normalized_grain = model.normalize_grain_name(grain)
    return _cached_query_metric(
        str(Path(workspace_path).resolve()),
        _metric_query_cache_signature(
            catalog,
            workspace_path,
            metric_name,
            normalized_grain,
        ),
        metric_name,
        tuple(group_by or []),
        _stable_json(dict(filters or {})),
        normalized_grain,
        _date_cache_key(start),
        _date_cache_key(end),
        compare,
        include_state_columns,
        include_quantile_suite,
        include_curve_columns,
    )


def _metric_query_cache_signature(
    catalog: model.Catalog,
    workspace_path: str | Path,
    metric_name: str,
    grain: str,
) -> str:
    metric = catalog.metrics.metrics.get(metric_name)
    processor = _processor_for_metric(catalog, metric_name)
    payload: dict[str, Any] = {
        "catalog": catalog.model_dump(mode="json", by_alias=True),
        "metric": metric.model_dump(mode="json", by_alias=True) if metric is not None else None,
        "processor_hash": (
            processor_computation_hash(catalog, processor) if processor is not None else None
        ),
        "aggregate": _aggregate_cache_signature(workspace_path, processor, grain),
        "ledger": _ledger_cache_signature(workspace_path),
    }
    return _stable_json(payload)


def _aggregate_cache_signature(
    workspace_path: str | Path,
    processor: model.Processor | None,
    grain: str,
) -> list[tuple[str, int, int, int]]:
    if processor is None:
        return []
    signatures: list[tuple[str, int, int, int]] = []
    for candidate in grain_levels.aggregate_grain_candidates(processor, grain):
        base = aggregate_dir(
            workspace_path,
            source_id=processor.source,
            processor_id=processor.id,
            grain=candidate,
        )
        count = 0
        latest_mtime = 0
        total_size = 0
        if base.exists():
            for path in base.glob("**/*.parquet"):
                if not path.is_file():
                    continue
                stat = path.stat()
                count += 1
                latest_mtime = max(latest_mtime, stat.st_mtime_ns)
                total_size += stat.st_size
        signatures.append((candidate, count, latest_mtime, total_size))
    return signatures


def cached_metric_freshness(
    workspace_path: str | Path,
    catalog: model.Catalog,
    metric_name: str,
    *,
    grain: str,
) -> Freshness:
    """Return metric freshness, cached on processor config + aggregate/ledger state.

    Any ingestion run touches the ledger databases, and any catalog edit
    changes the processor hash, so the signature invalidates automatically.
    """
    processor = _processor_for_metric(catalog, metric_name)
    signature = _stable_json(
        {
            "processor_hash": (
                processor_computation_hash(catalog, processor) if processor is not None else None
            ),
            "aggregate": _aggregate_cache_signature(workspace_path, processor, grain),
            "ledger": _ledger_cache_signature(workspace_path),
        }
    )
    return _cached_metric_freshness(
        catalog,
        str(Path(workspace_path).resolve()),
        signature,
        metric_name,
        grain,
    )


@st.cache_data(show_spinner=False, max_entries=512)
def _cached_metric_freshness(
    _catalog: model.Catalog,
    workspace_path: str,
    cache_signature: str,
    metric_name: str,
    grain: str,
) -> Freshness:
    del cache_signature
    return metric_freshness(workspace_path, _catalog, metric_name, grain=grain)


def _ledger_cache_signature(workspace_path: str | Path) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    base = meta_dir(workspace_path)
    for name in ("chunks.duckdb", "pipeline_runs.duckdb"):
        path = base / name
        if not path.exists():
            out.append((name, 0, 0))
            continue
        stat = path.stat()
        out.append((name, stat.st_mtime_ns, stat.st_size))
    return out


def _stable_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _date_cache_key(value: dt.date | None) -> str | None:
    return value.isoformat() if value is not None else None


def filter_capabilities_for_page(
    catalog: model.Catalog,
    page: model.DashboardPage,
) -> list[FilterCapability]:
    """Return explicitly configured page filters with exact tile coverage."""

    explicit = list(page.filters)
    out: list[FilterCapability] = []
    for authored in explicit:
        field = authored.field
        supported = tuple(
            tile.id
            for tile in page.tiles
            if field in available_filter_columns_for_tile(catalog, tile)
        )
        unsupported = tuple(tile.id for tile in page.tiles if tile.id not in supported)
        out.append(
            FilterCapability(
                field=field,
                label=authored.label or _humanize_identifier(field),
                display=authored.display,
                scope=authored.scope,
                control=authored.control,
                supported_tile_ids=supported,
                unsupported_tile_ids=unsupported,
                explicit=True,
            )
        )
    return out


def partition_filters_for_tile(
    catalog: model.Catalog,
    tile: model.Tile,
    filters: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Split page filters into supported values and explicitly ignored fields."""

    supported_columns = available_filter_columns_for_tile(catalog, tile)
    applied: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in (filters or {}).items():
        column = _canonical_column(str(key), supported_columns)
        if column is None:
            ignored.append(str(key))
        elif value not in (None, "", []):
            applied[column] = value
    return applied, tuple(ignored)


def parse_filter_text(raw: str) -> dict[str, str | list[str]]:
    """Parse UI filter text in ``key=value`` form."""
    filters: dict[str, str | list[str]] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        values = [item.strip() for item in value.split(",") if item.strip()]
        if values:
            filters[key.strip()] = values[0] if len(values) == 1 else values
    return filters


__all__ = [
    "FilterCapability",
    "available_filter_columns_for_page",
    "available_filter_columns_for_tile",
    "cached_metric_freshness",
    "filter_capabilities_for_page",
    "filter_columns_for_tile",
    "grain_for_tile",
    "group_by_for_tile",
    "parse_filter_text",
    "partition_filters_for_tile",
    "query_metric_cached",
    "query_tile",
    "tile_to_dict",
]
