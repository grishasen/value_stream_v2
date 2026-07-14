"""Dashboard query helpers for the Streamlit app."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.processors import grain_levels
from valuestream.query import query_metric
from valuestream.store.meta import meta_dir
from valuestream.store.parquet import aggregate_dir
from valuestream.ui.freshness import Freshness, metric_freshness

_TIME_GRAINS = {
    "Day": "daily",
    "day": "daily",
    "as_of_date": "daily",
    "Week": "weekly",
    "week": "weekly",
    "Month": "monthly",
    "month": "monthly",
    "period": "monthly",
    "Quarter": "quarterly",
    "quarter": "quarterly",
    "Year": "yearly",
    "year": "yearly",
}
_TIME_GRAIN_RANK = {
    "daily": 0,
    "weekly": 1,
    "monthly": 2,
    "quarterly": 3,
    "yearly": 4,
    "summary": 5,
}
_TIME_COLUMNS = set(_TIME_GRAINS)
_TIME_COLUMN_EQUIVALENTS = {
    "daily": ("Day", "day", "as_of_date"),
    "weekly": ("Week", "week"),
    "monthly": ("Month", "month", "period"),
    "quarterly": ("Quarter", "quarter"),
    "yearly": ("Year", "year"),
}
_FACET_FIELDS = ("facet_col", "facet_column", "facet_row", "facets")
_CURVE_CHARTS = {"roc_curve", "precision_recall_curve", "gain_curve", "lift_curve"}
_CHART_DIMENSION_FIELDS = {
    "line": ("x", *_FACET_FIELDS, "color"),
    "stacked_area": ("x", *_FACET_FIELDS, "color"),
    "bar": ("x", *_FACET_FIELDS, "color"),
    "kpi_card": ("group_by",),
    "waterfall": ("x", *_FACET_FIELDS, "color"),
    "pareto": ("x", *_FACET_FIELDS, "color"),
    "treemap": ("path", "x", "names", "color"),
    "heatmap": ("x", "y"),
    "cohort_heatmap": ("x", "y"),
    "scatter": ("animation_frame", "animation_group", *_FACET_FIELDS, "color"),
    "combo": ("x", *_FACET_FIELDS, "color"),
    "interval": ("x", *_FACET_FIELDS, "color"),
    "donut": ("names", "x", "color"),
    "geo_map": ("locations", "location", "lat", "lon", *_FACET_FIELDS, "color"),
    "table": ("group_by", "columns"),
    "calendar_heatmap": ("date", "x"),
    "bar_polar": ("theta", "color"),
    "sankey": ("source", "target"),
    "gauge": ("facet_row", "facet_col", "facet_column", "facets", "group_by"),
    "funnel": (*_FACET_FIELDS, "color"),
    "boxplot": ("x", *_FACET_FIELDS, "color"),
    "histogram": (*_FACET_FIELDS, "color"),
    "calibration_curve": (*_FACET_FIELDS, "color"),
    "roc_curve": (*_FACET_FIELDS, "color"),
    "precision_recall_curve": (*_FACET_FIELDS, "color"),
    "gain_curve": (*_FACET_FIELDS, "color"),
    "lift_curve": (*_FACET_FIELDS, "color"),
    "corr": ("color",),
    "descriptive_line": ("x", *_FACET_FIELDS, "color"),
    "descriptive_boxplot": ("x", *_FACET_FIELDS, "color"),
    "descriptive_histogram": (*_FACET_FIELDS, "color"),
    "descriptive_heatmap": ("x", "y"),
    "descriptive_funnel": (*_FACET_FIELDS, "color"),
    "experiment_z_score": ("x", "y", *_FACET_FIELDS, "color"),
    "experiment_odds_ratio": ("x", "y", *_FACET_FIELDS, "color"),
}
_DEFAULT_DIMENSION_FIELDS = (
    "group_by",
    "path",
    "x",
    "color",
    *_FACET_FIELDS,
    "animation_frame",
    "animation_group",
)
_STATEFUL_CHARTS = {"funnel"}


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


def tile_to_dict(tile: model.Tile) -> dict[str, Any]:
    """Return a plain dict including permissive tile extras."""
    return {**tile.model_dump(exclude={"model_extra"}), **dict(tile.model_extra or {})}


def query_tile(
    workspace_path: str | Path,
    catalog: model.Catalog,
    tile: model.Tile,
    *,
    filters: Mapping[str, Any] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pl.DataFrame:
    """Query aggregate data for a dashboard tile."""
    tile_dict = tile_to_dict(tile)
    dimension_aliases = _dimension_aliases(catalog, tile.metric, tile_dict)
    canonical_tile = _canonicalize_tile_dimensions(tile_dict, dimension_aliases)
    tile_filters = dict(tile_dict.get("filters") or {})
    filter_columns = available_filter_columns_for_tile(catalog, tile)
    if filters:
        for key, value in filters.items():
            column = _canonical_column(key, filter_columns)
            if value not in (None, "") and column is not None:
                tile_filters[column] = value
    tile_filters = _canonicalize_filter_keys(tile_filters, filter_columns)
    grain = grain_for_tile(canonical_tile)
    group_by = _processor_group_columns(
        catalog,
        tile.metric,
        group_by_for_tile(canonical_tile),
    )
    rows = _cached_query_metric(
        str(Path(workspace_path).resolve()),
        _metric_query_cache_signature(catalog, workspace_path, tile.metric, grain),
        tile.metric,
        tuple(group_by),
        _stable_json(tile_filters),
        grain,
        _date_cache_key(start),
        _date_cache_key(end),
        None,
        tile.chart in _STATEFUL_CHARTS
        or tile.chart.startswith("descriptive_")
        or _tile_references_scalar_state(catalog, tile.metric, canonical_tile),
        tile.chart in {"boxplot", "descriptive_boxplot", "combo"},
        tile.chart in _CURVE_CHARTS,
    )
    return _restore_dimension_aliases(_restore_time_columns(rows, tile_dict), dimension_aliases)


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


def available_filter_columns_for_tile(catalog: model.Catalog, tile: model.Tile) -> list[str]:
    """Return aggregate dimensions that can filter a tile without raw rows."""
    processor = _processor_for_metric(catalog, tile.metric)
    if processor is None:
        return filter_columns_for_tile(tile_to_dict(tile))
    return [column for column in processor.group_by if column not in _TIME_COLUMNS]


def available_filter_columns_for_page(catalog: model.Catalog, page: Any) -> list[str]:
    """Return all page-level aggregate filter columns in display order."""
    out: list[str] = []
    for tile in page.tiles:
        for column in available_filter_columns_for_tile(catalog, tile):
            if column not in out:
                out.append(column)
    return out


def filter_capabilities_for_page(
    catalog: model.Catalog,
    page: model.DashboardPage,
) -> list[FilterCapability]:
    """Return explicit or safely inferred page filters with tile coverage."""

    explicit = list(page.filters)
    fields = (
        [item.field for item in explicit]
        if explicit
        else available_filter_columns_for_page(catalog, page)
    )
    explicit_by_key = {_dimension_key(item.field): item for item in explicit}
    out: list[FilterCapability] = []
    inferred_primary = 0
    for field in fields:
        supported = tuple(
            tile.id
            for tile in page.tiles
            if _canonical_column(field, available_filter_columns_for_tile(catalog, tile))
            is not None
        )
        unsupported = tuple(tile.id for tile in page.tiles if tile.id not in supported)
        authored = explicit_by_key.get(_dimension_key(field))
        if authored is not None:
            label = authored.label or _humanize_identifier(field)
            display = authored.display
            scope = authored.scope
            control = authored.control
        else:
            applies_to_all = not unsupported
            display = "primary" if applies_to_all and inferred_primary < 3 else "secondary"
            if display == "primary":
                inferred_primary += 1
            label = _humanize_identifier(field)
            scope = "all_tiles" if applies_to_all else "compatible_tiles"
            control = "multiselect"
        out.append(
            FilterCapability(
                field=field,
                label=label,
                display=display,
                scope=scope,
                control=control,
                supported_tile_ids=supported,
                unsupported_tile_ids=unsupported,
                explicit=authored is not None,
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


def grain_for_tile(tile: Mapping[str, Any]) -> str:
    """Choose the physical grain for a tile."""
    grain = _grain_for_dimension_fields(tile)
    if grain is not None:
        return grain
    if "grain" in tile:
        return model.normalize_grain_name(str(tile["grain"]))
    grain = _grain_for_value(tile.get("group_by"))
    if grain is not None:
        return grain
    return "summary"


def group_by_for_tile(tile: Mapping[str, Any]) -> list[str]:
    """Infer the dimensions represented by a tile's rendered marks."""
    candidates: list[str] = []
    for field in _dimension_fields_for_tile(tile):
        _append_dimensions(candidates, tile.get(field))
    return [candidate for candidate in candidates if candidate not in _TIME_COLUMNS]


def filter_columns_for_tile(tile: Mapping[str, Any]) -> list[str]:
    """Return tile dimensions that are useful as report filters."""
    candidates: list[str] = []
    _append_dimensions(candidates, tile.get("group_by"))
    for candidate in group_by_for_tile(tile):
        _append_dimensions(candidates, candidate)
    return [candidate for candidate in candidates if candidate not in _TIME_COLUMNS]


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


def _processor_group_columns(
    catalog: model.Catalog,
    metric_name: str,
    candidates: list[str],
) -> list[str]:
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return candidates
    processor = next(
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.source),
        None,
    )
    if processor is None:
        return candidates
    out: list[str] = []
    for candidate in candidates:
        column = _canonical_column(candidate, processor.group_by)
        if column is not None and column not in out:
            out.append(column)
    return out


def _processor_for_metric(catalog: model.Catalog, metric_name: str) -> model.Processor | None:
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return None
    return next(
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.source),
        None,
    )


def _tile_references_scalar_state(
    catalog: model.Catalog,
    metric_name: str,
    tile: Mapping[str, Any],
) -> bool:
    processor = _processor_for_metric(catalog, metric_name)
    if processor is None:
        return False
    scalar_states = {
        name
        for name, state in model.effective_processor_states(processor).items()
        if state.type in {"count", "value_sum", "min", "max", "pooled_mean", "pooled_variance"}
    }
    if not scalar_states:
        return False
    candidates: list[str] = []
    _append_dimensions(candidates, tile)
    return any(candidate in scalar_states for candidate in candidates)


def _append_dimensions(candidates: list[str], value: Any) -> None:
    values: Iterable[Any]
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    for item in values:
        if item in (None, "", "---"):
            continue
        candidate = str(item)
        if candidate not in candidates:
            candidates.append(candidate)


def _dimension_aliases(
    catalog: model.Catalog,
    metric_name: str,
    tile: Mapping[str, Any],
) -> dict[str, str]:
    processor = _processor_for_metric(catalog, metric_name)
    if processor is None:
        return {}
    candidates: list[str] = []
    for field in _dimension_fields_for_tile(tile):
        _append_dimensions(candidates, tile.get(field))
    aliases: dict[str, str] = {}
    for candidate in candidates:
        column = _canonical_column(candidate, processor.group_by)
        if column is not None and column != candidate:
            aliases[candidate] = column
    return aliases


def _canonicalize_tile_dimensions(
    tile: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> dict[str, Any]:
    return {key: _canonicalize_dimension_value(value, aliases) for key, value in tile.items()}


def _canonicalize_dimension_value(value: Any, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonicalize_dimension_value(nested_value, aliases)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_dimension_value(item, aliases) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_dimension_value(item, aliases) for item in value)
    if isinstance(value, str):
        return aliases.get(value, value)
    return value


def _canonicalize_filter_keys(
    filters: Mapping[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in filters.items():
        out[_canonical_column(key, columns) or key] = value
    return out


def _canonical_column(candidate: str, columns: Iterable[str]) -> str | None:
    choices = list(columns)
    if candidate in choices:
        return candidate
    key = _dimension_key(candidate)
    if not key:
        return None
    for column in choices:
        if _dimension_key(column) == key:
            return column
    startswith_matches = [
        column
        for column in choices
        if _dimension_key(column).startswith(key) or key.startswith(_dimension_key(column))
    ]
    if len(startswith_matches) == 1:
        return startswith_matches[0]
    return None


def _dimension_key(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _humanize_identifier(value: str) -> str:
    spaced = "".join(
        f" {character}"
        if index and character.isupper() and not value[index - 1].isupper()
        else character
        for index, character in enumerate(value.replace("_", " ").replace("-", " "))
    )
    return " ".join(spaced.split()).strip().capitalize()


def _grain_for_value(value: Any) -> str | None:
    candidates: list[str] = []
    _append_dimensions(candidates, value)
    return next(
        (_TIME_GRAINS[candidate] for candidate in candidates if candidate in _TIME_GRAINS), None
    )


def _grain_for_dimension_fields(tile: Mapping[str, Any]) -> str | None:
    grains = []
    for field in _dimension_fields_for_tile(tile):
        grain = _grain_for_value(tile.get(field))
        if grain is not None:
            grains.append(grain)
    if not grains:
        return None
    return min(grains, key=lambda grain: _TIME_GRAIN_RANK[grain])


def _dimension_fields_for_tile(tile: Mapping[str, Any]) -> tuple[str, ...]:
    fields = _CHART_DIMENSION_FIELDS.get(str(tile.get("chart", "")), _DEFAULT_DIMENSION_FIELDS)
    if str(tile.get("chart", "")).casefold() == "gauge" and _has_facet_dimensions(tile):
        return tuple(field for field in fields if field != "group_by")
    return fields


def _has_facet_dimensions(tile: Mapping[str, Any]) -> bool:
    facets = tile.get("facets")
    return bool(
        tile.get("facet_row")
        or tile.get("facet_col")
        or tile.get("facet_column")
        or (
            isinstance(facets, Mapping) and any(facets.get(key) for key in ("row", "col", "column"))
        )
    )


def _restore_time_columns(rows: pl.DataFrame, tile: Mapping[str, Any]) -> pl.DataFrame:
    out = rows
    candidates: list[str] = []
    for field in _dimension_fields_for_tile(tile):
        _append_dimensions(candidates, tile.get(field))
    for candidate in candidates:
        grain = _TIME_GRAINS.get(candidate)
        if grain is None or candidate in out.columns:
            continue
        source = next(
            (column for column in _TIME_COLUMN_EQUIVALENTS[grain] if column in out.columns),
            None,
        )
        if source is not None:
            out = out.with_columns(pl.col(source).alias(candidate))
    return out


def _restore_dimension_aliases(rows: pl.DataFrame, aliases: Mapping[str, str]) -> pl.DataFrame:
    out = rows
    for alias, source in aliases.items():
        if alias not in out.columns and source in out.columns:
            out = out.with_columns(pl.col(source).alias(alias))
    return out


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
