"""Phase 1 metric resolver/planner/executor."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl

from valuestream.algorithms import rfm, stats
from valuestream.algorithms.curves import calibration_from_digests, curve_from_digests
from valuestream.config import model
from valuestream.config.canonical import catalog_config_hash, processor_computation_hash
from valuestream.config.loader import load
from valuestream.engine.ledger import aggregate_lineage_paths, successful_chunk_keys
from valuestream.expr.translator import translate
from valuestream.processors import (
    BinaryOutcomeProcessor,
    EntityLifecycleProcessor,
    EntitySetProcessor,
    FunnelProcessor,
    NumericDistributionProcessor,
    ScoreDistributionProcessor,
    SnapshotProcessor,
    grain_levels,
)
from valuestream.processors.registry import create_processor
from valuestream.states import cpc, hll, kll, tdigest, theta, topk
from valuestream.store.parquet import aggregate_exists, scan_aggregate
from valuestream.utils.timer import timed

_SketchPayload = bytes | bytearray | memoryview | None
_Processor = (
    BinaryOutcomeProcessor
    | NumericDistributionProcessor
    | ScoreDistributionProcessor
    | EntityLifecycleProcessor
    | EntitySetProcessor
    | FunnelProcessor
    | SnapshotProcessor
)
_LIFECYCLE_OUTPUT_COLUMNS = [
    "customers_count",
    "unique_holdings",
    "lifetime_value",
    "MinPurchasedDate",
    "MaxPurchasedDate",
    "frequency",
    "recency",
    "tenure",
    "monetary_value",
    "r_quartile",
    "f_quartile",
    "m_quartile",
    "rfm_seg",
    "rfm_segment",
    "rfm_score",
]
_QUANTILE_SUITE = {
    "Median": 0.5,
    "p25": 0.25,
    "p75": 0.75,
    "p90": 0.9,
    "p95": 0.95,
}
_VARIANT_OUTPUT_COLUMNS = [
    "Count",
    "Positives",
    "Negatives",
    "CTR",
    "TestCTR",
    "ControlCTR",
    "TestSampleSize",
    "ControlSampleSize",
    "AbsoluteRateDifference",
    "AbsoluteRateDifference_CI_Low",
    "AbsoluteRateDifference_CI_High",
    "Lift",
    "Lift_Z_Score",
    "Lift_P_Val",
    "StdErr",
]
_CONTINGENCY_OUTPUT_COLUMNS = [
    "Count",
    "Positives",
    "Negatives",
    "chi2_stat",
    "chi2_dof",
    "chi2_p_val",
    "chi2_odds_ratio_stat",
    "chi2_odds_ratio_ci_low",
    "chi2_odds_ratio_ci_high",
    "g_stat",
    "g_dof",
    "g_p_val",
    "g_odds_ratio_stat",
    "g_odds_ratio_ci_low",
    "g_odds_ratio_ci_high",
    "z_score",
    "z_p_val",
]
_PROPORTION_OUTPUT_COLUMNS = [
    "Count",
    "Positives",
    "Negatives",
    "z_score",
    "z_p_val",
]
_CURVE_OUTPUT_COLUMNS = [
    "roc_auc",
    "average_precision",
    "tpr",
    "fpr",
    "precision",
    "recall",
    "pos_fraction",
]


@dataclass(frozen=True)
class QueryProvenance:
    """Stable provenance for one governed metric-query result."""

    metric: str
    source_id: str
    processor_id: str
    requested_grain: str
    stored_grain: str
    catalog_hash: str
    computation_hash: str
    pipeline_run_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    aggregate_rows_scanned: int
    latest_created_at: dt.datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "source_id": self.source_id,
            "processor_id": self.processor_id,
            "requested_grain": self.requested_grain,
            "stored_grain": self.stored_grain,
            "catalog_hash": self.catalog_hash,
            "computation_hash": self.computation_hash,
            "pipeline_run_ids": list(self.pipeline_run_ids),
            "chunk_ids": list(self.chunk_ids),
            "aggregate_rows_scanned": self.aggregate_rows_scanned,
            "latest_created_at": (
                self.latest_created_at.isoformat() if self.latest_created_at is not None else None
            ),
        }


@dataclass(frozen=True)
class MetricQueryResult:
    """Metric rows plus the aggregate/config lineage that produced them."""

    rows: pl.DataFrame
    provenance: QueryProvenance


class AggregateNotReadyError(ValueError):
    """The current processor contract has no queryable aggregate version yet."""


@timed
def query_metric(
    workspace_path: str | Path,
    metric_name: str,
    *,
    group_by: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    grain: str = "daily",
    start: dt.date | dt.datetime | str | None = None,
    end: dt.date | dt.datetime | str | None = None,
    having: dict[str, Any] | None = None,
    order_by: list[str] | None = None,
    top_n: int | None = None,
    top_n_by: str | None = None,
    top_n_within: list[str] | None = None,
    compare: str | None = None,
    include_state_columns: bool = False,
    include_quantile_suite: bool = False,
    include_curve_columns: bool = False,
    _provenance_sink: list[QueryProvenance] | None = None,
) -> pl.DataFrame:
    """Query one metric from persisted aggregates.

    ``filters`` values may be scalars (equality), lists (membership), or
    operator specs like ``{"op": ">=", "value": 3}``. ``having`` applies the
    same specs to metric output columns after derivation. ``order_by`` accepts
    column names with an optional ``-`` prefix for descending order. ``top_n``
    keeps the largest rows by ``top_n_by`` (first metric output by default),
    optionally per ``top_n_within`` group. ``compare="prior_period"`` adds
    ``*_prev``/``*_delta``/``*_pct_change`` columns over the query time axis.
    """
    grain = model.normalize_grain_name(grain)
    catalog = load(workspace_path)
    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        raise ValueError(f"unknown metric {metric_name!r}")

    processor_config = next(
        (processor for processor in catalog.processors.processors if processor.id == metric.source),
        None,
    )
    if processor_config is None:
        raise ValueError(f"metric {metric_name!r} references unknown processor {metric.source!r}")

    processor = _make_processor(
        processor_config,
        computation_hash=processor_computation_hash(catalog, processor_config),
    )
    windowed_set_op = isinstance(metric, model.SetOpMetric) and any(
        operand.time_window for operand in metric.operands
    )
    if windowed_set_op and grain != "summary":
        raise ValueError("time-window set_op metrics currently require grain='summary'")
    storage_grain = "daily" if windowed_set_op else grain
    frame, stored_grain = _load_current_aggregate(
        workspace_path,
        processor,
        storage_grain,
        metric_name=metric_name,
        start=start,
        end=end,
    )
    frame = _with_calendar_columns(
        frame,
        stored_grain,
        processor.config.aggregation_level_for(stored_grain),
    )
    frame = _apply_filters(frame, filters or {})
    frame = _apply_time_range(frame, stored_grain, start=start, end=end)
    if _provenance_sink is not None:
        _provenance_sink.append(
            _query_provenance(
                frame,
                metric_name=metric_name,
                processor=processor,
                requested_grain=grain,
                stored_grain=stored_grain,
                catalog_hash_value=catalog_config_hash(catalog),
            )
        )

    group_columns = _group_columns(frame, group_by or [], grain)
    merge_columns = _metric_group_columns(group_columns, metric, processor, frame)
    if windowed_set_op:
        assert isinstance(metric, model.SetOpMetric)
        derived = _derive_windowed_set_op(
            frame,
            metric_name,
            metric,
            processor,
            group_columns=merge_columns,
            anchor=_date_bound(end) if end is not None else None,
        )
    else:
        merged = processor.merge_for_query(frame, merge_columns)
        derived = _derive_metric(
            merged,
            metric_name,
            metric,
            catalog.metrics.metrics,
            state_specs=processor.state_specs,
            group_columns=group_columns,
            include_curve_columns=include_curve_columns,
        )
    result_columns = (
        merge_columns if isinstance(metric, model.LifecycleSummaryMetric) else group_columns
    )
    result_group_columns = [column for column in result_columns if column in derived.columns]
    if not include_state_columns:
        derived = derived.select(
            [
                *result_group_columns,
                *_metric_output_columns(
                    metric_name,
                    metric,
                    derived,
                    include_quantile_suite=include_quantile_suite,
                    include_curve_columns=include_curve_columns,
                ),
            ]
        )
    if compare:
        derived = _with_prior_period_comparison(
            derived,
            compare,
            grain=grain,
            group_columns=result_group_columns,
        )
    derived = _apply_having(derived, having or {})
    derived = _apply_top_n(
        derived,
        top_n,
        top_n_by,
        top_n_within,
        group_columns=result_group_columns,
    )
    order_specs = _order_by_specs(order_by, derived.columns)
    if order_specs:
        return derived.sort(
            [name for name, _ in order_specs],
            descending=[descending for _, descending in order_specs],
        )
    if top_n is not None:
        return derived
    return derived.sort(result_group_columns) if result_group_columns else derived


def query_metric_result(
    workspace_path: str | Path,
    metric_name: str,
    **query_options: Any,
) -> MetricQueryResult:
    """Execute a metric query and return rows with stable provenance metadata."""

    options = dict(query_options)
    options.pop("_provenance_sink", None)
    provenance: list[QueryProvenance] = []
    rows = query_metric(
        workspace_path,
        metric_name,
        _provenance_sink=provenance,
        **options,
    )
    if not provenance:  # pragma: no cover - query_metric always records it
        raise RuntimeError("metric query did not produce provenance")
    return MetricQueryResult(rows=rows, provenance=provenance[0])


def _query_provenance(
    frame: pl.DataFrame,
    *,
    metric_name: str,
    processor: _Processor,
    requested_grain: str,
    stored_grain: str,
    catalog_hash_value: str,
) -> QueryProvenance:
    run_ids = _string_values(frame, "pipeline_run_id")
    chunk_ids = _string_values(frame, "chunk_id")
    latest_created_at = (
        frame["created_at"].max()
        if "created_at" in frame.columns and not frame.is_empty()
        else None
    )
    return QueryProvenance(
        metric=metric_name,
        source_id=processor.source_id,
        processor_id=processor.id,
        requested_grain=requested_grain,
        stored_grain=stored_grain,
        catalog_hash=catalog_hash_value,
        computation_hash=processor.config_hash,
        pipeline_run_ids=run_ids,
        chunk_ids=chunk_ids,
        aggregate_rows_scanned=frame.height,
        latest_created_at=cast(dt.datetime | None, latest_created_at),
    )


def _string_values(frame: pl.DataFrame, column: str) -> tuple[str, ...]:
    if column not in frame.columns or frame.is_empty():
        return ()
    return tuple(sorted({str(value) for value in frame[column].drop_nulls().to_list()}))


@timed
def _load_current_aggregate(
    workspace_path: str | Path,
    processor: _Processor,
    grain: str,
    *,
    metric_name: str,
    start: dt.date | dt.datetime | str | None = None,
    end: dt.date | dt.datetime | str | None = None,
) -> tuple[pl.DataFrame, str]:
    found_aggregate = False
    found_stale = False
    found_unpublished = False
    successful_keys: set[tuple[str, str]] | None = None
    for stored_grain in grain_levels.aggregate_grain_candidates(processor.config, grain):
        if not aggregate_exists(
            workspace_path,
            source_id=processor.source_id,
            processor_id=processor.id,
            grain=stored_grain,
        ):
            continue
        found_aggregate = True
        selected_paths, lineage_status = _current_lineage_paths(
            workspace_path,
            processor,
            stored_grain,
        )
        if lineage_status != "ready":
            found_stale |= lineage_status == "stale"
            found_unpublished |= lineage_status == "unpublished"
            continue
        scanned = scan_aggregate(
            workspace_path,
            source_id=processor.source_id,
            processor_id=processor.id,
            grain=stored_grain,
            paths=selected_paths,
        )
        # Stale detection counts config-matching rows *before* any value/time
        # predicate, so "no rows in range" is never mistaken for "config changed".
        matching = scanned.filter(pl.col("config_hash") == processor.config_hash)
        level = processor.config.aggregation_level_for(stored_grain)
        pruned = _prune_periods(matching, level, start=start, end=end)
        if successful_keys is None:
            successful_keys = successful_chunk_keys(
                workspace_path,
                source_id=processor.source_id,
            )
        # One collection graph lets Polars eliminate the shared aggregate scan
        # across status, range-visible, and all-period visibility branches.
        plans = [
            scanned.select(
                pl.len().alias("rows"),
                (pl.col("config_hash") == processor.config_hash).sum().alias("matching_rows"),
            ),
            _filter_successful_chunks_lazy(pruned, successful_keys),
        ]
        if start is not None or end is not None:
            plans.append(
                _filter_successful_chunks_lazy(matching, successful_keys).select(
                    pl.len().alias("visible_rows")
                )
            )
        collected = pl.collect_all(plans)
        counts = collected[0].row(0, named=True)
        if int(counts["matching_rows"] or 0) == 0:
            if int(counts["rows"] or 0) > 0:
                found_stale = True
            continue
        visible = collected[1]
        if visible.is_empty():
            has_visible_rows_any_period = (start is not None or end is not None) and int(
                collected[2].item() or 0
            ) > 0
            if not has_visible_rows_any_period:
                found_unpublished = True
                continue
        return visible, stored_grain
    if found_stale or found_unpublished:
        raise AggregateNotReadyError(
            f"metric {metric_name!r} is configured, but aggregate data for "
            f"{processor.source_id}/{processor.id}/{grain} has not been published for the "
            "current processor configuration; run ingestion for a new workspace or "
            "backfill/reprocess existing data"
        )
    if not found_aggregate:
        raise FileNotFoundError(
            f"no aggregate data for {processor.source_id}/{processor.id}/{grain}; run ingestion first"
        )
    raise FileNotFoundError(
        f"no aggregate rows for {processor.source_id}/{processor.id}/{grain}; run ingestion first"
    )


def _current_lineage_paths(
    workspace_path: str | Path,
    processor: _Processor,
    grain: str,
) -> tuple[tuple[Path, ...] | None, Literal["ready", "stale", "unpublished"]]:
    """Select only physical files produced under the current processor contract."""
    paths_by_hash = aggregate_lineage_paths(
        workspace_path,
        source_id=processor.source_id,
        processor_id=processor.id,
        grain=grain,
    )
    if not paths_by_hash:
        return None, "ready"
    current_paths = paths_by_hash.get(processor.config_hash, ())
    if not current_paths:
        return None, "stale"
    existing_paths = tuple(path for path in current_paths if path.exists())
    if not existing_paths:
        return None, "unpublished"
    return existing_paths, "ready"


def _filter_successful_chunks_lazy(
    frame: pl.LazyFrame,
    keys: set[tuple[str, str]],
) -> pl.LazyFrame:
    names = set(frame.collect_schema().names())
    if not {"pipeline_run_id", "chunk_id"} <= names or not keys:
        return frame.head(0)
    ordered_keys = sorted(keys)
    key_frame = pl.LazyFrame(
        {
            "pipeline_run_id": [run_id for run_id, _ in ordered_keys],
            "chunk_id": [chunk_id for _, chunk_id in ordered_keys],
        }
    )
    joined = frame.with_columns(pl.col("pipeline_run_id").cast(pl.String)).join(
        key_frame,
        on=["pipeline_run_id", "chunk_id"],
        how="inner",
    )
    if "created_at" not in names:
        return joined
    return joined.filter(pl.col("created_at") == pl.col("created_at").max().over("chunk_id"))


def _prune_periods(
    frame: pl.LazyFrame,
    level: str,
    *,
    start: dt.date | dt.datetime | str | None,
    end: dt.date | dt.datetime | str | None,
) -> pl.LazyFrame:
    """Prune whole hive ``period=`` partitions outside the requested window.

    Only applied when the physical ``period`` is a ``YYYY-MM`` string (daily or
    monthly levels). This is a coarse optimization; the executor still applies
    the exact time-range filter post-collect, so results are unchanged.
    """
    if level not in {"daily", "monthly"} or (start is None and end is None):
        return frame
    period = pl.col("period").cast(pl.String)
    out = frame
    if start is not None:
        out = out.filter(period >= _month_bound(start))
    if end is not None:
        out = out.filter(period <= _month_bound(end))
    return out


_FILTER_OP_ALIASES = {
    "eq": "eq",
    "=": "eq",
    "==": "eq",
    "ne": "ne",
    "!=": "ne",
    "<>": "ne",
    "gt": "gt",
    ">": "gt",
    "gte": "gte",
    ">=": "gte",
    "lt": "lt",
    "<": "lt",
    "lte": "lte",
    "<=": "lte",
    "in": "in",
    "not_in": "not_in",
    "not in": "not_in",
    "contains": "contains",
    "starts_with": "starts_with",
    "ends_with": "ends_with",
    "is_null": "is_null",
    "not_null": "not_null",
    "is_not_null": "not_null",
}


def _apply_filters(
    frame: pl.DataFrame,
    filters: dict[str, Any],
) -> pl.DataFrame:
    out = frame
    for key, value in filters.items():
        column = key
        if column not in out.columns:
            raise ValueError(f"filter column {key!r} not present in aggregate")
        out = out.filter(_filter_expression(column, value))
    return out


def _apply_having(
    frame: pl.DataFrame,
    having: dict[str, Any],
) -> pl.DataFrame:
    out = frame
    for key, value in having.items():
        if key not in out.columns:
            available = ", ".join(out.columns)
            raise ValueError(
                f"having column {key!r} is not in the query result; use one of: {available}"
            )
        out = out.filter(_filter_expression(key, value))
    return out


def _filter_expression(column: str, value: Any) -> pl.Expr:  # noqa: PLR0911, PLR0912 — operator dispatch
    col = pl.col(column)
    if isinstance(value, dict):
        op_raw = str(value.get("op", "") or "").strip().lower()
        op = _FILTER_OP_ALIASES.get(op_raw)
        if op is None:
            supported = ", ".join(sorted(set(_FILTER_OP_ALIASES.values())))
            raise ValueError(
                f"unsupported filter operator {value.get('op')!r} for column {column!r}; "
                f"use one of: {supported}"
            )
        if op == "is_null":
            return col.is_null()
        if op == "not_null":
            return col.is_not_null()
        operand = value.get("values", value.get("value"))
        if op == "in":
            return col.is_in(_operand_values(operand))
        if op == "not_in":
            return ~col.is_in(_operand_values(operand))
        if operand is None:
            raise ValueError(f"filter operator {op!r} on column {column!r} requires a value")
        if op == "eq":
            return cast(pl.Expr, col == operand)
        if op == "ne":
            return cast(pl.Expr, col != operand)
        if op == "gt":
            return cast(pl.Expr, col > operand)
        if op == "gte":
            return cast(pl.Expr, col >= operand)
        if op == "lt":
            return cast(pl.Expr, col < operand)
        if op == "lte":
            return cast(pl.Expr, col <= operand)
        if op == "contains":
            return col.cast(pl.String).str.contains(str(operand), literal=True)
        if op == "starts_with":
            return col.cast(pl.String).str.starts_with(str(operand))
        return col.cast(pl.String).str.ends_with(str(operand))
    if isinstance(value, list):
        return col.is_in(value)
    return col.is_in([value])


def _operand_values(operand: Any) -> list[Any]:
    if isinstance(operand, list):
        return operand
    if operand is None:
        raise ValueError("membership filter requires a list of values")
    return [operand]


def _order_by_specs(
    order_by: list[str] | None,
    columns: Iterable[str],
) -> list[tuple[str, bool]]:
    available = set(columns)
    specs: list[tuple[str, bool]] = []
    for item in order_by or []:
        raw = str(item).strip()
        if not raw:
            continue
        descending = raw.startswith("-")
        name = raw[1:].strip() if descending else raw
        if name not in available:
            options = ", ".join(sorted(available))
            raise ValueError(
                f"order_by column {name!r} is not in the query result; use one of: {options}"
            )
        specs.append((name, descending))
    return specs


def _apply_top_n(
    frame: pl.DataFrame,
    top_n: int | None,
    top_n_by: str | None,
    top_n_within: list[str] | None,
    *,
    group_columns: list[str],
) -> pl.DataFrame:
    if top_n is None:
        return frame
    n = max(1, int(top_n))
    by = top_n_by or _default_rank_column(frame, group_columns)
    if by not in frame.columns:
        options = ", ".join(frame.columns)
        raise ValueError(
            f"top_n_by column {by!r} is not in the query result; use one of: {options}"
        )
    within = [column for column in (top_n_within or []) if column]
    missing = [column for column in within if column not in frame.columns]
    if missing:
        raise ValueError(f"top_n_within column(s) not in the query result: {', '.join(missing)}")
    ranked = frame.sort(by, descending=True, nulls_last=True)
    if within:
        return ranked.group_by(within, maintain_order=True).head(n)
    return ranked.head(n)


def _default_rank_column(frame: pl.DataFrame, group_columns: list[str]) -> str:
    for column in frame.columns:
        if column in group_columns:
            continue
        if frame.schema[column].is_numeric():
            return column
    return frame.columns[-1]


def _with_prior_period_comparison(
    frame: pl.DataFrame,
    compare: str,
    *,
    grain: str,
    group_columns: list[str],
) -> pl.DataFrame:
    if compare != "prior_period":
        raise ValueError(f"unsupported compare mode {compare!r}; use 'prior_period'")
    time_column = next(
        (
            candidate
            for candidate in grain_levels.TIME_COLUMN_CANDIDATES.get(
                model.normalize_grain_name(grain), ()
            )
            if candidate in frame.columns
        ),
        None,
    )
    if time_column is None:
        raise ValueError(
            "compare='prior_period' requires a time-bucketed query; "
            "ask for a Day/Month/Quarter/Year time axis"
        )
    dimensions = [column for column in group_columns if column != time_column]
    value_columns = [
        column
        for column in frame.columns
        if column not in group_columns and frame.schema[column].is_numeric()
    ]
    if not value_columns:
        return frame
    ordered = frame.sort([*dimensions, time_column])
    expressions: list[pl.Expr] = []
    for column in value_columns:
        prev = pl.col(column).shift(1).over(dimensions) if dimensions else pl.col(column).shift(1)
        expressions.append(prev.alias(f"{column}_prev"))
        expressions.append((pl.col(column) - prev).alias(f"{column}_delta"))
        expressions.append(
            pl.when(prev != 0)
            .then((pl.col(column) - prev) / prev)
            .otherwise(None)
            .alias(f"{column}_pct_change")
        )
    return ordered.with_columns(expressions)


def _apply_time_range(
    frame: pl.DataFrame,
    grain: str,
    *,
    start: dt.date | dt.datetime | str | None,
    end: dt.date | dt.datetime | str | None,
) -> pl.DataFrame:
    if start is None and end is None:
        return frame
    grain = model.normalize_grain_name(grain)
    if grain == "daily":
        column = next(
            (candidate for candidate in ("Day", "day", "as_of_date") if candidate in frame.columns),
            None,
        )
        if column is None:
            return frame
        expr = pl.col(column).cast(pl.Date)
        out = frame
        if start is not None:
            out = out.filter(expr >= _date_bound(start))
        if end is not None:
            out = out.filter(expr <= _date_bound(end))
        return out
    if grain == "monthly":
        column = next(
            (candidate for candidate in ("Month", "month", "period") if candidate in frame.columns),
            None,
        )
        if column is None:
            return frame
        out = frame
        if start is not None:
            out = out.filter(pl.col(column).cast(pl.String) >= _month_bound(start))
        if end is not None:
            out = out.filter(pl.col(column).cast(pl.String) <= _month_bound(end))
        return out
    return frame


def _group_columns(
    frame: pl.DataFrame,
    group_by: list[str],
    grain: str,
) -> list[str]:
    grain = model.normalize_grain_name(grain)
    columns: list[str] = []
    for candidate in grain_levels.TIME_COLUMN_CANDIDATES.get(grain, ()):
        if candidate in frame.columns:
            columns.append(candidate)
            break

    for column in group_by:
        if column not in frame.columns:
            raise ValueError(f"group_by column {column!r} not present in aggregate")
        if column not in columns:
            columns.append(column)
    return columns


def _with_calendar_columns(frame: pl.DataFrame, grain: str, physical_level: str) -> pl.DataFrame:
    """Expose camel-case calendar columns for report builders and charts."""

    out = frame
    if "Day" not in out.columns and "day" in out.columns:
        out = out.with_columns(pl.col("day").alias("Day"))
    if "Month" not in out.columns:
        if "month" in out.columns:
            out = out.with_columns(pl.col("month").cast(pl.String).alias("Month"))
        elif physical_level == "monthly" and "period" in out.columns:
            out = out.with_columns(pl.col("period").cast(pl.String).alias("Month"))
    if "Year" not in out.columns and "year" in out.columns:
        out = out.with_columns(pl.col("year").alias("Year"))
    if "Quarter" not in out.columns and "quarter" in out.columns:
        out = out.with_columns(pl.col("quarter").cast(pl.String).alias("Quarter"))

    if "Quarter" not in out.columns and physical_level == "quarterly" and "period" in out.columns:
        out = out.with_columns(pl.col("period").cast(pl.String).alias("Quarter"))
    if "Year" not in out.columns and physical_level == "yearly" and "period" in out.columns:
        out = out.with_columns(pl.col("period").cast(pl.String).alias("Year"))
    return grain_levels.with_calendar_columns(out)


def _metric_group_columns(
    group_columns: list[str],
    metric: model.Metric,
    processor: _Processor,
    frame: pl.DataFrame,
) -> list[str]:
    columns = list(group_columns)
    if isinstance(metric, model.LifecycleSummaryMetric) and isinstance(
        processor, EntityLifecycleProcessor
    ):
        entity_column = processor.entity_column
        if entity_column in frame.columns and entity_column not in columns:
            columns.append(entity_column)
    if (
        isinstance(
            metric,
            model.VariantCompareMetric | model.ContingencyTestMetric | model.ProportionTestMetric,
        )
        and metric.variant_column in frame.columns
        and metric.variant_column not in columns
    ):
        columns.append(metric.variant_column)
    return columns


def _metric_output_columns(  # noqa: PLR0911
    metric_name: str,
    metric: model.Metric,
    frame: pl.DataFrame,
    *,
    include_quantile_suite: bool,
    include_curve_columns: bool,
) -> list[str]:
    if isinstance(metric, model.LifecycleSummaryMetric):
        configured = [column for column in metric.outputs if column in frame.columns]
        return configured or [
            column for column in _LIFECYCLE_OUTPUT_COLUMNS if column in frame.columns
        ]
    if isinstance(metric, model.VariantCompareMetric):
        return _available_columns(frame, [*_VARIANT_OUTPUT_COLUMNS, *metric.outputs])
    if isinstance(metric, model.ContingencyTestMetric):
        return _available_columns(frame, [*_CONTINGENCY_OUTPUT_COLUMNS, *metric.outputs])
    if isinstance(metric, model.ProportionTestMetric):
        return _available_columns(frame, [*_PROPORTION_OUTPUT_COLUMNS, *metric.outputs])
    if isinstance(metric, model.TdigestQuantileMetric):
        if not include_quantile_suite:
            return [metric_name]
        prop = _quantile_property(metric.state)
        suite = [
            metric_name,
            *(f"{prop}_{suffix}" for suffix in _QUANTILE_SUITE),
            f"{prop}_Min",
            f"{prop}_Max",
        ]
        return _available_columns(frame, suite)
    if isinstance(metric, model.CurveFromDigestsMetric):
        if include_curve_columns:
            return _available_columns(frame, [metric_name, *_CURVE_OUTPUT_COLUMNS])
        return [metric_name]
    return [metric_name]


def _available_columns(frame: pl.DataFrame, columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for column in columns:
        if column in frame.columns and column not in out:
            out.append(column)
    return out


def _quantile_property(state: str) -> str:
    for suffix in ("_tdigest", "_kll"):
        if state.endswith(suffix):
            return state.removesuffix(suffix)
    return state


def _cardinality_estimator(
    state: str,
    state_specs: dict[str, model.StateSpec] | None,
) -> Any:
    state_type = state_specs[state].type if state_specs and state in state_specs else None
    if state_type is None:
        if state.endswith("_cpc"):
            state_type = "cpc"
        elif state.endswith("_hll"):
            state_type = "hll"
        elif state.endswith("_theta"):
            state_type = "theta"
    if state_type == "cpc":
        return cpc.estimate
    if state_type == "hll":
        return hll.estimate
    if state_type == "theta":
        return theta.estimate
    raise ValueError(
        f"approx_distinct_count state {state!r} must be configured as a CPC, HLL, or "
        "Theta state"
    )


@timed
def _derive_metric(  # noqa: PLR0911, PLR0912
    frame: pl.DataFrame,
    metric_name: str,
    metric: model.Metric,
    metrics: dict[str, model.Metric],
    *,
    state_specs: dict[str, model.StateSpec] | None = None,
    group_columns: list[str] | None = None,
    include_curve_columns: bool = False,
) -> pl.DataFrame:
    working = frame
    if isinstance(metric, model.FormulaMetric):
        for dep in metric.depends_on:
            dep_metric = metrics.get(dep)
            if dep_metric is None:
                raise ValueError(f"metric {metric_name!r} depends on unknown metric {dep!r}")
            if dep not in working.columns:
                working = _derive_metric(
                    working,
                    dep,
                    dep_metric,
                    metrics,
                    state_specs=state_specs,
                    group_columns=group_columns,
                    include_curve_columns=include_curve_columns,
                )
        return working.with_columns(translate(metric.expression).alias(metric_name))
    if isinstance(metric, model.ApproxDistinctCountMetric):
        if metric.state not in working.columns:
            raise ValueError(f"state {metric.state!r} not present for metric {metric_name!r}")
        estimate = _cardinality_estimator(metric.state, state_specs)
        return working.with_columns(
            pl.col(metric.state).map_elements(estimate, return_dtype=pl.Float64).alias(metric_name)
        )
    if isinstance(metric, model.TopKItemsMetric):
        if metric.state not in working.columns:
            raise ValueError(f"state {metric.state!r} not present for metric {metric_name!r}")
        return working.with_columns(
            pl.col(metric.state)
            .map_elements(
                partial(
                    _topk_metric,
                    limit=metric.limit,
                    error_type=metric.error_type,
                ),
                return_dtype=_topk_dtype(),
            )
            .alias(metric_name)
        )
    if isinstance(metric, model.TdigestQuantileMetric):
        if metric.state not in working.columns:
            raise ValueError(f"state {metric.state!r} not present for metric {metric_name!r}")
        state_type = (
            state_specs[metric.state].type
            if state_specs is not None and metric.state in state_specs
            else "kll"
            if metric.state.endswith("_kll")
            else "tdigest"
        )
        quantile_fn = kll.quantile if state_type == "kll" else tdigest.quantile
        prop = _quantile_property(metric.state)
        quantiles = {
            metric_name: metric.quantile,
            **{f"{prop}_{suffix}": quantile for suffix, quantile in _QUANTILE_SUITE.items()},
        }
        return working.with_columns(
            *[
                pl.col(metric.state)
                .map_elements(
                    partial(
                        _quantile_metric,
                        quantile_fn=quantile_fn,
                        quantile=quantile,
                    ),
                    return_dtype=pl.Float64,
                )
                .alias(column)
                for column, quantile in quantiles.items()
            ]
        )
    if isinstance(metric, model.CurveFromDigestsMetric):
        _ensure_columns(working, [metric.positive_state, metric.negative_state], metric_name)
        curve_col = f"__{metric_name}_curve"
        out = working.with_columns(
            pl.struct([metric.positive_state, metric.negative_state])
            .map_elements(
                _curve_metric,
                return_dtype=_curve_dtype(),
            )
            .alias(curve_col)
        )
        if include_curve_columns:
            out = out.unnest(curve_col)
            if metric_name not in out.columns:
                out = out.with_columns(pl.col(metric.output).alias(metric_name))
            return out
        return out.with_columns(
            pl.col(curve_col).struct.field(metric.output).alias(metric_name)
        ).drop(curve_col)
    if isinstance(metric, model.CalibrationFromDigestsMetric):
        _ensure_columns(working, [metric.positive_state, metric.negative_state], metric_name)
        return working.with_columns(
            pl.struct([metric.positive_state, metric.negative_state])
            .map_elements(_calibration_metric, return_dtype=_calibration_dtype())
            .alias(metric_name)
        )
    if isinstance(metric, model.SetOpMetric):
        states = _set_op_states(metric)
        _ensure_columns(working, states, metric_name)
        return working.with_columns(
            pl.struct(states)
            .map_elements(
                lambda row: _set_op_metric(row, metric, states),
                return_dtype=pl.Float64,
            )
            .alias(metric_name)
        )
    if isinstance(metric, model.FunnelDropoffMetric):
        from_col = f"{metric.from_stage}_Count"
        to_col = f"{metric.to_stage}_Count"
        _ensure_columns(working, [from_col, to_col], metric_name)
        dropoff = pl.col(from_col) - pl.col(to_col)
        if metric.output == "count":
            return working.with_columns(dropoff.alias(metric_name))
        return working.with_columns(
            pl.when(pl.col(from_col) == 0)
            .then(0.0)
            .otherwise(dropoff / pl.col(from_col))
            .alias(metric_name)
        )
    if isinstance(metric, model.LifecycleSummaryMetric):
        return _lifecycle_summary(working, metric)
    if isinstance(metric, model.VariantCompareMetric):
        return _derive_variant_compare(working, metric, group_columns or [])
    if isinstance(metric, model.ContingencyTestMetric):
        return _derive_contingency_test(working, metric, group_columns or [])
    if isinstance(metric, model.ProportionTestMetric):
        return _derive_proportion_test(working, metric, group_columns or [])
    raise NotImplementedError(f"query does not support metric kind {metric.kind!r}")


def _derive_variant_compare(
    frame: pl.DataFrame,
    metric: model.VariantCompareMetric,
    group_columns: list[str],
) -> pl.DataFrame:
    _ensure_columns(frame, [metric.variant_column, "Positives", "Negatives"], "variant_compare")
    out: list[dict[str, Any]] = []
    for groups, rows in _partition_groups(frame, group_columns):
        control = rows.filter(pl.col(metric.variant_column) == metric.control_role)
        test = rows.filter(pl.col(metric.variant_column) == metric.test_role)
        values: dict[str, Any]
        if control.is_empty() or test.is_empty():
            values = dict.fromkeys(_VARIANT_OUTPUT_COLUMNS)
        else:
            values = stats.variant_comparison(
                test_positives=_sum_column(test, "Positives"),
                test_negatives=_sum_column(test, "Negatives"),
                control_positives=_sum_column(control, "Positives"),
                control_negatives=_sum_column(control, "Negatives"),
                confidence_level=metric.confidence_level,
            )
        out.append({**groups, **values})
    return pl.DataFrame(out) if out else _empty_metric_frame(group_columns, _VARIANT_OUTPUT_COLUMNS)


def _derive_contingency_test(
    frame: pl.DataFrame,
    metric: model.ContingencyTestMetric,
    group_columns: list[str],
) -> pl.DataFrame:
    _ensure_columns(frame, [metric.variant_column, "Positives", "Negatives"], "contingency_test")
    out: list[dict[str, Any]] = []
    for groups, rows in _partition_groups(frame, group_columns):
        variants = rows.group_by(metric.variant_column).agg(
            pl.col("Positives").sum(),
            pl.col("Negatives").sum(),
        )
        positives = _sum_column(variants, "Positives")
        negatives = _sum_column(variants, "Negatives")
        tests = stats.contingency_tests(
            zip(variants["Positives"].to_list(), variants["Negatives"].to_list(), strict=True)
        )
        out.append(
            {
                **groups,
                "Count": positives + negatives,
                "Positives": positives,
                "Negatives": negatives,
                **tests,
            }
        )
    return (
        pl.DataFrame(out)
        if out
        else _empty_metric_frame(group_columns, _CONTINGENCY_OUTPUT_COLUMNS)
    )


def _derive_proportion_test(
    frame: pl.DataFrame,
    metric: model.ProportionTestMetric,
    group_columns: list[str],
) -> pl.DataFrame:
    _ensure_columns(frame, [metric.variant_column, "Positives", "Negatives"], "proportion_test")
    out: list[dict[str, Any]] = []
    for groups, rows in _partition_groups(frame, group_columns):
        test = rows.filter(pl.col(metric.variant_column) == metric.test_role)
        control = rows.filter(pl.col(metric.variant_column) == metric.control_role)
        if test.is_empty() or control.is_empty():
            values = dict.fromkeys(_PROPORTION_OUTPUT_COLUMNS)
        else:
            test_positives = _sum_column(test, "Positives")
            test_negatives = _sum_column(test, "Negatives")
            control_positives = _sum_column(control, "Positives")
            control_negatives = _sum_column(control, "Negatives")
            values = {
                "Count": test_positives + test_negatives + control_positives + control_negatives,
                "Positives": test_positives + control_positives,
                "Negatives": test_negatives + control_negatives,
                **stats.proportions_ztest(
                    test_positives=test_positives,
                    test_total=test_positives + test_negatives,
                    control_positives=control_positives,
                    control_total=control_positives + control_negatives,
                ),
            }
        out.append({**groups, **values})
    return (
        pl.DataFrame(out) if out else _empty_metric_frame(group_columns, _PROPORTION_OUTPUT_COLUMNS)
    )


def _partition_groups(
    frame: pl.DataFrame,
    group_columns: list[str],
) -> Iterable[tuple[dict[str, Any], pl.DataFrame]]:
    if not group_columns:
        return [({}, frame)]
    partitions = frame.partition_by(group_columns, as_dict=True)
    return [
        (
            dict(zip(group_columns, key if isinstance(key, tuple) else (key,), strict=True)),
            rows,
        )
        for key, rows in partitions.items()
    ]


def _sum_column(frame: pl.DataFrame, column: str) -> int:
    return int(frame[column].sum() or 0)


def _empty_metric_frame(group_columns: list[str], output_columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict.fromkeys([*group_columns, *output_columns], pl.Float64))


def _quantile_metric(
    payload: _SketchPayload,
    *,
    quantile_fn: Any,
    quantile: float,
) -> float:
    return float(quantile_fn(payload, quantile))


def _topk_metric(
    payload: _SketchPayload,
    *,
    limit: int,
    error_type: Literal["NO_FALSE_POSITIVES", "NO_FALSE_NEGATIVES"],
) -> list[topk.FrequentItem]:
    return topk.frequent_items(payload, error_type=error_type)[:limit]


def _make_processor(
    config: model.Processor,
    *,
    computation_hash: str | None = None,
) -> _Processor:
    return create_processor(config, computation_hash=computation_hash)


def _ensure_columns(frame: pl.DataFrame, columns: list[str], metric_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"metric {metric_name!r} missing state column(s): {', '.join(missing)}")


def _curve_metric(row: dict[str, object]) -> dict[str, object]:
    keys = list(row)
    result = curve_from_digests(
        cast(_SketchPayload, row[keys[0]]),
        cast(_SketchPayload, row[keys[1]]),
    )
    return {
        "roc_auc": result.roc_auc,
        "average_precision": result.average_precision,
        "tpr": list(result.tpr),
        "fpr": list(result.fpr),
        "precision": list(result.precision),
        "recall": list(result.recall),
        "pos_fraction": result.pos_fraction,
    }


def _calibration_metric(row: dict[str, object]) -> dict[str, list[float]]:
    keys = list(row)
    result = calibration_from_digests(
        cast(_SketchPayload, row[keys[0]]),
        cast(_SketchPayload, row[keys[1]]),
    )
    return {
        "bin": list(result.bin),
        "predicted": list(result.predicted),
        "observed": list(result.observed),
    }


def _calibration_dtype() -> pl.DataType:
    return pl.Struct(
        {
            "bin": pl.List(pl.Float64),
            "predicted": pl.List(pl.Float64),
            "observed": pl.List(pl.Float64),
        }
    )


def _curve_dtype() -> pl.DataType:
    return pl.Struct(
        {
            "roc_auc": pl.Float64,
            "average_precision": pl.Float64,
            "tpr": pl.List(pl.Float64),
            "fpr": pl.List(pl.Float64),
            "precision": pl.List(pl.Float64),
            "recall": pl.List(pl.Float64),
            "pos_fraction": pl.Float64,
        }
    )


def _topk_dtype() -> pl.DataType:
    return pl.List(
        pl.Struct(
            {
                "item": pl.String,
                "estimate": pl.UInt64,
                "lower_bound": pl.UInt64,
                "upper_bound": pl.UInt64,
            }
        )
    )


def _set_op_states(metric: model.SetOpMetric) -> list[str]:
    if metric.states:
        return list(metric.states)
    return [operand.state for operand in metric.operands]


def _derive_windowed_set_op(
    frame: pl.DataFrame,
    metric_name: str,
    metric: model.SetOpMetric,
    processor: _Processor,
    *,
    group_columns: list[str],
    anchor: dt.date | None,
) -> pl.DataFrame:
    """Merge one theta operand per configured relative time window."""

    if not metric.operands:
        raise ValueError("time-window set_op requires operands")
    date_column = _set_window_date_column(frame)
    dates = frame[date_column].cast(pl.Date)
    effective_anchor = anchor or cast(dt.date | None, dates.max())
    if effective_anchor is None:
        return _empty_metric_frame(group_columns, [metric_name])

    operand_frames: list[pl.DataFrame] = []
    operand_columns: list[str] = []
    empty_payloads: dict[str, bytes] = {}
    available_groups = [column for column in group_columns if column in frame.columns]
    universe = frame.select(available_groups).unique() if available_groups else None
    for index, operand in enumerate(metric.operands):
        if operand.state not in frame.columns:
            raise ValueError(f"state {operand.state!r} not present for metric {metric_name!r}")
        filtered = _apply_set_time_window(
            frame,
            date_column=date_column,
            window=operand.time_window,
            anchor=effective_anchor,
        )
        merged = processor.merge_for_query(filtered, group_columns)
        alias = f"__set_operand_{index}"
        operand_columns.append(alias)
        spec = processor.state_specs.get(operand.state)
        lg_k = int((spec.model_extra or {}).get("lg_k", 12)) if spec is not None else 12
        empty_payloads[alias] = theta.build([], lg_k=lg_k)
        selected = merged.select(
            *[column for column in group_columns if column in merged.columns],
            pl.col(operand.state).alias(alias),
        )
        if available_groups and universe is not None:
            selected = universe.join(selected, on=available_groups, how="left")
        elif selected.is_empty():
            selected = pl.DataFrame({alias: [empty_payloads[alias]]})
        operand_frames.append(selected)

    combined = operand_frames[0]
    join_columns = [column for column in group_columns if column in combined.columns]
    for operand_frame in operand_frames[1:]:
        if join_columns:
            combined = combined.join(operand_frame, on=join_columns, how="full", coalesce=True)
        else:
            combined = combined.join(operand_frame, how="cross")
    combined = combined.with_columns(
        *[pl.col(column).fill_null(empty_payloads[column]) for column in operand_columns]
    )
    return combined.with_columns(
        pl.struct(operand_columns)
        .map_elements(
            lambda row: _set_op_metric(row, metric, operand_columns),
            return_dtype=pl.Float64,
        )
        .alias(metric_name)
    )


def _set_window_date_column(frame: pl.DataFrame) -> str:
    for column in ("Day", "as_of_date"):
        if column in frame.columns:
            return column
    raise ValueError("time-window set_op requires a daily aggregate with a Day column")


def _apply_set_time_window(
    frame: pl.DataFrame,
    *,
    date_column: str,
    window: dict[str, Any] | None,
    anchor: dt.date,
) -> pl.DataFrame:
    if not window:
        return frame
    dates = pl.col(date_column).cast(pl.Date)
    if "last" in window:
        days = _duration_days(window["last"], allow_negative=False)
        start = anchor - dt.timedelta(days=max(days - 1, 0))
        return frame.filter(dates.is_between(start, anchor, closed="both"))
    between = window.get("between")
    if isinstance(between, list | tuple) and len(between) == 2:
        start_offset = _duration_days(between[0], allow_negative=True)
        end_offset = _duration_days(between[1], allow_negative=True)
        start = anchor + dt.timedelta(days=start_offset)
        end = anchor + dt.timedelta(days=end_offset)
        if start > end:
            start, end = end, start
        return frame.filter(dates.is_between(start, end, closed="both"))
    raise ValueError("set_op time_window must define 'last' or a two-value 'between' range")


def _duration_days(value: object, *, allow_negative: bool) -> int:
    match = re.fullmatch(r"([+-]?)(\d+)([dDwW])", str(value).strip())
    if match is None:
        raise ValueError(f"invalid set_op duration {value!r}; use values such as '7d' or '-1d'")
    sign, amount, unit = match.groups()
    number = int(amount) * (7 if unit.casefold() == "w" else 1)
    if sign == "-":
        number = -number
    if not allow_negative and number <= 0:
        raise ValueError("set_op 'last' duration must be positive")
    return number


def _set_op_metric(row: dict[str, object], metric: model.SetOpMetric, states: list[str]) -> float:
    payloads = [cast(_SketchPayload, row[state]) for state in states]
    if metric.op == "union":
        return theta.estimate(theta.merge(payloads))
    if metric.op == "intersection":
        return theta.estimate(theta.intersect(payloads))
    if metric.op in {"a_not_b", "diff"}:
        if len(payloads) != 2:
            raise ValueError("set_op diff/a_not_b requires exactly two states")
        return theta.estimate(theta.a_not_b(payloads[0], payloads[1]))
    raise ValueError(f"unsupported set_op {metric.op!r}")


def _lifecycle_summary(frame: pl.DataFrame, metric: model.LifecycleSummaryMetric) -> pl.DataFrame:
    _ensure_columns(
        frame,
        ["unique_holdings", "lifetime_value", "MinPurchasedDate", "MaxPurchasedDate"],
        "lifecycle_summary",
    )
    group_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "unique_holdings",
            "lifetime_value",
            "MinPurchasedDate",
            "MaxPurchasedDate",
            "UniquePurchasers_cpc",
            "UniquePurchasers_hll",
            "config_hash",
        }
    ]
    entity_column = _infer_lifecycle_entity_column(group_columns)
    summary = frame.group_by(group_columns).agg(
        pl.n_unique(entity_column).alias("customers_count"),
        pl.col("unique_holdings").sum().alias("unique_holdings"),
        pl.col("lifetime_value").sum().alias("lifetime_value"),
        pl.col("MinPurchasedDate").min().alias("MinPurchasedDate"),
        pl.col("MaxPurchasedDate").max().alias("MaxPurchasedDate"),
    )
    preset = str((metric.model_extra or {}).get("segment_preset", "default"))
    return rfm.with_rfm(summary, segment_preset=cast(rfm.SegmentPreset, preset))


def _infer_lifecycle_entity_column(group_columns: list[str]) -> str:
    for candidate in ("CustomerID", "customer_id", "customer", "Customer"):
        if candidate in group_columns:
            return candidate
    if group_columns:
        return group_columns[-1]
    return "__entity"


def _date_bound(value: dt.date | dt.datetime | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(value[:10])


def _month_bound(value: dt.date | dt.datetime | str) -> str:
    return _date_bound(value).strftime("%Y-%m")


__all__ = ["MetricQueryResult", "QueryProvenance", "query_metric", "query_metric_result"]
