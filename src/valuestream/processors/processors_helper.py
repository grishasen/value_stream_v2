"""Shared helpers for Phase 3 processors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypeVar, cast

import polars as pl

from valuestream.config import model
from valuestream.expr import ast
from valuestream.expr.parser import parse
from valuestream.expr.translator import translate
from valuestream.processors.context import PROVENANCE_COLUMNS, ChunkContext
from valuestream.states import cpc, hll, kll, tdigest, theta, topk

_FrameLike = pl.DataFrame | pl.LazyFrame
_TFrame = TypeVar("_TFrame", pl.DataFrame, pl.LazyFrame)
DistributionSketchSpec = tuple[str, str, int, pl.Expr]
_DISTRIBUTION_SKETCH_STRUCT = "__valuestream_distribution_sketches"


def group_by_columns(config: model.Processor) -> list[str]:
    """Return direct aggregate group-by columns for a processor."""
    return list(config.group_by)


def extra(processor: model.Processor) -> dict[str, Any]:
    return dict(processor.model_extra or {})


def spec_extra(spec: model.StateSpec) -> dict[str, Any]:
    return dict(spec.model_extra or {})


def list_extra(raw_extra: dict[str, Any], key: str) -> list[str]:
    """Return ``raw_extra[key]`` coerced to a list of strings, or ``[]``."""
    raw = raw_extra.get(key, [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def expression(value: Any) -> ast.Expr:
    if isinstance(value, dict):
        return parse(value)
    return cast(ast.Expr, value)


def where_expr(raw_extra: dict[str, Any]) -> pl.Expr | None:
    raw = raw_extra.get("where")
    if raw is None:
        return None
    return translate(expression(raw))


def filtered_column(column: str, raw_extra: dict[str, Any]) -> pl.Expr:
    condition = where_expr(raw_extra)
    base = pl.col(column)
    if condition is None:
        return base
    return pl.when(condition).then(base).otherwise(None)


def count_expr(raw_extra: dict[str, Any], *, alias: str) -> pl.Expr:
    condition = where_expr(raw_extra)
    if condition is None:
        return pl.len().alias(alias)
    return pl.when(condition).then(1).otherwise(0).sum().alias(alias)


def value_sum_expr(column: str, raw_extra: dict[str, Any], *, alias: str) -> pl.Expr:
    condition = where_expr(raw_extra)
    value = pl.col(column).cast(pl.Float64)
    if condition is not None:
        value = pl.when(condition).then(value).otherwise(0.0)
    return value.sum().alias(alias)


def time_column(existing: set[str]) -> str | None:
    for column in ("Day", "day"):
        if column in existing:
            return column
    return None


def period_from_column(column: str | None) -> pl.Expr:
    if column:
        return pl.col(column).cast(pl.String).str.slice(0, 7)
    return pl.lit("ALL")


def with_provenance(
    frame: _TFrame,
    processor_config_hash: str,
    ctx: ChunkContext,
    *,
    period: pl.Expr,
) -> _TFrame:
    return frame.with_columns(
        pl.lit(ctx.pipeline_run_id).alias("pipeline_run_id"),
        pl.lit(ctx.chunk_id).alias("chunk_id"),
        period.alias("period"),
        pl.lit(ctx.created_at).alias("created_at"),
        pl.lit(processor_config_hash).alias("config_hash"),
    )


def with_static_provenance(
    frame: pl.DataFrame,
    processor_config_hash: str,
    ctx: ChunkContext,
) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(ctx.pipeline_run_id).alias("pipeline_run_id"),
        pl.lit(ctx.chunk_id).alias("chunk_id"),
        pl.lit(ctx.created_at).alias("created_at"),
        pl.lit(processor_config_hash).alias("config_hash"),
    )


def with_group_columns(
    frame: pl.DataFrame, group_columns: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    if group_columns:
        return frame, group_columns
    return frame.with_columns(pl.lit("ALL").alias("__valuestream_all")), ["__valuestream_all"]


def default_group_columns(
    frame: pl.DataFrame,
    state_specs: dict[str, model.StateSpec],
) -> list[str]:
    state_names = set(state_specs)
    ignored = state_names | {column for column in PROVENANCE_COLUMNS if column != "period"}
    return [column for column in frame.columns if column not in ignored]


def sketch_build_expr(
    name: str,
    spec: model.StateSpec,
    *,
    existing: set[str],
    default_source_column: str,
) -> tuple[pl.Expr | None, tuple[str, str, int]]:
    """Return a list-aggregation expression and sketch metadata."""
    raw_extra = spec_extra(spec)
    source_column = str(raw_extra.get("source_column", default_source_column))
    if source_column not in existing:
        return None, (name, spec.type, _sketch_k(spec))
    helper = f"__values_{name}"
    values = filtered_column(source_column, raw_extra).drop_nulls()
    if spec.type in {"cpc", "hll", "theta"}:
        values = values.unique()
    return values.alias(helper), (name, spec.type, _sketch_k(spec))


def postprocess_sketches(
    frame: _TFrame,
    sketch_columns: list[tuple[str, str, int]],
) -> _TFrame:
    out = frame
    columns = _frame_columns(out)
    for name, state_type, sketch_k in sketch_columns:
        helper = f"__values_{name}"
        if helper not in columns:
            continue
        build_fn: Callable[..., bytes]
        kwarg: str
        if state_type == "cpc":
            build_fn = cpc.build
            kwarg = "lg_k"
        elif state_type == "hll":
            build_fn = hll.build
            kwarg = "lg_k"
        elif state_type == "theta":
            build_fn = theta.build
            kwarg = "lg_k"
        elif state_type == "tdigest":
            build_fn = tdigest.build
            kwarg = "k"
        elif state_type == "kll":
            build_fn = kll.build
            kwarg = "k"
        elif state_type == "topk":
            build_fn = topk.build
            kwarg = "lg_max_map_size"
        else:
            continue
        out = out.with_columns(
            pl.col(helper)
            .map_elements(
                partial(_build_sketch, build_fn=build_fn, kwarg=kwarg, k=sketch_k),
                return_dtype=pl.Binary,
            )
            .alias(name)
        ).drop(helper)
    return out


def distribution_sketch_expr(sketches: list[DistributionSketchSpec]) -> pl.Expr:
    """Build all t-digest/KLL states for a group through one Python callback."""

    metadata = tuple((name, state_type, k) for name, state_type, k, _ in sketches)
    return pl.map_groups(
        exprs=[expression for _, _, _, expression in sketches],
        function=partial(_build_distribution_sketches, metadata=metadata),
        return_dtype=pl.Struct([pl.Field(name, pl.Binary) for name, _, _ in metadata]),
        returns_scalar=True,
    ).alias(_DISTRIBUTION_SKETCH_STRUCT)


def unnest_distribution_sketches(frame: _TFrame) -> _TFrame:
    """Unnest the shared distribution-sketch struct when it is present."""

    if _DISTRIBUTION_SKETCH_STRUCT not in _frame_columns(frame):
        return frame
    return frame.unnest(_DISTRIBUTION_SKETCH_STRUCT)


def merge_for_query(
    merge: Callable[..., pl.DataFrame],
    frame: pl.DataFrame,
    group_columns: list[str],
    config_hash: str,
) -> pl.DataFrame:
    """Drop provenance, merge rows, and stamp the current processor config hash."""
    working = frame.drop([column for column in PROVENANCE_COLUMNS if column in frame.columns])
    return merge(working, group_columns=group_columns).with_columns(
        pl.lit(config_hash).alias("config_hash")
    )


def compact_state_frame(
    frame: pl.DataFrame,
    state_specs: dict[str, model.StateSpec],
    group_columns: list[str],
    merge: Callable[..., pl.DataFrame],
    *,
    identity_level: bool,
) -> pl.DataFrame:
    """Project unique finest-grain state rows, otherwise use the normal merge.

    A chunk aggregate already has one row per finest-grain key.  Re-merging
    such singleton groups needlessly deserializes and serializes every sketch.
    The native duplicate check keeps this shortcut safe for callers that pass
    arbitrary or already-concatenated partial frames to ``compact``.
    """

    if identity_level and _groups_are_unique(frame, group_columns):
        return frame.select(
            [*group_columns, *_identity_state_expressions(frame, state_specs, group_columns)]
        )
    return merge(frame, group_columns=group_columns)


def merge_state_frame(  # noqa: PLR0912
    frame: pl.DataFrame,
    state_specs: dict[str, model.StateSpec],
    group_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Merge a state frame using the core Phase 3 state rules."""
    if frame.is_empty():
        return frame
    if group_columns is None:
        group_columns = default_group_columns(frame, state_specs)
    working, actual_groups = with_group_columns(frame, group_columns)
    agg_exprs: list[pl.Expr] = []
    sketch_columns: list[tuple[str, str, int]] = []
    for name, spec in state_specs.items():
        if name not in working.columns:
            continue
        if spec.type in {"count", "value_sum"}:
            agg_exprs.append(pl.col(name).sum().alias(name))
        elif spec.type == "min":
            agg_exprs.append(pl.col(name).min().alias(name))
        elif spec.type == "max":
            agg_exprs.append(pl.col(name).max().alias(name))
        elif spec.type in {"cpc", "hll", "theta", "tdigest", "kll", "topk"}:
            helper = f"__merge_{name}"
            agg_exprs.append(pl.col(name).alias(helper))
            sketch_columns.append((name, spec.type, _sketch_k(spec)))
        elif spec.type == "pooled_mean":
            weight = str(spec_extra(spec).get("weight", "Count"))
            if weight in working.columns:
                agg_exprs.append(weighted_mean_expr(name, weight).alias(name))
        elif spec.type == "pooled_variance":
            mean_col, count_col = _variance_companions(name, spec)
            if mean_col not in working.columns or count_col not in working.columns:
                raise ValueError(
                    f"pooled_variance state {name!r} requires companion mean column "
                    f"{mean_col!r} and weight column {count_col!r}; set them via the "
                    "state's 'mean'/'weight' fields"
                )
            group_mean = f"__group_mean_{name}"
            working = working.with_columns(
                weighted_mean_expr(mean_col, count_col).over(actual_groups).alias(group_mean)
            )
            numerator = ((pl.col(count_col) - 1) * pl.col(name).fill_null(0)).sum() + (
                pl.col(count_col) * (pl.col(mean_col) - pl.col(group_mean)).pow(2)
            ).sum()
            agg_exprs.append(
                pl.when(pl.col(count_col).sum() > 1)
                .then(numerator / (pl.col(count_col).sum() - 1))
                .otherwise(None)
                .alias(name)
            )
    merged = working.group_by(actual_groups).agg(agg_exprs)
    if not group_columns:
        merged = merged.drop("__valuestream_all")
    for name, state_type, sketch_k in sketch_columns:
        helper = f"__merge_{name}"
        if state_type == "cpc":
            merged = _merge_sketch_column(merged, name, helper, cpc.merge, "lg_k", sketch_k)
        elif state_type == "hll":
            merged = _merge_sketch_column(merged, name, helper, hll.merge, "lg_k", sketch_k)
        elif state_type == "theta":
            merged = _merge_sketch_column(merged, name, helper, theta.merge, "lg_k", sketch_k)
        elif state_type == "tdigest":
            merged = _merge_sketch_column(merged, name, helper, tdigest.merge, "k", sketch_k)
        elif state_type == "kll":
            merged = _merge_sketch_column(merged, name, helper, kll.merge, "k", sketch_k)
        elif state_type == "topk":
            merged = _merge_sketch_column(
                merged,
                name,
                helper,
                topk.merge,
                "lg_max_map_size",
                sketch_k,
            )
    return merged


def ensure_state_columns(
    frame: _TFrame,
    state_specs: dict[str, model.StateSpec],
) -> _TFrame:
    """Materialize missing state columns with neutral values."""
    out = frame
    columns = _frame_columns(out)
    for name, spec in state_specs.items():
        if name in columns:
            continue
        sketch_k = _sketch_k(spec)
        if spec.type == "cpc":
            out = out.with_columns(pl.lit(cpc.build([], lg_k=sketch_k)).alias(name))
        elif spec.type == "hll":
            out = out.with_columns(pl.lit(hll.build([], lg_k=sketch_k)).alias(name))
        elif spec.type == "theta":
            out = out.with_columns(pl.lit(theta.build([], lg_k=sketch_k)).alias(name))
        elif spec.type == "tdigest":
            out = out.with_columns(pl.lit(tdigest.build([], k=sketch_k)).alias(name))
        elif spec.type == "kll":
            out = out.with_columns(pl.lit(kll.build([], k=sketch_k)).alias(name))
        elif spec.type == "topk":
            out = out.with_columns(pl.lit(topk.build([], lg_max_map_size=sketch_k)).alias(name))
        elif spec.type in {"value_sum", "min", "max", "pooled_mean", "pooled_variance"}:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
        else:
            out = out.with_columns(pl.lit(0).alias(name))
    return out


def _frame_columns(frame: _FrameLike) -> set[str]:
    if isinstance(frame, pl.LazyFrame):
        return set(frame.collect_schema().names())
    return set(frame.columns)


def _groups_are_unique(frame: pl.DataFrame, group_columns: list[str]) -> bool:
    if frame.height <= 1:
        return True
    if not group_columns:
        return False
    return not frame.select(group_columns).is_duplicated().any()


def _identity_state_expressions(
    frame: pl.DataFrame,
    state_specs: dict[str, model.StateSpec],
    group_columns: list[str],
) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for name, spec in state_specs.items():
        if name not in frame.columns or name in group_columns:
            continue
        if spec.type == "pooled_mean":
            weight = str(spec_extra(spec).get("weight", "Count"))
            if weight in frame.columns:
                expressions.append(_singleton_weighted_mean_expr(name, weight).alias(name))
            continue
        if spec.type == "pooled_variance":
            mean_col, count_col = _variance_companions(name, spec)
            if mean_col not in frame.columns or count_col not in frame.columns:
                raise ValueError(
                    f"pooled_variance state {name!r} requires companion mean column "
                    f"{mean_col!r} and weight column {count_col!r}; set them via the "
                    "state's 'mean'/'weight' fields"
                )
            count = pl.col(count_col)
            group_mean = _singleton_weighted_mean_expr(mean_col, count_col)
            within = ((count - 1) * pl.col(name).fill_null(0)).fill_null(0)
            between = (count * (pl.col(mean_col) - group_mean).pow(2)).fill_null(0)
            count_sum = count.fill_null(0)
            expressions.append(
                pl.when(count_sum > 1)
                .then((within + between) / (count_sum - 1))
                .otherwise(None)
                .alias(name)
            )
            continue
        expressions.append(pl.col(name))
    return expressions


def _singleton_weighted_mean_expr(value_col: str, weight_col: str) -> pl.Expr:
    weight = pl.col(weight_col)
    return (pl.col(value_col) * weight).fill_null(0.0) / weight.fill_null(0)


def series_or_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, pl.Series):
        return value.to_list()
    if isinstance(value, list):
        return value
    return [value]


def _build_sketch(values: Any, build_fn: Callable[..., bytes], kwarg: str, k: int) -> bytes:
    return build_fn(series_or_list(values), **{kwarg: k})


def _build_distribution_sketches(
    values: Sequence[pl.Series],
    *,
    metadata: tuple[tuple[str, str, int], ...],
) -> dict[str, bytes]:
    sketches: dict[str, bytes] = {}
    for series, (name, state_type, k) in zip(values, metadata, strict=True):
        build_fn = kll.build if state_type == "kll" else tdigest.build
        sketches[name] = build_fn(series, k=k)
    return sketches


def _merge_sketch_column(
    frame: pl.DataFrame,
    name: str,
    helper: str,
    merge_fn: Callable[..., bytes],
    kwarg: str,
    k: int,
) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(helper)
        .map_elements(
            lambda values: merge_fn(series_or_list(values), **{kwarg: k}),
            return_dtype=pl.Binary,
        )
        .alias(name)
    ).drop(helper)


def weighted_mean_expr(value_col: str, weight_col: str) -> pl.Expr:
    """Return a count-weighted mean expression: ``sum(value*weight)/sum(weight)``."""
    return (pl.col(value_col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()


def _variance_companions(name: str, spec: model.StateSpec) -> tuple[str, str]:
    extra = spec_extra(spec)
    base = name[:-4] if name.endswith("_Var") else name
    mean_col = str(extra.get("mean", f"{base}_Mean"))
    count_col = str(extra.get("weight", f"{base}_Count"))
    return mean_col, count_col


def _sketch_k(spec: model.StateSpec) -> int:
    raw_extra = spec_extra(spec)
    if spec.type == "topk":
        return int(raw_extra.get("lg_max_map_size", raw_extra.get("k", 10)))
    if spec.type == "cpc":
        return int(raw_extra.get("lg_k", 11))
    if spec.type in {"hll", "theta"}:
        return int(raw_extra.get("lg_k", 12))
    if spec.type == "kll":
        return int(raw_extra.get("k", 200))
    return int(raw_extra.get("k", 500))


__all__ = [
    "DistributionSketchSpec",
    "compact_state_frame",
    "count_expr",
    "default_group_columns",
    "distribution_sketch_expr",
    "ensure_state_columns",
    "expression",
    "extra",
    "filtered_column",
    "group_by_columns",
    "list_extra",
    "merge_for_query",
    "merge_state_frame",
    "period_from_column",
    "postprocess_sketches",
    "series_or_list",
    "sketch_build_expr",
    "spec_extra",
    "time_column",
    "unnest_distribution_sketches",
    "value_sum_expr",
    "weighted_mean_expr",
    "where_expr",
    "with_group_columns",
    "with_provenance",
    "with_static_provenance",
]
