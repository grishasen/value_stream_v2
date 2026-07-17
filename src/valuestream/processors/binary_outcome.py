"""Phase 1 ``binary_outcome`` processor."""

from __future__ import annotations

from typing import TypeVar

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.context import PROVENANCE_COLUMNS, ChunkContext
from valuestream.processors.outcomes import compatible_values, is_in_values, parse_outcome
from valuestream.utils.timer import timed

_TFrame = TypeVar("_TFrame", pl.DataFrame, pl.LazyFrame)


class BinaryOutcomeProcessor:
    """Aggregate positive/negative outcomes into mergeable state rows."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.BinaryOutcomeProcessor):
            raise TypeError(f"expected binary_outcome processor, got {config.kind!r}")
        self.config = config
        self.config_hash = computation_hash or processor_config_hash(config)

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def source_id(self) -> str:
        return self.config.source

    @property
    def group_by_columns(self) -> list[str]:
        return list(self.config.group_by)

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return daily partial aggregates for one transformed source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy daily partial aggregate plan for one transformed source chunk."""
        extra = p3.extra(self.config)
        outcome = parse_outcome(extra)
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        schema = source.collect_schema()
        positive_values = compatible_values(outcome.positive_values, schema[outcome.column])
        negative_values = compatible_values(outcome.negative_values, schema[outcome.column])
        source = source.filter(
            is_in_values(outcome.column, [*positive_values, *negative_values])
        ).with_columns(
            is_in_values(outcome.column, positive_values)
            .cast(pl.Int64)
            .alias("__valuestream_positive")
        )

        dedup_keys = [
            key
            for key in p3.list_extra(extra, "dedup_keys")
            if key in source.collect_schema().names()
        ]
        if dedup_keys:
            source = source.filter(
                pl.col("__valuestream_positive")
                == pl.col("__valuestream_positive").max().over(dedup_keys)
            ).unique(subset=dedup_keys, keep="first")

        source_schema = source.collect_schema()
        existing = set(source_schema.names())
        time_columns = grain_levels.chunk_time_group_columns(existing, self.config)
        group_keys = [
            column
            for column in [*self.group_by_columns, *time_columns]
            if column and column in existing
        ]
        variant_column = extra.get("variant_column")
        if isinstance(variant_column, str) and variant_column in existing:
            group_keys.append(variant_column)

        agg_exprs = self._agg_exprs(existing, source_schema)
        grouped_lazy = source.group_by(group_keys).agg(agg_exprs)
        grouped_lazy = p3.postprocess_sketches(grouped_lazy, self._sketch_columns(existing))
        grouped_lazy = self._ensure_state_columns(grouped_lazy)
        grouped_lazy = self._with_negatives(grouped_lazy)
        return p3.with_provenance(
            grouped_lazy,
            self.config_hash,
            ctx,
            period=grain_levels.base_period_expr(existing, self.config),
        )

    @timed
    def compact(self, frame: pl.DataFrame, target_grain: str, ctx: ChunkContext) -> pl.DataFrame:
        """Compact a base partial frame to the target grain's configured level."""
        if frame.is_empty():
            return frame
        target_grain = grain_levels.normalize_target_grain(
            self.config, target_grain, "binary_outcome"
        )
        plan = grain_levels.prepare_compaction(
            frame,
            config=self.config,
            state_specs=self.state_specs,
            target_grain=target_grain,
        )
        merged = p3.compact_state_frame(
            plan.frame,
            self.state_specs,
            plan.group_columns,
            self.merge,
            identity_level=(
                self.config.aggregation_level_for(target_grain)
                == grain_levels.finest_configured_level(self.config)
            ),
        )
        merged = self._with_negatives(merged)
        return p3.with_static_provenance(merged, self.config_hash, ctx)

    @timed
    def merge(self, frame: pl.DataFrame, group_columns: list[str] | None = None) -> pl.DataFrame:
        """Merge partial aggregate rows using state-specific rules."""
        if frame.is_empty():
            return frame
        return self._with_negatives(p3.merge_state_frame(frame, self.state_specs, group_columns))

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve the current processor config hash for query-time formulas."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _agg_exprs(self, existing: set[str], source_schema: pl.Schema) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        sketch_helpers: list[pl.Expr] = []
        for name, spec in self.state_specs.items():
            extra = p3.spec_extra(spec)
            if name == "Count":
                exprs.append(pl.len().alias(name))
            elif name == "Positives":
                exprs.append(pl.col("__valuestream_positive").sum().alias(name))
            elif name == "Negatives":
                continue
            elif spec.type == "value_sum":
                source_column = str(extra.get("source_column", name))
                if source_column in existing:
                    exprs.append(pl.col(source_column).sum().cast(pl.Float64).alias(name))
            elif spec.type == "min":
                source_column = str(extra.get("source_column", name))
                if source_column in existing:
                    exprs.append(pl.col(source_column).min().alias(name))
            elif spec.type == "max":
                source_column = str(extra.get("source_column", name))
                if source_column in existing:
                    exprs.append(pl.col(source_column).max().alias(name))
            elif spec.type in {"cpc", "hll", "theta", "topk"}:
                sketch_expr, _ = p3.sketch_build_expr(
                    name,
                    spec,
                    existing=existing,
                    default_source_column=name,
                    source_dtypes=source_schema,
                )
                if sketch_expr is not None:
                    sketch_helpers.append(sketch_expr)
        return [*exprs, *sketch_helpers]

    def _sketch_columns(self, existing: set[str]) -> list[tuple[str, str, int]]:
        columns: list[tuple[str, str, int]] = []
        for name, spec in self.state_specs.items():
            if spec.type not in {"cpc", "hll", "theta", "topk"}:
                continue
            _, metadata = p3.sketch_build_expr(
                name,
                spec,
                existing=existing,
                default_source_column=name,
            )
            columns.append(metadata)
        return columns

    def _ensure_state_columns(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        out = frame
        columns = set(out.collect_schema().names())
        for name, spec in self.state_specs.items():
            if name in columns or name == "Negatives":
                continue
            if spec.type in {"cpc", "hll", "theta", "topk"}:
                out = p3.ensure_state_columns(out, {name: spec})
            elif spec.type == "value_sum":
                out = out.with_columns(pl.lit(0.0).alias(name))
            else:
                out = out.with_columns(pl.lit(0).alias(name))
        return out

    def _with_negatives(self, frame: _TFrame) -> _TFrame:
        columns = (
            set(frame.collect_schema().names())
            if isinstance(frame, pl.LazyFrame)
            else set(frame.columns)
        )
        if {"Count", "Positives"} <= columns:
            return frame.with_columns((pl.col("Count") - pl.col("Positives")).alias("Negatives"))
        return frame


__all__ = ["PROVENANCE_COLUMNS", "BinaryOutcomeProcessor", "ChunkContext"]
