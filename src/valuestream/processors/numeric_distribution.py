"""Phase 2 ``numeric_distribution`` processor."""

from __future__ import annotations

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.context import ChunkContext
from valuestream.utils.timer import timed


class NumericDistributionProcessor:
    """Aggregate numeric property distributions into mergeable state rows."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.NumericDistributionProcessor):
            raise TypeError(f"expected numeric_distribution processor, got {config.kind!r}")
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
    def properties(self) -> list[str]:
        return list(self.config.properties)

    @property
    def quantile_engine(self) -> str:
        return self.config.quantile_engine

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return daily descriptive aggregates for one transformed source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(  # noqa: PLR0912
        self, frame: pl.LazyFrame, ctx: ChunkContext
    ) -> pl.LazyFrame:
        """Return a lazy daily descriptive aggregate plan for one source chunk."""
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        schema = source.collect_schema()
        existing = set(schema.names())
        time_columns = grain_levels.chunk_time_group_columns(existing, self.config)
        group_keys = list(
            dict.fromkeys(
                column
                for column in [*self.group_by_columns, *time_columns]
                if column and column in existing
            )
        )

        agg_exprs: list[pl.Expr] = []
        distribution_sketches: list[p3.DistributionSketchSpec] = []
        remaining_sketches: list[tuple[pl.Expr, tuple[str, str, int]]] = []
        for name, spec in self.state_specs.items():
            raw_extra = p3.spec_extra(spec)
            if spec.type == "count":
                if spec.source_column:
                    _require_source_column(self.id, name, spec.source_column, existing)
                agg_exprs.append(p3.count_expr(raw_extra, alias=name))
            elif spec.type == "value_sum":
                _require_source_column(self.id, name, spec.source_column, existing)
                agg_exprs.append(p3.value_sum_expr(spec.source_column, raw_extra, alias=name))
            elif spec.type == "pooled_mean":
                if spec.source_column is None:
                    raise ValueError(
                        f"numeric_distribution processor {self.id!r} state {name!r} "
                        "requires source_column"
                    )
                _require_source_column(self.id, name, spec.source_column, existing)
                agg_exprs.append(pl.col(spec.source_column).mean().alias(name))
            elif spec.type == "pooled_variance":
                _require_source_column(self.id, name, spec.source_column, existing)
                agg_exprs.append(pl.col(spec.source_column).var().alias(name))
            elif spec.type == "min":
                _require_source_column(self.id, name, spec.source_column, existing)
                agg_exprs.append(pl.col(spec.source_column).min().alias(name))
            elif spec.type == "max":
                _require_source_column(self.id, name, spec.source_column, existing)
                agg_exprs.append(pl.col(spec.source_column).max().alias(name))
            elif spec.type in {"tdigest", "kll"}:
                _require_source_column(self.id, name, spec.source_column, existing)
                distribution_sketches.append(
                    (name, spec.type, _sketch_k(spec), pl.col(spec.source_column).drop_nulls())
                )
            elif spec.type in {"cpc", "hll", "theta", "topk"}:
                expression, metadata = p3.sketch_build_expr(
                    name,
                    spec,
                    existing=existing,
                    source_dtypes=schema,
                )
                if expression is not None:
                    remaining_sketches.append((expression, metadata))
        if distribution_sketches:
            agg_exprs.append(p3.distribution_sketch_expr(distribution_sketches))
        agg_exprs.extend(expression for expression, _ in remaining_sketches)
        grouped = source.group_by(group_keys).agg(agg_exprs)
        grouped = p3.unnest_distribution_sketches(grouped)
        grouped = p3.postprocess_sketches(
            grouped,
            [metadata for _, metadata in remaining_sketches],
        )
        grouped = self._ensure_state_columns(grouped)
        return p3.with_provenance(
            grouped,
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
            self.config, target_grain, "numeric_distribution"
        )
        plan = grain_levels.prepare_compaction(
            frame,
            config=self.config,
            state_specs=self.state_specs,
            target_grain=target_grain,
        )
        return p3.with_static_provenance(
            p3.compact_state_frame(
                plan.frame,
                self.state_specs,
                plan.group_columns,
                self.merge,
                identity_level=(
                    self.config.aggregation_level_for(target_grain)
                    == grain_levels.finest_configured_level(self.config)
                ),
            ),
            self.config_hash,
            ctx,
        )

    @timed
    def merge(self, frame: pl.DataFrame, group_columns: list[str] | None = None) -> pl.DataFrame:
        """Merge partial aggregate rows using Phase 2 state rules."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve config hash for query-time metrics."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _ensure_state_columns(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return p3.ensure_state_columns(frame, self.state_specs)


def _sketch_k(spec: model.StateSpec) -> int:
    return int(getattr(spec, "k", None) or (500 if spec.type == "tdigest" else 200))


def _require_source_column(
    processor_id: str,
    state_name: str,
    source_column: str,
    existing: set[str],
) -> None:
    if source_column not in existing:
        raise ValueError(
            f"numeric_distribution processor {processor_id!r} state {state_name!r} "
            f"requires missing source column {source_column!r}"
        )


__all__ = ["NumericDistributionProcessor"]
