"""Phase 3 ``entity_set`` processor."""

from __future__ import annotations

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.utils.timer import timed


class EntitySetProcessor:
    """Aggregate mergeable approximate sets for cohort and overlap metrics."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.EntitySetProcessor):
            raise TypeError(f"expected entity_set processor, got {config.kind!r}")
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
        return p3.group_by_columns(self.config)

    @property
    def entity_column(self) -> str:
        return model.entity_set_column(self.config)

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return partial approximate-set aggregates for one source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy approximate-set aggregate plan for one source chunk."""
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        source_schema = source.collect_schema()
        existing = set(source_schema.names())
        time_columns = grain_levels.chunk_time_group_columns(existing, self.config)
        group_keys = [
            column
            for column in [*self.group_by_columns, *time_columns]
            if column and column in existing
        ]
        grouped = source.group_by(group_keys).agg(self._agg_exprs(existing, source_schema))
        grouped = p3.postprocess_sketches(grouped, self._sketch_columns(existing))
        grouped = p3.ensure_state_columns(grouped, self.state_specs)
        return p3.with_provenance(
            grouped,
            self.config_hash,
            ctx,
            period=grain_levels.base_period_expr(existing, self.config),
        )

    @timed
    def compact(self, frame: pl.DataFrame, target_grain: str, ctx: ChunkContext) -> pl.DataFrame:
        """Compact a base set aggregate to the target grain's configured level."""
        if frame.is_empty():
            return frame
        target_grain = grain_levels.normalize_target_grain(self.config, target_grain, "entity_set")
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
        """Merge partial set aggregate rows."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve config hash for query-time metrics."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _agg_exprs(self, existing: set[str], source_schema: pl.Schema) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        for name, spec in self.state_specs.items():
            raw_extra = p3.spec_extra(spec)
            if spec.type == "count":
                exprs.append(p3.count_expr(raw_extra, alias=name))
            elif spec.type == "value_sum":
                source_column = str(raw_extra.get("source_column", name))
                if source_column in existing:
                    exprs.append(p3.value_sum_expr(source_column, raw_extra, alias=name))
            elif spec.type == "min":
                source_column = str(raw_extra.get("source_column", name))
                if source_column in existing:
                    exprs.append(p3.filtered_column(source_column, raw_extra).min().alias(name))
            elif spec.type == "max":
                source_column = str(raw_extra.get("source_column", name))
                if source_column in existing:
                    exprs.append(p3.filtered_column(source_column, raw_extra).max().alias(name))
            elif spec.type in {"cpc", "hll", "theta", "topk"}:
                expr, _ = p3.sketch_build_expr(
                    name,
                    spec,
                    existing=existing,
                    default_source_column=self.entity_column,
                    source_dtypes=source_schema,
                )
                if expr is not None:
                    exprs.append(expr)
        return exprs

    def _sketch_columns(self, existing: set[str]) -> list[tuple[str, str, int]]:
        columns: list[tuple[str, str, int]] = []
        for name, spec in self.state_specs.items():
            if spec.type not in {"cpc", "hll", "theta", "topk"}:
                continue
            _, meta = p3.sketch_build_expr(
                name,
                spec,
                existing=existing,
                default_source_column=self.entity_column,
            )
            columns.append(meta)
        return columns


__all__ = ["EntitySetProcessor"]
