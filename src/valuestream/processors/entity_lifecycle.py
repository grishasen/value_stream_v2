"""Phase 3 ``entity_lifecycle`` processor."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class LifecycleKeys:
    """Column mapping for lifecycle aggregation."""

    customer_id: str
    order_id: str
    monetary: str
    purchase_date: str


class EntityLifecycleProcessor:
    """Aggregate customer purchase lifecycle state for RFM/CLV metrics."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.EntityLifecycleProcessor):
            raise TypeError(f"expected entity_lifecycle processor, got {config.kind!r}")
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
    def keys(self) -> LifecycleKeys:
        return LifecycleKeys(**model.entity_lifecycle_keys(self.config))

    @property
    def entity_column(self) -> str:
        return self.keys.customer_id

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return lifecycle state rows for one source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy lifecycle state aggregate plan for one source chunk."""
        keys = self.keys
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))
        source = self._apply_lifespan_filter(source, keys.purchase_date)
        source = source.with_columns(
            pl.col(keys.monetary).cast(pl.Float64).alias("__lifecycle_monetary"),
            pl.col(keys.purchase_date).dt.year().cast(pl.String).alias("Year"),
            pl.concat_str(
                [
                    pl.col(keys.purchase_date).dt.year().cast(pl.String),
                    pl.lit("_Q"),
                    pl.col(keys.purchase_date).dt.quarter().cast(pl.String),
                ]
            ).alias("Quarter"),
        )

        existing = set(source.collect_schema().names())
        group_keys = [
            column
            for column in [*self.group_by_columns, keys.customer_id, "Year", "Quarter"]
            if column in existing
        ]
        grouped = source.group_by(group_keys).agg(self._agg_exprs(existing))
        grouped = p3.postprocess_sketches(grouped, self._sketch_columns(existing))
        grouped = p3.ensure_state_columns(grouped, self.state_specs)
        return p3.with_provenance(
            grouped,
            self.config_hash,
            ctx,
            period=pl.col("MaxPurchasedDate").cast(pl.String).str.slice(0, 7),
        )

    @timed
    def compact(self, frame: pl.DataFrame, target_grain: str, ctx: ChunkContext) -> pl.DataFrame:
        """Compact lifecycle state to the target grain's configured level."""
        if frame.is_empty():
            return frame
        target_grain = grain_levels.normalize_target_grain(
            self.config, target_grain, "entity_lifecycle"
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
        """Merge lifecycle state rows."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve config hash for query-time metrics."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _agg_exprs(self, existing: set[str]) -> list[pl.Expr]:
        keys = self.keys
        exprs: list[pl.Expr] = []
        for name, spec in self.state_specs.items():
            raw_extra = p3.spec_extra(spec)
            source_column = str(raw_extra.get("source_column", name))
            if name == "unique_holdings" and keys.order_id in existing:
                exprs.append(pl.col(keys.order_id).n_unique().alias(name))
            elif name == "lifetime_value":
                exprs.append(pl.col("__lifecycle_monetary").sum().alias(name))
            elif name == "MinPurchasedDate" and keys.purchase_date in existing:
                exprs.append(pl.col(keys.purchase_date).min().alias(name))
            elif name == "MaxPurchasedDate" and keys.purchase_date in existing:
                exprs.append(pl.col(keys.purchase_date).max().alias(name))
            elif spec.type == "count":
                exprs.append(p3.count_expr(raw_extra, alias=name))
            elif spec.type == "value_sum" and source_column in existing:
                exprs.append(p3.value_sum_expr(source_column, raw_extra, alias=name))
            elif spec.type == "min" and source_column in existing:
                exprs.append(p3.filtered_column(source_column, raw_extra).min().alias(name))
            elif spec.type == "max" and source_column in existing:
                exprs.append(p3.filtered_column(source_column, raw_extra).max().alias(name))
            elif spec.type in {"cpc", "hll", "theta", "topk"}:
                expr, _ = p3.sketch_build_expr(
                    name,
                    spec,
                    existing=existing,
                    default_source_column=keys.customer_id,
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
                default_source_column=self.keys.customer_id,
            )
            columns.append(meta)
        return columns

    def _apply_lifespan_filter(self, frame: pl.LazyFrame, purchase_date: str) -> pl.LazyFrame:
        lifespan_years = p3.extra(self.config).get("lifespan_years")
        if lifespan_years is None:
            return frame
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=365 * int(lifespan_years))
        return frame.filter(pl.col(purchase_date) >= pl.lit(cutoff).cast(pl.Datetime("us")))


__all__ = ["EntityLifecycleProcessor", "LifecycleKeys"]
