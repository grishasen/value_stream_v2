"""Phase 2 ``numeric_distribution`` processor."""

from __future__ import annotations

from functools import partial
from typing import Any

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.context import ChunkContext
from valuestream.states import kll, tdigest
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
        return p3.list_extra(p3.extra(self.config), "properties")

    @property
    def quantile_engine(self) -> str:
        return str(p3.extra(self.config).get("quantile_engine", "tdigest"))

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return daily descriptive aggregates for one transformed source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy daily descriptive aggregate plan for one source chunk."""
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        schema = source.collect_schema()
        existing = set(schema.names())
        properties = [prop for prop in self.properties if prop in existing]
        numeric_props = [prop for prop in properties if schema[prop].is_numeric()]
        time_columns = grain_levels.chunk_time_group_columns(existing, self.config)
        group_keys = [
            column
            for column in [*self.group_by_columns, *time_columns]
            if column and column in existing
        ]

        agg_exprs: list[pl.Expr] = []
        for prop in properties:
            agg_exprs.append(pl.col(prop).count().alias(f"{prop}_Count"))
        for prop in numeric_props:
            agg_exprs.extend(
                [
                    pl.col(prop).sum().cast(pl.Float64).alias(f"{prop}_Sum"),
                    pl.col(prop).mean().alias(f"{prop}_Mean"),
                    pl.col(prop).var().alias(f"{prop}_Var"),
                    pl.col(prop).min().alias(f"{prop}_Min"),
                    pl.col(prop).max().alias(f"{prop}_Max"),
                    pl.col(prop).drop_nulls().alias(f"__values_{prop}"),
                ]
            )
        generic_sketches = self._generic_sketch_columns(existing, numeric_props)
        agg_exprs.extend(expression for expression, _ in generic_sketches)
        grouped = source.group_by(group_keys).agg(agg_exprs)
        grouped = self._postprocess_sketches(grouped, numeric_props)
        grouped = p3.postprocess_sketches(
            grouped,
            [metadata for _, metadata in generic_sketches],
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
            self.merge(plan.frame, group_columns=plan.group_columns),
            self.config_hash,
            ctx,
        )

    @timed
    def merge(
        self, frame: pl.DataFrame, group_columns: list[str] | None = None
    ) -> pl.DataFrame:
        """Merge partial aggregate rows using Phase 2 state rules."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve config hash for query-time metrics."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _postprocess_sketches(self, frame: pl.LazyFrame, numeric_props: list[str]) -> pl.LazyFrame:
        out = frame
        columns = set(out.collect_schema().names())
        for prop in numeric_props:
            helper = f"__values_{prop}"
            if helper not in columns:
                continue
            if self.quantile_engine == "kll":
                out = out.with_columns(
                    pl.col(helper)
                    .map_elements(
                        partial(_build_kll, k=_sketch_k(self.state_specs[f"{prop}_kll"])),
                        return_dtype=pl.Binary,
                    )
                    .alias(f"{prop}_kll")
                )
            else:
                out = out.with_columns(
                    pl.col(helper)
                    .map_elements(
                        partial(_build_tdigest, k=_sketch_k(self.state_specs[f"{prop}_tdigest"])),
                        return_dtype=pl.Binary,
                    )
                    .alias(f"{prop}_tdigest")
                )
            out = out.drop(helper)
        return out

    def _ensure_state_columns(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        return p3.ensure_state_columns(frame, self.state_specs)

    def _generic_sketch_columns(
        self,
        existing: set[str],
        numeric_props: list[str],
    ) -> list[tuple[pl.Expr, tuple[str, str, int]]]:
        columns: list[tuple[pl.Expr, tuple[str, str, int]]] = []
        standard_digest_names = {
            f"{prop}_{self.quantile_engine}" for prop in numeric_props
        }
        for name, spec in self.state_specs.items():
            if spec.type not in {"tdigest", "kll", "cpc", "hll", "theta", "topk"}:
                continue
            if name in standard_digest_names:
                continue
            expression, metadata = p3.sketch_build_expr(
                name,
                spec,
                existing=existing,
                default_source_column=_state_source_column(name, spec),
            )
            if expression is not None:
                columns.append((expression, metadata))
        return columns


def _state_source_column(name: str, spec: model.StateSpec) -> str:
    if source_column := p3.spec_extra(spec).get("source_column"):
        return str(source_column)
    for suffix in ("_tdigest", "_kll", "_cpc", "_hll", "_theta", "_topk"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def _sketch_k(spec: model.StateSpec) -> int:
    extra = dict(spec.model_extra or {})
    return int(extra.get("k", 500 if spec.type == "tdigest" else 200))


def _build_tdigest(values: Any, k: int) -> bytes:
    return tdigest.build(p3.series_or_list(values), k=k)


def _build_kll(values: Any, k: int) -> bytes:
    return kll.build(p3.series_or_list(values), k=k)


__all__ = ["NumericDistributionProcessor"]
