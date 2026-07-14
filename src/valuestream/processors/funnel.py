"""Phase 3 ``funnel`` processor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class FunnelStage:
    """One configured funnel stage."""

    name: str
    when: Any


class FunnelProcessor:
    """Aggregate per-stage funnel counts and optional cardinality sketches."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.FunnelProcessor):
            raise TypeError(f"expected funnel processor, got {config.kind!r}")
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
    def stages(self) -> list[FunnelStage]:
        raw = p3.extra(self.config).get("stages", [])
        if not isinstance(raw, list):
            return []
        stages: list[FunnelStage] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            stages.append(FunnelStage(name=str(item["name"]), when=item["when"]))
        return stages

    @property
    def entity_column(self) -> str | None:
        raw = p3.extra(self.config).get("entity")
        return str(raw) if raw is not None else None

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return per-stage counts for one source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy per-stage aggregate plan for one source chunk."""
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))

        existing = set(source.collect_schema().names())
        time_columns = grain_levels.chunk_time_group_columns(existing, self.config)
        group_keys = [
            column
            for column in [*self.group_by_columns, *time_columns]
            if column and column in existing
        ]
        grouped = source.group_by(group_keys).agg(self._agg_exprs(existing))
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
        """Compact a base funnel aggregate to the target grain's configured level."""
        if frame.is_empty():
            return frame
        target_grain = grain_levels.normalize_target_grain(self.config, target_grain, "funnel")
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
    def merge(self, frame: pl.DataFrame, group_columns: list[str] | None = None) -> pl.DataFrame:
        """Merge partial funnel aggregate rows."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve config hash for query-time metrics."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _agg_exprs(self, existing: set[str]) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        entity = self.entity_column
        for stage in self.stages:
            condition = translate(p3.expression(stage.when))
            exprs.append(pl.when(condition).then(1).otherwise(0).sum().alias(f"{stage.name}_Count"))
            if entity is not None and entity in existing:
                for name, _ in self._stage_cardinality_states(stage):
                    exprs.append(
                        pl.when(condition)
                        .then(pl.col(entity))
                        .otherwise(None)
                        .drop_nulls()
                        .unique()
                        .alias(f"__values_{name}")
                    )
        stage_states = self._stage_state_names()
        for name, spec in self.state_specs.items():
            if name in stage_states or spec.type not in {"cpc", "hll", "theta", "topk"}:
                continue
            expression, _ = p3.sketch_build_expr(
                name,
                spec,
                existing=existing,
                default_source_column=entity or name,
            )
            if expression is not None:
                exprs.append(expression)
        return exprs

    def _sketch_columns(self, existing: set[str]) -> list[tuple[str, str, int]]:
        entity = self.entity_column
        columns: list[tuple[str, str, int]] = []
        if entity is not None and entity in existing:
            for stage in self.stages:
                for name, spec in self._stage_cardinality_states(stage):
                    default = 11 if spec.type == "cpc" else 12
                    lg_k = int(p3.spec_extra(spec).get("lg_k", default))
                    columns.append((name, spec.type, lg_k))
        stage_states = self._stage_state_names()
        for name, spec in self.state_specs.items():
            if name in stage_states or spec.type not in {"cpc", "hll", "theta", "topk"}:
                continue
            _, metadata = p3.sketch_build_expr(
                name,
                spec,
                existing=existing,
                default_source_column=entity or name,
            )
            columns.append(metadata)
        return columns

    def _stage_state_names(self) -> set[str]:
        names: set[str] = set()
        for stage in self.stages:
            names.add(f"{stage.name}_Count")
            names.update(name for name, _ in self._stage_cardinality_states(stage))
        return names

    def _stage_cardinality_states(
        self,
        stage: FunnelStage,
    ) -> list[tuple[str, model.StateSpec]]:
        names = {f"{stage.name}_Customers_cpc", f"{stage.name}_Customers_hll"}
        return [
            (name, spec)
            for name, spec in self.state_specs.items()
            if name in names and spec.type in {"cpc", "hll"}
        ]


__all__ = ["FunnelProcessor", "FunnelStage"]
