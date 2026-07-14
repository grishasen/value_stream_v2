"""Phase 3 ``snapshot`` processor."""

from __future__ import annotations

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.binary_outcome import PROVENANCE_COLUMNS, ChunkContext
from valuestream.utils.timer import timed


class SnapshotProcessor:
    """Aggregate periodic and accumulating snapshot state."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.SnapshotProcessor):
            raise TypeError(f"expected snapshot processor, got {config.kind!r}")
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
    def entity_column(self) -> str | None:
        raw = p3.extra(self.config).get("entity")
        return str(raw) if raw is not None else None

    @property
    def state_specs(self) -> dict[str, model.StateSpec]:
        return model.effective_processor_states(self.config)

    @timed
    def chunk_aggregate(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.DataFrame:
        """Return snapshot state rows for one source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy snapshot state aggregate plan for one source chunk."""
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))
        if self.config.snapshot_kind == "accumulating":
            source = self._latest_entity_rows(source, ctx)
        else:
            source = self._with_periodic_as_of(source, ctx)

        existing = set(source.collect_schema().names())
        group_keys = self._chunk_group_keys(existing)
        grouped = source.group_by(group_keys).agg(self._agg_exprs(existing))
        grouped = p3.postprocess_sketches(grouped, self._sketch_columns(existing))
        grouped = p3.ensure_state_columns(grouped, self.state_specs)
        return p3.with_provenance(
            grouped,
            self.config_hash,
            ctx,
            period=pl.col("as_of_date").cast(pl.String).str.slice(0, 7),
        )

    @timed
    def compact(self, frame: pl.DataFrame, target_grain: str, ctx: ChunkContext) -> pl.DataFrame:
        """Compact snapshot rows while preserving latest-as-of semantics."""
        if frame.is_empty():
            return frame
        target_grain = grain_levels.normalize_target_grain(self.config, target_grain, "snapshot")
        if target_grain == "daily":
            return frame
        provenance_to_drop = [column for column in PROVENANCE_COLUMNS if column != "period"]
        working = frame.drop([column for column in provenance_to_drop if column in frame.columns])
        working = grain_levels.with_calendar_columns(working)
        level = self.config.aggregation_level_for(target_grain)
        working = working.with_columns(
            grain_levels.period_expr_for_level(set(working.columns), level).alias("period")
        )
        if target_grain != "daily":
            if self.config.snapshot_kind == "periodic":
                latest_by = [*self.group_by_columns, "period"]
            else:
                latest_by = [column for column in [self.entity_column, "period"] if column]
            working = _latest_by(working, latest_by)
        group_columns = p3.default_group_columns(working, self.state_specs)
        return p3.with_static_provenance(
            self.merge(working, group_columns=group_columns),
            self.config_hash,
            ctx,
        )

    @timed
    def merge(self, frame: pl.DataFrame, group_columns: list[str] | None = None) -> pl.DataFrame:
        """Merge snapshot state rows."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows after retaining the latest relevant ``as_of_date``."""
        working = frame
        if "as_of_date" in working.columns:
            if self.config.snapshot_kind == "accumulating" and self.entity_column is not None:
                working = _latest_entity_state(working, self.entity_column)
            else:
                working = _latest_by(working, group_columns)
        working = working.drop(
            [column for column in PROVENANCE_COLUMNS if column in working.columns]
        )
        return self.merge(working, group_columns=group_columns).with_columns(
            pl.lit(self.config_hash).alias("config_hash")
        )

    def _agg_exprs(self, existing: set[str]) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        for name, spec in self.state_specs.items():
            raw_extra = p3.spec_extra(spec)
            source_column = str(raw_extra.get("source_column", name))
            if spec.type == "count":
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
                    default_source_column=self.entity_column or "CustomerID",
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
                default_source_column=self.entity_column or "CustomerID",
            )
            columns.append(meta)
        return columns

    def _chunk_group_keys(self, existing: set[str]) -> list[str]:
        columns: list[str | None] = [*self.group_by_columns, "as_of_date"]
        if self.config.snapshot_kind == "accumulating":
            columns.insert(len(self.group_by_columns), self.entity_column)
        return [column for column in columns if column is not None and column in existing]

    def _with_periodic_as_of(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        existing = set(frame.collect_schema().names())
        as_of_column = str(
            p3.extra(self.config).get(
                "as_of_column",
                next((column for column in ("as_of_date", "Day", "day") if column in existing), ""),
            )
        )
        if as_of_column and as_of_column in existing:
            return frame.with_columns(pl.col(as_of_column).cast(pl.Date).alias("as_of_date"))
        return frame.with_columns(pl.lit(ctx.created_at.date()).alias("as_of_date"))

    def _latest_entity_rows(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        entity = self.entity_column
        if entity is None:
            raise ValueError(
                "accumulating snapshot requires an 'entity' column in processor config"
            )
        frame = frame.with_columns(self._accumulating_as_of_expr(ctx).alias("as_of_date"))
        return frame.sort("as_of_date").unique(subset=[entity], keep="last")

    def _accumulating_as_of_expr(self, ctx: ChunkContext) -> pl.Expr:
        raw_milestones = p3.extra(self.config).get("milestones", [])
        columns = [
            str(item["column"])
            for item in raw_milestones
            if isinstance(item, dict) and "column" in item
        ]
        if columns:
            return pl.max_horizontal([pl.col(column) for column in columns]).cast(pl.Date)
        return pl.lit(ctx.created_at.date())


def _latest_by(frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    if "as_of_date" not in frame.columns:
        return frame
    if not group_columns:
        max_as_of = frame.select(pl.col("as_of_date").max()).item()
        return frame.filter(pl.col("as_of_date") == max_as_of)
    return frame.filter(pl.col("as_of_date") == pl.col("as_of_date").max().over(group_columns))


def _latest_entity_state(frame: pl.DataFrame, entity_column: str) -> pl.DataFrame:
    sort_columns = ["as_of_date"]
    if "created_at" in frame.columns:
        sort_columns.append("created_at")
    return frame.sort(sort_columns).unique(subset=[entity_column], keep="last")


__all__ = ["SnapshotProcessor"]
