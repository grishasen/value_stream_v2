"""Phase 2 ``score_distribution`` processor."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

import valuestream.processors.processors_helper as p3
from valuestream.algorithms import ml_helpers
from valuestream.config import model
from valuestream.config.canonical import processor_config_hash
from valuestream.expr.translator import translate
from valuestream.processors import grain_levels
from valuestream.processors.context import SOURCE_ORDER_COLUMN, ChunkContext
from valuestream.processors.outcomes import compatible_values, is_in_values, parse_outcome
from valuestream.states import kll, tdigest
from valuestream.utils.timer import timed


class ScoreDistributionProcessor:
    """Aggregate score-vs-outcome distributions for model metrics."""

    def __init__(self, config: model.Processor, *, computation_hash: str | None = None) -> None:
        if not isinstance(config, model.ScoreDistributionProcessor):
            raise TypeError(f"expected score_distribution processor, got {config.kind!r}")
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
        """Return daily score aggregates for one transformed source chunk."""
        return self.chunk_aggregate_lazy(frame, ctx).collect()

    @timed
    def chunk_aggregate_lazy(self, frame: pl.LazyFrame, ctx: ChunkContext) -> pl.LazyFrame:
        """Return a lazy daily score aggregate plan for one source chunk."""
        outcome = parse_outcome(self.config.outcome)
        source = frame
        if self.config.filter is not None:
            source = source.filter(translate(self.config.filter))
        schema = source.collect_schema()
        positive_values = compatible_values(outcome.positive_values, schema[outcome.column])
        negative_values = compatible_values(outcome.negative_values, schema[outcome.column])
        source = source.filter(
            is_in_values(outcome.column, [*positive_values, *negative_values])
        ).with_columns(is_in_values(outcome.column, positive_values).alias("__vs_positive"))
        dedup_keys = [key for key in self.config.dedup_keys if key in source.collect_schema().names()]
        if dedup_keys:
            source = source.filter(
                pl.col("__vs_positive").cast(pl.Int8)
                == pl.col("__vs_positive").cast(pl.Int8).max().over(dedup_keys)
            ).unique(subset=dedup_keys, keep="first")

        existing = set(source.collect_schema().names())
        time_columns = grain_levels.chunk_time_group_columns(existing, self.config)
        group_keys = list(
            dict.fromkeys(
                column
                for column in [*self.group_by_columns, *time_columns]
                if column and column in existing
            )
        )
        grouped = source.group_by(group_keys).agg(self._agg_exprs(existing, schema))
        grouped = p3.unnest_distribution_sketches(grouped)
        frame_out = self._postprocess(grouped, existing)
        if "__PositiveCount" in frame_out.collect_schema().names():
            frame_out = frame_out.filter(pl.col("__PositiveCount") > 0).drop("__PositiveCount")
        frame_out = self._ensure_state_columns(frame_out)
        return p3.with_provenance(
            frame_out,
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
            self.config, target_grain, "score_distribution"
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
        """Merge score aggregate rows."""
        return p3.merge_state_frame(frame, self.state_specs, group_columns)

    @timed
    def merge_for_query(self, frame: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
        """Merge rows and preserve config hash for query-time metrics."""
        return p3.merge_for_query(self.merge, frame, group_columns, self.config_hash)

    def _agg_exprs(  # noqa: PLR0912, PLR0915
        self, existing: set[str], source_schema: pl.Schema
    ) -> list[pl.Expr]:
        exprs: list[pl.Expr] = [pl.col("__vs_positive").sum().alias("__PositiveCount")]
        distribution_sketches: list[p3.DistributionSketchSpec] = []
        for name, spec in self.state_specs.items():
            if spec.type == "count":
                if spec.outcome == "positive":
                    exprs.append(pl.col("__vs_positive").sum().alias(name))
                elif spec.outcome == "negative":
                    exprs.append((~pl.col("__vs_positive")).sum().alias(name))
                elif spec.source_column:
                    if spec.source_column not in existing:
                        raise ValueError(
                            f"score_distribution processor {self.id!r} state {name!r} "
                            f"requires missing source column {spec.source_column!r}"
                        )
                    count = (
                        pl.col(spec.source_column).n_unique()
                        if spec.distinct
                        else pl.col(spec.source_column).count()
                    )
                    exprs.append(count.alias(name))
                elif spec.stage:
                    raise ValueError(
                        f"score_distribution processor {self.id!r} state {name!r} "
                        "cannot use a funnel stage selector"
                    )
                else:
                    exprs.append(pl.len().alias(name))
            elif spec.type in {"tdigest", "kll"}:
                score_column = spec.source_column
                if score_column not in existing:
                    raise ValueError(
                        f"score_distribution processor {self.id!r} state {name!r} "
                        f"requires missing score column {score_column!r}"
                    )
                outcome_role = spec.outcome
                values = pl.col(score_column)
                if outcome_role in {"positive", "negative"}:
                    desired = outcome_role == "positive"
                    values = values.filter(pl.col("__vs_positive") == desired)
                distribution_sketches.append(
                    (name, spec.type, _sketch_k(spec), values.drop_nulls())
                )
            elif spec.type in {"cpc", "hll", "theta", "topk"}:
                expression, _ = p3.sketch_build_expr(
                    name,
                    spec,
                    existing=existing,
                    source_dtypes=source_schema,
                )
                if expression is not None:
                    exprs.append(expression)
            elif spec.type == "pooled_mean":
                if spec.recipe == "personalization":
                    required = {"CustomerID", "Name"}
                    function = _personalization_group
                    inputs = ("CustomerID", "Name")
                elif spec.recipe == "novelty":
                    required = {"CustomerID", "InteractionID", "Name"}
                    function = _novelty_group
                    inputs = ("CustomerID", "InteractionID", "Name")
                elif spec.source_column:
                    if spec.source_column not in existing:
                        raise ValueError(
                            f"score_distribution processor {self.id!r} state {name!r} "
                            f"requires missing source column {spec.source_column!r}"
                        )
                    exprs.append(pl.col(spec.source_column).mean().alias(name))
                    continue
                else:  # pragma: no cover - rejected by the state model
                    raise ValueError(f"pooled_mean state {name!r} has no computation")
                if not required <= existing:
                    missing = ", ".join(sorted(required - existing))
                    raise ValueError(
                        f"score_distribution processor {self.id!r} state {name!r} "
                        f"requires missing recipe columns: {missing}"
                    )
                exprs.append(
                    pl.map_groups(
                        exprs=_source_order_inputs(existing, *inputs),
                        function=function,
                        return_dtype=pl.Float64,
                        returns_scalar=True,
                    ).alias(name)
                )
            elif spec.type == "pooled_variance":
                if spec.source_column not in existing:
                    raise ValueError(
                        f"score_distribution processor {self.id!r} state {name!r} "
                        f"requires missing source column {spec.source_column!r}"
                    )
                exprs.append(pl.col(spec.source_column).var().alias(name))
        if distribution_sketches:
            exprs.append(p3.distribution_sketch_expr(distribution_sketches))
        return exprs

    def _postprocess(self, frame: pl.LazyFrame, existing: set[str]) -> pl.LazyFrame:
        return p3.postprocess_sketches(frame, self._sketch_columns(existing))

    def _ensure_state_columns(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        out = frame
        columns = set(out.collect_schema().names())
        for name, spec in self.state_specs.items():
            if name in columns:
                continue
            if spec.type == "tdigest":
                out = out.with_columns(pl.lit(tdigest.build([], k=_sketch_k(spec))).alias(name))
            elif spec.type == "kll":
                out = out.with_columns(pl.lit(kll.build([], k=_sketch_k(spec))).alias(name))
            elif spec.type in {
                "cpc",
                "hll",
                "theta",
                "topk",
                "pooled_mean",
                "pooled_variance",
                "count",
            }:
                out = p3.ensure_state_columns(out, {name: spec})
        return out

    def _sketch_columns(self, existing: set[str]) -> list[tuple[str, str, int]]:
        columns: list[tuple[str, str, int]] = []
        for name, spec in self.state_specs.items():
            if spec.type not in {"cpc", "hll", "theta", "topk"}:
                continue
            _, metadata = p3.sketch_build_expr(
                name,
                spec,
                existing=existing,
            )
            columns.append(metadata)
        return columns


def _sketch_k(spec: model.StateSpec) -> int:
    return int(p3.spec_extra(spec).get("k", 500))


def _personalization_group(values: Sequence[pl.Series]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = _source_ordered_values(values, value_columns=2)
    return ml_helpers.personalization(ordered[0], ordered[1])


def _novelty_group(values: Sequence[pl.Series]) -> float:
    if len(values) < 3:
        return 0.0
    ordered = _source_ordered_values(values, value_columns=3)
    return ml_helpers.novelty(ordered[0], ordered[1], ordered[2])


def _source_ordered_values(
    values: Sequence[pl.Series], *, value_columns: int
) -> tuple[pl.Series, ...]:
    selected = tuple(values[:value_columns])
    if len(values) <= value_columns:
        return selected
    order = values[value_columns].arg_sort()
    return tuple(series.gather(order) for series in selected)


def _source_order_inputs(existing: set[str], *columns: str) -> list[str]:
    inputs = list(columns)
    if SOURCE_ORDER_COLUMN in existing:
        inputs.append(SOURCE_ORDER_COLUMN)
    return inputs


__all__ = ["ScoreDistributionProcessor"]
