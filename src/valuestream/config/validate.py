"""Catalog-level validation.

Stitches the structural loader (``valuestream.config.loader``) and the
expression validator (``valuestream.expr.validator``) into a single pass:

1. Every Source referenced by a Processor exists.
2. Every Processor referenced by a Metric exists.
3. Every Metric referenced by a Tile exists.
4. Every metric ``formula`` expression type-checks against the processor's
   state columns (plus dependency metrics).
5. Per-source expression contexts (transform filters, processor filters)
   structurally parse — Pydantic already ensures that — and we sanity-check
   their column references against a best-effort "discovered" schema
   built from the source's transforms.

Per-state dtype inference (count → Int64, value_sum → Float64, …) is
documented inline. Sketch-typed states (cpc, hll, theta, tdigest, kll, topk)
have no public dtype — they're modeled as ``String`` placeholders so
column-existence checks succeed but type-checking on them is permissive.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from valuestream.config import model
from valuestream.expr import ast as expr_ast
from valuestream.expr.parser import ParseError
from valuestream.expr.parser import parse as parse_expr
from valuestream.expr.validator import (
    Issue,
    ValidationError,
)
from valuestream.expr.validator import (
    validate as validate_expr,
)
from valuestream.utils.names import capitalize_fields
from valuestream.utils.timer import timed

# State-type → public Dtype. Sketches surface as ``String`` (Phase-0
# placeholder) since the validator's Dtype enum has no Bytes/Sketch.
_STATE_TYPE_DTYPE: Mapping[str, expr_ast.Dtype] = {
    "count": "Int64",
    "value_sum": "Float64",
    "min": "Float64",
    "max": "Float64",
    "pooled_mean": "Float64",
    "pooled_variance": "Float64",
    "tdigest": "String",
    "kll": "String",
    "cpc": "String",
    "hll": "String",
    "theta": "String",
    "topk": "String",
}

# Each tuple is one required role; aliases reflect what the chart factory
# actually accepts (for example treemap path may be authored as ``x``).
_TILE_REQUIRED_ALTERNATIVES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "line": (("x",), ("y",)),
    "stacked_area": (("x",), ("y",), ("color",)),
    "bar": (("x",), ("y",)),
    "kpi_card": (("value", "y"),),
    "waterfall": (("x",), ("y",)),
    "pareto": (("x",), ("y",)),
    "treemap": (("path", "x", "names"),),
    "heatmap": (("x",), ("y",), ("color",)),
    "cohort_heatmap": (("x",), ("y",), ("color",)),
    "scatter": (("x",), ("y",)),
    "combo": (("x",), ("y",), ("y2",)),
    "interval": (("x",), ("y",)),
    "donut": (("names", "x"), ("values", "value", "y")),
    "calendar_heatmap": (("date", "x"), ("value", "y")),
    "bar_polar": (("r",), ("theta",), ("color",)),
    "sankey": (("source",), ("target",), ("value",)),
    "gauge": (("value", "y"),),
    "funnel": (("stages",), ("color",)),
    "boxplot": (("x",), ("y", "property")),
    "histogram": (("property", "x", "y"),),
    "corr": (("x",), ("y",)),
    "descriptive_line": (("x",), ("property",), ("score",)),
    "descriptive_boxplot": (("x",), ("property", "y")),
    "descriptive_histogram": (("property", "x", "y"),),
    "descriptive_heatmap": (("x",), ("y",), ("property",), ("score",)),
    "descriptive_funnel": (("x",), ("color",), ("stages",)),
    "experiment_z_score": (("x",), ("y",)),
    "experiment_odds_ratio": (("x",), ("y",)),
    "clv_treemap": (("path", "x", "names"),),
}


@dataclass(frozen=True)
class CatalogIssue:
    """One catalog-level validation finding."""

    location: str
    message: str
    severity: str = "error"


@dataclass
class CatalogValidationResult:
    """Outcome of :func:`validate_catalog`.

    ``ok`` is true iff there are no error-severity issues. Warnings (e.g.
    ``now()`` usage in an expression) do not flip ``ok``.
    """

    ok: bool
    issues: list[CatalogIssue] = field(default_factory=list)


@timed
def validate_catalog(catalog: model.Catalog) -> CatalogValidationResult:
    """Run all cross-reference and per-expression checks on ``catalog``."""
    issues: list[CatalogIssue] = []

    _validate_unique_ids(catalog, issues)

    source_by_id = {s.id: s for s in catalog.pipelines.sources}
    source_ids = {s.id for s in catalog.pipelines.sources}
    processor_by_id = {p.id: p for p in catalog.processors.processors}
    processor_ids = {p.id for p in catalog.processors.processors}
    metric_names = set(catalog.metrics.metrics.keys())

    seed_columns_by_source: dict[str, set[str]] = {}
    for processor in catalog.processors.processors:
        seed_columns_by_source.setdefault(processor.source, set()).update(
            _processor_source_columns(processor)
        )

    # Build per-source row schemas and validate source-level expressions.
    source_schemas: dict[str, dict[str, expr_ast.Dtype]] = {}
    for source in catalog.pipelines.sources:
        source_schemas[source.id] = _validate_source_expressions(
            source,
            issues,
            seed_columns=seed_columns_by_source.get(source.id, set()),
        )

    # 1. processor.source must resolve to a Source.
    for processor in catalog.processors.processors:
        if processor.source not in source_ids:
            issues.append(
                CatalogIssue(
                    location=f"processors[{processor.id}].source",
                    message=f"unknown source {processor.source!r}",
                )
            )
        _validate_processor_config(
            processor,
            source_schemas.get(processor.source, {}),
            issues,
        )

    # 2. metric.source must resolve to a Processor.
    for name, metric in catalog.metrics.metrics.items():
        if metric.source not in processor_ids:
            issues.append(
                CatalogIssue(
                    location=f"metrics.{name}.source",
                    message=f"unknown processor {metric.source!r}",
                )
            )
        for dep in metric.depends_on:
            if dep not in metric_names:
                issues.append(
                    CatalogIssue(
                        location=f"metrics.{name}.depends_on",
                        message=f"unknown metric {dep!r}",
                    )
                )
        bound = processor_by_id.get(metric.source)
        if bound is not None:
            _validate_metric_state_references(name, metric, bound, issues)
            _validate_metric_dimensions(name, metric, bound, issues)

    _validate_metric_dependencies(catalog.metrics.metrics, issues)

    # 3. tile.metric must resolve to a Metric.
    for dashboard in catalog.dashboards.dashboards:
        for page in dashboard.pages:
            for tile in page.tiles:
                if tile.metric not in metric_names:
                    issues.append(
                        CatalogIssue(
                            location=(
                                f"dashboards[{dashboard.id}]"
                                f".pages[{page.id}].tiles[{tile.id}].metric"
                            ),
                            message=f"unknown metric {tile.metric!r}",
                        )
                    )
                _validate_tile_config(dashboard.id, page.id, tile, issues)
            _validate_page_filters(
                dashboard.id,
                page,
                catalog.metrics.metrics,
                processor_by_id,
                issues,
            )

    _validate_dashboard_theme(catalog.dashboards.theme, issues)

    # 4. Type-check every metric formula against its processor's state schema.
    for processor in catalog.processors.processors:
        if processor.filter is None:
            continue
        bound_source = source_by_id.get(processor.source)
        if bound_source is None:
            continue
        schema = dict(source_schemas.get(bound_source.id, {}))
        for column in processor.group_by:
            schema.setdefault(column, "String")
        _validate_expr_collect(
            processor.filter,
            schema,
            issues,
            prefix=f"processors[{processor.id}].filter",
        )

    for name, metric in catalog.metrics.metrics.items():
        if not isinstance(metric, model.FormulaMetric):
            continue
        bound = processor_by_id.get(metric.source)
        if bound is None:
            continue  # missing-source error already reported above
        schema = _state_schema(bound)
        # Allow dependency metrics to appear as columns in the formula —
        # the query layer evaluates them in dependency order.
        for dep in metric.depends_on:
            schema.setdefault(dep, "Float64")
        _validate_expr_collect(
            metric.expression,
            schema,
            issues,
            prefix=f"metrics.{name}.expression",
        )

    errors = any(i.severity == "error" for i in issues)
    return CatalogValidationResult(ok=not errors, issues=issues)


def _validate_unique_ids(
    catalog: model.Catalog,
    issues: list[CatalogIssue],
) -> None:
    _append_duplicate_ids(
        [source.id for source in catalog.pipelines.sources],
        "pipelines.sources",
        issues,
    )
    _append_duplicate_ids(
        [processor.id for processor in catalog.processors.processors],
        "processors.processors",
        issues,
    )
    _append_duplicate_ids(
        [dashboard.id for dashboard in catalog.dashboards.dashboards],
        "dashboards.dashboards",
        issues,
    )
    for dashboard in catalog.dashboards.dashboards:
        _append_duplicate_ids(
            [page.id for page in dashboard.pages],
            f"dashboards[{dashboard.id}].pages",
            issues,
        )
        for page in dashboard.pages:
            _append_duplicate_ids(
                [tile.id for tile in page.tiles],
                f"dashboards[{dashboard.id}].pages[{page.id}].tiles",
                issues,
            )


def _append_duplicate_ids(
    values: list[str],
    location: str,
    issues: list[CatalogIssue],
) -> None:
    seen: set[str] = set()
    reported: set[str] = set()
    for value in values:
        if value in seen and value not in reported:
            issues.append(
                CatalogIssue(
                    location=location,
                    message=f"duplicate id {value!r}",
                )
            )
            reported.add(value)
        seen.add(value)


def _validate_tile_config(
    dashboard_id: str,
    page_id: str,
    tile: model.Tile,
    issues: list[CatalogIssue],
) -> None:
    values = tile.model_dump(by_alias=True, exclude_none=True)
    for alternatives in _TILE_REQUIRED_ALTERNATIVES.get(tile.chart, ()):
        if any(_configured_tile_value(values.get(field)) for field in alternatives):
            continue
        choices = " or ".join(repr(field) for field in alternatives)
        issues.append(
            CatalogIssue(
                location=(
                    f"dashboards[{dashboard_id}].pages[{page_id}]"
                    f".tiles[{tile.id}].{alternatives[0]}"
                ),
                message=f"chart {tile.chart!r} requires {choices}",
            )
        )
    location = f"dashboards[{dashboard_id}].pages[{page_id}].tiles[{tile.id}]"
    if tile.placement == "kpi_strip" and tile.chart != "kpi_card":
        issues.append(
            CatalogIssue(
                location=f"{location}.placement",
                message="placement 'kpi_strip' requires chart 'kpi_card'",
            )
        )
    if tile.kpi is not None and tile.chart != "kpi_card":
        issues.append(
            CatalogIssue(
                location=f"{location}.kpi",
                message="kpi settings require chart 'kpi_card'",
            )
        )
    if tile.chart == "kpi_card" and _configured_tile_value(values.get("group_by")):
        issues.append(
            CatalogIssue(
                location=f"{location}.group_by",
                message="kpi_card must be scalar and cannot define group_by",
            )
        )
    if values.get("summary_aggregation") not in (None, ""):
        issues.append(
            CatalogIssue(
                location=f"{location}.summary_aggregation",
                message=(
                    "summary_aggregation is deprecated; configure an ungrouped KPI metric "
                    "instead of reducing rendered chart rows"
                ),
                severity="warning",
            )
        )
    if tile.chart == "combo" and values.get("y") == values.get("y2"):
        issues.append(
            CatalogIssue(
                location=f"{location}.y2",
                message="combo y and y2 must reference different values",
            )
        )
    if tile.scale_mode != "absolute" and tile.chart not in {"line", "stacked_area"}:
        issues.append(
            CatalogIssue(
                location=f"{location}.scale_mode",
                message="index_100 and percent_change scale modes require a time-series chart",
            )
        )


def _validate_page_filters(
    dashboard_id: str,
    page: model.DashboardPage,
    metrics: Mapping[str, model.Metric],
    processors: Mapping[str, model.Processor],
    issues: list[CatalogIssue],
) -> None:
    """Ensure authored page filters are backed by persisted aggregate dimensions."""

    seen: set[str] = set()
    for index, filter_spec in enumerate(page.filters):
        location = f"dashboards[{dashboard_id}].pages[{page.id}].filters[{index}]"
        normalized = _dimension_key(filter_spec.field)
        if not normalized:
            issues.append(CatalogIssue(location=f"{location}.field", message="field is required"))
            continue
        if normalized in seen:
            issues.append(
                CatalogIssue(
                    location=f"{location}.field",
                    message=f"duplicate page filter field {filter_spec.field!r}",
                )
            )
            continue
        seen.add(normalized)

        supported: list[str] = []
        for tile in page.tiles:
            metric = metrics.get(tile.metric)
            processor = processors.get(metric.source) if metric is not None else None
            if processor is not None and _dimension_matches(filter_spec.field, processor.group_by):
                supported.append(tile.id)

        if not supported:
            issues.append(
                CatalogIssue(
                    location=f"{location}.field",
                    message=(
                        f"filter field {filter_spec.field!r} is not persisted by any tile processor"
                    ),
                )
            )
        elif filter_spec.scope == "all_tiles" and len(supported) != len(page.tiles):
            unsupported = [tile.id for tile in page.tiles if tile.id not in supported]
            issues.append(
                CatalogIssue(
                    location=f"{location}.scope",
                    message=(
                        f"all_tiles filter {filter_spec.field!r} is unsupported by tiles: "
                        f"{', '.join(unsupported)}"
                    ),
                )
            )


def _validate_dashboard_theme(theme: Mapping[str, Any], issues: list[CatalogIssue]) -> None:
    raw = theme.get("category_colors")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        issues.append(
            CatalogIssue(
                location="dashboards.theme.category_colors",
                message="category_colors must map dimension names to category/color mappings",
            )
        )
        return
    for dimension, category_map in raw.items():
        if not isinstance(category_map, Mapping) or not category_map:
            issues.append(
                CatalogIssue(
                    location=f"dashboards.theme.category_colors.{dimension}",
                    message="category color entry must be a non-empty mapping",
                )
            )
            continue
        for category, color in category_map.items():
            if not isinstance(color, str) or not color.strip():
                issues.append(
                    CatalogIssue(
                        location=(f"dashboards.theme.category_colors.{dimension}.{category}"),
                        message="category color must be a non-empty color string",
                    )
                )


def _dimension_matches(field: str, columns: list[str]) -> bool:
    key = _dimension_key(field)
    return any(_dimension_key(column) == key for column in columns)


def _dimension_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _configured_tile_value(value: object) -> bool:
    return value not in (None, "", "---", [], ())


def _validate_metric_dependencies(
    metrics: Mapping[str, model.Metric],
    issues: list[CatalogIssue],
) -> None:
    """Reject dependency cycles and dependencies bound to another processor."""

    for name, metric in metrics.items():
        for dependency in metric.depends_on:
            bound = metrics.get(dependency)
            if bound is not None and bound.source != metric.source:
                issues.append(
                    CatalogIssue(
                        location=f"metrics.{name}.depends_on",
                        message=(
                            f"metric dependency {dependency!r} uses processor {bound.source!r}; "
                            f"expected {metric.source!r}"
                        ),
                    )
                )

    visiting: list[str] = []
    visited: set[str] = set()
    reported_cycles: set[tuple[str, ...]] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            start = visiting.index(name)
            cycle = (*visiting[start:], name)
            if cycle not in reported_cycles:
                issues.append(
                    CatalogIssue(
                        location=f"metrics.{name}.depends_on",
                        message=f"metric dependency cycle: {' -> '.join(cycle)}",
                    )
                )
                reported_cycles.add(cycle)
            return
        visiting.append(name)
        metric = metrics.get(name)
        if metric is not None:
            for dependency in metric.depends_on:
                if dependency in metrics:
                    visit(dependency)
        visiting.pop()
        visited.add(name)

    for name in metrics:
        visit(name)


def _validate_metric_dimensions(
    metric_name: str,
    metric: model.Metric,
    processor: model.Processor,
    issues: list[CatalogIssue],
) -> None:
    if not isinstance(
        metric,
        model.VariantCompareMetric | model.ContingencyTestMetric | model.ProportionTestMetric,
    ):
        return
    dimensions = set(processor.group_by)
    extra_variant = (processor.model_extra or {}).get("variant_column")
    if isinstance(extra_variant, str):
        dimensions.add(extra_variant)
    if metric.variant_column not in dimensions:
        issues.append(
            CatalogIssue(
                location=f"metrics.{metric_name}.variant_column",
                message=(
                    f"variant column {metric.variant_column!r} is not persisted by processor "
                    f"{processor.id!r}; add it to processor group_by"
                ),
            )
        )


def _validate_source_expressions(
    source: model.Source,
    issues: list[CatalogIssue],
    *,
    seed_columns: set[str],
) -> dict[str, expr_ast.Dtype]:
    """Validate transform expressions while evolving a best-effort row schema.

    Phase 0 has no physical file reader yet, so the source schema is inferred
    from catalog declarations rather than observed data. That is intentionally
    conservative: natural keys, timestamp fields, defaults, and direct transform
    column references seed the schema; expression transforms are then checked in
    order and may add derived columns.
    """
    schema = _initial_source_schema(source)
    for column in seed_columns:
        schema.setdefault(column, "String")
    _seed_transform_columns(source, schema)

    for index, transform in enumerate(source.transforms):
        prefix = f"sources[{source.id}].transforms[{index}]"
        if isinstance(transform, model.RenameCapitalize):
            schema = _capitalize_schema(schema)
        elif isinstance(transform, model.ParseDatetime):
            for column in transform.columns:
                schema[column] = "Datetime"
        elif isinstance(transform, model.DeriveCalendar):
            schema.setdefault(transform.from_, "Datetime")
            for output in transform.outputs:
                schema[output] = _calendar_output_dtype(output)
        elif isinstance(transform, model.DeriveActionId):
            for part in transform.parts:
                schema.setdefault(part, "String")
            schema.setdefault("ActionID", "String")
        elif isinstance(transform, model.DeriveColumn):
            dtype = _validate_expr_collect(
                transform.expression,
                schema,
                issues,
                prefix=f"{prefix}.expression",
            )
            if dtype is not None:
                schema[transform.output] = dtype
        elif isinstance(transform, model.FilterTransform):
            _validate_expr_collect(
                transform.expression,
                schema,
                issues,
                prefix=f"{prefix}.expression",
            )
        elif isinstance(transform, model.Defaults):
            for column, value in transform.values.items():
                schema[column] = _infer_literal_dtype(value)
        elif isinstance(transform, model.Cast):
            schema.update(transform.columns)
        elif isinstance(transform, model.DropColumns):
            for column in transform.columns:
                schema.pop(column, None)
        elif isinstance(transform, model.Coalesce):
            for column in transform.columns:
                schema.setdefault(column, "String")
            schema[transform.output] = "String"
        elif isinstance(transform, model.Dedup):
            for key in transform.keys:
                schema.setdefault(key, "String")

    return schema


def _processor_source_columns(processor: model.Processor) -> set[str]:
    """Return source fields declared outside expressions on one processor."""

    columns = {str(column) for column in processor.group_by if str(column).strip()}
    if processor.time is not None and processor.time.column:
        columns.add(processor.time.column)
    raw = processor.model_dump(mode="python", by_alias=True, exclude_none=True)
    for key in ("outcome_column", "variant_column"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            columns.add(value)
    for key in ("properties", "score_properties"):
        values = raw.get(key)
        if isinstance(values, list):
            columns.update(str(value) for value in values if str(value).strip())
    for key in ("score_columns", "scores"):
        values = raw.get(key)
        if isinstance(values, dict):
            columns.update(str(value) for value in values.values() if str(value).strip())
    for key in ("outcome", "entities"):
        values = raw.get(key)
        if not isinstance(values, dict):
            continue
        for field_key in ("column", "subject"):
            value = values.get(field_key)
            if isinstance(value, str) and value.strip():
                columns.add(value)
    states = raw.get("states")
    if isinstance(states, dict):
        for state in states.values():
            if not isinstance(state, dict):
                continue
            source_column = state.get("source_column")
            if isinstance(source_column, str) and source_column.strip():
                columns.add(source_column)
    return columns


def _capitalize_schema(schema: Mapping[str, expr_ast.Dtype]) -> dict[str, expr_ast.Dtype]:
    names = list(schema)
    renamed = capitalize_fields(names)
    return {new: schema[old] for old, new in zip(names, renamed, strict=False)}


def _initial_source_schema(source: model.Source) -> dict[str, expr_ast.Dtype]:
    schema: dict[str, expr_ast.Dtype] = {}
    if source.schema_.timestamp_column is not None:
        schema[source.schema_.timestamp_column] = "Datetime"
    for key in source.schema_.natural_key:
        schema.setdefault(key, "String")
    for column in source.schema_.drop_columns:
        schema.setdefault(column, "String")
    for column, value in source.defaults.items():
        schema[column] = _infer_literal_dtype(value)
    return schema


def _seed_transform_columns(
    source: model.Source,
    schema: dict[str, expr_ast.Dtype],
) -> None:
    """Seed schema with direct column references from non-expression transforms."""
    for transform in source.transforms:
        if isinstance(transform, model.ParseDatetime):
            for column in transform.columns:
                schema.setdefault(column, "Datetime")
        elif isinstance(transform, model.DeriveCalendar):
            schema.setdefault(transform.from_, "Datetime")
        elif isinstance(transform, model.DeriveActionId):
            for part in transform.parts:
                schema.setdefault(part, "String")
        elif isinstance(transform, model.Dedup):
            for key in transform.keys:
                schema.setdefault(key, "String")
        elif isinstance(transform, model.Cast):
            for column, dtype in transform.columns.items():
                schema.setdefault(column, dtype)
        elif isinstance(transform, model.Coalesce):
            for column in transform.columns:
                schema.setdefault(column, "String")


def _validate_expr_collect(
    expression: expr_ast.Expr,
    schema: Mapping[str, expr_ast.Dtype],
    issues: list[CatalogIssue],
    *,
    prefix: str,
) -> expr_ast.Dtype | None:
    """Validate one expression and append errors/warnings with ``prefix``."""
    try:
        result = validate_expr(expression, schema)
    except ValidationError as exc:
        for ev in exc.issues:
            issues.append(_to_catalog_issue(ev, prefix=prefix))
        return None

    for warning in result.warnings:
        issues.append(_to_catalog_issue(warning, prefix=prefix))
    return result.dtype


def _infer_literal_dtype(value: Any) -> expr_ast.Dtype:
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, int):
        return "Int64"
    if isinstance(value, float):
        return "Float64"
    return "String"


def _calendar_output_dtype(output: str) -> expr_ast.Dtype:
    normalized = output.lower()
    if normalized == "day":
        return "Date"
    if normalized == "year":
        return "Int16"
    return "String"


def _state_schema(processor: model.Processor) -> dict[str, expr_ast.Dtype]:
    """Map a processor's state columns to public dtypes for the validator."""
    return {
        name: _STATE_TYPE_DTYPE[spec.type]
        for name, spec in model.effective_processor_states(processor).items()
    }


def _validate_metric_state_references(
    metric_name: str,
    metric: model.Metric,
    processor: model.Processor,
    issues: list[CatalogIssue],
) -> None:
    if isinstance(metric, model.FunnelDropoffMetric):
        stages = _funnel_stage_names(processor)
        for field_name in ("from_stage", "to_stage"):
            stage_name = getattr(metric, field_name)
            if stage_name not in stages:
                issues.append(
                    CatalogIssue(
                        location=f"metrics.{metric_name}.{field_name}",
                        message=(
                            f"unknown funnel stage {stage_name!r} on processor {processor.id!r}"
                        ),
                    )
                )
        return
    if isinstance(metric, model.ApproxDistinctCountMetric):
        _require_state_type(
            metric_name,
            processor,
            "state",
            metric.state,
            {"cpc", "hll", "theta"},
            issues,
        )
        return
    if isinstance(metric, model.TopKItemsMetric):
        _require_state_type(
            metric_name,
            processor,
            "state",
            metric.state,
            {"topk"},
            issues,
        )
        return
    if isinstance(metric, model.TdigestQuantileMetric):
        _require_state_type(
            metric_name,
            processor,
            "state",
            metric.state,
            {"tdigest", "kll"},
            issues,
        )
        return
    if isinstance(metric, model.SetOpMetric):
        _validate_set_op_metric(metric_name, metric, processor, issues)
        return
    if not isinstance(metric, model.CurveFromDigestsMetric | model.CalibrationFromDigestsMetric):
        return
    for field_name in ("positive_state", "negative_state"):
        _require_state_type(
            metric_name,
            processor,
            field_name,
            getattr(metric, field_name),
            {"tdigest"},
            issues,
        )


def _validate_set_op_metric(
    metric_name: str,
    metric: model.SetOpMetric,
    processor: model.Processor,
    issues: list[CatalogIssue],
) -> None:
    if metric.states and metric.operands:
        issues.append(
            CatalogIssue(
                location=f"metrics.{metric_name}",
                message="set_op must define either states or operands, not both",
            )
        )
    states = metric.states or [operand.state for operand in metric.operands]
    if not states:
        issues.append(
            CatalogIssue(
                location=f"metrics.{metric_name}",
                message="set_op must define at least one state or operand",
            )
        )
    if metric.op in {"a_not_b", "diff"} and len(states) != 2:
        issues.append(
            CatalogIssue(
                location=f"metrics.{metric_name}",
                message="set_op diff/a_not_b requires exactly two operands",
            )
        )
    for index, state in enumerate(states):
        _require_state_type(
            metric_name,
            processor,
            f"operands[{index}].state" if metric.operands else f"states[{index}]",
            state,
            {"theta"},
            issues,
        )
    for index, operand in enumerate(metric.operands):
        if operand.time_window is not None:
            _validate_set_time_window(metric_name, index, operand.time_window, issues)


def _validate_set_time_window(
    metric_name: str,
    operand_index: int,
    window: dict[str, Any],
    issues: list[CatalogIssue],
) -> None:
    location = f"metrics.{metric_name}.operands[{operand_index}].time_window"
    if set(window) == {"last"} and _is_duration(window["last"], positive=True):
        return
    between = window.get("between")
    if (
        set(window) == {"between"}
        and isinstance(between, list | tuple)
        and len(between) == 2
        and all(_is_duration(value, positive=False) for value in between)
    ):
        return
    issues.append(
        CatalogIssue(
            location=location,
            message=(
                "time_window must be {'last': '<positive Nd/Nw>'} or "
                "{'between': ['<offset>', '<offset>']}"
            ),
        )
    )


def _is_duration(value: object, *, positive: bool) -> bool:
    match = re.fullmatch(r"([+-]?)(\d+)([dDwW])", str(value).strip())
    if match is None:
        return False
    sign, amount, _ = match.groups()
    return not positive or (sign != "-" and int(amount) > 0)


def _require_state_type(
    metric_name: str,
    processor: model.Processor,
    field_name: str,
    state_name: str,
    allowed_types: set[str],
    issues: list[CatalogIssue],
) -> None:
    states = model.effective_processor_states(processor)
    state = states.get(state_name)
    location = f"metrics.{metric_name}.{field_name}"
    if state is None:
        issues.append(
            CatalogIssue(
                location=location,
                message=f"unknown state {state_name!r} on processor {processor.id!r}",
            )
        )
        return
    if state.type not in allowed_types:
        expected = ", ".join(repr(kind) for kind in sorted(allowed_types))
        issues.append(
            CatalogIssue(
                location=location,
                message=f"state {state_name!r} must have type {expected}, got {state.type!r}",
            )
        )


def _validate_processor_config(
    processor: model.Processor,
    source_schema: Mapping[str, expr_ast.Dtype],
    issues: list[CatalogIssue],
) -> None:
    variant_column = (processor.model_extra or {}).get("variant_column")
    if isinstance(variant_column, str) and any(
        _dimension_key(variant_column) == _dimension_key(column) for column in processor.group_by
    ):
        issues.append(
            CatalogIssue(
                location=f"processors[{processor.id}].variant_column",
                message=(
                    f"variant column {variant_column!r} is already present in group_by; "
                    "keep it only as variant_column because the processor persists it automatically"
                ),
            )
        )

    if not isinstance(processor, model.FunnelProcessor):
        return
    raw_stages = (processor.model_extra or {}).get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        issues.append(
            CatalogIssue(
                location=f"processors[{processor.id}].stages",
                message="funnel processor must define at least one stage",
            )
        )
        return
    for index, stage in enumerate(raw_stages):
        location = f"processors[{processor.id}].stages[{index}]"
        if not isinstance(stage, Mapping):
            issues.append(
                CatalogIssue(
                    location=location,
                    message="funnel stage must be a mapping",
                )
            )
            continue
        if not stage.get("name"):
            issues.append(
                CatalogIssue(
                    location=f"{location}.name",
                    message="field required",
                )
            )
        if "when" not in stage:
            issues.append(
                CatalogIssue(
                    location=f"{location}.when",
                    message="field required",
                )
            )
            continue
        try:
            when = parse_expr(stage["when"])
        except ParseError as exc:
            issues.append(
                CatalogIssue(
                    location=f"{location}.when",
                    message=str(exc),
                )
            )
            continue
        dtype = _validate_expr_collect(
            when,
            source_schema,
            issues,
            prefix=f"{location}.when",
        )
        if dtype is not None and dtype != "Boolean":
            issues.append(
                CatalogIssue(
                    location=f"{location}.when",
                    message=f"funnel stage condition must be Boolean, got {dtype}",
                )
            )


def _funnel_stage_names(processor: model.Processor) -> set[str]:
    if not isinstance(processor, model.FunnelProcessor):
        return set()
    raw_stages = (processor.model_extra or {}).get("stages")
    if not isinstance(raw_stages, list):
        return set()
    names: set[str] = set()
    for stage in raw_stages:
        if isinstance(stage, Mapping) and stage.get("name"):
            names.add(str(stage["name"]))
    return names


def _to_catalog_issue(ev: Issue, prefix: str) -> CatalogIssue:
    location = f"{prefix}.{ev.path}" if ev.path and ev.path != "<root>" else prefix
    return CatalogIssue(
        location=location,
        message=ev.message,
        severity=ev.severity,
    )


__all__ = ["CatalogIssue", "CatalogValidationResult", "validate_catalog"]
