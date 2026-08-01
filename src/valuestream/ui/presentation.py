"""Resolve catalog presentation metadata without changing metric semantics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from valuestream.config import model


@dataclass(frozen=True)
class MetricQuality:
    """Short user-facing method metadata for one metric."""

    label: str = ""
    help: str = ""
    approximate: bool = False


def resolve_tile_presentation(  # noqa: PLR0912
    catalog: model.Catalog,
    tile: model.Tile | Mapping[str, Any],
) -> dict[str, Any]:
    """Merge metric display defaults into a tile while preserving tile overrides."""

    if isinstance(tile, BaseModel):
        out = tile.model_dump(mode="python", exclude_none=True)
    else:
        out = dict(tile)
    metric_name = str(out.get("metric", ""))
    metric = catalog.metrics.metrics.get(metric_name)
    display = metric.display if metric is not None else None

    if display is not None:
        if display.value_format and not out.get("value_format"):
            out["value_format"] = display.value_format
        if display.unit:
            out.setdefault("unit", display.unit)
        out.setdefault("direction", display.direction)
    if metric is not None and metric.description and not out.get("description"):
        out["description"] = metric.description

    labels = dict(out.get("labels") or {})
    for field in _PRESENTATION_FIELDS:
        value = out.get(field)
        if isinstance(value, str) and value:
            labels.setdefault(
                value,
                chart_field_label(catalog, metric_name, field, value),
            )
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    labels.setdefault(
                        item,
                        chart_field_label(catalog, metric_name, field, item),
                    )
    if metric_name:
        labels.setdefault(
            metric_name,
            chart_field_label(catalog, metric_name, "metric_output", metric_name),
        )
    out["labels"] = labels

    x = out.get("x")
    y = out.get("y") or out.get("metric_output") or metric_name
    color = out.get("color")
    if isinstance(x, str):
        out.setdefault("x_axis_title", labels.get(x, humanize_identifier(x)))
    if isinstance(y, str):
        y_label = labels.get(y, humanize_identifier(y))
        if (
            display is not None
            and display.unit
            and str(out.get("chart", ""))
            not in {"experiment_z_score", "experiment_odds_ratio"}
        ):
            y_label = _label_with_unit(y_label, display.unit)
        out.setdefault("y_axis_title", y_label)
    if isinstance(color, str):
        out.setdefault("legend_title", labels.get(color, humanize_identifier(color)))

    quality = metric_quality(catalog, metric_name)
    if quality.label:
        out["quality_label"] = quality.label
        out["quality_help"] = quality.help
    return out


def metric_quality(catalog: model.Catalog, metric_name: str) -> MetricQuality:
    """Derive trustworthy method metadata from the metric and processor state."""

    metric = catalog.metrics.metrics.get(metric_name)
    if metric is None:
        return MetricQuality()
    processor = next(
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.processor),
        None,
    )
    if isinstance(metric, model.ApproxDistinctCountMetric):
        state = (
            model.effective_processor_states(processor).get(metric.state)
            if processor is not None
            else None
        )
        lg_k = int(getattr(state, "lg_k", None) or 12) if state is not None else 12
        relative_error = 1.04 / math.sqrt(2**lg_k)
        return MetricQuality(
            label="Approximate",
            help=(
                f"HyperLogLog distinct-count estimate (lg_k={lg_k}); "
                f"about ±{relative_error:.1%} relative standard error at one standard deviation."
            ),
            approximate=True,
        )
    if isinstance(
        metric,
        model.DistributionMetric
        | model.QuantileMetric
        | model.CurveFromDigestsMetric
        | model.CalibrationFromDigestsMetric,
    ):
        return MetricQuality(
            label="Approximate",
            help="Estimated from mergeable t-digest aggregate state; raw event rows are not queried.",
            approximate=True,
        )
    if isinstance(
        metric,
        model.VariantCompareMetric | model.ProportionTestMetric | model.ContingencyTestMetric,
    ):
        return MetricQuality(
            label="Statistical estimate",
            help="Calculated from aggregate positive/negative counts with the configured test roles.",
        )
    return MetricQuality()


def humanize_identifier(value: str) -> str:
    """Turn stable catalog identifiers into conservative display labels."""

    text = re.sub(r"[_-]+", " ", str(value)).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    if text.startswith("VS "):
        text = text[3:]
    return text[:1].upper() + text[1:] if text else ""


def chart_field_label(
    catalog: model.Catalog,
    metric_name: str,
    field_name: str,
    option: str,
    *,
    humanize: bool = True,
) -> str:
    """Resolve one chart field's display label without renaming its stored value."""

    if not option:
        return "None"

    primary_metric = catalog.metrics.metrics.get(metric_name)
    if field_name == "secondary_metric":
        secondary_metric = catalog.metrics.metrics.get(option)
        if (
            secondary_metric is not None
            and secondary_metric.display is not None
            and secondary_metric.display.label
        ):
            return secondary_metric.display.label
    if (
        option == metric_name
        and primary_metric is not None
        and primary_metric.display is not None
        and primary_metric.display.label
    ):
        return primary_metric.display.label

    processor = next(
        (
            candidate
            for candidate in catalog.processors.processors
            if primary_metric is not None and candidate.id == primary_metric.processor
        ),
        None,
    )
    if (
        isinstance(processor, model.FrequencyResponseProcessor)
        and option == processor.frequency_column
    ):
        return "Number of impressions"
    return humanize_identifier(option) if humanize else option


def _label_with_unit(label: str, unit: str) -> str:
    normalized = unit.strip().casefold()
    if not normalized or normalized in {"count", "number", "score", "percent"}:
        return label
    return f"{label} ({unit})"


_PRESENTATION_FIELDS = (
    "x",
    "y",
    "secondary_metric",
    "metric_output",
    "lower_output",
    "upper_output",
    "color",
    "names",
    "locations",
    "lat",
    "lon",
    "theta",
    "property",
    "score",
    "columns",
    "path",
)


__all__ = [
    "MetricQuality",
    "chart_field_label",
    "humanize_identifier",
    "metric_quality",
    "resolve_tile_presentation",
]
