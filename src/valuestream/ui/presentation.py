"""Resolve catalog presentation metadata without changing metric semantics."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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

    if isinstance(tile, model.Tile):
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
            labels.setdefault(value, _field_label(value, metric_name, display))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    labels.setdefault(item, _field_label(item, metric_name, display))
    if metric_name:
        labels.setdefault(metric_name, _field_label(metric_name, metric_name, display))
    out["labels"] = labels

    x = out.get("x", out.get("date"))
    y = out.get("y", out.get("value", out.get("values")))
    color = out.get("color")
    if isinstance(x, str):
        out.setdefault("x_axis_title", labels.get(x, humanize_identifier(x)))
    if isinstance(y, str):
        y_label = labels.get(y, humanize_identifier(y))
        if display is not None and display.unit:
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
        (candidate for candidate in catalog.processors.processors if candidate.id == metric.source),
        None,
    )
    if isinstance(metric, model.ApproxDistinctCountMetric):
        state = (
            model.effective_processor_states(processor).get(metric.state)
            if processor is not None
            else None
        )
        lg_k = int((state.model_extra or {}).get("lg_k", 12)) if state is not None else 12
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
        model.TdigestQuantileMetric
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


def _field_label(
    field: str,
    metric_name: str,
    display: model.MetricDisplaySpec | None,
) -> str:
    if field == metric_name and display is not None and display.label:
        return display.label
    return humanize_identifier(field)


def _label_with_unit(label: str, unit: str) -> str:
    normalized = unit.strip().casefold()
    if not normalized or normalized in {"count", "number", "score", "percent"}:
        return label
    return f"{label} ({unit})"


_PRESENTATION_FIELDS = (
    "x",
    "y",
    "y2",
    "value",
    "values",
    "color",
    "date",
    "names",
    "locations",
    "source",
    "target",
    "columns",
    "path",
)


__all__ = [
    "MetricQuality",
    "humanize_identifier",
    "metric_quality",
    "resolve_tile_presentation",
]
