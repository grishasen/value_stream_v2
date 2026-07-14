"""Dimension profiling helpers for group-by selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from valuestream.config import model
from valuestream.readers.discovery import discover
from valuestream.readers.io import cleanup_temporaries, read
from valuestream.transforms import apply_transforms
from valuestream.ui.context import ValueStreamContext, processors_for_source
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

PROFILE_SAMPLE_ROWS = 10_000
RECOMMENDED_CARDINALITY_MIN = 3
LOW_CARDINALITY_MAX = 50
MEDIUM_CARDINALITY_MAX = 500
HIGH_CARDINALITY_MAX = 5_000
IDENTITY_HINTS = (
    "email",
    "guid",
    "phone",
    "token",
    "uuid",
)
ENTITY_HINTS = ("account", "customer", "subject")
MEASURE_OR_TIME_HINTS = (
    "datetime",
    "probability",
    "propensity",
    "rank",
    "response_time",
    "score",
    "timestamp",
)
ACTIVE_PROCESSOR_REASON = "Already modeled in processor group-by."
ACTIVE_SELECTION_REASON = "Already selected as a group-by field."
PEGA_CDH_CORE_PACK = "Pega/CDH Core"
DIMENSION_PACKS: dict[str, tuple[str, ...]] = {
    PEGA_CDH_CORE_PACK: (
        "Channel",
        "Issue",
        "Group",
        "Direction",
        "PlacementType",
        "Treatment",
        "CustomerType",
        "ModelControlGroup",
        "Campaign",
        "CampaignName",
        "ActionName",
        "Offer",
        "Proposition",
        "BusinessIssue",
        "InteractionType",
    )
}


@dataclass(frozen=True)
class DimensionProfileRow:
    """Profile one source field for group-by suitability."""

    field: str
    dtype: str
    non_null: int
    null_rate: float
    cardinality: int
    cardinality_rate: float
    sample_values: str
    current_usage: str
    recommendation: str
    safe_for_group_by: str
    reason: str


@dataclass(frozen=True)
class AggregateSizePreview:
    """Estimate how much a dimension change expands sampled aggregate rows."""

    current_rows: int
    projected_rows: int
    added_rows: int
    expansion_factor: float
    sample_rows: int


def source_profile_sample(
    ctx: ValueStreamContext,
    source: model.Source,
    *,
    limit: int = PROFILE_SAMPLE_ROWS,
) -> pl.DataFrame | None:
    """Return a transformed sample frame for profiling source dimensions."""
    try:
        chunks = discover(ctx.workspace, source)
        if not chunks:
            return None
        frame = apply_transforms(read(source.reader, chunks[0].files), source)
        return frame.limit(limit).collect()
    except Exception:
        logger.exception("Failed to profile source dimensions: source=%s", source.id)
        return None
    finally:
        cleanup_temporaries()


def source_dimension_profile_rows(
    ctx: ValueStreamContext,
    source: model.Source,
    sample: pl.DataFrame,
) -> list[DimensionProfileRow]:
    """Profile transformed source fields using existing processors as context."""
    processors = processors_for_source(ctx, source.id)
    current_usage: dict[str, list[str]] = {}
    for processor in processors:
        for field in processor.group_by:
            current_usage.setdefault(str(field), []).append(processor.id)

    natural_keys = {str(field) for field in source.schema_.natural_key}
    protected_fields = {
        field
        for field in [
            source.schema_.timestamp_column,
            *natural_keys,
            *source_required_fields(source),
            *[field for processor in processors for field in processor_field_references(processor)],
        ]
        if field
    }
    return dimension_profile_rows(
        sample,
        current_usage=current_usage,
        protected_fields=protected_fields,
        active_reason=ACTIVE_PROCESSOR_REASON,
    )


def selection_dimension_profile_rows(
    sample: pl.DataFrame,
    *,
    selected_fields: Sequence[str],
    required_fields: Iterable[str] = (),
) -> list[DimensionProfileRow]:
    """Profile working fields using the current group-by selection as context."""
    current_usage = {
        str(field): ["Selected"] for field in selected_fields if field in sample.columns
    }
    return dimension_profile_rows(
        sample,
        current_usage=current_usage,
        protected_fields=required_fields,
        active_reason=ACTIVE_SELECTION_REASON,
    )


def dimension_profile_rows(
    sample: pl.DataFrame,
    *,
    current_usage: Mapping[str, Sequence[str]] | None = None,
    protected_fields: Iterable[str] = (),
    active_reason: str = ACTIVE_PROCESSOR_REASON,
) -> list[DimensionProfileRow]:
    """Profile fields for suitability as aggregate dimensions."""
    current_usage_map = {
        str(field): [str(item) for item in usage if str(item)]
        for field, usage in (current_usage or {}).items()
    }
    protected_set = {str(field) for field in protected_fields if str(field)}
    rows: list[DimensionProfileRow] = []
    total_rows = sample.height
    for field in sample.columns:
        series = sample.get_column(field)
        non_null = int(series.len() - series.null_count())
        non_null_values = series.drop_nulls()
        cardinality = int(non_null_values.n_unique()) if non_null else 0
        null_rate = float(series.null_count() / total_rows) if total_rows else 0.0
        cardinality_rate = float(cardinality / non_null) if non_null else 0.0
        field_usage = current_usage_map.get(field, [])
        recommendation, reason = dimension_recommendation(
            field,
            dtype=str(sample.schema.get(field, "")),
            current_usage=field_usage,
            protected=field in protected_set,
            non_null=non_null,
            cardinality=cardinality,
            cardinality_rate=cardinality_rate,
            null_rate=null_rate,
            active_reason=active_reason,
        )
        safe_for_group_by = group_by_safety(recommendation)
        rows.append(
            DimensionProfileRow(
                field=field,
                dtype=str(sample.schema.get(field, "")),
                non_null=non_null,
                null_rate=null_rate,
                cardinality=cardinality,
                cardinality_rate=cardinality_rate,
                sample_values=sample_values_text(non_null_values),
                current_usage=", ".join(field_usage),
                recommendation=recommendation,
                safe_for_group_by=safe_for_group_by,
                reason=reason,
            )
        )
    return sorted(rows, key=profile_sort_key)


def profile_frame(rows: Sequence[DimensionProfileRow]) -> pl.DataFrame:
    """Return display rows for a dimension profile table."""
    return pl.DataFrame(
        [
            {
                "Field": row.field,
                "Type": row.dtype,
                "Non-null": row.non_null,
                "Null %": round(row.null_rate * 100, 1),
                "Cardinality": row.cardinality,
                "Cardinality %": round(row.cardinality_rate * 100, 1),
                "Current Usage": row.current_usage,
                "Recommendation": row.recommendation,
                "Safe For Group-By": row.safe_for_group_by,
                "Reason": row.reason,
                "Sample Values": row.sample_values,
            }
            for row in rows
        ]
    )


def recommended_fields(
    rows: Sequence[DimensionProfileRow],
    *,
    allowed_fields: Iterable[str] | None = None,
    existing_fields: Iterable[str] = (),
) -> list[str]:
    """Return recommended dimension fields in profile order."""
    allowed_set = {str(field) for field in allowed_fields} if allowed_fields is not None else None
    existing_set = {str(field) for field in existing_fields}
    out: list[str] = []
    for row in rows:
        if row.recommendation != "Recommended" or row.field in existing_set:
            continue
        if allowed_set is not None and row.field not in allowed_set:
            continue
        out.append(row.field)
    return out


def default_group_by_fields(
    sample: pl.DataFrame,
    approved_fields: Sequence[str],
    *,
    required_fields: Iterable[str] = (),
    limit: int = 5,
) -> list[str]:
    """Choose initial group-by fields from approved low-cardinality dimensions."""
    approved_set = {str(field) for field in approved_fields if field in sample.columns}
    rows = dimension_profile_rows(sample, protected_fields=required_fields)
    return [
        row.field
        for row in rows
        if row.field in approved_set and row.recommendation == "Recommended"
    ][:limit]


def dimension_pack_names() -> list[str]:
    """Return the built-in dimension pack names."""
    return list(DIMENSION_PACKS)


def dimension_pack_fields(
    available_fields: Iterable[str],
    pack_name: str = PEGA_CDH_CORE_PACK,
) -> list[str]:
    """Return pack fields present in the available source fields."""
    available = {str(field).casefold(): str(field) for field in available_fields if str(field)}
    return [
        available[field.casefold()]
        for field in DIMENSION_PACKS.get(pack_name, ())
        if field.casefold() in available
    ]


def aggregate_size_preview(
    sample: pl.DataFrame,
    current_fields: Iterable[str],
    added_fields: Iterable[str] = (),
) -> AggregateSizePreview:
    """Estimate distinct aggregate groups before and after adding dimensions."""
    current = aggregate_tuple_count(sample, current_fields)
    projected = aggregate_tuple_count(sample, [*current_fields, *added_fields])
    return AggregateSizePreview(
        current_rows=current,
        projected_rows=projected,
        added_rows=max(0, projected - current),
        expansion_factor=float(projected / current) if current else 0.0,
        sample_rows=sample.height,
    )


def aggregate_tuple_count(sample: pl.DataFrame, fields: Iterable[str]) -> int:
    """Return distinct tuple count for fields that exist in the sample."""
    valid_fields = _dedupe([str(field) for field in fields if str(field) in sample.columns])
    if not valid_fields:
        return 1 if sample.height else 0
    return int(sample.select(valid_fields).unique().height)


def sketch_recommendations(
    rows: Sequence[DimensionProfileRow],
    *,
    existing_fields: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Return high-cardinality fields that are better explored as sketches."""
    existing = {str(field) for field in existing_fields}
    recommendations: list[dict[str, str]] = []
    for row in rows:
        if row.field in existing:
            continue
        if row.recommendation == "Avoid" and (
            "High-cardinality" in row.reason or looks_like_identity_field(row.field)
        ):
            state_type = "cpc" if looks_like_identity_field(row.field) else "topk"
            question = (
                "How many unique values exist?"
                if state_type == "cpc"
                else "What are the most frequent values?"
            )
            recommendations.append(
                {
                    "Field": row.field,
                    "Sketch": state_type.upper(),
                    "Question": question,
                    "Reason": row.reason,
                }
            )
    return recommendations


def profile_sort_key(row: DimensionProfileRow) -> tuple[int, str]:
    """Sort recommended candidates first and avoid fields last."""
    order = {"Recommended": 0, "Review": 1, "Active": 2, "Avoid": 3}
    return (order.get(row.recommendation, 9), row.field.casefold())


def dimension_recommendation(
    field: str,
    *,
    dtype: str,
    current_usage: Sequence[str],
    protected: bool,
    non_null: int,
    cardinality: int,
    cardinality_rate: float,
    null_rate: float,
    active_reason: str = ACTIVE_PROCESSOR_REASON,
) -> tuple[str, str]:
    """Classify one field for aggregate dimension suitability."""
    if current_usage:
        recommendation = "Active"
        reason = active_reason
    elif protected or looks_like_identity_field(field) or looks_like_measure_or_time_field(field):
        recommendation = "Avoid"
        reason = "Identity, timestamp, measure, outcome, or processor-control field."
    elif non_null == 0:
        recommendation = "Avoid"
        reason = "Only null values in sample."
    elif cardinality < RECOMMENDED_CARDINALITY_MIN:
        recommendation = "Review" if cardinality > 1 else "Avoid"
        reason = "Fewer than 3 distinct values; not useful as a default breakdown."
    elif cardinality <= LOW_CARDINALITY_MAX and null_rate <= 0.5:
        recommendation = "Recommended"
        reason = "Low-cardinality field suitable for filters and breakdowns."
    elif cardinality <= MEDIUM_CARDINALITY_MAX and cardinality_rate <= 0.5:
        recommendation = "Review"
        reason = "Moderate cardinality; consider Month/Summary grains or business approval."
    elif dtype_is_numeric_or_temporal(dtype) and cardinality > LOW_CARDINALITY_MAX:
        recommendation = "Avoid"
        reason = "High-cardinality numeric/time field; bin or derive a category first."
    elif cardinality > HIGH_CARDINALITY_MAX or cardinality_rate > 0.8:
        recommendation = "Avoid"
        reason = "High-cardinality field will expand aggregate size."
    else:
        recommendation = "Review"
        reason = "Potentially useful, but check cardinality, null rate, and privacy risk."
    return recommendation, reason


def group_by_safety(recommendation: str) -> str:
    """Return a compact group-by safety label for display."""
    if recommendation in {"Recommended", "Active"}:
        return "Yes"
    if recommendation == "Review":
        return "Review"
    return "No"


def source_required_fields(source: model.Source) -> list[str]:
    """Return fields required by source-level transforms."""
    fields: list[str] = []
    for transform in source.transforms:
        if isinstance(transform, model.ParseDatetime):
            fields.extend(str(column) for column in transform.columns)
        elif isinstance(transform, model.DeriveCalendar):
            fields.append(str(transform.from_))
            fields.extend(str(output) for output in transform.outputs)
    return fields


def processor_field_references(processor: model.Processor) -> list[str]:
    """Return source fields that processor logic reads outside dimensions."""
    fields: list[str] = []
    extra = dict(processor.model_extra or {})
    entities = extra.get("entities")
    if isinstance(entities, dict):
        fields.append(str(entities.get("subject", "") or ""))
    outcome = extra.get("outcome")
    if isinstance(outcome, dict):
        fields.append(str(outcome.get("column", "") or ""))
    for spec in model.effective_processor_states(processor).values():
        state_extra = dict(spec.model_extra or {})
        fields.append(str(state_extra.get("source_column", "") or ""))
    for key in (
        "value_column",
        "score_column",
        "entity_column",
        "stage_column",
        "snapshot_time_column",
    ):
        fields.append(str(extra.get(key, "") or ""))
    return fields


def looks_like_identity_field(field: str) -> bool:
    """Return whether a field name is likely an identity or natural key."""
    lower = field.casefold()
    compact = lower.replace("_", "").replace("-", "").replace(" ", "")
    if lower in {"id", "key"} or lower.endswith(("_id", "-id", " id")):
        return True
    if field.endswith(("ID", "Id")) and len(field) > 2:
        return True
    if any(hint in compact for hint in ENTITY_HINTS) and compact.endswith(
        ("id", "key", "number", "token")
    ):
        return True
    return any(hint in compact for hint in IDENTITY_HINTS if hint not in {"id", "key"})


def looks_like_measure_or_time_field(field: str) -> bool:
    """Return whether a field name suggests a measure or time/control field."""
    lower = field.casefold()
    compact = lower.replace("_", "").replace("-", "").replace(" ", "")
    if compact.endswith(("date", "time", "timestamp")):
        return True
    snakeish = lower.replace("-", "_").replace(" ", "_")
    return any(hint in compact or hint in snakeish for hint in MEASURE_OR_TIME_HINTS)


def dtype_is_numeric_or_temporal(dtype: str) -> bool:
    """Return whether a dtype should usually be binned before grouping."""
    normalized = dtype.casefold()
    return any(
        token in normalized
        for token in (
            "date",
            "datetime",
            "decimal",
            "duration",
            "float",
            "int",
            "time",
            "uint",
        )
    )


def sample_values_text(series: pl.Series, *, limit: int = 5) -> str:
    """Return compact distinct sample values for profile display."""
    values = []
    for value in series.unique(maintain_order=True).head(limit).to_list():
        text = str(value)
        if len(text) > 32:
            text = f"{text[:29]}..."
        values.append(text)
    return ", ".join(values)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
