"""Parameterised catalog fragments shared by MCP examples and executable draft templates."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TemplateName = Literal["daily_rows", "interaction_metrics"]
PEGA_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%.3f %Z"


class TemplateBindings(BaseModel):
    """Physical column names. Defaults match the Pega IH sample."""

    model_config = ConfigDict(extra="forbid", str_min_length=1)
    timestamp: str = "pxOutcomeTime"
    channel: str = "pyChannel"
    outcome: str = "pyOutcome"
    score: str = "pyPropensity"
    visitor: str = "pySubjectID"
    action: str = "pyName"


class OutcomeRules(BaseModel):
    """Classified rows form the denominator. Each row has one final outcome."""

    model_config = ConfigDict(extra="forbid", str_min_length=1)
    positive: list[str] = Field(default_factory=lambda: ["Clicked", "Conversion"], min_length=1)
    negative: list[str] = Field(default_factory=lambda: ["Impression", "Pending"], min_length=1)
    conversion: list[str] = Field(default_factory=lambda: ["Conversion"], min_length=1)

    @model_validator(mode="after")
    def check_classes(self) -> Self:
        positive, negative, conversion = (
            set(self.positive),
            set(self.negative),
            set(self.conversion),
        )
        if positive & negative:
            raise ValueError("Positive and negative outcome classes must be disjoint")
        if not conversion < (positive | negative):
            raise ValueError("Conversion must be a proper subset of the classified outcomes")
        return self


def example_fragments(
    source_id: str = "ih",
    bindings: TemplateBindings | None = None,
    outcomes: OutcomeRules | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    fields = bindings or TemplateBindings()
    rules = outcomes or OutcomeRules()
    dimension = "channel" if fields.channel == "pyChannel" else fields.channel
    outcome_spec = {
        "column": fields.outcome,
        "positive_values": rules.positive,
        "negative_values": rules.negative,
    }
    daily = {
        "source": source_id,
        "time": {"property": fields.timestamp, "grain": "daily"},
        "group_by": ["Day", fields.channel],
    }
    return {
        "transform": {
            "parse_datetime": {
                "kind": "parse_datetime",
                "columns": [fields.timestamp],
                "format": PEGA_TIMESTAMP_FORMAT,
            },
            "derive_calendar": {
                "kind": "derive_calendar",
                "from": fields.timestamp,
                "outputs": ["Day"],
            },
        },
        "processor": {
            "funnel": {
                **daily,
                "id": f"{source_id}_records",
                "kind": "funnel",
                "stages": [{"name": "Records", "when": {"lit": True}}],
                "states": {"Count": {"type": "count"}},
            },
            "binary_outcome": {
                **daily,
                "id": f"{source_id}_engagement",
                "kind": "binary_outcome",
                "outcome": outcome_spec,
                "states": {
                    "Count": {"type": "count"},
                    "Positives": {"type": "count", "outcome": "positive"},
                    "Visitors": {"type": "cpc", "source_column": fields.visitor},
                },
            },
            "score_distribution": {
                **daily,
                "id": f"{source_id}_scores",
                "kind": "score_distribution",
                "score_properties": [{"column": fields.score, "role": "primary"}],
                "outcome": outcome_spec,
                "states": {
                    name: {
                        "type": "tdigest",
                        "source_column": fields.score,
                        "score_property": fields.score,
                        "outcome": outcome,
                    }
                    for name, outcome in (("P", "positive"), ("N", "negative"))
                },
            },
        },
        "metric": {
            "formula": {
                "kind": "formula",
                "processor": f"{source_id}_engagement",
                "description": "Share of classified interactions with positive outcomes.",
                "expression": {
                    "op": "safe_div",
                    "num": {"col": "Positives"},
                    "den": {"col": "Count"},
                },
                "display": {"value_format": "percent"},
            },
            "curve_from_digests": {
                "kind": "curve_from_digests",
                "processor": f"{source_id}_scores",
                "description": "ROC AUC of response propensity.",
                "positive_state": "P",
                "negative_state": "N",
                "output": "roc_auc",
            },
            "approx_distinct_count": {
                "kind": "approx_distinct_count",
                "processor": f"{source_id}_engagement",
                "description": "Estimated distinct visitors.",
                "state": "Visitors",
            },
        },
        "chart": {
            "line": {
                "id": "daily_rows",
                "title": f"Daily interactions by {dimension}",
                "metric": f"{source_id}_Rows",
                "chart": "line",
                "x": "Day",
                "color": fields.channel,
            }
        },
    }


def worked_examples(
    source_id: str = "ih",
    bindings: TemplateBindings | None = None,
    outcomes: OutcomeRules | None = None,
) -> dict[str, Any]:
    """A connected example, expressed in the same operations accepted by the draft tool."""
    fields = bindings or TemplateBindings()
    rules = outcomes or OutcomeRules()
    dimension = "channel" if fields.channel == "pyChannel" else fields.channel
    fragments = example_fragments(source_id, fields, rules)
    processors = fragments["processor"]
    metrics = fragments["metric"]
    binary = processors["binary_outcome"]
    engagement = {
        **binary,
        "states": {**binary["states"], "Actions": {"type": "cpc", "source_column": fields.action}},
    }
    conversions = {
        **binary,
        "id": f"{source_id}_conversions",
        "outcome": {
            **binary["outcome"],
            "positive_values": rules.conversion,
            "negative_values": [
                v for v in (*rules.positive, *rules.negative) if v not in rules.conversion
            ],
        },
        "states": {
            "Count": {"type": "count"},
            "Positives": {"type": "count", "outcome": "positive"},
        },
    }
    metric_defs = {
        "CTR": {
            **metrics["formula"],
            "description": f"Positive outcomes ({', '.join(rules.positive)}) divided by classified interactions.",
        },
        "ConversionRate": {
            **metrics["formula"],
            "processor": f"{source_id}_conversions",
            "description": f"Conversions ({', '.join(rules.conversion)}) divided by classified interactions.",
        },
        "RocAuc": metrics["curve_from_digests"],
        "UniqueVisitors": metrics["approx_distinct_count"],
        "UniqueActions": {
            **metrics["approx_distinct_count"],
            "state": "Actions",
            "description": "Estimated distinct action names.",
        },
    }
    target = {"op": "set_tile", "dashboard": f"{source_id}_overview", "page": "overview"}
    return {
        "daily_rows": {
            "instructions": "For source ih, append these transforms in order to its existing pipeline transforms and submit the complete pipelines section via catalog_draft_update. Then pass operations to catalog_draft_apply_operations using the updated revision. Adapt IDs and field names to your draft.",
            "append_transforms": list(fragments["transform"].values()),
            "operations": [
                {"op": "set_processor", "processor": processors["funnel"]},
                {**target, "tile": fragments["chart"]["line"]},
            ],
        },
        "interaction_metrics": {
            "prerequisite": "Use daily_rows first. The example assumes one outcome per row, the listed outcome classes, nonempty visitor/action IDs, and a confirmed propensity score. Adapt these assumptions to the user's goals and sample.",
            "operations": [
                *(
                    {"op": "set_processor", "processor": proc}
                    for proc in (engagement, conversions, processors["score_distribution"])
                ),
                *(
                    {"op": "set_metric", "name": name, "metric": metric}
                    for name, metric in metric_defs.items()
                ),
                *(
                    {
                        **target,
                        "tile": {
                            "id": f"{name}_total",
                            "title": name,
                            "metric": name,
                            "chart": "kpi_card",
                            "placement": "kpi_strip",
                        },
                    }
                    for name in metric_defs
                ),
                {
                    **target,
                    "tile": {
                        "id": "volume",
                        "title": f"Interaction volume by {dimension}",
                        "metric": f"{source_id}_Rows",
                        "chart": "bar",
                        "x": fields.channel,
                    },
                },
                *(
                    {
                        **target,
                        "tile": {
                            **fragments["chart"]["line"],
                            "id": f"{name}_day",
                            "metric": name,
                            "title": f"{name} by day and {dimension}",
                        },
                    }
                    for name in metric_defs
                ),
            ],
        },
    }


def template_edits(
    source_id: str,
    template: TemplateName,
    bindings: TemplateBindings,
    outcomes: OutcomeRules,
    *,
    timestamp_format: str | None,
    approved_fields: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roles = (
        {"timestamp", "channel"} if template == "daily_rows" else set(TemplateBindings.model_fields)
    )
    required = set(bindings.model_dump(include=roles).values())
    if missing := required.difference(approved_fields):
        raise ValueError(
            f"Unapproved template bindings: {', '.join(sorted(missing))}. Check field names or create a draft with these fields approved."
        )
    examples = worked_examples(source_id, bindings, outcomes)
    daily = examples["daily_rows"]
    parse, calendar = daily["append_transforms"]
    transforms = (
        [calendar]
        if timestamp_format is None
        else [{**parse, "format": timestamp_format}, calendar]
    )
    operations = daily["operations"]
    if template == "interaction_metrics":
        operations.extend(examples[template]["operations"])
    return transforms, operations
