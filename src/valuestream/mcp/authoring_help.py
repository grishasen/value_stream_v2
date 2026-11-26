"""Targeted, workspace-independent reference for the MCP catalog authoring tools."""

from __future__ import annotations

import re
from functools import cache
from typing import Any, Literal, get_args

from pydantic import TypeAdapter, ValidationError

from valuestream.ai.copilot import OPERATION_DICTIONARY
from valuestream.ai.studio import catalog_prompt_dictionaries
from valuestream.config import model
from valuestream.recipes import load_builtin_kpi_recipes

HelpTopic = Literal[
    "workflow",
    "transform",
    "processor",
    "state",
    "metric",
    "chart",
    "expression",
    "operation",
    "example",
    "recipe",
]

AUTHORING_INSTRUCTIONS = (
    "For catalog authoring use the listed Value Stream tools directly. "
    "Paths starting with data/ are relative to this server's workspace. "
    "Use sample_profile for sample inspection. When a field, shape, operation or KPI is "
    "unclear, call catalog_authoring_help for that topic before editing. "
    "Use this live reference in place of browsing source files, local catalogs or shell. "
    "Edit the draft, validate, preview the requested changes, then obtain authorization to apply."
)

_SCHEMA_TYPES: dict[str, Any] = {
    "transform": model.Transform,
    "processor": model.Processor,
    "state": model.StateSpec,
    "metric": model.Metric,
    "chart": model.Tile,
}

# Small compatible fragments of a daily IH catalog. Substitute the sample's approved
# fields, source ID, confirmed date format and business-specific outcome classes.
_OUTCOME = {
    "column": "pyOutcome",
    "positive_values": ["Clicked", "Conversion"],
    "negative_values": ["Impression", "Pending"],
}
_DAILY = {
    "source": "ih",
    "time": {"property": "pxOutcomeTime", "grain": "daily"},
    "group_by": ["Day", "pyChannel"],
}
_EXAMPLES: dict[str, dict[str, dict[str, Any]]] = {
    "transform": {
        "parse_datetime": {
            "kind": "parse_datetime",
            "columns": ["pxOutcomeTime"],
            "format": "%Y%m%dT%H%M%S%.3f %Z",
        },
        "derive_calendar": {"kind": "derive_calendar", "from": "pxOutcomeTime", "outputs": ["Day"]},
    },
    "processor": {
        "funnel": {
            **_DAILY,
            "id": "ih_records",
            "kind": "funnel",
            "stages": [{"name": "Records", "when": {"lit": True}}],
            "states": {"Count": {"type": "count"}},
        },
        "binary_outcome": {
            **_DAILY,
            "id": "ih_engagement",
            "kind": "binary_outcome",
            "outcome": _OUTCOME,
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count", "outcome": "positive"},
                "Visitors": {"type": "cpc", "source_column": "pySubjectID"},
            },
        },
        "score_distribution": {
            **_DAILY,
            "id": "ih_scores",
            "kind": "score_distribution",
            "score_properties": [{"column": "pyPropensity", "role": "primary"}],
            "outcome": _OUTCOME,
            "states": {
                name: {
                    "type": "tdigest",
                    "source_column": "pyPropensity",
                    "score_property": "pyPropensity",
                    "outcome": outcome,
                }
                for name, outcome in (("P", "positive"), ("N", "negative"))
            },
        },
    },
    "metric": {
        "formula": {
            "kind": "formula",
            "processor": "ih_engagement",
            "description": "Share of classified interactions with positive outcomes.",
            "expression": {"op": "safe_div", "num": {"col": "Positives"}, "den": {"col": "Count"}},
            "display": {"value_format": "percent"},
        },
        "curve_from_digests": {
            "kind": "curve_from_digests",
            "processor": "ih_scores",
            "description": "ROC AUC of response propensity.",
            "positive_state": "P",
            "negative_state": "N",
            "output": "roc_auc",
        },
        "approx_distinct_count": {
            "kind": "approx_distinct_count",
            "processor": "ih_engagement",
            "description": "Estimated distinct visitors.",
            "state": "Visitors",
        },
    },
    "chart": {
        "line": {
            "id": "daily_rows",
            "title": "Daily interactions by channel",
            "metric": "ih_Rows",
            "chart": "line",
            "x": "Day",
            "color": "pyChannel",
        }
    },
}


@cache
def _variant_schema(topic: str) -> dict[str, Any]:
    return TypeAdapter(_SCHEMA_TYPES[topic]).json_schema()


def _lookup(entries: dict[str, Any], topic: str, name: str) -> Any:
    if name not in entries:
        raise ValueError(f"Unknown {topic} {name!r}. Available: {', '.join(sorted(entries))}")
    return entries[name]


def _schema_help(topic: str, name: str, definition: str) -> dict[str, Any]:
    union = _variant_schema(topic)
    variants = union["discriminator"]["mapping"]
    if not name:
        return {"names": sorted(variants)}
    root = _lookup(variants, topic, name).split("/")[-1]
    # Return one definition per call. All references resolve against the same union.
    schema = _lookup(union["$defs"], "definition", definition or root)
    result = {
        "schema": schema,
        "reference_lookup": "Pass a $defs reference's last component as definition with the same topic and name.",
    }
    if definition:
        return result
    dictionaries = catalog_prompt_dictionaries()
    if topic in {"processor", "metric"}:
        result["guidance"] = dictionaries[f"{topic}_kinds"].get(name, {})
    examples = _EXAMPLES.get(topic, {})
    if topic == "chart":
        examples = {**dictionaries["chart_required_fields"]["tile_examples"], **examples}
    if name in examples:
        result["example"] = examples[name]
    return result


def _worked_examples() -> dict[str, Any]:
    """A connected example, expressed in the same operations accepted by the draft tool."""
    processors = _EXAMPLES["processor"]
    metrics = _EXAMPLES["metric"]
    binary = processors["binary_outcome"]
    engagement = {
        **binary,
        "states": {**binary["states"], "Actions": {"type": "cpc", "source_column": "pyName"}},
    }
    conversions = {
        **binary,
        "id": "ih_conversions",
        "outcome": {
            **_OUTCOME,
            "positive_values": ["Conversion"],
            "negative_values": ["Clicked", "Impression", "Pending"],
        },
        "states": {
            "Count": {"type": "count"},
            "Positives": {"type": "count", "outcome": "positive"},
        },
    }
    metric_defs = {
        "CTR": {
            **metrics["formula"],
            "description": "Clicked or Conversion outcomes divided by all four classified outcomes.",
        },
        "ConversionRate": {
            **metrics["formula"],
            "processor": "ih_conversions",
            "description": "Conversion outcomes divided by classified interactions.",
        },
        "RocAuc": metrics["curve_from_digests"],
        "UniqueVisitors": metrics["approx_distinct_count"],
        "UniqueActions": {
            **metrics["approx_distinct_count"],
            "state": "Actions",
            "description": "Estimated distinct action names.",
        },
    }
    target = {"op": "set_tile", "dashboard": "ih_overview", "page": "overview"}
    return {
        "daily_rows": {
            "instructions": "For source ih, append these transforms in order to its existing pipeline transforms and submit the complete pipelines section via catalog_draft_update. Then pass operations to catalog_draft_apply_operations using the updated revision. Adapt IDs and field names to your draft.",
            "append_transforms": list(_EXAMPLES["transform"].values()),
            "operations": [
                {"op": "set_processor", "processor": processors["funnel"]},
                {**target, "tile": _EXAMPLES["chart"]["line"]},
            ],
        },
        "interaction_metrics": {
            "prerequisite": "Use daily_rows first. The example assumes one outcome per row, the four listed outcome values, nonempty visitor/action IDs, and a confirmed propensity score. Adapt these assumptions to the user's goals and sample.",
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
                        "title": "Interaction volume by channel",
                        "metric": "ih_Rows",
                        "chart": "bar",
                        "x": "pyChannel",
                    },
                },
                *(
                    {
                        **target,
                        "tile": {
                            **_EXAMPLES["chart"]["line"],
                            "id": f"{name}_day",
                            "metric": name,
                            "title": f"{name} by day and channel",
                        },
                    }
                    for name in metric_defs
                ),
            ],
        },
    }


def _expression_help(name: str) -> dict[str, Any]:
    ast = catalog_prompt_dictionaries()["expression_ast"]
    forms = ast["operator_forms"]
    if not name:
        return {
            "atoms": ast["atoms"],
            "names": sorted({op for form in forms.values() for op in form["ops"]}),
        }
    shapes = [form["shape"] for form in forms.values() if name in form["ops"]]
    if not shapes:
        raise ValueError(f"Unknown expression {name!r}. Omit name to list operators.")
    return {"name": name, "forms": shapes, "allowed_values": ast["allowed_values"]}


def authoring_help(
    topic: HelpTopic = "workflow", name: str = "", definition: str = ""
) -> dict[str, Any]:
    """Describe one piece of the contract without reading a workspace or a sample."""
    if topic in _SCHEMA_TYPES:
        return _schema_help(topic, name, definition)
    if topic == "workflow":
        return {
            "instructions": AUTHORING_INSTRUCTIONS,
            "topics": [*_SCHEMA_TYPES, "expression", "operation", "recipe", "example"],
            "scope": "Keep the sampled source id and reader. Edit its schema and transforms. Preserve unsampled sources and their dependent objects. Approved physical fields are fixed at creation; create a new draft if the goal needs more fields.",
            "edit": "Use catalog_draft_update for complete changed JSON sections, including source transforms. For individual processors, metrics, tiles or KPI recipes use catalog_draft_apply_operations. Both take draft_id and expected_revision from the latest response.",
            "daily_count": "Call topic=example, name=daily_rows for a complete edit. See transform/parse_datetime then transform/derive_calendar, processor/funnel and chart/line for compatible examples. Keep the baseline row metric bound to the daily processor. Adapt example names and date format to the sample.",
            "rates_and_auc": "Call topic=example, name=interaction_metrics for a five-metric batch. See processor/binary_outcome, metric/formula, processor/score_distribution, metric/curve_from_digests and metric/approx_distinct_count. Confirm classes, score and denominator. For predefined KPIs list recipe names.",
            "review": "Validation checks structure and execution. Verify that the requested metrics and tiles appear in preview before presenting that revision for approval.",
        }
    if topic == "expression":
        return _expression_help(name)
    if topic == "recipe":
        entries = {r.id: r for r in load_builtin_kpi_recipes().recipes}
    elif topic == "example":
        entries = _worked_examples()
    else:
        entries = OPERATION_DICTIONARY
    if not name:
        return {"names": sorted(entries)}
    entry = _lookup(entries, topic, name)
    if topic == "recipe":
        entry = entry.model_dump(mode="json", exclude_none=True)
    return {"name": name, "contract": entry}


def help_call(issue: str | Exception, *, fallback: str = "workflow") -> dict[str, Any]:
    """Prefer the validator's tagged field path over words from its human-readable message."""
    if isinstance(issue, ValidationError):
        location = issue.errors(include_input=False, include_url=False)[0]["loc"]
        words = [str(part) for part in reversed(location)]
    else:
        message = str(issue).splitlines()[0] if str(issue) else ""
        # Unknown names list valid alternatives. Suggest the index, not its first entry.
        unknown = re.match(r"Unknown (?:KPI )?(\w+)", message)
        if unknown and unknown[1] in get_args(HelpTopic):
            return {"tool": "catalog_authoring_help", "arguments": {"topic": unknown[1]}}
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", message)
    topics = {
        "transforms": "transform",
        "states": "state",
        "tiles": "chart",
        "processors": "processor",
        "metrics": "metric",
    }
    for word in words:
        if word in OPERATION_DICTIONARY:
            return {
                "tool": "catalog_authoring_help",
                "arguments": {"topic": "operation", "name": word},
            }
        for topic in _SCHEMA_TYPES:
            if word in _variant_schema(topic)["discriminator"]["mapping"]:
                return {
                    "tool": "catalog_authoring_help",
                    "arguments": {"topic": topic, "name": word},
                }
    topic = next((topics[word] for word in words if word in topics), fallback)
    return {"tool": "catalog_authoring_help", "arguments": {"topic": topic}}
