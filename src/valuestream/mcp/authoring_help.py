"""Targeted, workspace-independent reference for the MCP catalog authoring tools."""

from __future__ import annotations

import re
from functools import cache
from typing import Any, Literal, get_args

from pydantic import TypeAdapter, ValidationError

from valuestream.ai.catalog_templates import (
    OutcomeRules,
    TemplateBindings,
    TemplateName,
    example_fragments,
    worked_examples,
)
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
    "template",
    "recipe",
]

AUTHORING_INSTRUCTIONS = (
    "Load the valuestream-catalog-authoring skill for requests to build catalogs or dashboards. "
    "For daily counts or Interaction History KPIs call catalog_draft_apply_template after draft creation. "
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
_EXAMPLES = example_fragments()


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


def _template_help(name: str) -> dict[str, Any]:
    names = get_args(TemplateName)
    if not name:
        return {"names": list(names)}
    if name not in names:
        raise ValueError(f"Unknown template {name!r}. Available: {', '.join(names)}")
    defaults = TemplateBindings().model_dump()
    arguments = {
        "template": name,
        "bindings": defaults
        if name == "interaction_metrics"
        else {k: defaults[k] for k in ("timestamp", "channel")},
    }
    if name == "interaction_metrics":
        arguments["outcomes"] = OutcomeRules().model_dump()
    return {
        "tool": "catalog_draft_apply_template",
        "arguments": arguments,
        "instructions": "Supply draft_id and expected_revision from the latest edit. Bind confirmed fields and outcome rules. Set timestamp_format for text, or null for a temporal field.",
    }


def authoring_help(
    topic: HelpTopic = "workflow", name: str = "", definition: str = ""
) -> dict[str, Any]:
    """Describe one piece of the contract without reading a workspace or a sample."""
    if topic in _SCHEMA_TYPES:
        return _schema_help(topic, name, definition)
    if topic == "template":
        return _template_help(name)
    if topic == "workflow":
        return {
            "instructions": AUTHORING_INSTRUCTIONS,
            "topics": [*_SCHEMA_TYPES, "expression", "operation", "recipe", "template", "example"],
            "scope": "Keep the sampled source id and reader. Edit its schema and transforms. Preserve unsampled sources and their dependent objects. Approved physical fields are fixed at creation; create a new draft if the goal needs more fields.",
            "edit": "Use catalog_draft_update for complete changed JSON sections, including source transforms. For individual processors, metrics, tiles or KPI recipes use catalog_draft_apply_operations. Both take draft_id and expected_revision from the latest response.",
            "daily_count": "Use catalog_draft_apply_template with template=daily_rows. Bind timestamp and channel (the grouping field) and set the timestamp format. See topic=template, name=daily_rows for defaults.",
            "rates_and_auc": "Use catalog_draft_apply_template with template=interaction_metrics for five KPIs, daily trends and volume. Confirm outcome classes, score and denominator. See topic=template, name=interaction_metrics for bindings.",
            "review": "Pass the requested metric names and dashboard/page/tile IDs as expected_metrics and expected_tiles to catalog_draft_preview. Present that revision for approval when the checks pass.",
        }
    if topic == "expression":
        return _expression_help(name)
    if topic == "recipe":
        entries = {r.id: r for r in load_builtin_kpi_recipes().recipes}
    elif topic == "example":
        entries = worked_examples()
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
