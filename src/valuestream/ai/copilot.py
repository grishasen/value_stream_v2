"""Copilot support for the AI Configuration Studio.

The copilot answers free-form requests with a small closed set of structured
draft operations instead of full YAML files. Each operation is applied to a
copy of the session draft and held in the same pending review as generated
drafts, so every change stays individually reviewable before it can touch the
workspace catalog.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import yaml

from valuestream.ai.studio import (
    catalog_prompt_dictionaries,
    filter_draft_by_selection,
    prompt_draft_sections,
    tile_keys,
)
from valuestream.config import model
from valuestream.recipes import (
    instantiate_metric,
    instantiate_tile,
    load_builtin_kpi_recipes,
    processor_with_recipe_states,
    recipe_readiness,
)
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_HISTORY_LIMIT = 8
_COVERAGE_STATUSES = ("covered", "partial", "missing")

OPERATION_DICTIONARY: dict[str, Any] = {
    "set_processor": {
        "purpose": "Add or replace one processor definition.",
        "shape": {
            "op": "set_processor",
            "previous_id": "existing id when renaming, otherwise omit",
            "processor": "complete processor mapping with id",
        },
    },
    "remove_processor": {
        "purpose": "Remove one processor; dependent metrics and tiles are removed automatically.",
        "shape": {"op": "remove_processor", "id": "existing processor id"},
    },
    "set_metric": {
        "purpose": "Add or replace one metric definition.",
        "shape": {
            "op": "set_metric",
            "previous_name": "existing id when renaming, otherwise omit",
            "name": "metric id",
            "metric": "complete metric mapping",
        },
    },
    "remove_metric": {
        "purpose": "Remove one metric; dependent tiles are removed automatically.",
        "shape": {"op": "remove_metric", "name": "existing metric id"},
    },
    "set_tile": {
        "purpose": "Add or replace one report tile; the dashboard and page are created if missing.",
        "shape": {
            "op": "set_tile",
            "dashboard": "dashboard id",
            "page": "page id",
            "tile": "complete tile mapping with id",
        },
    },
    "remove_tile": {
        "purpose": "Remove one report tile.",
        "shape": {
            "op": "remove_tile",
            "dashboard": "dashboard id",
            "page": "page id",
            "id": "tile id",
        },
    },
    "set_dashboards": {
        "purpose": "Replace the whole dashboards section; use only for large reworks.",
        "shape": {"op": "set_dashboards", "dashboards": "complete dashboards.yaml mapping"},
    },
    "install_recipe": {
        "purpose": "Install one built-in KPI recipe against an existing compatible processor.",
        "shape": {
            "op": "install_recipe",
            "recipe_id": "built-in recipe id",
            "processor": "existing processor id",
            "metric_id": "new metric id",
            "bindings": "recipe role to existing processor state mapping; omit when unambiguous",
            "dashboard": "optional existing or new dashboard id",
            "page": "optional existing or new page id",
            "tile_id": "required when dashboard and page are supplied",
        },
    },
}

_STEP_HINTS: dict[str, str] = {
    "Sample": "The user is describing the source sample and runtime settings.",
    "Required Fields": "The user is mapping subject, outcome, and time columns.",
    "Defaults": "The user is filling default values for missing source fields.",
    "Filters": "The user is restricting which source rows are ingested.",
    "Calculations": "The user is adding derived columns to the working schema.",
    "Approve Fields": "The user decides which fields AI generation may use.",
    "AI Draft": "The user is generating the first full catalog draft.",
    "Draft": "The user is generating the first full catalog draft.",
    "Workspace Draft": "The user is reviewing the draft loaded from the workspace catalog.",
    "Processors": "The user is reviewing processor definitions.",
    "Metrics": "The user is reviewing metric definitions.",
    "AI Reports": "The user is regenerating dashboards from the current metrics.",
    "Reports": "The user is regenerating dashboards from the current metrics.",
    "Reports Review": "The user is reviewing dashboard tiles.",
    "Chat": "The user is preparing Chat With Data descriptions.",
    "Settings": "The user is adjusting dashboard theme and layout settings.",
    "Save & Export": "The user is validating and applying the draft to the workspace.",
}


@dataclass(frozen=True)
class CopilotQuestion:
    """One clarifying question with optional quick-reply options."""

    question: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class CopilotTurn:
    """Parsed copilot response: a reply plus optional operations and questions."""

    reply: str
    operations: list[dict[str, Any]] = field(default_factory=list)
    questions: list[CopilotQuestion] = field(default_factory=list)


@dataclass(frozen=True)
class RequirementCoverage:
    """Coverage judgement for one distinct business requirement."""

    requirement: str
    status: str
    metrics: tuple[str, ...] = ()
    tiles: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class DraftPatch:
    """One independently reviewable structural change between two drafts."""

    key: str
    section: str
    object_id: str
    change: str
    before: Any = None
    after: Any = None


@dataclass(frozen=True)
class CopilotRun:
    """Result of the bounded governed-operation loop."""

    turn: CopilotTurn
    pending_draft: dict[str, Any] | None = None
    summaries: tuple[str, ...] = ()
    validation_issues: tuple[str, ...] = ()
    responses: tuple[str, ...] = ()
    iterations: int = 0


class RecipeInstallRequestLike(Protocol):
    """Minimal recipe request contract shared with the Streamlit recipe browser."""

    metric_id: str
    metric_def: dict[str, Any]
    processor_id: str
    state_additions: dict[str, dict[str, Any]] | None
    processor_def: dict[str, Any] | None
    report_target: Any
    tile_def: dict[str, Any] | None


def apply_draft_operations(
    draft: dict[str, Any], operations: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Apply structured operations to a copy of the draft and describe each change."""

    updated = copy.deepcopy(draft)
    summaries: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each operation must be a mapping")
        kind = str(operation.get("op") or "")
        handler = _OPERATION_HANDLERS.get(kind)
        if handler is None:
            known = ", ".join(sorted(_OPERATION_HANDLERS))
            raise ValueError(f"Unknown operation {kind!r}; expected one of: {known}")
        updated, summary = handler(updated, operation)
        summaries.append(summary)
    return updated, summaries


def update_processor_definition(
    draft: dict[str, Any],
    old_id: str,
    new_id: str,
    processor: dict[str, Any],
) -> dict[str, Any]:
    """Add or update a processor and preserve metric references across a rename."""

    updated = copy.deepcopy(draft)
    processors = updated.setdefault("processors", {}).setdefault("processors", [])
    if not isinstance(processors, list):
        raise ValueError("Draft processors section is not a list")
    for index, existing in enumerate(processors):
        if isinstance(existing, dict) and str(existing.get("id") or "") == old_id:
            processors[index] = copy.deepcopy(processor)
            break
    else:
        processors.append(copy.deepcopy(processor))
    if old_id and old_id != new_id:
        metrics = updated.get("metrics", {}).get("metrics", {})
        if isinstance(metrics, dict):
            for metric in metrics.values():
                if isinstance(metric, dict) and metric.get("source") == old_id:
                    metric["source"] = new_id
    return updated


def update_metric_definition(
    draft: dict[str, Any],
    old_name: str,
    new_name: str,
    metric: dict[str, Any],
) -> dict[str, Any]:
    """Add or update a metric and preserve tile references across a rename."""

    updated = copy.deepcopy(draft)
    metrics = updated.setdefault("metrics", {}).setdefault("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Draft metrics section is not a mapping")
    if old_name and old_name != new_name:
        metrics.pop(old_name, None)
        _rename_tile_metric_references(updated, old_name, new_name)
    metrics[new_name] = copy.deepcopy(metric)
    return updated


def install_recipe_request_in_draft(
    draft: dict[str, Any], request: RecipeInstallRequestLike
) -> dict[str, Any]:
    """Apply the exact recipe-browser request through the shared operation layer."""

    updated = copy.deepcopy(draft)
    if request.processor_id and (request.processor_def or request.state_additions):
        processor_section = updated.setdefault("processors", {})
        processor_defs = processor_section.setdefault("processors", [])
        if not isinstance(processor_defs, list):
            raise TypeError("draft processors section must contain a processors list")
        typed_processors = model.Processors.model_validate(processor_section).processors
        processor = next(
            (item for item in typed_processors if item.id == request.processor_id),
            None,
        )
        if processor is None:
            raise ValueError(f"processor ID {request.processor_id!r} no longer exists")
        replacement = request.processor_def
        if replacement is None:
            configured = processor_with_recipe_states(processor, request.state_additions or {})
            replacement = _processor_definition(configured)
        processor_defs[:] = [
            copy.deepcopy(replacement)
            if isinstance(item, dict) and item.get("id") == request.processor_id
            else item
            for item in processor_defs
        ]
        model.Processors.model_validate(processor_section)

    metrics = updated.setdefault("metrics", {}).setdefault("metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("draft metrics section must contain a metrics mapping")
    if request.metric_id in metrics:
        raise ValueError(f"metric ID {request.metric_id!r} already exists")
    metrics[request.metric_id] = copy.deepcopy(request.metric_def)
    model.Metrics.model_validate(updated["metrics"])

    target = request.report_target
    if target is not None and request.tile_def:
        updated, _ = _apply_set_tile(
            updated,
            {
                "dashboard": target.dashboard_id,
                "dashboard_title": target.dashboard_title,
                "page": target.page_id,
                "page_title": target.page_title,
                "tile": request.tile_def,
            },
        )
    return updated


def _apply_set_processor(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    processor = op.get("processor")
    if not isinstance(processor, dict) or not str(processor.get("id") or "").strip():
        raise ValueError("set_processor requires a processor mapping with an id")
    processor_id = str(processor["id"]).strip()
    previous_id = str(op.get("previous_id") or processor_id).strip()
    existing_ids = {
        str(item.get("id"))
        for item in draft.get("processors", {}).get("processors", [])
        if isinstance(item, dict) and item.get("id")
    }
    updated = update_processor_definition(draft, previous_id, processor_id, processor)
    verb = "Updated" if previous_id in existing_ids else "Added"
    suffix = f" (renamed from '{previous_id}')" if previous_id != processor_id else ""
    return updated, f"{verb} processor '{processor_id}'{suffix}"


def _apply_remove_processor(
    draft: dict[str, Any], op: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    processor_id = str(op.get("id") or "").strip()
    if not processor_id:
        raise ValueError("remove_processor requires an id")
    existing = [
        str(processor.get("id"))
        for processor in draft.get("processors", {}).get("processors", [])
        if isinstance(processor, dict) and processor.get("id")
    ]
    if processor_id not in existing:
        raise ValueError(f"Processor '{processor_id}' does not exist in the draft")
    remaining = [item for item in existing if item != processor_id]
    filtered = filter_draft_by_selection(draft, selected_processors=remaining)
    return filtered, f"Removed processor '{processor_id}' and its dependent metrics and tiles"


def _apply_set_metric(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    name = str(op.get("name") or "").strip()
    metric = op.get("metric")
    if not name or not isinstance(metric, dict):
        raise ValueError("set_metric requires a name and a metric mapping")
    metrics = draft.get("metrics", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Draft metrics section is not a mapping")
    previous_name = str(op.get("previous_name") or name).strip()
    verb = "Updated" if previous_name in metrics else "Added"
    updated = update_metric_definition(draft, previous_name, name, metric)
    suffix = f" (renamed from '{previous_name}')" if previous_name != name else ""
    return updated, f"{verb} metric '{name}'{suffix}"


def _apply_remove_metric(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    name = str(op.get("name") or "").strip()
    if not name:
        raise ValueError("remove_metric requires a name")
    metrics = draft.get("metrics", {}).get("metrics", {})
    if not isinstance(metrics, dict) or name not in metrics:
        raise ValueError(f"Metric '{name}' does not exist in the draft")
    remaining = [item for item in metrics if item != name]
    filtered = filter_draft_by_selection(draft, selected_metrics=remaining)
    return filtered, f"Removed metric '{name}' and its dependent tiles"


def _apply_set_tile(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    dashboard_id = str(op.get("dashboard") or "").strip()
    page_id = str(op.get("page") or "").strip()
    tile = op.get("tile")
    if not dashboard_id or not page_id or not isinstance(tile, dict):
        raise ValueError("set_tile requires dashboard, page, and a tile mapping")
    tile_id = str(tile.get("id") or "").strip()
    if not tile_id:
        raise ValueError("set_tile requires a tile id")
    dashboards = draft.setdefault("dashboards", {}).setdefault("dashboards", [])
    if not isinstance(dashboards, list):
        raise ValueError("Draft dashboards section is not a list")
    dashboard = _find_by_id(dashboards, dashboard_id)
    if dashboard is None:
        dashboard = {
            "id": dashboard_id,
            "title": str(op.get("dashboard_title") or _title_from_identifier(dashboard_id)),
            "pages": [],
        }
        dashboards.append(dashboard)
    pages = dashboard.setdefault("pages", [])
    page = _find_by_id(pages, page_id)
    if page is None:
        page = {
            "id": page_id,
            "title": str(op.get("page_title") or _title_from_identifier(page_id)),
            "tiles": [],
        }
        pages.append(page)
    tiles = page.setdefault("tiles", [])
    key = f"{dashboard_id}/{page_id}/{tile_id}"
    for index, existing in enumerate(tiles):
        if isinstance(existing, dict) and existing.get("id") == tile_id:
            tiles[index] = tile
            return draft, f"Updated tile '{key}'"
    tiles.append(tile)
    return draft, f"Added tile '{key}'"


def _apply_remove_tile(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    dashboard_id = str(op.get("dashboard") or "").strip()
    page_id = str(op.get("page") or "").strip()
    tile_id = str(op.get("id") or "").strip()
    if not dashboard_id or not page_id or not tile_id:
        raise ValueError("remove_tile requires dashboard, page, and id")
    key = f"{dashboard_id}/{page_id}/{tile_id}"
    dashboard = _find_by_id(draft.get("dashboards", {}).get("dashboards", []), dashboard_id)
    page = _find_by_id(dashboard.get("pages", []), page_id) if dashboard else None
    tiles = page.get("tiles", []) if page else []
    remaining = [
        tile for tile in tiles if not (isinstance(tile, dict) and tile.get("id") == tile_id)
    ]
    if page is None or len(remaining) == len(tiles):
        raise ValueError(f"Tile '{key}' does not exist in the draft")
    page["tiles"] = remaining
    return draft, f"Removed tile '{key}'"


def _apply_set_dashboards(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    dashboards = op.get("dashboards")
    if not isinstance(dashboards, dict) or not isinstance(dashboards.get("dashboards"), list):
        raise ValueError("set_dashboards requires a mapping with a dashboards list")
    draft["dashboards"] = dashboards
    return draft, "Replaced the dashboards section"


def _apply_install_recipe(draft: dict[str, Any], op: dict[str, Any]) -> tuple[dict[str, Any], str]:
    recipe_id = str(op.get("recipe_id") or "").strip()
    processor_id = str(op.get("processor") or "").strip()
    metric_id = str(op.get("metric_id") or "").strip()
    if not recipe_id or not processor_id or not metric_id:
        raise ValueError("install_recipe requires recipe_id, processor, and metric_id")
    recipe = next(
        (item for item in load_builtin_kpi_recipes().recipes if item.id == recipe_id),
        None,
    )
    if recipe is None:
        raise ValueError(f"Unknown KPI recipe '{recipe_id}'")
    processors = model.Processors.model_validate(draft.get("processors", {})).processors
    processor = next((item for item in processors if item.id == processor_id), None)
    if processor is None:
        raise ValueError(f"Processor '{processor_id}' does not exist in the draft")
    readiness = recipe_readiness(recipe, processor)
    bindings_raw = op.get("bindings")
    bindings = (
        {str(key): str(value) for key, value in bindings_raw.items()}
        if isinstance(bindings_raw, dict)
        else dict(readiness.resolved_inputs)
    )
    expected_roles = {item.role for item in recipe.inputs}
    if set(bindings) != expected_roles:
        missing = ", ".join(sorted(expected_roles - set(bindings))) or "ambiguous bindings"
        raise ValueError(
            f"Recipe '{recipe_id}' needs clarification for: {missing}. "
            "Ask the user to choose from the recipe input options."
        )
    metrics = draft.get("metrics", {}).get("metrics", {})
    if isinstance(metrics, dict) and metric_id in metrics:
        raise ValueError(f"Metric '{metric_id}' already exists in the draft")
    metric = instantiate_metric(recipe, processor, metric_id, bindings)
    updated = update_metric_definition(draft, metric_id, metric_id, metric)
    dashboard_id = str(op.get("dashboard") or "").strip()
    page_id = str(op.get("page") or "").strip()
    tile_id = str(op.get("tile_id") or "").strip()
    if any((dashboard_id, page_id, tile_id)):
        if not all((dashboard_id, page_id, tile_id)):
            raise ValueError(
                "install_recipe requires dashboard, page, and tile_id together when adding a tile"
            )
        tile = instantiate_tile(recipe, metric_id, tile_id)
        updated, _ = _apply_set_tile(
            updated,
            {"dashboard": dashboard_id, "page": page_id, "tile": tile},
        )
    return updated, f"Installed recipe '{recipe_id}' as metric '{metric_id}'"


_OPERATION_HANDLERS = {
    "set_processor": _apply_set_processor,
    "remove_processor": _apply_remove_processor,
    "set_metric": _apply_set_metric,
    "remove_metric": _apply_remove_metric,
    "set_tile": _apply_set_tile,
    "remove_tile": _apply_remove_tile,
    "set_dashboards": _apply_set_dashboards,
    "install_recipe": _apply_install_recipe,
}


def _find_by_id(items: Any, item_id: str) -> dict[str, Any] | None:
    if not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    return None


def _title_from_identifier(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title() or value


def _rename_tile_metric_references(draft: dict[str, Any], old_name: str, new_name: str) -> None:
    for dashboard in draft.get("dashboards", {}).get("dashboards", []) or []:
        if not isinstance(dashboard, dict):
            continue
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            for tile in page.get("tiles", []) or []:
                if isinstance(tile, dict) and tile.get("metric") == old_name:
                    tile["metric"] = new_name


def _processor_definition(processor: model.Processor) -> dict[str, Any]:
    data = cast(
        dict[str, Any],
        processor.model_dump(mode="json", by_alias=True, exclude_none=True),
    )
    group_by = data.pop("group_by", None)
    if group_by:
        data["dimensions"] = group_by
    if not processor.states:
        data.pop("states", None)
    return data


def draft_patches(base: dict[str, Any], proposed: dict[str, Any]) -> list[DraftPatch]:
    """Return independently reviewable structural changes between two drafts."""

    patches: list[DraftPatch] = []
    patches.extend(
        _mapping_patches(
            "processors",
            _processors_by_id(base),
            _processors_by_id(proposed),
        )
    )
    patches.extend(
        _mapping_patches(
            "metrics",
            _metrics_by_name(base),
            _metrics_by_name(proposed),
        )
    )
    base_structure = _dashboard_structure(base)
    proposed_structure = _dashboard_structure(proposed)
    if base_structure != proposed_structure:
        patches.append(
            DraftPatch(
                key="dashboards:structure",
                section="dashboards",
                object_id="structure and settings",
                change=_change_kind(base_structure, proposed_structure),
                before=base_structure,
                after=proposed_structure,
            )
        )
    patches.extend(_mapping_patches("tiles", _tiles_by_key(base), _tiles_by_key(proposed)))
    base_chat = base.get("chat_with_data")
    proposed_chat = proposed.get("chat_with_data")
    if base_chat != proposed_chat:
        patches.append(
            DraftPatch(
                key="chat_with_data:settings",
                section="chat_with_data",
                object_id="settings",
                change=_change_kind(base_chat, proposed_chat),
                before=copy.deepcopy(base_chat),
                after=copy.deepcopy(proposed_chat),
            )
        )
    return patches


def merge_selected_draft_patches(
    base: dict[str, Any],
    proposed: dict[str, Any],
    accepted_patch_keys: list[str] | set[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Apply only accepted structural patches to a copy of the base draft."""

    accepted = set(accepted_patch_keys)
    patches = draft_patches(base, proposed)
    merged = copy.deepcopy(base)
    for patch in patches:
        if patch.section not in {"processors", "metrics"} or patch.key not in accepted:
            continue
        if patch.section == "processors":
            _set_processor_value(merged, patch.object_id, patch.after)
        else:
            _set_metric_value(merged, patch.object_id, patch.after)

    structure_patch = next(
        (patch for patch in patches if patch.key == "dashboards:structure"),
        None,
    )
    if structure_patch is not None and structure_patch.key in accepted:
        merged["dashboards"] = copy.deepcopy(proposed.get("dashboards", {}))
    else:
        merged["dashboards"] = copy.deepcopy(base.get("dashboards", {}))

    for patch in (item for item in patches if item.section == "tiles"):
        use_proposed = patch.key in accepted
        desired = patch.after if use_proposed else patch.before
        source = proposed if use_proposed else base
        _set_tile_value(merged, patch.object_id, desired, source)

    chat_patch = next(
        (patch for patch in patches if patch.section == "chat_with_data"),
        None,
    )
    if chat_patch is not None and chat_patch.key in accepted:
        if chat_patch.after is None:
            merged.pop("chat_with_data", None)
        else:
            merged["chat_with_data"] = copy.deepcopy(chat_patch.after)
    return merged


def run_copilot_tool_loop(
    *,
    prompt: str,
    draft: dict[str, Any],
    call_model: Callable[[str], str],
    validate: Callable[[dict[str, Any]], tuple[bool, list[str]]],
    max_iterations: int = 3,
) -> CopilotRun:
    """Run a bounded operation/validation loop without mutating the accepted draft."""

    limit = max(1, min(max_iterations, 5))
    candidate = copy.deepcopy(draft)
    responses: list[str] = []
    summaries: list[str] = []
    issues: list[str] = []
    current_prompt = prompt
    last_turn = CopilotTurn(reply="No copilot response was produced.")
    for iteration in range(1, limit + 1):
        response = call_model(current_prompt)
        responses.append(response)
        turn = parse_copilot_response(response)
        last_turn = turn
        if turn.questions:
            return CopilotRun(
                turn=turn,
                responses=tuple(responses),
                iterations=iteration,
            )
        if not turn.operations:
            return CopilotRun(
                turn=turn,
                responses=tuple(responses),
                iterations=iteration,
            )
        try:
            candidate, operation_summaries = apply_draft_operations(candidate, turn.operations)
        except (TypeError, ValueError) as exc:
            issues = [str(exc)]
        else:
            summaries.extend(operation_summaries)
            ok, issues = validate(candidate)
            if ok:
                return CopilotRun(
                    turn=turn,
                    pending_draft=candidate,
                    summaries=tuple(summaries),
                    responses=tuple(responses),
                    iterations=iteration,
                )
        if iteration < limit:
            current_prompt = _tool_correction_prompt(
                original_prompt=prompt,
                candidate=candidate,
                issues=issues,
            )
    failed_reply = (
        f"{last_turn.reply}\n\nNo pending change was created because the proposed "
        f"operations still failed validation after {limit} attempts."
    )
    return CopilotRun(
        turn=CopilotTurn(reply=failed_reply),
        validation_issues=tuple(issues),
        responses=tuple(responses),
        iterations=limit,
    )


def _tool_correction_prompt(
    *,
    original_prompt: str,
    candidate: dict[str, Any],
    issues: list[str],
) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous governed operations were applied to a temporary draft and failed. "
        "Return corrected operations against the current temporary draft. Do not repeat an "
        "invalid operation. Ask a clarifying question instead of guessing when the errors "
        "cannot be resolved from the approved context.\n\n"
        f"Validation or operation errors:\n{yaml.safe_dump(issues, sort_keys=False)}\n"
        "Current temporary draft:\n"
        f"{yaml.safe_dump(prompt_draft_sections(candidate), sort_keys=False)}\n"
        "Return the same JSON response contract only."
    )


def _mapping_patches(
    section: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[DraftPatch]:
    patches: list[DraftPatch] = []
    for object_id in sorted(set(before) | set(after), key=str.casefold):
        old = before.get(object_id)
        new = after.get(object_id)
        if old == new:
            continue
        patches.append(
            DraftPatch(
                key=f"{section}:{object_id}",
                section=section,
                object_id=object_id,
                change=_change_kind(old, new),
                before=copy.deepcopy(old),
                after=copy.deepcopy(new),
            )
        )
    return patches


def _change_kind(before: Any, after: Any) -> str:
    if before is None:
        return "added"
    if after is None:
        return "removed"
    return "changed"


def _processors_by_id(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    processors = draft.get("processors", {}).get("processors", [])
    if not isinstance(processors, list):
        return {}
    return {
        str(item["id"]): item for item in processors if isinstance(item, dict) and item.get("id")
    }


def _metrics_by_name(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = draft.get("metrics", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {str(name): item for name, item in metrics.items() if isinstance(item, dict)}


def _tiles_by_key(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dashboard in draft.get("dashboards", {}).get("dashboards", []) or []:
        if not isinstance(dashboard, dict) or not dashboard.get("id"):
            continue
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict) or not page.get("id"):
                continue
            for tile in page.get("tiles", []) or []:
                if isinstance(tile, dict) and tile.get("id"):
                    key = f"{dashboard['id']}/{page['id']}/{tile['id']}"
                    out[key] = tile
    return out


def _dashboard_structure(draft: dict[str, Any]) -> dict[str, Any]:
    raw = draft.get("dashboards", {})
    dashboards: dict[str, Any] = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    for dashboard in dashboards.get("dashboards", []):
        if not isinstance(dashboard, dict):
            continue
        for page in dashboard.get("pages", []) or []:
            if isinstance(page, dict):
                page["tiles"] = []
    return dashboards


def _set_processor_value(draft: dict[str, Any], processor_id: str, value: Any) -> None:
    processors = draft.setdefault("processors", {}).setdefault("processors", [])
    if not isinstance(processors, list):
        raise ValueError("Draft processors section is not a list")
    index = next(
        (
            index
            for index, item in enumerate(processors)
            if isinstance(item, dict) and item.get("id") == processor_id
        ),
        None,
    )
    if value is None:
        if index is not None:
            processors.pop(index)
        return
    if index is None:
        processors.append(copy.deepcopy(value))
    else:
        processors[index] = copy.deepcopy(value)


def _set_metric_value(draft: dict[str, Any], metric_id: str, value: Any) -> None:
    metrics = draft.setdefault("metrics", {}).setdefault("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Draft metrics section is not a mapping")
    if value is None:
        metrics.pop(metric_id, None)
    else:
        metrics[metric_id] = copy.deepcopy(value)


def _set_tile_value(
    draft: dict[str, Any],
    key: str,
    value: Any,
    metadata_source: dict[str, Any],
) -> None:
    dashboard_id, page_id, tile_id = key.split("/", 2)
    if value is None:
        dashboard = _find_by_id(draft.get("dashboards", {}).get("dashboards", []), dashboard_id)
        page = _find_by_id(dashboard.get("pages", []), page_id) if dashboard else None
        if page is not None:
            page["tiles"] = [
                item
                for item in page.get("tiles", []) or []
                if not (isinstance(item, dict) and item.get("id") == tile_id)
            ]
        return
    dashboard_title, page_title = _dashboard_page_titles(
        metadata_source,
        dashboard_id,
        page_id,
    )
    _apply_set_tile(
        draft,
        {
            "dashboard": dashboard_id,
            "dashboard_title": dashboard_title,
            "page": page_id,
            "page_title": page_title,
            "tile": value,
        },
    )


def _dashboard_page_titles(
    draft: dict[str, Any], dashboard_id: str, page_id: str
) -> tuple[str, str]:
    dashboard = _find_by_id(draft.get("dashboards", {}).get("dashboards", []), dashboard_id)
    page = _find_by_id(dashboard.get("pages", []), page_id) if dashboard else None
    return (
        str((dashboard or {}).get("title") or _title_from_identifier(dashboard_id)),
        str((page or {}).get("title") or _title_from_identifier(page_id)),
    )


def prompt_for_copilot(
    *,
    step: str,
    user_message: str,
    history: list[dict[str, str]],
    user_goals: str,
    approved_schema: list[dict[str, Any]],
    approved_fields: list[str],
    hidden_fields: list[str],
    current_draft: dict[str, Any],
) -> str:
    """Build the step-aware copilot prompt for one user message."""

    goals_text = ""
    if user_goals.strip():
        goals_text = f"Business requirements from the user:\n{user_goals.strip()}\n\n"
    transcript_lines = []
    for item in history[-_HISTORY_LIMIT:]:
        content = str(item.get("content") or "").strip()
        if content:
            role = str(item.get("role") or "user")
            transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines)
    history_text = f"Conversation so far:\n{transcript}\n\n" if transcript else ""
    return (
        "You are the configuration copilot inside Value Stream's AI Configuration Studio. "
        "You help the user turn free-form requests into reviewable draft changes.\n\n"
        f"The user is on the {_step_name(step)!r} studio step. {_step_hint(step)}\n\n"
        "Respond with a single JSON object and nothing else:\n"
        '{"reply": str, "operations": [Operation], "questions": '
        '[{"question": str, "options": [str]}]}\n'
        "- reply: short plain-language answer for the user.\n"
        "- operations: draft operations, only when the user asked for a concrete change. "
        "Leave empty when you are only advising or asking questions.\n"
        "- questions: when the request is ambiguous, ask before guessing and offer two to "
        "four concrete options per question. Leave empty otherwise.\n"
        "- Never return operations and questions in the same response. Resolve questions first.\n"
        "- Use only approved fields, existing processor ids, and existing metric ids in "
        "operations. Never invent source fields.\n\n"
        "Operation dictionary:\n"
        f"{yaml.safe_dump(OPERATION_DICTIONARY, sort_keys=False)}\n"
        "Built-in KPI recipe ids:\n"
        f"{yaml.safe_dump([item.id for item in load_builtin_kpi_recipes().recipes], sort_keys=False)}\n"
        "Catalog dictionaries:\n"
        f"{yaml.safe_dump(catalog_prompt_dictionaries(), sort_keys=False)}\n"
        f"{goals_text}"
        f"Approved fields:\n{yaml.safe_dump(approved_fields, sort_keys=False)}\n"
        f"Hidden fields:\n{yaml.safe_dump(hidden_fields, sort_keys=False)}\n"
        f"Approved schema preview:\n{yaml.safe_dump(approved_schema, sort_keys=False)}\n"
        f"Current draft:\n{yaml.safe_dump(prompt_draft_sections(current_draft), sort_keys=False)}\n"
        f"{history_text}"
        f"User message:\n{user_message.strip()}\n\n"
        "Return valid JSON only. Do not wrap the answer in prose or Markdown fences."
    )


def _step_name(step: str) -> str:
    return re.sub(r"^\d+\.\s*", "", step).strip() or step


def _step_hint(step: str) -> str:
    return _STEP_HINTS.get(_step_name(step), "")


def parse_copilot_response(text: str) -> CopilotTurn:
    """Parse the copilot JSON response into a validated turn."""

    payload = _json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("Copilot response must be a JSON object")
    reply = str(payload.get("reply") or "").strip()
    operations_raw = payload.get("operations") or []
    questions_raw = payload.get("questions") or []
    if not isinstance(operations_raw, list) or not isinstance(questions_raw, list):
        raise ValueError("Copilot operations and questions must be lists")
    operations = [item for item in operations_raw if isinstance(item, dict)]
    questions: list[CopilotQuestion] = []
    for item in questions_raw:
        if isinstance(item, str) and item.strip():
            questions.append(CopilotQuestion(question=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        options_raw = item.get("options") or []
        options = tuple(
            str(option).strip()
            for option in options_raw
            if isinstance(option, str | int | float) and str(option).strip()
        )
        questions.append(CopilotQuestion(question=question, options=options))
    if not reply and not operations and not questions:
        raise ValueError("Copilot response contains no reply, operations, or questions")
    if not reply:
        reply = "Proposed draft changes." if operations else questions[0].question
    return CopilotTurn(reply=reply, operations=operations, questions=questions)


def prompt_for_coverage(*, user_goals: str, draft: dict[str, Any]) -> str:
    """Build the prompt that maps business requirements onto the current draft."""

    metrics = sorted(draft.get("metrics", {}).get("metrics", {}), key=str.casefold)
    return (
        "Judge how well this Value Stream catalog draft covers the user's business "
        "requirements. Split the requirements into distinct individual requirements "
        "(at most 12). Judge only from the draft content.\n\n"
        "Respond with a single JSON array and nothing else:\n"
        '[{"requirement": str, "status": "covered"|"partial"|"missing", '
        '"metrics": [str], "tiles": [str], "note": str}]\n'
        "- metrics: existing metric ids that cover the requirement.\n"
        "- tiles: existing dashboard/page/tile keys that report it.\n"
        "- note: one short sentence explaining the judgement.\n\n"
        f"Business requirements:\n{user_goals.strip()}\n\n"
        f"Metric ids in the draft:\n{yaml.safe_dump(metrics, sort_keys=False)}\n"
        f"Tile keys in the draft:\n{yaml.safe_dump(tile_keys(draft), sort_keys=False)}\n"
        f"Current draft:\n{yaml.safe_dump(prompt_draft_sections(draft), sort_keys=False)}\n"
        "Return valid JSON only. Do not wrap the answer in prose or Markdown fences."
    )


def parse_coverage_response(
    text: str,
    *,
    draft: dict[str, Any] | None = None,
) -> list[RequirementCoverage]:
    """Parse coverage JSON and reject model references absent from the draft."""

    payload = _json_payload(text)
    if isinstance(payload, dict):
        payload = payload.get("coverage") or payload.get("requirements")
    if not isinstance(payload, list):
        raise ValueError("Coverage response must be a JSON array")
    rows: list[RequirementCoverage] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        requirement = str(item.get("requirement") or "").strip()
        if not requirement:
            continue
        status = str(item.get("status") or "").strip().casefold()
        if status not in _COVERAGE_STATUSES:
            raise ValueError(f"Coverage status must be one of {_COVERAGE_STATUSES}, got {status!r}")
        metrics = _string_tuple(item.get("metrics"))
        tiles = _string_tuple(item.get("tiles"))
        note = str(item.get("note") or "").strip()
        if draft is not None:
            known_metrics = set(_metrics_by_name(draft))
            known_tiles = set(_tiles_by_key(draft))
            unknown = [
                *(f"metric {name!r}" for name in metrics if name not in known_metrics),
                *(f"tile {name!r}" for name in tiles if name not in known_tiles),
            ]
            metrics = tuple(name for name in metrics if name in known_metrics)
            tiles = tuple(name for name in tiles if name in known_tiles)
            if unknown:
                suffix = "Ignored unknown draft references: " + ", ".join(unknown) + "."
                note = f"{note} {suffix}".strip()
            if status in {"covered", "partial"} and not metrics and not tiles:
                status = "missing"
                note = (
                    f"{note} Coverage was downgraded because no referenced metric or tile exists."
                ).strip()
        rows.append(
            RequirementCoverage(
                requirement=requirement,
                status=status,
                metrics=metrics,
                tiles=tiles,
                note=note,
            )
        )
    if not rows:
        raise ValueError("Coverage response contains no requirements")
    return rows


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _json_payload(text: str) -> Any:
    fenced = _JSON_FENCE_RE.search(text)
    raw = fenced.group(1).strip() if fenced else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Response does not contain valid JSON")
