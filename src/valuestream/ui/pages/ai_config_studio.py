"""Guided configuration studio page."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
import yaml
from streamlit.errors import StreamlitAPIException

from valuestream.ai import (
    AICallSettings,
    DraftPatch,
    call_litellm,
    classify_draft_validation_issues,
    draft_object_counts,
    draft_patches,
    filter_draft_by_selection,
    generate_schema_preview,
    install_recipe_request_in_draft,
    merge_draft_sections,
    merge_selected_draft_patches,
    parse_ai_yaml_sections,
    parse_coverage_response,
    prompt_for_config_draft,
    prompt_for_copilot,
    prompt_for_coverage,
    prompt_for_draft_refinement,
    prompt_for_repair,
    prompt_for_report_refresh,
    run_copilot_tool_loop,
    tile_keys,
    update_metric_definition,
    update_processor_definition,
    validate_draft_catalog,
    validation_trace_for_repair,
)
from valuestream.ai.settings import load_llm_settings_config, write_chat_with_data_config
from valuestream.config import model
from valuestream.engine import run_source
from valuestream.expr import parser as expr_parser
from valuestream.expr.translator import translate
from valuestream.ui import (
    builder,
    components,
    config_help,
    dimension_profile,
    field_remap,
    forms,
    recipe_library,
)
from valuestream.ui.context import ValueStreamContext
from valuestream.utils.logger import get_logger
from valuestream.utils.names import capitalize_fields

logger = get_logger(__name__)

STEPS = [
    "1. Sample",
    "2. Required Fields",
    "3. Defaults",
    "4. Filters",
    "5. Calculations",
    "6. Approve Fields",
    "7. AI Draft",
    "8. Processors",
    "9. Metrics",
    "10. AI Reports",
    "11. Reports Review",
    "12. Chat",
    "13. Settings",
    "14. Save & Export",
]
DETERMINISTIC_STEPS = [
    "1. Sample",
    "2. Required Fields",
    "3. Defaults",
    "4. Filters",
    "5. Calculations",
    "6. Approve Fields",
    "7. Draft",
    "8. Processors",
    "9. Metrics",
    "10. Reports",
    "11. Reports Review",
    "12. Chat",
    "13. Settings",
    "14. Save & Export",
]
CATALOG_DRAFT_STEPS = [
    "Workspace Draft",
    "Processors",
    "Metrics",
    "Reports",
    "Reports Review",
    "Chat",
    "Settings",
    "Save & Export",
]
AI_CALLS_ENABLED_STATE_KEY = "ai_studio_ai_calls_enabled"
AI_SHARING_CONFIRMATION_STATE_KEY = "ai_studio_ai_sharing_confirmed_signature"
AI_SHARING_CONFIRMATION_WIDGET_PREFIX = "ai_studio_ai_sharing_consent_"
AI_SHARING_CONTRACT_STATE_KEY = "ai_studio_ai_sharing_contract_signature"
CATALOG_DRAFT_SOURCE = "catalog"
STUDIO_PHASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Data", (0, 1, 2, 3, 4, 5)),
    ("Draft", (6,)),
    ("Review", (7, 8, 9, 10)),
    ("Publish", (11, 12, 13)),
)
_COPILOT_HISTORY_DISPLAY = 8
_PREPROCESSING_PATCH_SECTIONS = frozenset(
    {"source_defaults", "source_filters", "calculated_fields"}
)
_PREPROCESSING_SYNC_STATE_KEY = "ai_studio_preprocessing_sync_sections"

TIME_TARGETS = ("OutcomeTime", "DecisionTime")
FIELD_APPROVAL_EDITOR_COLUMNS = [
    "Approve",
    "Send To AI",
    "Column",
    "Data Type",
    "Unique Count",
    "Most occurring",
    "Values",
    "Field Tags",
]
EDITABLE_FIELD_COLUMNS = ("Approve", "Send To AI")
REASONING_EFFORT_OPTIONS = ("", "minimal", "low", "medium", "high", "xhigh")
VERBOSITY_OPTIONS = ("", "low", "medium", "high")


def render(ctx: ValueStreamContext) -> None:
    """Render the guided AI catalog studio."""
    _render_studio(
        ctx,
        ai_calls_enabled=True,
        title="AI Configuration Studio",
        subtitle=(
            "Prepare a source sample, approve fields, review AI-generated YAML, then "
            "apply or export it."
        ),
        status_label="Guided AI draft",
    )


def _render_studio(
    ctx: ValueStreamContext,
    *,
    ai_calls_enabled: bool,
    title: str,
    subtitle: str,
    status_label: str,
    include_header: bool = True,
) -> None:
    """Render the shared guided catalog studio workflow."""
    if include_header:
        components.render_page_header(
            title,
            subtitle,
            status="pending",
            status_label=status_label,
        )

    st.session_state[AI_CALLS_ENABLED_STATE_KEY] = ai_calls_enabled
    if ai_calls_enabled:
        _initialize_ai_settings(ctx.workspace)
    raw_sample = _load_sample(ctx.workspace, ai_calls_enabled=ai_calls_enabled)
    if raw_sample is None:
        if not ai_calls_enabled:
            st.info(
                "Upload a CSV, Parquet, JSON, NDJSON, gzip, or zip sample in the sidebar "
                "to build from data, or review the current catalog draft below."
            )
            _current_catalog_draft_editor(ctx)
            return
        st.info("Upload a CSV, Parquet, JSON, NDJSON, gzip, or zip sample in the sidebar to start.")
        return

    _initialize_state(raw_sample)
    _consume_preprocessing_editor_sync()
    _sync_ai_rename_capitalize_state(raw_sample)
    schema_sample = _schema_sample(raw_sample)
    _set_effective_schema_state(schema_sample)
    working, preprocessing_error = _working_sample(schema_sample)
    approved_fields = _approve_fields(working)
    _studio_status_bar(ctx, raw_sample, working, approved_fields, preprocessing_error)

    steps = _studio_steps(ai_calls_enabled=ai_calls_enabled)
    next_step = st.session_state.pop("ai_studio_next_step", None)
    next_step = _normalize_studio_step(next_step, steps)
    if next_step in steps:
        st.session_state["ai_studio_step"] = next_step
        current_step = next_step
    else:
        current_step = st.session_state.get("ai_studio_step", STEPS[0])
    current_step = _normalize_studio_step(current_step, steps)
    if current_step not in steps:
        current_step = steps[0]
        st.session_state["ai_studio_step"] = current_step
    current_phase = _phase_for_step(current_step, steps)
    st.session_state["ai_studio_phase"] = current_phase
    phase_statuses = _phase_statuses(approved_fields, preprocessing_error)
    st.segmented_control(
        "Studio Phase",
        [name for name, _ in STUDIO_PHASES],
        selection_mode="single",
        label_visibility="collapsed",
        key="ai_studio_phase",
        format_func=lambda name: _phase_label(name, phase_statuses.get(name, "empty")),
        help=config_help.field_help("editor.studio_phase"),
        on_change=_jump_to_phase_start,
        args=(steps,),
    )
    step_kwargs: dict[str, Any] = {
        "selection_mode": "single",
        "label_visibility": "collapsed",
        "key": "ai_studio_step",
    }
    if "ai_studio_step" not in st.session_state:
        step_kwargs["default"] = current_step
    step = st.segmented_control(
        "Studio Step",
        _phase_step_options(current_phase, steps),
        help=config_help.field_help("editor.studio_step"),
        **step_kwargs,
    )
    step = step or current_step

    if ai_calls_enabled:
        _render_ai_data_sharing_confirmation(approved_fields)
        content_col, copilot_col = st.columns([2.35, 1], gap="large")
        with copilot_col:
            _render_copilot_panel(step, working, approved_fields)
        with content_col:
            _render_selected_step(
                ctx,
                step,
                raw_sample,
                schema_sample,
                working,
                approved_fields,
                preprocessing_error,
                ai_calls_enabled=ai_calls_enabled,
            )
    else:
        _render_selected_step(
            ctx,
            step,
            raw_sample,
            schema_sample,
            working,
            approved_fields,
            preprocessing_error,
            ai_calls_enabled=ai_calls_enabled,
        )


def _studio_steps(*, ai_calls_enabled: bool) -> list[str]:
    return STEPS if ai_calls_enabled else DETERMINISTIC_STEPS


def _normalize_studio_step(step: object, steps: list[str]) -> str | None:
    if not isinstance(step, str):
        return None
    if step in steps:
        return step
    index = _studio_step_index(step)
    if index is None or index >= len(steps):
        return None
    return steps[index]


def _studio_step_index(step: str) -> int | None:
    for steps in (STEPS, DETERMINISTIC_STEPS):
        if step in steps:
            return steps.index(step)
    return None


def _phase_for_step(step: str, steps: list[str]) -> str:
    index = steps.index(step) if step in steps else 0
    for name, indexes in STUDIO_PHASES:
        if index in indexes:
            return name
    return STUDIO_PHASES[0][0]


def _phase_step_options(phase: str, steps: list[str]) -> list[str]:
    for name, indexes in STUDIO_PHASES:
        if name == phase:
            return [steps[index] for index in indexes if index < len(steps)]
    return list(steps)


def _jump_to_phase_start(steps: list[str]) -> None:
    phase = st.session_state.get("ai_studio_phase")
    for name, indexes in STUDIO_PHASES:
        if name == phase and indexes:
            st.session_state["ai_studio_step"] = steps[indexes[0]]
            return


def _phase_statuses(
    approved_fields: list[str], preprocessing_error: str | None = None
) -> dict[str, str]:
    draft = st.session_state.get("ai_studio_draft")
    pending = st.session_state.get("ai_studio_pending_draft") is not None
    statuses = {
        "Data": "complete" if approved_fields and not preprocessing_error else "attention",
        "Draft": "attention" if pending else ("complete" if draft is not None else "empty"),
        "Review": "empty",
        "Publish": "empty",
    }
    if isinstance(draft, dict):
        ok, _ = validate_draft_catalog(draft)
        statuses["Review"] = "attention" if pending or not ok else "complete"
        signature = _draft_signature(draft)
        if st.session_state.get("ai_studio_published_signature") == signature:
            statuses["Publish"] = "complete"
        elif ok:
            statuses["Publish"] = "attention"
    return statuses


def _phase_label(name: str, status: str) -> str:
    marker = {"complete": "✓", "attention": "!", "empty": "○"}.get(status, "○")
    return f"{marker} {name}"


def _render_selected_step(
    ctx: ValueStreamContext,
    step: str,
    raw_sample: pl.DataFrame,
    schema_sample: pl.DataFrame,
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
    *,
    ai_calls_enabled: bool = True,
) -> None:
    handlers = {
        0: lambda: _sample_step(ctx, raw_sample, schema_sample),
        1: lambda: _required_fields(schema_sample, working),
        2: lambda: _defaults(schema_sample),
        3: lambda: _filters(schema_sample),
        4: _calculations,
        5: lambda: _render_approve_fields_step(working),
        6: lambda: _ai_draft(
            ctx,
            working,
            approved_fields,
            preprocessing_error,
            ai_calls_enabled=ai_calls_enabled,
        ),
        7: lambda: _processors_review(working, approved_fields),
        8: lambda: _metrics_review(working, approved_fields),
        9: lambda: _ai_reports(
            working,
            approved_fields,
            preprocessing_error,
            ai_calls_enabled=ai_calls_enabled,
        ),
        10: lambda: _reports_review(working, approved_fields),
        11: _chat_review,
        12: _settings_review,
        13: lambda: _save_export(ctx, working, approved_fields, preprocessing_error),
    }
    index = _studio_step_index(step)
    handlers.get(index, handlers[0])()


def _current_catalog_draft_editor(ctx: ValueStreamContext) -> None:
    _initialize_current_catalog_draft(ctx)
    draft = st.session_state.get("ai_studio_draft")
    if not isinstance(draft, dict):
        draft = _draft_from_catalog(ctx)
        _set_draft(draft)
    approved_fields = _catalog_approved_fields(draft)
    working = _empty_working_frame(approved_fields)

    with components.bordered_panel(
        "Current Catalog Draft",
        "Edit the loaded workspace with the same non-AI review tools used after draft generation.",
    ):
        action_cols = st.columns([0.24, 0.58, 0.18], vertical_alignment="center")
        if action_cols[0].button("Reload Current Catalog Draft", icon=":material/refresh:"):
            _load_current_catalog_draft(ctx)
            st.rerun()
        action_cols[1].caption(
            "This draft starts from the active catalog. Changes are held in session state "
            "until the save action writes them to the workspace."
        )
        with action_cols[2]:
            _render_workspace_save_bar(ctx)
        _draft_counts(draft)
        ok, issues = validate_draft_catalog(draft)
        _render_draft_validation(ok, issues, expanded=not ok)

    step = (
        st.segmented_control(
            "Current Catalog Draft Step",
            CATALOG_DRAFT_STEPS,
            default=st.session_state.get("ai_studio_catalog_draft_step", CATALOG_DRAFT_STEPS[0]),
            selection_mode="single",
            label_visibility="collapsed",
            key="ai_studio_catalog_draft_step",
            help=config_help.field_help("editor.draft_step"),
        )
        or CATALOG_DRAFT_STEPS[0]
    )
    handlers = {
        CATALOG_DRAFT_STEPS[0]: lambda: _catalog_draft_overview(ctx, draft, approved_fields),
        CATALOG_DRAFT_STEPS[1]: lambda: _processors_review(working, approved_fields),
        CATALOG_DRAFT_STEPS[2]: lambda: _metrics_review(working, approved_fields),
        CATALOG_DRAFT_STEPS[3]: lambda: _ai_reports(
            working,
            approved_fields,
            None,
            ai_calls_enabled=False,
        ),
        CATALOG_DRAFT_STEPS[4]: lambda: _reports_review(working, approved_fields),
        CATALOG_DRAFT_STEPS[5]: _chat_review,
        CATALOG_DRAFT_STEPS[6]: _settings_review,
        CATALOG_DRAFT_STEPS[7]: lambda: _save_export(ctx, working, approved_fields, None),
    }
    handlers.get(step, handlers[CATALOG_DRAFT_STEPS[0]])()


def _initialize_current_catalog_draft(ctx: ValueStreamContext) -> None:
    if (
        st.session_state.get("ai_studio_draft_source") == CATALOG_DRAFT_SOURCE
        and st.session_state.get("ai_studio_catalog_draft_hash") == ctx.catalog_hash
        and isinstance(st.session_state.get("ai_studio_draft"), dict)
    ):
        return
    _load_current_catalog_draft(ctx)


def _load_current_catalog_draft(ctx: ValueStreamContext) -> None:
    _set_draft(_draft_from_catalog(ctx))
    st.session_state["ai_studio_draft_source"] = CATALOG_DRAFT_SOURCE
    st.session_state["ai_studio_catalog_draft_hash"] = ctx.catalog_hash


def _draft_from_catalog(ctx: ValueStreamContext) -> dict[str, Any]:
    return {
        "pipelines": ctx.catalog.pipelines.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        ),
        "processors": {
            "processors": [
                builder.processor_to_dict(processor)
                for processor in ctx.catalog.processors.processors
            ]
        },
        "metrics": {
            "metrics": {
                name: builder.metric_to_dict(metric)
                for name, metric in sorted(
                    ctx.catalog.metrics.metrics.items(),
                    key=lambda item: item[0].casefold(),
                )
            }
        },
        "dashboards": ctx.catalog.dashboards.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        ),
    }


def _catalog_draft_overview(
    ctx: ValueStreamContext,
    draft: dict[str, Any],
    approved_fields: list[str],
) -> None:
    components.metric_strip(
        [{"label": key, "value": value} for key, value in draft_object_counts(draft).items()]
    )
    components.render_validation_summary(ctx.validation.issues, ok=ctx.validation.ok)
    with components.bordered_panel(
        "Field Catalog",
        "Field options inferred from sources, processors, metrics, and report tiles.",
    ):
        st.caption(
            "These fields seed processor, metric, and report controls when no sample is uploaded."
        )
        st.dataframe(
            [{"Field": field} for field in approved_fields],
            hide_index=True,
            width="stretch",
            height=320,
        )
    with components.bordered_panel("Draft YAML", "Current in-session catalog draft."):
        for filename, section in _draft_files(draft).items():
            with st.expander(filename, expanded=filename in {"processors.yaml", "metrics.yaml"}):
                st.code(yaml.safe_dump(section, sort_keys=False), language="yaml")


def _catalog_approved_fields(draft: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for source in draft.get("pipelines", {}).get("sources", []) or []:
        if not isinstance(source, dict):
            continue
        schema = source.get("schema") if isinstance(source.get("schema"), dict) else {}
        fields.append(str(schema.get("timestamp_column", "") or ""))
        fields.extend(builder.string_list(schema.get("natural_key")))
        fields.extend(builder.string_list(schema.get("drop_columns")))
        defaults = source.get("defaults") if isinstance(source.get("defaults"), dict) else {}
        fields.extend(str(field) for field in defaults)
        for transform in source.get("transforms", []) or []:
            if not isinstance(transform, dict):
                continue
            fields.extend(_fields_from_catalog_transform(transform))
    for processor in _draft_processor_definitions(draft):
        fields.extend(_processor_dimensions(processor))
        time_def = processor.get("time") if isinstance(processor.get("time"), dict) else {}
        fields.append(str(time_def.get("column", "") or ""))
        fields.extend(_fields_from_catalog_processor(processor))
    for metric_name, metric_def in _draft_metric_definitions(draft).items():
        fields.append(metric_name)
        fields.extend(_metric_output_fields(metric_name, metric_def))
    for tile in _tile_inventory_rows(draft):
        fields.append(str(tile.get("Metric", "") or ""))
    return builder.dedupe([field for field in fields if field])


def _fields_from_catalog_transform(transform: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    fields.extend(builder.string_list(transform.get("columns")))
    fields.extend(builder.string_list(transform.get("outputs")))
    for key in ("from", "output"):
        value = str(transform.get(key, "") or "")
        if value:
            fields.append(value)
    values = transform.get("values")
    if isinstance(values, dict):
        fields.extend(str(field) for field in values)
    fields.extend(_fields_from_expression(transform.get("expression")))
    return fields


def _fields_from_catalog_processor(processor: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    entities = processor.get("entities") if isinstance(processor.get("entities"), dict) else {}
    fields.append(str(entities.get("subject", "") or ""))
    outcome = processor.get("outcome") if isinstance(processor.get("outcome"), dict) else {}
    fields.append(str(outcome.get("column", "") or processor.get("outcome_column", "") or ""))
    fields.append(str(processor.get("variant_column", "") or ""))
    fields.extend(builder.string_list(processor.get("properties")))
    fields.extend(builder.string_list(processor.get("score_properties")))
    score_columns = (
        processor.get("score_columns")
        if isinstance(processor.get("score_columns"), dict)
        else processor.get("scores")
        if isinstance(processor.get("scores"), dict)
        else {}
    )
    fields.extend(str(field) for field in score_columns.values() if field)
    states = processor.get("states") if isinstance(processor.get("states"), dict) else {}
    for spec in states.values():
        if isinstance(spec, dict):
            fields.append(str(spec.get("source_column", "") or ""))
    fields.extend(_fields_from_expression(processor.get("filter")))
    return fields


def _fields_from_expression(value: Any) -> list[str]:
    fields: list[str] = []
    if isinstance(value, dict):
        for key in ("col", "column"):
            field = str(value.get(key, "") or "")
            if field:
                fields.append(field)
        for child in value.values():
            fields.extend(_fields_from_expression(child))
    elif isinstance(value, list):
        for child in value:
            fields.extend(_fields_from_expression(child))
    return fields


def _empty_working_frame(fields: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict.fromkeys(fields, pl.Utf8))


def _load_sample(
    workspace: Path,
    *,
    ai_calls_enabled: bool = True,
) -> pl.DataFrame | None:
    active_workspace_sample = str(st.session_state.get("ai_studio_workspace_sample_active") or "")
    with st.sidebar:
        st.write("### Studio Controls" if ai_calls_enabled else "### Builder Studio Controls")
        upload = st.file_uploader(
            "Source sample",
            type=["csv", "parquet", "json", "ndjson", "zip", "gz", "gzip"],
            key="ai_studio_sample",
            help=config_help.field_help("ai.source_sample"),
        )
        st.number_input(
            "Preview Rows",
            min_value=100,
            max_value=100_000,
            value=10_000,
            step=500,
            key="ai_studio_sample_rows",
            help=config_help.field_help("ai.preview_rows"),
        )
        workspace_samples = _workspace_sample_files(workspace)
        if workspace_samples:
            selected_workspace_sample = st.selectbox(
                "Workspace sample",
                ["", *workspace_samples],
                format_func=lambda value: value or "Select a file",
                key="ai_studio_workspace_sample_choice",
                help=config_help.field_help("ai.workspace_sample"),
            )
            if st.button(
                "Use Workspace Sample",
                icon=":material/folder_open:",
                disabled=not selected_workspace_sample,
                key="ai_studio_use_workspace_sample",
            ):
                active_workspace_sample = selected_workspace_sample
                st.session_state["ai_studio_workspace_sample_active"] = active_workspace_sample
            if active_workspace_sample:
                st.caption(f"Using workspace sample `{active_workspace_sample}`.")
        if ai_calls_enabled:
            _ai_sidebar_controls()
    if upload is None and not active_workspace_sample:
        return None
    try:
        if upload is not None:
            sample_name = upload.name
            data = upload.getvalue()
        else:
            sample_path = (workspace / active_workspace_sample).resolve()
            data_root = (workspace / "data").resolve()
            if not sample_path.is_relative_to(data_root) or not sample_path.is_file():
                raise ValueError("workspace sample must be a file under the workspace data folder")
            sample_name = sample_path.name
            data = sample_path.read_bytes()
        st.session_state["ai_studio_sample_name"] = sample_name
        st.session_state["ai_studio_sample_identity"] = hashlib.sha256(data).hexdigest()
        limit = int(st.session_state.get("ai_studio_sample_rows", 10_000))
        frame = _read_sample_bytes(sample_name, data)
        return frame.head(limit)
    except Exception as exc:
        _log_ai_operation_failure("Sample read", exc)
        st.sidebar.error(f"Could not read sample: {exc}")
        return None


def _workspace_sample_files(workspace: Path) -> list[str]:
    data_root = workspace / "data"
    if not data_root.is_dir():
        return []
    accepted = {".csv", ".parquet", ".json", ".ndjson", ".zip", ".gz", ".gzip"}
    return [
        path.relative_to(workspace).as_posix()
        for path in sorted(data_root.rglob("*"))
        if path.is_file() and path.suffix.casefold() in accepted
    ]


def _ai_sidebar_controls() -> None:
    _ensure_ai_sidebar_state_defaults()
    with st.expander("AI Settings", expanded=False):
        st.text_input(
            "Model",
            key="ai_studio_ai_model",
            help=config_help.field_help("ai.model"),
        )
        st.text_input(
            "API Base",
            key="ai_studio_ai_api_base",
            placeholder="http://localhost:11434",
            help=config_help.field_help("ai.api_base"),
        )
        st.text_input(
            "Custom Provider",
            key="ai_studio_ai_provider",
            help=config_help.field_help("ai.custom_provider"),
        )
        api_key_env = str(st.session_state.get("ai_studio_api_key_env") or "").strip()
        st.text_input(
            "API Key",
            type="password",
            key="ai_studio_api_key",
            help=(
                f"{config_help.field_help('ai.api_key')}\n\nConfigured environment variable: "
                f"`{api_key_env}`."
                if api_key_env
                else config_help.field_help("ai.api_key")
            ),
        )
        if st.toggle(
            "Override Temperature",
            key="ai_studio_ai_temperature_enabled",
            help=config_help.field_help("ai.temperature_override"),
        ):
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="ai_studio_ai_temperature",
                help=config_help.field_help("ai.temperature"),
            )
        st.selectbox(
            "Reasoning Effort",
            REASONING_EFFORT_OPTIONS,
            format_func=_ai_option_label,
            key="ai_studio_ai_reasoning_effort",
            help=config_help.field_help("ai.reasoning_effort"),
        )
        st.selectbox(
            "Verbosity",
            VERBOSITY_OPTIONS,
            format_func=_ai_option_label,
            key="ai_studio_ai_verbosity",
            help=config_help.field_help("ai.verbosity"),
        )
        st.number_input(
            "Timeout Seconds",
            min_value=10,
            max_value=600,
            step=10,
            key="ai_studio_ai_timeout_seconds",
            help=config_help.field_help("ai.timeout"),
        )
        if st.session_state.get("ai_studio_ai_config_path"):
            st.caption(f"Loaded defaults from `{st.session_state['ai_studio_ai_config_path']}`.")
        st.caption(
            "Examples: `openai/gpt-4.1-mini`, `anthropic/claude-sonnet-4-5`, "
            "`ollama/llama3.1` with API Base `http://localhost:11434`, "
            "or a model served by a LiteLLM proxy."
        )


def _ensure_ai_sidebar_state_defaults() -> None:
    defaults: dict[str, Any] = {
        "ai_studio_ai_model": "openai/gpt-4.1-mini",
        "ai_studio_ai_api_base": "",
        "ai_studio_ai_provider": "",
        "ai_studio_ai_temperature_enabled": False,
        "ai_studio_ai_temperature": 1.0,
        "ai_studio_ai_reasoning_effort": "",
        "ai_studio_ai_verbosity": "",
        "ai_studio_ai_timeout_seconds": 90,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    api_key_env = str(st.session_state.get("ai_studio_api_key_env") or "").strip()
    configured_env_key = os.environ.get(api_key_env, "") if api_key_env else ""
    default_key = (
        configured_env_key
        or os.environ.get("LITELLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    st.session_state.setdefault("ai_studio_api_key", default_key)

    _normalize_ai_selectbox_state("ai_studio_ai_reasoning_effort", REASONING_EFFORT_OPTIONS)
    _normalize_ai_selectbox_state("ai_studio_ai_verbosity", VERBOSITY_OPTIONS)


def _normalize_ai_selectbox_state(key: str, options: tuple[str, ...]) -> None:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def _ai_option_label(value: str) -> str:
    return value if value else "Model default"


def _ai_calls_enabled() -> bool:
    return bool(st.session_state.get(AI_CALLS_ENABLED_STATE_KEY, True))


def _read_sample_bytes(file_name: str, data: bytes) -> pl.DataFrame:
    lower = file_name.lower()
    if lower.endswith(".csv"):
        return pl.read_csv(BytesIO(data), infer_schema_length=500)
    if lower.endswith(".parquet"):
        return pl.read_parquet(BytesIO(data))
    if lower.endswith((".json", ".ndjson")):
        return _read_json_payload(data)
    if lower.endswith((".gz", ".gzip")):
        return _read_json_payload(gzip.decompress(data))
    if lower.endswith(".zip"):
        rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(BytesIO(data)) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith((".json", ".ndjson")):
                    rows.extend(_json_records(zf.read(name)))
        return pl.from_dicts(rows) if rows else pl.DataFrame()
    return _read_json_payload(data)


def _read_json_payload(data: bytes) -> pl.DataFrame:
    records = _json_records(data)
    return pl.from_dicts(records) if records else pl.DataFrame()


def _initialize_ai_settings(workspace: Path) -> None:
    config_path, config = _load_ai_settings_config(workspace)
    signature = (
        str(config_path) if config_path else str(workspace.resolve()),
        config_path.stat().st_mtime_ns if config_path and config_path.exists() else 0,
    )
    if st.session_state.get("ai_studio_ai_config_signature") == signature:
        return
    st.session_state["ai_studio_ai_config_signature"] = signature
    st.session_state["ai_studio_ai_config_path"] = str(config_path) if config_path else ""
    if not config:
        return
    mappings = {
        "model": "ai_studio_ai_model",
        "api_base": "ai_studio_ai_api_base",
        "custom_provider": "ai_studio_ai_provider",
        "custom_llm_provider": "ai_studio_ai_provider",
        "api_key_env": "ai_studio_api_key_env",
        "temperature": "ai_studio_ai_temperature",
        "reasoning_effort": "ai_studio_ai_reasoning_effort",
        "verbosity": "ai_studio_ai_verbosity",
        "timeout_seconds": "ai_studio_ai_timeout_seconds",
    }
    for config_key, state_key in mappings.items():
        value = config.get(config_key)
        if value is not None:
            st.session_state[state_key] = value
    if config.get("temperature") is not None:
        st.session_state["ai_studio_ai_temperature_enabled"] = True


def _load_ai_settings_config(workspace: Path) -> tuple[Path | None, dict[str, Any]]:
    return load_llm_settings_config(workspace)


def _json_records(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        loaded = json.loads(text)
        return [row for row in loaded if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _initialize_state(sample: pl.DataFrame) -> None:
    signature = (
        str(st.session_state.get("ai_studio_sample_identity") or ""),
        tuple((name, str(dtype)) for name, dtype in sample.schema.items()),
    )
    if st.session_state.get("ai_studio_sample_signature") == signature:
        return
    st.session_state["ai_studio_sample_signature"] = signature
    # Business requirements describe intent, not the sample, so they survive sample changes.
    st.session_state.setdefault("ai_studio_user_goals", "")
    st.session_state["ai_studio_source_id"] = "ih"
    st.session_state["ai_studio_reader_kind"] = "pega_ds_export"
    st.session_state["ai_studio_file_pattern"] = "**/*.zip"
    st.session_state["ai_studio_group_pattern"] = r"\d{8}(?=\d{6}_)"
    st.session_state["ai_studio_streaming"] = True
    st.session_state["ai_studio_hive_partitioning"] = False
    st.session_state["ai_studio_timestamp_format"] = "%Y%m%dT%H%M%S%.3f %Z"
    st.session_state["ai_studio_subject"] = _default_subject_column(sample.columns)
    st.session_state["ai_studio_outcome_time"] = _default_time_column(sample.columns, "OutcomeTime")
    st.session_state["ai_studio_decision_time"] = _default_time_column(
        sample.columns, "DecisionTime", fallback=False
    )
    st.session_state["ai_studio_outcome_column"] = _default_column(sample.columns, "Outcome")
    st.session_state["ai_studio_day_column"] = _default_column(
        sample.columns, "Day", fallback=False
    )
    st.session_state["ai_studio_month_column"] = _default_column(
        sample.columns, "Month", fallback=False
    )
    st.session_state["ai_studio_year_column"] = _default_column(
        sample.columns, "Year", fallback=False
    )
    st.session_state["ai_studio_quarter_column"] = _default_column(
        sample.columns, "Quarter", fallback=False
    )
    st.session_state["ai_studio_defaults"] = [builder.blank_default_row()]
    st.session_state["ai_studio_filter_rows"] = [builder.blank_filter_row()]
    st.session_state["ai_studio_filter_mode"] = "Rules"
    st.session_state["ai_studio_raw_filter"] = ""
    st.session_state["ai_studio_calculations"] = [_blank_ai_calculation_row()]
    st.session_state["ai_studio_rename_capitalize"] = False
    st.session_state["ai_studio_rename_capitalize_applied"] = False
    st.session_state["ai_studio_approved_fields"] = []
    st.session_state["ai_studio_example_fields"] = []
    st.session_state["ai_studio_group_by_fields"] = []
    st.session_state["ai_studio_field_approval_initialized"] = False
    _clear_ai_sharing_confirmation()
    _clear_schema_widget_state()
    st.session_state["ai_studio_draft"] = None
    st.session_state["ai_studio_pending_draft"] = None
    st.session_state["ai_studio_pending_base_draft"] = None
    st.session_state["ai_studio_pending_kind"] = ""
    st.session_state["ai_studio_pending_prompt"] = ""
    st.session_state["ai_studio_last_ai_response"] = ""
    st.session_state["ai_studio_raw_metrics_yaml"] = ""
    st.session_state["ai_studio_raw_dashboards_yaml"] = ""
    st.session_state["ai_studio_copilot_history"] = []
    st.session_state["ai_studio_copilot_questions"] = []
    st.session_state["ai_studio_copilot_last_prompt"] = ""
    st.session_state.pop("ai_studio_copilot_queued_message", None)
    st.session_state["ai_studio_coverage_rows"] = []
    st.session_state["ai_studio_coverage_signature"] = ""
    st.session_state["ai_studio_published_signature"] = ""


def _sample_step(
    ctx: ValueStreamContext,
    raw_sample: pl.DataFrame,
    schema_sample: pl.DataFrame,
) -> None:
    if _ai_calls_enabled():
        with components.bordered_panel(
            "Business Requirements",
            "Describe what you want to measure; the AI draft and report steps use this "
            "alongside the approved schema.",
        ):
            _render_user_goals_editor()
    with components.bordered_panel("Runtime Settings"):
        (
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
        ) = st.columns([1, 1, 1, 1, 0.5, 0.5], gap="xsmall")
        st.session_state["ai_studio_source_id"] = c1.text_input(
            "Source ID",
            value=st.session_state["ai_studio_source_id"],
            help=config_help.field_help("source.id"),
        )
        st.session_state["ai_studio_reader_kind"] = c2.selectbox(
            "Reader",
            ["pega_ds_export", "parquet", "csv", "xlsx"],
            index=["pega_ds_export", "parquet", "csv", "xlsx"].index(
                st.session_state["ai_studio_reader_kind"]
            ),
            help=config_help.field_help("source.reader"),
        )
        st.session_state["ai_studio_file_pattern"] = c3.text_input(
            "File Pattern",
            value=st.session_state["ai_studio_file_pattern"],
            help=config_help.field_help("source.file_pattern"),
        )
        st.session_state["ai_studio_group_pattern"] = c4.text_input(
            "Group Pattern",
            value=st.session_state["ai_studio_group_pattern"],
            help=config_help.field_help("source.group_pattern"),
        )
        c5.markdown("Streaming", help=config_help.field_help("source.streaming"))
        st.session_state["ai_studio_streaming"] = c5.toggle(
            "Streaming",
            value=st.session_state["ai_studio_streaming"],
            help=config_help.field_help("source.streaming"),
            label_visibility="collapsed",
        )
        c6.markdown("Hive-style", help=config_help.field_help("source.hive_partitioning"))
        st.session_state["ai_studio_hive_partitioning"] = c6.toggle(
            "Hive Partitioned",
            value=st.session_state["ai_studio_hive_partitioning"],
            help=config_help.field_help("source.hive_partitioning"),
            label_visibility="collapsed",
        )
        tmstmp, _others = st.columns([2, 3], gap="xsmall")
        st.session_state["ai_studio_timestamp_format"] = tmstmp.text_input(
            "Timestamp Format",
            value=st.session_state["ai_studio_timestamp_format"],
            help=config_help.field_help("source.timestamp_format"),
        )
        st.session_state["ai_studio_rename_capitalize"] = st.toggle(
            "Use Rename / Capitalize Transform",
            key="ai_studio_rename_capitalize_transform",
            help=config_help.field_help("source.rename_capitalize"),
        )
        if st.session_state["ai_studio_rename_capitalize"]:
            st.caption(
                "`rename_capitalize` converts source columns to the legacy Pega-aware "
                "capitalized schema, for example `pyName` to `Name`."
            )

        st.code(
            yaml.safe_dump(
                {
                    "transforms": (
                        [{"kind": "rename_capitalize"}]
                        if st.session_state["ai_studio_rename_capitalize"]
                        else []
                    )
                },
                sort_keys=False,
            ),
            language="yaml",
        )
    _set_effective_schema_state(raw_sample)
    _sample_preview(raw_sample, schema_sample)
    with components.bordered_panel(
        "Current Workspace Sources", "Existing catalog sources are shown for context."
    ):
        rows = [
            {
                "source": source.id,
                "reader": source.reader.kind,
                "pattern": source.reader.file_pattern,
                "processors": len(processors),
            }
            for source in ctx.catalog.pipelines.sources
            for processors in [
                [p for p in ctx.catalog.processors.processors if p.source == source.id]
            ]
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


def _required_fields(sample: pl.DataFrame, working: pl.DataFrame) -> None:
    with components.bordered_panel(
        "Required Field Mapping", "Enter the source fields that identify rows, outcomes, and time."
    ):
        c1, c2, c3, c4 = st.columns(4)
        st.session_state["ai_studio_subject"] = c1.text_input(
            "Subject ID Field",
            value=str(st.session_state.get("ai_studio_subject", "")),
            placeholder="CustomerID",
            help=config_help.field_help("mapping.subject"),
        ).strip()
        st.session_state["ai_studio_outcome_column"] = c2.text_input(
            "Outcome Field",
            value=str(st.session_state.get("ai_studio_outcome_column", "")),
            placeholder="Outcome",
            help=config_help.field_help("mapping.outcome"),
        ).strip()
        st.session_state["ai_studio_outcome_time"] = c3.text_input(
            "Outcome Timestamp",
            value=str(st.session_state.get("ai_studio_outcome_time", "")),
            placeholder="EventTime",
            help=config_help.field_help("mapping.outcome_time"),
        ).strip()
        st.session_state["ai_studio_decision_time"] = c4.text_input(
            "Decision Timestamp",
            value=str(st.session_state.get("ai_studio_decision_time", "")),
            placeholder="DecisionTime",
            help=config_help.field_help("mapping.decision_time"),
        ).strip()
        st.caption("Date fields (if available)")
        c5, c6, c7, c8 = st.columns(4)
        st.session_state["ai_studio_day_column"] = c5.text_input(
            "Day",
            value=str(st.session_state.get("ai_studio_day_column", "")),
            placeholder="Day",
            help=config_help.field_help("mapping.day"),
        ).strip()
        st.session_state["ai_studio_month_column"] = c6.text_input(
            "Month",
            value=str(st.session_state.get("ai_studio_month_column", "")),
            placeholder="Month",
            help=config_help.field_help("mapping.month"),
        ).strip()
        st.session_state["ai_studio_quarter_column"] = c7.text_input(
            "Quarter",
            value=str(st.session_state.get("ai_studio_quarter_column", "")),
            placeholder="Quarter",
            help=config_help.field_help("mapping.quarter"),
        ).strip()
        st.session_state["ai_studio_year_column"] = c8.text_input(
            "Year",
            value=str(st.session_state.get("ai_studio_year_column", "")),
            placeholder="Year",
            help=config_help.field_help("mapping.year"),
        ).strip()
        st.dataframe(
            _field_mapping_rows(sample),
            hide_index=True,
            width="stretch",
            height=280,
        )
    preview_cols = [
        column
        for column in [
            "SubjectID",
            "OutcomeTime",
            "DecisionTime",
            "Day",
            "Month",
            "Year",
            "Quarter",
        ]
        if column in working.columns
    ]
    if preview_cols:
        with components.bordered_panel(
            "Derived Field Preview", "Required aliases and calendar fields after preprocessing."
        ):
            st.dataframe(working.select(preview_cols).head(50), hide_index=True, width="stretch")


def _defaults(sample: pl.DataFrame) -> None:
    _render_defaults_editor(sample)


@st.fragment()
def _render_defaults_editor(sample: pl.DataFrame) -> None:
    with components.bordered_panel("Default Values", "Defaults are applied before filters."):
        editor_key = _schema_widget_key("ai_studio_defaults_editor")
        picker_key = _schema_widget_key("ai_studio_defaults_field_picker")
        picker_col, action_col = st.columns([0.78, 0.22], vertical_alignment="bottom")
        selected_fields = picker_col.multiselect(
            "Add Field",
            [str(column) for column in sample.columns],
            accept_new_options=True,
            key=picker_key,
            placeholder="Select existing or type new",
            help=config_help.field_help("default.field"),
        )
        action_col.button(
            "Add",
            icon=":material/add:",
            disabled=not selected_fields,
            key=f"{picker_key}_add",
            on_click=components.add_default_fields_from_picker,
            args=("ai_studio_defaults", picker_key, editor_key),
        )
        default_frame = builder.editor_frame(
            st.session_state.get("ai_studio_defaults", []),
            ["Field", "Default Value", "Enabled"],
            builder.blank_default_row,
        )
        edited = st.data_editor(
            default_frame,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=editor_key,
            column_config={
                "Field": st.column_config.TextColumn(
                    "Field", width="medium", help=config_help.field_help("default.field")
                ),
                "Default Value": st.column_config.TextColumn(
                    "Default Value",
                    width="medium",
                    help=config_help.field_help("default.value"),
                ),
                "Enabled": st.column_config.CheckboxColumn(
                    "Enabled", width="small", help=config_help.field_help("row.enabled")
                ),
            },
        )
        st.session_state["ai_studio_defaults"] = builder.normalize_editor_rows(edited)
        missing = [
            name
            for name in ("SubjectID", "OutcomeTime", "Day")
            if name not in sample.columns
            and name not in builder.build_default_values(st.session_state["ai_studio_defaults"])
        ]
        if missing:
            st.caption(
                "Mapped or derived aliases can provide missing fields: " + ", ".join(missing)
            )


def _filters(sample: pl.DataFrame) -> None:
    filter_frame = builder.editor_frame(
        st.session_state.get("ai_studio_filter_rows", []),
        ["Field", "Operator", "Value", "Enabled"],
        builder.blank_filter_row,
    )
    _render_filters_editor(sample, filter_frame)


@st.fragment()
def _render_filters_editor(sample: pl.DataFrame, filter_frame: pl.DataFrame) -> None:
    with components.bordered_panel(
        "Filters", "Define dataset-level filters before calculated fields."
    ):
        st.session_state["ai_studio_filter_mode"] = st.segmented_control(
            "Filter Mode",
            ["Rules", "Raw AST"],
            default=st.session_state["ai_studio_filter_mode"],
            help=config_help.field_help("source.filter_mode"),
        )
        if st.session_state["ai_studio_filter_mode"] == "Rules":
            edited = st.data_editor(
                filter_frame,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key=_schema_widget_key("ai_studio_filter_editor"),
                column_config={
                    "Field": st.column_config.SelectboxColumn(
                        "Field",
                        options=sample.columns,
                        required=False,
                        help=config_help.field_help("filter.field"),
                    ),
                    "Operator": st.column_config.SelectboxColumn(
                        "Operator",
                        options=builder.FILTER_OPERATORS,
                        help=config_help.field_help("filter.operator"),
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value", help=config_help.field_help("filter.value")
                    ),
                    "Enabled": st.column_config.CheckboxColumn(
                        "Enabled", help=config_help.field_help("row.enabled")
                    ),
                },
            )
            st.session_state["ai_studio_filter_rows"] = builder.normalize_editor_rows(edited)
            compiled = builder.compile_filter_rows(st.session_state["ai_studio_filter_rows"])
            st.caption("Compiled AST")
            st.code(builder.expression_yaml(compiled) or "{}", language="yaml")
        else:
            st.session_state["ai_studio_raw_filter"] = st.text_area(
                "Filter AST YAML",
                value=st.session_state["ai_studio_raw_filter"],
                height=220,
                placeholder="op: in\ncolumn: Outcome\nvalues: [Impression, Clicked, Pending, Conversion]",
                help=config_help.field_help("source.filter_ast"),
            )


def _calculations() -> None:
    calculation_frame = builder.editor_frame(
        st.session_state.get("ai_studio_calculations", []),
        ["Name", "Mode", "Left", "Right Kind", "Right", "Expression", "Enabled"],
        _blank_ai_calculation_row,
    )
    _render_calculations_editor(calculation_frame)


@st.fragment()
def _render_calculations_editor(
    calculation_frame: pl.DataFrame,
) -> None:
    with components.bordered_panel("Calculated Fields", "Create named `derive_column` transforms."):
        with st.popover("Examples", icon=":material/flare:"):
            st.code(
                "Name: Margin\nMode: Subtract\nLeft: Revenue\nRight Kind: Field\nRight: Cost",
                language="yaml",
            )
            st.code(
                "op: date_diff\nunit: seconds\nend: {col: OutcomeTime}\nstart: {col: DecisionTime}",
                language="yaml",
            )
            st.code(
                'pl.col("Revenue") - pl.col("Cost")',
                language="python",
            )
        edited = st.data_editor(
            calculation_frame,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=_schema_widget_key("ai_studio_calculation_editor"),
            column_config={
                "Name": st.column_config.TextColumn(
                    "Name", width="small", help=config_help.field_help("calculation.name")
                ),
                "Mode": st.column_config.SelectboxColumn(
                    "Mode",
                    options=builder.CALCULATION_MODES,
                    width="medium",
                    help=config_help.field_help("calculation.mode"),
                ),
                "Left": st.column_config.TextColumn(
                    "Left", width="medium", help=config_help.field_help("calculation.left")
                ),
                "Right Kind": st.column_config.SelectboxColumn(
                    "Right Kind",
                    options=["Field", "Literal"],
                    width="small",
                    help=config_help.field_help("calculation.right_kind"),
                ),
                "Right": st.column_config.TextColumn(
                    "Right", width="medium", help=config_help.field_help("calculation.right")
                ),
                "Expression": st.column_config.TextColumn(
                    "Expression",
                    width="large",
                    help=config_help.field_help("calculation.expression"),
                ),
                "Enabled": st.column_config.CheckboxColumn(
                    "Enabled", width="small", help=config_help.field_help("row.enabled")
                ),
            },
        )
        st.session_state["ai_studio_calculations"] = builder.normalize_editor_rows(edited)
        try:
            transforms = builder.build_derive_column_transforms(
                st.session_state["ai_studio_calculations"]
            )
            st.caption("Generated calculated transforms")
            st.code(yaml.safe_dump({"transforms": transforms}, sort_keys=False), language="yaml")
        except Exception as exc:
            _log_ai_operation_failure("Calculated transform preview", exc)
            st.error(str(exc))


def _approve_fields(sample: pl.DataFrame) -> list[str]:
    available_fields = sorted(sample.columns, key=str.casefold)
    required = [field for field in _studio_required_fields(sample) if field in available_fields]
    approved_fields, _, _ = _sync_field_approval_state(
        sample,
        available_fields,
        required,
    )
    return approved_fields


@st.fragment()
def _render_approve_fields_step(sample: pl.DataFrame) -> None:
    available_fields = sorted(sample.columns, key=str.casefold)
    required = [field for field in _studio_required_fields(sample) if field in available_fields]
    approved_fields, example_fields, group_by = _sync_field_approval_state(
        sample,
        available_fields,
        required,
    )
    previous_approved_fields = list(approved_fields)
    previous_example_fields = list(example_fields)
    with components.bordered_panel(
        "Approve Fields And Data Sharing With AI",
        "Approve exposes a field to the AI stage; Share Sample Values additionally "
        "sends example values from that field in AI prompts.",
    ):
        if required:
            st.caption("Required IH fields are always included: " + ", ".join(required))
        st.caption(
            "This is the post-processed schema. Derived time fields are already visible here."
        )
        query = st.text_input(
            "Filter fields",
            key=_schema_widget_key("ai_studio_field_approval_search"),
            placeholder="Filter fields",
            icon=":material/search:",
            label_visibility="collapsed",
            help=config_help.field_help("ai.field_search"),
        )
        normalized_query = query.strip().casefold()
        visible_fields = (
            [field for field in available_fields if normalized_query in field.casefold()]
            if normalized_query
            else available_fields
        )
        editor_frame = _field_approval_editor_frame(
            sample,
            visible_fields,
            required_fields=required,
            approved_fields=approved_fields,
            example_fields=example_fields,
        )
        edited = st.data_editor(
            editor_frame,
            width="stretch",
            hide_index=True,
            height=520,
            # The key scopes stored edits to one filter view; the checkbox
            # state itself lives in the approval session keys below.
            key=_schema_widget_key(f"ai_studio_field_approval_editor_{normalized_query}"),
            disabled=[
                column for column in editor_frame.columns if column not in EDITABLE_FIELD_COLUMNS
            ],
            column_config={
                "Approve": st.column_config.CheckboxColumn(
                    "Approve",
                    help=config_help.field_help("ai.field_approve"),
                    width="small",
                ),
                "Send To AI": st.column_config.CheckboxColumn(
                    "Share Sample Values",
                    help=config_help.field_help("ai.field_send_values"),
                    width="small",
                ),
                "Column": st.column_config.TextColumn(
                    "Column", width="medium", help=config_help.field_help("ai.field_name")
                ),
                "Data Type": st.column_config.TextColumn(
                    "Data Type", width="small", help=config_help.field_help("ai.field_type")
                ),
                "Unique Count": st.column_config.NumberColumn(
                    "Unique Count",
                    width="small",
                    help=config_help.field_help("ai.field_unique_count"),
                ),
                "Most occurring": st.column_config.TextColumn(
                    "Most occurring",
                    width="medium",
                    help=config_help.field_help("ai.field_mode"),
                ),
                "Values": st.column_config.TextColumn(
                    "Values",
                    help=config_help.field_help("ai.field_values"),
                    width="large",
                ),
                "Field Tags": st.column_config.TextColumn(
                    "Field Tags", width="large", help=config_help.field_help("ai.field_tags")
                ),
            },
        )
        selected, example_fields = _apply_field_approval_edits(
            builder.normalize_editor_rows(edited),
            available_fields=available_fields,
            required_fields=required,
            approved_fields=approved_fields,
            example_fields=example_fields,
        )
        selected, example_fields, group_by = _normalize_field_approval_state(
            available_fields=available_fields,
            required_fields=required,
            approved_fields=selected,
            example_fields=example_fields,
            group_by_fields=group_by,
        )
        st.session_state["ai_studio_approved_fields"] = selected
        st.session_state["ai_studio_example_fields"] = example_fields
        st.session_state["ai_studio_group_by_fields"] = group_by
        _invalidate_ai_sharing_confirmation_if_scope_changed(
            previous_approved_fields=previous_approved_fields,
            previous_example_fields=previous_example_fields,
            approved_fields=selected,
            example_fields=example_fields,
        )

    with components.bordered_panel(
        "Suggested Group-By Fields",
        "Choose approved fields that should become processor dimensions.",
    ):
        group_key = _schema_widget_key("ai_studio_group_by_field_selector")
        if group_key not in st.session_state:
            st.session_state[group_key] = [field for field in group_by if field in selected]
        else:
            st.session_state[group_key] = [
                field for field in st.session_state[group_key] if field in selected
            ]
        group_by = _render_dimension_profile_panel(
            sample,
            selected_fields=selected,
            required_fields=required,
            group_by_fields=[field for field in st.session_state[group_key] if field in selected],
            group_key=group_key,
        )
        st.session_state[group_key] = group_by
        group_by = st.multiselect(
            "Group-By Fields",
            selected,
            help=config_help.field_help("processor.group_by"),
            key=group_key,
        )
        group_by = _ordered_fields(selected, set(group_by))
        st.session_state["ai_studio_group_by_fields"] = group_by

    current_example_fields = [
        field for field in st.session_state.get("ai_studio_example_fields", []) if field in selected
    ]
    _privacy_summary(sample, selected, current_example_fields, group_by)
    with st.expander("Preview working sample after preprocessing", expanded=False):
        preview_fields = [field for field in selected if field in sample.columns]
        if preview_fields:
            st.dataframe(
                sample.select(preview_fields).head(100),
                hide_index=True,
                width="stretch",
                height=320,
            )
        else:
            st.info("No fields are currently approved.")


def _render_dimension_profile_panel(
    sample: pl.DataFrame,
    *,
    selected_fields: list[str],
    required_fields: list[str],
    group_by_fields: list[str],
    group_key: str,
) -> list[str]:
    profile_fields = [field for field in selected_fields if field in sample.columns]
    if not profile_fields:
        st.info("Approve fields before profiling aggregate dimensions.")
        return []
    profile_sample = sample.select(profile_fields)
    rows = dimension_profile.selection_dimension_profile_rows(
        profile_sample,
        selected_fields=group_by_fields,
        required_fields=required_fields,
    )
    if not rows:
        st.info("No approved fields found in the working sample.")
        return []

    profile_frame = dimension_profile.profile_frame(rows).rename(
        {"Current Usage": "Current Selection"}
    )
    components.metric_cards(
        [
            {"label": "Profiled fields", "value": len(rows)},
            {
                "label": "Recommended",
                "value": sum(row.recommendation == "Recommended" for row in rows),
            },
            {
                "label": "Needs review",
                "value": sum(row.recommendation == "Review" for row in rows),
            },
            {
                "label": "Already active",
                "value": sum(row.recommendation == "Active" for row in rows),
            },
        ],
        columns=4,
    )
    filter_choice = st.segmented_control(
        "Profile Filter",
        ["All", "Recommended", "Review", "Avoid", "Active"],
        default="All",
        key=_schema_widget_key("ai_studio_dimension_profile_filter"),
        help=config_help.field_help("dimension.profile_filter"),
    )
    filtered = profile_frame
    if filter_choice and filter_choice != "All":
        filtered = filtered.filter(pl.col("Recommendation") == filter_choice)
    st.dataframe(filtered, hide_index=True, width="stretch", height=360)

    recommended = dimension_profile.recommended_fields(
        rows,
        allowed_fields=selected_fields,
        existing_fields=group_by_fields,
    )
    if recommended:
        st.caption(
            "Recommended fields are approved, low-cardinality, non-identity fields in the working sample."
        )
        if st.button(
            "Add recommended to selection",
            icon=":material/add:",
            key=_schema_widget_key("ai_studio_dimension_add_recommended"),
        ):
            current = [
                str(value)
                for value in st.session_state.get(group_key, group_by_fields)
                if str(value)
            ]
            updated = _ordered_fields(selected_fields, {*current, *recommended})
            st.session_state[group_key] = updated
            st.session_state["ai_studio_group_by_fields"] = updated
            st.rerun()

    pack_fields = dimension_profile.dimension_pack_fields(selected_fields)
    pack_candidates = [field for field in pack_fields if field not in group_by_fields]
    if pack_candidates and st.button(
        "Add Pega/CDH core dimensions",
        icon=":material/library_add:",
        key=_schema_widget_key("ai_studio_dimension_add_pack"),
    ):
        current = [
            str(value) for value in st.session_state.get(group_key, group_by_fields) if str(value)
        ]
        updated = _ordered_fields(selected_fields, {*current, *pack_candidates})
        st.session_state[group_key] = updated
        st.session_state["ai_studio_group_by_fields"] = updated
        st.rerun()

    return _ordered_fields(selected_fields, set(group_by_fields))


@st.fragment()
def _ai_draft(  # noqa: PLR0912, PLR0915
    ctx: ValueStreamContext,
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
    *,
    ai_calls_enabled: bool = True,
) -> None:
    if preprocessing_error:
        st.warning("Resolve preprocessing errors before generating a draft.")
        st.code(preprocessing_error, language="text")
        return
    baseline = _build_draft_catalog(working, approved_fields)
    st.write("### AI Configuration Draft" if ai_calls_enabled else "### Configuration Draft")
    if ai_calls_enabled:
        st.caption(
            "The AI receives only the approved working schema and the deterministic source draft. "
            "Generated sections are held for review before they can update the draft."
        )
    else:
        st.caption(
            "The draft is built locally from the approved working schema. Review and edit it "
            "before applying it to the workspace."
        )
    user_goals = ""
    if ai_calls_enabled:
        with components.bordered_panel(
            "Business Requirements",
            "Free-form requirements sent with the approved schema when generating the draft.",
        ):
            user_goals = _render_user_goals_editor()
    schema_preview = _schema_preview_for_ai(working, approved_fields)
    hidden_fields = sorted(set(working.columns) - set(approved_fields), key=str.casefold)
    prompt = ""
    if ai_calls_enabled:
        prompt = prompt_for_config_draft(
            file_name=_sample_file_name(),
            approved_schema=schema_preview,
            approved_fields=approved_fields,
            hidden_fields=hidden_fields,
            baseline_draft=baseline,
            user_goals=user_goals,
        )

    if ai_calls_enabled:
        _ai_privacy_summary(working, approved_fields, prompt)
    else:
        _privacy_summary(
            working,
            approved_fields,
            [
                field
                for field in st.session_state.get("ai_studio_example_fields", [])
                if field in approved_fields
            ],
            [
                field
                for field in st.session_state.get("ai_studio_group_by_fields", [])
                if field in approved_fields
            ],
        )
    overview_cols = st.columns(2)
    with overview_cols[0], st.container(border=True):
        st.write("#### Approved Schema")
        st.dataframe(
            _schema_preview_display_rows(schema_preview),
            hide_index=True,
            width="stretch",
            height=360,
        )
    with overview_cols[1], st.container(border=True):
        st.write("#### Deterministic Baseline")
        _draft_counts(baseline)
        with st.expander("Baseline YAML", expanded=False):
            for filename, section in _draft_files(baseline).items():
                st.caption(filename)
                st.code(yaml.safe_dump(section, sort_keys=False), language="yaml")

    if ai_calls_enabled and st.session_state.get("ai_studio_pending_draft") is not None:
        return

    ai_settings = _current_ai_settings() if ai_calls_enabled else None
    sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
    action_col1, action_col2, action_col3 = st.columns(
        [0.28, 0.28, 0.44] if ai_calls_enabled else [0.32, 0.32, 0.36],
        vertical_alignment="center",
    )
    if action_col1.button("Use Deterministic Draft", type="secondary"):
        _set_draft(baseline)
        st.success("Deterministic draft accepted for review.")
    if ai_calls_enabled and action_col2.button(
        "Generate AI Draft",
        type="primary",
        disabled=ai_settings is None or not sharing_confirmed,
        help=(
            "Configure a LiteLLM model in the sidebar to enable AI generation."
            if ai_settings is None
            else "Confirm the AI data-sharing scope above to enable AI generation."
            if not sharing_confirmed
            else "Generate a governed draft with the confirmed sharing scope."
        ),
    ):
        try:
            with st.status("Generating AI draft", expanded=True) as status:
                status.write("Sending approved schema and baseline catalog to the model...")
                response = _call_litellm_for_current_sample(
                    ai_settings,
                    prompt,
                    approved_fields=approved_fields,
                )
                sections = parse_ai_yaml_sections(response)
                pending = merge_draft_sections(baseline, sections)
                st.session_state["ai_studio_pending_draft"] = pending
                st.session_state["ai_studio_pending_base_draft"] = baseline
                st.session_state["ai_studio_pending_kind"] = "draft"
                st.session_state["ai_studio_pending_prompt"] = prompt
                st.session_state["ai_studio_last_ai_response"] = response
                status.write("AI response parsed as catalog YAML.")
                status.update(label="Draft ready for review", state="complete")
            st.rerun()
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _log_ai_operation_failure("AI draft generation", exc)
            st.error(f"AI draft generation failed: {exc}")
    if ai_calls_enabled:
        with action_col3, st.popover("Show Prompt", icon=":material/psychology:"):
            st.code(prompt, language="text")
    else:
        action_col2.info("AI generation is disabled here; use AI Config Studio for model drafts.")

    current = st.session_state.get("ai_studio_draft")
    if current:
        st.success(f"Draft accepted for review. Current workspace hash: `{ctx.catalog_hash}`.")
        _draft_counts(current)
    elif ai_calls_enabled and not ai_settings:
        st.info("Configure a LiteLLM model in the sidebar or use the deterministic draft.")


def _save_export(
    ctx: ValueStreamContext,
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
) -> None:
    components.render_validation_summary(ctx.validation.issues, ok=ctx.validation.ok)
    if preprocessing_error:
        st.warning("Resolve preprocessing errors before applying a draft.")
        return
    draft = _current_or_deterministic_draft(working, approved_fields)
    st.write("### Save & Export")
    st.caption(
        "Export the reviewed draft here. Use the consistent save action at the top "
        "of the editor to write it to the workspace."
    )
    _draft_counts(draft)
    ok, issues = validate_draft_catalog(draft)
    _render_draft_validation(ok, issues, expanded=not ok)
    _render_ai_repair_panel(draft, working, approved_fields, issues)
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    _render_coverage_panel(draft, working, approved_fields)
    files = _draft_files(draft)
    for filename, section in files.items():
        text = yaml.safe_dump(section, sort_keys=False)
        with st.expander(filename, expanded=filename in {"pipelines.yaml", "dashboards.yaml"}):
            st.code(text, language="yaml")
            st.download_button(
                f"Download {filename}",
                data=text,
                file_name=filename,
                mime="text/yaml",
                key=f"ai_studio_download_{filename}",
                disabled=not ok,
            )
    if st.button(
        "Save Draft & Run Source",
        icon=":material/play_arrow:",
        disabled=not ok,
        help="Write the draft, validate the workspace catalog, then run generated sources so aggregates are materialized.",
    ):
        try:
            _apply_draft_and_run_sources(ctx, draft)
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _log_ai_operation_failure("AI draft apply and source run", exc)
            st.error(str(exc))


def _render_workspace_save_bar(ctx: ValueStreamContext) -> None:
    """Publish the accepted AI draft from one consistent top-of-editor action."""

    feedback = st.session_state.pop("ai_studio_workspace_save_feedback", None)
    draft = st.session_state.get("ai_studio_draft")
    pending = st.session_state.get("ai_studio_pending_draft") is not None
    ok = False
    issues: list[str] = []
    if isinstance(draft, dict):
        ok, issues = validate_draft_catalog(draft)

    published = bool(
        isinstance(draft, dict)
        and st.session_state.get("ai_studio_published_signature") == _draft_signature(draft)
    )
    if not isinstance(draft, dict):
        caption = "Generate and accept a draft before saving it to the workspace."
    elif pending:
        caption = "Review or reject the pending AI changes before saving the accepted draft."
    elif not ok:
        caption = f"Resolve {len(issues)} draft validation issue(s) before saving."
    elif published:
        caption = "The accepted draft is already saved to the active workspace."
    else:
        caption = "Save the accepted in-session draft to the active workspace catalog."

    if components.editor_save_bar(
        key="ai_studio_workspace_save",
        caption=caption,
        label="Saved" if published else "Save draft",
        disabled=not isinstance(draft, dict) or pending or not ok or published,
        help=(
            "This writes the accepted draft only. Use the update buttons inside a review "
            "step to accept its current controls into the draft first."
        ),
    ):
        try:
            _apply_draft(ctx, draft)
            _mark_draft_published(draft)
            workspace_ok, workspace_issues = builder.validate_workspace(ctx.workspace)
            st.session_state["ai_studio_workspace_save_feedback"] = {
                "ok": workspace_ok,
                "issues": workspace_issues,
            }
            st.rerun()
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _log_ai_operation_failure("AI draft workspace save", exc)
            st.toast(f"Draft could not be saved: {exc}", icon=":material/error:")
    if isinstance(feedback, dict):
        if feedback.get("ok"):
            st.toast(
                "Draft saved to the workspace and the catalog validates.",
                icon=":material/check_circle:",
            )
        else:
            st.toast(
                "Draft saved, but the workspace catalog needs attention.",
                icon=":material/warning:",
            )


def _processors_review(working: pl.DataFrame, approved_fields: list[str]) -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Processors Review")
    st.caption("Review generated processor definitions before editing dependent metrics.")
    _draft_counts(draft)
    ok, issues = validate_draft_catalog(draft)
    _render_draft_validation(ok, issues, expanded=False)
    _render_ai_repair_panel(draft, working, approved_fields, issues)
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    _render_ai_refine_panel(draft, working, approved_fields)

    processors = _draft_processor_definitions(draft)
    processor_ids = [
        str(processor.get("id", "")) for processor in processors if processor.get("id")
    ]
    if not processor_ids:
        st.warning("The draft does not contain any processors.")
        return

    selected = st.multiselect(
        "Processors To Keep",
        options=processor_ids,
        default=processor_ids,
        key="ai_studio_processors_to_keep",
        help="Metrics and tiles that depend on rejected processors are removed automatically.",
    )
    if st.button("Update Draft: Processor Selection", type="primary", disabled=not selected):
        _set_draft(filter_draft_by_selection(draft, selected_processors=selected))
        st.rerun()

    _render_processor_parameter_editor(draft, working, approved_fields)

    with components.bordered_panel(
        "Raw Processors YAML",
        "Use this only when the visual controls are too narrow for the generated processor.",
    ):
        text = yaml.safe_dump(draft.get("processors", {}), sort_keys=False)
        components.sync_text_area("ai_studio_raw_processors_yaml", text)
        raw = st.text_area(
            "processors.yaml",
            key="ai_studio_raw_processors_yaml",
            height=360,
            help=config_help.field_help("ai.raw_yaml"),
        )
        if st.button("Update Draft From Processors YAML", type="secondary"):
            try:
                sections = parse_ai_yaml_sections(raw)
                if "processors" not in sections:
                    raise ValueError("YAML must include a processors section")
                _set_draft(merge_draft_sections(draft, sections))
                st.rerun()
            except Exception as exc:
                _log_ai_operation_failure("Raw processor YAML apply", exc)
                st.error(str(exc))
    with st.expander("Approved field catalog", expanded=False):
        st.write(", ".join(approved_fields) if approved_fields else "No approved fields.")


def _render_processor_parameter_editor(
    draft: dict[str, Any],
    working: pl.DataFrame,
    approved_fields: list[str],
) -> None:
    processors = _draft_processor_definitions(draft)
    if not processors:
        return
    processors_by_id = {
        str(processor.get("id")): processor for processor in processors if processor.get("id")
    }
    processor_ids = sorted(processors_by_id, key=str.casefold)
    if not processor_ids:
        return

    with components.bordered_panel(
        "Processor Parameter Editor",
        "Select one generated processor, edit its parameters and states, then apply the change.",
    ):
        current_processor = st.session_state.get(
            "ai_studio_processor_editor_processor", processor_ids[0]
        )
        processor_id = st.selectbox(
            "Processor",
            processor_ids,
            index=builder.option_index(processor_ids, current_processor),
            key="ai_studio_processor_editor_processor",
            help=config_help.field_help("processor.selector"),
        )
        processor_def = dict(processors_by_id.get(processor_id, {}))
        key_prefix = f"ai_studio_processor_editor_{builder.widget_key_fragment(processor_id)}"

        source_ids = _draft_source_ids(draft)
        current_source = str(processor_def.get("source", "") or "")
        current_kind = str(processor_def.get("kind", "") or "")
        source_options = forms.with_current(source_ids, current_source)
        kind_options = forms.with_current(list(forms.PROCESSOR_KIND_OPTIONS), current_kind)

        id_col, source_col, kind_col, description_col = st.columns(
            [1, 1, 1, 1],
            gap="xsmall",
            vertical_alignment="bottom",
        )
        new_processor_id = id_col.text_input(
            "Processor ID",
            value=processor_id,
            key=f"{key_prefix}_id",
            help=config_help.field_help("processor.id"),
        ).strip()
        source = source_col.selectbox(
            "Source",
            source_options or [""],
            index=builder.option_index(source_options or [""], current_source),
            key=f"{key_prefix}_source",
            help=config_help.field_help("processor.source"),
        )
        kind = kind_col.selectbox(
            "Processor Kind",
            kind_options,
            index=builder.option_index(kind_options, current_kind),
            key=f"{key_prefix}_kind",
            help=config_help.field_help("processor.kind"),
        )

        description = description_col.text_input(
            "Description",
            value=str(processor_def.get("description", "") or ""),
            key=f"{key_prefix}_description",
            help=config_help.field_help("processor.description"),
        ).strip()

        field_options = forms.with_current(
            sorted(approved_fields or list(working.columns), key=str.casefold),
            _processor_dimensions(processor_def),
        )
        dimensions = st.multiselect(
            "Dimensions",
            field_options,
            default=[
                field for field in _processor_dimensions(processor_def) if field in field_options
            ],
            key=f"{key_prefix}_dimensions",
            help=config_help.field_help("processor.group_by"),
        )
        time_def = processor_def.get("time") if isinstance(processor_def.get("time"), dict) else {}
        time_col, grains_col = st.columns(2, gap="xsmall", vertical_alignment="bottom")
        with time_col:
            time_column = forms.select_or_text(
                "Time Column",
                forms.with_current(list(working.columns), str(time_def.get("column", "") or "")),
                time_def.get("column", ""),
                key=f"{key_prefix}_time_column",
                help_key="processor.time_column",
            )
        grains = grains_col.multiselect(
            "Grains",
            list(forms.PROCESSOR_GRAIN_OPTIONS),
            default=[
                builder.display_grain(grain)
                for grain in builder.string_list(time_def.get("grains")) or ["Summary"]
                if builder.display_grain(grain) in forms.PROCESSOR_GRAIN_OPTIONS
            ],
            key=f"{key_prefix}_grains",
            help=config_help.field_help("processor.grains"),
        )

        kind_fields = _processor_kind_parameter_fields(
            processor_def,
            kind,
            working,
            approved_fields,
            key_prefix=key_prefix,
        )
        states = _processor_state_editor(processor_def, key_prefix=key_prefix)
        filter_value = _processor_filter_editor(processor_def, key_prefix=key_prefix)

        edited_processor = _without_empty(
            {
                **_processor_preserved_fields(processor_def),
                "id": new_processor_id,
                "source": source,
                "kind": kind,
                "description": description,
                "dimensions": dimensions,
                "time": {"column": time_column or None, "grains": grains or ["Summary"]},
                **kind_fields,
                "states": states,
                "filter": filter_value,
            }
        )

        with st.expander("Processor YAML Preview", expanded=False):
            st.code(
                yaml.safe_dump({"processors": [edited_processor]}, sort_keys=False),
                language="yaml",
            )

        if st.button("Update Processor In Draft", type="primary", key=f"{key_prefix}_apply"):
            if not new_processor_id:
                st.error("Processor ID is required.")
                return
            if new_processor_id != processor_id and new_processor_id in processors_by_id:
                st.error(f"Processor {new_processor_id!r} already exists.")
                return
            _set_draft(
                _update_processor_definition(
                    draft,
                    processor_id,
                    new_processor_id,
                    edited_processor,
                )
            )
            st.rerun()


def _processor_kind_parameter_fields(
    processor_def: dict[str, Any],
    kind: str,
    working: pl.DataFrame,
    approved_fields: list[str],
    *,
    key_prefix: str,
) -> dict[str, Any]:
    field_options = sorted(approved_fields or list(working.columns), key=str.casefold)
    numeric_options = _numeric_field_options(working, field_options)
    return forms.processor_kind_fields(
        processor_def,
        kind,
        field_options=field_options,
        numeric_field_options=numeric_options or None,
        key_prefix=key_prefix,
    )


def _processor_state_editor(processor_def: dict[str, Any], *, key_prefix: str) -> dict[str, Any]:
    rows = _processor_state_rows(processor_def)
    frame = builder.editor_frame(
        rows,
        ["State", "Type", "Source Column", "Enabled"],
        _blank_processor_state_row,
    )
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"{key_prefix}_states",
        column_config={
            "State": st.column_config.TextColumn(
                "State", width="medium", help=config_help.field_help("state.name")
            ),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=list(builder.STATE_TYPES),
                width="xsmall",
                help=config_help.field_help("state.type"),
            ),
            "Source Column": st.column_config.TextColumn(
                "Source Column",
                width="medium",
                help=config_help.field_help("state.source_column"),
            ),
            "Enabled": st.column_config.CheckboxColumn(
                "Enabled", width="xsmall", help=config_help.field_help("row.enabled")
            ),
        },
    )
    return _processor_states_from_rows(
        builder.normalize_editor_rows(edited),
        _processor_state_specs(processor_def),
    )


def _processor_filter_editor(
    processor_def: dict[str, Any], *, key_prefix: str
) -> dict[str, Any] | None:
    filter_value = processor_def.get("filter")
    default_text = yaml.safe_dump(filter_value, sort_keys=False).strip() if filter_value else ""
    components.sync_text_area(f"{key_prefix}_filter", default_text)
    raw_filter = st.text_area(
        "Processor Filter YAML",
        key=f"{key_prefix}_filter",
        height=160,
        help=config_help.field_help("processor.filter_ast"),
    )
    if not raw_filter.strip():
        return None
    try:
        parsed = yaml.safe_load(raw_filter)
    except yaml.YAMLError as exc:
        st.warning(f"Filter YAML could not be parsed: {exc}")
        return filter_value if isinstance(filter_value, dict) else None
    if not isinstance(parsed, dict):
        st.warning("Processor filter must be a YAML mapping.")
        return filter_value if isinstance(filter_value, dict) else None
    return parsed


def _processor_state_rows(processor_def: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = _processor_state_specs(processor_def)
    seen: set[str] = set()
    raw_states = processor_def.get("states")
    if isinstance(raw_states, dict):
        for name, spec in raw_states.items():
            spec_dict = spec if isinstance(spec, dict) else {"type": spec}
            merged_spec = specs.get(str(name), spec_dict)
            rows.append(
                {
                    "State": str(name),
                    "Type": str(merged_spec.get("type", "") or ""),
                    "Source Column": str(merged_spec.get("source_column", "") or ""),
                    "Enabled": True,
                }
            )
            seen.add(str(name))
    elif isinstance(raw_states, list):
        for spec in raw_states:
            if isinstance(spec, dict):
                name = spec.get("name") or spec.get("id") or spec.get("state") or spec.get("output")
                merged_spec = specs.get(str(name or ""), spec)
                rows.append(
                    {
                        "State": str(name or ""),
                        "Type": str(merged_spec.get("type", "") or ""),
                        "Source Column": str(merged_spec.get("source_column", "") or ""),
                        "Enabled": True,
                    }
                )
                if name:
                    seen.add(str(name))
    for name, spec in specs.items():
        if name not in seen:
            rows.append(
                {
                    "State": name,
                    "Type": str(spec.get("type", "") or ""),
                    "Source Column": str(spec.get("source_column", "") or ""),
                    "Enabled": True,
                }
            )
    return rows or [_blank_processor_state_row()]


def _processor_state_specs(processor_def: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return explicit state specs filled from kind-derived draft defaults."""
    specs = {name: dict(spec) for name, spec in _inferred_state_specs(processor_def).items()}
    for name, spec in builder.state_spec_definitions(processor_def).items():
        merged = dict(specs.get(name, {}))
        merged.update(spec)
        specs[name] = merged
    return specs


def _processor_states_from_rows(
    rows: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    existing_specs = existing or {}
    states: dict[str, Any] = {}
    for row in rows:
        if not _truthy_editor_value(row.get("Enabled", True)):
            continue
        name = str(row.get("State", "") or "").strip()
        state_type = str(row.get("Type", "") or "").strip()
        if not name or not state_type:
            continue
        spec: dict[str, Any] = dict(existing_specs.get(name, {}))
        spec["type"] = state_type
        source_column = str(row.get("Source Column", "") or "").strip()
        if source_column:
            spec["source_column"] = source_column
        else:
            spec.pop("source_column", None)
        states[name] = spec
    return states


def _blank_processor_state_row() -> dict[str, Any]:
    return {"State": "", "Type": "count", "Source Column": "", "Enabled": True}


def _processor_preserved_fields(processor_def: dict[str, Any]) -> dict[str, Any]:
    handled = {
        "id",
        "source",
        "kind",
        "description",
        "dimensions",
        "group_by",
        "time",
        "states",
        "filter",
        "entities",
        "outcome",
        "outcome_column",
        "positive_values",
        "negative_values",
        "variant_column",
        "properties",
        "quantile_engine",
        "sketch_build_mode",
        "score_properties",
        "score_columns",
        "stages",
        "snapshot_kind",
        "cadence",
    }
    return {key: value for key, value in processor_def.items() if key not in handled}


def _update_processor_definition(
    draft: dict[str, Any],
    old_id: str,
    new_id: str,
    processor_def: dict[str, Any],
) -> dict[str, Any]:
    return update_processor_definition(draft, old_id, new_id, processor_def)


def _draft_processor_definitions(draft: dict[str, Any]) -> list[dict[str, Any]]:
    processors = draft.get("processors", {}).get("processors", [])
    if not isinstance(processors, list):
        return []
    return [dict(processor) for processor in processors if isinstance(processor, dict)]


def _draft_source_ids(draft: dict[str, Any]) -> list[str]:
    sources = draft.get("pipelines", {}).get("sources", [])
    if not isinstance(sources, list):
        return []
    return [
        str(source.get("id")) for source in sources if isinstance(source, dict) and source.get("id")
    ]


def _processor_dimensions(processor_def: dict[str, Any]) -> list[str]:
    dimensions = processor_def.get("dimensions", processor_def.get("group_by", []))
    return builder.string_list(dimensions)


def _numeric_field_options(working: pl.DataFrame, fields: list[str]) -> list[str]:
    return [
        field
        for field in fields
        if field in working.columns
        and working.schema.get(field) is not None
        and working.schema[field].is_numeric()
        and working.schema[field] != pl.Boolean
    ]


def _draft_metric_choice_label(draft: dict[str, Any], metric_name: str) -> str:
    metric_def = _draft_metric_definitions(draft).get(metric_name)
    if metric_def is None:
        return metric_name
    source = str(metric_def.get("source", "") or "unknown")
    kind = str(metric_def.get("kind", "") or "unknown")
    return f"{metric_name} · {source} · {builder.metric_kind_label(kind)}"


def _draft_metric_source_ids(draft: dict[str, Any]) -> list[str]:
    metric_defs = _draft_metric_definitions(draft)
    metric_sources = [
        str(metric_def.get("source", "") or "")
        for metric_def in metric_defs.values()
        if metric_def.get("source")
    ]
    ordered_sources = [
        processor_id
        for processor_id in _draft_processor_ids(draft)
        if processor_id in metric_sources
    ]
    unknown_sources = sorted(
        {source for source in metric_sources if source and source not in ordered_sources},
        key=str.casefold,
    )
    return [*ordered_sources, *unknown_sources]


def _draft_metric_kinds_for_source(draft: dict[str, Any], source: str) -> list[str]:
    kinds = [
        str(metric_def.get("kind", "") or "")
        for metric_def in _draft_metric_definitions(draft).values()
        if metric_def.get("source") == source
    ]
    return sorted(builder.dedupe([kind for kind in kinds if kind]), key=builder.metric_kind_label)


def _draft_metric_names_for_source_kind(draft: dict[str, Any], source: str, kind: str) -> list[str]:
    return sorted(
        [
            name
            for name, metric_def in _draft_metric_definitions(draft).items()
            if metric_def.get("source") == source and metric_def.get("kind") == kind
        ],
        key=str.casefold,
    )


def _metrics_review(working: pl.DataFrame, approved_fields: list[str]) -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Metrics Review")
    st.caption("Review generated metric definitions before refreshing reports.")
    _draft_counts(draft)
    ok, issues = validate_draft_catalog(draft)
    _render_draft_validation(ok, issues, expanded=False)
    _render_ai_repair_panel(draft, working, approved_fields, issues)
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    _render_ai_refine_panel(draft, working, approved_fields)
    _render_coverage_panel(draft, working, approved_fields)
    _render_ai_recipe_library(draft)
    metrics = sorted(draft.get("metrics", {}).get("metrics", {}), key=str.casefold)
    if not metrics:
        st.info("The draft does not contain any metrics yet. Add one from the recipe library.")
        return
    selected = st.multiselect(
        "Metrics To Keep",
        options=metrics,
        default=metrics,
        format_func=lambda name: _draft_metric_choice_label(draft, name),
        key="ai_studio_metrics_to_keep",
        help="Tiles for rejected metrics are removed automatically.",
    )
    if st.button("Update Draft: Metric Selection", type="primary", disabled=not selected):
        _set_draft(filter_draft_by_selection(draft, selected_metrics=selected))
        st.rerun()

    _render_metric_parameter_editor(draft)

    with components.bordered_panel(
        "Raw Metrics YAML",
        "Use this only when the visual controls are too narrow for the generated metric.",
    ):
        text = yaml.safe_dump(draft.get("metrics", {}), sort_keys=False)
        components.sync_text_area("ai_studio_raw_metrics_yaml", text)
        raw = st.text_area(
            "metrics.yaml",
            key="ai_studio_raw_metrics_yaml",
            height=360,
            help=config_help.field_help("ai.raw_yaml"),
        )
        if st.button("Update Draft From Metrics YAML", type="secondary"):
            try:
                sections = parse_ai_yaml_sections(raw)
                if "metrics" not in sections:
                    raise ValueError("YAML must include a metrics section")
                _set_draft(merge_draft_sections(draft, sections))
                st.rerun()
            except Exception as exc:
                _log_ai_operation_failure("Raw metric YAML apply", exc)
                st.error(str(exc))
    with st.expander("Approved field catalog", expanded=False):
        st.write(", ".join(approved_fields) if approved_fields else "No approved fields.")


def _render_ai_recipe_library(draft: dict[str, Any]) -> None:
    """Install a shared KPI recipe into the session-local AI draft."""

    materialization_feedback = st.session_state.pop(
        "ai_studio_recipe_materialization_feedback", None
    )
    if isinstance(materialization_feedback, dict):
        source_id = str(materialization_feedback.get("source_id", "") or "")
        states = ", ".join(
            f"`{value}`" for value in materialization_feedback.get("state_names", [])
        )
        st.info(
            f"Recipe added to the draft. Use **Save Draft & Run Source** in Save & "
            f"Export to materialize {states or 'the new aggregate state'} from "
            f"source `{source_id}`."
        )

    try:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": draft.get("pipelines", {}),
                "processors": draft.get("processors", {}),
                "metrics": draft.get("metrics", {}),
                "dashboards": draft.get("dashboards", {}),
            }
        )
    except ValueError as exc:
        st.warning(
            f"Recipe compatibility cannot be evaluated until the draft shape is valid: {exc}"
        )
        return
    request = recipe_library.render_recipe_library(
        catalog=catalog,
        key_prefix="ai_studio_kpi_recipes",
        submit_label="Add recipe to draft",
        expanded=not bool(_draft_metric_definitions(draft)),
    )
    if request is None:
        return
    try:
        updated = _install_recipe_in_draft(draft, request)
        _set_draft(updated)
        if request.materialization:
            st.session_state["ai_studio_recipe_materialization_feedback"] = {
                "source_id": request.materialization.source_id,
                "state_names": list(request.materialization.state_names),
            }
        st.toast("KPI recipe added to the draft.", icon=":material/check:")
        st.rerun()
    except (TypeError, ValueError) as exc:
        st.error(str(exc))


def _install_recipe_in_draft(
    draft: dict[str, Any],
    request: recipe_library.RecipeInstallRequest,
) -> dict[str, Any]:
    """Return a copy of an AI draft with one recipe metric and optional tile."""
    return install_recipe_request_in_draft(draft, request)


def _render_metric_parameter_editor(draft: dict[str, Any]) -> None:
    metrics = _draft_metric_definitions(draft)
    if not metrics:
        return

    with components.bordered_panel(
        "Metric Parameter Editor",
        "Select one generated metric, edit its catalog parameters, then apply the change.",
    ):
        metric_names = sorted(metrics, key=str.casefold)
        current_metric = st.session_state.get(
            "ai_studio_metric_editor_selected_id",
            st.session_state.get("ai_studio_metric_editor_metric", metric_names[0]),
        )
        current_metric_def = (
            metrics.get(str(current_metric), {}) if isinstance(current_metric, str) else {}
        )
        source_options = _draft_metric_source_ids(draft)
        if not source_options:
            st.info("No metric sources are available.")
            return
        current_source = str(current_metric_def.get("source", "") or "")
        source_key = "ai_studio_metric_editor_source"
        if st.session_state.get(source_key) not in source_options:
            st.session_state.pop(source_key, None)
        source = st.selectbox(
            "Source Processor",
            source_options,
            index=builder.option_index(source_options, current_source),
            key=source_key,
            help=config_help.field_help("metric.processor"),
        )
        kind_options = _draft_metric_kinds_for_source(draft, source)
        if not kind_options:
            st.info("Selected processor has no editable metric kinds.")
            return
        current_kind = (
            str(current_metric_def.get("kind", "") or "")
            if current_metric_def.get("source") == source
            else ""
        )
        kind_key = f"ai_studio_metric_editor_kind_{builder.widget_key_fragment(source)}"
        if st.session_state.get(kind_key) not in kind_options:
            st.session_state.pop(kind_key, None)
        kind = st.selectbox(
            "Metric Kind",
            kind_options,
            index=builder.option_index(kind_options, current_kind),
            format_func=builder.metric_kind_label,
            key=kind_key,
            help=config_help.field_help("metric.kind"),
        )
        metric_choices = _draft_metric_names_for_source_kind(draft, source, kind)
        if not metric_choices:
            st.info("Selected processor and kind have no editable metrics.")
            return
        metric_key = f"ai_studio_metric_editor_metric_{builder.widget_key_fragment(source)}_{builder.widget_key_fragment(kind)}"
        if st.session_state.get(metric_key) not in metric_choices:
            st.session_state.pop(metric_key, None)
            current_metric = None
        metric_name = st.selectbox(
            "Metric",
            metric_choices,
            index=builder.option_index(metric_choices, current_metric),
            format_func=lambda name: _draft_metric_choice_label(draft, name),
            key=metric_key,
            help=config_help.field_help("metric.selector"),
        )
        st.session_state["ai_studio_metric_editor_selected_id"] = metric_name
        metric_def = dict(metrics.get(metric_name, {}))
        key_prefix = f"ai_studio_metric_editor_{builder.widget_key_fragment(metric_name)}"

        st.caption(f"Metric ID: `{metric_name}`")
        renamed_metric = metric_name
        description = st.text_input(
            "Description",
            value=str(metric_def.get("description", "") or ""),
            key=f"{key_prefix}_description",
            help=config_help.field_help("metric.description"),
        ).strip()
        depends_on = st.multiselect(
            "Depends On",
            forms.with_current(
                [name for name in metric_names if name != metric_name],
                builder.string_list(metric_def.get("depends_on")),
            ),
            default=builder.string_list(metric_def.get("depends_on")),
            key=f"{key_prefix}_depends_on",
            help=config_help.field_help("metric.depends_on"),
        )
        display = _draft_metric_display_controls(metric_def.get("display"), key_prefix)

        edited_metric = _without_empty(
            {
                "source": source,
                "kind": kind,
                "description": description,
                "depends_on": depends_on,
                "display": display,
                **_metric_kind_parameter_fields(
                    draft,
                    source,
                    kind,
                    metric_def,
                    key_prefix=key_prefix,
                ),
            }
        )

        with st.expander("Metric YAML Preview", expanded=False):
            st.code(
                yaml.safe_dump({renamed_metric or metric_name: edited_metric}, sort_keys=False),
                language="yaml",
            )

        if st.button("Update Metric In Draft", type="primary", key=f"{key_prefix}_apply"):
            _set_draft(_update_metric_definition(draft, metric_name, renamed_metric, edited_metric))
            st.rerun()


def _draft_metric_display_controls(raw: Any, key_prefix: str) -> dict[str, Any]:
    seed = dict(raw) if isinstance(raw, dict) else {}
    with st.expander("Report presentation", expanded=False):
        label = st.text_input(
            "Display label",
            value=str(seed.get("label", "") or ""),
            key=f"{key_prefix}_display_label",
            help=config_help.field_help("metric.display_label"),
        ).strip()
        unit = st.text_input(
            "Unit",
            value=str(seed.get("unit", "") or ""),
            key=f"{key_prefix}_display_unit",
            help=config_help.field_help("metric.unit"),
        ).strip()
        formats = ["", "percent", "integer", "number", "currency"]
        value_format = st.selectbox(
            "Default value format",
            formats,
            index=builder.option_index(formats, seed.get("value_format")),
            format_func=lambda value: "Unspecified" if not value else value.title(),
            key=f"{key_prefix}_display_format",
            help=config_help.field_help("metric.value_format"),
        )
        directions = ["neutral", "higher_is_better", "lower_is_better"]
        direction = st.selectbox(
            "Direction",
            directions,
            index=builder.option_index(directions, seed.get("direction") or "neutral"),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"{key_prefix}_display_direction",
            help=config_help.field_help("metric.direction"),
        )
    return _without_empty(
        {
            "label": label,
            "unit": unit,
            "value_format": value_format,
            "direction": direction if direction != "neutral" else "",
        }
    )


def _metric_kind_parameter_fields(
    draft: dict[str, Any],
    source: str,
    kind: str,
    metric_def: dict[str, Any],
    *,
    key_prefix: str,
) -> dict[str, Any]:
    processor = _draft_processor_by_id(draft, source)
    roles = processor.get("variant_role_map", {})
    ctx = forms.MetricFormContext(
        state_options=lambda types: _draft_state_options(draft, source, state_types=set(types)),
        digest_pairs=_draft_digest_pair_options(draft, source),
        funnel_stages=_draft_funnel_stage_options(draft, source),
        default_variant_column=str(processor.get("variant_column", "") or ""),
        variant_roles=dict(roles) if isinstance(roles, dict) else {},
    )
    fields = forms.metric_kind_fields(kind, metric_def, ctx, key_prefix=key_prefix)
    if fields is None:
        # Keep the previous kind-specific fields instead of writing a broken
        # definition into the draft.
        return {
            key: value for key, value in metric_def.items() if key not in forms.METRIC_BASE_FIELDS
        }
    return fields


def _draft_digest_pair_options(
    draft: dict[str, Any],
    processor_id: str,
) -> list[tuple[str, str, str]]:
    processor = _draft_processor_by_id(draft, processor_id)
    explicit_pairs = builder.digest_pair_options_from_definition(processor)
    if explicit_pairs:
        return explicit_pairs
    return [
        (
            property_name,
            f"{property_name}_tdigest_positives",
            f"{property_name}_tdigest_negatives",
        )
        for property_name in builder.score_properties_from_definition(processor)
    ]


def _update_metric_definition(
    draft: dict[str, Any],
    old_name: str,
    new_name: str,
    metric_def: dict[str, Any],
) -> dict[str, Any]:
    return update_metric_definition(draft, old_name, new_name, metric_def)


def _draft_metric_definitions(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = draft.get("metrics", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {str(name): dict(value) for name, value in metrics.items() if isinstance(value, dict)}


def _draft_processor_ids(draft: dict[str, Any]) -> list[str]:
    processors = draft.get("processors", {}).get("processors", [])
    if not isinstance(processors, list):
        return []
    return [
        str(processor.get("id"))
        for processor in processors
        if isinstance(processor, dict) and processor.get("id")
    ]


def _draft_processor_by_id(draft: dict[str, Any], processor_id: str) -> dict[str, Any]:
    processors = draft.get("processors", {}).get("processors", [])
    if not isinstance(processors, list):
        return {}
    for processor in processors:
        if isinstance(processor, dict) and str(processor.get("id", "")) == processor_id:
            return processor
    return {}


def _draft_state_options(
    draft: dict[str, Any],
    processor_id: str,
    *,
    state_types: set[str] | None = None,
) -> list[str]:
    wanted = state_types or set()
    states = _draft_state_types(draft, processor_id)
    return [name for name, state_type in states.items() if not wanted or state_type in wanted]


def _draft_state_types(draft: dict[str, Any], processor_id: str) -> dict[str, str]:
    processor = _draft_processor_by_id(draft, processor_id)
    return {
        name: state_type
        for name, spec in _processor_state_specs(processor).items()
        if (state_type := _state_type_from_spec(spec))
    }


def _explicit_state_types(processor: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    raw_states = processor.get("states")
    if isinstance(raw_states, dict):
        for name, spec in raw_states.items():
            out[str(name)] = _state_type_from_spec(spec)
    elif isinstance(raw_states, list):
        for spec in raw_states:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name") or spec.get("id") or spec.get("state") or spec.get("output")
            if name:
                out[str(name)] = _state_type_from_spec(spec)
    return {name: state_type for name, state_type in out.items() if state_type}


def _state_type_from_spec(spec: Any) -> str:
    if isinstance(spec, dict):
        return str(spec.get("type", "") or "")
    if isinstance(spec, str):
        return spec
    return ""


def _inferred_state_types(processor: dict[str, Any]) -> dict[str, str]:
    return {
        name: state_type
        for name, spec in _inferred_state_specs(processor).items()
        if (state_type := _state_type_from_spec(spec))
    }


def _inferred_state_specs(processor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    kind = str(processor.get("kind", "") or "")
    if kind == "binary_outcome":
        states = {
            "Count": {"type": "count"},
            "Positives": {"type": "count"},
            "Negatives": {"type": "count"},
        }
        subject = _processor_subject(processor)
        if subject:
            states["UniqueSubjects_cpc"] = {
                "type": "cpc",
                "source_column": subject,
                "lg_k": 11,
            }
        return states
    if kind == "numeric_distribution":
        engine = "kll" if str(processor.get("quantile_engine", "") or "") == "kll" else "tdigest"
        return _numeric_distribution_state_specs(
            builder.string_list(processor.get("properties")),
            engine,
        )
    if kind == "score_distribution":
        subject = _processor_subject(processor) or "CustomerID"
        unique_state = "UniqueCustomers_cpc" if subject == "CustomerID" else "UniqueSubjects_cpc"
        states = {
            "Count": {"type": "count"},
            "personalization": {"type": "pooled_mean", "weight": "Count"},
            "novelty": {"type": "pooled_mean", "weight": "Count"},
            unique_state: {"type": "cpc", "source_column": subject, "lg_k": 11},
        }
        for property_name in builder.score_properties_from_definition(processor):
            states[f"{property_name}_tdigest_positives"] = {
                "type": "tdigest",
                "source_column": property_name,
                "outcome": "positive",
                "score_property": property_name,
                "k": 500,
            }
            states[f"{property_name}_tdigest_negatives"] = {
                "type": "tdigest",
                "source_column": property_name,
                "outcome": "negative",
                "score_property": property_name,
                "k": 500,
            }
        return states
    if kind in {"snapshot", "entity_lifecycle"}:
        return {"Count": {"type": "count"}}
    if kind == "funnel":
        return {
            f"{stage}_Count": {"type": "count"}
            for stage in forms.stage_names_from_definition(processor.get("stages"))
        }
    return {}


def _numeric_distribution_state_types(properties: list[str], engine: str) -> dict[str, str]:
    return {
        name: state_type
        for name, spec in _numeric_distribution_state_specs(properties, engine).items()
        if (state_type := _state_type_from_spec(spec))
    }


def _numeric_distribution_state_specs(
    properties: list[str],
    engine: str,
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for prop in properties:
        states.update(
            {
                f"{prop}_Count": {"type": "count", "per_property": True},
                f"{prop}_Sum": {"type": "value_sum", "per_property": True},
                f"{prop}_Mean": {
                    "type": "pooled_mean",
                    "per_property": True,
                    "weight": f"{prop}_Count",
                },
                f"{prop}_Var": {"type": "pooled_variance", "per_property": True},
                f"{prop}_Min": {"type": "min", "per_property": True},
                f"{prop}_Max": {"type": "max", "per_property": True},
                f"{prop}_{engine}": {"type": engine, "per_property": True},
            }
        )
    return states


def _processor_subject(processor: dict[str, Any]) -> str:
    entities = processor.get("entities")
    if isinstance(entities, dict):
        return str(entities.get("subject", "") or "")
    return ""


def _draft_funnel_stage_options(draft: dict[str, Any], processor_id: str) -> list[str]:
    processor = _draft_processor_by_id(draft, processor_id)
    stages = forms.stage_names_from_definition(processor.get("stages"))
    if stages:
        return stages
    suffix = "_Count"
    return [
        name[: -len(suffix)]
        for name, state_type in _draft_state_types(draft, processor_id).items()
        if state_type == "count" and name.endswith(suffix)
    ]


def _ai_reports(
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
    *,
    ai_calls_enabled: bool = True,
) -> None:
    if preprocessing_error:
        st.warning("Resolve preprocessing errors before refreshing reports.")
        return
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### AI Reports" if ai_calls_enabled else "### Reports")
    st.caption(
        "Use a second AI pass to replace dashboards.yaml from the current metrics."
        if ai_calls_enabled
        else "Rebuild starter dashboards locally from the current draft metrics."
    )
    if ai_calls_enabled and st.session_state.get("ai_studio_pending_draft") is not None:
        return
    ok, issues = validate_draft_catalog(draft)
    _render_draft_validation(ok, issues, expanded=False)
    _draft_counts(draft)
    if not ai_calls_enabled:
        rows = _tile_inventory_rows(draft)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch", height=260)
        if st.button("Rebuild Reports From Metrics", type="primary"):
            dashboards = _deterministic_dashboards_from_metrics(draft, working, approved_fields)
            _set_draft(merge_draft_sections(draft, {"dashboards": dashboards}))
            st.success("Starter reports rebuilt from the current metrics.")
            st.rerun()
        return

    schema_preview = _schema_preview_for_ai(working, approved_fields)
    hidden_fields = sorted(set(working.columns) - set(approved_fields), key=str.casefold)
    prompt = prompt_for_report_refresh(
        file_name=_sample_file_name(),
        approved_schema=schema_preview,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=draft,
        user_goals=_current_user_goals(),
    )
    ai_settings = _current_ai_settings()
    sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
    action_col1, action_col2 = st.columns([0.3, 0.7], vertical_alignment="center")
    if action_col1.button(
        "Refresh Reports From Metrics",
        type="primary",
        disabled=ai_settings is None or not sharing_confirmed,
        help=(
            "Configure a LiteLLM model in the sidebar to enable AI report refresh."
            if ai_settings is None
            else "Confirm the AI data-sharing scope above to enable AI report refresh."
            if not sharing_confirmed
            else "Refresh reports with the confirmed sharing scope."
        ),
    ):
        try:
            with st.status("Refreshing reports", expanded=True) as status:
                response = _call_litellm_for_current_sample(
                    ai_settings,
                    prompt,
                    approved_fields=approved_fields,
                )
                sections = parse_ai_yaml_sections(response)
                if "dashboards" not in sections:
                    raise ValueError("AI response did not include dashboards")
                pending = merge_draft_sections(
                    draft,
                    {"dashboards": _with_generated_report_ids(sections["dashboards"])},
                )
                st.session_state["ai_studio_pending_draft"] = pending
                st.session_state["ai_studio_pending_base_draft"] = draft
                st.session_state["ai_studio_pending_kind"] = "reports"
                st.session_state["ai_studio_pending_prompt"] = prompt
                st.session_state["ai_studio_last_ai_response"] = response
                status.update(label="Reports ready for review", state="complete")
            st.rerun()
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _log_ai_operation_failure("Report refresh", exc)
            st.error(f"Report refresh failed: {exc}")
    with action_col2, st.popover("Show Prompt", icon=":material/psychology:"):
        st.code(prompt, language="text")
    if not ai_settings:
        st.info("Configure a LiteLLM model in the sidebar to refresh reports.")


def _deterministic_dashboards_from_metrics(
    draft: dict[str, Any],
    working: pl.DataFrame,
    approved_fields: list[str],
) -> dict[str, Any]:
    metrics = _draft_metric_definitions(draft)
    processors = {
        str(processor.get("id")): processor
        for processor in _draft_processor_definitions(draft)
        if processor.get("id")
    }
    tiles: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for metric_name, metric_def in sorted(metrics.items(), key=lambda item: item[0].casefold()):
        processor = processors.get(str(metric_def.get("source", "") or ""), {})
        title = builder.title_from_identifier(metric_name)
        chart_kind, fields = _deterministic_tile_fields(
            metric_name,
            metric_def,
            processor,
            working,
            approved_fields,
        )
        if metric_def.get("description"):
            fields["description"] = str(metric_def["description"])
        if chart_kind == "kpi_card":
            fields.update(
                {
                    "placement": "kpi_strip",
                    "kpi": {
                        "comparison": "previous_period",
                        "comparison_period": "month",
                        "sparkline_grain": "monthly",
                        "sparkline_points": 12,
                    },
                }
            )
        tile_id = _unique_tile_id(
            builder.random_catalog_id(title, fallback="tile"),
            used_ids,
        )
        tiles.append(
            builder.build_tile(
                tile_id=tile_id,
                title=title,
                metric_name=metric_name,
                chart_kind=chart_kind,
                fields=fields,
            )
        )
    dashboards = draft.get("dashboards") if isinstance(draft.get("dashboards"), dict) else {}
    dashboard_title = "Builder Overview"
    page_title = "Metrics"
    page_filters = _deterministic_page_filters(
        metrics,
        processors,
        approved_fields,
    )
    return {
        "theme": dict(dashboards.get("theme", {})) if isinstance(dashboards, dict) else {},
        "dashboards": [
            {
                "id": builder.random_catalog_id(dashboard_title, fallback="dashboard"),
                "title": dashboard_title,
                "layout": "tabs",
                "pages": [
                    {
                        "id": builder.random_catalog_id(page_title, fallback="page"),
                        "title": page_title,
                        "filters": page_filters,
                        "time_filter": {
                            "default": "all_time",
                            "presets": [
                                "last_30_days",
                                "last_90_days",
                                "year_to_date",
                                "custom",
                                "all_time",
                            ],
                        },
                        "tiles": tiles,
                    }
                ],
            }
        ],
    }


def _deterministic_page_filters(
    metrics: dict[str, dict[str, Any]],
    processors: dict[str, dict[str, Any]],
    approved_fields: list[str],
) -> list[dict[str, Any]]:
    dimensions_by_metric = {
        metric_name: [
            field
            for field in _processor_dimensions(
                processors.get(str(metric_def.get("source", "") or ""), {})
            )
            if field in approved_fields
        ]
        for metric_name, metric_def in metrics.items()
    }
    fields = builder.dedupe(
        [field for dimensions in dimensions_by_metric.values() for field in dimensions]
    )
    filters: list[dict[str, Any]] = []
    primary_count = 0
    for field in fields[:8]:
        coverage = sum(field in dimensions for dimensions in dimensions_by_metric.values())
        all_tiles = bool(dimensions_by_metric) and coverage == len(dimensions_by_metric)
        display = "primary" if all_tiles and primary_count < 3 else "secondary"
        if display == "primary":
            primary_count += 1
        filters.append(
            {
                "field": field,
                "label": builder.title_from_identifier(field),
                "display": display,
                "scope": "all_tiles" if all_tiles else "compatible_tiles",
                "control": "multiselect",
            }
        )
    return filters


def _with_generated_report_ids(dashboards: dict[str, Any]) -> dict[str, Any]:
    """Return a dashboards section whose report ids come from display names."""
    out = {**dashboards}
    normalized_dashboards: list[dict[str, Any]] = []
    for dashboard in dashboards.get("dashboards", []) or []:
        if not isinstance(dashboard, dict):
            continue
        dashboard_title = str(dashboard.get("title") or dashboard.get("id") or "Dashboard")
        dashboard_copy = {
            **dashboard,
            "id": builder.random_catalog_id(dashboard_title, fallback="dashboard"),
            "title": dashboard_title,
        }
        normalized_pages: list[dict[str, Any]] = []
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            page_title = str(page.get("title") or page.get("id") or "Page")
            page_copy = {
                **page,
                "id": builder.random_catalog_id(page_title, fallback="page"),
                "title": page_title,
            }
            used_tile_ids: set[str] = set()
            normalized_tiles: list[dict[str, Any]] = []
            for tile in page.get("tiles", []) or []:
                if not isinstance(tile, dict):
                    continue
                tile_title = str(
                    tile.get("title") or tile.get("metric") or tile.get("id") or "Tile"
                )
                normalized_tiles.append(
                    {
                        **tile,
                        "id": _unique_tile_id(
                            builder.random_catalog_id(tile_title, fallback="tile"),
                            used_tile_ids,
                        ),
                        "title": tile_title,
                    }
                )
            page_copy["tiles"] = normalized_tiles
            normalized_pages.append(page_copy)
        dashboard_copy["pages"] = normalized_pages
        normalized_dashboards.append(dashboard_copy)
    out["dashboards"] = normalized_dashboards
    return out


def _deterministic_tile_fields(
    metric_name: str,
    metric_def: dict[str, Any],
    processor: dict[str, Any],
    working: pl.DataFrame,
    approved_fields: list[str],
) -> tuple[str, dict[str, Any]]:
    kind = str(metric_def.get("kind", "") or "")
    if kind == "calibration_from_digests":
        return "calibration_curve", {}
    if kind == "curve_from_digests":
        output = str(metric_def.get("output", "") or "")
        chart = "precision_recall_curve" if output == "average_precision" else "roc_curve"
        color = _first_processor_dimension(processor, approved_fields)
        return chart, {"color": color} if color else {}
    outputs = _metric_output_fields(metric_name, metric_def)
    dimensions = _processor_dimensions(processor)
    first_dimension = _first_existing_field(dimensions, approved_fields)
    time_field = _first_time_field(processor, working, approved_fields)
    value = outputs[0] if outputs else metric_name
    if kind == "lifecycle_summary":
        columns = builder.dedupe([*dimensions[:3], *outputs[:6]])
        return "table", {"columns": columns or [value]}
    if not time_field and not first_dimension:
        return "kpi_card", {"value": value}
    if time_field:
        fields = {"x": time_field, "y": value}
        if first_dimension:
            fields["color"] = first_dimension
        return "line", fields
    return "bar", {"x": first_dimension, "y": value}


def _metric_output_fields(metric_name: str, metric_def: dict[str, Any]) -> list[str]:
    outputs = builder.string_list(metric_def.get("outputs"))
    if outputs:
        return outputs
    kind = str(metric_def.get("kind", "") or "")
    if kind == "variant_compare":
        return [
            "CTR",
            "TestCTR",
            "ControlCTR",
            "AbsoluteRateDifference",
            "AbsoluteRateDifference_CI_Low",
            "AbsoluteRateDifference_CI_High",
            "Lift",
            "Lift_P_Val",
            "TestSampleSize",
            "ControlSampleSize",
        ]
    if kind in {"contingency_test", "proportion_test"}:
        return ["z_score", "z_p_val"]
    if kind == "lifecycle_summary":
        return list(forms.LIFECYCLE_OUTPUT_OPTIONS)
    return [metric_name]


def _first_time_field(
    processor: dict[str, Any],
    working: pl.DataFrame,
    approved_fields: list[str],
) -> str:
    time_def = processor.get("time") if isinstance(processor.get("time"), dict) else {}
    time_column = str(time_def.get("column", "") or "")
    grains = [builder.display_grain(grain) for grain in builder.string_list(time_def.get("grains"))]
    candidates = [grain for grain in ("Day", "Month", "Quarter", "Year") if grain in grains]
    candidates.append(time_column)
    candidates.extend(["Day", "Month", "Quarter", "Year"])
    return _first_existing_field(candidates, [*approved_fields, *working.columns])


def _first_processor_dimension(processor: dict[str, Any], approved_fields: list[str]) -> str:
    return _first_existing_field(_processor_dimensions(processor), approved_fields)


def _first_existing_field(candidates: list[str], available_fields: list[str]) -> str:
    available = set(available_fields)
    return next((field for field in candidates if field and field in available), "")


def _unique_tile_id(base: str, used_ids: set[str]) -> str:
    cleaned = base.strip("_") or "metric_tile"
    candidate = cleaned
    suffix = 2
    while candidate in used_ids:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _reports_review(working: pl.DataFrame, approved_fields: list[str]) -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Reports Review")
    st.caption("Review generated tiles, remove weak reports, or edit dashboards.yaml directly.")
    ok, issues = validate_draft_catalog(draft)
    _render_draft_validation(ok, issues, expanded=False)
    _render_ai_repair_panel(draft, None, approved_fields, issues)
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    _render_ai_refine_panel(draft, None, approved_fields)
    _render_coverage_panel(draft, working, approved_fields)

    keys = tile_keys(draft)
    if keys:
        selected_tiles = st.multiselect(
            "Tiles To Keep",
            options=keys,
            default=keys,
            key="ai_studio_tiles_to_keep",
            help=config_help.field_help("ai.keep_tiles"),
        )
        if st.button("Update Draft: Tile Selection", type="primary", disabled=not selected_tiles):
            _set_draft(filter_draft_by_selection(draft, selected_tiles=selected_tiles))
            st.rerun()
    else:
        st.warning("The draft does not contain any dashboard tiles.")

    rows = _tile_inventory_rows(draft)
    st.dataframe(rows, hide_index=True, width="stretch", height=320)
    _render_report_settings_editor(draft)
    with components.bordered_panel(
        "Raw Dashboards YAML",
        "Use this for chart settings not exposed by the compact review table.",
    ):
        text = yaml.safe_dump(draft.get("dashboards", {}), sort_keys=False)
        components.sync_text_area("ai_studio_raw_dashboards_yaml", text)
        raw = st.text_area(
            "dashboards.yaml",
            key="ai_studio_raw_dashboards_yaml",
            height=420,
            help=config_help.field_help("ai.raw_yaml"),
        )
        if st.button("Update Draft From Dashboards YAML", type="secondary"):
            try:
                sections = parse_ai_yaml_sections(raw)
                if "dashboards" not in sections:
                    raise ValueError("YAML must include a dashboards section")
                _set_draft(merge_draft_sections(draft, sections))
                st.rerun()
            except Exception as exc:
                _log_ai_operation_failure("Raw dashboard YAML apply", exc)
                st.error(str(exc))


def _render_report_settings_editor(  # noqa: PLR0912
    draft: dict[str, Any],
) -> None:
    pages: list[tuple[int, int, str, dict[str, Any]]] = []
    for dashboard_index, dashboard in enumerate(
        draft.get("dashboards", {}).get("dashboards", []) or []
    ):
        if not isinstance(dashboard, dict):
            continue
        for page_index, page in enumerate(dashboard.get("pages", []) or []):
            if isinstance(page, dict):
                label = f"{dashboard.get('title', dashboard.get('id', 'Dashboard'))} / {page.get('title', page.get('id', 'Page'))}"
                pages.append((dashboard_index, page_index, label, page))
    if not pages:
        return
    with components.bordered_panel(
        "Report Settings",
        "Edit aggregate-backed page controls and the selected tile without replacing other draft properties.",
    ):
        page_key = st.selectbox(
            "Page",
            list(range(len(pages))),
            format_func=lambda index: pages[index][2],
            key="ai_studio_report_settings_page",
            help=config_help.field_help("report.page"),
        )
        dashboard_index, page_index, _, page = pages[int(page_key)]
        filter_rows = [dict(row) for row in page.get("filters", []) if isinstance(row, dict)]
        if not filter_rows:
            filter_rows = [
                {
                    "field": "",
                    "label": "",
                    "display": "secondary",
                    "scope": "compatible_tiles",
                    "control": "multiselect",
                }
            ]
        edited_filters = st.data_editor(
            filter_rows,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=f"ai_studio_report_filters_{dashboard_index}_{page_index}",
            column_config={
                "field": st.column_config.TextColumn(
                    "Aggregate field", help=config_help.field_help("report.filter_field")
                ),
                "label": st.column_config.TextColumn(
                    "Display label", help=config_help.field_help("report.filter_label")
                ),
                "display": st.column_config.SelectboxColumn(
                    "Placement",
                    options=["primary", "secondary"],
                    help=config_help.field_help("report.filter_placement"),
                ),
                "scope": st.column_config.SelectboxColumn(
                    "Coverage",
                    options=["all_tiles", "compatible_tiles"],
                    help=config_help.field_help("report.filter_scope"),
                ),
                "control": st.column_config.SelectboxColumn(
                    "Control",
                    options=["multiselect", "selectbox", "text"],
                    help=config_help.field_help("report.filter_control"),
                ),
            },
        )
        time_seed = page.get("time_filter") if isinstance(page.get("time_filter"), dict) else {}
        all_presets = [
            "last_7_days",
            "last_30_days",
            "last_90_days",
            "year_to_date",
            "custom",
            "all_time",
        ]
        presets = st.multiselect(
            "Available time ranges",
            all_presets,
            default=[
                value
                for value in time_seed.get("presets", model.TimeFilterSpec().presets)
                if value in all_presets
            ],
            key=f"ai_studio_report_time_presets_{dashboard_index}_{page_index}",
            help=config_help.field_help("report.available_ranges"),
        )
        default_options = presets or ["all_time"]
        default_time = st.selectbox(
            "Default time range",
            default_options,
            index=builder.option_index(default_options, time_seed.get("default") or "all_time"),
            key=f"ai_studio_report_time_default_{dashboard_index}_{page_index}",
            help=config_help.field_help("report.default_range"),
        )

        tiles = [tile for tile in page.get("tiles", []) or [] if isinstance(tile, dict)]
        tile_index = (
            st.selectbox(
                "Tile",
                list(range(len(tiles))),
                format_func=lambda index: str(
                    tiles[index].get("title") or tiles[index].get("id") or "Tile"
                ),
                key=f"ai_studio_report_tile_{dashboard_index}_{page_index}",
                help=config_help.field_help("report.tile_title"),
            )
            if tiles
            else None
        )
        tile_updates: dict[str, Any] = {}
        if tile_index is not None:
            tile = tiles[int(tile_index)]
            description = st.text_area(
                "Tile description",
                value=str(tile.get("description", "") or ""),
                height=80,
                key=f"ai_studio_report_tile_description_{dashboard_index}_{page_index}_{tile_index}",
                help=config_help.field_help("report.description"),
            ).strip()
            tile_updates["description"] = description
            chart = str(tile.get("chart", ""))
            if chart in {"line", "stacked_area"}:
                tile_updates["scale_mode"] = st.selectbox(
                    "Scale",
                    ["absolute", "index_100", "percent_change"],
                    index=builder.option_index(
                        ["absolute", "index_100", "percent_change"],
                        tile.get("scale_mode") or "absolute",
                    ),
                    key=f"ai_studio_report_tile_scale_{dashboard_index}_{page_index}_{tile_index}",
                    help=config_help.field_help("report.scale"),
                )
            if chart == "kpi_card":
                placement = st.selectbox(
                    "Placement",
                    ["content", "kpi_strip"],
                    index=builder.option_index(
                        ["content", "kpi_strip"], tile.get("placement") or "content"
                    ),
                    key=f"ai_studio_report_tile_placement_{dashboard_index}_{page_index}_{tile_index}",
                    help=config_help.field_help("report.placement"),
                )
                tile_updates["placement"] = placement
                if placement == "kpi_strip":
                    raw_kpi = tile.get("kpi") if isinstance(tile.get("kpi"), dict) else {}
                    tile_updates["kpi"] = _ai_kpi_settings(
                        raw_kpi,
                        key=f"{dashboard_index}_{page_index}_{tile_index}",
                    )
                else:
                    tile_updates["kpi"] = None

        if st.button(
            "Update Report Settings In Draft",
            type="primary",
            disabled=not presets,
            key=f"ai_studio_report_settings_apply_{dashboard_index}_{page_index}",
        ):
            updated = copy.deepcopy(draft)
            target_page = updated["dashboards"]["dashboards"][dashboard_index]["pages"][page_index]
            target_page["filters"] = _normalize_report_filter_rows(edited_filters)
            target_page["time_filter"] = {"default": default_time, "presets": presets}
            if tile_index is not None:
                target_tile = target_page["tiles"][int(tile_index)]
                for key, value in tile_updates.items():
                    if (
                        (value in (None, "", {}, []) and key != "description")
                        or (key == "scale_mode" and value == "absolute")
                        or (key == "placement" and value == "content")
                    ):
                        target_tile.pop(key, None)
                    else:
                        target_tile[key] = value
            _set_draft(updated)
            st.rerun()


def _ai_kpi_settings(seed: dict[str, Any], *, key: str) -> dict[str, Any]:
    comparison = st.selectbox(
        "KPI comparison",
        ["none", "previous_period"],
        index=builder.option_index(["none", "previous_period"], seed.get("comparison") or "none"),
        key=f"ai_studio_report_kpi_comparison_{key}",
        help=config_help.field_help("report.comparison"),
    )
    period = st.selectbox(
        "KPI period",
        ["day", "week", "month", "quarter", "year"],
        index=builder.option_index(
            ["day", "week", "month", "quarter", "year"],
            seed.get("comparison_period") or "month",
        ),
        key=f"ai_studio_report_kpi_period_{key}",
        help=config_help.field_help("report.comparison_period"),
    )
    sparkline = st.selectbox(
        "KPI sparkline",
        ["", "daily", "weekly", "monthly"],
        index=builder.option_index(
            ["", "daily", "weekly", "monthly"], seed.get("sparkline_grain") or ""
        ),
        format_func=lambda value: "None" if not value else value.title(),
        key=f"ai_studio_report_kpi_sparkline_{key}",
        help=config_help.field_help("report.sparkline_grain"),
    )
    return {
        "comparison": comparison,
        "comparison_period": period,
        "sparkline_grain": sparkline or None,
        "sparkline_points": int(seed.get("sparkline_points") or 30),
        "target": seed.get("target"),
    }


def _normalize_report_filter_rows(value: Any) -> list[dict[str, Any]]:
    rows = value.to_dict(orient="records") if hasattr(value, "to_dict") else value
    return [
        {
            "field": str(row.get("field", "")).strip(),
            "label": str(row.get("label", "") or "").strip(),
            "display": str(row.get("display") or "secondary"),
            "scope": str(row.get("scope") or "compatible_tiles"),
            "control": str(row.get("control") or "multiselect"),
        }
        for row in rows or []
        if isinstance(row, dict) and str(row.get("field", "")).strip()
    ]


def _chat_review() -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Chat Review")
    st.caption(
        "The Chat page queries the active workspace after this draft is applied. "
        "Use this step to check that the metrics exposed to chat are coherent."
    )
    metrics = sorted(draft.get("metrics", {}).get("metrics", {}), key=str.casefold)
    processors = {
        processor.get("id"): processor
        for processor in draft.get("processors", {}).get("processors", [])
        if isinstance(processor, dict)
    }
    rows = []
    for name, metric_def in draft.get("metrics", {}).get("metrics", {}).items():
        processor = (
            processors.get(metric_def.get("source")) if isinstance(metric_def, dict) else None
        )
        rows.append(
            {
                "Metric": name,
                "Kind": metric_def.get("kind", "") if isinstance(metric_def, dict) else "",
                "Processor": metric_def.get("source", "") if isinstance(metric_def, dict) else "",
                "Group By": ", ".join(processor.get("dimensions", processor.get("group_by", [])))
                if isinstance(processor, dict)
                else "",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch", height=360)
    st.info(
        "After applying the draft, open the Chat With Data page to ask questions over these "
        f"{len(metrics)} aggregate metric(s)."
    )


def _settings_review() -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Settings Review")
    st.caption("Review workspace defaults and dashboard theme before final export.")
    updated = copy.deepcopy(draft)
    pipelines = updated.setdefault("pipelines", {})
    defaults = pipelines.setdefault("defaults", {})
    calendar = defaults.setdefault("calendar", {})
    dashboards = updated.setdefault("dashboards", {})
    with components.bordered_panel("Workspace Defaults", ""):
        pipelines["workspace"] = st.text_input(
            "Workspace Name",
            value=str(pipelines.get("workspace", "studio")),
            key="ai_studio_settings_workspace",
            help=config_help.field_help("workspace.name"),
        )
        defaults["time_zone"] = st.text_input(
            "Time Zone",
            value=str(defaults.get("time_zone", "UTC")),
            key="ai_studio_settings_time_zone",
            help=config_help.field_help("workspace.time_zone"),
        )
        calendar["grains"] = st.multiselect(
            "Calendar Grains",
            ["Day", "Month", "Quarter", "Year", "Summary"],
            default=[
                grain
                for grain in (
                    calendar.get("grains") or ["Day", "Month", "Quarter", "Year", "Summary"]
                )
                if grain in {"Day", "Month", "Quarter", "Year", "Summary"}
            ],
            key="ai_studio_settings_grains",
            help=config_help.field_help("workspace.calendar_grains"),
        )
        calendar["week_start"] = st.selectbox(
            "Week Start",
            ["monday", "sunday"],
            index=builder.option_index(
                ["monday", "sunday"], str(calendar.get("week_start", "monday"))
            ),
            key="ai_studio_settings_week_start",
            help=config_help.field_help("workspace.week_start"),
        )
    with components.bordered_panel("Dashboard Theme", ""):
        theme_text = yaml.safe_dump(dashboards.get("theme", {}), sort_keys=False)
        components.sync_text_area("ai_studio_settings_theme_yaml", theme_text)
        raw_theme = st.text_area(
            "Theme YAML",
            key="ai_studio_settings_theme_yaml",
            height=180,
            help=config_help.field_help("workspace.theme_yaml"),
        )
        try:
            theme = yaml.safe_load(raw_theme) or {}
            if not isinstance(theme, dict):
                raise ValueError("theme must be a YAML mapping")
            dashboards["theme"] = theme
        except Exception as exc:
            _log_ai_operation_failure("Dashboard theme YAML parse", exc)
            st.warning(str(exc))
    if st.button("Update Settings In Draft", type="primary"):
        _set_draft(updated)
        st.success("Draft settings updated.")


def _render_pending_draft_review() -> None:
    if not _ai_calls_enabled():
        return
    pending = st.session_state.get("ai_studio_pending_draft")
    if pending is None:
        return
    base = st.session_state.get("ai_studio_pending_base_draft") or {}
    kind = st.session_state.get("ai_studio_pending_kind") or "draft"
    patches = draft_patches(base, pending)
    signature = _pending_review_signature(base, pending, kind)
    st.write(f"### Review pending AI {kind}")
    st.caption("Each card is independent. Reject keeps the accepted draft's previous definition.")
    if not patches:
        st.info("The AI response does not change the accepted draft.")
    accepted: list[str] = []
    for index, patch in enumerate(patches):
        patch_label = _draft_patch_label(patch.section)
        with st.container(border=True):
            st.write(f"**{patch.change.title()} {patch_label}: `{patch.object_id}`**")
            keep = st.checkbox(
                f"Accept {patch_label} patch",
                value=True,
                key=f"ai_studio_patch_{signature}_{index}",
                help=config_help.field_help("ai.patch_accept"),
            )
            if keep:
                accepted.append(patch.key)
            with st.expander("Inspect patch", expanded=False):
                st.caption("Before")
                st.code(
                    yaml.safe_dump(patch.before, sort_keys=False)
                    if patch.before is not None
                    else "null",
                    language="yaml",
                )
                st.caption("After")
                st.code(
                    yaml.safe_dump(patch.after, sort_keys=False)
                    if patch.after is not None
                    else "null",
                    language="yaml",
                )

    reviewed = merge_selected_draft_patches(base, pending, accepted)
    ok, issues = validate_draft_catalog(reviewed)
    _render_draft_validation(ok, issues, expanded=not ok)
    if issues:
        st.warning(
            "This patch selection is not internally consistent. Adjust the cards or accept "
            "it for repair before publishing."
        )
    with st.container(horizontal=True):
        if st.button("Accept patches", type="primary", disabled=not patches):
            _set_draft(reviewed)
            _queue_preprocessing_editor_sync(patches, accepted)
            _clear_pending_ai_draft()
            if not ok:
                st.session_state["ai_studio_next_step"] = STEPS[7]
            st.rerun()
        if st.button("Discard all"):
            _clear_pending_ai_draft()
            st.rerun()
        with st.popover("Prompt and response", icon=":material/article:"):
            st.caption("Prompt")
            st.code(st.session_state.get("ai_studio_pending_prompt", ""), language="text")
            st.caption("Response")
            st.code(st.session_state.get("ai_studio_last_ai_response", ""), language="json")


def _pending_review_signature(base: dict[str, Any], pending: dict[str, Any], kind: str) -> str:
    payload = {"kind": kind, "base": base, "pending": pending}
    return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _draft_patch_label(section: str) -> str:
    return {
        "source_defaults": "source default",
        "source_filters": "source filter",
        "calculated_fields": "calculated field",
    }.get(section, section)


def _queue_preprocessing_editor_sync(patches: list[DraftPatch], accepted: list[str]) -> None:
    accepted_keys = set(accepted)
    queued: dict[str, set[str]] = {}
    queued_raw = st.session_state.get(_PREPROCESSING_SYNC_STATE_KEY) or {}
    if isinstance(queued_raw, dict):
        for source_id, sections in queued_raw.items():
            queued[str(source_id)] = {str(section) for section in sections}
    elif isinstance(queued_raw, list):
        for source_id in queued_raw:
            queued[str(source_id)] = {"source_defaults", "calculated_fields"}
    for patch in patches:
        if patch.key not in accepted_keys or patch.section not in _PREPROCESSING_PATCH_SECTIONS:
            continue
        source_id = (
            patch.object_id
            if patch.section == "source_filters"
            else patch.object_id.partition("/")[0]
        )
        if source_id:
            queued.setdefault(source_id, set()).add(patch.section)
    if queued:
        st.session_state[_PREPROCESSING_SYNC_STATE_KEY] = {
            source_id: sorted(sections, key=str.casefold)
            for source_id, sections in sorted(queued.items())
        }


def _render_ai_repair_panel(
    draft: dict[str, Any],
    working: pl.DataFrame | None,
    approved_fields: list[str],
    issues: list[str],
) -> None:
    if not _ai_calls_enabled():
        return
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    if not issues:
        return
    ai_settings = _current_ai_settings()
    sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
    schema_preview = _schema_preview_for_ai(working, approved_fields) if working is not None else []
    hidden_fields = (
        sorted(set(working.columns) - set(approved_fields), key=str.casefold)
        if working is not None
        else []
    )
    prompt = prompt_for_repair(
        file_name=_sample_file_name(),
        approved_schema=schema_preview,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=draft,
        validation_issues=issues,
        validation_trace=validation_trace_for_repair(draft),
    )
    with st.container(border=True):
        st.write("### AI Repair")
        st.caption("Ask AI to repair invalid processor, metric, or dashboard YAML.")
        cols = st.columns(2)
        cols[0].metric("Validation Issues", len(issues))
        cols[1].metric("AI Available", "Yes" if ai_settings else "No")
        action_col1, action_col2 = st.columns([0.32, 0.68], vertical_alignment="center")
        if action_col1.button(
            "Generate AI Repair",
            type="primary",
            disabled=ai_settings is None or not sharing_confirmed,
            help=(
                "Configure a LiteLLM model in the sidebar to enable AI repair."
                if ai_settings is None
                else "Confirm the AI data-sharing scope above to enable AI repair."
                if not sharing_confirmed
                else "Generate a repair with the confirmed sharing scope."
            ),
        ):
            try:
                with st.status("Generating repair", expanded=True) as status:
                    response = _call_litellm_for_current_sample(
                        ai_settings,
                        prompt,
                        approved_fields=approved_fields,
                    )
                    sections = parse_ai_yaml_sections(response)
                    pending = merge_draft_sections(draft, sections)
                    st.session_state["ai_studio_pending_draft"] = pending
                    st.session_state["ai_studio_pending_base_draft"] = draft
                    st.session_state["ai_studio_pending_kind"] = "repair"
                    st.session_state["ai_studio_pending_prompt"] = prompt
                    st.session_state["ai_studio_last_ai_response"] = response
                    status.update(label="Repair ready for review", state="complete")
                st.rerun()
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _log_ai_operation_failure("AI repair", exc)
                st.error(f"AI repair failed: {exc}")
        with action_col2, st.popover("Show Repair Prompt", icon=":material/build:"):
            st.code(prompt, language="text")


@st.fragment()
def _render_copilot_panel(
    step: str,
    working: pl.DataFrame,
    approved_fields: list[str],
) -> None:
    if not _ai_calls_enabled():
        return
    history = st.session_state.setdefault("ai_studio_copilot_history", [])
    has_pending = st.session_state.get("ai_studio_pending_draft") is not None
    sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
    queued = st.session_state.get("ai_studio_copilot_queued_message")
    st.write("## AI Copilot")
    st.caption(
        "Ask about the active step or request a draft change. Every valid change waits "
        "for explicit patch review."
    )
    if has_pending:
        _render_pending_draft_review()
    if history:
        with st.container(height=320, border=False):
            for message in history[-_COPILOT_HISTORY_DISPLAY:]:
                role = str(message.get("role") or "user")
                with st.chat_message("assistant" if role == "assistant" else "user"):
                    st.markdown(str(message.get("content") or ""))
    if not has_pending:
        _render_copilot_questions()
    else:
        st.caption("Accept or discard the pending patches before sending another message.")
    message_text = st.chat_input(
        "Describe a change or ask about this step",
        key="ai_studio_copilot_input",
        disabled=has_pending or not sharing_confirmed,
        submit_mode="disable",
    )
    if not sharing_confirmed:
        st.caption("Confirm the current sample's AI data-sharing scope before using Copilot.")
    last_prompt = str(st.session_state.get("ai_studio_copilot_last_prompt") or "")
    if last_prompt:
        with st.popover("Last prompt", icon=":material/psychology:"):
            st.code(last_prompt, language="text")
    prompt_text = ""
    if not has_pending and queued:
        prompt_text = str(
            st.session_state.pop("ai_studio_copilot_queued_message", "") or ""
        ).strip()
    elif not has_pending and message_text:
        prompt_text = str(message_text).strip()
    if prompt_text:
        _handle_copilot_message(prompt_text, step, working, approved_fields)


def _render_copilot_questions() -> None:
    questions = st.session_state.get("ai_studio_copilot_questions") or []
    for question_index, question in enumerate(questions):
        text = str(question.get("question") or "").strip()
        if not text:
            continue
        st.markdown(text)
        options = [
            str(option).strip() for option in (question.get("options") or []) if str(option).strip()
        ]
        if not options:
            continue
        columns = st.columns(min(len(options), 4))
        for option_index, option in enumerate(options[:4]):
            if columns[option_index].button(
                option,
                key=f"ai_studio_copilot_option_{question_index}_{option_index}",
            ):
                st.session_state["ai_studio_copilot_queued_message"] = option
                st.session_state["ai_studio_copilot_questions"] = []
                _rerun_copilot_fragment()


def _handle_copilot_message(
    message: str,
    step: str,
    working: pl.DataFrame,
    approved_fields: list[str],
) -> None:
    if st.session_state.get("ai_studio_pending_draft") is not None:
        st.info("Review the pending patches before asking the copilot for another change.")
        return
    ai_settings = _current_ai_settings()
    if ai_settings is None:
        st.info("Configure a LiteLLM model in the sidebar to use the copilot.")
        return
    if not _ai_data_sharing_confirmed(approved_fields):
        st.info("Confirm the current sample's AI data-sharing scope before using the copilot.")
        return
    history = st.session_state.setdefault("ai_studio_copilot_history", [])
    accepted_draft = st.session_state.get("ai_studio_draft")
    draft = (
        accepted_draft
        if isinstance(accepted_draft, dict)
        else _build_draft_catalog(working, approved_fields)
    )
    schema_preview = _schema_preview_for_ai(working, approved_fields)
    hidden_fields = sorted(set(working.columns) - set(approved_fields), key=str.casefold)
    prompt = prompt_for_copilot(
        step=step,
        user_message=message,
        history=history,
        user_goals=_current_user_goals(),
        approved_schema=schema_preview,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=draft,
    )
    st.session_state["ai_studio_copilot_last_prompt"] = prompt
    try:
        with st.status("Running governed draft tools", expanded=False) as status:
            result = run_copilot_tool_loop(
                prompt=prompt,
                draft=draft,
                call_model=lambda iteration_prompt: _call_litellm_for_current_sample(
                    ai_settings,
                    iteration_prompt,
                    approved_fields=approved_fields,
                ),
                validate=validate_draft_catalog,
                max_iterations=3,
                operation_policy=_copilot_operation_policy(step),
                hidden_fields=hidden_fields,
            )
            status.update(
                label=f"Copilot finished after {result.iterations} iteration(s)",
                state="complete" if not result.validation_issues else "error",
            )
    except Exception as exc:  # pragma: no cover - Streamlit display path
        _log_ai_operation_failure("Copilot request", exc)
        st.error(_copilot_request_error_message(exc))
        return
    history.append({"role": "user", "content": message})
    reply_lines = [result.turn.reply]
    if result.pending_draft is not None and draft_patches(draft, result.pending_draft):
        st.session_state["ai_studio_pending_draft"] = result.pending_draft
        st.session_state["ai_studio_pending_base_draft"] = draft
        st.session_state["ai_studio_pending_kind"] = "copilot"
        st.session_state["ai_studio_pending_prompt"] = prompt
        st.session_state["ai_studio_last_ai_response"] = "\n\n".join(result.responses)
        reply_lines.append("")
        reply_lines.extend(f"- {summary}" for summary in result.summaries)
        reply_lines.append("")
        reply_lines.append("The validated changes are waiting in patch review.")
    elif result.validation_issues:
        reply_lines.append("")
        reply_lines.extend(f"- {issue}" for issue in result.validation_issues)
    st.session_state["ai_studio_copilot_questions"] = [
        {"question": question.question, "options": list(question.options)}
        for question in result.turn.questions
    ]
    history.append({"role": "assistant", "content": "\n".join(reply_lines)})
    if result.pending_draft is not None:
        st.rerun()
    _rerun_copilot_fragment()


def _copilot_request_error_message(exc: Exception) -> str:
    error = str(exc).strip()
    if "insufficient permissions" in error.casefold():
        model_name = str(st.session_state.get("ai_studio_ai_model") or "the selected model")
        return (
            f"The provider denied access to `{model_name}` for the current API project/key. "
            "Choose a model available to that project in **AI Settings**, or grant the project "
            "model usage and the key write permission. No draft operations were applied."
        )
    return f"Copilot request failed: {error or type(exc).__name__}"


def _copilot_operation_policy(step: str) -> dict[str, str]:
    step_name = step.split(". ", 1)[-1]
    if step_name != "Filters":
        return {}
    message = (
        "The Filters step edits the source pipeline before processor fan-out. "
        "Use set_source_filter or remove_source_filter; do not edit a processor."
    )
    return {"set_processor": message, "remove_processor": message}


def _rerun_copilot_fragment() -> None:
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        # AppTest and the first full-page render are not fragment reruns yet.
        st.rerun()


def _render_coverage_panel(
    draft: dict[str, Any],
    working: pl.DataFrame,
    approved_fields: list[str],
) -> None:
    if not _ai_calls_enabled():
        return
    with st.container(border=True):
        st.write("### Requirements Coverage")
        goals = _current_user_goals()
        if not goals:
            st.caption(
                "Add business requirements on the Sample or AI Draft step to check coverage."
            )
            return
        st.caption("Ask AI to map each business requirement to the metrics and tiles covering it.")
        ai_settings = _current_ai_settings()
        sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
        hidden_fields = sorted(set(working.columns) - set(approved_fields), key=str.casefold)
        prompt = prompt_for_coverage(
            user_goals=goals,
            draft=draft,
            hidden_fields=hidden_fields,
        )
        signature = _coverage_signature(goals, draft)
        action_col, prompt_col = st.columns([0.32, 0.68], vertical_alignment="center")
        if (
            action_col.button(
                "Check Coverage",
                key="ai_studio_coverage_check",
                type="primary",
                disabled=ai_settings is None or not sharing_confirmed,
                help=(
                    "Configure a LiteLLM model in the sidebar to check coverage."
                    if ai_settings is None
                    else "Confirm the AI data-sharing scope above to check coverage."
                    if not sharing_confirmed
                    else "Check coverage with the confirmed sharing scope."
                ),
            )
            and ai_settings is not None
        ):
            try:
                with st.status("Checking requirements coverage", expanded=False):
                    response = _call_litellm_for_current_sample(
                        ai_settings,
                        prompt,
                        approved_fields=approved_fields,
                    )
                    coverage = parse_coverage_response(response, draft=draft)
                st.session_state["ai_studio_coverage_rows"] = [
                    {
                        "requirement": row.requirement,
                        "status": row.status,
                        "metrics": list(row.metrics),
                        "tiles": list(row.tiles),
                        "note": row.note,
                    }
                    for row in coverage
                ]
                st.session_state["ai_studio_coverage_signature"] = signature
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _log_ai_operation_failure("Coverage check", exc)
                st.error(f"Coverage check failed: {exc}")
        with prompt_col, st.popover("Show Prompt", icon=":material/psychology:"):
            st.code(prompt, language="text")
        stored_rows: list[dict[str, Any]] = st.session_state.get("ai_studio_coverage_rows") or []
        if not stored_rows:
            return
        if st.session_state.get("ai_studio_coverage_signature") != signature:
            st.warning("The draft or requirements changed since the last coverage check.")
        counts = Counter(str(row.get("status")) for row in stored_rows)
        components.metric_strip(
            [
                {"label": "Covered", "value": counts.get("covered", 0)},
                {"label": "Partial", "value": counts.get("partial", 0)},
                {"label": "Missing", "value": counts.get("missing", 0)},
            ]
        )
        display_rows: list[dict[str, Any]] = [
            {
                "Status": str(row.get("status") or ""),
                "Requirement": str(row.get("requirement") or ""),
                "Metrics": ", ".join(row.get("metrics") or []),
                "Tiles": ", ".join(row.get("tiles") or []),
                "Note": str(row.get("note") or ""),
            }
            for row in stored_rows
        ]
        st.dataframe(display_rows, hide_index=True, width="stretch")
        uncovered = [row for row in stored_rows if row.get("status") != "covered"]
        for index, row in enumerate(uncovered[:4]):
            requirement = str(row.get("requirement") or "")
            if st.button(
                f"Ask Copilot To Cover: {requirement[:60]}",
                key=f"ai_studio_coverage_fix_{index}",
            ):
                st.session_state["ai_studio_copilot_queued_message"] = (
                    f"Cover this requirement with concrete draft changes: {requirement}"
                )
                st.rerun()


def _coverage_signature(goals: str, draft: dict[str, Any]) -> str:
    payload = goals + "\n" + yaml.safe_dump(draft, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _draft_signature(draft: dict[str, Any]) -> str:
    payload = yaml.safe_dump(draft, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_user_goals_editor(*, height: int = 120) -> str:
    st.session_state["ai_studio_user_goals"] = st.text_area(
        "Business Requirements",
        value=str(st.session_state.get("ai_studio_user_goals", "") or ""),
        height=height,
        help=config_help.field_help("ai.user_goals"),
        placeholder=(
            "For example: weekly conversion by channel, average revenue per customer, "
            "and an A/B test readout for the latest campaign."
        ),
    )
    return _current_user_goals()


def _current_user_goals() -> str:
    return str(st.session_state.get("ai_studio_user_goals", "") or "").strip()


def _render_ai_refine_panel(
    draft: dict[str, Any],
    working: pl.DataFrame | None,
    approved_fields: list[str],
) -> None:
    if not _ai_calls_enabled():
        return
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    ai_settings = _current_ai_settings()
    sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
    with st.container(border=True):
        st.write("### AI Revision")
        st.caption(
            "Describe a change in free form. Revised sections are held for review "
            "before they can update the draft."
        )
        instruction = st.text_area(
            "Change Request",
            key="ai_studio_refine_instruction",
            height=100,
            help=config_help.field_help("ai.refine_instruction"),
            placeholder=(
                "For example: add weekly revenue by channel and a KPI card for total orders."
            ),
        )
        instruction = (instruction or "").strip()
        schema_preview = _schema_preview_for_ai(working, approved_fields)
        hidden_fields = (
            sorted(set(working.columns) - set(approved_fields), key=str.casefold)
            if working is not None
            else []
        )
        prompt = prompt_for_draft_refinement(
            file_name=_sample_file_name(),
            approved_schema=schema_preview,
            approved_fields=approved_fields,
            hidden_fields=hidden_fields,
            current_draft=draft,
            instruction=instruction,
            user_goals=_current_user_goals(),
        )
        action_col1, action_col2 = st.columns([0.32, 0.68], vertical_alignment="center")
        if (
            action_col1.button(
                "Generate AI Revision",
                type="primary",
                disabled=ai_settings is None or not instruction or not sharing_confirmed,
                help=(
                    "Configure a LiteLLM model in the sidebar and enter a change request."
                    if ai_settings is None
                    else "Confirm the AI data-sharing scope above before generating a revision."
                    if not sharing_confirmed
                    else "Enter a change request."
                    if not instruction
                    else "Generate a revision with the confirmed sharing scope."
                ),
            )
            and ai_settings is not None
        ):
            try:
                with st.status("Generating revision", expanded=True) as status:
                    response = _call_litellm_for_current_sample(
                        ai_settings,
                        prompt,
                        approved_fields=approved_fields,
                    )
                    sections = parse_ai_yaml_sections(response)
                    pending = merge_draft_sections(draft, sections)
                    st.session_state["ai_studio_pending_draft"] = pending
                    st.session_state["ai_studio_pending_base_draft"] = draft
                    st.session_state["ai_studio_pending_kind"] = "revision"
                    st.session_state["ai_studio_pending_prompt"] = prompt
                    st.session_state["ai_studio_last_ai_response"] = response
                    status.update(label="Revision ready for review", state="complete")
                st.rerun()
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _log_ai_operation_failure("AI revision", exc)
                st.error(f"AI revision failed: {exc}")
        with action_col2, st.popover("Show Prompt", icon=":material/psychology:"):
            st.code(prompt, language="text")


def _current_ai_settings() -> AICallSettings | None:
    model_name = str(st.session_state.get("ai_studio_ai_model") or "").strip()
    if not model_name:
        return None
    temperature = (
        float(st.session_state.get("ai_studio_ai_temperature", 1.0))
        if st.session_state.get("ai_studio_ai_temperature_enabled")
        else None
    )
    return AICallSettings(
        model=model_name,
        api_key=str(st.session_state.get("ai_studio_api_key") or "").strip(),
        api_base=str(st.session_state.get("ai_studio_ai_api_base") or "").strip(),
        custom_llm_provider=str(st.session_state.get("ai_studio_ai_provider") or "").strip(),
        temperature=temperature,
        reasoning_effort=str(st.session_state.get("ai_studio_ai_reasoning_effort") or "").strip(),
        verbosity=str(st.session_state.get("ai_studio_ai_verbosity") or "").strip(),
        timeout_seconds=int(st.session_state.get("ai_studio_ai_timeout_seconds", 90)),
    )


def _sample_file_name() -> str:
    return str(st.session_state.get("ai_studio_sample_name") or "source_sample")


def _schema_preview_for_ai(
    working: pl.DataFrame | None, approved_fields: list[str]
) -> list[dict[str, Any]]:
    if working is None:
        return []
    example_fields = st.session_state.get("ai_studio_example_fields") or []
    return generate_schema_preview(
        working,
        approved_fields=approved_fields,
        example_fields=[field for field in example_fields if field in approved_fields],
    )


def _schema_preview_display_rows(schema_preview: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in schema_preview:
        display_row = {
            "column": str(row.get("column", "") or ""),
            "dtype": str(row.get("dtype", "") or ""),
            "nulls": int(row.get("nulls", 0) or 0),
            "unique": int(row.get("unique", 0) or 0),
            "examples": _examples_display_text(row.get("examples")),
        }
        rows.append(display_row)
    return rows


def _examples_display_text(examples: Any) -> str:
    if examples is None:
        return ""
    if isinstance(examples, list):
        return "[" + ", ".join(_sample_display_value(value) for value in examples) + "]"
    return _sample_display_value(examples)


def _ai_sharing_contract(approved_fields: list[str] | None = None) -> dict[str, Any]:
    approved = (
        st.session_state.get("ai_studio_approved_fields") or []
        if approved_fields is None
        else approved_fields
    )
    approved = sorted({str(field) for field in approved if str(field)}, key=str.casefold)
    example_fields = sorted(
        {
            str(field)
            for field in st.session_state.get("ai_studio_example_fields", [])
            if str(field) in approved
        },
        key=str.casefold,
    )
    model_name = str(st.session_state.get("ai_studio_ai_model") or "").strip()
    custom_provider = str(st.session_state.get("ai_studio_ai_provider") or "").strip()
    inferred_provider = model_name.partition("/")[0] if "/" in model_name else ""
    api_base = str(st.session_state.get("ai_studio_ai_api_base") or "").strip()
    return {
        "sample_identity": str(st.session_state.get("ai_studio_sample_identity") or ""),
        "sample_name": _sample_file_name(),
        "model": model_name,
        "provider": custom_provider or inferred_provider or "Configured model provider",
        "destination": "Custom endpoint configured" if api_base else "Provider default endpoint",
        "endpoint_fingerprint": (
            hashlib.sha256(api_base.encode("utf-8")).hexdigest() if api_base else ""
        ),
        "approved_fields": approved,
        "example_fields": example_fields,
    }


def _ai_sharing_signature(approved_fields: list[str] | None = None) -> str:
    payload = json.dumps(
        _ai_sharing_contract(approved_fields),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_sharing_confirmation_required() -> bool:
    return bool(st.session_state.get("ai_studio_sample_identity"))


def _ai_data_sharing_confirmed(approved_fields: list[str] | None = None) -> bool:
    if not _sample_sharing_confirmation_required():
        return True
    return st.session_state.get(AI_SHARING_CONFIRMATION_STATE_KEY) == _ai_sharing_signature(
        approved_fields
    )


def _clear_ai_sharing_confirmation() -> None:
    """Clear approval widgets and sample/provider-scoped Copilot context."""

    st.session_state[AI_SHARING_CONFIRMATION_STATE_KEY] = ""
    st.session_state[AI_SHARING_CONTRACT_STATE_KEY] = ""
    for key in list(st.session_state):
        if str(key).startswith(AI_SHARING_CONFIRMATION_WIDGET_PREFIX):
            st.session_state.pop(key, None)
    st.session_state["ai_studio_copilot_history"] = []
    st.session_state["ai_studio_copilot_questions"] = []
    st.session_state["ai_studio_copilot_last_prompt"] = ""
    st.session_state.pop("ai_studio_copilot_queued_message", None)


def _render_ai_data_sharing_confirmation(approved_fields: list[str]) -> None:
    """Require an explicit, sample-scoped review before any model call."""
    if not _sample_sharing_confirmation_required():
        return
    contract = _ai_sharing_contract(approved_fields)
    signature = _ai_sharing_signature(approved_fields)
    stored_signature = str(st.session_state.get(AI_SHARING_CONFIRMATION_STATE_KEY) or "")
    previous_contract_signature = str(st.session_state.get(AI_SHARING_CONTRACT_STATE_KEY) or "")
    confirmed_scope_changed = bool(stored_signature and stored_signature != signature)
    rendered_scope_changed = bool(
        previous_contract_signature and previous_contract_signature != signature
    )
    if confirmed_scope_changed or rendered_scope_changed:
        # Confirmation is single-use for the current sharing contract. Once a
        # model/provider/field scope changes, returning to an older scope must
        # still require a new review, and prior Copilot text cannot cross scopes.
        _clear_ai_sharing_confirmation()
    st.session_state[AI_SHARING_CONTRACT_STATE_KEY] = signature
    example_fields = contract["example_fields"]
    likely_identifiers = [field for field in example_fields if _looks_like_id(field)]
    already_confirmed = _ai_data_sharing_confirmed(approved_fields)
    with st.container(border=True):
        st.write("### Review data sent to AI")
        st.caption(
            "Confirm this sharing scope for the current sample before any AI action can run. "
            "Changing the sample, model, provider, approved fields, or example sharing requires "
            "a new confirmation. Prompts also include your business requirements and the "
            "relevant deterministic catalog or current draft settings."
        )
        summary_cols = st.columns(4)
        summary_cols[0].metric("Provider", contract["provider"])
        summary_cols[1].metric("Model", contract["model"] or "Not configured")
        summary_cols[2].metric("Schema Fields", len(contract["approved_fields"]))
        summary_cols[3].metric("Fields With Examples", len(example_fields))
        st.caption(f"Destination: **{contract['destination']}**")
        if example_fields:
            st.warning("Sample values will be included for: " + ", ".join(example_fields) + ".")
        else:
            st.info(
                "No sample values will be sent. Approved field names, types, null counts, and "
                "unique counts, plus business requirements and relevant catalog or draft "
                "settings, are still included."
            )
        if likely_identifiers:
            st.warning(
                "Likely identifier fields are selected for sample-value sharing: "
                + ", ".join(likely_identifiers)
                + ". Verify that your provider is allowed to receive them."
            )
        with st.expander("Sharing scope", expanded=False):
            st.write(f"Sample: `{contract['sample_name']}`")
            st.write(
                "Approved schema fields: " + (", ".join(contract["approved_fields"]) or "none")
            )
            st.write("Fields with sample values: " + (", ".join(example_fields) or "none"))
            st.write(f"Destination: {contract['destination']}")
            st.write("Also included: business requirements and relevant catalog or draft settings")
            st.caption("Provider storage and retention follow your configured provider terms.")
        confirmed = st.checkbox(
            "I confirm the sharing scope above may be sent to the provider and model shown.",
            value=already_confirmed,
            key=f"{AI_SHARING_CONFIRMATION_WIDGET_PREFIX}{signature[:16]}",
            help=(
                "Required once for the current sample and sharing scope. Changing the sample, "
                "provider, model, approved schema, or example fields requires confirmation again."
            ),
        )
        if confirmed:
            st.session_state[AI_SHARING_CONFIRMATION_STATE_KEY] = signature
        elif already_confirmed:
            # The current checkbox has already been instantiated in this run;
            # clear only the approval marker and leave its now-false widget state.
            st.session_state[AI_SHARING_CONFIRMATION_STATE_KEY] = ""


def _call_litellm_for_current_sample(
    settings: AICallSettings,
    prompt: str,
    *,
    approved_fields: list[str] | None = None,
    **kwargs: Any,
) -> str:
    if not _ai_data_sharing_confirmed(approved_fields):
        raise PermissionError(
            "Review and confirm the current sample's AI data-sharing scope before continuing."
        )
    return call_litellm(settings, prompt, **kwargs)


def _log_ai_operation_failure(operation: str, exc: Exception) -> None:
    """Log Studio failures without sample, model, config, or exception payloads."""

    error_type = type(exc).__name__
    if len(error_type) > 64 or not error_type.isascii() or not error_type.isidentifier():
        error_type = "ApplicationError"
    logger.error("%s failed: error_type=%s", operation, error_type)


def _ai_privacy_summary(working: pl.DataFrame, approved_fields: list[str], prompt: str) -> None:
    example_fields = [
        field
        for field in st.session_state.get("ai_studio_example_fields", [])
        if field in approved_fields
    ]
    hidden = sorted(set(working.columns) - set(approved_fields), key=str.casefold)
    estimated_tokens = max(1, (len(prompt) + 3) // 4)
    components.metric_strip(
        [
            {"label": "Fields Sent", "value": len(approved_fields)},
            {"label": "Fields Hidden", "value": len(hidden)},
            {"label": "Fields With Examples", "value": len(example_fields)},
            {"label": "Prompt Size", "value": f"~{estimated_tokens:,} tokens"},
        ]
    )
    with st.expander("AI Sharing Details", expanded=False):
        st.write("Fields sent: " + (", ".join(approved_fields) if approved_fields else "none"))
        st.write("Fields hidden: " + (", ".join(hidden) if hidden else "none"))
        st.write(
            "Fields with examples: " + (", ".join(example_fields) if example_fields else "none")
        )


def _draft_counts(draft: dict[str, Any]) -> None:
    components.metric_strip(
        [{"label": label, "value": value} for label, value in draft_object_counts(draft).items()],
        columns=5,
    )


def _set_draft(draft: dict[str, Any]) -> None:
    st.session_state["ai_studio_draft"] = yaml.safe_load(yaml.safe_dump(draft, sort_keys=False))
    st.session_state.pop("ai_studio_raw_metrics_yaml_signature", None)
    st.session_state.pop("ai_studio_raw_dashboards_yaml_signature", None)
    st.session_state.pop("ai_studio_settings_theme_yaml_signature", None)


def _clear_pending_ai_draft() -> None:
    st.session_state["ai_studio_pending_draft"] = None
    st.session_state["ai_studio_pending_base_draft"] = None
    st.session_state["ai_studio_pending_kind"] = ""
    st.session_state["ai_studio_pending_prompt"] = ""
    st.session_state["ai_studio_last_ai_response"] = ""


def _current_or_deterministic_draft(
    working: pl.DataFrame,
    approved_fields: list[str],
) -> dict[str, Any]:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        draft = _build_draft_catalog(working, approved_fields)
        _set_draft(draft)
    return draft


def _render_draft_validation(ok: bool, issues: list[str], *, expanded: bool) -> None:
    with st.container(border=True):
        st.write("### Draft Validation")
        blocking_issues, repairable_issues = classify_draft_validation_issues(issues)
        status = "OK"
        if not ok:
            status = (
                "Needs repair" if repairable_issues and not blocking_issues else "Needs attention"
            )
        cols = st.columns(2)
        cols[0].metric("Status", status)
        cols[1].metric("Issues", len(issues))
        if not issues:
            st.success("Draft catalog validates.")
            return
        with st.expander("Validation Details", expanded=expanded):
            for issue in blocking_issues:
                st.error(issue)
            for issue in repairable_issues:
                st.warning(issue)


def _tile_inventory_rows(draft: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    dashboards = draft.get("dashboards", {}).get("dashboards", [])
    if not isinstance(dashboards, list):
        return rows
    for dashboard in dashboards:
        if not isinstance(dashboard, dict):
            continue
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            for tile in page.get("tiles", []) or []:
                if not isinstance(tile, dict):
                    continue
                rows.append(
                    {
                        "Dashboard": str(dashboard.get("id", "")),
                        "Page": str(page.get("id", "")),
                        "Tile": str(tile.get("id", "")),
                        "Title": str(tile.get("title", "")),
                        "Metric": str(tile.get("metric", "")),
                        "Chart": str(tile.get("chart", "")),
                    }
                )
    return rows


def _schema_sample(sample: pl.DataFrame) -> pl.DataFrame:
    if _rename_capitalize_enabled():
        return _rename_capitalize_frame(sample)
    return sample


def _set_effective_schema_state(sample: pl.DataFrame) -> None:
    columns = list(sample.columns)
    signature = _schema_signature(columns)
    st.session_state["ai_studio_effective_schema_columns"] = columns
    st.session_state["ai_studio_effective_schema_signature"] = signature


def _schema_signature(columns: list[str]) -> str:
    raw = "\x1f".join(columns).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _schema_widget_key(base: str) -> str:
    signature = st.session_state.get("ai_studio_effective_schema_signature", "schema")
    return f"{base}_{signature}"


def _clear_schema_widget_state() -> None:
    schema_widget_bases = (
        "ai_studio_defaults_editor",
        "ai_studio_filter_editor",
        "ai_studio_calculation_editor",
        "ai_studio_schema_preview_table",
        "ai_studio_group_by_field_selector",
        "ai_studio_field_approval_search",
        "ai_studio_field_approval_editor",
    )
    for key in list(st.session_state.keys()):
        if key in schema_widget_bases or any(
            key.startswith(f"{base}_") for base in schema_widget_bases
        ):
            st.session_state.pop(key, None)


def _clear_preprocessing_widget_state(sections: set[str]) -> None:
    section_widget_bases = {
        "source_defaults": (
            "ai_studio_defaults_editor",
            "ai_studio_defaults_field_picker",
        ),
        "source_filters": ("ai_studio_filter_editor",),
        "calculated_fields": ("ai_studio_calculation_editor",),
    }
    widget_bases = tuple(
        base for section in sections for base in section_widget_bases.get(section, ())
    )
    for key in list(st.session_state.keys()):
        if key in widget_bases or any(key.startswith(f"{base}_") for base in widget_bases):
            st.session_state.pop(key, None)


def _consume_preprocessing_editor_sync() -> None:
    queued = _preprocessing_sync_queue()
    active_source_id = str(st.session_state.get("ai_studio_source_id") or "").strip()
    if not queued or active_source_id not in queued:
        return
    sections = queued.pop(active_source_id)
    _store_preprocessing_sync_queue(queued)

    draft = st.session_state.get("ai_studio_draft") or {}
    source_defs = draft.get("pipelines", {}).get("sources", [])
    source_def = next(
        (
            item
            for item in source_defs
            if isinstance(item, dict) and item.get("id") == active_source_id
        ),
        None,
    )
    if source_def is None:
        return
    source = model.Source.model_validate(source_def)
    syncers = {
        "source_defaults": _sync_source_defaults_editor,
        "source_filters": _sync_source_filter_editor,
        "calculated_fields": _sync_calculated_fields_editor,
    }
    for section in sections:
        syncer = syncers.get(section)
        if syncer is not None:
            syncer(source)
    _clear_preprocessing_widget_state(sections)


def _preprocessing_sync_queue() -> dict[str, set[str]]:
    queued_raw = st.session_state.get(_PREPROCESSING_SYNC_STATE_KEY) or {}
    if isinstance(queued_raw, list):
        return {
            str(source_id): {"source_defaults", "calculated_fields"}
            for source_id in queued_raw
            if str(source_id).strip()
        }
    if isinstance(queued_raw, dict):
        return {
            str(source_id): {str(section) for section in sections}
            for source_id, sections in queued_raw.items()
        }
    return {}


def _store_preprocessing_sync_queue(queued: dict[str, set[str]]) -> None:
    if not queued:
        st.session_state.pop(_PREPROCESSING_SYNC_STATE_KEY, None)
        return
    st.session_state[_PREPROCESSING_SYNC_STATE_KEY] = {
        source_id: sorted(values, key=str.casefold) for source_id, values in sorted(queued.items())
    }


def _sync_source_defaults_editor(source: model.Source) -> None:
    default_rows = builder.default_rows_from_values(builder.source_defaults(source))
    for row in default_rows:
        if row.get("Default Value") is None:
            row["Default Value"] = "null"
    st.session_state["ai_studio_defaults"] = default_rows


def _sync_source_filter_editor(source: model.Source) -> None:
    expression = builder.first_filter_expression(source)
    filter_rows = builder.filter_rows_from_expression(expression)
    if filter_rows is None:
        st.session_state["ai_studio_filter_mode"] = "Raw AST"
        st.session_state["ai_studio_filter_rows"] = [builder.blank_filter_row()]
        st.session_state["ai_studio_raw_filter"] = builder.expression_yaml(expression)
        return
    st.session_state["ai_studio_filter_mode"] = "Rules"
    st.session_state["ai_studio_filter_rows"] = filter_rows
    st.session_state["ai_studio_raw_filter"] = ""


def _sync_calculated_fields_editor(source: model.Source) -> None:
    st.session_state["ai_studio_calculations"] = builder.calculated_rows_from_source(source)


def _working_sample(sample: pl.DataFrame) -> tuple[pl.DataFrame, str | None]:
    try:
        frame = sample
        defaults = builder.build_default_values(st.session_state.get("ai_studio_defaults", []))
        for column, value in defaults.items():
            if column in frame.columns:
                frame = frame.with_columns(pl.col(column).fill_null(value).alias(column))
            else:
                frame = frame.with_columns(pl.lit(value).alias(column))
        frame = _alias_required_fields(frame)
        frame = _derive_time_fields(frame)
        filter_expression = _current_filter_expression()
        if filter_expression:
            frame = frame.filter(translate(expr_parser.parse(filter_expression)))
        for transform in builder.build_derive_column_transforms(
            st.session_state.get("ai_studio_calculations", [])
        ):
            frame = frame.with_columns(
                translate(expr_parser.parse(transform["expression"])).alias(transform["output"])
            )
        return frame, None
    except Exception as exc:
        _log_ai_operation_failure("Working sample preprocessing", exc)
        return sample, str(exc)


def _rename_capitalize_enabled() -> bool:
    return bool(st.session_state.get("ai_studio_rename_capitalize"))


def _rename_capitalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename(_rename_capitalize_mapping(frame.columns))


def _rename_capitalize_mapping(columns: list[str]) -> dict[str, str]:
    return dict(zip(columns, capitalize_fields(columns), strict=False))


def _sync_ai_rename_capitalize_state(sample: pl.DataFrame) -> None:
    enabled = _rename_capitalize_enabled()
    applied = bool(st.session_state.get("ai_studio_rename_capitalize_applied", False))
    if applied == enabled:
        return

    forward = {
        source: target
        for source, target in _rename_capitalize_mapping(sample.columns).items()
        if source != target
    }
    mapping = forward if enabled else {target: source for source, target in forward.items()}
    if mapping:
        for key in (
            "ai_studio_subject",
            "ai_studio_outcome_column",
            "ai_studio_outcome_time",
            "ai_studio_decision_time",
            "ai_studio_day_column",
            "ai_studio_month_column",
            "ai_studio_quarter_column",
            "ai_studio_year_column",
        ):
            field_remap.remap_state_field(key, mapping)
        field_remap.remap_state_rows("ai_studio_defaults", mapping, ("Field",))
        field_remap.remap_state_rows("ai_studio_filter_rows", mapping, ("Field",))
        field_remap.remap_state_raw_expression("ai_studio_raw_filter", mapping)
        field_remap.remap_state_calculation_rows("ai_studio_calculations", mapping)
        for key in (
            "ai_studio_approved_fields",
            "ai_studio_example_fields",
            "ai_studio_group_by_fields",
            "ai_studio_group_by_field_selector",
        ):
            field_remap.remap_state_field_list(key, mapping)
    _clear_schema_widget_state()
    st.session_state["ai_studio_field_approval_initialized"] = False
    st.session_state["ai_studio_rename_capitalize_applied"] = enabled


def _alias_required_fields(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for target, source in _field_aliases().items():
        if source in frame.columns and source != target:
            expressions.append(pl.col(source).alias(target))
    return frame.with_columns(expressions) if expressions else frame


def _derive_time_fields(frame: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR0912
    fmt = st.session_state.get("ai_studio_timestamp_format", "")
    out = frame
    if "OutcomeTime" in out.columns and out.schema.get("OutcomeTime") == pl.String and fmt:
        out = out.with_columns(pl.col("OutcomeTime").str.strptime(pl.Datetime, fmt, strict=False))
    expressions: list[pl.Expr] = []
    if "OutcomeTime" in out.columns:
        time = pl.col("OutcomeTime")
        if "Day" not in out.columns:
            expressions.append(time.dt.date().alias("Day"))
        if "Month" not in out.columns:
            expressions.append(time.dt.strftime("%Y-%m").alias("Month"))
        if "Year" not in out.columns:
            expressions.append(time.dt.year().cast(pl.Int16).alias("Year"))
        if "Quarter" not in out.columns:
            expressions.append(
                (
                    time.dt.year().cast(pl.String)
                    + pl.lit("_Q")
                    + time.dt.quarter().cast(pl.String)
                ).alias("Quarter")
            )
    elif "Day" in out.columns:
        day = _day_expr(out)
        if "Month" not in out.columns:
            expressions.append(day.dt.strftime("%Y-%m").alias("Month"))
        if "Year" not in out.columns:
            expressions.append(day.dt.year().cast(pl.Int16).alias("Year"))
        if "Quarter" not in out.columns:
            expressions.append(
                (
                    day.dt.year().cast(pl.String) + pl.lit("_Q") + day.dt.quarter().cast(pl.String)
                ).alias("Quarter")
            )
    elif "Month" in out.columns:
        month_text = pl.col("Month").cast(pl.String)
        month_num = month_text.str.slice(5, 2).cast(pl.Int16, strict=False)
        if "Year" not in out.columns:
            expressions.append(
                month_text.str.slice(0, 4).cast(pl.Int16, strict=False).alias("Year")
            )
        if "Quarter" not in out.columns:
            quarter = ((month_num - 1) // 3) + 1
            expressions.append(
                pl.concat_str(
                    [month_text.str.slice(0, 4), pl.lit("_Q"), quarter.cast(pl.String)]
                ).alias("Quarter")
            )
    if "DecisionTime" in out.columns and "OutcomeTime" in out.columns:
        if out.schema.get("DecisionTime") == pl.String and fmt:
            out = out.with_columns(
                pl.col("DecisionTime").str.strptime(pl.Datetime, fmt, strict=False)
            )
        if "ResponseTime" not in out.columns:
            expressions.append(
                (pl.col("OutcomeTime") - pl.col("DecisionTime"))
                .dt.total_seconds()
                .alias("ResponseTime")
            )
    return out.with_columns(expressions) if expressions else out


def _blank_ai_calculation_row() -> dict[str, Any]:
    return {
        "Name": "",
        "Mode": "AST YAML",
        "Left": "",
        "Right Kind": "Field",
        "Right": "",
        "Expression": "",
        "Enabled": True,
    }


def _field_aliases() -> dict[str, str]:
    aliases = {
        "SubjectID": str(st.session_state.get("ai_studio_subject", "") or "").strip(),
        "OutcomeTime": str(st.session_state.get("ai_studio_outcome_time", "") or "").strip(),
        "DecisionTime": str(st.session_state.get("ai_studio_decision_time", "") or "").strip(),
        "Day": str(st.session_state.get("ai_studio_day_column", "") or "").strip(),
        "Month": str(st.session_state.get("ai_studio_month_column", "") or "").strip(),
        "Quarter": str(st.session_state.get("ai_studio_quarter_column", "") or "").strip(),
        "Year": str(st.session_state.get("ai_studio_year_column", "") or "").strip(),
    }
    return {target: source for target, source in aliases.items() if source}


def _calendar_outputs_to_derive() -> list[str]:
    mapped = {
        "Day": st.session_state.get("ai_studio_day_column", ""),
        "Month": st.session_state.get("ai_studio_month_column", ""),
        "Quarter": st.session_state.get("ai_studio_quarter_column", ""),
        "Year": st.session_state.get("ai_studio_year_column", ""),
    }
    return [output for output, source in mapped.items() if not str(source or "").strip()]


def _day_expr(frame: pl.DataFrame) -> pl.Expr:
    if frame.schema.get("Day") == pl.String:
        return pl.col("Day").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    return pl.col("Day").cast(pl.Date, strict=False)


def _field_mapping_rows(sample: pl.DataFrame) -> list[dict[str, str]]:
    aliases = _field_aliases()
    rows: list[dict[str, str]] = []
    for target in ("SubjectID", "OutcomeTime", "DecisionTime", "Day", "Month", "Quarter", "Year"):
        source = aliases.get(target, "")
        rows.append(
            {
                "working_field": target,
                "source_field": source,
                "sample_status": "found" if source in sample.columns else "manual/derived",
            }
        )
    return rows


def _studio_required_fields(sample: pl.DataFrame) -> list[str]:
    required = ["SubjectID"]
    if "OutcomeTime" in sample.columns:
        required.append("OutcomeTime")
    for field in ("Day", "Month", "Quarter", "Year"):
        if field in sample.columns:
            required.append(field)
    outcome = str(st.session_state.get("ai_studio_outcome_column", "") or "").strip()
    if outcome and outcome in sample.columns and outcome not in required:
        required.append(outcome)
    return required


def _primary_time_field(working: pl.DataFrame) -> str:
    for candidate in ("OutcomeTime", "Day", "Month", "Quarter", "Year"):
        if candidate in working.columns:
            return candidate
    return ""


def _processor_grains(working: pl.DataFrame) -> list[str]:
    grains = [grain for grain in ("Day", "Month", "Quarter", "Year") if grain in working.columns]
    return [*grains, "Summary"] if grains else ["Summary"]


def _primary_chart_x(working: pl.DataFrame, group_by: list[str]) -> str:
    for candidate in ("Day", "Month", "Quarter", "Year"):
        if candidate in working.columns:
            return candidate
    return group_by[0] if group_by else "period"


def _current_filter_expression() -> dict[str, Any] | None:
    if st.session_state.get("ai_studio_filter_mode") == "Raw AST":
        raw = st.session_state.get("ai_studio_raw_filter", "")
        return builder.parse_expression_yaml(raw) if raw.strip() else None
    return builder.compile_filter_rows(st.session_state.get("ai_studio_filter_rows", []))


def _build_draft_catalog(working: pl.DataFrame, approved_fields: list[str]) -> dict[str, Any]:
    source_id = str(st.session_state.get("ai_studio_source_id") or "ih").strip() or "ih"
    subject = (
        "SubjectID"
        if "SubjectID" in working.columns
        else st.session_state.get("ai_studio_subject", "")
    )
    time_column = _primary_time_field(working)
    outcome_column = st.session_state.get("ai_studio_outcome_column", "Outcome")
    group_by = st.session_state.get("ai_studio_group_by_fields") or _default_group_by_fields(
        working,
        approved_fields,
    )
    group_by = [field for field in group_by if field in approved_fields]
    grains = _processor_grains(working)
    primary_x = _primary_chart_x(working, group_by)
    processor_id = _generated_processor_id(source_id)
    reader: dict[str, Any] = {
        "kind": st.session_state.get("ai_studio_reader_kind", "pega_ds_export"),
        "file_pattern": st.session_state.get("ai_studio_file_pattern", "**/*.zip"),
        "streaming": bool(st.session_state.get("ai_studio_streaming", True)),
    }
    group_pattern = st.session_state.get("ai_studio_group_pattern", "")
    if group_pattern:
        reader["group_by_filename"] = group_pattern
    if st.session_state.get("ai_studio_hive_partitioning"):
        reader["hive_partitioning"] = True

    default_values = builder.build_default_values(st.session_state.get("ai_studio_defaults", []))
    rename_capitalize = _rename_capitalize_enabled()
    transforms: list[dict[str, Any]] = []
    if rename_capitalize:
        transforms.append({"kind": "rename_capitalize"})
    if rename_capitalize and default_values:
        transforms.append({"kind": "defaults", "values": default_values})
    for target, source_column in _field_aliases().items():
        if source_column and source_column != target:
            transforms.append(
                {
                    "kind": "derive_column",
                    "output": target,
                    "expression": {"col": source_column},
                }
            )
    parse_columns = [
        column
        for column in TIME_TARGETS
        if column in working.columns and st.session_state.get("ai_studio_timestamp_format", "")
    ]
    if parse_columns:
        transforms.append(
            {
                "kind": "parse_datetime",
                "columns": parse_columns,
                "format": st.session_state.get("ai_studio_timestamp_format", ""),
            }
        )
    calendar_outputs = _calendar_outputs_to_derive()
    if "OutcomeTime" in working.columns and calendar_outputs:
        transforms.append(
            {
                "kind": "derive_calendar",
                "from": "OutcomeTime",
                "outputs": calendar_outputs,
            }
        )
    filter_expression = _current_filter_expression()
    if filter_expression:
        transforms.append({"kind": "filter", "expression": filter_expression})
    transforms.extend(
        builder.build_derive_column_transforms(st.session_state.get("ai_studio_calculations", []))
    )
    if {"OutcomeTime", "DecisionTime"} <= set(working.columns):
        transforms.append(
            {
                "kind": "derive_column",
                "output": "ResponseTime",
                "expression": {
                    "op": "date_diff",
                    "unit": "seconds",
                    "end": {"col": "OutcomeTime"},
                    "start": {"col": "DecisionTime"},
                },
            }
        )

    dashboards = {
        "theme": {},
        "dashboards": [
            {
                "id": builder.random_catalog_id("Studio Overview", fallback="dashboard"),
                "title": "Studio Overview",
                "layout": "tabs",
                "pages": [
                    {
                        "id": builder.random_catalog_id("Engagement", fallback="page"),
                        "title": "Engagement",
                        "tiles": [
                            {
                                "id": builder.random_catalog_id("CTR Trend", fallback="tile"),
                                "title": "CTR Trend",
                                "metric": "Studio_CTR",
                                "chart": "line",
                                "x": primary_x,
                                "y": "Studio_CTR",
                                "color": group_by[0] if group_by else "",
                            },
                            {
                                "id": builder.random_catalog_id(
                                    "CTR By Dimension", fallback="tile"
                                ),
                                "title": "CTR By Dimension",
                                "metric": "Studio_CTR",
                                "chart": "bar",
                                "x": group_by[0] if group_by else primary_x,
                                "y": "Studio_CTR",
                            },
                        ],
                    }
                ],
            }
        ],
    }
    return {
        "pipelines": {
            "version": 1,
            "workspace": "studio",
            "defaults": {
                "time_zone": "UTC",
                "calendar": {
                    "grains": ["Day", "Month", "Quarter", "Year", "Summary"],
                    "week_start": "monday",
                },
            },
            "sources": [
                {
                    "id": source_id,
                    "description": "Generated from AI Configuration Studio.",
                    "reader": reader,
                    "schema": {
                        "timestamp_column": time_column or None,
                        "natural_key": _natural_key_fields(subject),
                        "drop_columns": [],
                    },
                    "defaults": {} if rename_capitalize else default_values,
                    "transforms": transforms,
                }
            ],
        },
        "processors": {
            "processors": [
                _without_empty(
                    {
                        "id": processor_id,
                        "source": source_id,
                        "kind": "binary_outcome",
                        "description": "Generated engagement processor.",
                        "dimensions": group_by,
                        "time": {"column": time_column, "grains": grains},
                        "entities": {"subject": subject} if subject else None,
                        "outcome": {
                            "column": outcome_column,
                            "positive_values": ["Clicked", "Conversion"],
                            "negative_values": ["Impression", "Pending"],
                        },
                    }
                )
            ]
        },
        "metrics": {
            "metrics": {
                "Studio_CTR": {
                    "source": processor_id,
                    "kind": "formula",
                    "description": "Click-through rate generated by AI Configuration Studio.",
                    "expression": {
                        "op": "safe_div",
                        "num": {"col": "Positives"},
                        "den": {"op": "add", "args": [{"col": "Positives"}, {"col": "Negatives"}]},
                    },
                },
                "Studio_Count": {
                    "source": processor_id,
                    "kind": "formula",
                    "description": "Total outcome rows generated by AI Configuration Studio.",
                    "expression": {"col": "Count"},
                },
            }
        },
        "dashboards": dashboards,
    }


def _generated_processor_id(source_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in source_id)
    return f"{cleaned or 'ih'}_engagement"


def _draft_files(draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = {
        "pipelines.yaml": draft["pipelines"],
        "processors.yaml": draft["processors"],
        "metrics.yaml": draft["metrics"],
        "dashboards.yaml": draft["dashboards"],
    }
    if isinstance(draft.get("chat_with_data"), dict):
        files["ai.yaml"] = {"chat_with_data": draft["chat_with_data"]}
    return files


def _natural_key_fields(subject: object, aliases: dict[str, str] | None = None) -> list[str]:
    subject_text = str(subject or "").strip()
    source_subject = (aliases or _field_aliases()).get("SubjectID", "").strip()
    natural_key = source_subject or subject_text
    return [natural_key] if natural_key else []


def _without_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "", [], {})}


def _apply_draft(ctx: ValueStreamContext, draft: dict[str, Any]) -> None:
    ok, issues = validate_draft_catalog(draft)
    if not ok:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"AI draft validation failed before apply:\n{details}")

    with builder.workspace_configuration_transaction(ctx.workspace):
        for source_def in draft["pipelines"]["sources"]:
            builder.write_source_definition(ctx.workspace, source_def)
        for processor_def in draft["processors"]["processors"]:
            builder.write_processor_definition(ctx.workspace, processor_def)
        for metric_name, metric_def in draft["metrics"]["metrics"].items():
            builder.write_metric_definition(ctx.workspace, metric_name, metric_def)
        # Persist the complete section once so dashboard theme/layout and page-level
        # filter/time settings survive the draft review/apply round trip.
        builder.write_dashboards_definition(ctx.workspace, draft["dashboards"])
        chat_settings = draft.get("chat_with_data")
        if isinstance(chat_settings, dict):
            write_chat_with_data_config(
                ctx.workspace,
                agent_prompt=str(chat_settings.get("agent_prompt") or ""),
                dataset_descriptions={
                    str(key): str(value)
                    for key, value in dict(chat_settings.get("dataset_descriptions") or {}).items()
                },
                metric_descriptions={
                    str(key): str(value)
                    for key, value in dict(chat_settings.get("metric_descriptions") or {}).items()
                },
            )
        builder.require_valid_workspace(ctx.workspace)


def _apply_draft_and_run_sources(ctx: ValueStreamContext, draft: dict[str, Any]) -> None:
    _apply_draft(ctx, draft)
    _mark_draft_published(draft)
    ok, issues = builder.validate_workspace(ctx.workspace)
    if not ok:
        st.warning("Draft applied, but catalog needs attention before sources can run.")
        st.code("\n".join(issues), language="text")
        return

    source_ids = [
        str(source_def.get("id", "") or "")
        for source_def in draft.get("pipelines", {}).get("sources", [])
        if isinstance(source_def, dict) and source_def.get("id")
    ]
    if not source_ids:
        st.warning("Draft applied and catalog validates, but no generated sources were found.")
        return

    results = []
    with st.status("Running generated sources", expanded=True) as status:
        chunk_progress = components.chunk_progress_indicator(include_source=len(source_ids) > 1)
        for source_id in source_ids:
            status.write(f"Running `{source_id}`...")
            result = run_source(ctx.workspace, source_id, progress_callback=chunk_progress)
            results.append(result)
            status.write(
                f"{source_id}: {result.chunks_ok} ok, {result.chunks_skipped} skipped, {result.chunks_failed} failed."
            )
        failed = [result for result in results if result.status == "failed"]
        state = "error" if failed else "complete"
        status.update(label="Generated source run finished", state=state)

    if any(result.status == "failed" for result in results):
        st.error("Draft applied, but at least one generated source run failed.")
    elif any(result.status == "partial" for result in results):
        st.warning("Draft applied and generated source run finished with partial failures.")
    else:
        st.success("Draft applied and generated source run finished.")


def _mark_draft_published(draft: dict[str, Any]) -> None:
    st.session_state["ai_studio_published_signature"] = _draft_signature(draft)


def _studio_status_bar(
    ctx: ValueStreamContext,
    raw: pl.DataFrame,
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
) -> None:
    """Compact review status: lifecycle badges plus one line of sample counts."""
    ai_calls_enabled = _ai_calls_enabled()
    draft = st.session_state.get("ai_studio_draft")
    pending = st.session_state.get("ai_studio_pending_draft")
    draft_ok, draft_issues = validate_draft_catalog(draft) if draft else (False, [])
    with st.container(border=True):
        status_col, save_col = st.columns([0.84, 0.16], vertical_alignment="center")
        with status_col:
            with st.container(horizontal=True):
                components.status_badge("Sample", "ready" if raw.height else "warning")
                components.status_badge(
                    "Preprocessing",
                    "blocked" if preprocessing_error else "ready",
                    help=preprocessing_error or "Working schema is available.",
                )
                components.status_badge("Field Approval", "ready" if approved_fields else "warning")
                components.status_badge(
                    "AI Draft" if ai_calls_enabled else "Draft",
                    (
                        "warning"
                        if ai_calls_enabled and pending
                        else ("ready" if draft else "pending")
                    ),
                    help=(
                        "Pending AI output needs review." if ai_calls_enabled and pending else None
                    ),
                )
                components.status_badge(
                    "Processors",
                    (
                        "ready"
                        if draft and draft.get("processors", {}).get("processors")
                        else "pending"
                    ),
                )
                components.status_badge(
                    "Metrics",
                    ("ready" if draft and draft.get("metrics", {}).get("metrics") else "pending"),
                )
                components.status_badge(
                    "Reports",
                    "ready" if draft and tile_keys(draft) else "pending",
                )
                components.status_badge(
                    "Export",
                    "ready" if draft_ok else ("blocked" if draft_issues else "pending"),
                    help=draft_issues[0] if draft_issues else None,
                )
            st.caption(
                f"{len(raw.columns)} raw columns · {len(working.columns)} working columns · "
                f"{len(approved_fields)} approved fields · {working.height:,} sample rows"
            )
        with save_col:
            _render_workspace_save_bar(ctx)


def _privacy_summary(
    working: pl.DataFrame,
    selected: list[str],
    example_fields: list[str],
    group_by: list[str],
) -> None:
    hidden = sorted(set(working.columns) - set(selected), key=str.casefold)
    components.metric_strip(
        [
            {"label": "Fields Available", "value": len(selected)},
            {"label": "Fields Hidden", "value": len(hidden)},
            {"label": "Fields With Examples", "value": len(example_fields)},
            {"label": "Grouping Fields", "value": len(group_by)},
        ]
    )
    with st.expander("Field Sharing Details", expanded=False):
        st.write("Approved fields: " + (", ".join(selected) if selected else "none"))
        st.write("Hidden fields: " + (", ".join(hidden) if hidden else "none"))
        st.write(
            "Fields with examples: " + (", ".join(example_fields) if example_fields else "none")
        )
        st.write("Grouping fields: " + (", ".join(group_by) if group_by else "none"))


def _sync_field_approval_state(
    sample: pl.DataFrame,
    available_fields: list[str],
    required_fields: list[str],
) -> tuple[list[str], list[str], list[str]]:
    initialized = bool(st.session_state.get("ai_studio_field_approval_initialized"))
    if initialized:
        approved_fields = st.session_state.get("ai_studio_approved_fields") or []
        example_fields = st.session_state.get("ai_studio_example_fields") or []
        group_by_fields = st.session_state.get("ai_studio_group_by_fields") or []
    else:
        approved_fields = list(available_fields)
        example_fields = []
        group_by_fields = _default_group_by_fields(sample, approved_fields, required_fields)
        st.session_state["ai_studio_field_approval_initialized"] = True

    approved_fields, example_fields, group_by_fields = _normalize_field_approval_state(
        available_fields=available_fields,
        required_fields=required_fields,
        approved_fields=approved_fields,
        example_fields=example_fields,
        group_by_fields=group_by_fields,
    )
    st.session_state["ai_studio_approved_fields"] = approved_fields
    st.session_state["ai_studio_example_fields"] = example_fields
    st.session_state["ai_studio_group_by_fields"] = group_by_fields
    return approved_fields, example_fields, group_by_fields


def _normalize_field_approval_state(
    *,
    available_fields: list[str],
    required_fields: list[str],
    approved_fields: list[str],
    example_fields: list[str],
    group_by_fields: list[str],
) -> tuple[list[str], list[str], list[str]]:
    available_set = set(available_fields)
    required_set = {field for field in required_fields if field in available_set}
    approved_set = {field for field in approved_fields if field in available_set} | required_set
    example_set = {field for field in example_fields if field in approved_set}
    group_by_set = {field for field in group_by_fields if field in approved_set}
    return (
        _ordered_fields(available_fields, approved_set),
        _ordered_fields(available_fields, example_set),
        _ordered_fields(available_fields, group_by_set),
    )


def _invalidate_ai_sharing_confirmation_if_scope_changed(
    *,
    previous_approved_fields: list[str],
    previous_example_fields: list[str],
    approved_fields: list[str],
    example_fields: list[str],
) -> None:
    if approved_fields == previous_approved_fields and example_fields == previous_example_fields:
        return
    # Field approval is a fragment. Clear confirmation immediately so the
    # still-rendered Copilot cannot use the prior sharing scope.
    _clear_ai_sharing_confirmation()


def _field_approval_editor_rows(
    sample: pl.DataFrame,
    fields: list[str],
    *,
    required_fields: list[str],
    approved_fields: list[str],
    example_fields: list[str],
) -> list[dict[str, Any]]:
    required_set = set(required_fields)
    approved_set = set(approved_fields)
    example_set = set(example_fields)
    rows: list[dict[str, Any]] = []
    for field in fields:
        series = sample.get_column(field)
        rows.append(
            {
                "Approve": field in approved_set,
                "Send To AI": field in example_set,
                "Column": field,
                "Data Type": str(sample.schema.get(field, "")),
                "Unique Count": int(series.n_unique()),
                "Most occurring": _most_occurring_value_text(series),
                "Values": _schema_values_text(series),
                "Field Tags": _field_tags(field, series, required_set),
            }
        )
    return rows


def _field_approval_editor_frame(
    sample: pl.DataFrame,
    fields: list[str],
    *,
    required_fields: list[str],
    approved_fields: list[str],
    example_fields: list[str],
) -> pl.DataFrame:
    rows = _field_approval_editor_rows(
        sample,
        fields,
        required_fields=required_fields,
        approved_fields=approved_fields,
        example_fields=example_fields,
    )
    if not rows:
        return pl.DataFrame(
            schema={
                "Approve": pl.Boolean,
                "Send To AI": pl.Boolean,
                "Column": pl.String,
                "Data Type": pl.String,
                "Unique Count": pl.Int64,
                "Most occurring": pl.String,
                "Values": pl.String,
                "Field Tags": pl.String,
            }
        )
    return pl.DataFrame(rows).select(FIELD_APPROVAL_EDITOR_COLUMNS)


def _apply_field_approval_edits(
    rows: list[dict[str, Any]],
    *,
    available_fields: list[str],
    required_fields: list[str],
    approved_fields: list[str],
    example_fields: list[str],
) -> tuple[list[str], list[str]]:
    """Merge visible editor checkbox edits over the current approval state.

    Rows hidden by the field filter keep their existing approval and sharing
    membership; required fields stay approved.
    """
    available_set = set(available_fields)
    approved_set = set(approved_fields)
    example_set = set(example_fields)
    for row in rows:
        field = str(row.get("Column", "") or "")
        if field not in available_set:
            continue
        if _truthy_editor_value(row.get("Approve", True)):
            approved_set.add(field)
        else:
            approved_set.discard(field)
        if _truthy_editor_value(row.get("Send To AI", row.get("Share Sample Values"))):
            example_set.add(field)
        else:
            example_set.discard(field)
    approved_set |= {field for field in required_fields if field in available_set}
    return (
        _ordered_fields(available_fields, approved_set),
        _ordered_fields(available_fields, example_set & approved_set),
    )


def _field_tags(field: str, series: pl.Series, required_fields: set[str]) -> str:
    tags: list[str] = []
    if field in required_fields:
        tags.append("Required")
    if _is_numeric_series(series):
        tags.append("Numeric measure")
    elif _is_business_dimension(series):
        tags.append("Business dimension")
    if _looks_like_id(field):
        tags.append("Likely ID")
    return ", ".join(tags) if tags else "Field"


def _looks_like_id(field: str) -> bool:
    lowered = field.casefold()
    return lowered.endswith("id") or "id" in lowered


def _is_numeric_series(series: pl.Series) -> bool:
    return series.dtype.is_numeric() and series.dtype != pl.Boolean


def _is_business_dimension(series: pl.Series) -> bool:
    return series.dtype in {pl.String, pl.Categorical, pl.Boolean} or series.n_unique() <= 50


def _most_occurring_value_text(series: pl.Series) -> str:
    if _is_numeric_series(series):
        return "N/A"
    values = [_sample_display_value(value) for value in series.drop_nulls().to_list()]
    if not values:
        return "N/A"
    most_common = Counter(values).most_common(1)
    return f"[{most_common[0][0]}]" if most_common else "N/A"


def _schema_values_text(series: pl.Series, *, max_examples: int = 8) -> str:
    values = series.drop_nulls()
    if values.is_empty():
        return "N/A"
    if _is_numeric_series(series):
        return _numeric_summary_text(values)
    unique_values = values.unique(maintain_order=True).head(max_examples).to_list()
    return "[" + ", ".join(_sample_display_value(value) for value in unique_values) + "]"


def _numeric_summary_text(series: pl.Series) -> str:
    min_value = series.min()
    max_value = series.max()
    mean_value = series.mean()
    median_value = series.median()
    return (
        f"Min = {_format_summary_value(min_value)} "
        f"Max = {_format_summary_value(max_value)} "
        f"Mean = {_format_summary_value(mean_value)} "
        f"Median = {_format_summary_value(median_value)}"
    )


def _format_summary_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sample_display_value(value: Any) -> str:
    return repr(value)


def _ordered_fields(available_fields: list[str], selected_fields: set[str]) -> list[str]:
    return [field for field in available_fields if field in selected_fields]


def _truthy_editor_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _sample_preview(
    raw_sample: pl.DataFrame,
    schema_sample: pl.DataFrame,
) -> None:
    with components.bordered_panel(
        "Sample Preview",
    ):
        schema_rows = [
            {
                "column": name,
                "dtype": str(dtype),
                "unique": (
                    schema_sample.get_column(name).n_unique()
                    if name in schema_sample.columns
                    else 0
                ),
            }
            for name, dtype in schema_sample.schema.items()
        ]
        st.dataframe(schema_rows, hide_index=True, width="stretch", height=320)
        with st.expander("Rows", expanded=False):
            st.dataframe(schema_sample.head(100), hide_index=True, width="stretch", height=300)
        if schema_sample.columns != raw_sample.columns:
            with st.expander("Raw Rows", expanded=False):
                st.dataframe(raw_sample.head(100), hide_index=True, width="stretch", height=300)


def _default_subject_column(columns: list[str]) -> str:
    for candidate in ["SubjectID", "CustomerID", "CustomerId", "SubjectId", "CustomerKey"]:
        if candidate in columns:
            return candidate
    for column in columns:
        lowered = column.casefold()
        if ("subject" in lowered or "customer" in lowered) and "id" in lowered:
            return column
    return _default_column(columns, "id")


def _default_time_column(columns: list[str], preferred: str, *, fallback: bool = True) -> str:
    if preferred in columns:
        return preferred
    for column in columns:
        if preferred.casefold() in column.casefold():
            return column
    for column in columns:
        if "time" in column.casefold() or "date" in column.casefold():
            return column
    return columns[0] if fallback and columns else ""


def _default_column(columns: list[str], preferred: str, *, fallback: bool = True) -> str:
    if preferred in columns:
        return preferred
    for column in columns:
        if preferred.casefold() in column.casefold():
            return column
    return columns[0] if fallback and columns else ""


def _default_group_by_fields(
    sample: pl.DataFrame,
    approved_fields: list[str],
    required_fields: list[str] | None = None,
) -> list[str]:
    return dimension_profile.default_group_by_fields(
        sample,
        approved_fields,
        required_fields=required_fields or _studio_required_fields(sample),
        limit=5,
    )
