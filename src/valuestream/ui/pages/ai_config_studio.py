"""Guided configuration studio page."""

from __future__ import annotations

import copy
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import time
import zipfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
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
    validate_draft_field_contract,
    validation_trace_for_repair,
)
from valuestream.ai.copilot import (
    draft_patch_bundles,
    merge_selected_draft_patch_bundles,
)
from valuestream.ai.settings import load_llm_settings_config, write_chat_with_data_config
from valuestream.ai.studio import (
    AI_PROVIDER_PREFLIGHT_MAX_TIMEOUT_SECONDS,
    AIProviderCallError,
    DraftAttemptDiagnostic,
    DraftCandidateResult,
    ai_provider_preflight_cache_key,
    generate_validated_candidate,
    preflight_ai_provider,
)
from valuestream.config import model
from valuestream.config.canonical import catalog_config_hash, processor_computation_hash
from valuestream.expr import parser as expr_parser
from valuestream.expr.translator import translate
from valuestream.ui import (
    ai_studio_checkpoint,
    builder,
    components,
    config_help,
    dimension_profile,
    field_remap,
    forms,
    recipe_library,
)
from valuestream.ui.context import ValueStreamContext
from valuestream.ui.instrumentation import (
    AuthoringEvent,
    AuthoringOutcome,
    AuthoringStage,
    AuthoringWorkflow,
    record_event,
    start_journey,
)
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
    "14. Apply",
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
    "14. Apply",
]
CATALOG_DRAFT_STEPS = [
    "Workspace Draft",
    "Processors",
    "Metrics",
    "Reports",
    "Reports Review",
    "Chat",
    "Settings",
    "Apply",
]
AI_CALLS_ENABLED_STATE_KEY = "ai_studio_ai_calls_enabled"
AI_SHARING_CONFIRMATION_STATE_KEY = "ai_studio_ai_sharing_confirmed_signature"
AI_REPLACEMENT_CONFIRM_STATE_KEY = "ai_studio_replacement_confirmed_signature"
AI_STUDIO_AUTHORING_WORKFLOW_KEY = "ai_studio_authoring_workflow"
BUILDER_SOURCE_RETURN_URL = "/configuration_builder?from=ai_studio_source"
AI_PREFLIGHT_TIMEOUT_SECONDS = AI_PROVIDER_PREFLIGHT_MAX_TIMEOUT_SECONDS
AI_PREFLIGHT_NEGATIVE_TTL_SECONDS = 15
AI_SHARING_CONFIRMATION_WIDGET_PREFIX = "ai_studio_ai_sharing_consent_"
AI_SHARING_CONTRACT_STATE_KEY = "ai_studio_ai_sharing_contract_signature"
AI_SHARING_RECEIPT_STATE_KEY = "ai_studio_ai_sharing_consent_receipt"
CATALOG_DRAFT_SOURCE = "catalog"
STUDIO_PHASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Data", (0, 1, 2, 3, 4, 5)),
    ("Draft", (6,)),
    ("Review", (7, 8, 9, 10)),
    ("Apply", (11, 12, 13)),
)
STUDIO_READINESS_AREAS = ("Data", "Processor", "Metric", "Report", "Provider", "Runtime")
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
AI_STUDIO_UPLOAD_MAX_BYTES = 512 * 1024 * 1024
AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES = 1024 * 1024 * 1024
AI_STUDIO_ARCHIVE_MAX_MEMBERS = 64
AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY = "ai_studio_rename_capitalize_enabled"
AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY = "ai_studio_rename_capitalize"
AI_STUDIO_RENAME_CAPITALIZE_WIDGET_KEY = "ai_studio_rename_capitalize_transform"
AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY = "ai_studio_schema_contract_stale"
AI_STUDIO_CHECKPOINT_CONTEXT_KEY = "ai_studio_checkpoint_context"
AI_STUDIO_CHECKPOINT_LOADED_WORKSPACE_KEY = "ai_studio_checkpoint_loaded_workspace"
AI_STUDIO_CHECKPOINT_PENDING_KEY = "ai_studio_checkpoint_pending"
AI_STUDIO_CHECKPOINT_STAGED_KEY = "ai_studio_checkpoint_staged"
AI_STUDIO_CHECKPOINT_BASE_HASH_KEY = "ai_studio_checkpoint_base_catalog_hash"
AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY = "ai_studio_checkpoint_reconciliation"
AI_STUDIO_CHECKPOINT_NOTICE_KEY = "ai_studio_checkpoint_notice"
AI_STUDIO_CHECKPOINT_SUPPRESSED_WORKSPACE_KEY = "ai_studio_checkpoint_suppressed_workspace"
AI_STUDIO_CATALOG_DRAFT_DIRTY_KEY = "ai_studio_catalog_draft_dirty"


class SamplePreviewLimitError(ValueError):
    """An actionable preview rejection caused by a documented safety limit."""


@dataclass(frozen=True)
class SampleFormatCapability:
    """One advertised sample format shared by picker, preview, and validation."""

    key: str
    label: str
    suffixes: tuple[str, ...]
    upload_extensions: tuple[str, ...]
    archive_member_suffixes: tuple[str, ...] = ()


SAMPLE_FORMAT_CAPABILITIES = (
    SampleFormatCapability("csv", "CSV", (".csv",), ("csv",)),
    SampleFormatCapability("parquet", "Parquet", (".parquet",), ("parquet",)),
    SampleFormatCapability("json", "JSON", (".json",), ("json",)),
    SampleFormatCapability("ndjson", "NDJSON", (".ndjson",), ("ndjson",)),
    SampleFormatCapability(
        "gzip",
        "gzip-compressed JSON/NDJSON",
        (".gz", ".gzip"),
        ("gz", "gzip"),
        (".json", ".ndjson"),
    ),
    SampleFormatCapability(
        "zip",
        "ZIP containing JSON/NDJSON",
        (".zip",),
        ("zip",),
        (".json", ".ndjson"),
    ),
)


@dataclass(frozen=True)
class SampleSourcePlan:
    """Preview/runtime contract inferred from one selected sample."""

    format_label: str
    source_id: str
    reader_kind: str
    root: str
    file_pattern: str
    group_pattern: str = ""
    timestamp_format: str = ""
    production_ready: bool = False
    requires_runtime_confirmation: bool = False
    note: str = ""


@dataclass(frozen=True)
class DraftValidationSnapshot:
    """Validation evidence tied to an exact draft revision."""

    signature: str
    ok: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class StudioReadinessIssue:
    """One safe, actionable Apply-readiness finding."""

    area: str
    severity: str
    object_path: str
    current_value: str
    expected_contract: str
    remediation: str
    target_step: str
    runtime_only: bool = False


@dataclass(frozen=True)
class StudioReadinessSnapshot:
    """Canonical readiness evidence shared by summary, Export, and Apply."""

    issues: tuple[StudioReadinessIssue, ...]
    artifact_counts: dict[str, str]
    last_changes: dict[str, str]
    apply_ready: bool
    apply_disabled_reason: str
    export_ready: bool
    export_disabled_reason: str

    @property
    def blocker_count(self) -> int:
        return sum(issue.severity == "blocker" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)


def render(ctx: ValueStreamContext) -> None:
    """Render the guided AI catalog studio."""
    builder_source_handoff = _builder_source_handoff()
    _render_studio(
        ctx,
        ai_calls_enabled=not builder_source_handoff,
        title=(
            "Add source · Configuration Studio"
            if builder_source_handoff
            else "AI Configuration Studio"
        ),
        subtitle=(
            "Start with a workspace sample, review the deterministic additions, then apply "
            "and return to Configuration Builder."
            if builder_source_handoff
            else "Prepare a source sample, approve fields, review AI-generated YAML, then "
            "apply or export it."
        ),
        status_label="Deterministic source addition"
        if builder_source_handoff
        else "Guided AI draft",
        authoring_workflow=(
            AuthoringWorkflow.BUILDER if builder_source_handoff else AuthoringWorkflow.AI_STUDIO
        ),
        sample_required=builder_source_handoff,
    )


def _builder_source_handoff() -> bool:
    """Return whether Builder requested the deterministic Add-source journey."""

    return (
        str(st.query_params.get("from") or "") == "configuration_builder"
        and str(st.query_params.get("intent") or "") == "add_source"
        and str(st.query_params.get("mode") or "") == "deterministic"
    )


def _studio_authoring_workflow() -> AuthoringWorkflow:
    raw = str(
        st.session_state.get(AI_STUDIO_AUTHORING_WORKFLOW_KEY) or AuthoringWorkflow.AI_STUDIO.value
    )
    try:
        return AuthoringWorkflow(raw)
    except ValueError:
        return AuthoringWorkflow.AI_STUDIO


def _render_studio(  # noqa: PLR0912, PLR0915
    ctx: ValueStreamContext,
    *,
    ai_calls_enabled: bool,
    title: str,
    subtitle: str,
    status_label: str,
    include_header: bool = True,
    authoring_workflow: AuthoringWorkflow = AuthoringWorkflow.AI_STUDIO,
    sample_required: bool = False,
) -> None:
    """Render the shared guided catalog studio workflow."""
    st.session_state[AI_STUDIO_AUTHORING_WORKFLOW_KEY] = authoring_workflow.value
    start_journey(st.session_state, workflow=authoring_workflow)
    if include_header:
        components.render_page_header(
            title,
            subtitle,
            status="pending",
            status_label=status_label,
        )
    if sample_required:
        handoff_col, return_col = st.columns([0.72, 0.28], vertical_alignment="center")
        handoff_col.info(
            "You are adding to the active workspace. Existing sources, processors, metrics, "
            "and reports are carried into the reviewed revision."
        )
        return_col.link_button(
            "Cancel and return to Builder",
            f"{BUILDER_SOURCE_RETURN_URL}_cancelled",
            icon=":material/arrow_back:",
            width="stretch",
        )

    _initialize_ai_studio_checkpoint(ctx)
    _render_ai_studio_checkpoint_notice()
    _render_ai_studio_checkpoint_reconciliation()
    if _render_ai_studio_checkpoint_recovery():
        return

    st.session_state[AI_CALLS_ENABLED_STATE_KEY] = ai_calls_enabled
    st.session_state["ai_studio_active_workspace_name"] = ctx.catalog.pipelines.workspace
    if ai_calls_enabled:
        _initialize_ai_settings(ctx.workspace)
    raw_sample = _load_sample(ctx.workspace, ai_calls_enabled=ai_calls_enabled)
    if raw_sample is None:
        if not ai_calls_enabled and not sample_required:
            st.info("Choose a source sample above, or review the current catalog draft below.")
            _current_catalog_draft_editor(ctx)
            _persist_ai_studio_checkpoint()
            return
        _persist_ai_studio_checkpoint()
        return

    _initialize_state(raw_sample)
    _apply_staged_ai_studio_checkpoint(raw_sample, ai_calls_enabled=ai_calls_enabled)
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
        st.session_state["ai_studio_jump_step"] = next_step
        current_step = next_step
    else:
        current_step = st.session_state.get("ai_studio_step", STEPS[0])
    current_step = _normalize_studio_step(current_step, steps)
    if current_step not in steps:
        current_step = steps[0]
        st.session_state["ai_studio_step"] = current_step
    st.session_state["ai_studio_step"] = current_step
    step = _render_studio_step_header(
        current_step,
        steps,
        statuses=_phase_statuses(approved_fields, preprocessing_error),
    )
    _render_schema_contract_notice(steps)

    if ai_calls_enabled:
        _render_ai_data_sharing_confirmation(approved_fields)
        _render_copilot_panel(step, working, approved_fields)
        if st.session_state.get("ai_studio_pending_draft") is not None:
            st.info(
                "A validated proposal is waiting for review. Accept or discard the complete "
                "change bundles before editing the accepted revision."
            )
            _render_pending_draft_review()
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
    _render_studio_step_navigation(step, steps)
    _persist_ai_studio_checkpoint()


def _initialize_ai_studio_checkpoint(ctx: ValueStreamContext) -> None:
    """Load one recovery candidate before sample or provider initialization."""

    workspace = str(Path(ctx.workspace).resolve())
    current_hash = catalog_config_hash(ctx.catalog)
    previous = st.session_state.get(AI_STUDIO_CHECKPOINT_CONTEXT_KEY)
    previous_workspace = previous.get("workspace") if isinstance(previous, Mapping) else None
    if previous_workspace and previous_workspace != workspace:
        _persist_ai_studio_checkpoint()
        for key in (
            AI_STUDIO_CHECKPOINT_PENDING_KEY,
            AI_STUDIO_CHECKPOINT_STAGED_KEY,
            AI_STUDIO_CHECKPOINT_BASE_HASH_KEY,
            AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY,
            AI_STUDIO_CHECKPOINT_LOADED_WORKSPACE_KEY,
        ):
            st.session_state.pop(key, None)
    st.session_state[AI_STUDIO_CHECKPOINT_CONTEXT_KEY] = {
        "workspace": workspace,
        "catalog_hash": current_hash,
    }

    has_state = isinstance(st.session_state.get("ai_studio_draft"), dict) or bool(
        st.session_state.get("ai_studio_sample_identity")
    )
    base_hash = st.session_state.get(AI_STUDIO_CHECKPOINT_BASE_HASH_KEY)
    if has_state and isinstance(base_hash, str) and base_hash != current_hash:
        st.session_state[AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY] = {
            "saved_hash": base_hash,
            "current_hash": current_hash,
        }
    elif has_state and not isinstance(base_hash, str):
        st.session_state[AI_STUDIO_CHECKPOINT_BASE_HASH_KEY] = current_hash

    if st.session_state.get(AI_STUDIO_CHECKPOINT_LOADED_WORKSPACE_KEY) == workspace:
        return
    st.session_state[AI_STUDIO_CHECKPOINT_LOADED_WORKSPACE_KEY] = workspace
    if has_state:
        return
    result = ai_studio_checkpoint.load_ai_studio_checkpoint(
        workspace,
        current_catalog_hash=current_hash,
        allowed_steps=tuple(dict.fromkeys([*STEPS, *DETERMINISTIC_STEPS])),
    )
    if result.checkpoint is not None:
        checkpoint = result.checkpoint
        st.session_state[AI_STUDIO_CHECKPOINT_PENDING_KEY] = {
            "workspace": workspace,
            "status": result.status,
            "saved_at": checkpoint.saved_at,
            "base_catalog_hash": checkpoint.base_catalog_hash,
            "current_step": checkpoint.current_step,
            "requires_sample_reselect": checkpoint.requires_sample_reselect,
            "state": copy.deepcopy(checkpoint.state),
        }
    elif result.status == "expired":
        st.session_state[AI_STUDIO_CHECKPOINT_NOTICE_KEY] = (
            "An expired AI Configuration Studio checkpoint was removed."
        )
    elif result.status == "invalid":
        st.session_state[AI_STUDIO_CHECKPOINT_NOTICE_KEY] = (
            "An unreadable AI Configuration Studio checkpoint was removed."
        )


def _render_ai_studio_checkpoint_recovery() -> bool:
    """Render an explicit restore/discard decision and pause initialization."""

    pending = st.session_state.get(AI_STUDIO_CHECKPOINT_PENDING_KEY)
    if not isinstance(pending, Mapping):
        return False
    state = pending.get("state")
    has_draft = isinstance(state, Mapping) and isinstance(state.get("ai_studio_draft"), dict)
    requires_reselect = bool(pending.get("requires_sample_reselect"))
    saved_at = pending.get("saved_at")
    saved_label = (
        saved_at.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(saved_at, dt.datetime)
        else "an earlier session"
    )
    with st.container(border=True):
        if pending.get("status") == "reconciliation":
            st.warning("Reconciliation required before restoring this Studio checkpoint.")
            st.caption(
                "The workspace catalog changed after this checkpoint. Restore revalidates the "
                "accepted draft and clears its prior review before navigation or Apply."
            )
        else:
            st.info("An unapplied AI Configuration Studio checkpoint is available.")
            st.caption("Restore imports only committed, privacy-safe authoring state.")
        if requires_reselect:
            st.warning(
                "The original upload was not retained. Restore keeps only the accepted catalog "
                "draft metadata; reselect the sample before continuing."
            )
        st.caption(f"Saved {saved_label} · accepted catalog draft: {'yes' if has_draft else 'no'}.")
        restore_col, discard_col = st.columns(2)
        restore_col.button(
            "Restore Studio checkpoint",
            type="primary",
            icon=":material/restore:",
            width="stretch",
            on_click=_restore_ai_studio_checkpoint,
        )
        discard_col.button(
            "Discard Studio checkpoint",
            icon=":material/delete_sweep:",
            width="stretch",
            on_click=_discard_ai_studio_checkpoint,
        )
    return True


def _render_ai_studio_checkpoint_notice() -> None:
    notice = st.session_state.pop(AI_STUDIO_CHECKPOINT_NOTICE_KEY, None)
    if notice:
        st.info(str(notice))


def _render_ai_studio_checkpoint_reconciliation() -> None:
    if isinstance(st.session_state.get(AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY), Mapping):
        st.warning(
            "Reconciliation required: the catalog or selected workspace sample changed since "
            "this Studio draft was checkpointed. The restored draft was revalidated and must "
            "be reviewed again before Apply."
        )


def _restore_ai_studio_checkpoint() -> None:
    """Stage safe state so sample initialization cannot erase it."""

    pending = st.session_state.get(AI_STUDIO_CHECKPOINT_PENDING_KEY)
    context = st.session_state.get(AI_STUDIO_CHECKPOINT_CONTEXT_KEY)
    if not isinstance(pending, Mapping) or not isinstance(context, Mapping):
        return
    if pending.get("workspace") != context.get("workspace"):
        return
    state = pending.get("state")
    base_hash = pending.get("base_catalog_hash")
    if not isinstance(state, Mapping) or not isinstance(base_hash, str):
        return
    staged = {
        "state": copy.deepcopy(dict(state)),
        "current_step": pending.get("current_step"),
        "requires_sample_reselect": bool(pending.get("requires_sample_reselect")),
        "reconciliation": pending.get("status") == "reconciliation",
    }
    st.session_state[AI_STUDIO_CHECKPOINT_STAGED_KEY] = staged
    st.session_state[AI_STUDIO_CHECKPOINT_BASE_HASH_KEY] = base_hash
    relative = state.get("ai_studio_sample_workspace_relative")
    identity = state.get("ai_studio_sample_identity")
    if isinstance(relative, str) and relative and isinstance(identity, str):
        st.session_state["ai_studio_workspace_sample_active"] = relative
        st.session_state["ai_studio_sample_workspace_relative"] = relative
        st.session_state["ai_studio_sample_identity"] = identity
    else:
        st.session_state["ai_studio_workspace_sample_active"] = ""
        st.session_state[AI_STUDIO_CHECKPOINT_NOTICE_KEY] = (
            "Reselect the original upload (or another reviewed sample) to continue. Upload "
            "bytes and values were never checkpointed."
        )
    if staged["reconciliation"]:
        st.session_state[AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY] = {
            "saved_hash": base_hash,
            "current_hash": str(context.get("catalog_hash", "")),
        }
    st.session_state.pop(AI_STUDIO_CHECKPOINT_PENDING_KEY, None)


def _discard_ai_studio_checkpoint() -> None:
    context = st.session_state.get(AI_STUDIO_CHECKPOINT_CONTEXT_KEY)
    if isinstance(context, Mapping) and isinstance(context.get("workspace"), str):
        workspace = str(context["workspace"])
        ai_studio_checkpoint.discard_ai_studio_checkpoint(workspace)
        # Respect the explicit privacy decision for the rest of this session.
        # Accepting a new draft re-enables checkpointing below.
        st.session_state[AI_STUDIO_CHECKPOINT_SUPPRESSED_WORKSPACE_KEY] = workspace
    for key in (
        AI_STUDIO_CHECKPOINT_PENDING_KEY,
        AI_STUDIO_CHECKPOINT_STAGED_KEY,
        AI_STUDIO_CHECKPOINT_BASE_HASH_KEY,
        AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY,
    ):
        st.session_state.pop(key, None)


def _apply_staged_ai_studio_checkpoint(
    raw_sample: pl.DataFrame,
    *,
    ai_calls_enabled: bool,
) -> None:
    """Restore after sample initialization, then validate the accepted revision."""

    del raw_sample
    staged = st.session_state.get(AI_STUDIO_CHECKPOINT_STAGED_KEY)
    if not isinstance(staged, Mapping):
        return
    state = staged.get("state")
    if not isinstance(state, Mapping):
        st.session_state.pop(AI_STUDIO_CHECKPOINT_STAGED_KEY, None)
        return
    expected_identity = str(state.get("ai_studio_sample_identity") or "")
    actual_identity = str(st.session_state.get("ai_studio_sample_identity") or "")
    sample_matches = bool(expected_identity and expected_identity == actual_identity)
    draft_only = bool(staged.get("requires_sample_reselect")) or not sample_matches
    draft_keys = {
        "ai_studio_draft",
        "ai_studio_reviewed_signature",
        "ai_studio_draft_source",
        "ai_studio_catalog_draft_step",
    }
    for key, value in state.items():
        if key in {"ai_studio_draft", "ai_studio_reviewed_signature"}:
            continue
        if draft_only and key not in draft_keys:
            continue
        st.session_state[key] = copy.deepcopy(value)
    if (
        not draft_only
        and AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY not in state
        and AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY in state
    ):
        st.session_state[AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = bool(
            state[AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY]
        )
    _migrate_rename_capitalize_state()

    draft = state.get("ai_studio_draft")
    snapshot: DraftValidationSnapshot | None = None
    if isinstance(draft, dict):
        st.session_state["ai_studio_validation_cache"] = {}
        _set_draft(copy.deepcopy(draft))
        snapshot = _draft_validation_snapshot(draft)
        saved_review = state.get("ai_studio_reviewed_signature")
        may_restore_review = (
            not staged.get("reconciliation")
            and sample_matches
            and snapshot.ok
            and saved_review == snapshot.signature
        )
        st.session_state["ai_studio_reviewed_signature"] = (
            snapshot.signature if may_restore_review else ""
        )
    _clear_ai_sharing_confirmation()

    reconciliation = bool(staged.get("reconciliation")) or (
        bool(expected_identity) and not sample_matches
    )
    if reconciliation:
        context = st.session_state.get(AI_STUDIO_CHECKPOINT_CONTEXT_KEY)
        st.session_state[AI_STUDIO_CHECKPOINT_RECONCILIATION_KEY] = {
            "saved_hash": st.session_state.get(AI_STUDIO_CHECKPOINT_BASE_HASH_KEY, ""),
            "current_hash": (
                str(context.get("catalog_hash", "")) if isinstance(context, Mapping) else ""
            ),
        }
    desired = staged.get("current_step")
    steps = _studio_steps(ai_calls_enabled=ai_calls_enabled)
    restored_step = _normalize_studio_step(desired, steps)
    if snapshot is not None and not snapshot.ok:
        restored_step = steps[6]
    if restored_step in steps:
        st.session_state["ai_studio_step"] = restored_step
        st.session_state["ai_studio_jump_step"] = restored_step
    st.session_state.pop(AI_STUDIO_CHECKPOINT_STAGED_KEY, None)


def _persist_ai_studio_checkpoint() -> None:
    """Persist only allowlisted committed state for the active workspace."""

    context = st.session_state.get(AI_STUDIO_CHECKPOINT_CONTEXT_KEY)
    if not isinstance(context, Mapping):
        return
    workspace = context.get("workspace")
    current_hash = context.get("catalog_hash")
    if not isinstance(workspace, str) or not isinstance(current_hash, str):
        return
    if isinstance(st.session_state.get(AI_STUDIO_CHECKPOINT_PENDING_KEY), Mapping) or isinstance(
        st.session_state.get(AI_STUDIO_CHECKPOINT_STAGED_KEY), Mapping
    ):
        return
    catalog_draft_hash = st.session_state.get("ai_studio_catalog_draft_hash")
    clean_catalog_baseline = (
        st.session_state.get("ai_studio_draft_source") == CATALOG_DRAFT_SOURCE
        and not st.session_state.get(AI_STUDIO_CATALOG_DRAFT_DIRTY_KEY, False)
        and isinstance(catalog_draft_hash, str)
        and bool(catalog_draft_hash)
        and current_hash.startswith(catalog_draft_hash)
    )
    if clean_catalog_baseline:
        ai_studio_checkpoint.discard_ai_studio_checkpoint(workspace)
        return
    if st.session_state.get(AI_STUDIO_CHECKPOINT_SUPPRESSED_WORKSPACE_KEY) == workspace:
        return
    draft = st.session_state.get("ai_studio_draft")
    if isinstance(draft, dict) and st.session_state.get(
        "ai_studio_published_signature"
    ) == _draft_signature(draft):
        ai_studio_checkpoint.discard_ai_studio_checkpoint(workspace)
        return
    base_hash = st.session_state.get(AI_STUDIO_CHECKPOINT_BASE_HASH_KEY)
    if not isinstance(base_hash, str):
        base_hash = current_hash
        st.session_state[AI_STUDIO_CHECKPOINT_BASE_HASH_KEY] = base_hash
    step = str(st.session_state.get("ai_studio_step") or STEPS[0])
    allowed_steps = tuple(dict.fromkeys([*STEPS, *DETERMINISTIC_STEPS]))
    if step not in allowed_steps:
        step = STEPS[0]
    try:
        ai_studio_checkpoint.write_ai_studio_checkpoint(
            workspace,
            session_state=st.session_state,
            current_step=step,
            base_catalog_hash=base_hash,
        )
    except (OSError, TypeError, ValueError):
        logger.exception("Failed to persist AI Configuration Studio checkpoint")
        st.session_state[AI_STUDIO_CHECKPOINT_NOTICE_KEY] = (
            "This Studio state remains in the current session, but its recovery checkpoint "
            "could not be saved."
        )


def _render_studio_step_header(
    current_step: str,
    steps: list[str],
    *,
    statuses: dict[str, str] | None = None,
) -> str:
    """Render the compact phase rail and one phase-scoped step selector."""

    index = steps.index(current_step)
    current_phase = _phase_for_step(current_step, steps)
    phase_names = [name for name, _indexes in STUDIO_PHASES]
    phase_statuses = statuses or _phase_statuses([])
    phase_key = "ai_studio_phase"
    if st.session_state.get(phase_key) != current_phase:
        st.session_state[phase_key] = current_phase
    st.segmented_control(
        "Workflow phase",
        phase_names,
        key=phase_key,
        format_func=lambda phase: _phase_label(phase, phase_statuses.get(phase, "empty")),
        width="stretch",
        on_change=_jump_to_phase_start,
        args=(steps,),
    )
    st.progress(
        (index + 1) / len(steps),
        text=(
            f"{current_phase} · {_phase_status_text(phase_statuses.get(current_phase, 'empty'))} "
            f"· Step {index + 1} of {len(steps)}"
        ),
    )
    phase_steps = _phase_step_options(current_phase, steps)
    jump_key = "ai_studio_jump_step"
    if st.session_state.get(jump_key) not in phase_steps:
        st.session_state[jump_key] = current_step
    st.selectbox(
        f"Jump to step in {current_phase}",
        phase_steps,
        key=jump_key,
        on_change=_queue_studio_step_jump,
        args=(steps,),
        help="Move within this phase without changing accepted or committed work.",
    )
    return current_step


def _render_schema_contract_notice(steps: list[str]) -> None:
    """Keep a stale accepted revision visible while blocking unsafe downstream edits."""

    if not st.session_state.get(AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY):
        return
    st.warning(
        "Source field naming changed after this revision was created. The accepted revision "
        "is still available for comparison, but AI changes and Apply are blocked until you "
        "generate and review an updated draft against the effective schema."
    )
    st.button(
        "Review updated draft",
        key="ai_studio_reconcile_schema_contract",
        type="primary",
        icon=":material/rule:",
        on_click=_queue_schema_contract_review,
        args=(steps,),
    )


def _queue_schema_contract_review(steps: list[str]) -> None:
    """Queue Draft navigation before the phase-scoped step widget is rendered again."""

    target = steps[6]
    st.session_state["ai_studio_step"] = target
    st.session_state["ai_studio_jump_step"] = target
    st.session_state["ai_studio_next_step"] = target


def _render_studio_step_navigation(step: str, steps: list[str]) -> None:
    """Keep the next decision visible at the bottom of every Studio step."""

    index = steps.index(step)
    back_col, next_col = st.columns([1, 1], vertical_alignment="center")
    if back_col.button(
        "Back",
        icon=":material/arrow_back:",
        disabled=index == 0,
        key=f"ai_studio_back_{index}",
        width="stretch",
    ):
        target = steps[index - 1]
        st.session_state["ai_studio_next_step"] = target
        st.rerun()
    if next_col.button(
        "Continue" if index < len(steps) - 1 else "Review outcome",
        icon=":material/arrow_forward:",
        disabled=index == len(steps) - 1,
        key=f"ai_studio_continue_{index}",
        width="stretch",
    ):
        target = steps[index + 1]
        st.session_state["ai_studio_next_step"] = target
        st.rerun()


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
            target = steps[indexes[0]]
            st.session_state["ai_studio_step"] = target
            st.session_state["ai_studio_jump_step"] = target
            st.session_state["ai_studio_next_step"] = target
            return


def _queue_studio_step_jump(steps: list[str]) -> None:
    selected = _normalize_studio_step(st.session_state.get("ai_studio_jump_step"), steps)
    if selected in steps:
        st.session_state["ai_studio_step"] = selected
        st.session_state["ai_studio_next_step"] = selected


def _phase_statuses(
    approved_fields: list[str], preprocessing_error: str | None = None
) -> dict[str, str]:
    draft = st.session_state.get("ai_studio_draft")
    pending = st.session_state.get("ai_studio_pending_draft") is not None
    statuses = {
        "Data": "complete" if approved_fields and not preprocessing_error else "attention",
        "Draft": "attention" if pending else "empty",
        "Review": "empty",
        "Apply": "empty",
    }
    if isinstance(draft, dict):
        signature = _draft_signature(draft)
        snapshot = _draft_validation_snapshot(draft)
        statuses["Draft"] = "complete" if snapshot.ok and not pending else "attention"
        reviewed = st.session_state.get("ai_studio_reviewed_signature") == signature
        statuses["Review"] = "complete" if snapshot.ok and reviewed and not pending else "attention"
        if (
            snapshot.ok
            and reviewed
            and st.session_state.get("ai_studio_published_signature") == signature
        ):
            statuses["Apply"] = "complete"
        elif snapshot.ok and reviewed:
            statuses["Apply"] = "attention"
    return statuses


def _phase_label(name: str, status: str) -> str:
    return f"{name} · {_phase_status_text(status)}"


def _phase_status_text(status: str) -> str:
    return {
        "complete": "Complete",
        "attention": "Attention",
        "empty": "Not started",
    }.get(status, "Not started")


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
        _set_draft(draft, resume_checkpoint=False)
    approved_fields = _catalog_approved_fields(draft)
    working = _empty_working_frame(approved_fields)

    with components.bordered_panel(
        "Current Catalog Draft",
        "Edit the loaded workspace with the same non-AI review tools used after draft generation.",
    ):
        action_cols = st.columns([0.28, 0.72], vertical_alignment="center")
        if action_cols[0].button("Reload Current Catalog Draft", icon=":material/refresh:"):
            _load_current_catalog_draft(ctx)
            st.rerun()
        action_cols[1].caption(
            "This draft starts from the active catalog. Changes are held in session state "
            "until the save action writes them to the workspace."
        )
        _draft_counts(draft)
        snapshot = _draft_validation_snapshot(draft)
        _render_draft_validation(
            snapshot.ok,
            list(snapshot.issues),
            revision=snapshot.signature,
            expanded=not snapshot.ok,
        )

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
    _set_draft(_draft_from_catalog(ctx), resume_checkpoint=False)
    st.session_state["ai_studio_draft_source"] = CATALOG_DRAFT_SOURCE
    st.session_state["ai_studio_catalog_draft_hash"] = ctx.catalog_hash
    st.session_state[AI_STUDIO_CATALOG_DRAFT_DIRTY_KEY] = False


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


def _replace_exact_string_references(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_exact_string_references(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_string_references(child, replacements) for child in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _builder_source_addition_draft(
    ctx: ValueStreamContext,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Merge a deterministic source candidate into the active catalog draft.

    The handoff is additive: existing definitions keep their exact ids and
    content, while generated metric and dashboard ids are namespaced by the new
    source. A duplicate source id is rejected instead of becoming an implicit
    edit of the selected Source.
    """

    active = _draft_from_catalog(ctx)
    addition = copy.deepcopy(candidate)
    candidate_sources = addition.get("pipelines", {}).get("sources", [])
    if not isinstance(candidate_sources, list) or len(candidate_sources) != 1:
        raise ValueError("Add source expects exactly one deterministic source definition.")
    source_id = str(candidate_sources[0].get("id") or "").strip()
    existing_source_ids = {
        str(source.get("id"))
        for source in active["pipelines"].get("sources", [])
        if isinstance(source, dict) and source.get("id")
    }
    if not source_id:
        raise ValueError("Enter a Source ID before reviewing this addition.")
    if source_id in existing_source_ids:
        raise ValueError(
            f"Source ID {source_id!r} already exists. Choose a new Source ID; "
            "Add source never edits an existing source implicitly."
        )

    active_processors = active["processors"].get("processors", [])
    candidate_processors = addition.get("processors", {}).get("processors", [])
    if not isinstance(candidate_processors, list):
        raise ValueError("deterministic processors must be a list")
    used_processor_ids = {
        str(processor.get("id"))
        for processor in active_processors
        if isinstance(processor, dict) and processor.get("id")
    }
    processor_replacements: dict[str, str] = {}
    for processor in candidate_processors:
        old_id = str(processor.get("id") or "")
        new_id = builder.stable_catalog_id(
            old_id,
            fallback="processor",
            parent_id=source_id,
            existing_ids=used_processor_ids,
            preferred_id=old_id,
        )
        processor_replacements[old_id] = new_id
        used_processor_ids.add(new_id)
        processor["id"] = new_id

    active_metric_defs = active["metrics"].get("metrics", {})
    candidate_metric_defs = addition.get("metrics", {}).get("metrics", {})
    if not isinstance(candidate_metric_defs, dict):
        raise ValueError("deterministic metrics must be a mapping")
    used_metric_ids = {str(name) for name in active_metric_defs}
    metric_replacements: dict[str, str] = {}
    for old_id in candidate_metric_defs:
        new_id = builder.stable_catalog_id(
            str(old_id),
            fallback="metric",
            parent_id=source_id,
            existing_ids=used_metric_ids,
        )
        metric_replacements[str(old_id)] = new_id
        used_metric_ids.add(new_id)
    renamed_metric_defs: dict[str, Any] = {}
    replacements = {**processor_replacements, **metric_replacements}
    for old_id, definition in candidate_metric_defs.items():
        renamed_metric_defs[metric_replacements[str(old_id)]] = _replace_exact_string_references(
            definition,
            replacements,
        )

    active_dashboards = active["dashboards"].get("dashboards", [])
    candidate_dashboards = addition.get("dashboards", {}).get("dashboards", [])
    if not isinstance(candidate_dashboards, list):
        raise ValueError("deterministic dashboards must be a list")
    candidate_dashboards = _replace_exact_string_references(
        candidate_dashboards,
        metric_replacements,
    )
    used_dashboard_ids = {
        str(dashboard.get("id"))
        for dashboard in active_dashboards
        if isinstance(dashboard, dict) and dashboard.get("id")
    }
    for dashboard in candidate_dashboards:
        old_id = str(dashboard.get("id") or "")
        dashboard["id"] = builder.stable_catalog_id(
            str(dashboard.get("title") or old_id),
            fallback="dashboard",
            parent_id=source_id,
            existing_ids=used_dashboard_ids,
        )
        used_dashboard_ids.add(str(dashboard["id"]))

    merged = copy.deepcopy(active)
    merged["pipelines"]["sources"] = [
        *merged["pipelines"].get("sources", []),
        *candidate_sources,
    ]
    merged["processors"]["processors"] = [*active_processors, *candidate_processors]
    merged["metrics"]["metrics"] = {**active_metric_defs, **renamed_metric_defs}
    merged["dashboards"]["dashboards"] = [*active_dashboards, *candidate_dashboards]
    return merged


def _catalog_draft_overview(
    ctx: ValueStreamContext,
    draft: dict[str, Any],
    approved_fields: list[str],
) -> None:
    components.key_value_strip(
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
    for filename, section in _draft_files(draft).items():
        with st.expander(f"Technical details · {filename}", expanded=False):
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


def _load_sample(  # noqa: PLR0915
    workspace: Path,
    *,
    ai_calls_enabled: bool = True,
) -> pl.DataFrame | None:
    active_workspace_sample = str(st.session_state.get("ai_studio_workspace_sample_active") or "")
    with st.sidebar:
        st.write("### Studio Controls" if ai_calls_enabled else "### Builder Studio Controls")
        st.number_input(
            "Preview Rows",
            min_value=100,
            max_value=100_000,
            value=10_000,
            step=500,
            key="ai_studio_sample_rows",
            help=config_help.field_help("ai.preview_rows"),
        )
        if ai_calls_enabled:
            _ai_sidebar_controls()

    upload = None
    workspace_samples = _workspace_sample_files(workspace)
    if not active_workspace_sample:
        st.write("### Start with a source sample")
        st.caption(
            "Previewing does not change the workspace. Choose an existing workspace file, "
            "upload a local sample, or create the deterministic demo."
        )
        upload = st.file_uploader(
            "Upload a source sample",
            type=_sample_upload_extensions(),
            key="ai_studio_sample",
            help=config_help.field_help("ai.source_sample"),
        )
        st.caption(
            "Uploads are limited to 64 MiB. For larger CSV or Parquet files, place the "
            "file under workspace data/ and use the workspace picker."
        )
        workspace_col, demo_col = st.columns(2, vertical_alignment="bottom")
        selected_workspace_sample = workspace_col.selectbox(
            "Use workspace data",
            ["", *workspace_samples],
            format_func=lambda value: value or "Select a file under data/",
            key="ai_studio_workspace_sample_choice",
            help=config_help.field_help("ai.workspace_sample"),
        )
        if workspace_col.button(
            "Use workspace sample",
            icon=":material/folder_open:",
            disabled=not selected_workspace_sample,
            key="ai_studio_use_workspace_sample",
            width="stretch",
        ):
            active_workspace_sample = selected_workspace_sample
            st.session_state["ai_studio_workspace_sample_active"] = active_workspace_sample
            st.session_state["ai_studio_sample_origin"] = "workspace"
        if demo_col.button(
            "Try deterministic demo",
            icon=":material/science:",
            key="ai_studio_use_demo_sample",
            width="stretch",
            help="Creates a small CSV under data/studio so the preview and runtime source match.",
        ):
            active_workspace_sample = _create_demo_sample(workspace)
            st.session_state["ai_studio_workspace_sample_active"] = active_workspace_sample
            st.session_state["ai_studio_sample_origin"] = "demo"
    else:
        source_col, change_col = st.columns([0.75, 0.25], vertical_alignment="center")
        source_col.info(f"Source sample: `{active_workspace_sample}`")
        if change_col.button(
            "Choose another",
            key="ai_studio_change_sample",
            width="stretch",
        ):
            st.session_state["ai_studio_workspace_sample_active"] = ""
            st.session_state.pop("ai_studio_sample_bytes", None)
            st.rerun()

    if upload is None and not active_workspace_sample:
        return None
    try:
        limit = int(st.session_state.get("ai_studio_sample_rows", 10_000))
        if upload is not None:
            sample_name = Path(upload.name).name
            data = _uploaded_sample_bytes(upload)
            workspace_relative = ""
            st.session_state["ai_studio_sample_origin"] = "upload"
            st.session_state["ai_studio_sample_bytes"] = data
            sample_identity = hashlib.sha256(data).hexdigest()
            frame = _read_sample_bytes(sample_name, data, limit=limit)
        else:
            sample_path = (workspace / active_workspace_sample).resolve()
            data_root = (workspace / "data").resolve()
            if not sample_path.is_relative_to(data_root) or not sample_path.is_file():
                raise ValueError("workspace sample must be a file under the workspace data folder")
            sample_name = sample_path.name
            workspace_relative = active_workspace_sample
            st.session_state.pop("ai_studio_sample_bytes", None)
            sample_identity = _workspace_sample_identity(sample_path, workspace_relative)
            frame = _read_workspace_sample(sample_path, limit=limit)
        st.session_state["ai_studio_sample_name"] = sample_name
        st.session_state["ai_studio_sample_workspace_relative"] = workspace_relative
        st.session_state["ai_studio_sample_identity"] = sample_identity
        plan = _sample_source_plan(
            sample_name,
            frame.columns,
            workspace_relative=workspace_relative,
        )
        st.session_state["ai_studio_sample_source_plan"] = plan
        record_event(
            st.session_state,
            event=AuthoringEvent.SAMPLE_CHOSEN,
            workflow=_studio_authoring_workflow(),
            stage=AuthoringStage.SAMPLE,
            outcome=AuthoringOutcome.SUCCESS,
            once=True,
        )
        return frame.head(limit)
    except Exception as exc:
        _log_ai_operation_failure("Sample read", exc)
        st.error(f"Could not preview this sample. {_sample_error_message(exc)}")
        return None


def _create_demo_sample(workspace: Path) -> str:
    """Create the built-in sample after an explicit user click."""

    relative = Path("data/studio/value_stream_demo.csv")
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    _demo_sample().write_csv(temporary)
    os.replace(temporary, destination)
    return relative.as_posix()


def _demo_sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CustomerID": ["C-01", "C-02", "C-03", "C-01", "C-04", "C-02"],
            "OutcomeTime": [
                "2026-07-01T09:00:00Z",
                "2026-07-01T10:00:00Z",
                "2026-07-02T09:30:00Z",
                "2026-07-03T12:00:00Z",
                "2026-07-03T14:30:00Z",
                "2026-07-04T08:45:00Z",
            ],
            "Outcome": ["Clicked", "Impression", "Clicked", "Impression", "Clicked", "Clicked"],
            "Channel": ["Web", "Email", "Web", "Mobile", "Email", "Mobile"],
        }
    )


def _sample_source_plan(
    file_name: str,
    columns: list[str] | tuple[str, ...],
    *,
    workspace_relative: str = "",
) -> SampleSourcePlan:
    """Infer an honest preview/runtime plan without assuming every file is Pega."""

    capability = _sample_format_capability(file_name)
    column_names = {str(column).casefold() for column in columns}
    pega_like = bool(
        {"outcome", "outcometime"}.issubset(column_names)
        or {"pyoutcome", "pxoutcometime"}.issubset(column_names)
        or "pxinteractionid" in column_names
    )
    root = _sample_runtime_root(workspace_relative)
    suffix = Path(file_name).suffix
    pattern = f"**/*{suffix}" if suffix else Path(file_name).name
    if capability and capability.key == "csv":
        return SampleSourcePlan(
            "CSV",
            "sample",
            "csv",
            root,
            pattern,
            timestamp_format="%+" if Path(file_name).name == "value_stream_demo.csv" else "",
            production_ready=bool(workspace_relative),
            note="Preview and runtime both use the CSV reader.",
        )
    if capability and capability.key == "parquet":
        return SampleSourcePlan(
            "Parquet",
            "sample",
            "parquet",
            root,
            pattern,
            timestamp_format="%Y%m%dT%H%M%S%.3f %Z" if pega_like else "",
            production_ready=bool(workspace_relative),
            note="Preview and runtime both use the Parquet reader.",
        )
    if capability and capability.key in {"json", "ndjson", "zip", "gzip"}:
        return SampleSourcePlan(
            capability.label,
            "ih" if pega_like else "sample",
            "pega_ds_export",
            root,
            pattern,
            group_pattern=r"\d{8}(?=\d{6}_)" if pega_like else "",
            timestamp_format="%Y%m%dT%H%M%S%.3f %Z" if pega_like else "",
            production_ready=bool(workspace_relative) and pega_like,
            requires_runtime_confirmation=not pega_like,
            note=(
                "The schema looks like a Pega interaction export; preview and runtime use "
                "the Pega DS reader."
                if pega_like
                else "Preview is supported, but the built-in runtime JSON/archive reader is "
                "Pega-specific. Confirm Pega compatibility or convert the source to CSV/Parquet."
            ),
        )
    return SampleSourcePlan(
        "Unsupported",
        "sample",
        "csv",
        root,
        pattern,
        requires_runtime_confirmation=True,
        note="Convert this sample to CSV or Parquet before building a runtime source.",
    )


def _sample_runtime_root(workspace_relative: str) -> str:
    """Infer a dataset root above any Hive-style ``key=value`` partitions."""

    if not workspace_relative:
        return "data/studio"
    root = Path(workspace_relative).parent
    key, separator, value = root.name.partition("=")
    while key and separator and value:
        root = root.parent
        key, separator, value = root.name.partition("=")
    return root.as_posix()


def _sample_error_message(exc: Exception) -> str:
    if isinstance(exc, zipfile.BadZipFile):
        return "The ZIP archive is corrupt or incomplete."
    if isinstance(exc, SamplePreviewLimitError):
        return str(exc)
    message = str(exc)
    if "JSON or NDJSON" in message or "Unsupported sample format" in message:
        return message
    labels = ", ".join(capability.label for capability in SAMPLE_FORMAT_CAPABILITIES)
    return f"Check that the file matches one of the supported formats: {labels}."


def _sample_format_capability(file_name: str) -> SampleFormatCapability | None:
    lower = Path(file_name).name.casefold()
    return next(
        (
            capability
            for capability in SAMPLE_FORMAT_CAPABILITIES
            if any(lower.endswith(suffix) for suffix in capability.suffixes)
        ),
        None,
    )


def _sample_upload_extensions() -> list[str]:
    return builder.dedupe(
        [
            extension
            for capability in SAMPLE_FORMAT_CAPABILITIES
            for extension in capability.upload_extensions
        ]
    )


def _workspace_sample_files(workspace: Path) -> list[str]:
    data_root = workspace / "data"
    if not data_root.is_dir():
        return []
    return [
        path.relative_to(workspace).as_posix()
        for path in sorted(data_root.rglob("*"))
        if path.is_file() and _sample_format_capability(path.name) is not None
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


def _workspace_sample_identity(path: Path, workspace_relative: str) -> str:
    """Identify a workspace sample without allocating a full-file byte copy."""

    stat = path.stat()
    payload = {
        "relative_path": workspace_relative,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _uploaded_sample_bytes(upload: Any) -> bytes:
    """Reject an oversized upload before asking Streamlit to allocate its payload."""

    size = getattr(upload, "size", None)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise SamplePreviewLimitError(
            "Could not determine this upload's size safely. Place the file under workspace "
            "data/ and choose it with Use workspace sample."
        )
    _validate_buffered_preview_size(size, source="Uploads")
    data = upload.getvalue()
    _validate_buffered_preview_size(len(data), source="Uploads")
    return data


def _validate_buffered_preview_size(size: int, *, source: str) -> None:
    if size <= AI_STUDIO_UPLOAD_MAX_BYTES:
        return
    raise SamplePreviewLimitError(
        f"{source} are limited to 64 MiB for an in-memory preview. For larger CSV or "
        "Parquet files, place the file under workspace data/ and use the workspace picker; "
        "convert larger JSON or archive samples to CSV or Parquet first."
    )


def _read_workspace_sample(
    path: Path,
    *,
    limit: int,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Read a bounded workspace preview with format-native pushdown when available."""

    lower = path.name.casefold()
    if lower.endswith(".parquet"):
        preview = pl.scan_parquet(path)
        if columns is not None:
            preview = preview.select(list(columns))
        return preview.head(limit).collect()
    if lower.endswith(".csv"):
        return pl.read_csv(path, infer_schema_length=500, n_rows=limit)
    _validate_buffered_preview_size(path.stat().st_size, source="Workspace JSON/archive files")
    return _read_sample_bytes(path.name, path.read_bytes(), limit=limit)


def _read_sample_bytes(file_name: str, data: bytes, *, limit: int | None = None) -> pl.DataFrame:
    capability = _sample_format_capability(file_name)
    if capability is None:
        raise ValueError(
            f"Unsupported sample format for {Path(file_name).name!r}. "
            f"Choose one of: {', '.join(_sample_upload_extensions())}."
        )
    _validate_buffered_preview_size(len(data), source="Buffered sample files")
    if capability.key == "csv":
        return pl.read_csv(BytesIO(data), infer_schema_length=500, n_rows=limit)
    if capability.key == "parquet":
        return pl.read_parquet(BytesIO(data), n_rows=limit)
    if capability.key in {"json", "ndjson"}:
        return _read_json_payload(data, limit=limit)
    if capability.key == "gzip":
        return _read_json_payload(_bounded_gzip_payload(data), limit=limit)
    if capability.key == "zip":
        rows = _bounded_zip_records(data, capability=capability, limit=limit)
        frame = pl.from_dicts(rows) if rows else pl.DataFrame()
        return frame
    raise AssertionError(f"No preview reader registered for {capability.key!r}.")


def _bounded_gzip_payload(data: bytes) -> bytes:
    if len(data) >= 4:
        declared_size = int.from_bytes(data[-4:], byteorder="little")
        if declared_size > AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES:
            raise SamplePreviewLimitError(
                "The gzip sample exceeds the 128 MiB expanded preview budget. Convert it "
                "to Parquet or CSV and use a workspace sample."
            )
    with gzip.GzipFile(fileobj=BytesIO(data)) as compressed:
        payload = compressed.read(AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES + 1)
    if len(payload) > AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES:
        raise SamplePreviewLimitError(
            "The gzip sample exceeds the 128 MiB expanded preview budget. Convert it to "
            "Parquet or CSV and use a workspace sample."
        )
    return payload


def _bounded_zip_records(
    data: bytes,
    *,
    capability: SampleFormatCapability,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(BytesIO(data)) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if info.filename.casefold().endswith(capability.archive_member_suffixes)
            ),
            key=lambda info: info.filename.casefold(),
        )
        if not members:
            raise ValueError("ZIP samples must contain at least one JSON or NDJSON file.")
        if len(members) > AI_STUDIO_ARCHIVE_MAX_MEMBERS:
            raise SamplePreviewLimitError(
                "ZIP previews support at most 64 JSON/NDJSON members. Split the archive or "
                "convert it to Parquet or CSV."
            )
        if any(info.flag_bits & 0x1 for info in members):
            raise SamplePreviewLimitError(
                "Encrypted ZIP members cannot be previewed. Provide an unencrypted sample."
            )
        declared_size = sum(info.file_size for info in members)
        if declared_size > AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES:
            raise SamplePreviewLimitError(
                "The ZIP sample exceeds the 128 MiB expanded preview budget. Split the "
                "archive or convert it to Parquet or CSV."
            )

        expanded_size = 0
        for member in members:
            if limit is not None and len(rows) >= limit:
                break
            remaining_budget = AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES - expanded_size
            with archive.open(member) as member_file:
                payload = member_file.read(remaining_budget + 1)
            expanded_size += len(payload)
            if expanded_size > AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES:
                raise SamplePreviewLimitError(
                    "The ZIP sample exceeds the 128 MiB expanded preview budget. Split the "
                    "archive or convert it to Parquet or CSV."
                )
            remaining_rows = None if limit is None else limit - len(rows)
            rows.extend(_json_records(payload, limit=remaining_rows))
    return rows


def _read_json_payload(data: bytes, *, limit: int | None = None) -> pl.DataFrame:
    records = _json_records(data, limit=limit)
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


def _json_records(data: bytes, *, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        return []
    text = data.decode("utf-8")
    position = _skip_json_whitespace(text, 0)
    if position >= len(text):
        return []
    if text[position] == "[":
        return _json_array_records(text, position=position + 1, limit=limit)

    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    while position < len(text):
        loaded, position = decoder.raw_decode(text, position)
        if isinstance(loaded, dict):
            rows.append(loaded)
            if limit is not None and len(rows) >= limit:
                return rows
        position = _skip_json_whitespace(text, position)
    return rows


def _json_array_records(
    text: str,
    *,
    position: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    position = _skip_json_whitespace(text, position)
    while position < len(text):
        if text[position] == "]":
            trailing = _skip_json_whitespace(text, position + 1)
            if trailing != len(text):
                raise json.JSONDecodeError("Extra data", text, trailing)
            return rows
        loaded, position = decoder.raw_decode(text, position)
        if isinstance(loaded, dict):
            rows.append(loaded)
            if limit is not None and len(rows) >= limit:
                return rows
        position = _skip_json_whitespace(text, position)
        if position < len(text) and text[position] == ",":
            position = _skip_json_whitespace(text, position + 1)
            if position >= len(text) or text[position] == "]":
                raise json.JSONDecodeError("Expecting value", text, position)
            continue
        if position < len(text) and text[position] == "]":
            continue
        raise json.JSONDecodeError("Expecting ',' delimiter", text, position)
    raise json.JSONDecodeError("Expecting ']'", text, position)


def _skip_json_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _initialize_state(sample: pl.DataFrame) -> None:  # noqa: PLR0915
    signature = (
        str(st.session_state.get("ai_studio_sample_identity") or ""),
        tuple((name, str(dtype)) for name, dtype in sample.schema.items()),
    )
    if st.session_state.get("ai_studio_sample_signature") == signature:
        return
    st.session_state["ai_studio_sample_signature"] = signature
    # Business requirements describe intent, not the sample, so they survive sample changes.
    st.session_state.setdefault("ai_studio_user_goals", "")
    plan = st.session_state.get("ai_studio_sample_source_plan")
    if not isinstance(plan, SampleSourcePlan):
        plan = _sample_source_plan(
            _sample_file_name(),
            sample.columns,
            workspace_relative=str(
                st.session_state.get("ai_studio_sample_workspace_relative") or ""
            ),
        )
    st.session_state["ai_studio_source_id"] = plan.source_id
    st.session_state["ai_studio_reader_kind"] = plan.reader_kind
    st.session_state["ai_studio_reader_root"] = plan.root
    st.session_state["ai_studio_file_pattern"] = plan.file_pattern
    st.session_state["ai_studio_group_pattern"] = plan.group_pattern
    st.session_state["ai_studio_streaming"] = plan.reader_kind == "pega_ds_export"
    st.session_state["ai_studio_hive_partitioning"] = False
    st.session_state["ai_studio_timestamp_format"] = plan.timestamp_format
    st.session_state["ai_studio_subject"] = _default_subject_column(sample.columns)
    st.session_state["ai_studio_outcome_time"] = _default_time_column(sample.columns, "OutcomeTime")
    st.session_state["ai_studio_decision_time"] = _default_time_column(
        sample.columns, "DecisionTime", fallback=False
    )
    st.session_state["ai_studio_outcome_column"] = _default_outcome_column(sample)
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
    st.session_state["ai_studio_defaults"] = []
    st.session_state["ai_studio_filter_rows"] = []
    st.session_state["ai_studio_filter_mode"] = "Rules"
    st.session_state["ai_studio_raw_filter"] = ""
    st.session_state["ai_studio_calculations"] = []
    st.session_state[AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = False
    st.session_state.pop(AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY, None)
    st.session_state.pop(AI_STUDIO_RENAME_CAPITALIZE_WIDGET_KEY, None)
    st.session_state["ai_studio_rename_capitalize_applied"] = False
    st.session_state["ai_studio_approved_fields"] = []
    st.session_state["ai_studio_example_fields"] = []
    st.session_state["ai_studio_group_by_fields"] = []
    st.session_state["ai_studio_field_approval_initialized"] = False
    _clear_ai_sharing_confirmation()
    _clear_schema_widget_state()
    st.session_state["ai_studio_draft"] = None
    st.session_state["ai_studio_draft_source"] = ""
    st.session_state["ai_studio_pending_draft"] = None
    st.session_state["ai_studio_pending_base_draft"] = None
    st.session_state["ai_studio_pending_kind"] = ""
    st.session_state["ai_studio_pending_prompt"] = ""
    st.session_state["ai_studio_last_ai_response"] = ""
    st.session_state.pop("ai_studio_last_candidate_failure", None)
    st.session_state["ai_studio_raw_metrics_yaml"] = ""
    st.session_state["ai_studio_raw_dashboards_yaml"] = ""
    st.session_state["ai_studio_copilot_history"] = []
    st.session_state["ai_studio_copilot_questions"] = []
    st.session_state["ai_studio_copilot_last_prompt"] = ""
    st.session_state.pop("ai_studio_copilot_queued_message", None)
    st.session_state["ai_studio_coverage_rows"] = []
    st.session_state["ai_studio_coverage_signature"] = ""
    st.session_state["ai_studio_published_signature"] = ""
    st.session_state["ai_studio_reviewed_signature"] = ""
    st.session_state["ai_studio_validation_cache"] = {}
    st.session_state["ai_studio_outcome_receipt"] = None
    st.session_state[AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY] = False


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
        c1, c2 = st.columns(2, gap="small")
        st.session_state["ai_studio_source_id"] = c1.text_input(
            "Source ID",
            value=st.session_state["ai_studio_source_id"],
        )
        st.session_state["ai_studio_reader_kind"] = c2.selectbox(
            "Reader",
            ["pega_ds_export", "parquet", "csv", "xlsx"],
            index=["pega_ds_export", "parquet", "csv", "xlsx"].index(
                st.session_state["ai_studio_reader_kind"]
            ),
            help=config_help.field_help("source.reader"),
        )
        c3, c4 = st.columns(2, gap="small")
        st.session_state["ai_studio_reader_root"] = c3.text_input(
            "Workspace Root",
            value=str(st.session_state.get("ai_studio_reader_root") or "data"),
            help="Path relative to the workspace. Runtime discovery is constrained to this root.",
        )
        st.session_state["ai_studio_file_pattern"] = c4.text_input(
            "File Pattern",
            value=st.session_state["ai_studio_file_pattern"],
            help=config_help.field_help("source.file_pattern"),
        )
        c5, c6 = st.columns(2, gap="small")
        st.session_state["ai_studio_group_pattern"] = c5.text_input(
            "Group Pattern",
            value=st.session_state["ai_studio_group_pattern"],
            help=config_help.field_help("source.group_pattern"),
        )
        st.session_state["ai_studio_timestamp_format"] = c6.text_input(
            "Timestamp Format",
            value=st.session_state["ai_studio_timestamp_format"],
            help=config_help.field_help("source.timestamp_format"),
        )
        c7, c8 = st.columns(2, gap="small")
        st.session_state["ai_studio_streaming"] = c7.toggle(
            "Streaming",
            value=st.session_state["ai_studio_streaming"],
            help=config_help.field_help("source.streaming"),
        )
        st.session_state["ai_studio_hive_partitioning"] = c8.toggle(
            "Hive Partitioned",
            value=st.session_state["ai_studio_hive_partitioning"],
            help=config_help.field_help("source.hive_partitioning"),
        )
        _render_rename_capitalize_toggle()
        if _rename_capitalize_enabled():
            st.caption(
                "`rename_capitalize` converts source columns to the legacy Pega-aware "
                "capitalized schema, for example `pyName` to `Name`."
            )

        with st.expander(
            "Technical details · planned transforms",
            expanded=_rename_capitalize_enabled(),
        ):
            st.code(
                yaml.safe_dump(
                    {
                        "transforms": (
                            [{"kind": "rename_capitalize"}] if _rename_capitalize_enabled() else []
                        )
                    },
                    sort_keys=False,
                ),
                language="yaml",
            )
    _set_effective_schema_state(schema_sample)
    _sample_preview(raw_sample, schema_sample)
    with components.bordered_panel(
        "Current Workspace Sources", "Existing catalog sources are shown for context."
    ):
        rows = [
            {
                "Source": source.description or builder.title_from_identifier(source.id),
                "Reader": builder.title_from_identifier(source.reader.kind),
                "Pattern": source.reader.file_pattern,
                "Processors": len(processors),
            }
            for source in ctx.catalog.pipelines.sources
            for processors in [
                [p for p in ctx.catalog.processors.processors if p.source == source.id]
            ]
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        with st.expander("Technical details · source IDs", expanded=False):
            components.key_value_strip(
                [
                    {
                        "label": source.description or builder.title_from_identifier(source.id),
                        "value": source.id,
                    }
                    for source in ctx.catalog.pipelines.sources
                ]
            )


def _stage_uploaded_sample(workspace: Path) -> str:
    """Persist an in-memory upload only after explicit user confirmation."""

    data = st.session_state.get("ai_studio_sample_bytes")
    if not isinstance(data, bytes) or not data:
        raise ValueError("The uploaded sample is no longer available. Upload it again.")
    file_name = Path(_sample_file_name()).name
    relative = Path("data/studio") / file_name
    destination = workspace / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return relative.as_posix()


def _required_fields(sample: pl.DataFrame, working: pl.DataFrame) -> None:
    with components.bordered_panel(
        "Required Field Mapping",
        "Choose fields from the active schema. Studio never accepts an unknown column name here.",
    ):
        c1, c2 = st.columns(2)
        _schema_field_selector(
            c1,
            "Subject ID Field",
            "ai_studio_subject",
            sample,
            help=config_help.field_help("mapping.subject"),
        )
        _schema_field_selector(
            c2,
            "Outcome Field",
            "ai_studio_outcome_column",
            sample,
        )
        c3, c4 = st.columns(2)
        _schema_field_selector(
            c3,
            "Outcome Timestamp",
            "ai_studio_outcome_time",
            sample,
        )
        _schema_field_selector(
            c4,
            "Decision Timestamp",
            "ai_studio_decision_time",
            sample,
        )
        st.caption("Date fields (if available)")
        c5, c6 = st.columns(2)
        _schema_field_selector(
            c5,
            "Day",
            "ai_studio_day_column",
            sample,
        )
        _schema_field_selector(
            c6,
            "Month",
            "ai_studio_month_column",
            sample,
        )
        c7, c8 = st.columns(2)
        _schema_field_selector(
            c7,
            "Quarter",
            "ai_studio_quarter_column",
            sample,
        )
        _schema_field_selector(
            c8,
            "Year",
            "ai_studio_year_column",
            sample,
        )
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


def _schema_field_selector(
    container: Any,
    label: str,
    state_key: str,
    sample: pl.DataFrame,
    *,
    help: str | None = None,
) -> str:
    options = ["", *[str(column) for column in sample.columns]]
    option_labels = {
        column: (
            f"{column} · {sample.schema[column]} · "
            f"{sample.get_column(column).null_count():,} null · "
            f"{sample.get_column(column).n_unique():,} unique"
        )
        for column in sample.columns
    }
    current = str(st.session_state.get(state_key) or "")
    if current not in options:
        current = ""
    selected = container.selectbox(
        label,
        options,
        index=options.index(current),
        format_func=lambda value: option_labels.get(value, "Not mapped"),
        key=_schema_widget_key(f"{state_key}_selector"),
        help=help,
    )
    st.session_state[state_key] = selected
    return selected


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
            components.pinned_editor_input(editor_key, default_frame),
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
        _render_stale_preprocessing_field_feedback("Defaults")
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
    _persist_ai_studio_checkpoint()


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
            filter_editor_key = _schema_widget_key("ai_studio_filter_editor")
            edited = st.data_editor(
                components.pinned_editor_input(filter_editor_key, filter_frame),
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key=filter_editor_key,
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
            try:
                builder.compile_filter_rows(st.session_state["ai_studio_filter_rows"])
            except ValueError as exc:
                st.error(f"Filters: {exc}")
        else:
            try:
                compiled = builder.compile_filter_rows(st.session_state["ai_studio_filter_rows"])
            except ValueError as exc:
                compiled = None
                st.error(f"Filters: {exc}")
            st.code(builder.expression_yaml(compiled) or "{}", language="yaml")
        _render_stale_preprocessing_field_feedback("Filters")
    _persist_ai_studio_checkpoint()


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
        calculation_editor_key = _schema_widget_key("ai_studio_calculation_editor")
        edited = st.data_editor(
            components.pinned_editor_input(calculation_editor_key, calculation_frame),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=calculation_editor_key,
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
        _render_stale_preprocessing_field_feedback("Calculations")
        try:
            transforms = builder.build_derive_column_transforms(
                st.session_state["ai_studio_calculations"]
            )
            with st.expander("Technical details · generated transforms", expanded=False):
                st.code(
                    yaml.safe_dump({"transforms": transforms}, sort_keys=False),
                    language="yaml",
                )
        except Exception as exc:
            _log_ai_operation_failure("Calculated transform preview", exc)
            st.error(str(exc))
    _persist_ai_studio_checkpoint()


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
        editor_key = _schema_widget_key(f"ai_studio_field_approval_editor_{normalized_query}")
        edited = st.data_editor(
            editor_frame,
            width="stretch",
            hide_index=True,
            height=520,
            # The key scopes stored edits to one filter view; the checkbox
            # state itself lives in the approval session keys below.
            key=editor_key,
            on_change=_on_field_approval_editor_change,
            args=(
                editor_key,
                tuple(visible_fields),
                tuple(available_fields),
                tuple(required),
            ),
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
    _persist_ai_studio_checkpoint()


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
    if _builder_source_handoff():
        try:
            baseline = _builder_source_addition_draft(ctx, baseline)
        except ValueError as exc:
            st.error(str(exc))
            st.info("Return to the Sample step and choose a unique Source ID.")
            return
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
    if action_col1.button("Review deterministic draft", type="secondary"):
        ok, issues = _validate_draft_for_active_schema(baseline, approved_fields)
        if ok:
            if ai_calls_enabled:
                _queue_pending_candidate(
                    DraftCandidateResult(draft=baseline, issues=(), attempts=0),
                    base_draft=_draft_review_base(baseline),
                    kind="deterministic draft",
                    prompt="Generated locally from the approved schema; no provider call was made.",
                )
                st.rerun()
            else:
                _set_draft(baseline, reviewed=True)
                st.success("Deterministic draft accepted for review.")
        else:
            st.error(f"The deterministic draft has {len(issues)} blocking issue(s).")
    if ai_calls_enabled and action_col2.button(
        _ai_retry_label("draft", "Generate AI Draft"),
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
                result = _run_validated_ai_candidate(
                    settings=ai_settings,
                    operation="catalog_draft",
                    prompt=prompt,
                    base_draft=baseline,
                    approved_fields=approved_fields,
                    repair_prompt_factory=_candidate_repair_prompt_factory(
                        working, approved_fields
                    ),
                    status=status,
                )
                queued = _queue_pending_candidate(
                    result,
                    base_draft=_draft_review_base(result.draft or baseline),
                    kind="draft",
                    prompt=prompt,
                )
                status.update(
                    label=(
                        f"Validated draft ready after {_attempt_count_label(result.attempts)}"
                        if queued
                        else f"No valid draft after {_attempt_count_label(result.attempts)}"
                    ),
                    state="complete" if queued else "error",
                )
            if queued:
                st.rerun()
            _render_candidate_failure(result, operation_label="AI draft")
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _log_ai_operation_failure("AI draft generation", exc)
            _render_ai_operation_error(exc, operation="AI draft generation")
    if ai_calls_enabled:
        with action_col3, st.popover("Show Prompt", icon=":material/psychology:"):
            st.code(prompt, language="text")
    else:
        action_col2.info("AI generation is disabled here; use AI Config Studio for model drafts.")

    current = st.session_state.get("ai_studio_draft")
    if current:
        if _current_candidate_failure() is not None:
            st.info(
                "The current accepted draft was preserved and is separate from the failed "
                f"model response. Workspace hash: `{ctx.catalog_hash}`."
            )
        else:
            st.success(f"Draft accepted for review. Current workspace hash: `{ctx.catalog_hash}`.")
        _draft_counts(current)
    elif ai_calls_enabled and not ai_settings:
        st.info("Configure a LiteLLM model in the sidebar or use the deterministic draft.")


def _studio_apply_readiness(  # noqa: PLR0912
    ctx: ValueStreamContext,
    draft: dict[str, Any] | None,
    approved_fields: list[str],
    preprocessing_error: str | None,
    *,
    ai_calls_enabled: bool | None = None,
) -> StudioReadinessSnapshot:
    """Build the one readiness contract used by Apply and Export."""

    steps = _studio_steps(
        ai_calls_enabled=_ai_calls_enabled() if ai_calls_enabled is None else ai_calls_enabled
    )
    provider_enabled = _ai_calls_enabled() if ai_calls_enabled is None else ai_calls_enabled
    pending = st.session_state.get("ai_studio_pending_draft") is not None
    snapshot = _draft_validation_snapshot(draft) if isinstance(draft, dict) else None
    signature = snapshot.signature if snapshot is not None else ""
    reviewed = bool(
        snapshot
        and snapshot.ok
        and st.session_state.get("ai_studio_reviewed_signature") == signature
    )
    published = bool(
        snapshot
        and snapshot.ok
        and st.session_state.get("ai_studio_published_signature") == signature
    )
    source_ready = _production_source_ready()
    replacement = (
        _workspace_replacement_impact(getattr(ctx, "catalog", None), draft)
        if isinstance(draft, dict) and snapshot and snapshot.ok and not published
        else {}
    )
    replacement_confirmed = bool(
        not replacement or st.session_state.get(AI_REPLACEMENT_CONFIRM_STATE_KEY) == signature
    )

    counts = draft_object_counts(draft) if isinstance(draft, dict) else {}
    artifact_counts = {
        "Data": (
            f"{int(counts.get('Sources', 0))} source(s) · {len(approved_fields)} approved field(s)"
        ),
        "Processor": f"{int(counts.get('Processors', 0))} processor(s)",
        "Metric": f"{int(counts.get('Metrics', 0))} metric(s)",
        "Report": (
            f"{int(counts.get('Dashboards', 0))} dashboard(s) · "
            f"{int(counts.get('Tiles', 0))} tile(s)"
        ),
        "Provider": "Provider enabled" if provider_enabled else "Deterministic mode",
        "Runtime": "1 accepted revision" if snapshot else "0 accepted revisions",
    }
    revision_label = f"Accepted revision {signature[:12]}" if signature else "Not committed"
    last_changes = {
        "Data": (
            "Approved-field selection in this session" if approved_fields else "Not committed"
        ),
        "Processor": revision_label,
        "Metric": revision_label,
        "Report": revision_label,
        "Provider": "Provider settings in this session" if provider_enabled else "Not required",
        "Runtime": (
            f"Applied revision {signature[:12]}"
            if published
            else revision_label
            if signature
            else "Not committed"
        ),
    }

    findings: list[StudioReadinessIssue] = []
    if preprocessing_error:
        findings.append(
            StudioReadinessIssue(
                area="Data",
                severity="blocker",
                object_path="sample.preprocessing",
                current_value="The working sample could not be prepared safely.",
                expected_contract="Preprocessing completes with a valid working schema.",
                remediation="Open the indicated Data step and correct the mapping or transform.",
                target_step=_preprocessing_fix_step(preprocessing_error, steps),
            )
        )
    if not isinstance(draft, dict):
        findings.append(
            StudioReadinessIssue(
                area="Runtime",
                severity="blocker",
                object_path="draft.accepted_revision",
                current_value="No accepted revision",
                expected_contract="One validated draft is accepted into the current session.",
                remediation="Generate or review the deterministic draft, then accept it.",
                target_step=steps[6],
            )
        )
    else:
        if pending:
            findings.append(
                StudioReadinessIssue(
                    area="Provider",
                    severity="blocker",
                    object_path="draft.pending_proposal",
                    current_value="A proposal is waiting for review.",
                    expected_contract="No unreviewed provider proposal remains pending.",
                    remediation="Accept or discard the pending proposal on the Draft step.",
                    target_step=steps[6],
                )
            )
        if snapshot is not None and snapshot.issues:
            blocking, _repairable = classify_draft_validation_issues(list(snapshot.issues))
            blocking_set = set(blocking)
            for issue in snapshot.issues:
                area = _readiness_area_for_validation_issue(issue)
                findings.append(
                    StudioReadinessIssue(
                        area=area,
                        severity="blocker" if issue in blocking_set else "warning",
                        object_path=_readiness_issue_path(issue),
                        current_value=_safe_readiness_current_value(issue),
                        expected_contract=_readiness_expected_contract(area),
                        remediation=_readiness_remediation(area),
                        target_step=_readiness_target_step(area, steps),
                    )
                )
        if snapshot is not None and snapshot.ok and not pending:
            if not reviewed:
                findings.append(
                    StudioReadinessIssue(
                        area="Runtime",
                        severity="blocker",
                        object_path="draft.reviewed_revision",
                        current_value="The accepted revision is not marked reviewed.",
                        expected_contract="The exact validated revision has explicit business review.",
                        remediation="Mark this revision reviewed on the Apply step.",
                        target_step=steps[13],
                    )
                )
            elif not source_ready:
                findings.append(
                    StudioReadinessIssue(
                        area="Runtime",
                        severity="blocker",
                        object_path="source.runtime_reader",
                        current_value="Preview-only or staged reader settings do not match.",
                        expected_contract="The runtime reader, root, and file pattern match the staged source plan.",
                        remediation="Return to Sample and stage the file or align the runtime source settings.",
                        target_step=steps[0],
                        runtime_only=True,
                    )
                )
            elif not replacement_confirmed:
                findings.append(
                    StudioReadinessIssue(
                        area="Runtime",
                        severity="blocker",
                        object_path="workspace.replacement_consent",
                        current_value=(
                            f"{sum(len(names) for names in replacement.values())} existing "
                            "catalog object(s) would be removed."
                        ),
                        expected_contract="Removal is explicitly confirmed for this exact revision.",
                        remediation="Review the replacement disclosure and confirm the removal.",
                        target_step=steps[13],
                    )
                )

    validation_ok = bool(snapshot and snapshot.ok)
    apply_ready = bool(
        isinstance(draft, dict)
        and not preprocessing_error
        and not pending
        and validation_ok
        and reviewed
        and source_ready
        and replacement_confirmed
        and not published
    )
    export_ready = bool(
        isinstance(draft, dict) and not preprocessing_error and not pending and validation_ok
    )
    blocker_count = sum(finding.severity == "blocker" for finding in findings)
    warning_count = sum(finding.severity == "warning" for finding in findings)
    if published:
        apply_reason = "This revision is already applied to the active workspace."
    elif apply_ready:
        apply_reason = "All readiness contracts pass for this exact reviewed revision."
    else:
        apply_reason = _readiness_count_reason(
            blocker_count,
            warning_count,
            action="Apply",
        )
    if export_ready:
        export_reason = "Validated YAML is ready to download."
    else:
        export_reason = _readiness_count_reason(
            blocker_count,
            warning_count,
            action="Export",
        )
    return StudioReadinessSnapshot(
        issues=tuple(findings),
        artifact_counts=artifact_counts,
        last_changes=last_changes,
        apply_ready=apply_ready,
        apply_disabled_reason=apply_reason,
        export_ready=export_ready,
        export_disabled_reason=export_reason,
    )


def _readiness_count_reason(blockers: int, warnings: int, *, action: str) -> str:
    details: list[str] = []
    if blockers:
        details.append(f"{blockers} blocker{'s' if blockers != 1 else ''}")
    if warnings:
        details.append(f"{warnings} validation warning{'s' if warnings != 1 else ''}")
    joined = " and ".join(details) or "the unmet readiness contract"
    return f"{action} is unavailable: resolve {joined} shown in the readiness summary."


def _readiness_area_for_validation_issue(issue: str) -> str:
    normalized = issue.casefold()
    if any(token in normalized for token in ("dashboard", "report", "tile", "page")):
        return "Report"
    if "processor" in normalized:
        return "Processor"
    if "metric" in normalized:
        return "Metric"
    if any(
        token in normalized
        for token in ("pipeline", "source", "reader", "schema", "transform", "filter")
    ):
        return "Data"
    if any(token in normalized for token in ("provider", "model", "endpoint")):
        return "Provider"
    return "Runtime"


def _readiness_issue_path(issue: str) -> str:
    path, separator, _detail = issue.partition(":")
    return path.strip() if separator and path.strip() else "catalog"


def _safe_readiness_current_value(issue: str) -> str:
    _path, separator, detail = issue.partition(":")
    normalized = " ".join((detail if separator else issue).split()).strip()
    lowered = normalized.casefold()
    if not normalized:
        return "Invalid or missing catalog value"
    if any(secret in lowered for secret in ("api key", "api_key", "password", "bearer ")):
        return "Configured value is invalid or missing."
    if normalized.startswith(("/", "\\")) or "://" in normalized:
        return "Configured value is invalid; technical detail is available in validation."
    return normalized[:157] + "..." if len(normalized) > 160 else normalized


def _readiness_expected_contract(area: str) -> str:
    return {
        "Data": "Source and preprocessing configuration satisfies the catalog schema.",
        "Processor": "Processor configuration is valid and references an existing source.",
        "Metric": "Metric configuration is valid and references executable aggregate state.",
        "Report": "Report tiles reference valid metrics and chart-role fields.",
        "Provider": "Provider work is explicitly reviewed or is not required.",
        "Runtime": "The reviewed catalog can be applied through the guarded transaction.",
    }[area]


def _readiness_remediation(area: str) -> str:
    return {
        "Data": "Correct the source mapping or preprocessing definition in Data.",
        "Processor": "Open Processors and correct the affected processor definition.",
        "Metric": "Open Metrics and correct the affected metric definition.",
        "Report": "Open Reports Review and correct the affected tile or report.",
        "Provider": "Return to Draft and resolve the provider review state.",
        "Runtime": "Resolve the named review, source, or transaction prerequisite.",
    }[area]


def _readiness_target_step(area: str, steps: list[str]) -> str:
    return steps[
        {
            "Data": 1,
            "Processor": 7,
            "Metric": 8,
            "Report": 10,
            "Provider": 6,
            "Runtime": 13,
        }[area]
    ]


def _preprocessing_fix_step(error: str, steps: list[str]) -> str:
    normalized = error.casefold()
    if "filter" in normalized:
        return steps[3]
    if any(token in normalized for token in ("calculation", "expression", "derive")):
        return steps[4]
    if "default" in normalized:
        return steps[2]
    return steps[1]


def _readiness_area_status(
    area: str,
    snapshot: StudioReadinessSnapshot,
) -> str:
    area_issues = [issue for issue in snapshot.issues if issue.area == area]
    if area_issues:
        return "Attention"
    count_text = snapshot.artifact_counts.get(area, "")
    if count_text.startswith("0 ") and area not in {"Provider", "Runtime"}:
        return "Not started"
    return "Complete"


def _queue_readiness_fix(target_step: str) -> None:
    st.session_state["ai_studio_step"] = target_step
    st.session_state["ai_studio_jump_step"] = target_step
    st.session_state["ai_studio_next_step"] = target_step


def _render_apply_readiness(snapshot: StudioReadinessSnapshot) -> None:
    """Render grouped readiness evidence with actionable blocker targets."""

    st.write("#### Apply readiness")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        components.status_badge(
            f"{snapshot.blocker_count} blocker(s)",
            "blocked" if snapshot.blocker_count else "ready",
        )
        components.status_badge(
            f"{snapshot.warning_count} warning(s)",
            "warning" if snapshot.warning_count else "ready",
        )
    for area in STUDIO_READINESS_AREAS:
        area_issues = [issue for issue in snapshot.issues if issue.area == area]
        status = _readiness_area_status(area, snapshot)
        with st.expander(
            f"{area} · {status} · {snapshot.artifact_counts[area]}",
            expanded=bool(area_issues),
        ):
            st.caption(f"Last committed change: {snapshot.last_changes[area]}")
            if not area_issues:
                st.success("No readiness findings in this area.")
                continue
            for index, issue in enumerate(area_issues):
                message = (
                    f"**Object/path:** `{issue.object_path}`  \n"
                    f"**Current safe value:** {issue.current_value}  \n"
                    f"**Expected contract:** {issue.expected_contract}  \n"
                    f"**Remediation:** {issue.remediation}"
                )
                if issue.runtime_only:
                    message += "  \n**Scope:** Runtime-only readiness condition."
                if issue.severity == "blocker":
                    st.error(message)
                else:
                    st.warning(message)
                st.button(
                    "Jump to fix",
                    icon=":material/arrow_forward:",
                    key=f"ai_studio_readiness_jump_{area}_{index}",
                    on_click=_queue_readiness_fix,
                    args=(issue.target_step,),
                    help=f"Open {issue.target_step} without changing accepted work.",
                )


def _save_export(
    ctx: ValueStreamContext,
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
) -> None:
    components.render_validation_summary(ctx.validation.issues, ok=ctx.validation.ok)
    st.write("### Apply and export")
    raw_draft = st.session_state.get("ai_studio_draft")
    draft = raw_draft if isinstance(raw_draft, dict) else None
    readiness = _studio_apply_readiness(
        ctx,
        draft,
        approved_fields,
        preprocessing_error,
    )
    _render_apply_readiness(readiness)
    if preprocessing_error:
        st.warning("Resolve preprocessing errors before applying or exporting this revision.")
    if draft is None:
        st.info(
            "No accepted draft exists yet. Generate a deterministic or AI proposal, then "
            "review its complete change bundles before coming here."
        )
        _render_workspace_save_bar(ctx, readiness=readiness)
        st.write("#### Export YAML")
        st.warning(readiness.export_disabled_reason)
        if st.button("Go to Draft", icon=":material/arrow_back:"):
            st.session_state["ai_studio_next_step"] = _studio_steps(
                ai_calls_enabled=_ai_calls_enabled()
            )[6]
            st.rerun()
        return

    _render_outcome_receipt()
    _draft_counts(draft)
    snapshot = _draft_validation_snapshot(draft)
    _render_draft_validation(
        snapshot.ok,
        list(snapshot.issues),
        revision=snapshot.signature,
        expanded=not snapshot.ok,
    )
    _render_ai_repair_panel(draft, working, approved_fields, list(snapshot.issues))
    if st.session_state.get("ai_studio_pending_draft") is not None:
        _render_workspace_save_bar(ctx, readiness=readiness)
        st.write("#### Export YAML")
        st.warning(readiness.export_disabled_reason)
        return
    reviewed = bool(
        snapshot.ok and st.session_state.get("ai_studio_reviewed_signature") == snapshot.signature
    )
    if snapshot.ok and not reviewed:
        st.warning(
            "This exact revision has not been reviewed. Validation proves structural safety; "
            "review confirms the business intent."
        )
        if st.button("Mark this revision reviewed", type="primary"):
            st.session_state["ai_studio_reviewed_signature"] = snapshot.signature
            record_event(
                st.session_state,
                event=AuthoringEvent.REVIEWED,
                workflow=_studio_authoring_workflow(),
                stage=AuthoringStage.REVIEW,
                outcome=AuthoringOutcome.SUCCESS,
            )
            st.rerun()
    _render_workspace_save_bar(ctx, readiness=readiness)
    _render_coverage_panel(draft, working, approved_fields)
    files = _draft_files(draft)
    st.write("#### Export YAML")
    if readiness.export_ready:
        st.caption(
            "Downloads are available before raw YAML so the common path stays concise. "
            f"{readiness.export_disabled_reason}"
        )
    else:
        st.warning(readiness.export_disabled_reason)
    for filename, section in files.items():
        text = yaml.safe_dump(section, sort_keys=False)
        st.download_button(
            f"Download {filename}",
            data=text,
            file_name=filename,
            mime="text/yaml",
            key=f"ai_studio_download_{filename}",
            disabled=not readiness.export_ready,
            help=(None if readiness.export_ready else readiness.export_disabled_reason),
        )
        with st.expander(f"Technical details · {filename}", expanded=False):
            st.code(text, language="yaml")


def _render_workspace_save_bar(
    ctx: ValueStreamContext,
    *,
    readiness: StudioReadinessSnapshot | None = None,
) -> None:
    """Apply the accepted AI draft from one consistent final action."""

    feedback = st.session_state.pop("ai_studio_workspace_save_feedback", None)
    draft = st.session_state.get("ai_studio_draft")
    pending = st.session_state.get("ai_studio_pending_draft") is not None
    ok = False
    signature = ""
    if isinstance(draft, dict):
        snapshot = _draft_validation_snapshot(draft)
        ok = snapshot.ok
        signature = snapshot.signature
    if readiness is None:
        readiness = _studio_apply_readiness(
            ctx, draft if isinstance(draft, dict) else None, [], None
        )

    published = bool(
        isinstance(draft, dict)
        and ok
        and st.session_state.get("ai_studio_published_signature") == _draft_signature(draft)
    )
    replacement: dict[str, list[str]] = {}
    if isinstance(draft, dict) and ok and not published:
        replacement = _workspace_replacement_impact(getattr(ctx, "catalog", None), draft)
    replacement_confirmed = (
        not replacement or st.session_state.get(AI_REPLACEMENT_CONFIRM_STATE_KEY) == signature
    )
    if replacement and not pending:
        _render_replacement_disclosure(replacement, signature, confirmed=replacement_confirmed)
    caption = readiness.apply_disabled_reason
    st.caption(f"Apply status: {caption}")

    if components.editor_save_bar(
        key="ai_studio_workspace_save",
        caption=caption,
        label="Applied" if published else "Apply to workspace",
        disabled=not readiness.apply_ready,
        help=(
            "Writes the reviewed catalog revision only. Running source data is a separate, "
            "explicit action on the final step."
        ),
    ):
        try:
            requires_data_run = _draft_requires_data_run(ctx, draft)
            _apply_draft(ctx, draft)
            _mark_draft_published(draft)
            _discard_ai_studio_checkpoint()
            workspace_ok, workspace_issues = builder.validate_workspace(
                ctx.workspace,
                source_columns_by_id=_active_catalog_source_columns(),
            )
            st.session_state["ai_studio_workspace_save_feedback"] = {
                "ok": workspace_ok,
                "issues": workspace_issues,
            }
            st.session_state["ai_studio_outcome_receipt"] = {
                "revision": _draft_signature(draft)[:12],
                "applied": workspace_ok,
                "requires_data_run": requires_data_run,
                "run_status": "not_started" if requires_data_run else "not_required",
                "source_count": len(draft.get("pipelines", {}).get("sources", []) or []),
                "removed_object_count": sum(len(names) for names in replacement.values()),
            }
            record_event(
                st.session_state,
                event=AuthoringEvent.APPLIED,
                workflow=_studio_authoring_workflow(),
                stage=AuthoringStage.APPLY,
                outcome=AuthoringOutcome.SUCCESS if workspace_ok else AuthoringOutcome.BLOCKED,
                requires_data_run=requires_data_run,
            )
            st.rerun()
        except Exception as exc:  # pragma: no cover - Streamlit display path
            record_event(
                st.session_state,
                event=AuthoringEvent.FAILED,
                workflow=_studio_authoring_workflow(),
                stage=AuthoringStage.APPLY,
                outcome=_authoring_failure_outcome(exc),
            )
            _log_ai_operation_failure("AI draft workspace save", exc)
            st.toast(
                "The revision could not be applied. The workspace transaction was rolled back.",
                icon=":material/error:",
            )
    if isinstance(feedback, dict):
        if feedback.get("ok"):
            st.toast(
                "Revision applied to the workspace and the catalog validates.",
                icon=":material/check_circle:",
            )
        else:
            st.toast(
                "The revision was written, but the workspace catalog needs attention.",
                icon=":material/warning:",
            )


def _workspace_replacement_impact(
    catalog: Any,
    draft: dict[str, Any],
) -> dict[str, list[str]]:
    """Return active-catalog objects an apply of this draft would remove.

    ``_apply_draft`` replaces every catalog section wholesale, so a draft that
    was not built from the current workspace silently deletes anything it does
    not include. Review bundles diff draft revisions against the draft's own
    baseline and can never surface these removals, so the apply action must.
    """

    if catalog is None:
        return {}

    def draft_ids(section: Any, key: str) -> set[str]:
        items = section.get(key) if isinstance(section, dict) else None
        if isinstance(items, dict):
            return {str(name) for name in items}
        if isinstance(items, list):
            return {
                str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")
            }
        return set()

    draft_sources = draft_ids(draft.get("pipelines"), "sources")
    draft_processors = draft_ids(draft.get("processors"), "processors")
    draft_metrics = draft_ids(draft.get("metrics"), "metrics")
    draft_dashboards = draft_ids(draft.get("dashboards"), "dashboards")
    impact = {
        "sources": sorted(
            source.id for source in catalog.pipelines.sources if source.id not in draft_sources
        ),
        "processors": sorted(
            processor.id
            for processor in catalog.processors.processors
            if processor.id not in draft_processors
        ),
        "metrics": sorted(name for name in catalog.metrics.metrics if name not in draft_metrics),
        "dashboards": sorted(
            dashboard.id
            for dashboard in catalog.dashboards.dashboards
            if dashboard.id not in draft_dashboards
        ),
    }
    return {kind: names for kind, names in impact.items() if names}


def _render_replacement_disclosure(
    replacement: dict[str, list[str]],
    signature: str,
    *,
    confirmed: bool,
) -> None:
    """Require explicit consent before an apply removes existing catalog objects."""

    summary = " · ".join(f"{len(names)} {kind}" for kind, names in replacement.items())
    st.warning(
        "Applying this revision replaces the active catalog. It removes existing "
        f"objects the draft does not include: {summary}."
    )
    with st.expander("Removed by this apply", expanded=False):
        for kind, names in replacement.items():
            st.write(f"**{kind.title()}**: " + ", ".join(names))
    checked = st.checkbox(
        "Remove these existing objects when applying",
        value=confirmed,
        key=f"ai_studio_replace_confirm_{signature[:16]}",
        help=(
            "Required once for this exact draft revision. Changing the draft "
            "requires confirming the removals again."
        ),
    )
    if checked:
        st.session_state[AI_REPLACEMENT_CONFIRM_STATE_KEY] = signature
    elif st.session_state.get(AI_REPLACEMENT_CONFIRM_STATE_KEY) == signature:
        st.session_state[AI_REPLACEMENT_CONFIRM_STATE_KEY] = ""


def _processors_review(working: pl.DataFrame, approved_fields: list[str]) -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Processors Review")
    st.caption("Review generated processor definitions before editing dependent metrics.")
    _draft_counts(draft)
    snapshot = _draft_validation_snapshot(draft)
    _render_draft_validation(
        snapshot.ok,
        list(snapshot.issues),
        revision=snapshot.signature,
        expanded=False,
    )
    _render_ai_repair_panel(draft, working, approved_fields, list(snapshot.issues))
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

    _reconcile_keep_selection(
        st.session_state,
        key="ai_studio_processors_to_keep",
        options=processor_ids,
        revision=snapshot.signature,
    )
    selected = st.multiselect(
        "Processors To Keep",
        options=processor_ids,
        format_func=lambda processor_id: _processor_choice_label(draft, processor_id),
        key="ai_studio_processors_to_keep",
        help="Metrics and tiles that depend on rejected processors are removed automatically.",
    )
    if st.button("Update Draft: Processor Selection", type="primary", disabled=not selected):
        _set_draft(filter_draft_by_selection(draft, selected_processors=selected))
        st.rerun()
    with st.expander("Technical details · processor IDs", expanded=False):
        components.key_value_strip(
            [
                {"label": _processor_choice_label(draft, processor_id), "value": processor_id}
                for processor_id in processor_ids
            ]
        )
    _render_processor_parameter_editor(draft, working, approved_fields)

    with st.expander("Technical details · processors.yaml", expanded=False):
        st.caption("Use this only when the visual controls do not expose a required setting.")
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


def _processor_choice_label(draft: dict[str, Any], processor_id: str) -> str:
    processor = _draft_processor_by_id(draft, processor_id)
    description = str(processor.get("description") or "").strip()
    kind = builder.title_from_identifier(str(processor.get("kind") or "processor"))
    source = builder.title_from_identifier(str(processor.get("source") or "source"))
    title = description or builder.title_from_identifier(processor_id)
    return f"{title} · {kind} · {source} — {processor_id}"


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
            format_func=lambda value: _processor_choice_label(draft, value),
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
    display = metric_def.get("display") if isinstance(metric_def.get("display"), dict) else {}
    label = str(display.get("label") or "").strip()
    if label:
        return f"{label} — {metric_name} · {source} · {builder.metric_kind_label(kind)}"
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
    snapshot = _draft_validation_snapshot(draft)
    _render_draft_validation(
        snapshot.ok,
        list(snapshot.issues),
        revision=snapshot.signature,
        expanded=False,
    )
    _render_ai_repair_panel(draft, working, approved_fields, list(snapshot.issues))
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    _render_ai_refine_panel(draft, working, approved_fields)
    _render_coverage_panel(draft, working, approved_fields)
    _render_ai_recipe_library(draft)
    metrics = sorted(draft.get("metrics", {}).get("metrics", {}), key=str.casefold)
    if not metrics:
        st.info("The draft does not contain any metrics yet. Add one from the recipe library.")
        return
    _reconcile_keep_selection(
        st.session_state,
        key="ai_studio_metrics_to_keep",
        options=metrics,
        revision=snapshot.signature,
    )
    selected = st.multiselect(
        "Metrics To Keep",
        options=metrics,
        format_func=lambda name: _draft_metric_choice_label(draft, name),
        key="ai_studio_metrics_to_keep",
        help="Tiles for rejected metrics are removed automatically.",
    )
    if st.button("Update Draft: Metric Selection", type="primary", disabled=not selected):
        _set_draft(filter_draft_by_selection(draft, selected_metrics=selected))
        st.rerun()

    _render_metric_parameter_editor(draft)

    with st.expander("Technical details · metrics.yaml", expanded=False):
        st.caption("Use this only when the visual controls do not expose a required setting.")
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
            f"Recipe added to the draft. Review and **Apply to workspace**, then open "
            f"**Data Load** to materialize {states or 'the new aggregate state'} from "
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
    snapshot = _draft_validation_snapshot(draft)
    _render_draft_validation(
        snapshot.ok,
        list(snapshot.issues),
        revision=snapshot.signature,
        expanded=False,
    )
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
        _ai_retry_label("reports", "Refresh Reports From Metrics"),
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
                result = _run_validated_ai_candidate(
                    settings=ai_settings,
                    operation="report_refresh",
                    prompt=prompt,
                    base_draft=draft,
                    approved_fields=approved_fields,
                    repair_prompt_factory=_candidate_repair_prompt_factory(
                        working, approved_fields
                    ),
                    status=status,
                )
                queued = _queue_pending_candidate(
                    result,
                    base_draft=draft,
                    kind="reports",
                    prompt=prompt,
                )
                status.update(
                    label=(
                        "Validated reports ready for review"
                        if queued
                        else "Reports did not satisfy validation"
                    ),
                    state="complete" if queued else "error",
                )
            if queued:
                st.rerun()
            _render_candidate_failure(result, operation_label="Report proposal")
        except Exception as exc:  # pragma: no cover - Streamlit display path
            _log_ai_operation_failure("Report refresh", exc)
            _render_ai_operation_error(exc, operation="Report refresh")
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
        tile_id = builder.stable_catalog_id(
            title,
            fallback="tile",
            parent_id="metrics",
            existing_ids=used_ids,
        )
        used_ids.add(tile_id)
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
    dashboard_id = builder.stable_catalog_id(dashboard_title, fallback="dashboard")
    page_id = builder.stable_catalog_id(
        page_title,
        fallback="page",
        parent_id=dashboard_id,
    )
    page_filters = _deterministic_page_filters(
        metrics,
        processors,
        approved_fields,
    )
    return {
        "theme": dict(dashboards.get("theme", {})) if isinstance(dashboards, dict) else {},
        "dashboards": [
            {
                "id": dashboard_id,
                "title": dashboard_title,
                "layout": "tabs",
                "pages": [
                    {
                        "id": page_id,
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
    """Return a dashboards section with stable, reference-safe report ids."""
    out = {**dashboards}
    normalized_dashboards: list[dict[str, Any]] = []
    used_dashboard_ids: set[str] = set()
    for dashboard in dashboards.get("dashboards", []) or []:
        if not isinstance(dashboard, dict):
            continue
        dashboard_title = str(dashboard.get("title") or dashboard.get("id") or "Dashboard")
        dashboard_id = builder.stable_catalog_id(
            dashboard_title,
            fallback="dashboard",
            existing_ids=used_dashboard_ids,
            preferred_id=str(dashboard.get("id") or ""),
        )
        used_dashboard_ids.add(dashboard_id)
        dashboard_copy = {
            **dashboard,
            "id": dashboard_id,
            "title": dashboard_title,
        }
        normalized_pages: list[dict[str, Any]] = []
        used_page_ids: set[str] = set()
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            page_title = str(page.get("title") or page.get("id") or "Page")
            page_id = builder.stable_catalog_id(
                page_title,
                fallback="page",
                parent_id=dashboard_id,
                existing_ids=used_page_ids,
                preferred_id=str(page.get("id") or ""),
            )
            used_page_ids.add(page_id)
            page_copy = {
                **page,
                "id": page_id,
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
                        "id": builder.stable_catalog_id(
                            tile_title,
                            fallback="tile",
                            parent_id=page_id,
                            existing_ids=used_tile_ids,
                            preferred_id=str(tile.get("id") or ""),
                        ),
                        "title": tile_title,
                    }
                )
                used_tile_ids.add(str(normalized_tiles[-1]["id"]))
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


def _reports_review(working: pl.DataFrame, approved_fields: list[str]) -> None:
    draft = st.session_state.get("ai_studio_draft")
    if draft is None:
        st.info("Generate and accept a draft first.")
        return
    st.write("### Reports Review")
    st.caption("Review generated tiles, remove weak reports, or edit dashboards.yaml directly.")
    snapshot = _draft_validation_snapshot(draft)
    _render_draft_validation(
        snapshot.ok,
        list(snapshot.issues),
        revision=snapshot.signature,
        expanded=False,
    )
    _render_ai_repair_panel(draft, None, approved_fields, list(snapshot.issues))
    if st.session_state.get("ai_studio_pending_draft") is not None:
        return
    _render_ai_refine_panel(draft, None, approved_fields)
    _render_coverage_panel(draft, working, approved_fields)

    keys = tile_keys(draft)
    if keys:
        _render_tile_keep_table(draft, revision=snapshot.signature)
        with st.expander("Technical details · tile keys", expanded=False):
            components.key_value_strip(
                [{"label": _tile_choice_label(draft, key), "value": key} for key in keys]
            )
    else:
        st.warning("The draft does not contain any dashboard tiles.")
    _render_report_settings_editor(draft)
    with st.expander("Technical details · dashboards.yaml", expanded=False):
        st.caption("Use this for report settings not exposed by the compact review table.")
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


@st.fragment()
def _render_tile_keep_table(draft: dict[str, Any], *, revision: str) -> None:
    """Render the tile inventory with in-table Keep checkboxes.

    A fragment so checkbox edits rerun only this table; applying the
    selection escalates to a full app rerun. The editor input must stay
    byte-identical across reruns of one draft revision — if it changed,
    st.data_editor would reset its edit state and swallow every other
    click — so every row starts checked and the live selection is read
    back from the editor output only.
    """

    keys = tile_keys(draft)
    table_rows = [{"Keep": True, **row} for row in _tile_inventory_rows(draft)]
    edited_rows = st.data_editor(
        table_rows,
        hide_index=True,
        width="stretch",
        height=320,
        key=f"ai_studio_tiles_keep_editor_{revision}",
        column_config={
            "Keep": st.column_config.CheckboxColumn(
                "Keep",
                width="small",
                default=True,
                help=config_help.field_help("ai.keep_tiles"),
            ),
        },
        disabled=["Dashboard", "Page", "Report", "Measure", "Visualization"],
    )
    selected_tiles = [
        key
        for key, row in zip(keys, edited_rows, strict=True)
        if isinstance(row, dict) and bool(row.get("Keep"))
    ]
    st.session_state["ai_studio_tiles_to_keep"] = selected_tiles
    if st.button("Update Draft: Tile Selection", type="primary", disabled=not selected_tiles):
        _set_draft(filter_draft_by_selection(draft, selected_tiles=selected_tiles))
        st.rerun(scope="app")


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
    with st.expander("Technical details · dashboard theme YAML", expanded=False):
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


def _render_pending_draft_review() -> None:  # noqa: PLR0912, PLR0915
    if not _ai_calls_enabled():
        return
    pending = st.session_state.get("ai_studio_pending_draft")
    if pending is None:
        return
    base = st.session_state.get("ai_studio_pending_base_draft") or {}
    kind = st.session_state.get("ai_studio_pending_kind") or "draft"
    patches = draft_patches(base, pending)
    is_deterministic_review = kind == "deterministic draft"

    def validate_pending(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        # A deterministic draft is produced locally from the active schema and can
        # intentionally replace the previously accepted source graph. Comparing it
        # with that old graph here misclassifies the replacement as a provider
        # mutation outside the active source. Provider-authored proposals retain the
        # comparison guard.
        comparison_base = None if is_deterministic_review else base
        return _validate_draft_for_active_schema(
            candidate,
            baseline_draft=comparison_base,
        )

    bundles = draft_patch_bundles(base, pending, validate_pending)
    signature = _pending_review_signature(base, pending, kind)
    st.write(f"### Review pending {kind}")
    st.caption(
        "Changes are grouped with their processor, metric, and report dependencies. Removals "
        "start rejected and require an explicit individual selection."
    )
    if is_deterministic_review and any(bundle.is_removal for bundle in bundles):
        st.info(
            "This deterministic draft replaces existing source-dependent configuration. "
            "Choose Review individually, select Accept this complete deterministic "
            "replacement, then accept the selected bundle."
        )
    if not bundles:
        st.info("The AI response does not change the accepted draft.")
    safe_bundle_keys = [
        bundle.key for bundle in bundles if bundle.is_valid and not bundle.is_removal
    ]
    review_mode_key = f"ai_studio_individual_review_{signature}"
    action_col, review_col, reject_col = st.columns(3)
    if action_col.button(
        "Accept safe additions",
        type="primary",
        disabled=not safe_bundle_keys,
        key=f"ai_studio_accept_safe_{signature}",
        width="stretch",
    ):
        safe_candidate, safe_issues = merge_selected_draft_patch_bundles(
            base,
            pending,
            bundles,
            safe_bundle_keys,
            validate_pending,
        )
        if safe_candidate is None:
            st.warning(
                "The safe bundle combination is not valid as a whole. Review the bundles "
                "individually; nothing was accepted."
            )
            if safe_issues:
                with st.expander("Validation details", expanded=False):
                    for issue in safe_issues:
                        st.write(f"- {issue}")
        else:
            _accept_pending_bundles(
                safe_candidate,
                bundles=bundles,
                accepted_bundle_keys=safe_bundle_keys,
                patches=patches,
            )
            st.rerun()
    if review_col.button(
        "Review individually",
        key=f"ai_studio_review_individually_{signature}",
        width="stretch",
    ):
        st.session_state[review_mode_key] = True
        st.rerun()
    if reject_col.button(
        "Reject",
        key=f"ai_studio_reject_{signature}",
        width="stretch",
    ):
        _record_pending_review_discard(len(bundles))
        _clear_pending_ai_draft()
        st.rerun()

    individual_review = bool(st.session_state.get(review_mode_key))
    accepted_bundles: list[str] = []
    for index, bundle in enumerate(bundles):
        with st.container(border=True):
            st.write(f"#### {bundle.title}")
            st.write(bundle.summary)
            if bundle.is_removal:
                st.warning(
                    "This bundle removes existing configuration. It is excluded from safe "
                    "additions and starts unselected."
                )
            if not bundle.is_valid:
                st.warning("This bundle does not validate independently and cannot be accepted.")
                if bundle.validation_issues:
                    with st.expander("Why this bundle is blocked", expanded=False):
                        for issue in bundle.validation_issues:
                            st.write(f"- {issue}")
            elif not bundle.is_removal:
                st.caption(bundle.consequence)
            if individual_review:
                keep = st.checkbox(
                    (
                        "Accept this complete deterministic replacement"
                        if bundle.is_removal and is_deterministic_review
                        else "Explicitly include this removal"
                        if bundle.is_removal
                        else "Accept this complete bundle"
                    ),
                    value=False,
                    disabled=not bundle.is_valid,
                    key=f"ai_studio_bundle_{signature}_{index}",
                    help=config_help.field_help("ai.patch_accept"),
                )
                if keep:
                    accepted_bundles.append(bundle.key)
            with st.expander("Technical details · included changes", expanded=False):
                patch_by_key = {patch.key: patch for patch in patches}
                included_patches = [patch_by_key[patch_key] for patch_key in bundle.patch_keys]
                st.caption("Change summary")
                st.dataframe(
                    _pending_change_table_rows(included_patches),
                    hide_index=True,
                    width="stretch",
                )
                st.caption("YAML before and after")
                st.code(_pending_change_yaml(included_patches), language="yaml")

    reviewed, merge_issues = (None, ())
    if individual_review and accepted_bundles:
        reviewed, merge_issues = merge_selected_draft_patch_bundles(
            base,
            pending,
            bundles,
            accepted_bundles,
            validate_pending,
            allow_removals=True,
        )
    if reviewed is not None and accepted_bundles:
        snapshot = _draft_validation_snapshot(reviewed)
        _render_draft_validation(
            snapshot.ok,
            list(snapshot.issues),
            revision=snapshot.signature,
            expanded=not snapshot.ok,
        )
    elif merge_issues:
        st.warning("The selected bundle combination is not safe to apply.")
        with st.expander("Validation details", expanded=False):
            for issue in merge_issues:
                st.write(f"- {issue}")
    if individual_review and st.button(
        "Accept selected bundles",
        type="primary",
        disabled=reviewed is None or not accepted_bundles,
    ):
        _accept_pending_bundles(
            reviewed,
            bundles=bundles,
            accepted_bundle_keys=accepted_bundles,
            patches=patches,
        )
        st.rerun()
    with st.popover("Technical details", icon=":material/article:"):
        st.caption("Prompt")
        st.code(st.session_state.get("ai_studio_pending_prompt", ""), language="text")
        st.caption("Response")
        st.code(st.session_state.get("ai_studio_last_ai_response", ""), language="json")


def _accept_pending_bundles(
    reviewed: dict[str, Any],
    *,
    bundles: list[Any],
    accepted_bundle_keys: list[str],
    patches: list[DraftPatch],
) -> None:
    _set_draft(reviewed, reviewed=True)
    accepted = set(accepted_bundle_keys)
    accepted_patch_keys = [
        patch_key for bundle in bundles if bundle.key in accepted for patch_key in bundle.patch_keys
    ]
    _queue_preprocessing_editor_sync(patches, accepted_patch_keys)
    record_event(
        st.session_state,
        event=AuthoringEvent.REVIEWED,
        workflow=_studio_authoring_workflow(),
        stage=AuthoringStage.REVIEW,
        outcome=AuthoringOutcome.SUCCESS,
        count=len(accepted),
    )
    _clear_pending_ai_draft()


def _record_pending_review_discard(count: int) -> None:
    record_event(
        st.session_state,
        event=AuthoringEvent.REVIEWED,
        workflow=_studio_authoring_workflow(),
        stage=AuthoringStage.REVIEW,
        outcome=AuthoringOutcome.DISCARDED,
        count=count,
    )


def _pending_review_signature(base: dict[str, Any], pending: dict[str, Any], kind: str) -> str:
    payload = {"kind": kind, "base": base, "pending": pending}
    return hashlib.sha256(yaml.safe_dump(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _draft_patch_label(section: str) -> str:
    return {
        "sources": "source naming contract",
        "source_defaults": "source default",
        "source_filters": "source filter",
        "calculated_fields": "calculated field",
        "processors": "processor",
        "metrics": "metric",
        "dashboards": "report layout",
        "tiles": "report tile",
        "chat_with_data": "Chat With Data settings",
    }.get(section, section)


def _pending_change_table_rows(patches: list[DraftPatch]) -> list[dict[str, str]]:
    """Describe technical draft patches in user-facing review language."""

    actions = {"added": "Add", "changed": "Update", "removed": "Remove"}
    outcomes = {
        "added": "New configuration will be added.",
        "changed": "Existing configuration will be updated.",
        "removed": "Existing configuration will be removed.",
    }
    return [
        {
            "Change": actions.get(patch.change, patch.change.capitalize()),
            "Configuration": _draft_patch_label(patch.section).capitalize(),
            "Item": patch.object_id,
            "Result": outcomes.get(patch.change, "Configuration will change."),
        }
        for patch in patches
    ]


def _pending_change_yaml(patches: list[DraftPatch]) -> str:
    """Return one readable YAML view for all patches in a review bundle."""

    return yaml.safe_dump(
        {
            "changes": [
                {
                    "action": patch.change,
                    "configuration": _draft_patch_label(patch.section),
                    "item": patch.object_id,
                    "before": patch.before,
                    "after": patch.after,
                }
                for patch in patches
            ]
        },
        sort_keys=False,
    )


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
        if patch.key not in accepted_keys:
            continue
        if patch.section == "sources":
            if patch.object_id:
                queued.setdefault(patch.object_id, set()).update(_PREPROCESSING_PATCH_SECTIONS)
            continue
        if patch.section not in _PREPROCESSING_PATCH_SECTIONS:
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
            _ai_retry_label("repair", "Generate AI Repair"),
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
                    result = _run_validated_ai_candidate(
                        settings=ai_settings,
                        operation="catalog_repair",
                        prompt=prompt,
                        base_draft=draft,
                        approved_fields=approved_fields,
                        repair_prompt_factory=_candidate_repair_prompt_factory(
                            working, approved_fields
                        ),
                        status=status,
                    )
                    queued = _queue_pending_candidate(
                        result,
                        base_draft=draft,
                        kind="repair",
                        prompt=prompt,
                    )
                    status.update(
                        label=(
                            "Validated repair ready for review"
                            if queued
                            else "Repair did not satisfy validation"
                        ),
                        state="complete" if queued else "error",
                    )
                if queued:
                    st.rerun()
                _render_candidate_failure(result, operation_label="AI repair")
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _log_ai_operation_failure("AI repair", exc)
                _render_ai_operation_error(exc, operation="AI repair")
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
    with st.container(key="vs_ai_copilot_primary", border=True):
        _render_copilot_panel_contents(step, working, approved_fields)


def _render_copilot_panel_contents(
    step: str,
    working: pl.DataFrame,
    approved_fields: list[str],
) -> None:
    history = st.session_state.setdefault("ai_studio_copilot_history", [])
    has_pending = st.session_state.get("ai_studio_pending_draft") is not None
    sharing_confirmed = _ai_data_sharing_confirmed(approved_fields)
    queued = st.session_state.get("ai_studio_copilot_queued_message")
    components.status_badge("Recommended", "ready")
    st.write(f"### {_copilot_step_heading(step)}")
    st.caption(
        "Describe the outcome you want for this step. Copilot uses the effective schema, "
        "and every valid change waits for explicit patch review."
    )
    if history:
        with st.container(height=320, border=False):
            for message in history[-_COPILOT_HISTORY_DISPLAY:]:
                role = str(message.get("role") or "user")
                with st.chat_message("assistant" if role == "assistant" else "user"):
                    st.markdown(str(message.get("content") or ""))
    _render_copilot_questions()
    if has_pending:
        st.caption(
            "Proposal review is read-only: ask what a bundle changes or why it matters. "
            "Copilot cannot modify the proposal until review is complete."
        )
    message_text = st.chat_input(
        "Ask about the proposal" if has_pending else "Describe a change or ask about this step",
        key="ai_studio_copilot_input",
        disabled=not sharing_confirmed,
        submit_mode="disable",
    )
    if not sharing_confirmed:
        st.caption("Confirm the current sample's AI data-sharing scope before using Copilot.")
    prompt_text = ""
    if queued:
        prompt_text = str(
            st.session_state.pop("ai_studio_copilot_queued_message", "") or ""
        ).strip()
    elif message_text:
        prompt_text = str(message_text).strip()
    if prompt_text:
        _handle_copilot_message(prompt_text, step, working, approved_fields)
    last_prompt = str(st.session_state.get("ai_studio_copilot_last_prompt") or "")
    if last_prompt:
        with st.popover(
            "Last prompt",
            icon=":material/psychology:",
            key="ai_studio_copilot_last_prompt_popover",
        ):
            st.code(last_prompt, language="text")


def _copilot_step_heading(step: str) -> str:
    """Name the active numbered Studio step in the primary Copilot heading."""

    step_label = str(step).strip()
    match = re.fullmatch(r"(\d+)\.\s*(.+)", step_label)
    if match is None:
        return f"Configure {step_label} with AI" if step_label else "Configure this step with AI"
    step_number, step_name = match.groups()
    suffix = "" if re.search(r"\bAI\b", step_name, flags=re.IGNORECASE) else " with AI"
    return f"Configure step {step_number} · {step_name}{suffix}"


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
    pending_draft = st.session_state.get("ai_studio_pending_draft")
    read_only = isinstance(pending_draft, dict)
    history = st.session_state.setdefault("ai_studio_copilot_history", [])
    accepted_draft = st.session_state.get("ai_studio_draft")
    draft = (
        pending_draft
        if read_only
        else accepted_draft
        if isinstance(accepted_draft, dict)
        else _build_draft_catalog(working, approved_fields)
    )
    if not _draft_matches_active_naming_contract(draft):
        st.session_state[AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY] = True
        st.session_state["ai_studio_reviewed_signature"] = ""
        st.session_state["ai_studio_published_signature"] = ""
        history.extend(
            [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": (
                        "The source field-naming setting changed after this revision was "
                        "created, so I did not call the model or modify the draft. Open AI "
                        "Draft, generate the updated deterministic draft, and review its "
                        "change bundles before asking me to edit downstream fields."
                    ),
                },
            ]
        )
        _rerun_copilot_fragment()
        return
    ai_settings = _current_ai_settings()
    if ai_settings is None:
        st.info("Configure a LiteLLM model in the sidebar to use the copilot.")
        return
    if not _ai_data_sharing_confirmed(approved_fields):
        st.info("Confirm the current sample's AI data-sharing scope before using the copilot.")
        return
    schema_preview = _schema_preview_for_ai(working, approved_fields)
    hidden_fields = sorted(set(working.columns) - set(approved_fields), key=str.casefold)
    approved_field_name_mapping = {
        raw: effective
        for raw, effective in _rename_capitalize_mapping(
            list(st.session_state.get("ai_studio_raw_schema_columns") or [])
        ).items()
        if effective in approved_fields and raw != effective
    }
    prompt = prompt_for_copilot(
        step=step,
        user_message=message,
        history=history,
        user_goals=_current_user_goals(),
        approved_schema=schema_preview,
        approved_fields=approved_fields,
        hidden_fields=hidden_fields,
        current_draft=draft,
        rename_capitalize_enabled=_rename_capitalize_enabled(),
        approved_field_name_mapping=approved_field_name_mapping,
        read_only=read_only,
        pending_summary=(
            "A validated catalog proposal is awaiting explicit dependency-bundle review."
            if read_only
            else ""
        ),
    )
    st.session_state["ai_studio_copilot_last_prompt"] = prompt
    status = None
    try:
        with st.status("Running governed draft tools", expanded=False) as status:
            _preflight_ai_operation(
                ai_settings,
                operation="copilot_read" if read_only else "copilot_change",
                approved_fields=approved_fields,
                status=status,
            )
            result = run_copilot_tool_loop(
                prompt=prompt,
                draft=draft,
                call_model=lambda iteration_prompt: _call_litellm_for_current_sample(
                    ai_settings,
                    iteration_prompt,
                    approved_fields=approved_fields,
                ),
                # Structured Copilot operations are checked against approved fields by
                # run_copilot_tool_loop itself. Catalog validation here preserves valid
                # carried-forward objects that may use fields the user kept hidden from AI.
                validate=_validate_draft_catalog_for_active_source,
                max_iterations=3,
                operation_policy=_copilot_operation_policy(step),
                hidden_fields=hidden_fields,
                approved_fields=approved_fields,
                field_contract_source_id=str(st.session_state.get("ai_studio_source_id") or ""),
                field_contract_source_fields=_active_draft_source_fields(approved_fields),
                expected_rename_capitalize=_expected_rename_capitalize_contract(),
                field_name_mapping=approved_field_name_mapping,
                read_only=read_only,
                pending_summary=(
                    "A validated catalog proposal is awaiting explicit dependency-bundle review."
                    if read_only
                    else ""
                ),
            )
            status.update(
                label=f"Copilot finished after {result.iterations} iteration(s)",
                state="complete" if not result.validation_issues else "error",
            )
    except Exception as exc:  # pragma: no cover - Streamlit display path
        _render_copilot_operation_failure(status, exc)
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


def _render_copilot_operation_failure(status: Any, exc: Exception) -> None:
    if status is not None:
        status.update(label="Copilot request failed", state="error")
    _log_ai_operation_failure("Copilot request", exc)
    _render_ai_operation_error(exc, operation="Copilot request")


def _copilot_operation_policy(step: str) -> dict[str, str]:
    step_name = step.split(". ", 1)[-1]
    allowed = {
        "Defaults": {"set_source_default", "remove_source_default"},
        "Filters": {"set_source_filter", "remove_source_filter"},
        "Calculations": {"set_calculated_field", "remove_calculated_field"},
    }.get(step_name)
    if allowed is None:
        return {}
    message = {
        "Filters": (
            "The Filters step edits the source pipeline before processor fan-out. "
            "Use set_source_filter or remove_source_filter only."
        ),
        "Defaults": (
            "The Defaults step accepts source-default changes only. Use "
            "set_source_default or remove_source_default."
        ),
        "Calculations": (
            "The Calculations step accepts calculated-field changes only. Use "
            "set_calculated_field or remove_calculated_field."
        ),
    }[step_name]
    operation_names = {
        "set_source_default",
        "remove_source_default",
        "set_source_filter",
        "remove_source_filter",
        "set_calculated_field",
        "remove_calculated_field",
        "set_processor",
        "remove_processor",
        "set_metric",
        "remove_metric",
        "set_tile",
        "remove_tile",
        "set_dashboards",
        "install_recipe",
    }
    return dict.fromkeys(operation_names - allowed, message)


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
            approved_fields=approved_fields,
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
                with st.status("Checking requirements coverage", expanded=False) as status:
                    _preflight_ai_operation(
                        ai_settings,
                        operation="coverage_check",
                        approved_fields=approved_fields,
                        status=status,
                    )
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
                _render_ai_operation_error(exc, operation="Coverage check")
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


def _reconcile_keep_selection(
    state: Any,
    *,
    key: str,
    options: list[str],
    revision: str,
) -> list[str]:
    """Reconcile one Keep widget against a changed, stable-ID draft revision."""

    known_key = f"{key}__known_ids"
    rejected_key = f"{key}__rejected_ids"
    revision_key = f"{key}__revision"
    old_known = [str(value) for value in (state.get(known_key) or [])]
    current_raw = state.get(key) if key in state else None
    current_values = current_raw if isinstance(current_raw, (list, tuple, set)) else []
    current = {str(value) for value in current_values if str(value)}
    rejected = {str(value) for value in (state.get(rejected_key) or []) if str(value)}

    if old_known:
        for value in old_known:
            if value in current:
                rejected.discard(value)
            else:
                rejected.add(value)
        selected = [
            option
            for option in options
            if option in current or (option not in old_known and option not in rejected)
        ]
    else:
        selected = list(options)

    state[key] = selected
    state[known_key] = list(options)
    state[rejected_key] = sorted(rejected, key=str.casefold)
    state[revision_key] = revision
    return selected


def _active_draft_field_contract(
    approved_fields: list[str] | None = None,
) -> list[str] | None:
    if approved_fields is not None:
        return sorted({str(field) for field in approved_fields if str(field)}, key=str.casefold)
    if st.session_state.get("ai_studio_draft_source") == CATALOG_DRAFT_SOURCE:
        return None
    if "ai_studio_effective_schema_columns" not in st.session_state:
        return None
    return sorted(
        {
            str(field)
            for field in st.session_state.get("ai_studio_approved_fields", [])
            if str(field)
        },
        key=str.casefold,
    )


def _active_draft_field_contract_source_id() -> str | None:
    if st.session_state.get("ai_studio_draft_source") == CATALOG_DRAFT_SOURCE:
        return None
    source_id = str(st.session_state.get("ai_studio_source_id") or "").strip()
    return source_id or None


def _active_draft_source_fields(field_contract: list[str]) -> list[str]:
    raw_fields = st.session_state.get("ai_studio_effective_schema_columns")
    if not isinstance(raw_fields, list):
        return field_contract
    fields = sorted(
        {str(field) for field in raw_fields if str(field)},
        key=str.casefold,
    )
    return fields or field_contract


def _active_catalog_source_columns() -> dict[str, list[str]]:
    """Return observed physical columns for transform-aware catalog validation."""

    source_id = str(st.session_state.get("ai_studio_source_id") or "").strip()
    raw_columns = st.session_state.get("ai_studio_raw_schema_columns")
    if not source_id or not isinstance(raw_columns, list):
        return {}
    columns = [str(column) for column in raw_columns if str(column).strip()]
    return {source_id: columns} if columns else {}


def _validate_draft_catalog_for_active_source(
    draft: dict[str, Any],
) -> tuple[bool, list[str]]:
    return validate_draft_catalog(
        draft,
        source_columns_by_id=_active_catalog_source_columns(),
    )


def _expected_rename_capitalize_contract() -> bool | None:
    """Return the active sample's naming mode when sample-backed state is initialized."""

    if (
        AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY not in st.session_state
        and AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY not in st.session_state
    ):
        return None
    return _rename_capitalize_enabled()


def _validate_draft_for_active_schema(
    draft: dict[str, Any],
    approved_fields: list[str] | None = None,
    source_id: str | None = None,
    baseline_draft: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    catalog_ok, catalog_issues = _validate_draft_catalog_for_active_source(draft)
    field_contract = _active_draft_field_contract(approved_fields)
    if field_contract is None:
        return catalog_ok, catalog_issues
    fields_ok, field_issues = validate_draft_field_contract(
        draft,
        field_contract,
        source_id=source_id or _active_draft_field_contract_source_id(),
        source_fields=_active_draft_source_fields(field_contract),
        baseline_draft=baseline_draft,
        expected_rename_capitalize=_expected_rename_capitalize_contract(),
    )
    return catalog_ok and fields_ok, list(dict.fromkeys([*catalog_issues, *field_issues]))


def _draft_validation_cache_key(
    signature: str,
    field_contract: list[str] | None,
    source_id: str | None,
) -> str:
    if field_contract is None:
        return f"{signature}:catalog"
    payload = json.dumps(
        {
            "approved_fields": field_contract,
            "effective_schema_signature": str(
                st.session_state.get("ai_studio_effective_schema_signature") or ""
            ),
            "source_id": str(source_id or ""),
            "rename_capitalize": _expected_rename_capitalize_contract(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    contract_signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{signature}:{contract_signature}"


def _draft_validation_snapshot(
    draft: dict[str, Any],
    approved_fields: list[str] | None = None,
) -> DraftValidationSnapshot:
    """Return cached validation evidence for the draft and active field contract."""

    signature = _draft_signature(draft)
    field_contract = _active_draft_field_contract(approved_fields)
    source_id = _active_draft_field_contract_source_id()
    cache_key = _draft_validation_cache_key(signature, field_contract, source_id)
    raw_cache = st.session_state.setdefault("ai_studio_validation_cache", {})
    cache = raw_cache if isinstance(raw_cache, dict) else {}
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        snapshot = DraftValidationSnapshot(
            signature=signature,
            ok=bool(cached.get("ok")),
            issues=tuple(str(issue) for issue in cached.get("issues", [])),
        )
        _reconcile_reviewed_signature(snapshot)
        return snapshot
    ok, issues = _validate_draft_for_active_schema(draft, field_contract, source_id)
    snapshot = DraftValidationSnapshot(signature, ok, tuple(issues))
    cache[cache_key] = {"ok": ok, "issues": list(issues)}
    st.session_state["ai_studio_validation_cache"] = cache
    _reconcile_reviewed_signature(snapshot)
    return snapshot


def _reconcile_reviewed_signature(snapshot: DraftValidationSnapshot) -> None:
    """Invalidate review evidence when the same revision fails the current contract."""

    if (
        not snapshot.ok
        and st.session_state.get("ai_studio_reviewed_signature") == snapshot.signature
    ):
        st.session_state["ai_studio_reviewed_signature"] = ""


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
                _ai_retry_label("revision", "Generate AI Revision"),
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
                    result = _run_validated_ai_candidate(
                        settings=ai_settings,
                        operation="catalog_revision",
                        prompt=prompt,
                        base_draft=draft,
                        approved_fields=approved_fields,
                        repair_prompt_factory=_candidate_repair_prompt_factory(
                            working, approved_fields
                        ),
                        status=status,
                    )
                    queued = _queue_pending_candidate(
                        result,
                        base_draft=draft,
                        kind="revision",
                        prompt=prompt,
                    )
                    status.update(
                        label=(
                            "Validated revision ready for review"
                            if queued
                            else "Revision did not satisfy validation"
                        ),
                        state="complete" if queued else "error",
                    )
                if queued:
                    st.rerun()
                _render_candidate_failure(result, operation_label="AI revision")
            except Exception as exc:  # pragma: no cover - Streamlit display path
                _log_ai_operation_failure("AI revision", exc)
                _render_ai_operation_error(exc, operation="AI revision")
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
    st.session_state.pop(AI_SHARING_RECEIPT_STATE_KEY, None)
    for key in list(st.session_state):
        if str(key).startswith(AI_SHARING_CONFIRMATION_WIDGET_PREFIX):
            st.session_state.pop(key, None)
    st.session_state["ai_studio_copilot_history"] = []
    st.session_state["ai_studio_copilot_questions"] = []
    st.session_state["ai_studio_copilot_last_prompt"] = ""
    st.session_state.pop("ai_studio_copilot_queued_message", None)


def _on_ai_sharing_confirmation_change(
    signature: str,
    provider: str,
    model_name: str,
) -> None:
    """Record consent only from the confirmation input's change callback."""

    widget_key = f"{AI_SHARING_CONFIRMATION_WIDGET_PREFIX}{signature[:16]}"
    if bool(st.session_state.get(widget_key)):
        st.session_state[AI_SHARING_CONFIRMATION_STATE_KEY] = signature
        st.session_state[AI_SHARING_RECEIPT_STATE_KEY] = {
            "contract_signature": signature,
            "provider": provider,
            "model": model_name,
            "confirmed_at_epoch": int(time.time()),
            "interaction": "confirmation_input",
        }
        record_event(
            st.session_state,
            event=AuthoringEvent.CONSENT_CONFIRMED,
            workflow=_studio_authoring_workflow(),
            stage=AuthoringStage.CONSENT,
            outcome=AuthoringOutcome.SUCCESS,
        )
    elif st.session_state.get(AI_SHARING_CONFIRMATION_STATE_KEY) == signature:
        st.session_state[AI_SHARING_CONFIRMATION_STATE_KEY] = ""
        st.session_state.pop(AI_SHARING_RECEIPT_STATE_KEY, None)


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
    if already_confirmed:
        with st.container(border=True):
            with st.container(key="vs_ai_sharing_consent", gap="xsmall"):
                components.status_badge("AI access confirmed", "ready")
                st.caption(
                    f"Confirmed for {contract['provider']} · "
                    f"{contract['model'] or 'configured model'} · "
                    f"{len(contract['approved_fields'])} schema fields. AI generation, Copilot, "
                    "repair, and report refresh are enabled for this confirmed scope."
                )
            with st.expander("Review confirmed sharing scope", expanded=False):
                _render_ai_sharing_scope_details(
                    contract,
                    example_fields,
                    likely_identifiers,
                    include_help_expanders=False,
                )
                st.button(
                    "Revoke AI sharing confirmation",
                    key=f"ai_studio_ai_sharing_revoke_{signature[:16]}",
                    icon=":material/block:",
                    on_click=_clear_ai_sharing_confirmation,
                )
        return
    widget_key = f"{AI_SHARING_CONFIRMATION_WIDGET_PREFIX}{signature[:16]}"
    st.session_state.setdefault(widget_key, False)
    with st.container(border=True):
        st.write("### Review data sent to AI")
        st.caption(
            "Confirm this sharing scope for the current sample before any AI action can run. "
            "Changing the sample, model, provider, approved fields, or example sharing requires "
            "a new confirmation. Prompts also include your business requirements and the "
            "relevant deterministic catalog or current draft settings."
        )
        _render_ai_sharing_scope_details(
            contract,
            example_fields,
            likely_identifiers,
            include_help_expanders=True,
        )
        with st.container(key="vs_ai_sharing_consent", gap="xsmall"):
            components.status_badge("Required for AI editor", "warning")
            st.caption(
                "AI generation, Copilot, repair, and report refresh remain disabled "
                "until this confirmation is checked."
            )
            st.checkbox(
                "Review (changed) sharing scope and confirm it may be sent to the provider and model shown.",
                key=widget_key,
                on_change=_on_ai_sharing_confirmation_change,
                args=(signature, str(contract["provider"]), str(contract["model"])),
            )


def _render_ai_sharing_scope_details(
    contract: dict[str, Any],
    example_fields: list[str],
    likely_identifiers: list[str],
    *,
    include_help_expanders: bool,
) -> None:
    components.key_value_strip(
        [
            {"label": "Provider", "value": contract["provider"]},
            {"label": "Model", "value": contract["model"] or "Not configured"},
            {"label": "Schema fields", "value": len(contract["approved_fields"])},
            {"label": "Fields with examples", "value": len(example_fields)},
        ]
    )
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
    scope_lines = (
        f"Sample: `{contract['sample_name']}`\n\n"
        "Approved schema fields: "
        + (", ".join(contract["approved_fields"]) or "none")
        + "\n\nFields with sample values: "
        + (", ".join(example_fields) or "none")
        + f"\n\nDestination: {contract['destination']}"
    )
    if include_help_expanders:
        with st.expander("Sharing scope", expanded=False):
            st.markdown(scope_lines)
            st.write("Also included: business requirements and relevant catalog or draft settings")
            st.caption("Provider storage and retention follow your configured provider terms.")
        with st.expander("Why confirmation is required", expanded=False):
            st.caption(
                "Confirmation applies only to the sample, provider, model, approved schema, "
                "and example-value scope shown above. Reviewing this help does not change consent."
            )
    else:
        st.markdown(scope_lines)
        st.caption(
            "Also included: business requirements and relevant catalog or draft settings. "
            "Provider storage and retention follow your configured provider terms."
        )


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


def _preflight_ai_operation(
    settings: AICallSettings,
    *,
    operation: str,
    approved_fields: list[str],
    status: Any | None = None,
) -> None:
    """Check the exact provider/model operation after a user-triggered action."""

    if not _ai_data_sharing_confirmed(approved_fields):
        raise PermissionError(
            "Review and confirm the current sample's AI data-sharing scope before continuing."
        )
    cache_key = ai_provider_preflight_cache_key(settings, capability=operation)
    cache = st.session_state.setdefault("ai_studio_ai_preflight_cache", {})
    cached = cache.get(cache_key) if isinstance(cache, dict) else None
    force_retry = bool(st.session_state.pop("ai_studio_force_preflight_retry", False))
    if cached is True or (isinstance(cached, dict) and cached.get("ok") is True):
        if status is not None:
            status.write("Provider preflight · passed (cached for these settings)")
        return
    if isinstance(cached, dict) and cached.get("ok") is False:
        expires_at = float(cached.get("expires_at") or 0.0)
        if expires_at > time.monotonic() and not force_retry:
            if status is not None:
                status.write("Provider preflight · recent failure cached; Retry checks again")
            _raise_cached_preflight_failure(cached)
        cache.pop(cache_key, None)

    if status is not None:
        status.update(
            label=(
                "Provider preflight · checking model access and chat capability "
                f"(up to {AI_PREFLIGHT_TIMEOUT_SECONDS}s)"
            ),
            state="running",
            expanded=True,
        )

    def call_with_confirmed_scope(
        preflight_settings: AICallSettings,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return _call_litellm_for_current_sample(
            preflight_settings,
            prompt,
            approved_fields=approved_fields,
            **kwargs,
        )

    try:
        preflight_ai_provider(settings, call=call_with_confirmed_scope)
    except Exception as exc:
        if isinstance(cache, dict):
            cache[cache_key] = _preflight_failure_cache_entry(exc)
        raise
    if isinstance(cache, dict):
        cache[cache_key] = {"ok": True}
    if status is not None:
        status.write("Provider preflight · passed")


def _preflight_failure_cache_entry(exc: Exception) -> dict[str, Any]:
    """Return a short-lived, privacy-safe preflight failure receipt."""

    entry: dict[str, Any] = {
        "ok": False,
        "expires_at": time.monotonic() + AI_PREFLIGHT_NEGATIVE_TTL_SECONDS,
        "kind": "permission"
        if isinstance(exc, PermissionError)
        else "provider"
        if isinstance(exc, AIProviderCallError)
        else "timeout"
        if _authoring_failure_outcome(exc) is AuthoringOutcome.TIMEOUT
        else "error",
    }
    if isinstance(exc, AIProviderCallError):
        entry.update(
            {
                "call_id": exc.call_id,
                "error_type": exc.error_type,
                "permission_denied": exc.permission_denied,
                "category": str(exc.category),
                "retryable": exc.retryable,
            }
        )
    return entry


def _raise_cached_preflight_failure(entry: dict[str, Any]) -> None:
    """Raise the safe equivalent of a cached provider preflight failure."""

    kind = str(entry.get("kind") or "error")
    if kind == "timeout":
        raise TimeoutError("The recent provider preflight timed out; Retry checks again.")
    if kind == "permission":
        raise PermissionError("Provider access is not configured for this operation.")
    if kind == "provider":
        raise AIProviderCallError(
            call_id=str(entry.get("call_id") or "cached"),
            error_type=str(entry.get("error_type") or "ProviderError"),
            permission_denied=bool(entry.get("permission_denied")),
            category=str(entry.get("category") or "provider"),
        )
    raise RuntimeError("The recent provider preflight failed; Retry checks again.")


def _run_validated_ai_candidate(
    *,
    settings: AICallSettings,
    operation: str,
    prompt: str,
    base_draft: dict[str, Any],
    approved_fields: list[str],
    repair_prompt_factory: Any,
    status: Any,
) -> DraftCandidateResult:
    """Run one named, bounded AI operation and emit privacy-safe funnel evidence."""

    started_at = time.perf_counter()
    record_event(
        st.session_state,
        event=AuthoringEvent.DRAFT_REQUESTED,
        workflow=_studio_authoring_workflow(),
        stage=AuthoringStage.DRAFT,
        outcome=AuthoringOutcome.STARTED,
    )
    model_call_count = 0

    def call_model(iteration_prompt: str) -> str:
        nonlocal model_call_count
        model_call_count += 1
        stage = (
            "Generating proposal"
            if model_call_count == 1
            else f"Repair pass {model_call_count - 1}"
        )
        status.update(
            label=f"{stage} · waiting for provider response",
            state="running",
            expanded=True,
        )
        return _call_litellm_for_current_sample(
            settings,
            iteration_prompt,
            approved_fields=approved_fields,
        )

    try:
        _preflight_ai_operation(
            settings,
            operation=operation,
            approved_fields=approved_fields,
            status=status,
        )
        result = generate_validated_candidate(
            base_draft=base_draft,
            prompt=prompt,
            call=call_model,
            repair_prompt=repair_prompt_factory,
            max_repairs=2,
            operation=operation,
            validate=lambda candidate: _validate_draft_for_active_schema(
                candidate,
                approved_fields,
                baseline_draft=base_draft,
            ),
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1_000)
        record_event(
            st.session_state,
            event=AuthoringEvent.FAILED,
            workflow=_studio_authoring_workflow(),
            stage=AuthoringStage.DRAFT,
            outcome=_authoring_failure_outcome(exc),
            duration_ms=duration_ms,
            count=model_call_count,
        )
        st.session_state["ai_studio_last_candidate_failure"] = {
            "kind": operation,
            "stage": "provider",
            "issues": [],
            "attempts": model_call_count,
            "scope_signature": _ai_sharing_signature(approved_fields),
            "reference": exc.call_id if isinstance(exc, AIProviderCallError) else "",
        }
        raise
    duration_ms = int((time.perf_counter() - started_at) * 1_000)
    for diagnostic in result.attempt_diagnostics:
        status.write(_candidate_attempt_status_line(diagnostic))
    if result.ok:
        record_event(
            st.session_state,
            event=AuthoringEvent.VALID_PROPOSAL,
            workflow=_studio_authoring_workflow(),
            stage=AuthoringStage.DRAFT,
            outcome=AuthoringOutcome.SUCCESS,
            duration_ms=duration_ms,
            count=result.attempts,
        )
    else:
        record_event(
            st.session_state,
            event=AuthoringEvent.FAILED,
            workflow=_studio_authoring_workflow(),
            stage=AuthoringStage.DRAFT,
            outcome=AuthoringOutcome.BLOCKED,
            duration_ms=duration_ms,
            count=result.attempts,
        )
    return result


def _authoring_failure_outcome(exc: Exception) -> AuthoringOutcome:
    if isinstance(exc, AIProviderCallError) and str(exc.category) == "timeout":
        return AuthoringOutcome.TIMEOUT
    error_name = (type(exc).__name__ + " " + str(getattr(exc, "error_type", "") or "")).casefold()
    if isinstance(exc, TimeoutError) or "timeout" in error_name:
        return AuthoringOutcome.TIMEOUT
    if isinstance(exc, PermissionError):
        return AuthoringOutcome.BLOCKED
    return AuthoringOutcome.ERROR


def _candidate_repair_prompt_factory(
    working: pl.DataFrame | None,
    approved_fields: list[str],
) -> Any:
    schema_preview = _schema_preview_for_ai(working, approved_fields) if working is not None else []
    hidden_fields = (
        sorted(set(working.columns) - set(approved_fields), key=str.casefold)
        if working is not None
        else []
    )

    def build(candidate: dict[str, Any], issues: list[str], trace: str) -> str:
        return prompt_for_repair(
            file_name=_sample_file_name(),
            approved_schema=schema_preview,
            approved_fields=approved_fields,
            hidden_fields=hidden_fields,
            current_draft=candidate,
            validation_issues=issues,
            validation_trace=trace,
        )

    return build


def _queue_pending_candidate(
    result: DraftCandidateResult,
    *,
    base_draft: dict[str, Any],
    kind: str,
    prompt: str,
) -> bool:
    """Queue only a validated candidate and retain the accepted revision otherwise."""

    if result.draft is None:
        st.session_state["ai_studio_last_candidate_failure"] = {
            "kind": kind,
            "stage": result.failure_stage,
            "issues": list(result.issues),
            "attempts": result.attempts,
            "reference": result.reference,
            "scope_signature": _ai_sharing_signature(),
            "attempt_diagnostics": [
                _candidate_attempt_diagnostic_receipt(diagnostic)
                for diagnostic in result.attempt_diagnostics
            ],
        }
        return False
    st.session_state["ai_studio_pending_draft"] = result.draft
    st.session_state["ai_studio_pending_base_draft"] = copy.deepcopy(base_draft)
    st.session_state["ai_studio_pending_kind"] = kind
    st.session_state["ai_studio_pending_prompt"] = prompt
    st.session_state["ai_studio_last_ai_response"] = result.last_response
    st.session_state.pop("ai_studio_last_candidate_failure", None)
    return True


def _draft_review_base(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the accepted revision, or an empty catalog shell for first review."""

    accepted = st.session_state.get("ai_studio_draft")
    if isinstance(accepted, dict):
        return copy.deepcopy(accepted)
    return {
        "pipelines": copy.deepcopy(candidate.get("pipelines", {})),
        "processors": {"processors": []},
        "metrics": {"metrics": {}},
        "dashboards": {
            "theme": copy.deepcopy(candidate.get("dashboards", {}).get("theme", {})),
            "dashboards": [],
        },
    }


def _candidate_attempt_diagnostic_receipt(
    diagnostic: DraftAttemptDiagnostic,
) -> dict[str, Any]:
    """Return bounded in-session evidence for one rejected or accepted attempt."""

    return {
        "attempt": diagnostic.attempt,
        "role": diagnostic.role,
        "stage": diagnostic.stage,
        "issues": list(diagnostic.issues),
        "issue_count": diagnostic.issue_count,
        "issue_areas": dict(diagnostic.issue_areas),
        "sections": list(diagnostic.sections),
        "response_chars": diagnostic.response_chars,
        "error_type": diagnostic.error_type,
        "line": diagnostic.line,
        "column": diagnostic.column,
    }


def _candidate_attempt_status_line(diagnostic: DraftAttemptDiagnostic) -> str:
    """Return a terminal, user-facing status line for one model response."""

    role = _candidate_attempt_role_label(diagnostic).casefold()
    prefix = f"Attempt {diagnostic.attempt} · {role}"
    if diagnostic.stage == "validated":
        sections = ", ".join(diagnostic.sections) or "catalog sections"
        return f"{prefix} · accepted · {sections} passed the complete catalog contract"
    if diagnostic.stage == "parse":
        location = ""
        if diagnostic.line is not None and diagnostic.column is not None:
            location = f" at line {diagnostic.line}, column {diagnostic.column}"
        error_type = f" ({diagnostic.error_type})" if diagnostic.error_type else ""
        return f"{prefix} · rejected · catalog YAML could not be read{location}{error_type}"
    areas = (
        ", ".join(area.replace("_", " ") for area, _count in diagnostic.issue_areas) or "catalog"
    )
    return f"{prefix} · rejected · {diagnostic.issue_count} catalog contract issue(s) in {areas}"


def _candidate_attempt_rows(result: DraftCandidateResult) -> list[dict[str, Any]]:
    """Return a business-friendly table describing every bounded model attempt."""

    rows: list[dict[str, Any]] = []
    for diagnostic in result.attempt_diagnostics:
        if diagnostic.stage == "validated":
            outcome = "Accepted"
            explanation = "The returned sections passed the complete catalog contract."
        elif diagnostic.stage == "parse":
            outcome = "YAML could not be read"
            explanation = (
                diagnostic.issues[0]
                if diagnostic.issues
                else ("The response was not valid catalog YAML.")
            )
        else:
            outcome = "Catalog checks failed"
            examples = " ".join(diagnostic.issues[:2])
            explanation = f"{diagnostic.issue_count} contract issue(s). {examples}".strip()
        rows.append(
            {
                "Attempt": diagnostic.attempt,
                "Purpose": _candidate_attempt_role_label(diagnostic),
                "Outcome": outcome,
                "Sections Found": ", ".join(diagnostic.sections) or "None",
                "Issues": diagnostic.issue_count,
                "What Went Wrong": explanation,
            }
        )
    return rows


def _candidate_failure_message(
    result: DraftCandidateResult,
    *,
    operation_label: str,
) -> str:
    """Explain why a provider response was rejected without exposing its payload."""

    final = result.attempt_diagnostics[-1] if result.attempt_diagnostics else None
    if result.failure_stage == "parse":
        reason = (
            "the final response could not be read as catalog YAML. Review the attempt "
            "details, then retry to ask the model for complete YAML sections"
        )
    elif result.failure_stage == "validation":
        issue_count = final.issue_count if final is not None else len(result.issues)
        reason = (
            f"the final YAML still failed {issue_count} catalog contract check(s). "
            "Review the exact checks below; retry sends them through another bounded repair cycle"
        )
    else:
        reason = "none of the returned responses passed the complete catalog contract"
    return (
        f"{operation_label} was not created after {_attempt_count_label(result.attempts)}: "
        f"{reason}. "
        "The current accepted revision was not changed."
    )


def _render_candidate_failure(
    result: DraftCandidateResult,
    *,
    operation_label: str,
) -> None:
    """Render actionable, read-only evidence for a rejected model candidate."""

    st.error(_candidate_failure_message(result, operation_label=operation_label))
    with st.expander("Why the model response was rejected", expanded=True):
        final_stage = result.failure_stage.replace("_", " ").title() or "Catalog contract"
        components.key_value_strip(
            [
                {"label": "Diagnostic Reference", "value": result.reference or "Unavailable"},
                {"label": "Responses Checked", "value": result.attempts},
                {"label": "Final Failure", "value": final_stage},
                {"label": "Accepted Revision", "value": "Unchanged"},
            ]
        )
        rows = _candidate_attempt_rows(result)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        for diagnostic in result.attempt_diagnostics:
            if not diagnostic.issues:
                continue
            role = _candidate_attempt_role_label(diagnostic).casefold()
            st.markdown(f"**Attempt {diagnostic.attempt} · {role} details**")
            for issue in diagnostic.issues:
                st.write(issue)
            omitted = diagnostic.issue_count - len(diagnostic.issues)
            if omitted > 0:
                st.caption(f"{omitted} additional issue(s) were omitted from this receipt.")
        st.caption(
            "The prompt and raw model response are intentionally excluded from diagnostics. "
            "Use the reference above to correlate this receipt with privacy-safe debug logs."
        )


def _candidate_attempt_role_label(diagnostic: DraftAttemptDiagnostic) -> str:
    if diagnostic.role == "generation":
        return "Initial generation"
    if diagnostic.role == "repair":
        return f"Repair pass {max(diagnostic.attempt - 1, 1)}"
    return diagnostic.role.replace("_", " ").title()


def _attempt_count_label(attempts: int) -> str:
    return f"{attempts} attempt{'s' if attempts != 1 else ''}"


def _ai_operation_error_message(exc: Exception, *, operation: str) -> str:
    if isinstance(exc, PermissionError):
        return (
            f"Couldn't complete {operation}: provider access is not configured. "
            "Check the API key, project permission, model, or custom endpoint, then retry. "
            "The accepted revision was not changed."
        )
    if isinstance(exc, AIProviderCallError):
        category = str(exc.category)
        remediation = {
            "configuration": "Check the provider, model, credential, and endpoint settings.",
            "authentication": "Replace or refresh the configured API credential.",
            "authorization": "Grant this project access to the selected model or operation.",
            "rate_limit": "Wait briefly or reduce request frequency, then retry.",
            "timeout": "Retry or increase Timeout Seconds in AI Settings.",
            "network": "Check endpoint reachability and network connectivity, then retry.",
            "provider": "Retry; if it persists, check provider service health.",
            "response_validation": "Retry or choose a model that returns chat completions.",
            "internal": "Review the technical reference and application logs.",
        }.get(category, "Verify the provider settings, then retry.")
        return (
            f"Couldn't complete {operation}: {category.replace('_', ' ')} failure. "
            f"{remediation} The accepted revision was not changed."
        )
    if _authoring_failure_outcome(exc) is AuthoringOutcome.TIMEOUT:
        return (
            f"Couldn't complete {operation} before the provider timeout. Retry or increase "
            "Timeout Seconds in AI Settings. The accepted revision was not changed."
        )
    return (
        f"Couldn't complete {operation}. Verify the provider, model, credential, and endpoint, "
        "then retry. The accepted revision was not changed."
    )


def _render_ai_operation_error(exc: Exception, *, operation: str) -> None:
    # The next explicit user action is a genuine retry and must not be blocked
    # by the short-lived negative preflight cache.
    st.session_state["ai_studio_force_preflight_retry"] = True
    st.error(_ai_operation_error_message(exc, operation=operation))
    if isinstance(exc, AIProviderCallError):
        with st.expander("Technical details · provider reference", expanded=False):
            components.key_value_strip(
                [
                    {"label": "Reference", "value": exc.call_id},
                    {"label": "Category", "value": str(exc.category).replace("_", " ")},
                    {"label": "Retryable", "value": "Yes" if exc.retryable else "No"},
                    {"label": "Provider error type", "value": exc.error_type},
                ]
            )


def _ai_retry_label(kind: str, default: str) -> str:
    failure = _current_candidate_failure()
    if failure is None:
        return default
    failure_kind = str(failure.get("kind") or "")
    aliases = {
        "draft": {"draft", "catalog_draft"},
        "repair": {"repair", "catalog_repair"},
        "revision": {"revision", "catalog_revision"},
        "reports": {"reports", "report_refresh"},
    }
    if failure_kind not in aliases.get(kind, {kind}):
        return default
    return {
        "draft": "Retry AI Draft",
        "repair": "Retry AI Repair",
        "revision": "Retry AI Revision",
        "reports": "Retry Report Refresh",
    }.get(kind, f"Retry {default}")


def _current_candidate_failure() -> dict[str, Any] | None:
    """Return only a failure receipt that belongs to the active AI sharing scope."""

    failure = st.session_state.get("ai_studio_last_candidate_failure")
    if not isinstance(failure, dict):
        return None
    failure_scope = str(failure.get("scope_signature") or "")
    if failure_scope and failure_scope != _ai_sharing_signature():
        return None
    return failure


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
    components.key_value_strip(
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
    components.key_value_strip(
        [{"label": label, "value": value} for label, value in draft_object_counts(draft).items()]
    )


def _set_draft(
    draft: dict[str, Any],
    *,
    reviewed: bool = False,
    resume_checkpoint: bool = True,
) -> None:
    stored = yaml.safe_load(yaml.safe_dump(draft, sort_keys=False))
    st.session_state["ai_studio_draft"] = stored
    if (
        st.session_state.get("ai_studio_draft_source") != CATALOG_DRAFT_SOURCE
        and "ai_studio_effective_schema_columns" in st.session_state
    ):
        st.session_state[
            AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY
        ] = not _draft_matches_active_naming_contract(stored)
    st.session_state["ai_studio_reviewed_signature"] = _draft_signature(stored) if reviewed else ""
    if resume_checkpoint:
        st.session_state.pop(AI_STUDIO_CHECKPOINT_SUPPRESSED_WORKSPACE_KEY, None)
        if st.session_state.get("ai_studio_draft_source") == CATALOG_DRAFT_SOURCE:
            st.session_state[AI_STUDIO_CATALOG_DRAFT_DIRTY_KEY] = True
    st.session_state.pop("ai_studio_raw_metrics_yaml_signature", None)
    st.session_state.pop("ai_studio_raw_dashboards_yaml_signature", None)
    st.session_state.pop("ai_studio_settings_theme_yaml_signature", None)
    _persist_ai_studio_checkpoint()


def _clear_pending_ai_draft() -> None:
    st.session_state["ai_studio_pending_draft"] = None
    st.session_state["ai_studio_pending_base_draft"] = None
    st.session_state["ai_studio_pending_kind"] = ""
    st.session_state["ai_studio_pending_prompt"] = ""
    st.session_state["ai_studio_last_ai_response"] = ""


def _render_draft_validation(
    ok: bool,
    issues: list[str],
    *,
    revision: str,
    expanded: bool,
) -> None:
    with st.container(border=True):
        st.write(f"### Catalog validation · revision `{revision[:12]}`")
        blocking_issues, repairable_issues = classify_draft_validation_issues(issues)
        status = "OK"
        if not ok:
            status = (
                "Needs repair" if repairable_issues and not blocking_issues else "Needs attention"
            )
        components.key_value_strip(
            [
                {"label": "Status", "value": status},
                {"label": "Issues", "value": len(issues)},
            ]
        )
        if not issues:
            st.success("Draft catalog validates.")
            return
        with st.expander("Validation Details", expanded=expanded):
            for issue in blocking_issues:
                st.error(issue)
            for issue in repairable_issues:
                st.warning(issue)


def _tile_choice_label(draft: dict[str, Any], key: str) -> str:
    """Return a human-first Keep label with parent context and stable key."""

    try:
        dashboard_id, page_id, tile_id = key.split("/", 2)
    except ValueError:
        return key
    dashboards = draft.get("dashboards", {}).get("dashboards", [])
    for dashboard in dashboards if isinstance(dashboards, list) else []:
        if not isinstance(dashboard, dict) or str(dashboard.get("id")) != dashboard_id:
            continue
        for page in dashboard.get("pages", []) or []:
            if not isinstance(page, dict) or str(page.get("id")) != page_id:
                continue
            for tile in page.get("tiles", []) or []:
                if isinstance(tile, dict) and str(tile.get("id")) == tile_id:
                    title = str(tile.get("title") or builder.title_from_identifier(tile_id))
                    page_title = str(page.get("title") or builder.title_from_identifier(page_id))
                    return f"{title} · {page_title} — {key}"
    return key


def _tile_inventory_rows(draft: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metrics = _draft_metric_definitions(draft)
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
                # Skip id-less tiles so rows stay parallel to tile_keys(draft).
                if not isinstance(tile, dict) or not tile.get("id"):
                    continue
                metric_id = str(tile.get("metric", ""))
                metric = metrics.get(metric_id, {})
                display = metric.get("display") if isinstance(metric, dict) else {}
                metric_label = (
                    str(display.get("label") or "") if isinstance(display, dict) else ""
                ) or builder.title_from_identifier(metric_id)
                rows.append(
                    {
                        "Dashboard": str(
                            dashboard.get("title")
                            or builder.title_from_identifier(str(dashboard.get("id", "")))
                        ),
                        "Page": str(
                            page.get("title")
                            or builder.title_from_identifier(str(page.get("id", "")))
                        ),
                        "Report": str(
                            tile.get("title")
                            or builder.title_from_identifier(str(tile.get("id", "")))
                        ),
                        "Measure": metric_label,
                        "Visualization": builder.title_from_identifier(str(tile.get("chart", ""))),
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
        field_name_issues = _stale_preprocessing_field_name_issues()
        if field_name_issues:
            raise ValueError(" ".join(field_name_issues))
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


def _raw_to_effective_field_mapping() -> dict[str, str]:
    if not _rename_capitalize_enabled():
        return {}
    raw_columns = st.session_state.get("ai_studio_raw_schema_columns")
    if not isinstance(raw_columns, list):
        return {}
    return {
        raw: effective
        for raw, effective in _rename_capitalize_mapping(
            [str(item) for item in raw_columns]
        ).items()
        if raw != effective
    }


def _expression_stale_raw_fields(value: Any, mapping: dict[str, str]) -> set[str]:
    stale: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"column", "col"} and isinstance(item, str) and item in mapping:
                stale.add(item)
            elif key == "polars" and isinstance(item, str):
                stale.update(_polars_stale_raw_fields(item, mapping))
            else:
                stale.update(_expression_stale_raw_fields(item, mapping))
    elif isinstance(value, list):
        for item in value:
            stale.update(_expression_stale_raw_fields(item, mapping))
    return stale


def _polars_stale_raw_fields(expression: str, mapping: dict[str, str]) -> set[str]:
    return {
        raw
        for raw in mapping
        if re.search(rf"\bpl\.col\(\s*(['\"])({re.escape(raw)})\1\s*\)", expression)
    }


def _stale_preprocessing_field_name_issues(  # noqa: PLR0912
    section: str | None = None,
) -> list[str]:
    """Find raw Pega names used where the effective post-transform schema is required."""

    mapping = _raw_to_effective_field_mapping()
    if not mapping:
        return []
    findings: list[tuple[str, str]] = []
    if section in {None, "Defaults"}:
        for row in st.session_state.get("ai_studio_defaults", []) or []:
            if not isinstance(row, dict) or not builder.editor_row_enabled(row.get("Enabled")):
                continue
            field = str(row.get("Field") or "").strip()
            if field in mapping:
                findings.append(("Defaults", field))
    if section in {None, "Filters"}:
        if st.session_state.get("ai_studio_filter_mode") == "Raw AST":
            raw_filter = str(st.session_state.get("ai_studio_raw_filter") or "")
            try:
                expression = yaml.safe_load(raw_filter) if raw_filter.strip() else None
            except yaml.YAMLError:
                expression = None
            for field in _expression_stale_raw_fields(expression, mapping):
                findings.append(("Filters", field))
        else:
            for row in st.session_state.get("ai_studio_filter_rows", []) or []:
                if not isinstance(row, dict) or not builder.editor_row_enabled(row.get("Enabled")):
                    continue
                field = str(row.get("Field") or "").strip()
                if field in mapping:
                    findings.append(("Filters", field))
    if section in {None, "Calculations"}:
        for row in st.session_state.get("ai_studio_calculations", []) or []:
            if not isinstance(row, dict) or not builder.editor_row_enabled(row.get("Enabled")):
                continue
            direct_fields = {
                str(row.get("Name") or "").strip(),
                str(row.get("Left") or "").strip(),
            }
            if str(row.get("Right Kind") or "Field") == "Field":
                direct_fields.add(str(row.get("Right") or "").strip())
            mode = str(row.get("Mode") or "AST YAML")
            expression_text = str(row.get("Expression") or row.get("Expression YAML") or "")
            if mode == "Polars":
                direct_fields.update(_polars_stale_raw_fields(expression_text, mapping))
            elif mode == "AST YAML" and expression_text.strip():
                try:
                    expression = yaml.safe_load(expression_text)
                except yaml.YAMLError:
                    expression = None
                direct_fields.update(_expression_stale_raw_fields(expression, mapping))
            for field in direct_fields:
                if field in mapping:
                    findings.append(("Calculations", field))
    return [
        f"{area} uses raw field {raw!r}; use effective field {mapping[raw]!r} after "
        "Rename / Capitalize."
        for area, raw in sorted(set(findings))
    ]


def _render_stale_preprocessing_field_feedback(section: str) -> None:
    for issue in _stale_preprocessing_field_name_issues(section):
        st.error(issue)


def _rename_capitalize_enabled() -> bool:
    _migrate_rename_capitalize_state()
    return bool(st.session_state.get(AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY))


def _migrate_rename_capitalize_state() -> None:
    """Move pre-fix widget-owned state to the durable authoring key once."""

    if (
        AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY not in st.session_state
        and AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY in st.session_state
    ):
        st.session_state[AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = bool(
            st.session_state[AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY]
        )
    st.session_state.pop(AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY, None)


def _on_rename_capitalize_toggle_change() -> None:
    """Copy the transient widget value into durable, checkpointed authoring state."""

    st.session_state[AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = bool(
        st.session_state.get(AI_STUDIO_RENAME_CAPITALIZE_WIDGET_KEY)
    )


def _render_rename_capitalize_toggle() -> None:
    """Render the source transform control without coupling state to widget lifetime."""

    st.toggle(
        "Use Rename / Capitalize Transform",
        value=_rename_capitalize_enabled(),
        key=AI_STUDIO_RENAME_CAPITALIZE_WIDGET_KEY,
        help=config_help.field_help("source.rename_capitalize"),
        on_change=_on_rename_capitalize_toggle_change,
    )


def _rename_capitalize_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename(_rename_capitalize_mapping(frame.columns))


def _rename_capitalize_mapping(columns: list[str]) -> dict[str, str]:
    return dict(zip(columns, capitalize_fields(columns), strict=False))


def _draft_uses_rename_capitalize(draft: object) -> bool | None:
    """Return the active draft source's naming mode, or ``None`` when it is unavailable."""

    if not isinstance(draft, dict):
        return None
    source_id = str(st.session_state.get("ai_studio_source_id") or "").strip()
    sources = draft.get("pipelines", {}).get("sources", [])
    if not isinstance(sources, list):
        return None
    source = next(
        (
            item
            for item in sources
            if isinstance(item, dict) and str(item.get("id") or "") == source_id
        ),
        None,
    )
    if not isinstance(source, dict):
        return None
    transforms = source.get("transforms", [])
    if not isinstance(transforms, list):
        return False
    return any(
        isinstance(transform, dict) and transform.get("kind") == "rename_capitalize"
        for transform in transforms
    )


def _draft_matches_active_naming_contract(draft: object) -> bool:
    expected = _expected_rename_capitalize_contract()
    if expected is None or not str(st.session_state.get("ai_studio_source_id") or "").strip():
        return True
    actual = _draft_uses_rename_capitalize(draft)
    return actual is not None and actual == expected


def _sync_ai_rename_capitalize_state(sample: pl.DataFrame) -> None:
    st.session_state["ai_studio_raw_schema_columns"] = list(sample.columns)
    enabled = _rename_capitalize_enabled()
    applied = bool(st.session_state.get("ai_studio_rename_capitalize_applied", False))
    accepted_draft = st.session_state.get("ai_studio_draft")
    stale_draft = isinstance(accepted_draft, dict) and not _draft_matches_active_naming_contract(
        accepted_draft
    )
    was_stale = bool(st.session_state.get(AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY))
    st.session_state[AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY] = stale_draft
    if stale_draft:
        st.session_state["ai_studio_reviewed_signature"] = ""
        st.session_state["ai_studio_published_signature"] = ""
        st.session_state["ai_studio_validation_cache"] = {}
        if not was_stale:
            _clear_pending_ai_draft()
            _clear_ai_sharing_confirmation()
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
    _clear_pending_ai_draft()
    _clear_ai_sharing_confirmation()
    st.session_state["ai_studio_validation_cache"] = {}
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
    """Default stored grains: Day and Month only; coarser grains stay opt-in.

    Quarter, Year, and Summary remain selectable in the processor editors, and
    summary queries are still answered from the stored Month rows via the
    default aggregation levels.
    """
    grains: list[str] = [grain for grain in ("Day", "Month") if grain in working.columns]
    return grains if grains else ["Summary"]


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


def _observed_outcome_groups(
    working: pl.DataFrame,
    outcome_column: str,
) -> tuple[list[Any], list[Any]]:
    """Classify observed outcomes for a click-through processor without inventing values."""

    if outcome_column not in working.columns:
        return ["Clicked"], ["Impression"]
    observed = sorted(
        working.get_column(outcome_column).drop_nulls().unique().to_list(),
        key=lambda value: str(value).casefold(),
    )
    if not observed:
        return ["Clicked"], ["Impression"]
    positive = [value for value in observed if str(value).strip().casefold() == "clicked"]
    if not positive:
        positive_labels = {"positive", "yes", "true", "success"}
        positive = [value for value in observed if str(value).strip().casefold() in positive_labels]
    if not positive:
        positive = [observed[0]]
    positive_set = {str(value) for value in positive}
    negative = [value for value in observed if str(value) not in positive_set]
    return positive, negative


def _studio_baseline_dashboards(primary_x: str, group_by: list[str]) -> dict[str, Any]:
    """Build a useful three-page, six-tile no-provider report baseline."""

    dashboard_title = "Studio Overview"
    dashboard_id = builder.stable_catalog_id(dashboard_title, fallback="dashboard")
    dimension = group_by[0] if group_by else primary_x
    page_specs = (
        (
            "Engagement",
            (
                ("CTR Trend", "Studio_CTR", "line"),
                ("CTR By Dimension", "Studio_CTR", "bar"),
            ),
        ),
        (
            "Volume",
            (
                ("Outcome Volume Trend", "Studio_Count", "line"),
                ("Outcome Volume By Dimension", "Studio_Count", "bar"),
            ),
        ),
        (
            "Outcomes",
            (
                ("Positive Outcomes Trend", "Studio_Positive_Outcomes", "line"),
                ("Negative Outcomes Trend", "Studio_Negative_Outcomes", "line"),
            ),
        ),
    )
    pages: list[dict[str, Any]] = []
    used_page_ids: set[str] = set()
    for page_title, tile_specs in page_specs:
        page_id = builder.stable_catalog_id(
            page_title,
            fallback="page",
            parent_id=dashboard_id,
            existing_ids=used_page_ids,
        )
        used_page_ids.add(page_id)
        used_tile_ids: set[str] = set()
        tiles: list[dict[str, Any]] = []
        for title, metric_name, chart_kind in tile_specs:
            tile_id = builder.stable_catalog_id(
                title,
                fallback="tile",
                parent_id=page_id,
                existing_ids=used_tile_ids,
            )
            used_tile_ids.add(tile_id)
            fields: dict[str, Any] = {
                "x": primary_x if chart_kind == "line" else dimension,
                "y": metric_name,
            }
            if chart_kind == "line" and group_by:
                fields["color"] = group_by[0]
            tiles.append(
                builder.build_tile(
                    tile_id=tile_id,
                    title=title,
                    metric_name=metric_name,
                    chart_kind=chart_kind,
                    fields=fields,
                )
            )
        pages.append(
            {
                "id": page_id,
                "title": page_title,
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
        )
    return {
        "theme": {},
        "dashboards": [
            {
                "id": dashboard_id,
                "title": dashboard_title,
                "layout": "tabs",
                "pages": pages,
            }
        ],
    }


def _build_draft_catalog(working: pl.DataFrame, approved_fields: list[str]) -> dict[str, Any]:
    source_id = str(st.session_state.get("ai_studio_source_id") or "ih").strip() or "ih"
    subject = (
        "SubjectID"
        if "SubjectID" in working.columns
        else st.session_state.get("ai_studio_subject", "")
    )
    time_column = _primary_time_field(working)
    outcome_column = st.session_state.get("ai_studio_outcome_column", "Outcome")
    positive_values, negative_values = _observed_outcome_groups(working, outcome_column)
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
    reader_root = str(st.session_state.get("ai_studio_reader_root") or "").strip()
    if reader_root:
        reader["root"] = reader_root
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

    dashboards = _studio_baseline_dashboards(primary_x, group_by)
    return {
        "pipelines": {
            "version": 1,
            "workspace": str(
                st.session_state.get("ai_studio_active_workspace_name") or "workspace"
            ),
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
                            "positive_values": positive_values,
                            "negative_values": negative_values,
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
                    "display": {"label": "Click-through rate", "value_format": "percent"},
                },
                "Studio_Count": {
                    "source": processor_id,
                    "kind": "formula",
                    "description": "Total outcome rows generated by AI Configuration Studio.",
                    "expression": {"col": "Count"},
                    "display": {"label": "Total outcomes", "value_format": "integer"},
                },
                "Studio_Positive_Outcomes": {
                    "source": processor_id,
                    "kind": "formula",
                    "description": "Observed positive outcomes in the selected scope.",
                    "expression": {"col": "Positives"},
                    "display": {"label": "Positive outcomes", "value_format": "integer"},
                },
                "Studio_Negative_Outcomes": {
                    "source": processor_id,
                    "kind": "formula",
                    "description": "Observed negative outcomes in the selected scope.",
                    "expression": {"col": "Negatives"},
                    "display": {"label": "Negative outcomes", "value_format": "integer"},
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
    snapshot = _draft_validation_snapshot(draft)
    if not snapshot.ok:
        details = "\n".join(f"- {issue}" for issue in snapshot.issues)
        raise ValueError(f"AI draft validation failed before apply:\n{details}")

    with builder.workspace_configuration_transaction(ctx.workspace):
        # Every accepted section is authoritative for this revision. Full-section
        # replacement makes reviewed removals durable while the surrounding
        # workspace transaction preserves all-or-nothing apply behavior.
        builder.write_pipelines_definition(ctx.workspace, draft["pipelines"])
        builder.write_processors_definition(ctx.workspace, draft["processors"])
        builder.write_metrics_definition(ctx.workspace, draft["metrics"])
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
        builder.require_valid_workspace(
            ctx.workspace,
            source_columns_by_id=_active_catalog_source_columns(),
        )


def _draft_requires_data_run(ctx: ValueStreamContext, draft: dict[str, Any]) -> bool:
    """Return whether this revision changes any persisted computation contract."""

    try:
        proposed = model.Catalog.model_validate(
            {
                section: draft[section]
                for section in ("pipelines", "processors", "metrics", "dashboards")
            }
        )
        current_by_id = {processor.id: processor for processor in ctx.catalog.processors.processors}
        for processor in proposed.processors.processors:
            current = current_by_id.get(processor.id)
            if current is None:
                return True
            if processor_computation_hash(proposed, processor) != processor_computation_hash(
                ctx.catalog, current
            ):
                return True
        return False
    except Exception as exc:
        _log_ai_operation_failure("Draft impact classification", exc)
        # Conservative fallback: direct users to Data Load when impact cannot be proven.
        return True


def _production_source_ready() -> bool:
    if st.session_state.get("ai_studio_draft_source") == CATALOG_DRAFT_SOURCE:
        return True
    plan = st.session_state.get("ai_studio_sample_source_plan")
    if not isinstance(plan, SampleSourcePlan) or not plan.production_ready:
        return False
    return (
        str(st.session_state.get("ai_studio_reader_kind") or "") == plan.reader_kind
        and str(st.session_state.get("ai_studio_reader_root") or "") == plan.root
        and str(st.session_state.get("ai_studio_file_pattern") or "") == plan.file_pattern
    )


def _render_outcome_receipt() -> None:
    receipt = st.session_state.get("ai_studio_outcome_receipt")
    if not isinstance(receipt, dict):
        return
    with st.container(border=True):
        st.write("#### Revision receipt")
        if receipt.get("applied"):
            st.success("The reviewed revision is applied and the workspace validates.")
        else:
            st.warning("The workspace needs attention before this revision can be used.")
        requires_data_run = bool(receipt.get("requires_data_run"))
        rows = [
            {"label": "Revision", "value": receipt.get("revision", "")},
            {
                "label": "Configuration",
                "value": "Applied" if receipt.get("applied") else "Blocked",
            },
            {
                "label": "Data impact",
                "value": "Run required" if requires_data_run else "No computation change",
            },
            {"label": "Sources", "value": int(receipt.get("source_count", 0))},
        ]
        removed_count = int(receipt.get("removed_object_count", 0) or 0)
        if removed_count:
            rows.append({"label": "Removed existing objects", "value": removed_count})
        components.key_value_strip(rows)
        if _builder_source_handoff():
            st.link_button(
                "Return to Configuration Builder",
                f"{BUILDER_SOURCE_RETURN_URL}_applied",
                icon=":material/arrow_back:",
                type="primary",
            )
            if requires_data_run:
                st.caption(
                    "The source definition is applied. Return to Builder now; run its data "
                    "separately when you are ready to materialize aggregates."
                )
        elif requires_data_run:
            st.link_button(
                "Run data",
                "/data_load?from=ai_studio",
                icon=":material/database:",
                type="primary",
            )
        else:
            st.link_button(
                "Open report",
                "/reports?from=ai_studio",
                icon=":material/monitoring:",
                type="primary",
            )


def _mark_draft_published(draft: dict[str, Any]) -> None:
    st.session_state["ai_studio_published_signature"] = _draft_signature(draft)


def _studio_status_bar(
    ctx: ValueStreamContext,
    raw: pl.DataFrame,
    working: pl.DataFrame,
    approved_fields: list[str],
    preprocessing_error: str | None,
) -> None:
    """Show lifecycle evidence without inferring readiness from object counts."""
    del ctx  # Apply is intentionally available only on the final step.
    ai_calls_enabled = _ai_calls_enabled()
    draft = st.session_state.get("ai_studio_draft")
    pending = st.session_state.get("ai_studio_pending_draft")
    snapshot = _draft_validation_snapshot(draft) if isinstance(draft, dict) else None
    reviewed = bool(
        snapshot
        and snapshot.ok
        and st.session_state.get("ai_studio_reviewed_signature") == snapshot.signature
    )
    applied = bool(
        snapshot
        and snapshot.ok
        and st.session_state.get("ai_studio_published_signature") == snapshot.signature
    )
    with st.container(border=True):
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
                ("warning" if ai_calls_enabled and pending else ("ready" if draft else "pending")),
                help="Pending AI output needs review." if ai_calls_enabled and pending else None,
            )
            components.status_badge(
                "Validation",
                "ready" if snapshot and snapshot.ok else "blocked" if snapshot else "pending",
                help=snapshot.issues[0] if snapshot and snapshot.issues else None,
            )
            components.status_badge(
                "Review",
                "ready" if reviewed else "warning" if snapshot and snapshot.ok else "pending",
            )
            components.status_badge("Workspace", "ready" if applied else "pending")
        st.caption(
            f"{len(raw.columns)} raw columns · {len(working.columns)} working columns · "
            f"{len(approved_fields)} approved fields · {working.height:,} sample rows"
        )


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


def _on_field_approval_editor_change(
    editor_key: str,
    visible_fields: tuple[str, ...],
    available_fields: tuple[str, ...],
    required_fields: tuple[str, ...],
) -> None:
    """Commit one data-editor delta before its fragment reruns."""

    editor_state = st.session_state.get(editor_key)
    if not isinstance(editor_state, Mapping):
        return
    edited_rows = editor_state.get("edited_rows")
    if not isinstance(edited_rows, Mapping):
        return

    previous_approved_fields = list(st.session_state.get("ai_studio_approved_fields") or [])
    previous_example_fields = list(st.session_state.get("ai_studio_example_fields") or [])
    approved_set = set(previous_approved_fields)
    example_set = set(previous_example_fields)
    changed_rows: list[dict[str, Any]] = []
    for raw_index, changes in edited_rows.items():
        if not isinstance(changes, Mapping):
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(visible_fields):
            continue
        field = visible_fields[index]
        changed_rows.append(
            {
                "Column": field,
                "Approve": changes.get("Approve", field in approved_set),
                "Send To AI": changes.get(
                    "Send To AI",
                    changes.get("Share Sample Values", field in example_set),
                ),
            }
        )
    if not changed_rows:
        return

    approved_fields, example_fields = _apply_field_approval_edits(
        changed_rows,
        available_fields=list(available_fields),
        required_fields=list(required_fields),
        approved_fields=previous_approved_fields,
        example_fields=previous_example_fields,
    )
    approved_fields, example_fields, group_by_fields = _normalize_field_approval_state(
        available_fields=list(available_fields),
        required_fields=list(required_fields),
        approved_fields=approved_fields,
        example_fields=example_fields,
        group_by_fields=list(st.session_state.get("ai_studio_group_by_fields") or []),
    )
    st.session_state["ai_studio_approved_fields"] = approved_fields
    st.session_state["ai_studio_example_fields"] = example_fields
    st.session_state["ai_studio_group_by_fields"] = group_by_fields
    _invalidate_ai_sharing_confirmation_if_scope_changed(
        previous_approved_fields=previous_approved_fields,
        previous_example_fields=previous_example_fields,
        approved_fields=approved_fields,
        example_fields=example_fields,
    )


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


def _normalized_role_name(value: str) -> str:
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    if len(normalized) > 2 and normalized[:2] in {"px", "py", "pz"}:
        return normalized[2:]
    return normalized


def _default_time_column(columns: list[str], preferred: str, *, fallback: bool = True) -> str:
    if preferred in columns:
        return preferred
    preferred_key = _normalized_role_name(preferred)
    for column in columns:
        if _normalized_role_name(column) == preferred_key:
            return column
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
    preferred_key = _normalized_role_name(preferred)
    for column in columns:
        if _normalized_role_name(column) == preferred_key:
            return column
    for column in columns:
        if preferred.casefold() in column.casefold():
            return column
    return columns[0] if fallback and columns else ""


def _default_outcome_column(sample: pl.DataFrame) -> str:
    """Choose a categorical outcome field without mistaking time columns for outcomes."""

    scored: list[tuple[int, str]] = []
    for index, column in enumerate(sample.columns):
        normalized = _normalized_role_name(column)
        lowered = column.casefold()
        if any(token in lowered for token in ("time", "date", "timestamp")):
            continue
        score = 0
        if column == "Outcome":
            score += 100
        elif normalized == "outcome":
            score += 90
        elif "outcome" in normalized:
            score += 25
        else:
            continue
        dtype_name = str(sample.schema[column]).casefold()
        if any(token in dtype_name for token in ("str", "categorical", "enum", "bool")):
            score += 15
        unique_count = sample.get_column(column).n_unique()
        if unique_count <= 50:
            score += 20
        elif sample.height and unique_count > max(100, sample.height // 2):
            score -= 30
        scored.append((score * 1_000 - index, column))
    if scored:
        return max(scored)[1]
    return _default_column(sample.columns, "Outcome")


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
