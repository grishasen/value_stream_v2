"""Privacy-safe workspace checkpoints for accepted AI Studio authoring state."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from valuestream.ui import builder_checkpoint

CHECKPOINT_FILENAME = "ai_config_studio_checkpoint.json"
CHECKPOINT_VERSION = 1
CHECKPOINT_RETENTION = builder_checkpoint.CHECKPOINT_RETENTION

CheckpointStatus = Literal["missing", "ready", "reconciliation", "expired", "invalid"]

_CATALOG_SECTIONS = ("pipelines", "processors", "metrics", "dashboards")
_MAPPING_KEYS = (
    "ai_studio_source_id",
    "ai_studio_reader_kind",
    "ai_studio_reader_root",
    "ai_studio_file_pattern",
    "ai_studio_group_pattern",
    "ai_studio_streaming",
    "ai_studio_hive_partitioning",
    "ai_studio_timestamp_format",
    "ai_studio_subject",
    "ai_studio_outcome_time",
    "ai_studio_decision_time",
    "ai_studio_outcome_column",
    "ai_studio_day_column",
    "ai_studio_month_column",
    "ai_studio_year_column",
    "ai_studio_quarter_column",
    "ai_studio_rename_capitalize",
)
_PREPROCESSING_KEYS = (
    "ai_studio_defaults",
    "ai_studio_filter_mode",
    "ai_studio_filter_rows",
    "ai_studio_raw_filter",
    "ai_studio_calculations",
)
_FIELD_SET_KEYS = (
    "ai_studio_approved_fields",
    "ai_studio_example_fields",
    "ai_studio_group_by_fields",
)
_DRAFT_METADATA_KEYS = (
    "ai_studio_draft_source",
    "ai_studio_catalog_draft_step",
)
_WORKSPACE_AUTHORING_KEYS = (*_MAPPING_KEYS, *_PREPROCESSING_KEYS, *_FIELD_SET_KEYS)
_ALLOWED_STATE_KEYS = frozenset(
    {
        "ai_studio_sample_workspace_relative",
        "ai_studio_sample_identity",
        "ai_studio_draft",
        "ai_studio_reviewed_signature",
        *_DRAFT_METADATA_KEYS,
        *_WORKSPACE_AUTHORING_KEYS,
    }
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class AIStudioCheckpoint:
    """One validated AI Studio recovery candidate."""

    saved_at: dt.datetime
    base_catalog_hash: str
    current_step: str
    requires_sample_reselect: bool
    state: dict[str, Any]


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Safe AI Studio checkpoint load outcome."""

    status: CheckpointStatus
    checkpoint: AIStudioCheckpoint | None = None


def checkpoint_path(workspace: str | Path) -> Path:
    return Path(workspace) / "meta" / CHECKPOINT_FILENAME


def capture_safe_authoring_state(session_state: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Capture only committed, allowlisted state and report whether sample reselect is required."""

    safe: dict[str, Any] = {}
    draft = _safe_catalog_draft(session_state.get("ai_studio_draft"))
    if draft is not None:
        safe["ai_studio_draft"] = draft
        reviewed = session_state.get("ai_studio_reviewed_signature")
        if isinstance(reviewed, str) and _HASH_PATTERN.fullmatch(reviewed):
            safe["ai_studio_reviewed_signature"] = reviewed
        for key in _DRAFT_METADATA_KEYS:
            value = session_state.get(key)
            if isinstance(value, str):
                safe[key] = value

    relative = str(session_state.get("ai_studio_sample_workspace_relative") or "")
    identity = str(session_state.get("ai_studio_sample_identity") or "")
    workspace_sample = _valid_workspace_relative(relative) and bool(
        _HASH_PATTERN.fullmatch(identity)
    )
    if workspace_sample:
        workspace_state: dict[str, Any] = {
            "ai_studio_sample_workspace_relative": relative,
            "ai_studio_sample_identity": identity,
        }
        try:
            for key in _WORKSPACE_AUTHORING_KEYS:
                if key in session_state:
                    workspace_state[key] = builder_checkpoint.require_safe_json(session_state[key])
        except ValueError:
            workspace_state = {}
        safe.update(workspace_state)

    requires_reselect = not workspace_sample
    if requires_reselect:
        safe = {key: value for key, value in safe.items() if key not in _WORKSPACE_AUTHORING_KEYS}
        safe.pop("ai_studio_sample_workspace_relative", None)
        safe.pop("ai_studio_sample_identity", None)
    return safe, requires_reselect


def write_ai_studio_checkpoint(
    workspace: str | Path,
    *,
    session_state: Mapping[str, Any],
    current_step: str,
    base_catalog_hash: str,
    now: dt.datetime | None = None,
) -> Path | None:
    safe_state, requires_reselect = capture_safe_authoring_state(session_state)
    if not safe_state:
        discard_ai_studio_checkpoint(workspace)
        return None
    if not current_step:
        raise ValueError("current AI Studio step is required")
    if _HASH_PATTERN.fullmatch(base_catalog_hash) is None:
        raise ValueError("base catalog hash must be a full sha256 digest")
    saved_at = _utc_now(now)
    return builder_checkpoint.atomic_write_json(
        checkpoint_path(workspace),
        {
            "version": CHECKPOINT_VERSION,
            "saved_at": saved_at.isoformat().replace("+00:00", "Z"),
            "base_catalog_hash": base_catalog_hash,
            "current_step": current_step,
            "requires_sample_reselect": requires_reselect,
            "state": safe_state,
        },
    )


def load_ai_studio_checkpoint(
    workspace: str | Path,
    *,
    current_catalog_hash: str,
    allowed_steps: Iterable[str],
    now: dt.datetime | None = None,
    retention: dt.timedelta = CHECKPOINT_RETENTION,
) -> CheckpointLoadResult:
    path = checkpoint_path(workspace)
    if not path.is_file():
        return CheckpointLoadResult("missing")
    try:
        raw = builder_checkpoint.read_bounded_json(path)
        checkpoint = _parse_checkpoint(raw, allowed_steps=allowed_steps)
        current_time = _utc_now(now)
        if checkpoint.saved_at > current_time + dt.timedelta(minutes=5):
            raise ValueError("checkpoint timestamp is in the future")
        if current_time - checkpoint.saved_at > retention:
            discard_ai_studio_checkpoint(workspace)
            return CheckpointLoadResult("expired")
    except (OSError, TypeError, ValueError):
        discard_ai_studio_checkpoint(workspace)
        return CheckpointLoadResult("invalid")
    status: CheckpointStatus = (
        "ready" if checkpoint.base_catalog_hash == current_catalog_hash else "reconciliation"
    )
    return CheckpointLoadResult(status, checkpoint)


def discard_ai_studio_checkpoint(workspace: str | Path) -> None:
    checkpoint_path(workspace).unlink(missing_ok=True)


def _safe_catalog_draft(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not all(section in value for section in _CATALOG_SECTIONS):
        return None
    core = {section: value[section] for section in _CATALOG_SECTIONS}
    try:
        safe = builder_checkpoint.require_safe_json(core)
    except ValueError:
        return None
    return safe if isinstance(safe, dict) else None


def _parse_checkpoint(value: Any, *, allowed_steps: Iterable[str]) -> AIStudioCheckpoint:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "saved_at",
        "base_catalog_hash",
        "current_step",
        "requires_sample_reselect",
        "state",
    }:
        raise ValueError("invalid checkpoint shape")
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    base_hash = value.get("base_catalog_hash")
    if not isinstance(base_hash, str) or _HASH_PATTERN.fullmatch(base_hash) is None:
        raise ValueError("invalid base catalog hash")
    current_step = value.get("current_step")
    if not isinstance(current_step, str) or current_step not in tuple(allowed_steps):
        raise ValueError("invalid current step")
    requires_reselect = value.get("requires_sample_reselect")
    if not isinstance(requires_reselect, bool):
        raise ValueError("invalid sample recovery state")
    state = _validate_state(value.get("state"), requires_reselect=requires_reselect)
    return AIStudioCheckpoint(
        saved_at=_parse_utc(value.get("saved_at")),
        base_catalog_hash=base_hash,
        current_step=current_step,
        requires_sample_reselect=requires_reselect,
        state=state,
    )


def _validate_state(value: Any, *, requires_reselect: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or not value or not set(value).issubset(_ALLOWED_STATE_KEYS):
        raise ValueError("invalid checkpoint state")
    # The outer keys are an explicit Studio allowlist.  Validate each value in
    # isolation so the shared generic sanitizer does not reject intentional
    # names such as ``ai_studio_sample_identity`` merely because they contain a
    # privacy-sensitive token.  Nested user-authored keys and values still pass
    # through the strict sanitizer.
    safe = {key: builder_checkpoint.require_safe_json(nested) for key, nested in value.items()}
    draft = safe.get("ai_studio_draft")
    if draft is not None and _safe_catalog_draft(draft) != draft:
        raise ValueError("invalid accepted draft")
    if requires_reselect:
        forbidden = {
            "ai_studio_sample_workspace_relative",
            "ai_studio_sample_identity",
            *_WORKSPACE_AUTHORING_KEYS,
        }
        if forbidden.intersection(safe):
            raise ValueError("upload checkpoint contains sample-derived state")
    else:
        relative = safe.get("ai_studio_sample_workspace_relative")
        identity = safe.get("ai_studio_sample_identity")
        if not isinstance(relative, str) or not _valid_workspace_relative(relative):
            raise ValueError("invalid workspace sample path")
        if not isinstance(identity, str) or _HASH_PATTERN.fullmatch(identity) is None:
            raise ValueError("invalid workspace sample identity")
    return safe


def _valid_workspace_relative(value: str) -> bool:
    if not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.parts[0] == "data"


def _parse_utc(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("invalid saved timestamp")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("saved timestamp must be timezone-aware")
    return parsed.astimezone(dt.UTC)


def _utc_now(value: dt.datetime | None) -> dt.datetime:
    current = value or dt.datetime.now(dt.UTC)
    if current.tzinfo is None:
        raise ValueError("checkpoint time must be timezone-aware")
    return current.astimezone(dt.UTC)


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_RETENTION",
    "AIStudioCheckpoint",
    "CheckpointLoadResult",
    "capture_safe_authoring_state",
    "checkpoint_path",
    "discard_ai_studio_checkpoint",
    "load_ai_studio_checkpoint",
    "write_ai_studio_checkpoint",
]
