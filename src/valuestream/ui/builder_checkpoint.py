"""Privacy-safe workspace checkpoints for Configuration Builder drafts."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CHECKPOINT_FILENAME = "config_builder_checkpoint.json"
CHECKPOINT_VERSION = 1
CHECKPOINT_RETENTION = dt.timedelta(days=7)
CHECKPOINT_MAX_BYTES = 1_048_576

CheckpointStatus = Literal["missing", "ready", "reconciliation", "expired", "invalid"]

_ENTRY_FIELDS = (
    "revision",
    "baseline_hash",
    "draft_hash",
    "draft_payload",
    "widget_state",
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~+/=-]{8,}|sk-[a-z0-9_-]{8,}|"
    r"-----BEGIN\s+[A-Z ]*PRIVATE KEY-----)"
)
_SENSITIVE_TOKENS = frozenset(
    {
        "api",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "chat",
        "password",
        "prompt",
        "provider",
        "sample",
        "secret",
        "token",
        "upload",
    }
)
_SENSITIVE_COMPACT_MARKERS = (
    "accesskey",
    "apikey",
    "clientsecret",
    "datasetdescriptions",
    "metricdescriptions",
    "providerpayload",
    "providerrequest",
    "providerresponse",
    "rawprovider",
    "requestpayload",
    "responsepayload",
)
_OMIT = object()


@dataclass(frozen=True)
class BuilderCheckpoint:
    """One validated workspace-local Builder recovery candidate."""

    saved_at: dt.datetime
    base_catalog_hash: str
    current_step: str
    drafts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Safe checkpoint load outcome without file or parser internals."""

    status: CheckpointStatus
    checkpoint: BuilderCheckpoint | None = None


def checkpoint_path(workspace: str | Path) -> Path:
    """Return the workspace-local Builder checkpoint path."""

    return Path(workspace) / "meta" / CHECKPOINT_FILENAME


def sanitize_draft_registry(value: Any) -> dict[str, dict[str, Any]]:
    """Return only JSON-safe, non-secret fields from the Builder draft registry."""

    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, dict[str, Any]] = {}
    for draft_key, raw_entry in value.items():
        if not isinstance(draft_key, str) or not draft_key or not isinstance(raw_entry, Mapping):
            continue
        if draft_key.startswith("chat:"):
            continue
        entry: dict[str, Any] = {}
        incomplete = False
        for field in _ENTRY_FIELDS:
            if field not in raw_entry:
                continue
            safe_value = _json_safe_value(raw_entry[field])
            if safe_value is _OMIT:
                incomplete = field in {"draft_payload", "widget_state"}
                continue
            if field in {"draft_payload", "widget_state"} and safe_value != raw_entry[field]:
                incomplete = True
            entry[field] = safe_value
        if incomplete or not _valid_registry_entry(entry):
            continue
        sanitized[draft_key] = entry
    return sanitized


def require_safe_json(value: Any) -> Any:
    """Return an isolated JSON-safe value or reject any privacy-changing cleanup."""

    safe_value = _json_safe_value(value)
    if safe_value is _OMIT or safe_value != value:
        raise ValueError("checkpoint state contains unsupported or sensitive data")
    return safe_value


def write_builder_checkpoint(
    workspace: str | Path,
    *,
    drafts: Any,
    current_step: str,
    base_catalog_hash: str,
    now: dt.datetime | None = None,
) -> Path | None:
    """Atomically persist a bounded safe draft registry, or delete an empty one."""

    path = checkpoint_path(workspace)
    safe_drafts = sanitize_draft_registry(drafts)
    if not safe_drafts:
        discard_builder_checkpoint(workspace)
        return None
    if not current_step.strip():
        raise ValueError("current Builder step is required")
    if _HASH_PATTERN.fullmatch(base_catalog_hash) is None:
        raise ValueError("base catalog hash must be a full sha256 digest")

    saved_at = _utc_now(now)
    payload = {
        "version": CHECKPOINT_VERSION,
        "saved_at": _format_utc(saved_at),
        "base_catalog_hash": base_catalog_hash,
        "current_step": current_step,
        "drafts": safe_drafts,
    }
    return atomic_write_json(path, payload)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write one size-bounded JSON checkpoint document."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(serialized) > CHECKPOINT_MAX_BYTES:
        raise ValueError("Builder checkpoint exceeds the safe size limit")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def read_bounded_json(path: Path) -> Any:
    """Read one checkpoint JSON document after enforcing the shared size ceiling."""

    if path.stat().st_size > CHECKPOINT_MAX_BYTES:
        raise ValueError("checkpoint is oversized")
    return json.loads(path.read_text(encoding="utf-8"))


def load_builder_checkpoint(
    workspace: str | Path,
    *,
    current_catalog_hash: str,
    allowed_steps: Iterable[str] = (),
    now: dt.datetime | None = None,
    retention: dt.timedelta = CHECKPOINT_RETENTION,
) -> CheckpointLoadResult:
    """Load one safe checkpoint and classify catalog reconciliation requirements."""

    path = checkpoint_path(workspace)
    if not path.is_file():
        return CheckpointLoadResult("missing")
    try:
        raw = read_bounded_json(path)
        checkpoint = _parse_checkpoint(raw, allowed_steps=allowed_steps)
        current_time = _utc_now(now)
        if checkpoint.saved_at > current_time + dt.timedelta(minutes=5):
            raise ValueError("checkpoint timestamp is in the future")
        if current_time - checkpoint.saved_at > retention:
            discard_builder_checkpoint(workspace)
            return CheckpointLoadResult("expired")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        discard_builder_checkpoint(workspace)
        return CheckpointLoadResult("invalid")

    status: CheckpointStatus = (
        "ready" if checkpoint.base_catalog_hash == current_catalog_hash else "reconciliation"
    )
    return CheckpointLoadResult(status, checkpoint)


def discard_builder_checkpoint(workspace: str | Path) -> None:
    """Delete the workspace-local Builder checkpoint if it exists."""

    checkpoint_path(workspace).unlink(missing_ok=True)


def _parse_checkpoint(
    value: Any,
    *,
    allowed_steps: Iterable[str],
) -> BuilderCheckpoint:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "saved_at",
        "base_catalog_hash",
        "current_step",
        "drafts",
    }:
        raise ValueError("invalid checkpoint shape")
    if value.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    catalog_hash = value.get("base_catalog_hash")
    if not isinstance(catalog_hash, str) or _HASH_PATTERN.fullmatch(catalog_hash) is None:
        raise ValueError("invalid base catalog hash")
    current_step = value.get("current_step")
    if not isinstance(current_step, str) or not current_step:
        raise ValueError("invalid current step")
    allowed = tuple(allowed_steps)
    if allowed and current_step not in allowed:
        raise ValueError("unknown current step")
    raw_drafts = value.get("drafts")
    safe_drafts = sanitize_draft_registry(raw_drafts)
    if not safe_drafts or safe_drafts != raw_drafts:
        raise ValueError("unsafe or empty draft registry")
    saved_at = _parse_utc(value.get("saved_at"))
    return BuilderCheckpoint(
        saved_at=saved_at,
        base_catalog_hash=catalog_hash,
        current_step=current_step,
        drafts=safe_drafts,
    )


def _valid_registry_entry(entry: Mapping[str, Any]) -> bool:
    revision = entry.get("revision")
    baseline_hash = entry.get("baseline_hash")
    draft_hash = entry.get("draft_hash")
    widget_state = entry.get("widget_state")
    return (
        isinstance(revision, str)
        and bool(revision)
        and isinstance(baseline_hash, str)
        and _HASH_PATTERN.fullmatch(baseline_hash) is not None
        and isinstance(draft_hash, str)
        and _HASH_PATTERN.fullmatch(draft_hash) is not None
        and "draft_payload" in entry
        and isinstance(widget_state, dict)
        and bool(widget_state)
    )


def _json_safe_value(value: Any, *, depth: int = 0) -> Any:  # noqa: PLR0911
    if depth > 32:
        return _OMIT
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _OMIT
    if isinstance(value, str):
        return _OMIT if _SECRET_VALUE_PATTERN.search(value) else value
    if isinstance(value, bytes | bytearray | memoryview):
        return _OMIT
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or _sensitive_key(key):
                continue
            safe_nested = _json_safe_value(nested, depth=depth + 1)
            if safe_nested is not _OMIT:
                cleaned[key] = safe_nested
        return cleaned
    if isinstance(value, list | tuple):
        cleaned_items = []
        for nested in value:
            safe_nested = _json_safe_value(nested, depth=depth + 1)
            if safe_nested is not _OMIT:
                cleaned_items.append(safe_nested)
        return cleaned_items
    return _OMIT


def _sensitive_key(key: str) -> bool:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake.casefold()).strip("_")
    tokens = frozenset(token for token in normalized.split("_") if token)
    compact = normalized.replace("_", "")
    return bool(tokens.intersection(_SENSITIVE_TOKENS)) or any(
        marker in compact for marker in _SENSITIVE_COMPACT_MARKERS
    )


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


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_MAX_BYTES",
    "CHECKPOINT_RETENTION",
    "BuilderCheckpoint",
    "CheckpointLoadResult",
    "atomic_write_json",
    "checkpoint_path",
    "discard_builder_checkpoint",
    "load_builder_checkpoint",
    "read_bounded_json",
    "require_safe_json",
    "sanitize_draft_registry",
    "write_builder_checkpoint",
]
