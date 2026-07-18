"""Privacy-safe authoring funnel instrumentation.

The event contract intentionally accepts only enumerated dimensions and bounded
numbers. It has no generic metadata mapping, so field names, sample values,
prompts, credentials, local paths, and catalog object identifiers cannot be
added accidentally at a call site.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import MutableMapping
from enum import StrEnum
from typing import Any

from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

_JOURNEY_ID_KEY = "vs_authoring_journey_id"
_JOURNEY_STARTED_KEY = "vs_authoring_journey_started"
_JOURNEY_STAGE_KEY = "vs_authoring_journey_stage"
_JOURNEY_WORKFLOW_KEY = "vs_authoring_journey_workflow"
_JOURNEY_COMPLETE_KEY = "vs_authoring_journey_complete"
_ONCE_KEYS = "vs_authoring_event_once"


class AuthoringWorkflow(StrEnum):
    """Allowlisted authoring surfaces."""

    BUILD = "build"
    BUILDER = "builder"
    AI_STUDIO = "ai_studio"


class AuthoringStage(StrEnum):
    """Allowlisted funnel stages."""

    ENTRY = "entry"
    SAMPLE = "sample"
    CONSENT = "consent"
    DRAFT = "draft"
    REVIEW = "review"
    APPLY = "apply"
    RUN = "run"
    REPORT = "report"
    OUTCOME = "outcome"


class AuthoringEvent(StrEnum):
    """Allowlisted funnel events."""

    ENTERED = "entered"
    SAMPLE_CHOSEN = "sample_chosen"
    CONSENT_CONFIRMED = "consent_confirmed"
    DRAFT_REQUESTED = "draft_requested"
    VALID_PROPOSAL = "valid_proposal"
    REVIEWED = "reviewed"
    APPLIED = "applied"
    RUN_STARTED = "run_started"
    REPORT_OPENED = "report_opened"
    FAILED = "failed"
    ABANDONED = "abandoned"


class AuthoringOutcome(StrEnum):
    """Bounded result labels; never provider- or user-controlled text."""

    STARTED = "started"
    SUCCESS = "success"
    BLOCKED = "blocked"
    RETRY = "retry"
    TIMEOUT = "timeout"
    ERROR = "error"
    DISCARDED = "discarded"


def workflow_from_handoff(value: object) -> AuthoringWorkflow | None:
    """Resolve the only authoring origins allowed in a handoff query parameter."""

    normalized = str(value or "").strip().casefold()
    if normalized == AuthoringWorkflow.BUILDER.value:
        return AuthoringWorkflow.BUILDER
    if normalized == AuthoringWorkflow.AI_STUDIO.value:
        return AuthoringWorkflow.AI_STUDIO
    return None


def start_journey(
    session_state: MutableMapping[str, Any],
    *,
    workflow: AuthoringWorkflow,
) -> str:
    """Start or return the current anonymous authoring journey."""

    journey_id = str(session_state.get(_JOURNEY_ID_KEY) or "")
    if journey_id:
        if session_state.get(_JOURNEY_COMPLETE_KEY):
            _clear_journey_state(session_state)
        else:
            raw_workflow = str(session_state.get(_JOURNEY_WORKFLOW_KEY) or "")
            if raw_workflow and raw_workflow != workflow.value:
                abandon_active_journey(session_state)
            else:
                session_state[_JOURNEY_WORKFLOW_KEY] = workflow.value
                return journey_id
    journey_id = secrets.token_hex(8)
    session_state[_JOURNEY_ID_KEY] = journey_id
    session_state[_JOURNEY_STARTED_KEY] = time.monotonic()
    session_state[_JOURNEY_STAGE_KEY] = AuthoringStage.ENTRY.value
    session_state[_JOURNEY_WORKFLOW_KEY] = workflow.value
    session_state[_JOURNEY_COMPLETE_KEY] = False
    session_state[_ONCE_KEYS] = []
    record_event(
        session_state,
        event=AuthoringEvent.ENTERED,
        workflow=workflow,
        stage=AuthoringStage.ENTRY,
        outcome=AuthoringOutcome.STARTED,
        once=True,
    )
    return journey_id


def record_event(
    session_state: MutableMapping[str, Any],
    *,
    event: AuthoringEvent,
    workflow: AuthoringWorkflow,
    stage: AuthoringStage,
    outcome: AuthoringOutcome,
    duration_ms: int | None = None,
    count: int | None = None,
    requires_data_run: bool | None = None,
    once: bool = False,
) -> bool:
    """Record one bounded event and return whether it was emitted."""

    event = AuthoringEvent(event)
    workflow = AuthoringWorkflow(workflow)
    stage = AuthoringStage(stage)
    outcome = AuthoringOutcome(outcome)
    journey_id = str(session_state.get(_JOURNEY_ID_KEY) or "")
    if not journey_id:
        journey_id = secrets.token_hex(8)
        session_state[_JOURNEY_ID_KEY] = journey_id
        session_state[_JOURNEY_STARTED_KEY] = time.monotonic()
        session_state[_ONCE_KEYS] = []

    once_key = f"{workflow.value}:{event.value}:{stage.value}:{outcome.value}"
    emitted = {str(value) for value in session_state.get(_ONCE_KEYS, [])}
    if once and once_key in emitted:
        return False
    if once:
        emitted.add(once_key)
        session_state[_ONCE_KEYS] = sorted(emitted)

    safe_duration = _bounded_non_negative(duration_ms, maximum=86_400_000)
    if safe_duration is None and event is AuthoringEvent.VALID_PROPOSAL:
        started = session_state.get(_JOURNEY_STARTED_KEY)
        if isinstance(started, int | float):
            safe_duration = _bounded_non_negative(
                int((time.monotonic() - float(started)) * 1_000),
                maximum=86_400_000,
            )
    safe_count = _bounded_non_negative(count, maximum=1_000_000)

    session_state[_JOURNEY_STAGE_KEY] = stage.value
    session_state[_JOURNEY_WORKFLOW_KEY] = workflow.value
    if event in {AuthoringEvent.REPORT_OPENED, AuthoringEvent.RUN_STARTED}:
        session_state[_JOURNEY_COMPLETE_KEY] = True

    logger.info(
        "Authoring funnel event: journey_id=%s workflow=%s event=%s stage=%s "
        "outcome=%s duration_ms=%s count=%s requires_data_run=%s",
        journey_id,
        workflow.value,
        event.value,
        stage.value,
        outcome.value,
        safe_duration,
        safe_count,
        requires_data_run,
    )
    return True


def abandon_and_reset(
    session_state: MutableMapping[str, Any],
    *,
    workflow: AuthoringWorkflow,
) -> None:
    """Close an unfinished journey when the user explicitly starts over."""

    if session_state.get(_JOURNEY_ID_KEY) and not session_state.get(_JOURNEY_COMPLETE_KEY):
        raw_stage = str(session_state.get(_JOURNEY_STAGE_KEY) or AuthoringStage.ENTRY.value)
        stage = (
            AuthoringStage(raw_stage)
            if raw_stage in {item.value for item in AuthoringStage}
            else AuthoringStage.ENTRY
        )
        record_event(
            session_state,
            event=AuthoringEvent.ABANDONED,
            workflow=workflow,
            stage=stage,
            outcome=AuthoringOutcome.DISCARDED,
        )
    _clear_journey_state(session_state)


def abandon_active_journey(session_state: MutableMapping[str, Any]) -> bool:
    """Record and clear the unfinished journey when in-app navigation leaves authoring."""

    if not session_state.get(_JOURNEY_ID_KEY) or session_state.get(_JOURNEY_COMPLETE_KEY):
        return False
    raw_workflow = str(session_state.get(_JOURNEY_WORKFLOW_KEY) or "")
    workflow = (
        AuthoringWorkflow(raw_workflow)
        if raw_workflow in {item.value for item in AuthoringWorkflow}
        else AuthoringWorkflow.BUILD
    )
    raw_stage = str(session_state.get(_JOURNEY_STAGE_KEY) or AuthoringStage.ENTRY.value)
    stage = (
        AuthoringStage(raw_stage)
        if raw_stage in {item.value for item in AuthoringStage}
        else AuthoringStage.ENTRY
    )
    record_event(
        session_state,
        event=AuthoringEvent.ABANDONED,
        workflow=workflow,
        stage=stage,
        outcome=AuthoringOutcome.DISCARDED,
    )
    _clear_journey_state(session_state)
    return True


def _clear_journey_state(session_state: MutableMapping[str, Any]) -> None:
    for key in (
        _JOURNEY_ID_KEY,
        _JOURNEY_STARTED_KEY,
        _JOURNEY_STAGE_KEY,
        _JOURNEY_WORKFLOW_KEY,
        _JOURNEY_COMPLETE_KEY,
        _ONCE_KEYS,
    ):
        session_state.pop(key, None)


def _bounded_non_negative(value: int | None, *, maximum: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(number, maximum))


__all__ = [
    "AuthoringEvent",
    "AuthoringOutcome",
    "AuthoringStage",
    "AuthoringWorkflow",
    "abandon_active_journey",
    "abandon_and_reset",
    "record_event",
    "start_journey",
    "workflow_from_handoff",
]
