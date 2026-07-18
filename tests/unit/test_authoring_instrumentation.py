"""Privacy and lifecycle tests for configuration-authoring events."""

from __future__ import annotations

import logging

import pytest

from valuestream.ui import feature_flags, instrumentation


@pytest.mark.unit
def test_authoring_feature_flag_defaults_on_and_accepts_explicit_off() -> None:
    assert feature_flags.authoring_v2_enabled({}) is True
    assert feature_flags.authoring_v2_enabled({"VALUESTREAM_AUTHORING_V2": "off"}) is False
    assert feature_flags.authoring_v2_enabled({"VALUESTREAM_AUTHORING_V2": "yes"}) is True


@pytest.mark.unit
def test_authoring_events_emit_only_bounded_metadata(caplog: pytest.LogCaptureFixture) -> None:
    state: dict[str, object] = {}
    caplog.set_level(logging.INFO)

    instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
    )
    instrumentation.record_event(
        state,
        event=instrumentation.AuthoringEvent.VALID_PROPOSAL,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
        stage=instrumentation.AuthoringStage.DRAFT,
        outcome=instrumentation.AuthoringOutcome.SUCCESS,
        duration_ms=12_345,
        count=7,
        requires_data_run=True,
    )

    assert "workflow=ai_studio" in caplog.text
    assert "event=valid_proposal" in caplog.text
    assert "duration_ms=12345" in caplog.text
    assert "count=7" in caplog.text
    for forbidden in (
        "CustomerID",
        "alice@example.com",
        "/Users/",
        "sk-private",
        "prompt",
    ):
        assert forbidden not in caplog.text


@pytest.mark.unit
def test_authoring_event_contract_rejects_arbitrary_labels() -> None:
    state: dict[str, object] = {}

    with pytest.raises(ValueError, match=r"sample=alice@example\.com"):
        instrumentation.record_event(
            state,
            event="sample=alice@example.com",  # type: ignore[arg-type]
            workflow=instrumentation.AuthoringWorkflow.BUILDER,
            stage=instrumentation.AuthoringStage.SAMPLE,
            outcome=instrumentation.AuthoringOutcome.SUCCESS,
        )


@pytest.mark.unit
def test_handoff_origin_is_strictly_allowlisted() -> None:
    assert (
        instrumentation.workflow_from_handoff("builder")
        is instrumentation.AuthoringWorkflow.BUILDER
    )
    assert (
        instrumentation.workflow_from_handoff("ai_studio")
        is instrumentation.AuthoringWorkflow.AI_STUDIO
    )
    assert instrumentation.workflow_from_handoff("/private/sample.csv") is None
    assert instrumentation.workflow_from_handoff(None) is None


@pytest.mark.unit
def test_once_events_are_deduplicated() -> None:
    state: dict[str, object] = {}
    kwargs = {
        "event": instrumentation.AuthoringEvent.CONSENT_CONFIRMED,
        "workflow": instrumentation.AuthoringWorkflow.AI_STUDIO,
        "stage": instrumentation.AuthoringStage.CONSENT,
        "outcome": instrumentation.AuthoringOutcome.SUCCESS,
        "once": True,
    }

    assert instrumentation.record_event(state, **kwargs) is True
    assert instrumentation.record_event(state, **kwargs) is False


@pytest.mark.unit
def test_explicit_restart_records_abandonment_without_session_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state: dict[str, object] = {"private_sample": "PRIVATE-ROW-VALUE"}
    caplog.set_level(logging.INFO)
    instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.BUILDER,
    )

    instrumentation.abandon_and_reset(
        state,
        workflow=instrumentation.AuthoringWorkflow.BUILDER,
    )

    assert "event=abandoned" in caplog.text
    assert "PRIVATE-ROW-VALUE" not in caplog.text
    assert "vs_authoring_journey_id" not in state


@pytest.mark.unit
def test_completed_journey_starts_fresh_on_the_next_authoring_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state: dict[str, object] = {}
    caplog.set_level(logging.INFO)
    first = instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.BUILDER,
    )
    instrumentation.record_event(
        state,
        event=instrumentation.AuthoringEvent.REPORT_OPENED,
        workflow=instrumentation.AuthoringWorkflow.BUILDER,
        stage=instrumentation.AuthoringStage.REPORT,
        outcome=instrumentation.AuthoringOutcome.SUCCESS,
    )

    second = instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.BUILDER,
    )

    assert second != first
    assert caplog.text.count("event=entered") == 2


@pytest.mark.unit
def test_in_app_navigation_records_last_stage_abandonment(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state: dict[str, object] = {}
    caplog.set_level(logging.INFO)
    instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
    )
    instrumentation.record_event(
        state,
        event=instrumentation.AuthoringEvent.DRAFT_REQUESTED,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
        stage=instrumentation.AuthoringStage.DRAFT,
        outcome=instrumentation.AuthoringOutcome.STARTED,
    )

    assert instrumentation.abandon_active_journey(state)
    assert "event=abandoned stage=draft" in caplog.text
    assert "vs_authoring_journey_id" not in state


@pytest.mark.unit
def test_build_choice_transitions_into_studio_without_splitting_the_journey(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state: dict[str, object] = {}
    caplog.set_level(logging.INFO)
    journey = instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.BUILD,
    )
    instrumentation.record_event(
        state,
        event=instrumentation.AuthoringEvent.ENTERED,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
        stage=instrumentation.AuthoringStage.ENTRY,
        outcome=instrumentation.AuthoringOutcome.STARTED,
        once=True,
    )
    instrumentation.record_event(
        state,
        event=instrumentation.AuthoringEvent.SAMPLE_CHOSEN,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
        stage=instrumentation.AuthoringStage.SAMPLE,
        outcome=instrumentation.AuthoringOutcome.STARTED,
    )

    continued = instrumentation.start_journey(
        state,
        workflow=instrumentation.AuthoringWorkflow.AI_STUDIO,
    )

    assert continued == journey
    assert "event=abandoned" not in caplog.text
    studio_entered = caplog.text.index("workflow=ai_studio event=entered")
    sample_chosen = caplog.text.index("workflow=ai_studio event=sample_chosen")
    assert studio_entered < sample_chosen
