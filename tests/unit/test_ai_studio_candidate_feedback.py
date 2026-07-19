"""UI feedback contracts for bounded AI Studio candidate generation."""

from __future__ import annotations

import json

import pytest

from valuestream.ai.studio import DraftAttemptDiagnostic, DraftCandidateResult
from valuestream.ui.pages import ai_config_studio as studio_page

RAW_MODEL_RESPONSE = "PRIVATE-CUSTOMER-42 from /Users/alice/private-response.yaml"


def _diagnostic(
    *,
    attempt: int = 1,
    role: str = "generation",
    stage: str = "validation",
    issues: tuple[str, ...] = ("metrics.Total.source references missing processor 'orders'.",),
    issue_count: int = 1,
    issue_areas: tuple[tuple[str, int], ...] = (("metrics", 1),),
    sections: tuple[str, ...] = ("metrics",),
    error_type: str = "",
    line: int | None = None,
    column: int | None = None,
) -> DraftAttemptDiagnostic:
    return DraftAttemptDiagnostic(
        attempt=attempt,
        role=role,
        stage=stage,
        issues=issues,
        issue_count=issue_count,
        issue_areas=issue_areas,
        sections=sections,
        response_chars=len(RAW_MODEL_RESPONSE),
        error_type=error_type,
        line=line,
        column=column,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stage", "diagnostic", "expected"),
    [
        (
            "parse",
            _diagnostic(
                stage="parse",
                issues=("The model response was not valid catalog YAML.",),
                issue_areas=(),
                sections=(),
                error_type="ParserError",
                line=7,
                column=3,
            ),
            "final response could not be read as catalog YAML",
        ),
        (
            "validation",
            _diagnostic(issue_count=4, issue_areas=(("metrics", 3), ("dashboards", 1))),
            "final YAML still failed 4 catalog contract check(s)",
        ),
    ],
)
def test_candidate_failure_message_is_stage_specific_and_excludes_raw_response(
    stage: str,
    diagnostic: DraftAttemptDiagnostic,
    expected: str,
) -> None:
    result = DraftCandidateResult(
        draft=None,
        issues=diagnostic.issues,
        attempts=3,
        last_response=RAW_MODEL_RESPONSE,
        failure_stage=stage,
        attempt_diagnostics=(diagnostic,),
        reference="candidate-safe-reference",
    )

    message = studio_page._candidate_failure_message(result, operation_label="AI draft")

    assert expected in message
    assert "after 3 attempts" in message
    assert "current accepted revision was not changed" in message
    assert "provider settings" not in message
    assert RAW_MODEL_RESPONSE not in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("diagnostic", "expected_parts"),
    [
        (
            _diagnostic(
                stage="validated",
                issues=(),
                issue_count=0,
                issue_areas=(),
                sections=("processors", "metrics"),
            ),
            ("Attempt 1 · initial generation", "accepted", "processors, metrics"),
        ),
        (
            _diagnostic(
                attempt=2,
                role="repair_1",
                stage="parse",
                issues=("The model response was not valid catalog YAML.",),
                issue_areas=(),
                sections=(),
                error_type="ParserError",
                line=7,
                column=3,
            ),
            ("Attempt 2 · repair 1", "rejected", "line 7, column 3", "ParserError"),
        ),
        (
            _diagnostic(
                attempt=3,
                role="repair_2",
                issue_count=2,
                issue_areas=(("metrics", 1), ("dashboards", 1)),
            ),
            ("Attempt 3 · repair 2", "rejected", "2 catalog contract issue(s)"),
        ),
    ],
)
def test_candidate_attempt_status_line_is_terminal_and_safe(
    diagnostic: DraftAttemptDiagnostic,
    expected_parts: tuple[str, ...],
) -> None:
    line = studio_page._candidate_attempt_status_line(diagnostic)

    assert all(part in line for part in expected_parts)
    assert "waiting for provider response" not in line
    assert RAW_MODEL_RESPONSE not in line


@pytest.mark.unit
def test_candidate_attempt_rows_explain_each_attempt_without_raw_response() -> None:
    diagnostics = (
        _diagnostic(
            stage="parse",
            issues=("The model response was not valid catalog YAML.",),
            issue_areas=(),
            sections=(),
            error_type="ParserError",
        ),
        _diagnostic(
            attempt=2,
            role="repair_1",
            issue_count=3,
            issue_areas=(("processors", 2), ("metrics", 1)),
            sections=("processors", "metrics"),
        ),
        _diagnostic(
            attempt=3,
            role="repair_2",
            stage="validated",
            issues=(),
            issue_count=0,
            issue_areas=(),
            sections=("processors", "metrics", "dashboards"),
        ),
    )
    result = DraftCandidateResult(
        draft={"metrics": {"metrics": {}}},
        issues=(),
        attempts=3,
        last_response=RAW_MODEL_RESPONSE,
        attempt_diagnostics=diagnostics,
    )

    rows = studio_page._candidate_attempt_rows(result)

    assert [row["Attempt"] for row in rows] == [1, 2, 3]
    assert [row["Outcome"] for row in rows] == [
        "YAML could not be read",
        "Catalog checks failed",
        "Accepted",
    ]
    assert rows[1]["Issues"] == 3
    assert rows[2]["Sections Found"] == "processors, metrics, dashboards"
    assert RAW_MODEL_RESPONSE not in json.dumps(rows)
    assert "waiting for provider response" not in json.dumps(rows)


@pytest.mark.unit
def test_queue_pending_candidate_records_bounded_failure_receipt_without_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {
        "ai_studio_ai_model": "openai/gpt-test",
        "ai_studio_ai_provider": "openai",
        "ai_studio_approved_fields": ["Channel"],
        "ai_studio_example_fields": [],
        "ai_studio_sample_identity": "sample-one",
        "ai_studio_sample_name": "sample.parquet",
        "ai_studio_pending_draft": None,
    }
    monkeypatch.setattr(studio_page.st, "session_state", state)
    diagnostic = _diagnostic()
    result = DraftCandidateResult(
        draft=None,
        issues=diagnostic.issues,
        attempts=3,
        last_response=RAW_MODEL_RESPONSE,
        failure_stage="validation",
        attempt_diagnostics=(diagnostic,),
        reference="candidate-safe-reference",
    )

    queued = studio_page._queue_pending_candidate(
        result,
        base_draft={"revision": "accepted"},
        kind="draft",
        prompt="PRIVATE PROMPT",
    )

    assert queued is False
    assert state["ai_studio_pending_draft"] is None
    receipt = state["ai_studio_last_candidate_failure"]
    assert isinstance(receipt, dict)
    assert receipt["kind"] == "draft"
    assert receipt["stage"] == "validation"
    assert receipt["attempts"] == 3
    assert receipt["reference"] == "candidate-safe-reference"
    assert receipt["scope_signature"] == studio_page._ai_sharing_signature()
    assert receipt["attempt_diagnostics"][0]["response_chars"] == len(RAW_MODEL_RESPONSE)
    serialized = json.dumps(receipt)
    assert RAW_MODEL_RESPONSE not in serialized
    assert "PRIVATE PROMPT" not in serialized


@pytest.mark.unit
def test_queue_pending_candidate_success_clears_failure_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_failure = {
        "kind": "draft",
        "stage": "validation",
        "attempts": 3,
    }
    state: dict[str, object] = {
        "ai_studio_last_candidate_failure": previous_failure,
        "ai_studio_pending_draft": None,
    }
    monkeypatch.setattr(studio_page.st, "session_state", state)
    accepted_base = {"revision": "accepted"}
    candidate = {"revision": "candidate"}
    result = DraftCandidateResult(
        draft=candidate,
        issues=(),
        attempts=1,
        last_response="valid catalog response",
        attempt_diagnostics=(
            _diagnostic(stage="validated", issues=(), issue_count=0, issue_areas=()),
        ),
    )

    queued = studio_page._queue_pending_candidate(
        result,
        base_draft=accepted_base,
        kind="draft",
        prompt="safe prompt",
    )

    assert queued is True
    assert "ai_studio_last_candidate_failure" not in state
    assert state["ai_studio_pending_draft"] == candidate
    assert state["ai_studio_pending_base_draft"] == accepted_base
    assert state["ai_studio_pending_base_draft"] is not accepted_base
    assert state["ai_studio_pending_kind"] == "draft"
    assert state["ai_studio_pending_prompt"] == "safe prompt"


@pytest.mark.unit
def test_retry_label_uses_only_failure_receipt_from_current_sharing_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {
        "ai_studio_ai_model": "openai/gpt-test",
        "ai_studio_ai_provider": "openai",
        "ai_studio_approved_fields": ["Channel"],
        "ai_studio_example_fields": [],
        "ai_studio_sample_identity": "sample-one",
        "ai_studio_sample_name": "sample.parquet",
    }
    monkeypatch.setattr(studio_page.st, "session_state", state)
    state["ai_studio_last_candidate_failure"] = {
        "kind": "catalog_draft",
        "scope_signature": studio_page._ai_sharing_signature(),
    }

    assert studio_page._ai_retry_label("draft", "Generate AI Draft") == "Retry AI Draft"
    assert (
        studio_page._ai_retry_label("reports", "Refresh Reports From Metrics")
        == "Refresh Reports From Metrics"
    )

    state["ai_studio_ai_model"] = "openai/gpt-other"

    assert studio_page._ai_retry_label("draft", "Generate AI Draft") == "Generate AI Draft"


@pytest.mark.unit
def test_prior_draft_uses_neutral_preserved_wording_after_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    monkeypatch.setattr(studio_page, "_build_draft_catalog", lambda *_args: {"baseline": True})
    monkeypatch.setattr(studio_page, "_builder_source_handoff", lambda: False)
    monkeypatch.setattr(studio_page, "_render_user_goals_editor", lambda: "")
    monkeypatch.setattr(studio_page, "_schema_preview_for_ai", lambda *_args: [])
    monkeypatch.setattr(studio_page, "prompt_for_config_draft", lambda **_kwargs: "safe prompt")
    monkeypatch.setattr(studio_page, "_ai_privacy_summary", lambda *_args: None)
    monkeypatch.setattr(studio_page, "_draft_counts", lambda *_args: None)
    monkeypatch.setattr(studio_page, "_draft_files", lambda *_args: {})

    def app() -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_draft"] = {"revision": "accepted"}
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_approved_fields"] = ["Channel"]
        st.session_state["ai_studio_last_candidate_failure"] = {
            "kind": "draft",
            "stage": "validation",
            "attempts": 3,
            "scope_signature": page._ai_sharing_signature(),
        }
        page._ai_draft(
            SimpleNamespace(catalog_hash="workspace-hash"),
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
            None,
            ai_calls_enabled=True,
        )

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    assert any(
        "current accepted draft was preserved" in info.value.casefold() for info in rendered.info
    )
    assert all("Draft accepted for review" not in success.value for success in rendered.success)
