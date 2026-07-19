"""AI Configuration Studio workspace-checkpoint tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from valuestream.config.canonical import catalog_config_hash
from valuestream.config.loader import load
from valuestream.ui import ai_studio_checkpoint, builder
from valuestream.ui.pages import ai_config_studio as studio_page


def _catalog_draft(workspace_name: str = "workspace") -> dict[str, object]:
    return {
        "pipelines": {"version": 1, "workspace": workspace_name, "sources": []},
        "processors": {"processors": []},
        "metrics": {"metrics": {}},
        "dashboards": {"theme": {}, "dashboards": []},
    }


def _workspace_state(sample_relative: str, sample_identity: str) -> dict[str, object]:
    draft = _catalog_draft()
    return {
        "ai_studio_sample_workspace_relative": sample_relative,
        "ai_studio_sample_identity": sample_identity,
        "ai_studio_source_id": "events",
        "ai_studio_reader_kind": "csv",
        "ai_studio_reader_root": "data",
        "ai_studio_file_pattern": "sample.csv",
        "ai_studio_group_pattern": "",
        "ai_studio_streaming": False,
        "ai_studio_hive_partitioning": False,
        "ai_studio_timestamp_format": "",
        "ai_studio_subject": "CustomerID",
        "ai_studio_outcome_time": "OutcomeTime",
        "ai_studio_decision_time": "",
        "ai_studio_outcome_column": "Outcome",
        "ai_studio_day_column": "",
        "ai_studio_month_column": "",
        "ai_studio_year_column": "",
        "ai_studio_quarter_column": "",
        "ai_studio_rename_capitalize_enabled": False,
        "ai_studio_defaults": [{"Field": "Channel", "Default Value": "Unknown"}],
        "ai_studio_filter_mode": "Rules",
        "ai_studio_filter_rows": [
            {"Field": "Outcome", "Operator": "==", "Value": "Clicked", "Enabled": True}
        ],
        "ai_studio_raw_filter": "",
        "ai_studio_calculations": [
            {"Name": "DayCopy", "Mode": "AST YAML", "Expression": "{col: OutcomeTime}"}
        ],
        "ai_studio_approved_fields": ["CustomerID", "Outcome", "OutcomeTime"],
        "ai_studio_example_fields": ["Outcome"],
        "ai_studio_group_by_fields": ["Outcome"],
        "ai_studio_draft": draft,
        "ai_studio_reviewed_signature": studio_page._draft_signature(draft),
        "ai_studio_draft_source": "sample",
        "ai_studio_catalog_draft_step": "Metrics",
        "ai_studio_sample_bytes": b"PRIVATE-UPLOAD-BYTES",
        "ai_studio_schema_preview_table": pl.DataFrame({"CustomerID": ["PRIVATE-CUSTOMER-42"]}),
        "ai_studio_pending_draft": {"raw_provider_payload": "PRIVATE-PROVIDER"},
        "ai_studio_pending_prompt": "PRIVATE-PROMPT",
        "ai_studio_last_ai_response": "PRIVATE-RESPONSE",
        "ai_studio_copilot_history": [{"content": "PRIVATE-COPILOT"}],
        "ai_studio_api_key": "sk-private-token-123456",
        "ai_studio_ai_provider": "private-provider",
        "ai_studio_ai_sharing_consent_receipt": {"approved": True},
    }


def _prepare_workspace(workspace: Path) -> tuple[Path, str, str]:
    builder.ensure_minimum_workspace(workspace)
    sample = workspace / "data" / "sample.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        "CustomerID,OutcomeTime,Outcome,Channel\nC-1,2026-07-01T09:00:00Z,Clicked,Web\n",
        encoding="utf-8",
    )
    relative = "data/sample.csv"
    identity = studio_page._workspace_sample_identity(sample, relative)
    return sample, relative, identity


def _render_deterministic_studio(workspace: str) -> None:
    from valuestream.ui.context import load_context  # noqa: PLC0415
    from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

    page._render_studio(
        load_context(workspace),
        ai_calls_enabled=False,
        title="Checkpoint test",
        subtitle="Checkpoint test",
        status_label="Checkpoint test",
        include_header=False,
    )


@pytest.mark.unit
def test_ai_studio_checkpoint_safe_roundtrip_omits_private_runtime_state(
    tmp_path: Path,
) -> None:
    _sample, relative, identity = _prepare_workspace(tmp_path)
    now = dt.datetime(2026, 7, 18, 12, 30, tzinfo=dt.UTC)
    session = _workspace_state(relative, identity)
    draft_with_prompt = dict(session["ai_studio_draft"])  # type: ignore[arg-type]
    draft_with_prompt["chat_with_data"] = {"agent_prompt": "PRIVATE-DRAFT-PROMPT"}
    session["ai_studio_draft"] = draft_with_prompt

    path = ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=session,
        current_step="14. Apply",
        base_catalog_hash="a" * 64,
        now=now,
    )

    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "PRIVATE" not in text
    for forbidden in (
        "sample_bytes",
        "schema_preview_table",
        "pending_draft",
        "pending_prompt",
        "last_ai_response",
        "copilot_history",
        "api_key",
        "ai_provider",
        "consent_receipt",
        "chat_with_data",
    ):
        assert forbidden not in text

    result = ai_studio_checkpoint.load_ai_studio_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=studio_page.DETERMINISTIC_STEPS,
        now=now + dt.timedelta(minutes=1),
    )
    assert result.status == "ready"
    assert result.checkpoint is not None
    assert not result.checkpoint.requires_sample_reselect
    assert result.checkpoint.state["ai_studio_subject"] == "CustomerID"
    assert result.checkpoint.state["ai_studio_approved_fields"] == [
        "CustomerID",
        "Outcome",
        "OutcomeTime",
    ]
    assert set(result.checkpoint.state["ai_studio_draft"]) == {
        "pipelines",
        "processors",
        "metrics",
        "dashboards",
    }


@pytest.mark.unit
def test_uploaded_sample_checkpoint_keeps_only_safe_catalog_metadata(tmp_path: Path) -> None:
    draft = _catalog_draft()
    session = {
        **_workspace_state("", "f" * 64),
        "ai_studio_sample_origin": "upload",
        "ai_studio_sample_workspace_relative": "",
        "ai_studio_sample_bytes": b"PRIVATE-UPLOAD-BYTES",
        "ai_studio_draft": draft,
        "ai_studio_reviewed_signature": studio_page._draft_signature(draft),
    }

    path = ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=session,
        current_step="14. Apply",
        base_catalog_hash="a" * 64,
    )
    result = ai_studio_checkpoint.load_ai_studio_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=studio_page.DETERMINISTIC_STEPS,
    )

    assert path is not None
    assert result.status == "ready"
    assert result.checkpoint is not None
    assert result.checkpoint.requires_sample_reselect
    assert set(result.checkpoint.state).issubset(
        {
            "ai_studio_draft",
            "ai_studio_reviewed_signature",
            "ai_studio_draft_source",
            "ai_studio_catalog_draft_step",
        }
    )
    assert "PRIVATE" not in path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_ai_studio_checkpoint_drift_expiry_invalid_and_cleanup(tmp_path: Path) -> None:
    _sample, relative, identity = _prepare_workspace(tmp_path)
    now = dt.datetime(2026, 7, 18, 12, 30, tzinfo=dt.UTC)
    path = ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=_workspace_state(relative, identity),
        current_step="14. Apply",
        base_catalog_hash="a" * 64,
        now=now,
    )
    assert path is not None
    drift = ai_studio_checkpoint.load_ai_studio_checkpoint(
        tmp_path,
        current_catalog_hash="b" * 64,
        allowed_steps=studio_page.DETERMINISTIC_STEPS,
        now=now,
    )
    assert drift.status == "reconciliation"
    expired = ai_studio_checkpoint.load_ai_studio_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=studio_page.DETERMINISTIC_STEPS,
        now=now + ai_studio_checkpoint.CHECKPOINT_RETENTION + dt.timedelta(seconds=1),
    )
    assert expired.status == "expired"
    assert not path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 1, "state": "PRIVATE"}', encoding="utf-8")
    invalid = ai_studio_checkpoint.load_ai_studio_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=studio_page.DETERMINISTIC_STEPS,
        now=now,
    )
    assert invalid.status == "invalid"
    assert not path.exists()

    ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=_workspace_state(relative, identity),
        current_step="14. Apply",
        base_catalog_hash="a" * 64,
        now=now,
    )
    ai_studio_checkpoint.discard_ai_studio_checkpoint(tmp_path)
    assert not path.exists()


@pytest.mark.unit
def test_ai_studio_checkpoint_restores_after_restart_and_revalidates(tmp_path: Path) -> None:
    _sample, relative, identity = _prepare_workspace(tmp_path)
    state = _workspace_state(relative, identity)
    state.pop("ai_studio_rename_capitalize_enabled")
    state["ai_studio_rename_capitalize"] = True
    ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=state,
        current_step="14. Apply",
        base_catalog_hash=catalog_config_hash(load(tmp_path)),
    )

    refreshed = AppTest.from_function(
        _render_deterministic_studio,
        kwargs={"workspace": str(tmp_path)},
    ).run()
    assert any(button.label == "Restore Studio checkpoint" for button in refreshed.button)
    refreshed = (
        next(button for button in refreshed.button if button.label == "Restore Studio checkpoint")
        .click()
        .run()
    )

    assert not refreshed.exception
    assert refreshed.session_state["ai_studio_subject"] == "CustomerID"
    assert refreshed.session_state["ai_studio_defaults"] == [
        {"Field": "Channel", "Default Value": "Unknown"}
    ]
    assert refreshed.session_state["ai_studio_rename_capitalize_enabled"] is True
    assert "ai_studio_rename_capitalize" not in refreshed.session_state
    restored_draft = refreshed.session_state["ai_studio_draft"]
    restored_signature = studio_page._draft_signature(restored_draft)
    assert refreshed.session_state["ai_studio_reviewed_signature"] == ""
    validation_cache = refreshed.session_state["ai_studio_validation_cache"]
    matching_entries = [
        entry for key, entry in validation_cache.items() if key.startswith(f"{restored_signature}:")
    ]
    assert matching_entries
    assert any(
        "active field-contract source 'events' does not exist" in issue.lower()
        for entry in matching_entries
        for issue in entry["issues"]
    )
    assert "ai_studio_sample_bytes" not in refreshed.session_state
    assert studio_page.AI_SHARING_RECEIPT_STATE_KEY not in refreshed.session_state


@pytest.mark.unit
def test_ai_studio_checkpoint_drift_and_upload_recovery_are_explicit(tmp_path: Path) -> None:
    _sample, relative, identity = _prepare_workspace(tmp_path)
    state = _workspace_state(relative, identity)
    old_hash = catalog_config_hash(load(tmp_path))
    ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=state,
        current_step="14. Apply",
        base_catalog_hash=old_hash,
    )
    pipelines_path = tmp_path / "catalog" / "pipelines.yaml"
    pipelines = builder._read_yaml(pipelines_path)
    pipelines["workspace"] = "external_change"
    builder._write_yaml(pipelines_path, pipelines)

    drifted = AppTest.from_function(
        _render_deterministic_studio,
        kwargs={"workspace": str(tmp_path)},
    ).run()
    assert "Reconciliation required" in "\n".join(item.value for item in drifted.warning)
    drifted = (
        next(button for button in drifted.button if button.label == "Restore Studio checkpoint")
        .click()
        .run()
    )
    assert drifted.session_state["ai_studio_reviewed_signature"] == ""
    assert "revalidated and must be reviewed again" in "\n".join(
        item.value for item in drifted.warning
    )

    ai_studio_checkpoint.discard_ai_studio_checkpoint(tmp_path)
    upload_state = {
        **_workspace_state("", "f" * 64),
        "ai_studio_sample_workspace_relative": "",
        "ai_studio_sample_bytes": b"PRIVATE-UPLOAD-BYTES",
    }
    ai_studio_checkpoint.write_ai_studio_checkpoint(
        tmp_path,
        session_state=upload_state,
        current_step="14. Apply",
        base_catalog_hash=catalog_config_hash(load(tmp_path)),
    )
    upload = AppTest.from_function(
        _render_deterministic_studio,
        kwargs={"workspace": str(tmp_path)},
    ).run()
    warnings = "\n".join(item.value for item in upload.warning)
    assert "original upload was not retained" in warnings
    upload = (
        next(button for button in upload.button if button.label == "Restore Studio checkpoint")
        .click()
        .run()
    )
    assert upload.get("file_uploader")
    assert "ai_studio_sample_bytes" not in upload.session_state
    assert studio_page.AI_STUDIO_CHECKPOINT_STAGED_KEY in upload.session_state

    discarded = AppTest.from_function(
        _render_deterministic_studio,
        kwargs={"workspace": str(tmp_path)},
    ).run()
    discarded = (
        next(button for button in discarded.button if button.label == "Discard Studio checkpoint")
        .click()
        .run()
    )
    assert not ai_studio_checkpoint.checkpoint_path(tmp_path).exists()
