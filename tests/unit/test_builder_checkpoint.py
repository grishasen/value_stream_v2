"""Configuration Builder workspace-checkpoint tests."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from valuestream.ui import builder, builder_checkpoint


def _safe_registry(*, workspace_name: str = "recovered_workspace") -> dict[str, object]:
    state: dict[str, object] = {
        "builder_settings_workspace": workspace_name,
        "builder_settings_calendar_grains": ["Day", "Summary"],
    }
    status = builder.builder_draft_status(
        "settings:workspace",
        {"workspace_name": "workspace"},
        {"workspace_name": workspace_name},
    )
    builder.update_builder_draft_registry(
        state,
        status,
        widget_prefixes=("builder_settings_",),
    )
    return state[builder.BUILDER_DRAFTS_KEY]  # type: ignore[return-value]


def _render_builder(workspace: str) -> None:
    from valuestream.ui.context import load_context  # noqa: PLC0415
    from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

    _builder_steps(load_context(workspace))


def _create_settings_checkpoint(workspace: Path, workspace_name: str) -> AppTest:
    builder.ensure_minimum_workspace(workspace)
    rendered = AppTest.from_function(
        _render_builder,
        kwargs={"workspace": str(workspace)},
    ).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Settings").run()
    name = next(item for item in rendered.text_input if item.label == "Workspace Name")
    return name.set_value(workspace_name).run()


@pytest.mark.unit
def test_checkpoint_round_trip_is_json_safe_and_redacts_private_state(tmp_path: Path) -> None:
    now = dt.datetime(2026, 7, 18, 12, 30, tzinfo=dt.UTC)
    registry = _safe_registry()
    chat_state: dict[str, object] = {
        "builder_chat_agent_prompt": "PRIVATE PROMPT VALUE",
        "builder_chat_dataset_descriptions": {"customer": "PRIVATE SAMPLE VALUE"},
        "builder_chat_provider_payload": {"response": "PRIVATE PROVIDER VALUE"},
        "builder_chat_sample_frame": pl.DataFrame({"CustomerID": ["PRIVATE-CUSTOMER-42"]}),
        "builder_chat_upload_bytes": b"private-upload",
        "builder_chat_api_key": "sk-private-token-123456",
        "builder_chat_password": "plain-password",
    }
    chat_status = builder.builder_draft_status(
        "chat:guidance",
        {"agent_prompt": "baseline"},
        {
            "agent_prompt": "PRIVATE PROMPT VALUE",
            "dataset_descriptions": {"customer": "PRIVATE SAMPLE VALUE"},
        },
    )
    builder.update_builder_draft_registry(
        chat_state,
        chat_status,
        widget_prefixes=("builder_chat_",),
    )
    registry.update(chat_state[builder.BUILDER_DRAFTS_KEY])  # type: ignore[arg-type]

    unsafe_state: dict[str, object] = {
        "builder_source_name": "safe-name",
        "builder_source_api_key": "sk-private-token-123456",
    }
    unsafe_status = builder.builder_draft_status(
        "source:unsafe",
        {"id": "unsafe", "description": "before"},
        {
            "id": "unsafe",
            "description": "after",
            "raw_provider_payload": {"response": "PRIVATE PROVIDER VALUE"},
        },
    )
    builder.update_builder_draft_registry(
        unsafe_state,
        unsafe_status,
        widget_prefixes=("builder_source_",),
    )
    registry.update(unsafe_state[builder.BUILDER_DRAFTS_KEY])  # type: ignore[arg-type]

    path = builder_checkpoint.write_builder_checkpoint(
        tmp_path,
        drafts=registry,
        current_step="Settings",
        base_catalog_hash="a" * 64,
        now=now,
    )

    assert path == tmp_path / "meta" / builder_checkpoint.CHECKPOINT_FILENAME
    assert path is not None
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "PRIVATE" not in text
    assert "sk-private" not in text
    assert "password" not in text
    assert "sample" not in text
    assert "provider" not in text
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))

    result = builder_checkpoint.load_builder_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=("Settings",),
        now=now + dt.timedelta(minutes=1),
    )

    assert result.status == "ready"
    assert result.checkpoint is not None
    assert result.checkpoint.saved_at == now
    assert result.checkpoint.base_catalog_hash == "a" * 64
    assert result.checkpoint.current_step == "Settings"
    restored = result.checkpoint.drafts["settings:workspace"]
    assert set(result.checkpoint.drafts) == {"settings:workspace"}
    assert restored["widget_state"]["builder_settings_workspace"] == "recovered_workspace"
    json.dumps(result.checkpoint.drafts, allow_nan=False)


@pytest.mark.unit
def test_checkpoint_changed_catalog_expiry_invalid_file_and_cleanup(tmp_path: Path) -> None:
    now = dt.datetime(2026, 7, 18, 12, 30, tzinfo=dt.UTC)
    registry = _safe_registry()
    path = builder_checkpoint.write_builder_checkpoint(
        tmp_path,
        drafts=registry,
        current_step="Settings",
        base_catalog_hash="a" * 64,
        now=now,
    )
    assert path is not None

    changed = builder_checkpoint.load_builder_checkpoint(
        tmp_path,
        current_catalog_hash="b" * 64,
        allowed_steps=("Settings",),
        now=now,
    )
    assert changed.status == "reconciliation"
    assert changed.checkpoint is not None
    assert path.is_file()

    expired = builder_checkpoint.load_builder_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=("Settings",),
        now=now + builder_checkpoint.CHECKPOINT_RETENTION + dt.timedelta(seconds=1),
    )
    assert expired.status == "expired"
    assert not path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 1, "drafts": "PRIVATE"}', encoding="utf-8")
    invalid = builder_checkpoint.load_builder_checkpoint(
        tmp_path,
        current_catalog_hash="a" * 64,
        allowed_steps=("Settings",),
        now=now,
    )
    assert invalid.status == "invalid"
    assert not path.exists()

    builder_checkpoint.write_builder_checkpoint(
        tmp_path,
        drafts=registry,
        current_step="Settings",
        base_catalog_hash="a" * 64,
        now=now,
    )
    assert path.is_file()
    assert (
        builder_checkpoint.write_builder_checkpoint(
            tmp_path,
            drafts={},
            current_step="Settings",
            base_catalog_hash="a" * 64,
            now=now,
        )
        is None
    )
    assert not path.exists()


@pytest.mark.unit
def test_builder_checkpoint_restores_after_fresh_app_session(tmp_path: Path) -> None:
    created = _create_settings_checkpoint(tmp_path, "recovered_workspace")
    assert not created.exception
    assert builder_checkpoint.checkpoint_path(tmp_path).is_file()

    refreshed = AppTest.from_function(
        _render_builder,
        kwargs={"workspace": str(tmp_path)},
    ).run()
    labels = [button.label for button in refreshed.button]
    assert labels.count("Restore checkpoint") == 1
    assert labels.count("Discard checkpoint") == 1
    assert not any(button.label == "Apply to workspace" for button in refreshed.button)

    refreshed = (
        next(button for button in refreshed.button if button.label == "Restore checkpoint")
        .click()
        .run()
    )
    assert refreshed.session_state["builder_step"] == "Settings"
    restore_draft = next(button for button in refreshed.button if button.label == "Restore draft")
    assert not restore_draft.disabled

    refreshed = restore_draft.click().run()
    workspace_name = next(item for item in refreshed.text_input if item.label == "Workspace Name")
    assert workspace_name.value == "recovered_workspace"
    assert any(button.label == "Apply to workspace" for button in refreshed.button)


@pytest.mark.unit
def test_builder_checkpoint_requires_reconciliation_after_catalog_change(tmp_path: Path) -> None:
    created = _create_settings_checkpoint(tmp_path, "recovered_workspace")
    assert not created.exception
    pipelines_path = tmp_path / "catalog" / "pipelines.yaml"
    pipelines = builder._read_yaml(pipelines_path)
    pipelines["workspace"] = "external_workspace_change"
    builder._write_yaml(pipelines_path, pipelines)

    refreshed = AppTest.from_function(
        _render_builder,
        kwargs={"workspace": str(tmp_path)},
    ).run()
    warnings = "\n".join(item.value for item in refreshed.warning)
    assert "Reconciliation required" in warnings

    refreshed = (
        next(button for button in refreshed.button if button.label == "Restore checkpoint")
        .click()
        .run()
    )
    warnings = "\n".join(item.value for item in refreshed.warning)
    assert "catalog changed since these drafts were checkpointed" in warnings
    restore_draft = next(button for button in refreshed.button if button.label == "Restore draft")
    assert restore_draft.disabled

    refreshed = (
        next(button for button in refreshed.button if button.label == "Discard draft").click().run()
    )
    assert not builder_checkpoint.checkpoint_path(tmp_path).exists()


@pytest.mark.unit
def test_clean_builder_navigation_round_trips_through_url_query(tmp_path: Path) -> None:
    builder.ensure_minimum_workspace(tmp_path)

    def app(workspace: str, initial_query_step: str) -> None:
        import streamlit as st  # noqa: PLC0415

        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages import config_builder as page  # noqa: PLC0415

        if not st.session_state.get("qa_builder_query_seeded"):
            st.query_params[page.BUILDER_STEP_QUERY_PARAM] = initial_query_step
            st.session_state["qa_builder_query_seeded"] = True
        page._builder_steps(load_context(workspace))
        st.caption(f"URL step: {st.query_params.get(page.BUILDER_STEP_QUERY_PARAM, 'missing')}")

    opened = AppTest.from_function(
        app,
        kwargs={"workspace": str(tmp_path), "initial_query_step": "settings"},
    ).run()

    assert not opened.exception
    assert opened.session_state["builder_step"] == "Settings"
    assert not builder_checkpoint.checkpoint_path(tmp_path).exists()

    jump = next(item for item in opened.selectbox if item.label == "Jump to step")
    navigated = jump.set_value("Reports / Tiles").run()
    assert not navigated.exception
    assert navigated.session_state["builder_step"] == "Reports / Tiles"
    assert any(item.value == "URL step: reports" for item in navigated.caption)

    reloaded = AppTest.from_function(
        app,
        kwargs={"workspace": str(tmp_path), "initial_query_step": "reports"},
    ).run()
    assert not reloaded.exception
    assert reloaded.session_state["builder_step"] == "Reports / Tiles"
    assert not builder_checkpoint.checkpoint_path(tmp_path).exists()

    obsolete_link = AppTest.from_function(
        app,
        kwargs={"workspace": str(tmp_path), "initial_query_step": "removed-step"},
    ).run()
    assert not obsolete_link.exception
    assert obsolete_link.session_state["builder_step"] == "Workspace Health"
    assert any(item.value == "URL step: health" for item in obsolete_link.caption)
