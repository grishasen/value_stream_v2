"""Phase 5 Builder helper tests."""

from __future__ import annotations

import ast
import copy
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import plotly.io as pio  # type: ignore[import-untyped]
import polars as pl
import pytest
import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

from valuestream.charts.recipes import RECIPES
from valuestream.config import model
from valuestream.config.loader import load
from valuestream.expr import parser as expr_parser
from valuestream.query import executor
from valuestream.states import kll, topk
from valuestream.ui import builder, dimension_profile, forms, theme
from valuestream.ui.pages import config_builder


@pytest.mark.unit
def test_editor_save_bar_supports_one_out_of_order_top_action() -> None:
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import components

with st.container(horizontal=True, horizontal_alignment="right"):
    save_slot = st.empty()
    save_slot.caption("")

@st.fragment
def editor(slot):
    components.editor_save_bar(
        key="test_editor_ready",
        caption="Complete the editor below.",
        disabled=True,
        placeholder=slot,
    )
    components.editor_save_bar(
        key="test_editor",
        caption="Save the current editor values.",
        placeholder=slot,
    )

editor(save_slot)
st.write("Editor body")
"""
    ).run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Save"]
    assert not app.button[0].disabled


@pytest.mark.unit
def test_builder_draft_status_uses_canonical_object_equality() -> None:
    baseline = {"id": "engagement", "states": {"Count": {"type": "count"}}}
    equivalent = {
        "states": {"Count": {"type": "count", "description": None}},
        "id": "engagement",
    }
    changed = {"id": "engagement", "states": {"Count": {"type": "value_sum"}}}

    clean = builder.builder_draft_status("processor:engagement", baseline, equivalent)
    dirty = builder.builder_draft_status("processor:engagement", baseline, changed)

    assert not clean.dirty
    assert dirty.dirty
    assert dirty.revision == dirty.draft_hash[:12]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "supported"),
    [
        ("binary_outcome", True),
        ("score_distribution", True),
        ("numeric_distribution", False),
        ("entity_lifecycle", False),
        ("entity_set", False),
        ("funnel", False),
        ("snapshot", False),
    ],
)
def test_processor_dedup_control_matches_runtime_support(kind: str, supported: bool) -> None:
    assert builder.processor_supports_dedup(kind) is supported


@pytest.mark.unit
def test_builder_create_template_stays_clean_until_edited_and_discard_resets_it() -> None:
    state: dict[str, object] = {"builder_processor_mode": "Create New Processor"}
    template = {"id": "ih_processor", "description": ""}

    clean = builder.builder_template_draft_status(
        state,
        "processor:new",
        "builder_proc_template_baseline_ih_processor",
        template,
    )
    assert not clean.dirty

    state["builder_proc_desc_ih_processor"] = "QA processor"
    dirty = builder.builder_template_draft_status(
        state,
        "processor:new",
        "builder_proc_template_baseline_ih_processor",
        {**template, "description": "QA processor"},
    )
    builder.update_builder_draft_registry(
        state,
        dirty,
        widget_prefixes=("builder_proc_",),
    )
    assert dirty.dirty
    assert list(state[builder.BUILDER_DRAFTS_KEY]) == ["processor:new"]

    builder.discard_builder_draft(
        state,
        dirty.key,
        widget_prefixes=("builder_proc_",),
        preserve_widget_keys=("builder_processor_mode",),
    )
    assert state[builder.BUILDER_DRAFTS_KEY] == {}
    assert state["builder_processor_mode"] == "Create New Processor"
    assert "builder_proc_desc_ih_processor" not in state
    assert "builder_proc_template_baseline_ih_processor" not in state

    reset = builder.builder_template_draft_status(
        state,
        "processor:new",
        "builder_proc_template_baseline_ih_processor",
        template,
    )
    assert not reset.dirty


@pytest.mark.unit
def test_discarded_create_draft_continues_on_the_first_click() -> None:
    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui import builder  # noqa: PLC0415
        from valuestream.ui.pages import config_builder as page  # noqa: PLC0415

        st.session_state.setdefault("builder_step", "Processors")
        st.session_state.setdefault("builder_processor_mode", "Create New Processor")
        save_slot = st.empty()
        draft_slot = st.empty()
        description = st.text_input(
            "Description",
            value="",
            key="builder_proc_desc_template",
        )
        status = builder.builder_template_draft_status(
            st.session_state,
            "processor:new",
            "builder_proc_template_baseline",
            {"id": "template", "description": description},
        )
        page._render_editor_primary_action(
            save_slot=save_slot,
            draft_slot=draft_slot,
            status=status,
            valid=True,
            widget_prefixes=("builder_proc_",),
            preserve_widget_keys=("builder_processor_mode",),
            help_text="Apply test processor",
        )

    rendered = AppTest.from_function(app).run()
    rendered = rendered.text_input[0].set_value("QA processor").run()
    assert any(button.label == "Apply to workspace" for button in rendered.button)

    rendered = (
        next(button for button in rendered.button if button.label == "Discard draft").click().run()
    )
    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)

    rendered = (
        next(button for button in rendered.button if button.label == "Continue").click().run()
    )
    assert not rendered.exception
    assert rendered.session_state["builder_step"] == "Metrics"


@pytest.mark.unit
def test_post_apply_cleanup_removes_editor_state_and_preserves_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {
        "builder_processor_mode": "Create New Processor",
        "builder_proc_desc_template": "QA processor",
        "builder_proc_template_baseline": {"description": ""},
    }
    status = builder.builder_draft_status(
        "processor:new",
        {"description": ""},
        {"description": "QA processor"},
    )
    builder.update_builder_draft_registry(
        state,
        status,
        widget_prefixes=("builder_proc_",),
    )
    monkeypatch.setattr(config_builder, "st", SimpleNamespace(session_state=state))

    config_builder._queue_builder_post_apply_cleanup(
        status=status,
        widget_prefixes=("builder_proc_",),
        preserve_widget_keys=("builder_processor_mode",),
        state_updates={"builder_processor_mode": "Edit Existing Processor"},
    )
    config_builder._consume_builder_post_apply_cleanup()

    assert state[builder.BUILDER_DRAFTS_KEY] == {}
    assert state["builder_processor_mode"] == "Edit Existing Processor"
    assert "builder_proc_desc_template" not in state
    assert "builder_proc_template_baseline" not in state
    assert config_builder.BUILDER_POST_APPLY_CLEANUP_KEY not in state


@pytest.mark.unit
def test_builder_draft_registry_preserves_and_discards_step_state() -> None:
    state: dict[str, object] = {
        "builder_source_id_ih": "events",
        "builder_metric_name": "keep",
    }
    status = builder.builder_draft_status("source:ih", {"id": "ih"}, {"id": "events"})

    assert not builder.update_builder_draft_registry(
        state,
        status,
        widget_prefixes=("builder_source_",),
    )
    assert builder.update_builder_draft_registry(
        state,
        status,
        widget_prefixes=("builder_source_",),
    )
    assert state[builder.BUILDER_DRAFTS_KEY] == {
        "source:ih": {
            "revision": status.revision,
            "baseline_hash": status.baseline_hash,
            "draft_hash": status.draft_hash,
            "draft_payload": {"id": "events"},
            "widget_state": {"builder_source_id_ih": "events"},
        }
    }

    state.pop("builder_source_id_ih")
    assert builder.restore_builder_draft(state, status.key)
    assert state["builder_source_id_ih"] == "events"
    assert builder.registered_builder_draft(state, status.key)["draft_payload"] == {"id": "events"}

    builder.discard_builder_draft(
        state,
        status.key,
        widget_prefixes=("builder_source_",),
    )

    assert state[builder.BUILDER_DRAFTS_KEY] == {}
    assert "builder_source_id_ih" not in state
    assert state["builder_metric_name"] == "keep"


@pytest.mark.unit
def test_builder_step_draft_can_be_restored_after_navigation() -> None:
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import builder
from valuestream.ui.pages.config_builder import _render_editor_primary_action

def move(step):
    st.session_state["draft_test_step"] = step

save_slot = st.empty()
draft_slot = st.empty()
if st.session_state.get("draft_test_step", "edit") == "edit":
    description = st.text_input(
        "Description",
        value="Persisted description",
        key="builder_source_description_demo",
    )
    status = builder.builder_draft_status(
        "source:demo",
        {"description": "Persisted description"},
        {"description": description},
    )
    _render_editor_primary_action(
        save_slot=save_slot,
        draft_slot=draft_slot,
        status=status,
        valid=True,
        widget_prefixes=("builder_source_",),
        help_text="Apply test source",
    )
    st.button("Leave editor", on_click=move, args=("away",))
else:
    st.write("Another step")
    st.button("Return to editor", on_click=move, args=("edit",))
"""
    ).run()

    assert not app.exception
    assert [button.label for button in app.button].count("Continue") == 1
    assert [button.label for button in app.button].count("Apply to workspace") == 0

    app = app.text_input[0].set_value("Session proposal").run()
    assert not app.exception
    assert [button.label for button in app.button].count("Continue") == 0
    assert [button.label for button in app.button].count("Apply to workspace") == 1

    app = next(button for button in app.button if button.label == "Leave editor").click().run()
    app = next(button for button in app.button if button.label == "Return to editor").click().run()

    assert not app.exception
    assert [button.label for button in app.button].count("Restore draft") == 1
    app = next(button for button in app.button if button.label == "Restore draft").click().run()

    assert not app.exception
    assert app.text_input[0].value == "Session proposal"
    assert [button.label for button in app.button].count("Apply to workspace") == 1


@pytest.mark.unit
def test_builder_apply_outcome_separates_report_ready_from_data_refresh() -> None:
    report = builder.builder_apply_outcome("Executive summary")
    processor = builder.builder_apply_outcome(
        "Engagement",
        source_ids=["ih", "ih"],
        requires_data_run=True,
    )

    assert not report.requires_data_run
    assert report.action == "open_report"
    assert processor.requires_data_run
    assert processor.action == "run_data"
    assert processor.source_ids == ("ih",)


@pytest.mark.unit
def test_builder_materialization_impact_ignores_prose_and_presentation() -> None:
    source = {
        "id": "ih",
        "description": "Before",
        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
    }
    processor = {
        "id": "engagement",
        "source": "ih",
        "description": "Before",
        "states": {"Count": {"type": "count"}},
    }
    settings = {
        "workspace_name": "Demo",
        "time_zone": "UTC",
        "calendar_grains": ["Day", "Summary"],
        "week_start": "monday",
        "dashboard_theme": {"accent": "pine"},
    }

    assert not builder.builder_requires_data_run(
        "source", source, {**source, "description": "After"}
    )
    assert not builder.builder_requires_data_run(
        "processor", processor, {**processor, "description": "After"}
    )
    assert not builder.builder_requires_data_run(
        "workspace_settings",
        settings,
        {**settings, "workspace_name": "Renamed", "dashboard_theme": {"accent": "gold"}},
    )
    assert not builder.builder_requires_data_run(
        "metric", {"description": "A"}, {"kind": "formula"}
    )
    # The common-dimension list itself is authoring metadata; without
    # extended processors nothing recomputes.
    assert not builder.builder_requires_data_run(
        "dimensions",
        {"dimensions": ["Channel"], "processors": {}},
        {"dimensions": ["Channel", "Issue"], "processors": {}},
    )


@pytest.mark.unit
def test_builder_materialization_impact_detects_computation_changes() -> None:
    source = {
        "id": "ih",
        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
    }
    processor = {
        "id": "engagement",
        "source": "ih",
        "states": {"Count": {"type": "count"}},
    }
    settings = {
        "time_zone": "UTC",
        "calendar_grains": ["Day", "Summary"],
        "week_start": "monday",
    }

    assert builder.builder_requires_data_run(
        "source",
        source,
        {**source, "reader": {"kind": "parquet", "file_pattern": "new/*.parquet"}},
    )
    assert builder.builder_requires_data_run(
        "processor",
        processor,
        {**processor, "states": {"Count": {"type": "value_sum", "column": "Value"}}},
    )
    assert builder.builder_requires_data_run(
        "workspace_settings",
        settings,
        {**settings, "time_zone": "Europe/Berlin"},
    )
    assert builder.builder_requires_data_run(
        "dimensions",
        {"dimensions": ["Channel"], "processors": {}},
        {
            "dimensions": ["Channel", "Issue"],
            "processors": {"engagement": {**processor, "dimensions": ["Channel", "Issue"]}},
        },
    )


@pytest.mark.unit
def test_direct_builder_render_starts_builder_authoring_journey() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    render_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render"
    )
    start_call = next(
        node
        for node in ast.walk(render_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "start_journey"
    )

    workflow = next(keyword.value for keyword in start_call.keywords if keyword.arg == "workflow")

    assert isinstance(workflow, ast.Attribute)
    assert workflow.attr == "BUILDER"


@pytest.mark.unit
@pytest.mark.parametrize("requires_data_run", [True, False])
def test_builder_apply_event_records_materialization_requirement(
    monkeypatch: pytest.MonkeyPatch,
    requires_data_run: bool,
) -> None:
    captured: dict[str, object] = {}
    session_state: dict[str, object] = {}

    def capture(state: object, **kwargs: object) -> bool:
        captured.update({"state": state, **kwargs})
        return True

    monkeypatch.setattr(config_builder, "record_event", capture)
    monkeypatch.setattr(config_builder, "st", SimpleNamespace(session_state=session_state))

    config_builder._record_builder_applied(
        builder.builder_apply_outcome(
            "Test object",
            requires_data_run=requires_data_run,
        )
    )

    assert captured["state"] is session_state
    assert captured["event"].value == "applied"  # type: ignore[union-attr]
    assert captured["workflow"].value == "builder"  # type: ignore[union-attr]
    assert captured["stage"].value == "apply"  # type: ignore[union-attr]
    assert captured["outcome"].value == "success"  # type: ignore[union-attr]
    assert captured["requires_data_run"] is requires_data_run


@pytest.mark.unit
def test_builder_ready_review_and_apply_events_are_ordered_and_private(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui import builder  # noqa: PLC0415
        from valuestream.ui.pages import config_builder as page  # noqa: PLC0415

        status = builder.builder_draft_status(
            "source:private",
            {"description": "baseline"},
            {"description": "PRIVATE-CUSTOMER-42"},
        )
        if page._render_editor_primary_action(
            save_slot=st.empty(),
            draft_slot=st.empty(),
            status=status,
            valid=True,
            widget_prefixes=("builder_private_",),
            help_text="Apply the test proposal.",
        ):
            page._record_builder_applied(builder.builder_apply_outcome("Private object"))

    rendered = AppTest.from_function(app).run()
    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run()
    )

    assert not rendered.exception
    valid = caplog.text.index("workflow=builder event=valid_proposal")
    reviewed = caplog.text.index("workflow=builder event=reviewed")
    applied = caplog.text.index("workflow=builder event=applied")
    assert valid < reviewed < applied
    assert "PRIVATE-CUSTOMER-42" not in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("private timeout details"), "timeout"),
        (ValueError("private validation details"), "blocked"),
        (RuntimeError("PRIVATE-CUSTOMER-42"), "error"),
    ],
)
def test_builder_apply_failure_uses_only_allowlisted_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: str,
) -> None:
    captured: dict[str, object] = {}

    def capture(_state: object, **kwargs: object) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(config_builder, "record_event", capture)
    monkeypatch.setattr(config_builder, "st", SimpleNamespace(session_state={}))

    config_builder._record_builder_apply_failed(error)

    assert captured["event"].value == "failed"  # type: ignore[union-attr]
    assert captured["workflow"].value == "builder"  # type: ignore[union-attr]
    assert captured["stage"].value == "apply"  # type: ignore[union-attr]
    assert captured["outcome"].value == expected  # type: ignore[union-attr]
    assert all("PRIVATE" not in str(value) for value in captured.values())


@pytest.mark.unit
def test_builder_preserves_unresolved_data_refresh_across_later_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state: dict[str, object] = {}
    monkeypatch.setattr(config_builder, "st", SimpleNamespace(session_state=session_state))

    config_builder._store_builder_apply_outcome(
        builder.builder_apply_outcome(
            "Engagement processor",
            source_ids=["ih"],
            requires_data_run=True,
        )
    )
    config_builder._store_builder_apply_outcome(builder.builder_apply_outcome("Executive report"))

    pending = session_state[config_builder.BUILDER_LAST_OUTCOME_KEY]
    assert pending["action"] == "run_data"
    assert pending["source_ids"] == ("ih",)

    config_builder._store_builder_apply_outcome(
        builder.builder_apply_outcome(
            "Customer processor",
            source_ids=["customers"],
            requires_data_run=True,
        )
    )

    merged = session_state[config_builder.BUILDER_LAST_OUTCOME_KEY]
    assert merged["action"] == "run_data"
    assert merged["source_ids"] == ("customers", "ih")


@pytest.mark.unit
def test_source_contract_apply_notice_surfaces_data_refresh_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "id": "ih",
        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
    }
    changed_source = {
        **source,
        "transforms": [
            {
                "kind": "derive_column",
                "column": "HighValue",
                "expression": {"op": "gt", "left": {"col": "Value"}, "right": 100},
            }
        ],
    }
    warnings: list[str] = []
    links: list[tuple[str, str]] = []
    session_state: dict[str, object] = {"builder_apply_notice": "Source applied."}
    fake_streamlit = SimpleNamespace(
        session_state=session_state,
        success=lambda *_args, **_kwargs: None,
        warning=lambda message, **_kwargs: warnings.append(str(message)),
        link_button=lambda label, url, **_kwargs: links.append((str(label), str(url))),
    )
    monkeypatch.setattr(config_builder, "st", fake_streamlit)

    requires_data_run = builder.builder_requires_data_run("source", source, changed_source)
    config_builder._store_builder_apply_outcome(
        builder.builder_apply_outcome(
            "Interaction history",
            source_ids=["ih"],
            requires_data_run=requires_data_run,
        )
    )
    config_builder._render_apply_notice()

    assert requires_data_run
    assert warnings == [
        "**Data refresh required · Interaction history**\n\n"
        "The workspace configuration is valid, but its aggregate computation contract "
        "changed. Run ih from Data Load to publish matching aggregates."
    ]
    assert links == [("Run data", "/data_load?from=builder")]


@pytest.mark.unit
def test_builder_navigation_supports_jump_back_and_continue(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()

    assert not rendered.exception
    labels = [button.label for button in rendered.button]
    assert labels.count("Back") == 1
    assert labels.count("Continue") == 1
    assert labels.count("Apply to workspace") == 0
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Sources").run()

    assert not rendered.exception
    source_labels = [button.label for button in rendered.button]
    assert source_labels.count("Continue") == 1
    assert source_labels.count("Apply to workspace") == 0

    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Export current workspace").run()

    assert not rendered.exception
    assert rendered.session_state["builder_step"] == "Export current workspace"
    assert {item.label for item in rendered.get("download_button")} == {
        "Download pipelines.yaml",
        "Download processors.yaml",
        "Download metrics.yaml",
        "Download dashboards.yaml",
    }


@pytest.mark.unit
def test_new_processor_template_is_clean_and_apply_reopens_clean_object(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Processors").run()
    rendered = (
        next(button for button in rendered.button if button.label == "Create New Processor")
        .click()
        .run()
    )

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)

    description = next(item for item in rendered.text_input if item.label == "Description")
    rendered = description.set_value("QA processor").run()
    assert any(button.label == "Apply to workspace" for button in rendered.button)

    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert rendered.session_state[builder.BUILDER_DRAFTS_KEY] == {}
    assert rendered.session_state["builder_processor_mode"] == "Edit Existing Processor"
    assert (
        rendered.session_state[config_builder.BUILDER_LAST_OUTCOME_KEY]["label"] == "QA processor"
    )
    created = next(
        processor
        for processor in load(tmp_path).processors.processors
        if processor.id == "ih_processor"
    )
    assert builder.processor_to_dict(created)["entities"] == {"subject": "InteractionID"}


@pytest.mark.unit
def test_builder_edit_selectors_lead_with_stable_id_and_keep_human_context(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Sources").run()
    source = next(item for item in rendered.selectbox if item.label == "Source")
    assert source.options == ["ih — Ih · Parquet"]

    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Processors").run()
    processor = next(item for item in rendered.selectbox if item.label == "Processor")
    processor_source = next(item for item in rendered.selectbox if item.label == "Source")
    assert processor.options == ["engagement — Engagement · Binary outcome"]
    assert processor_source.options == ["ih — Ih · Parquet"]

    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Dimensions").run()
    profile_source = next(item for item in rendered.selectbox if item.label == "Profile Source")
    assert profile_source.options == ["ih — Ih · Parquet"]


@pytest.mark.unit
def test_sources_offer_one_click_deterministic_studio_handoff(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Sources").run()

    assert not rendered.exception
    add_source = next(item for item in rendered.get("link_button") if item.label == "Add source")
    assert add_source.url == config_builder.BUILDER_ADD_SOURCE_URL
    assert "mode=deterministic" in add_source.url
    assert "intent=add_source" in add_source.url
    assert "return_to=configuration_builder" in add_source.url


@pytest.mark.unit
def test_processor_delete_dialog_uses_human_labels_and_cancel_is_read_only(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    paths = [tmp_path / "catalog" / name for name in builder.CATALOG_FILENAMES]
    before = {path: path.read_bytes() for path in paths}

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Processors").run(timeout=15)
    rendered = (
        next(button for button in rendered.button if button.label == "Delete processor")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert any("Engagement — engagement" in str(item.value) for item in rendered.warning)
    assert any("overview/portfolio/ctr" in str(item.value) for item in rendered.code)
    assert next(
        button for button in rendered.button if button.label == "Delete processor and dependencies"
    ).disabled
    rendered = next(button for button in rendered.button if button.label == "Cancel").click().run()

    assert not rendered.exception
    assert {path: path.read_bytes() for path in paths} == before
    assert {processor.id for processor in load(tmp_path).processors.processors} == {
        "engagement",
        "holdings_lifecycle",
    }


@pytest.mark.unit
def test_metric_delete_dialog_requires_explicit_tile_choice_and_cancel_is_read_only(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    paths = [tmp_path / "catalog" / name for name in builder.CATALOG_FILENAMES]
    before = {path: path.read_bytes() for path in paths}

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Metrics").run(timeout=15)
    rendered = (
        next(button for button in rendered.button if button.label == "Edit Existing Metric")
        .click()
        .run(timeout=15)
    )
    rendered = (
        next(button for button in rendered.button if button.label == "Delete metric")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    warnings = [str(item.value) for item in rendered.warning]
    assert any("CTR · Formula / state passthrough · Engagement" in value for value in warnings), (
        warnings
    )
    assert any("overview/portfolio/ctr" in str(item.value) for item in rendered.code)
    delete_action = next(
        button for button in rendered.button if button.label == "Delete metric and tiles"
    )
    assert delete_action.disabled
    assert any("dependent report tile" in item.label for item in rendered.checkbox)
    rendered = next(button for button in rendered.button if button.label == "Cancel").click().run()

    assert not rendered.exception
    assert {path: path.read_bytes() for path in paths} == before
    assert "CTR" in load(tmp_path).metrics.metrics


@pytest.mark.unit
def test_new_processor_empty_entity_fallback_is_explicit(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    pipelines_path = tmp_path / "catalog" / "pipelines.yaml"
    pipelines = yaml.safe_load(pipelines_path.read_text(encoding="utf-8"))
    pipelines["sources"][0]["schema"]["natural_key"] = []
    pipelines_path.write_text(yaml.safe_dump(pipelines, sort_keys=False), encoding="utf-8")
    processors_path = tmp_path / "catalog" / "processors.yaml"
    processors = yaml.safe_load(processors_path.read_text(encoding="utf-8"))
    # An unsupported reference in another processor is not source-schema evidence.
    processors["processors"][0]["entities"] = {"subject": "SubjectID"}
    processors_path.write_text(yaml.safe_dump(processors, sort_keys=False), encoding="utf-8")

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Processors").run()
    rendered = (
        next(button for button in rendered.button if button.label == "Create New Processor")
        .click()
        .run()
    )

    assert not rendered.exception
    warning_text = "\n".join(item.value for item in rendered.warning)
    assert "No subject/entity field could be inferred" in warning_text
    assert "intentionally empty" in warning_text
    subject = next(item for item in rendered.selectbox if item.label == "Subject Entity Field")
    # No selection renders the placeholder (value None); callers receive "".
    assert not subject.value


@pytest.mark.unit
def test_builder_metric_mode_switch_can_fill_claimed_fragment_slots(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    builder.write_metric_definition(
        tmp_path,
        "CTR",
        {
            **builder.build_formula_metric("engagement", "Positives", "Count"),
            "display": {"label": "Click-through rate"},
        },
    )

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Metrics").run()
    rendered = (
        next(button for button in rendered.button if button.label == "Edit Existing Metric")
        .click()
        .run()
    )

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)


@pytest.mark.unit
def test_metric_from_scratch_template_and_post_apply_editor_are_clean(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Metrics").run()
    create_from = next(item for item in rendered.segmented_control if item.label == "Create from")
    rendered = create_from.set_value("From Scratch").run()

    processor = next(item for item in rendered.selectbox if item.label == "Processor")
    rendered = processor.set_value("engagement").run()
    metric_kind = next(item for item in rendered.selectbox if item.label == "Metric Kind")
    rendered = metric_kind.set_value("formula").run()

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)

    metric_name = next(item for item in rendered.text_input if item.label == "Metric Display Name")
    rendered = metric_name.set_value("QA rate").run()
    metric_id = next(item for item in rendered.text_input if item.label == "Metric ID")
    assert metric_id.value == ""
    assert any(button.label == "Apply to workspace" for button in rendered.button)
    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)
    assert rendered.session_state["builder_metric_mode"] == "Edit Existing Metric"
    metrics = load(tmp_path).metrics.metrics
    assert len(metrics) == 2
    assert "engagement_metric_qa_rate" in metrics
    assert metrics["engagement_metric_qa_rate"].display.label == "QA rate"


@pytest.mark.unit
def test_export_downloads_render_before_collapsed_yaml_previews() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_export_current_workspace"
    )
    downloads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "download_button"
    ]
    expanders = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expander"
    ]

    assert downloads
    assert expanders
    assert max(node.lineno for node in downloads) < min(node.lineno for node in expanders)
    assert all(
        any(
            keyword.arg == "expanded"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
        for node in expanders
    )


@pytest.mark.unit
def test_empty_editor_frame_has_columns_without_phantom_row() -> None:
    frame = builder.editor_frame(
        [],
        ["Field", "Value", "Enabled"],
        builder.blank_filter_row,
    )

    assert frame.is_empty()
    assert frame.columns == ["Field", "Value", "Enabled"]
    assert frame.schema == {
        "Field": pl.String,
        "Value": pl.String,
        "Enabled": pl.Boolean,
    }
    assert builder.default_rows_from_values({}) == []
    assert builder.filter_rows_from_expression(None) == []


@pytest.mark.unit
def test_binary_outcome_editor_uses_compact_logical_columns() -> None:
    app = AppTest.from_string(
        """
from valuestream.ui import forms

processor = {
    "entities": {"subject": "SubjectID"},
    "outcome": {
        "column": "Outcome",
        "positive_values": ["Clicked"],
        "negative_values": ["Impression"],
    },
    "variant_column": "Variant",
}
forms.processor_kind_fields(
    processor,
    "binary_outcome",
    field_options=["SubjectID", "Outcome", "Variant"],
    key_prefix="compact",
)
"""
    ).run()

    assert not app.exception
    assert [item.label for item in app.selectbox] == [
        "Subject Entity Field",
        "Variant Column",
        "Outcome Column",
    ]
    assert [item.label for item in app.text_input] == ["Positive Values", "Negative Values"]
    assert len(app.get("column")) == 5


@pytest.mark.unit
def test_build_formula_metric_uses_safe_div_when_denominator_selected() -> None:
    metric = builder.build_formula_metric("engagement", "Positives", "Count")

    assert metric == {
        "source": "engagement",
        "kind": "formula",
        "expression": {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {"col": "Count"},
        },
    }


@pytest.mark.unit
def test_metric_kind_options_prioritize_score_distribution_curve_metrics() -> None:
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "ih_ml",
            "source": "ih",
            "kind": "score_distribution",
            "score_properties": ["final_propensity"],
            "entities": {"subject": "SubjectID"},
        }
    )

    options = builder.metric_kind_options(processor)

    assert options[:2] == ["curve_from_digests", "calibration_from_digests"]
    assert "formula" in options
    assert "approx_distinct_count" in options
    assert builder.default_curve_digest_states(processor) == (
        "final_propensity_tdigest_positives",
        "final_propensity_tdigest_negatives",
    )
    assert builder.default_curve_digest_states(processor, final=True) == (
        "final_propensity_tdigest_positives",
        "final_propensity_tdigest_negatives",
    )


@pytest.mark.unit
def test_score_distribution_digest_pairs_use_source_column_metadata() -> None:
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "ih_ml",
            "source": "ih",
            "kind": "score_distribution",
            "score_properties": ["propensity", "final_propensity"],
            "states": {
                "Count": {"type": "count"},
                "primary_clicked": {
                    "type": "tdigest",
                    "source_column": "propensity",
                    "outcome": "positive",
                },
                "primary_impressed": {
                    "type": "tdigest",
                    "source_column": "propensity",
                    "outcome": "negative",
                },
                "calibrated_clicked": {
                    "type": "tdigest",
                    "source_column": "final_propensity",
                    "outcome": "positive",
                },
                "calibrated_impressed": {
                    "type": "tdigest",
                    "source_column": "final_propensity",
                    "outcome": "negative",
                },
            },
        }
    )

    assert builder.digest_state_pair_options(processor) == [
        ("propensity", "primary_clicked", "primary_impressed"),
        ("final_propensity", "calibrated_clicked", "calibrated_impressed"),
    ]
    assert builder.default_curve_digest_states(processor) == (
        "primary_clicked",
        "primary_impressed",
    )
    assert builder.default_curve_digest_states(processor, final=True) == (
        "primary_clicked",
        "primary_impressed",
    )
    assert "calibration_from_digests" in builder.metric_kind_options(processor)


@pytest.mark.unit
def test_score_distribution_without_outcome_digest_pair_does_not_offer_curve_metric() -> None:
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "ih_ml",
            "source": "ih",
            "kind": "score_distribution",
            "states": {
                "Count": {"type": "count"},
                "final_propensity_tdigest": {
                    "type": "tdigest",
                    "source_column": "final_propensity",
                },
            },
        }
    )

    options = builder.metric_kind_options(processor)

    assert "curve_from_digests" not in options
    assert "calibration_from_digests" not in options
    assert "tdigest_quantile" in options


@pytest.mark.unit
def test_metric_kind_options_cover_processor_specific_metric_shapes() -> None:
    numeric = model.NumericDistributionProcessor.model_validate(
        {
            "id": "descriptive",
            "source": "ih",
            "kind": "numeric_distribution",
            "properties": ["final_propensity"],
        }
    )
    binary = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "variant_column": "ModelControlGroup",
            "entities": {"subject": "CustomerID"},
        }
    )
    entity_set = model.EntitySetProcessor.model_validate(
        {
            "id": "unique_users",
            "source": "ih",
            "kind": "entity_set",
            "states": {
                "Visitors_theta": {"type": "theta", "source_column": "CustomerID"},
                "Clickers_theta": {"type": "theta", "source_column": "CustomerID"},
                "Visitors_hll": {"type": "hll", "source_column": "CustomerID"},
            },
        }
    )
    funnel = model.FunnelProcessor.model_validate(
        {
            "id": "action_funnel",
            "source": "ih",
            "kind": "funnel",
            "stages": [
                {"name": "Impression", "when": {"op": "eq", "column": "Outcome", "value": 0}},
                {"name": "Clicked", "when": {"op": "eq", "column": "Outcome", "value": 1}},
            ],
        }
    )
    lifecycle = model.EntityLifecycleProcessor.model_validate(
        {"id": "customer_lifecycle", "source": "orders", "kind": "entity_lifecycle"}
    )

    assert builder.metric_kind_options(numeric)[0] == "tdigest_quantile"
    assert {"variant_compare", "contingency_test"} <= set(builder.metric_kind_options(binary))
    assert "proportion_test" in builder.metric_kind_options(binary)
    assert {"set_op", "approx_distinct_count"} <= set(builder.metric_kind_options(entity_set))
    assert builder.metric_kind_options(funnel)[0] == "funnel_dropoff"
    assert builder.metric_kind_options(lifecycle)[0] == "lifecycle_summary"


@pytest.mark.unit
def test_theta_only_processor_can_build_approx_distinct_metric() -> None:
    processor = model.EntitySetProcessor.model_validate(
        {
            "id": "unique_users",
            "source": "ih",
            "kind": "entity_set",
            "states": {"Visitors_theta": {"type": "theta", "source_column": "CustomerID"}},
        }
    )

    numeric = model.NumericDistributionProcessor.model_validate(
        {
            "id": "descriptive",
            "source": "ih",
            "kind": "numeric_distribution",
            "states": {"Visitors_theta": {"type": "theta", "source_column": "CustomerID"}},
        }
    )

    assert "approx_distinct_count" in builder.metric_kind_options(processor)
    assert "approx_distinct_count" in builder.metric_kind_options(numeric)


@pytest.mark.unit
def test_config_builder_warns_when_funnel_has_no_stages() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "demo",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {"id": "action_funnel", "source": "ih", "kind": "funnel"},
                ]
            },
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
        }
    )

    warnings = config_builder._funnel_stage_warnings(catalog)

    assert warnings == [
        "`action_funnel` is a funnel processor but has no stages. "
        "Add at least one `stages` entry with a name and Boolean `when` expression."
    ]


@pytest.mark.unit
def test_config_builder_warns_when_funnel_stage_lacks_when_expression() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "demo",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "action_funnel",
                        "source": "ih",
                        "kind": "funnel",
                        "stages": [
                            {
                                "name": "Impression",
                                "when": {"op": "eq", "column": "Outcome", "value": "Impression"},
                            },
                            {"name": "Clicked"},
                        ],
                    },
                ]
            },
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
        }
    )

    warnings = config_builder._funnel_stage_warnings(catalog)

    assert warnings == [
        "`action_funnel` has funnel stage(s) without a `when` expression: "
        "Clicked. Add a Boolean `when` expression to each stage."
    ]


@pytest.mark.unit
def test_merge_stage_definitions_preserves_when_expressions() -> None:
    existing = [
        {"name": "Impression", "when": {"op": "eq", "column": "Outcome", "value": "Impression"}},
        {"name": "Clicked", "when": {"op": "eq", "column": "Outcome", "value": "Clicked"}},
    ]

    merged = builder.merge_stage_definitions(existing, ["Impression", "Clicked", "Conversion"])

    assert merged == [
        {"name": "Impression", "when": {"op": "eq", "column": "Outcome", "value": "Impression"}},
        {"name": "Clicked", "when": {"op": "eq", "column": "Outcome", "value": "Clicked"}},
        {"name": "Conversion"},
    ]
    assert builder.stage_names_missing_when(merged) == ["Conversion"]


@pytest.mark.unit
def test_merge_stage_definitions_drops_removed_stages_and_dedupes() -> None:
    existing = [
        {"name": "Impression", "when": {"op": "eq", "column": "Outcome", "value": "Impression"}},
        {"name": "Clicked", "when": {"op": "eq", "column": "Outcome", "value": "Clicked"}},
    ]

    merged = builder.merge_stage_definitions(existing, ["Clicked", "Clicked"])

    assert merged == [
        {"name": "Clicked", "when": {"op": "eq", "column": "Outcome", "value": "Clicked"}},
    ]


@pytest.mark.unit
def test_stage_condition_rows_maps_and_joined_expressions() -> None:
    clicked = {"op": "eq", "column": "Outcome", "value": "Clicked"}

    assert builder.stage_condition_rows(None) == ([], "all", True)
    assert builder.stage_condition_rows(clicked) == (
        [{"Field": "Outcome", "Operator": "==", "Value": "Clicked", "Enabled": True}],
        "all",
        True,
    )

    both = {"op": "and", "args": [clicked, {"op": "gt", "column": "Revenue", "value": 5}]}
    rows, combine, representable = builder.stage_condition_rows(both)
    assert (combine, representable) == ("all", True)
    assert builder.compile_condition_rows(rows, combine=combine) == both


@pytest.mark.unit
def test_stage_condition_rows_maps_or_joined_expressions() -> None:
    either = {
        "op": "or",
        "args": [
            {"op": "eq", "column": "Outcome", "value": "Clicked"},
            {"op": "eq", "column": "Outcome", "value": "Accepted"},
        ],
    }

    rows, combine, representable = builder.stage_condition_rows(either)

    assert (combine, representable) == ("any", True)
    assert builder.compile_condition_rows(rows, combine=combine) == either


@pytest.mark.unit
def test_stage_condition_rows_flags_unrepresentable_expressions() -> None:
    nested = {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [
                    {"op": "eq", "column": "Outcome", "value": "Clicked"},
                    {"op": "eq", "column": "Channel", "value": "Web"},
                ],
            },
            {"op": "eq", "column": "Outcome", "value": "Accepted"},
        ],
    }

    assert builder.stage_condition_rows(nested) == ([], "all", False)
    assert builder.stage_condition_rows({"op": "case"}) == ([], "all", False)
    assert builder.stage_condition_rows("Outcome == 'Clicked'") == ([], "all", False)


@pytest.mark.unit
def test_formula_simplicity_detection_guards_compound_expressions() -> None:
    assert forms.is_simple_formula(None)
    assert forms.is_simple_formula({"col": "Count"})
    assert forms.is_simple_formula(
        {"op": "safe_div", "num": {"col": "Positives"}, "den": {"col": "Count"}}
    )
    assert not forms.is_simple_formula(
        {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {"op": "add", "args": [{"col": "Positives"}, {"col": "Negatives"}]},
        }
    )
    assert not forms.is_simple_formula({"op": "mul", "args": [{"col": "Count"}, {"lit": 2}]})


@pytest.mark.unit
def test_score_properties_for_editor_prefers_existing_score_like_fields() -> None:
    assert forms._score_properties_for_editor(
        {},
        ["Channel", "model_score", "calibrated_score"],
    ) == ["model_score", "calibrated_score"]


@pytest.mark.unit
def test_metric_kind_options_offer_topk_items_for_topk_states() -> None:
    processor = model.EntitySetProcessor.model_validate(
        {
            "id": "campaign_exploration",
            "source": "ih",
            "kind": "entity_set",
            "states": {
                "TopCampaign_topk": {"type": "topk", "source_column": "Campaign"},
                "Visitors_hll": {"type": "hll", "source_column": "CustomerID"},
            },
        }
    )

    options = builder.metric_kind_options(processor)

    assert "topk_items" in options
    assert builder.default_metric_name(processor, "topk_items") == "campaign_exploration_topk"


@pytest.mark.unit
def test_build_specialized_metric_definitions_are_concise_yaml_shapes() -> None:
    assert builder.build_curve_from_digests_metric(
        "ih_ml",
        "Propensity_tdigest_positives",
        "Propensity_tdigest_negatives",
        "average_precision",
    ) == {
        "source": "ih_ml",
        "kind": "curve_from_digests",
        "positive_state": "Propensity_tdigest_positives",
        "negative_state": "Propensity_tdigest_negatives",
        "output": "average_precision",
    }
    assert builder.build_approx_distinct_metric("ih_ml", "UniqueSubjects_hll") == {
        "source": "ih_ml",
        "kind": "approx_distinct_count",
        "state": "UniqueSubjects_hll",
    }
    assert builder.build_topk_items_metric("explore", "TopCampaign_topk", limit=5) == {
        "source": "explore",
        "kind": "topk_items",
        "state": "TopCampaign_topk",
        "limit": 5,
    }
    assert builder.build_topk_items_metric(
        "explore",
        "TopCampaign_topk",
        limit=5,
        error_type="NO_FALSE_NEGATIVES",
    ) == {
        "source": "explore",
        "kind": "topk_items",
        "state": "TopCampaign_topk",
        "limit": 5,
        "error_type": "NO_FALSE_NEGATIVES",
    }
    assert builder.build_funnel_dropoff_metric("funnel", "Impression", "Clicked") == {
        "source": "funnel",
        "kind": "funnel_dropoff",
        "from_stage": "Impression",
        "to_stage": "Clicked",
        "output": "rate",
    }
    assert builder.build_proportion_test_metric("engagement", outputs=["z_score"]) == {
        "source": "engagement",
        "kind": "proportion_test",
        "variant_column": "ModelControlGroup",
        "test_role": "Test",
        "control_role": "Control",
        "outputs": ["z_score"],
    }


@pytest.mark.unit
def test_metric_output_columns_return_multi_column_defaults() -> None:
    metric = model.VariantCompareMetric.model_validate(
        {
            "source": "engagement",
            "kind": "variant_compare",
            "variant_column": "ModelControlGroup",
            "test_role": "Test",
            "control_role": "Control",
        }
    )

    outputs = builder.metric_output_columns("Lift", metric)

    assert {"Lift", "Lift_Z_Score", "Lift_P_Val"} <= set(outputs)


@pytest.mark.unit
def test_chart_choices_filter_by_metric_processor_kind(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    choices = builder.chart_choices_for_metric(catalog, "CTR")

    assert "line" in choices
    assert "gauge" in choices
    assert "rfm_density" not in choices


@pytest.mark.unit
def test_default_tile_fields_use_processor_group_by_columns(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    fields = builder.default_tile_fields(catalog, "CTR", "line")

    assert fields == {"x": "Day", "y": "CTR", "color": "Channel"}


@pytest.mark.unit
def test_descriptive_tile_defaults_use_metric_property_and_score() -> None:
    catalog = _numeric_distribution_catalog()

    fields = builder.default_tile_fields(catalog, "ResponseP50", "descriptive_line")

    assert fields["x"] == "Month"
    assert fields["property"] == "ResponseTime"
    assert fields["score"] == "p50"
    assert fields["color"] == "Channel"


@pytest.mark.unit
def test_descriptive_property_and_score_options_are_property_specific() -> None:
    catalog = _numeric_distribution_catalog()

    assert builder.descriptive_property_options(catalog, "ResponseP50") == [
        "Propensity",
        "ResponseTime",
    ]
    assert builder.descriptive_score_options(catalog, "ResponseP50", "Propensity") == [
        "Count",
        "Sum",
        "Mean",
        "Var",
        "Min",
        "Max",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
    ]


@pytest.mark.unit
def test_chart_field_options_include_scalar_states_as_metric_values(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    options = builder.chart_field_options(catalog, "CTR")

    assert {"Count", "Positives", "Negatives", "CTR"} <= set(options)
    assert options.index("Count") < options.index("CTR")


@pytest.mark.unit
def test_scatter_defaults_to_count_marker_size_when_available(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    fields = builder.default_tile_fields(catalog, "CTR", "scatter")

    assert fields == {"x": "CTR", "y": "CTR", "color": "Channel", "size": "Count"}


@pytest.mark.unit
def test_marketing_chart_defaults_cover_supported_plotly_shapes(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    assert builder.default_tile_fields(catalog, "CTR", "kpi_card") == {"value": "CTR"}
    assert builder.default_tile_fields(catalog, "CTR", "gauge") == {
        "value": "CTR",
        "facet_row": "Channel",
    }
    assert builder.default_tile_fields(catalog, "CTR", "pareto") == {
        "x": "Channel",
        "y": "CTR",
    }
    assert builder.default_tile_fields(catalog, "CTR", "stacked_area") == {
        "x": "Day",
        "y": "CTR",
        "color": "Channel",
    }
    assert builder.default_tile_fields(catalog, "CTR", "donut") == {
        "names": "Channel",
        "values": "CTR",
    }
    assert builder.default_tile_fields(catalog, "CTR", "calendar_heatmap") == {
        "date": "Day",
        "value": "CTR",
    }
    assert builder.default_tile_fields(catalog, "CTR", "bar_polar") == {
        "r": "CTR",
        "theta": "Channel",
        "color": "Channel",
    }


@pytest.mark.unit
def test_tile_field_controls_show_polar_radius_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_selectbox(
        label: str,
        options: list[str],
        *,
        index: int = 0,
        key: str,
        **_: object,
    ) -> str:
        calls.append({"label": label, "options": options, "index": index, "key": key})
        return options[index]

    monkeypatch.setattr(st, "selectbox", fake_selectbox)
    monkeypatch.setattr(config_builder, "_chart_setting_controls", lambda *_, **__: {})

    fields = config_builder._tile_field_controls(
        "bar_polar",
        {"r": "CTR", "theta": "Channel", "color": "Placement"},
        ["", "Day", "Month", "Channel", "Placement", "CTR"],
        key_suffix="polar",
    )

    assert fields == {"r": "CTR", "theta": "Channel", "color": "Placement"}
    assert [call["label"] for call in calls] == ["R", "Theta", "Color"]
    assert [call["key"] for call in calls] == [
        "builder_tile_r_polar",
        "builder_tile_theta_polar",
        "builder_tile_color_polar",
    ]


@pytest.mark.unit
def test_tile_field_controls_show_gauge_facet_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_selectbox(
        label: str,
        options: list[str],
        *,
        index: int = 0,
        key: str,
        **_: object,
    ) -> str:
        calls.append({"label": label, "options": options, "index": index, "key": key})
        return options[index]

    monkeypatch.setattr(st, "selectbox", fake_selectbox)
    monkeypatch.setattr(config_builder, "_chart_setting_controls", lambda *_, **__: {})

    fields = config_builder._tile_field_controls(
        "gauge",
        {"value": "CTR", "facet_row": "Channel", "facet_col": "Placement"},
        ["", "Day", "Month", "Channel", "Placement", "CTR"],
        key_suffix="gauge",
    )

    assert fields == {"value": "CTR", "facet_row": "Channel", "facet_col": "Placement"}
    assert [call["label"] for call in calls] == ["Value", "Facet_Row", "Facet_Col"]
    assert [call["key"] for call in calls] == [
        "builder_tile_value_gauge",
        "builder_tile_facet_row_gauge",
        "builder_tile_facet_col_gauge",
    ]


@pytest.mark.unit
def test_tile_field_controls_show_scatter_animation_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_selectbox(
        label: str,
        options: list[str],
        *,
        index: int = 0,
        key: str,
        **_: object,
    ) -> str:
        calls.append({"label": label, "options": options, "index": index, "key": key})
        return options[index]

    monkeypatch.setattr(st, "selectbox", fake_selectbox)
    monkeypatch.setattr(config_builder, "_chart_setting_controls", lambda *_, **__: {})

    fields = config_builder._tile_field_controls(
        "scatter",
        {
            "x": "Count",
            "y": "Lift",
            "color": "Group",
            "size": "Positives",
            "animation_frame": "Month",
            "animation_group": "Issue",
        },
        ["", "Month", "Issue", "Group", "Count", "Positives", "Lift"],
        key_suffix="scatter",
    )

    assert fields == {
        "x": "Count",
        "y": "Lift",
        "color": "Group",
        "size": "Positives",
        "animation_frame": "Month",
        "animation_group": "Issue",
        "facet_row": "",
        "facet_col": "",
    }
    assert "builder_tile_animation_frame_scatter" in [call["key"] for call in calls]
    assert "builder_tile_animation_group_scatter" in [call["key"] for call in calls]


@pytest.mark.unit
def test_chart_field_controls_start_with_required_metadata() -> None:
    for chart_kind, required_fields in builder.CHART_REQUIRED_FIELDS.items():
        controls = builder.chart_field_controls(chart_kind)

        assert controls[: len(required_fields)] == required_fields
        assert set(required_fields) <= set(controls)


@pytest.mark.unit
def test_tile_field_controls_render_descriptive_funnel_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectbox_calls: list[dict[str, object]] = []
    text_calls: list[dict[str, object]] = []

    def fake_selectbox(
        label: str,
        options: list[str],
        *,
        index: int = 0,
        key: str,
        **_: object,
    ) -> str:
        selectbox_calls.append({"label": label, "options": options, "index": index, "key": key})
        return options[index]

    def fake_text_input(label: str, *, value: str = "", key: str, **_: object) -> str:
        text_calls.append({"label": label, "value": value, "key": key})
        return value

    monkeypatch.setattr(st, "selectbox", fake_selectbox)
    monkeypatch.setattr(st, "text_input", fake_text_input)
    monkeypatch.setattr(config_builder, "_chart_setting_controls", lambda *_, **__: {})

    fields = config_builder._tile_field_controls(
        "descriptive_funnel",
        {
            "x": "Outcome",
            "color": "Issue",
            "stages": ["Impression", "Clicked", "Conversion"],
            "facet_col": "Channel",
        },
        ["", "Outcome", "Issue", "Channel", "Outcome_Count"],
        key_suffix="desc_funnel",
    )

    assert fields == {
        "x": "Outcome",
        "color": "Issue",
        "stages": ["Impression", "Clicked", "Conversion"],
        "facet_row": "",
        "facet_col": "Channel",
    }
    assert [call["key"] for call in selectbox_calls] == [
        "builder_tile_x_desc_funnel",
        "builder_tile_color_desc_funnel",
        "builder_tile_facet_row_desc_funnel",
        "builder_tile_facet_col_desc_funnel",
    ]
    assert text_calls == [
        {
            "label": "Stages",
            "value": "Impression, Clicked, Conversion",
            "key": "builder_tile_stages_desc_funnel",
        }
    ]


@pytest.mark.unit
def test_tile_field_controls_show_size_only_for_supported_charts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_selectbox(
        label: str,
        options: list[str],
        *,
        index: int = 0,
        key: str,
        **_: object,
    ) -> str:
        calls.append({"label": label, "options": options, "index": index, "key": key})
        return options[index]

    monkeypatch.setattr(st, "selectbox", fake_selectbox)
    monkeypatch.setattr(config_builder, "_chart_setting_controls", lambda *_, **__: {})

    line_fields = config_builder._tile_field_controls(
        "line",
        {"x": "Month", "y": "CTR", "color": "Group", "size": "Count"},
        ["", "Month", "Group", "CTR", "Count"],
        key_suffix="line",
    )
    geo_fields = config_builder._tile_field_controls(
        "geo_map",
        {"locations": "Country", "value": "CTR", "size": "Count"},
        ["", "Country", "CTR", "Count"],
        key_suffix="geo",
    )

    assert "size" not in line_fields
    assert geo_fields["size"] == "Count"
    assert "builder_tile_size_line" not in [call["key"] for call in calls]
    assert "builder_tile_size_geo" in [call["key"] for call in calls]


@pytest.mark.unit
def test_gauge_chart_settings_include_optional_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExpander:
        def __enter__(self) -> FakeExpander:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(st, "expander", lambda *_, **__: FakeExpander())
    monkeypatch.setattr(st, "selectbox", lambda _label, options, *, index=0, **__: options[index])
    monkeypatch.setattr(st, "checkbox", lambda label, **__: label == "Reference")
    monkeypatch.setattr(st, "number_input", lambda _label, *, value=0.0, **__: value)
    monkeypatch.setattr(st, "text_area", lambda *_, **__: "")

    settings = config_builder._chart_setting_controls(
        "gauge",
        {"reference": 0.12},
        ["", "CTR"],
        "gauge",
    )

    assert settings["reference"] == 0.12


@pytest.mark.unit
def test_default_tile_fields_leave_calibration_axes_empty() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "ih_ml",
                        "source": "ih",
                        "kind": "score_distribution",
                        "group_by": ["placement_type"],
                        "score_properties": ["propensity", "final_propensity"],
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "MIL_Calibration": {
                        "source": "ih_ml",
                        "kind": "calibration_from_digests",
                        "positive_state": "final_propensity_tdigest_positives",
                        "negative_state": "final_propensity_tdigest_negatives",
                    }
                }
            },
            "dashboards": {"dashboards": []},
        }
    )

    fields = builder.default_tile_fields(catalog, "MIL_Calibration", "calibration_curve")

    assert fields == {}


@pytest.mark.unit
def test_curve_metric_chart_choices_include_model_curves() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "ih_ml",
                        "source": "ih",
                        "kind": "score_distribution",
                        "group_by": ["placement_type"],
                        "score_properties": ["propensity"],
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "MIL_ROC_AUC": {
                        "source": "ih_ml",
                        "kind": "curve_from_digests",
                        "positive_state": "propensity_tdigest_positives",
                        "negative_state": "propensity_tdigest_negatives",
                        "output": "roc_auc",
                    }
                }
            },
            "dashboards": {"dashboards": []},
        }
    )

    choices = builder.chart_choices_for_metric(catalog, "MIL_ROC_AUC")
    fields = builder.default_tile_fields(catalog, "MIL_ROC_AUC", "roc_curve")

    assert {"roc_curve", "precision_recall_curve", "gain_curve", "lift_curve"} <= set(choices)
    assert fields == {"color": "placement_type"}


@pytest.mark.unit
def test_tile_field_default_preserves_legacy_facet_defaults() -> None:
    assert config_builder._tile_field_default({"facet_column": "placement_type"}, "facet_col") == (
        "placement_type"
    )
    assert config_builder._tile_field_default({"facets": {"row": "region"}}, "facet_row") == (
        "region"
    )
    assert config_builder._tile_field_default({"group_by": ["region"]}, "facet_row") == "region"


@pytest.mark.unit
def test_processor_to_dict_uses_authoring_dimensions() -> None:
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
        }
    )

    data = builder.processor_to_dict(processor)

    assert data["dimensions"] == ["Channel"]
    assert "group_by" not in data
    assert "states" not in data


@pytest.mark.unit
def test_processor_to_dict_materializes_bulk_sketch_default() -> None:
    processor = model.NumericDistributionProcessor.model_validate(
        {
            "id": "descriptive",
            "source": "ih",
            "kind": "numeric_distribution",
            "properties": ["Revenue"],
        }
    )

    data = builder.processor_to_dict(processor)

    assert data["sketch_build_mode"] == "bulk"
    assert "sketch_build_mode" in forms.PROCESSOR_KIND_MANAGED_FIELDS


@pytest.mark.unit
def test_quantile_processor_editors_default_to_bulk_and_keep_legacy_escape_hatch() -> None:
    app = AppTest.from_string(
        """
from valuestream.ui import forms

forms.processor_kind_fields(
    {"properties": ["Revenue"]},
    "numeric_distribution",
    field_options=["Revenue"],
    numeric_field_options=["Revenue"],
    key_prefix="numeric",
)
forms.processor_kind_fields(
    {
        "score_properties": ["Propensity"],
        "sketch_build_mode": "legacy",
        "outcome": {
            "column": "Outcome",
            "positive_values": ["Clicked"],
            "negative_values": ["Impression"],
        },
    },
    "score_distribution",
    field_options=["CustomerID", "Outcome", "Propensity"],
    numeric_field_options=["Propensity"],
    key_prefix="score",
)
"""
    ).run()

    assert not app.exception
    mode_selectors = [item for item in app.selectbox if item.label == "Sketch Build Mode"]
    assert [item.value for item in mode_selectors] == ["bulk", "legacy"]
    assert all(item.options == ["bulk", "legacy"] for item in mode_selectors)


@pytest.mark.unit
def test_metric_to_dict_omits_empty_base_fields() -> None:
    metric = model.FormulaMetric.model_validate(
        {
            "source": "engagement",
            "kind": "formula",
            "expression": {
                "op": "safe_div",
                "num": {"col": "Positives"},
                "den": {"col": "Count"},
            },
        }
    )

    data = builder.metric_to_dict(metric)

    assert data == {
        "source": "engagement",
        "kind": "formula",
        "expression": {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {"col": "Count"},
        },
    }


@pytest.mark.unit
def test_metric_to_dict_keeps_sparse_authored_display_fields() -> None:
    metric = model.FormulaMetric.model_validate(
        {
            "source": "engagement",
            "kind": "formula",
            "expression": {"col": "Count"},
            "display": {"label": "Interactions"},
        }
    )

    data = builder.metric_to_dict(metric)

    assert data["display"] == {"label": "Interactions"}


@pytest.mark.unit
def test_lifecycle_metric_does_not_invent_outputs_for_existing_sparse_yaml() -> None:
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import forms

ctx = forms.MetricFormContext(state_options=lambda _types: [])
st.session_state["result"] = forms.metric_kind_fields(
    "lifecycle_summary",
    {"source": "lifecycle", "kind": "lifecycle_summary"},
    ctx,
    key_prefix="lifecycle_sparse",
)
"""
    ).run()

    assert not app.exception
    assert app.session_state["result"] == {}


@pytest.mark.unit
def test_set_op_editor_flags_unsupported_window_shapes() -> None:
    """Legacy start/end windows are engine-invalid; the editor demands fixes.

    The engine only accepts ``last`` and two-value ``between`` windows, so a
    metric carrying another shape surfaces blank Between offsets plus
    warnings instead of silently passing the broken window through.
    """
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import forms

seed = {
    "source": "audiences",
    "kind": "set_op",
    "op": "intersection",
    "operands": [
        {"state": "Active_theta", "time_window": {"start": "-30d", "end": "-1d"}},
        {"state": "Active_theta", "time_window": {"start": "-1d", "end": "now"}},
    ],
}
ctx = forms.MetricFormContext(
    state_options=lambda _types: ["Active_theta", "Known_theta"],
)
st.session_state["result"] = forms.metric_kind_fields(
    "set_op", seed, ctx, key_prefix="windowed_set"
)
"""
    ).run()

    assert not app.exception
    assert app.session_state["result"] is None
    assert any("anchor-relative offset" in item.value for item in app.warning)


@pytest.mark.unit
def test_build_state_defs_uses_direct_editor_rows() -> None:
    processor = model.NumericDistributionProcessor.model_validate(
        {
            "id": "descriptive",
            "source": "ih",
            "kind": "numeric_distribution",
            "properties": ["Revenue"],
        }
    )

    states = config_builder._build_state_defs(
        processor,
        [
            {
                "State": "Revenue_Sum",
                "Type": "value_sum",
                "Source Column": "NetRevenue",
                "Enabled": True,
            },
            {
                "State": "Revenue_tdigest",
                "Type": "tdigest",
                "Source Column": "Revenue",
                "Enabled": False,
            },
        ],
    )

    assert states == {
        "Revenue_Sum": {
            "type": "value_sum",
            "per_property": True,
            "source_column": "NetRevenue",
        }
    }


@pytest.mark.unit
def test_default_rows_with_fields_appends_selected_fields_without_duplicates() -> None:
    rows = [
        {"Field": "", "Default Value": "", "Enabled": True},
        {"Field": "Channel", "Default Value": "Unknown", "Enabled": True},
    ]

    updated = builder.default_rows_with_fields(rows, ["Channel", "Outcome", " NewField "])

    assert updated == [
        {"Field": "Channel", "Default Value": "Unknown", "Enabled": True},
        {"Field": "Outcome", "Default Value": "", "Enabled": True},
        {"Field": "NewField", "Default Value": "", "Enabled": True},
    ]


@pytest.mark.unit
def test_build_source_definition_can_add_rename_capitalize_defaults_transform() -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {
                "kind": "parquet",
                "file_pattern": "data/*.parquet",
            },
            "schema": {
                "timestamp_column": "OutcomeTime",
                "natural_key": ["InteractionID"],
                "drop_columns": [],
            },
            "defaults": {"Revenue": 0.0},
            "transforms": [
                {"kind": "parse_datetime", "columns": ["OutcomeTime"], "format": "%Y-%m-%d"}
            ],
        }
    )

    source_def = config_builder._build_source_definition(
        source=source,
        source_id="ih",
        description="",
        reader_kind="parquet",
        file_pattern="data/*.parquet",
        group_by_filename=None,
        root="",
        streaming=False,
        hive_partitioning=False,
        timestamp_column="OutcomeTime",
        natural_key=["InteractionID"],
        drop_columns=[],
        default_rows=[{"Field": "Revenue", "Default Value": "0.0", "Enabled": True}],
        use_rename_capitalize=True,
        filter_expression=None,
        calculated_rows=[],
    )

    assert source_def["defaults"] == {}
    assert source_def["transforms"][:2] == [
        {"kind": "rename_capitalize"},
        {"kind": "parse_datetime", "columns": ["OutcomeTime"], "format": "%Y-%m-%d"},
    ]
    assert source_def["transforms"][2] == {"kind": "defaults", "values": {"Revenue": 0.0}}


@pytest.mark.unit
def test_source_row_add_edit_delete_operations_all_dirty_the_outer_draft() -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "schema": {"natural_key": ["InteractionID"]},
            "defaults": {"Channel": "Unknown"},
            "transforms": [
                {
                    "kind": "filter",
                    "expression": {"op": "eq", "column": "Channel", "value": "Web"},
                },
                {
                    "kind": "derive_column",
                    "output": "Margin",
                    "expression": {
                        "op": "sub",
                        "args": [{"col": "Revenue"}, {"col": "Cost"}],
                    },
                },
            ],
        }
    )
    baseline = builder.source_to_dict(source)
    default_rows = builder.default_rows_from_values(builder.source_defaults(source))
    filter_rows = builder.filter_rows_from_expression(builder.first_filter_expression(source))
    calculated_rows = builder.calculated_rows_from_source(source)
    assert filter_rows is not None

    changes: list[tuple[list[dict], list[dict], list[dict]]] = []

    added_defaults = copy.deepcopy(default_rows)
    added_defaults.append({"Field": "Region", "Default Value": "Unknown", "Enabled": True})
    edited_defaults = copy.deepcopy(default_rows)
    edited_defaults[0]["Default Value"] = "Other"
    changes.extend(
        [
            (added_defaults, filter_rows, calculated_rows),
            (edited_defaults, filter_rows, calculated_rows),
            ([], filter_rows, calculated_rows),
        ]
    )

    added_filters = copy.deepcopy(filter_rows)
    added_filters.append({"Field": "Region", "Operator": "==", "Value": "EU", "Enabled": True})
    edited_filters = copy.deepcopy(filter_rows)
    edited_filters[0]["Value"] = "Mobile"
    changes.extend(
        [
            (default_rows, added_filters, calculated_rows),
            (default_rows, edited_filters, calculated_rows),
            (default_rows, [], calculated_rows),
        ]
    )

    added_calculations = copy.deepcopy(calculated_rows)
    added_calculations.append(
        {
            "Name": "RevenueCopy",
            "Mode": "AST YAML",
            "Left": "",
            "Right Kind": "Field",
            "Right": "",
            "Expression": "col: Revenue",
            "Enabled": True,
        }
    )
    edited_calculations = copy.deepcopy(calculated_rows)
    # The loader now recognizes sub(Revenue, Cost) as a Subtract-mode row, so a
    # meaningful edit changes its operands rather than the unused Expression.
    assert edited_calculations[0]["Mode"] == "Subtract"
    edited_calculations[0]["Right"] = "Discount"
    changes.extend(
        [
            (default_rows, filter_rows, added_calculations),
            (default_rows, filter_rows, edited_calculations),
            (default_rows, filter_rows, []),
        ]
    )

    for proposed_defaults, proposed_filters, proposed_calculations in changes:
        proposed = config_builder._build_source_definition(
            source=source,
            source_id=source.id,
            description=source.description,
            reader_kind=source.reader.kind,
            file_pattern=source.reader.file_pattern,
            group_by_filename=source.reader.group_by_filename,
            root="",
            streaming=False,
            hive_partitioning=False,
            timestamp_column=source.schema_.timestamp_column,
            natural_key=list(source.schema_.natural_key),
            drop_columns=list(source.schema_.drop_columns),
            default_rows=proposed_defaults,
            use_rename_capitalize=False,
            filter_expression=builder.compile_filter_rows(proposed_filters),
            calculated_rows=proposed_calculations,
        )

        assert builder.builder_draft_status("source:ih", baseline, proposed).dirty


@pytest.mark.unit
def test_nested_source_and_processor_row_editors_rerun_the_owning_fragment() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    row_editor_names = {
        "_render_default_values_editor",
        "_render_filter_rows_editor",
        "_render_calculated_rows_editor",
        "_render_state_rows_editor",
    }
    row_editors = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in row_editor_names
    }

    assert set(row_editors) == row_editor_names
    for function in row_editors.values():
        assert not any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "fragment"
            for decorator in function.decorator_list
        )


@pytest.mark.unit
def test_build_source_definition_preserves_untouched_transform_order() -> None:
    source = model.Source.model_validate(
        {
            "id": "sample",
            "description": "Generated sample",
            "reader": {
                "kind": "csv",
                "file_pattern": "sample.csv",
                "root": "data/studio",
            },
            "schema": {
                "timestamp_column": "OutcomeTime",
                "natural_key": ["CustomerID"],
                "drop_columns": [],
            },
            "transforms": [
                {
                    "kind": "derive_column",
                    "output": "SubjectID",
                    "expression": {"col": "CustomerID"},
                },
                {
                    "kind": "parse_datetime",
                    "columns": ["OutcomeTime"],
                    "format": "%+",
                },
                {
                    "kind": "derive_calendar",
                    "from": "OutcomeTime",
                    "outputs": ["Day", "Month", "Quarter", "Year"],
                },
            ],
        }
    )

    source_def = config_builder._build_source_definition(
        source=source,
        source_id=source.id,
        description=source.description,
        reader_kind=source.reader.kind,
        file_pattern=source.reader.file_pattern,
        group_by_filename=source.reader.group_by_filename,
        root="data/studio",
        streaming=False,
        hive_partitioning=False,
        timestamp_column=source.schema_.timestamp_column,
        natural_key=list(source.schema_.natural_key),
        drop_columns=list(source.schema_.drop_columns),
        default_rows=[],
        use_rename_capitalize=False,
        filter_expression=None,
        calculated_rows=builder.calculated_rows_from_source(source),
    )

    assert source_def == builder.source_to_dict(source)
    assert not builder.builder_draft_status(
        "source:sample", builder.source_to_dict(source), source_def
    ).dirty


@pytest.mark.unit
def test_preserve_untouched_processor_definition_ignores_materialized_default_states() -> None:
    processor = model.Processors.model_validate(
        {
            "processors": [
                {
                    "id": "sample_engagement",
                    "source": "sample",
                    "kind": "binary_outcome",
                    "description": "Generated engagement processor.",
                    "dimensions": ["Channel"],
                    "time": {
                        "column": "OutcomeTime",
                        "grains": ["Day", "Month", "Summary"],
                        "aggregation_levels": {"Month": "Day"},
                    },
                    "entities": {"subject": "SubjectID"},
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked", "Conversion"],
                        "negative_values": ["Impression", "Pending"],
                    },
                }
            ]
        }
    ).processors[0]
    proposed = builder.processor_to_dict(processor)
    proposed["states"] = {
        name: spec.model_dump(mode="json", by_alias=True, exclude_none=True)
        for name, spec in model.effective_processor_states(processor).items()
    }

    preserved = config_builder._preserve_untouched_processor_definition(
        processor,
        proposed,
    )

    assert preserved == builder.processor_to_dict(processor)
    assert "states" not in preserved
    assert preserved["time"]["aggregation_levels"] == {"Month": "Day"}


@pytest.mark.unit
def test_preserve_untouched_processor_definition_keeps_real_edits() -> None:
    processor = model.Processors.model_validate(
        {
            "processors": [
                {
                    "id": "engagement",
                    "source": "events",
                    "kind": "binary_outcome",
                    "description": "Original description",
                    "entities": {"subject": "SubjectID"},
                    "outcome": {"column": "Outcome"},
                }
            ]
        }
    ).processors[0]
    proposed = builder.processor_to_dict(processor)
    proposed["description"] = "Updated description"
    proposed["time"] = {"column": None, "grains": ["Day", "Month", "Summary"]}
    proposed["states"] = {
        name: spec.model_dump(mode="json", by_alias=True, exclude_none=True)
        for name, spec in model.effective_processor_states(processor).items()
    }

    preserved = config_builder._preserve_untouched_processor_definition(
        processor,
        proposed,
    )

    assert preserved["description"] == "Updated description"


@pytest.mark.unit
def test_source_field_options_apply_rename_capitalize_to_reader_columns(monkeypatch) -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "schema": {
                "timestamp_column": "pxOutcomeTime",
                "natural_key": ["pyCustomerID"],
                "drop_columns": [],
            },
            "transforms": [
                {"kind": "rename_capitalize"},
                {"kind": "parse_datetime", "columns": ["OutcomeTime"], "format": "%Y-%m-%d"},
                {"kind": "derive_column", "output": "ResponseTime", "expression": {"col": "Name"}},
            ],
        }
    )
    ctx = SimpleNamespace(
        catalog=SimpleNamespace(processors=SimpleNamespace(processors=[])),
        workspace=Path("."),
    )

    def fake_sample_columns(
        _ctx: object,
        _source: model.Source,
        *,
        rename_capitalize: bool = False,
    ) -> list[str]:
        columns = ["pyName", "pxOutcomeTime", "pyCustomerID"]
        return config_builder._rename_capitalize_fields(columns, rename_capitalize)

    monkeypatch.setattr(config_builder, "_source_sample_columns", fake_sample_columns)

    options = config_builder._source_field_options(ctx, source)

    assert "Name" in options
    assert "OutcomeTime" in options
    assert "CustomerID" in options
    assert "ResponseTime" in options
    assert "pyName" not in options
    assert "pxOutcomeTime" not in options
    assert "pyCustomerID" not in options


@pytest.mark.unit
def test_source_inspection_shares_reads_and_invalidates_on_file_or_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "events.parquet"
    source_file.write_text("first", encoding="utf-8")
    source = model.Source.model_validate(
        {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "events.parquet"},
            "schema": {"natural_key": ["CustomerID"]},
        }
    )
    ctx = SimpleNamespace(workspace=tmp_path)
    calls = {"discover": 0, "read": 0, "cleanup": 0, "status": 0}
    status_updates: list[dict[str, object]] = []

    def fake_discover(_workspace: Path, _source: model.Source) -> list[SimpleNamespace]:
        calls["discover"] += 1
        return [SimpleNamespace(files=(source_file,))]

    def fake_read(_reader: model.Reader, _files: tuple[Path, ...]) -> pl.LazyFrame:
        calls["read"] += 1
        return pl.DataFrame({"CustomerID": ["c1", "c2", "c3"], "Outcome": ["Clicked"] * 3}).lazy()

    def fake_status(_label: str, **_kwargs: object):
        calls["status"] += 1
        return nullcontext(
            SimpleNamespace(update=lambda **kwargs: status_updates.append(dict(kwargs)))
        )

    monkeypatch.setattr(config_builder, "discover", fake_discover)
    monkeypatch.setattr(config_builder, "read", fake_read)
    monkeypatch.setattr(
        config_builder,
        "cleanup_temporaries",
        lambda: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    )
    monkeypatch.setattr(
        config_builder,
        "st",
        SimpleNamespace(session_state={}, status=fake_status, error=lambda *_args: None),
    )

    config_builder._clear_source_inspection_cache()
    assert config_builder._source_sample_columns(ctx, source) == ["CustomerID", "Outcome"]
    sample = config_builder._source_profile_sample(ctx, source)
    assert sample is not None
    assert sample.height == 3
    assert calls == {"discover": 1, "read": 1, "cleanup": 1, "status": 1}

    config_builder._begin_source_inspection_scope()
    assert config_builder._source_sample_columns(ctx, source) == ["CustomerID", "Outcome"]
    assert calls == {"discover": 2, "read": 1, "cleanup": 2, "status": 1}

    source_file.write_text("second version", encoding="utf-8")
    config_builder._begin_source_inspection_scope()
    config_builder._source_sample_columns(ctx, source)
    assert calls == {"discover": 3, "read": 2, "cleanup": 3, "status": 2}

    changed_source = source.model_copy(update={"description": "Changed canonical source"})
    config_builder._begin_source_inspection_scope()
    config_builder._source_sample_columns(ctx, changed_source)
    assert calls == {"discover": 4, "read": 3, "cleanup": 4, "status": 3}
    assert status_updates[-1]["state"] == "complete"


@pytest.mark.unit
def test_source_inspection_failure_retry_invalidates_only_failed_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "broken": tmp_path / "broken.parquet",
        "healthy": tmp_path / "healthy.parquet",
    }
    for path in files.values():
        path.write_text(path.stem, encoding="utf-8")
    sources = {
        source_id: model.Source.model_validate(
            {
                "id": source_id,
                "reader": {"kind": "parquet", "file_pattern": f"{source_id}/*.parquet"},
            }
        )
        for source_id in files
    }
    ctx = SimpleNamespace(workspace=tmp_path)
    fail_broken = {"value": True}
    calls = {"broken": 0, "healthy": 0, "cleanup": 0}
    errors: list[str] = []
    buttons: list[dict[str, object]] = []

    def fake_discover(_workspace: Path, source: model.Source) -> list[SimpleNamespace]:
        return [SimpleNamespace(files=(files[source.id],))]

    def fake_read(_reader: model.Reader, selected: tuple[Path, ...]) -> pl.LazyFrame:
        source_id = selected[0].stem
        calls[source_id] += 1
        if source_id == "broken" and fail_broken["value"]:
            raise ValueError("private reader details")
        return pl.DataFrame({"CustomerID": [source_id]}).lazy()

    def fake_button(label: str, **kwargs: object) -> bool:
        buttons.append({"label": label, **kwargs})
        return False

    monkeypatch.setattr(config_builder, "discover", fake_discover)
    monkeypatch.setattr(config_builder, "read", fake_read)
    monkeypatch.setattr(
        config_builder,
        "cleanup_temporaries",
        lambda: calls.__setitem__("cleanup", calls["cleanup"] + 1),
    )
    monkeypatch.setattr(
        config_builder,
        "st",
        SimpleNamespace(
            session_state={},
            status=lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(update=lambda **_update: None)
            ),
            error=errors.append,
            button=fake_button,
        ),
    )

    config_builder._clear_source_inspection_cache()
    assert config_builder._source_sample_columns(ctx, sources["broken"]) == []
    assert config_builder._source_sample_columns(ctx, sources["broken"]) == []
    assert config_builder._source_sample_columns(ctx, sources["healthy"]) == ["CustomerID"]
    assert calls == {"broken": 1, "healthy": 1, "cleanup": 2}
    assert len(errors) == 1
    assert "source `broken`" in errors[0]
    assert "path pattern `broken/*.parquet`" in errors[0]
    assert "private reader details" not in errors[0]
    assert [button["label"] for button in buttons] == ["Retry source inspection"]

    retry = buttons[0]
    retry["on_click"](*retry["args"])  # type: ignore[operator]
    fail_broken["value"] = False
    config_builder._begin_source_inspection_scope()
    assert config_builder._source_sample_columns(ctx, sources["healthy"]) == ["CustomerID"]
    assert config_builder._source_sample_columns(ctx, sources["broken"]) == ["CustomerID"]
    assert calls == {"broken": 2, "healthy": 1, "cleanup": 4}


@pytest.mark.unit
def test_source_inspection_cache_keeps_only_bounded_transformed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "large.parquet"
    source_file.write_text("identity", encoding="utf-8")
    source = model.Source.model_validate(
        {
            "id": "large",
            "reader": {"kind": "parquet", "file_pattern": "large.parquet"},
        }
    )
    ctx = SimpleNamespace(workspace=tmp_path)
    raw = pl.DataFrame({"RowID": list(range(100)), "Value": list(range(100))})

    monkeypatch.setattr(
        config_builder,
        "discover",
        lambda *_args: [SimpleNamespace(files=(source_file,))],
    )
    monkeypatch.setattr(config_builder, "read", lambda *_args: raw.lazy())
    monkeypatch.setattr(config_builder, "cleanup_temporaries", lambda: None)
    monkeypatch.setattr(
        config_builder,
        "st",
        SimpleNamespace(
            session_state={},
            status=lambda *_args, **_kwargs: nullcontext(
                SimpleNamespace(update=lambda **_update: None)
            ),
            error=lambda *_args: None,
        ),
    )

    config_builder._clear_source_inspection_cache()
    result = config_builder._source_inspection(ctx, source, limit=5)

    assert result.sample is not None
    assert result.sample.height == 5
    assert result.raw_schema == (("RowID", "Int64"), ("Value", "Int64"))
    assert set(vars(result)) == {"key", "raw_schema", "sample", "error_kind"}
    assert not any(isinstance(value, pl.LazyFrame) for value in vars(result).values())
    scope = config_builder.st.session_state[config_builder.BUILDER_SOURCE_INSPECTION_SCOPE_KEY]
    assert all(
        isinstance(value, config_builder.SourceInspectionKey) for value in scope["keys"].values()
    )
    assert len(config_builder._SOURCE_INSPECTION_CACHE) == 1
    cached = next(iter(config_builder._SOURCE_INSPECTION_CACHE.values()))
    assert cached.sample is not None
    assert cached.sample.height == 5


@pytest.mark.unit
def test_source_field_options_include_entity_subject_and_are_ordered(monkeypatch) -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
        }
    )
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "dimensions": ["Issue"],
            "entities": {"subject": "SubjectID"},
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
            "states": {"Count": {"type": "count"}},
        }
    )
    ctx = SimpleNamespace(
        catalog=SimpleNamespace(processors=SimpleNamespace(processors=[processor])),
        workspace=Path("."),
    )
    monkeypatch.setattr(
        config_builder,
        "_source_sample_columns",
        lambda *_args, **_kwargs: ["ExperimentName", "AppliedModel"],
    )

    options = config_builder._source_field_options(ctx, source)

    assert options == ["AppliedModel", "ExperimentName", "Issue", "Outcome", "SubjectID"]


@pytest.mark.unit
def test_source_rename_mapping_remaps_editor_fields(monkeypatch) -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "transforms": [],
        }
    )
    ctx = SimpleNamespace(
        catalog=SimpleNamespace(processors=SimpleNamespace(processors=[])),
        workspace=Path("."),
    )

    def fake_sample_columns(
        _ctx: object,
        _source: model.Source,
        *,
        rename_capitalize: bool = False,
    ) -> list[str]:
        columns = ["pyName", "pxOutcomeTime"]
        return config_builder._rename_capitalize_fields(columns, rename_capitalize)

    monkeypatch.setattr(config_builder, "_source_sample_columns", fake_sample_columns)

    assert config_builder._source_rename_mapping(ctx, source, True) == {
        "pyName": "Name",
        "pxOutcomeTime": "OutcomeTime",
    }
    assert config_builder._source_rename_mapping(ctx, source, False) == {
        "Name": "pyName",
        "OutcomeTime": "pxOutcomeTime",
    }


@pytest.mark.unit
def test_dimension_profile_classifies_group_by_candidates() -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "schema": {
                "timestamp_column": "OutcomeTime",
                "natural_key": ["CustomerID"],
            },
        }
    )
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
            "states": {
                "Count": {"type": "count"},
                "PropensityDigest": {
                    "type": "tdigest",
                    "source_column": "Propensity",
                },
            },
        }
    )
    ctx = SimpleNamespace(
        catalog=SimpleNamespace(
            processors=SimpleNamespace(processors=[processor]),
        )
    )
    sample = pl.DataFrame(
        {
            "Channel": ["Web", "Mobile", "Web", "Branch", "Mobile"],
            "Issue": ["Cards", "Loans", "Cards", "Cards", "Loans"],
            "CustomerType": ["Mass", "Premier", "Mass", "Mass", "Premier"],
            "CustomerID": ["c1", "c2", "c3", "c4", "c5"],
            "Outcome": ["Clicked", "Impression", "Clicked", "Impression", "Clicked"],
            "OutcomeTime": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
                "2026-01-05",
            ],
            "Propensity": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    rows = {
        row.field: row
        for row in dimension_profile.source_dimension_profile_rows(ctx, source, sample)
    }

    assert rows["Channel"].recommendation == "Active"
    assert rows["Issue"].recommendation == "Review"
    assert rows["CustomerType"].recommendation == "Review"
    assert rows["CustomerID"].recommendation == "Avoid"
    assert rows["Outcome"].recommendation == "Avoid"
    assert rows["OutcomeTime"].recommendation == "Avoid"
    assert rows["Propensity"].recommendation == "Avoid"
    assert rows["Issue"].safe_for_group_by == "Review"
    assert rows["CustomerID"].safe_for_group_by == "No"
    assert "Safe For Group-By" in dimension_profile.profile_frame(list(rows.values())).columns
    assert dimension_profile.dimension_pack_fields(sample.columns) == [
        "Channel",
        "Issue",
    ]
    assert dimension_profile.sketch_recommendations(list(rows.values()))[0]["Sketch"] == "CPC"


@pytest.mark.unit
def test_dimension_recommendation_requires_at_least_three_distinct_values() -> None:
    common = {
        "field": "Segment",
        "dtype": "String",
        "current_usage": [],
        "protected": False,
        "non_null": 10,
        "cardinality_rate": 0.2,
        "null_rate": 0.0,
    }

    assert dimension_profile.dimension_recommendation(**common, cardinality=1) == (
        "Avoid",
        "Fewer than 3 distinct values; not useful as a default breakdown.",
    )
    assert dimension_profile.dimension_recommendation(**common, cardinality=2) == (
        "Review",
        "Fewer than 3 distinct values; not useful as a default breakdown.",
    )
    assert dimension_profile.dimension_recommendation(**common, cardinality=3) == (
        "Recommended",
        "Low-cardinality field suitable for filters and breakdowns.",
    )


@pytest.mark.unit
def test_dimension_promotion_ranks_reviewed_fields_before_avoid() -> None:
    def row(field: str, recommendation: str) -> dimension_profile.DimensionProfileRow:
        return dimension_profile.DimensionProfileRow(
            field=field,
            dtype="String",
            non_null=10,
            null_rate=0.0,
            cardinality=4,
            cardinality_rate=0.4,
            sample_values="",
            current_usage="",
            recommendation=recommendation,
            safe_for_group_by="No" if recommendation == "Avoid" else "Review",
            reason="test",
        )

    candidates = config_builder._dimension_promotion_candidates(
        ["CustomerID", "Channel", "Campaign"],
        [],
        [
            row("CustomerID", "Avoid"),
            row("Channel", "Recommended"),
            row("Campaign", "Review"),
        ],
    )

    assert candidates == ["Channel", "Campaign", "CustomerID"]


@pytest.mark.unit
def test_dimension_pack_and_promotion_summaries_keep_exact_profile_values() -> None:
    row = dimension_profile.DimensionProfileRow(
        field="Campaign",
        dtype="String",
        non_null=9,
        null_rate=0.1,
        cardinality=4,
        cardinality_rate=0.4,
        sample_values="Retention, CrossSell",
        current_usage="",
        recommendation="Review",
        safe_for_group_by="Review",
        reason="Moderate cardinality; review aggregate growth.",
    )

    assert config_builder._dimension_pack_summary(
        ["Channel", "Campaign"],
        ["Channel"],
        ["CustomerType"],
    ) == {
        "Available in source": ("Channel", "Campaign"),
        "Already selected": ("Channel",),
        "Missing from source": ("CustomerType",),
    }
    assert config_builder._dimension_promotion_summary(row) == {
        "Recommendation": "Review",
        "Group-by safety": "Review",
        "Cardinality": 4,
        "Null values": "10.0%",
        "Reason": "Moderate cardinality; review aggregate growth.",
    }


@pytest.mark.unit
def test_dimension_badges_render_as_chips_with_collapsed_json_details() -> None:
    def app() -> None:
        import json  # noqa: PLC0415

        import streamlit as st  # noqa: PLC0415

        from valuestream.ui.pages import config_builder as page  # noqa: PLC0415

        summary = page._dimension_pack_summary(
            ["Channel", "Campaign"],
            ["Channel"],
            ["CustomerType"],
        )
        page._render_dimension_field_badges(
            "Available in source", summary["Available in source"], color="blue"
        )
        page._render_dimension_field_badges(
            "Already selected", summary["Already selected"], color="green"
        )
        page._render_dimension_field_badges(
            "Missing from source", summary["Missing from source"], color="orange"
        )
        with st.expander("Technical details · Dimension pack", expanded=False):
            st.code(json.dumps(summary, indent=2), language="json")

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    badge_values = {item.value for item in rendered.markdown if "-badge[" in item.value}
    assert badge_values == {
        ":blue-badge[Channel]",
        ":blue-badge[Campaign]",
        ":green-badge[Channel]",
        ":orange-badge[CustomerType]",
    }
    assert rendered.expander[0].label == "Technical details · Dimension pack"
    assert rendered.code[0].language == "json"


@pytest.mark.unit
def test_dimension_profile_estimates_aggregate_size_expansion() -> None:
    sample = pl.DataFrame(
        {
            "Channel": ["Web", "Web", "Mobile", "Mobile"],
            "Issue": ["Cards", "Loans", "Cards", "Loans"],
        }
    )

    preview = dimension_profile.aggregate_size_preview(sample, ["Channel"], ["Issue"])

    assert preview.current_rows == 2
    assert preview.projected_rows == 4
    assert preview.added_rows == 2
    assert preview.expansion_factor == 2.0


@pytest.mark.unit
def test_sketch_state_rows_build_processor_sketch_grid_entries() -> None:
    rows = config_builder._sketch_state_rows(
        topk_field="Campaign",
        entity_field="CustomerID",
        include_cpc=True,
        include_theta=True,
    )

    assert [(row["State"], row["Type"], row["Source Column"]) for row in rows] == [
        ("TopCampaign_topk", "topk", "Campaign"),
        ("UniqueCustomerid_cpc", "cpc", "CustomerID"),
        ("AudienceCustomerid_theta", "theta", "CustomerID"),
    ]
    assert all(row["Enabled"] for row in rows)
    # The rows compile into valid state definitions on the target processor.
    processor = model.EntitySetProcessor.model_validate(
        {"id": "audience", "source": "ih", "kind": "entity_set"}
    )
    defs = config_builder._build_state_defs(processor, rows)
    assert {name: spec["type"] for name, spec in defs.items()} == {
        "TopCampaign_topk": "topk",
        "UniqueCustomerid_cpc": "cpc",
        "AudienceCustomerid_theta": "theta",
    }
    model.EntitySetProcessor.model_validate(
        {"id": "audience", "source": "ih", "kind": "entity_set", "states": defs}
    )


@pytest.mark.unit
def test_sketch_state_rows_respect_disabled_helpers() -> None:
    assert (
        config_builder._sketch_state_rows(
            topk_field="",
            entity_field="CustomerID",
            include_cpc=False,
            include_theta=False,
        )
        == []
    )
    only_theta = config_builder._sketch_state_rows(
        topk_field="",
        entity_field="CustomerID",
        include_cpc=False,
        include_theta=True,
    )
    assert [row["State"] for row in only_theta] == ["AudienceCustomerid_theta"]


@pytest.mark.unit
def test_topk_items_metric_derives_frequent_item_rows() -> None:
    frame = pl.DataFrame(
        {"TopCampaign_topk": [topk.build(["Retention", "Retention", "CrossSell", "Retention"])]}
    )
    metric = model.TopKItemsMetric.model_validate(
        {
            "source": "campaign_exploration",
            "kind": "topk_items",
            "state": "TopCampaign_topk",
            "limit": 1,
        }
    )

    out = executor._derive_metric(frame, "TopCampaigns", metric, {})

    assert out["TopCampaigns"][0][0] == {
        "item": "Retention",
        "estimate": 3,
        "lower_bound": 3,
        "upper_bound": 3,
    }


@pytest.mark.unit
def test_quantile_metric_uses_configured_state_type_not_state_name_suffix() -> None:
    frame = pl.DataFrame({"custom_quantile_state": [kll.build([1.0, 2.0, 100.0])]})
    metric = model.TdigestQuantileMetric.model_validate(
        {
            "source": "distribution",
            "kind": "tdigest_quantile",
            "state": "custom_quantile_state",
            "quantile": 0.5,
        }
    )
    state_specs = {
        "custom_quantile_state": model.StateSpec.model_validate(
            {"type": "kll", "source_column": "Value", "k": 200}
        )
    }

    out = executor._derive_metric(
        frame,
        "Median",
        metric,
        {},
        state_specs=state_specs,
    )

    assert out["Median"].to_list() == [pytest.approx(2.0)]


@pytest.mark.unit
def test_source_rename_sync_remaps_filters_and_clears_raw_editor(monkeypatch) -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "transforms": [],
        }
    )
    ctx = SimpleNamespace(
        catalog=SimpleNamespace(processors=SimpleNamespace(processors=[])),
        workspace=Path("."),
    )

    def fake_sample_columns(
        _ctx: object,
        _source: model.Source,
        *,
        rename_capitalize: bool = False,
    ) -> list[str]:
        columns = ["pyName", "pxOutcomeTime"]
        return config_builder._rename_capitalize_fields(columns, rename_capitalize)

    monkeypatch.setattr(config_builder, "_source_sample_columns", fake_sample_columns)
    st.session_state.clear()
    st.session_state["builder_source_rename_capitalize_applied_ih"] = False
    st.session_state["builder_source_filter_rows_ih"] = [
        {"Field": "pyName", "Operator": "Equals", "Value": "Offer", "Enabled": True}
    ]
    st.session_state["builder_source_raw_filter_ih"] = "op: eq\ncolumn: pyName\nvalue: Offer\n"
    st.session_state["builder_source_filter_editor_ih"] = "stale"
    st.session_state["builder_source_raw_filter_ih_editor"] = "stale"

    config_builder._sync_source_rename_capitalize_state(ctx, source, True)

    assert st.session_state["builder_source_filter_rows_ih"][0]["Field"] == "Name"
    assert "column: Name" in st.session_state["builder_source_raw_filter_ih"]
    assert "builder_source_filter_editor_ih" not in st.session_state
    assert "builder_source_raw_filter_ih_editor" not in st.session_state
    assert "builder_next_step" not in st.session_state


@pytest.mark.unit
def test_tile_editor_token_includes_dashboard_and_page() -> None:
    first = config_builder._tile_editor_token(
        "dashboard",
        "overview",
        {"id": "summary"},
    )
    second = config_builder._tile_editor_token(
        "dashboard",
        "detail",
        {"id": "summary"},
    )

    assert first == "dashboard__overview__summary"
    assert second == "dashboard__detail__summary"
    assert first != second


@pytest.mark.unit
def test_start_new_tile_editor_uses_blank_seed_and_fresh_token() -> None:
    state: dict[str, object] = {
        "builder_tile_seed": ("dashboard", "overview", {"id": "summary"}),
        "builder_tile_editor_token": "dashboard__overview__summary",
    }

    config_builder._start_new_tile_editor(state)

    assert state["builder_tile_seed"] == (None, None, {})
    assert state["builder_tile_editor_token"] == "new_1"

    config_builder._start_new_tile_editor(state)

    assert state["builder_tile_seed"] == (None, None, {})
    assert state["builder_tile_editor_token"] == "new_2"


@pytest.mark.unit
def test_report_library_groups_every_supported_chart_once() -> None:
    categorized = [
        chart_type
        for group in config_builder.REPORT_LIBRARY_GROUPS.values()
        for chart_type in group.chart_types
    ]

    assert len(categorized) == len(set(categorized))
    assert set(categorized) == set(builder.CHART_REQUIRED_FIELDS)
    assert set(config_builder.REPORT_LIBRARY_CHART_DESCRIPTIONS) == set(
        builder.CHART_REQUIRED_FIELDS
    )


@pytest.mark.unit
def test_shared_chart_catalog_labels_and_purposes_cover_every_supported_kind() -> None:
    supported = set(builder.CHART_REQUIRED_FIELDS)

    assert set(builder.CHART_DISPLAY_LABELS) == supported
    assert set(builder.CHART_DISPLAY_PURPOSES) == supported
    assert all(builder.chart_kind_label(kind) for kind in supported)
    assert all(builder.chart_kind_purpose(kind).endswith(".") for kind in supported)
    assert builder.chart_kind_selector_label("All") == "All chart types"
    assert builder.chart_kind_selector_label("kpi_card") == "KPI card"
    assert builder.chart_kind_selector_label("bar_polar") == "Polar bar"
    assert builder.chart_kind_selector_label("experiment_z_score") == "Experiment z-score"
    assert config_builder._report_library_chart_label("roc_curve") == "ROC curve"


@pytest.mark.unit
def test_report_library_plotly_previews_cover_every_supported_chart() -> None:
    for chart_type in builder.CHART_REQUIRED_FIELDS:
        figure = config_builder._chart_library_preview(chart_type, theme_base="light")

        assert figure.data
        assert figure.layout.paper_bgcolor == "#ffffff"
        assert figure.layout.plot_bgcolor == "#ffffff"
        assert pio.to_json(figure, validate=True)


@pytest.mark.unit
def test_report_library_dark_previews_share_the_app_chart_theme() -> None:
    figure = config_builder._chart_library_preview("combo", theme_base="dark")

    assert figure.data[0].marker.color == theme.PLOTLY_DARK_COLORWAY[0]
    assert figure.data[1].line.color == theme.PLOTLY_DARK_COLORWAY[1]
    assert figure.layout.font.color == "#f7faff"
    assert figure.layout.paper_bgcolor == "#162438"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_theme", "expected"),
    [
        (None, "app"),
        ({}, "app"),
        ({"base": "light", "template": "valuestream_light"}, "light"),
        ({"template": "valuestream_dark"}, "dark"),
        ({"template": "plotly_white"}, "custom"),
        ({"base": "light", "paper_bgcolor": "#fff"}, "custom"),
    ],
)
def test_tile_theme_mode_preserves_simple_presets_and_custom_yaml(
    raw_theme: object,
    expected: str,
) -> None:
    assert config_builder._tile_theme_mode(raw_theme) == expected


@pytest.mark.unit
def test_report_library_searches_business_and_technical_tile_context(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)
    catalog = load(tmp_path)

    options = config_builder._report_library_options(
        catalog,
        search="holdings",
        metric_filter="All",
        chart_filter="All",
    )

    assert [config_builder._tile_option_key(option) for option in options] == [
        "overview/portfolio/holdings_value",
        "overview/portfolio/holdings_rate",
    ]
    assert set(options[0][3]) == {"id", "title", "metric", "chart", "value"}


@pytest.mark.unit
def test_report_tile_semantic_noop_preserves_exact_authored_defaults() -> None:
    seed = {
        "id": "trend",
        "title": "Trend",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
        "placement": "content",
    }
    rebuilt = {
        "id": "trend",
        "title": "Trend",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
    }

    preserved = config_builder._preserve_untouched_tile_definition(seed, rebuilt)

    assert preserved == seed


@pytest.mark.unit
def test_visual_tile_merge_preserves_unowned_settings_and_legacy_facets() -> None:
    seed = {
        "id": "trend",
        "title": "Trend",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
        "facets": {"row": "Channel", "column": "Issue"},
        "axis_title_standoff": 18,
        "goal_lines": [{"value": 0.25, "label": "Target"}],
    }
    rebuilt = {
        "id": "trend",
        "title": "Trend",
        "metric": "CTR",
        "chart": "line",
        "x": "Day",
        "y": "CTR",
        "facet_row": "Channel",
        "facet_col": "Issue",
    }

    merged = config_builder._merge_visual_tile_definition(seed, rebuilt, "line")

    assert merged == seed


@pytest.mark.unit
def test_visual_tile_merge_allows_clearing_a_controlled_setting() -> None:
    seed = {
        "id": "ranked",
        "title": "Ranked",
        "metric": "CTR",
        "chart": "bar",
        "x": "Channel",
        "y": "CTR",
        "top_n": 5,
        "axis_title_standoff": 18,
    }
    rebuilt = {
        "id": "ranked",
        "title": "Ranked",
        "metric": "CTR",
        "chart": "bar",
        "x": "Channel",
        "y": "CTR",
    }

    merged = config_builder._merge_visual_tile_definition(seed, rebuilt, "bar")

    assert "top_n" not in merged
    assert merged["axis_title_standoff"] == 18


@pytest.mark.unit
def test_existing_report_tile_opens_clean_in_visual_editor(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Reports / Tiles").run(timeout=15)

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)


@pytest.mark.unit
def test_dashboard_manager_confirmation_is_scoped_to_the_exact_target(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    dashboards_path = tmp_path / "catalog" / "dashboards.yaml"
    dashboards = yaml.safe_load(dashboards_path.read_text(encoding="utf-8"))
    dashboards["dashboards"][0]["pages"].append(
        {"id": "secondary", "title": "Secondary", "tiles": []}
    )
    dashboards_path.write_text(yaml.safe_dump(dashboards, sort_keys=False), encoding="utf-8")
    builder.require_valid_workspace(tmp_path)

    def app(workspace: str) -> None:
        from pathlib import Path  # noqa: PLC0415 - isolated AppTest source

        from valuestream.config.loader import load  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import (  # noqa: PLC0415
            _render_dashboard_manager,
        )

        _render_dashboard_manager(Path(workspace), load(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()

    page_confirmation = next(
        item
        for item in rendered.checkbox
        if item.key == "builder_manage_confirm_page_overview_portfolio"
    )
    rendered = page_confirmation.set_value(True).run()
    assert not next(button for button in rendered.button if button.label == "Delete page").disabled
    assert next(button for button in rendered.button if button.label == "Delete dashboard").disabled

    page_selector = next(item for item in rendered.selectbox if item.label == "Page to manage")
    rendered = page_selector.set_value("secondary").run()

    assert not rendered.exception
    new_confirmation = next(
        item
        for item in rendered.checkbox
        if item.key == "builder_manage_confirm_page_overview_secondary"
    )
    assert new_confirmation.value is False
    assert next(button for button in rendered.button if button.label == "Delete page").disabled


@pytest.mark.unit
@pytest.mark.parametrize(
    "selected",
    [
        "ih_value_stream_overview/executive_summary/interactions_by_segment",
        ("ih_value_stream_overview/experiment_monitoring/model_control_engagement_compare"),
        ("ih_value_stream_overview/aggregate_report_type_coverage/interactions_pareto"),
        ("ih_value_stream_overview/numeric_report_type_coverage/response_time_boxplot_by_outcome"),
    ],
)
def test_demo_advanced_tiles_open_clean_in_visual_editor(selected: str) -> None:
    workspace = Path(__file__).resolve().parents[2] / "examples" / "demo"

    def app(workspace_path: str, selected_tile: str) -> None:
        import streamlit as st  # noqa: PLC0415

        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        st.session_state.setdefault("builder_step", "Reports / Tiles")
        st.session_state.setdefault("builder_selected_tile_key", selected_tile)
        _builder_steps(load_context(workspace_path))

    rendered = AppTest.from_function(
        app,
        kwargs={"workspace_path": str(workspace), "selected_tile": selected},
    ).run(timeout=30)

    assert not rendered.exception
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)
    if selected.endswith("interactions_pareto"):
        assert next(item for item in rendered.number_input if item.label == "Top N").value == 12


@pytest.mark.unit
def test_new_report_tile_template_is_clean_until_edited(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Reports / Tiles").run(timeout=15)
    rendered = (
        next(button for button in rendered.button if button.label == "New").click().run(timeout=15)
    )

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)


def _render_reports_step(workspace: str) -> None:
    import streamlit as st  # noqa: PLC0415 - isolated AppTest source

    from valuestream.ui.context import load_context  # noqa: PLC0415
    from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

    st.session_state.setdefault("builder_step", "Reports / Tiles")
    _builder_steps(load_context(workspace))


@pytest.mark.unit
def test_report_library_offers_every_purpose_and_creates_from_empty_kinds(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)

    rendered = AppTest.from_function(_render_reports_step, kwargs={"workspace": str(tmp_path)}).run(
        timeout=15
    )
    assert not rendered.exception

    # Every purpose tab is offered even though the fixture only has KPI cards.
    purpose = next(item for item in rendered.segmented_control if item.label == "Purpose")
    rendered = purpose.set_value("trend").run(timeout=15)
    assert not rendered.exception

    trend_kinds = config_builder.REPORT_LIBRARY_GROUPS["trend"].chart_types
    create_keys = {
        button.key
        for button in rendered.button
        if str(button.key or "").startswith("builder_report_library_create_")
    }
    assert create_keys == {
        f"builder_report_library_create_{chart_type}" for chart_type in trend_kinds
    }

    rendered = (
        next(
            button
            for button in rendered.button
            if button.key == "builder_report_library_create_line"
        )
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert rendered.session_state["builder_selected_tile_key"] == config_builder.NEW_TILE_KEY
    assert rendered.session_state["builder_tile_seed"] == (None, None, {"chart": "line"})
    # The Tile Editor is below the gallery, so the click announces itself on
    # the clicked card. (A toast is not used: fragment reruns drop them.)
    clicked = next(
        button for button in rendered.button if button.key == "builder_report_library_create_line"
    )
    assert clicked.label == "Restart draft"
    assert any(
        "A new draft with this chart is open" in str(item.value) for item in rendered.caption
    )

    metric = [item for item in rendered.selectbox if item.label == "Metric"][-1]
    rendered = metric.set_value("CTR").run(timeout=15)
    chart = [item for item in rendered.selectbox if item.label == "Chart"][-1]
    assert chart.value == "line"


@pytest.mark.unit
def test_report_library_new_keeps_draft_selection_and_search_hides_empty_kinds(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)

    rendered = AppTest.from_function(_render_reports_step, kwargs={"workspace": str(tmp_path)}).run(
        timeout=15
    )
    rendered = (
        next(button for button in rendered.button if button.label == "New").click().run(timeout=15)
    )

    assert not rendered.exception
    assert rendered.session_state["builder_selected_tile_key"] == config_builder.NEW_TILE_KEY

    search = next(item for item in rendered.text_input if item.label == "Search")
    rendered = search.set_value("CTR").run(timeout=15)

    assert not rendered.exception
    assert not any(
        str(button.key or "").startswith("builder_report_library_create_")
        for button in rendered.button
    )


@pytest.mark.unit
def test_new_report_tile_apply_reopens_the_written_tile_clean(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Reports / Tiles").run(timeout=15)
    rendered = (
        next(button for button in rendered.button if button.label == "New").click().run(timeout=15)
    )
    metric = [item for item in rendered.selectbox if item.label == "Metric"][-1]
    rendered = metric.set_value("CTR").run(timeout=15)
    chart = [item for item in rendered.selectbox if item.label == "Chart"][-1]
    assert "KPI card" in chart.options
    assert "kpi_card" not in chart.options
    rendered = chart.set_value("kpi_card").run(timeout=15)
    assert any("Technical chart kind: `kpi_card`" in item.value for item in rendered.caption)
    title = next(item for item in rendered.text_input if item.label == "Tile Title")
    rendered = title.set_value("QA tile").run(timeout=15)

    assert any(button.label == "Apply to workspace" for button in rendered.button)
    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)
    tiles = load(tmp_path).dashboards.dashboards[0].pages[0].tiles
    assert len(tiles) == 4
    assert any(tile.title == "QA tile" for tile in tiles)


@pytest.mark.unit
def test_tile_delete_button_names_and_stages_the_exact_library_target(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Reports / Tiles").run(timeout=15)
    delete_button = next(button for button in rendered.button if button.label.startswith("Delete "))

    assert delete_button.label == "Delete 'CTR'"
    rendered = delete_button.click().run(timeout=15)

    assert not rendered.exception
    assert rendered.session_state[config_builder.BUILDER_PENDING_TILE_DELETE_KEY] == {
        "dashboard_id": "overview",
        "page_id": "portfolio",
        "tile_id": "ctr",
        "title": "CTR",
    }
    assert any("overview/portfolio/ctr" in item.value for item in rendered.warning)
    assert any(button.label == "Apply to workspace" for button in rendered.button)


@pytest.mark.unit
def test_visual_report_library_replaces_inventory_dataframe(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.config.loader import load  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import (  # noqa: PLC0415
            _render_report_library_browser,
        )

        catalog = load(workspace)
        _render_report_library_browser(catalog, sorted(catalog.metrics.metrics))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()

    assert not rendered.exception
    assert not rendered.dataframe
    assert rendered.get("plotly_chart")
    assert rendered.segmented_control[0].value == "summary"
    assert rendered.pills
    chart_filter = next(item for item in rendered.selectbox if item.label == "Chart type")
    # Every catalog chart kind is selectable, not only the configured ones.
    assert chart_filter.options[0] == "All chart types"
    assert len(chart_filter.options) == 1 + len(config_builder.REPORT_LIBRARY_GROUP_BY_CHART)
    assert "KPI card" in chart_filter.options
    assert chart_filter.value == "All"


@pytest.mark.unit
def test_report_inventory_uses_human_labels_and_keeps_ids_in_details(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    builder.write_metric_definition(
        tmp_path,
        "CTR_generated_123",
        {
            **builder.build_formula_metric("engagement", "Positives", "Count"),
            "display": {"label": "Click-through rate"},
        },
    )
    builder.write_tile_definition(
        tmp_path,
        dashboard_id="marketing_generated_123",
        dashboard_title="Marketing overview",
        page_id="executive_generated_123",
        page_title="Executive summary",
        tile={
            "id": "ctr_generated_123",
            "title": "Engagement rate",
            "metric": "CTR_generated_123",
            "chart": "kpi_card",
            "value": "CTR_generated_123",
        },
    )
    catalog = load(tmp_path)

    human_rows = config_builder._report_inventory_rows(catalog)
    technical_rows = config_builder._report_inventory_rows(catalog, technical=True)

    assert human_rows == [
        {
            "Dashboard": "Marketing overview",
            "Page": "Executive summary",
            "Report": "Engagement rate",
            "Metric": "Click-through rate",
            "Chart": "KPI card",
        }
    ]
    assert technical_rows[0]["Metric ID"] == "CTR_generated_123"
    assert technical_rows[0]["Tile ID"] == "ctr_generated_123"


@pytest.mark.unit
def test_large_report_type_group_uses_compact_selector() -> None:
    tile_options = [
        (
            "overview",
            "executive",
            f"kpi_{index}",
            {
                "id": f"kpi_{index}",
                "title": f"KPI {index}",
                "metric": "CTR",
                "chart": "kpi_card",
                "value": "CTR",
            },
        )
        for index in range(config_builder.REPORT_LIBRARY_PILLS_MAX + 1)
    ]

    def app(options: list[tuple[str, str, str, dict]]) -> None:
        from valuestream.config import model as config_model  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import (  # noqa: PLC0415
            _render_report_library_chart_group,
            _tile_option_key,
        )

        catalog = config_model.Catalog.model_validate(
            {
                "pipelines": {"version": 1, "workspace": "labels", "sources": []},
                "processors": {"processors": []},
                "metrics": {"metrics": {}},
                "dashboards": {
                    "dashboards": [{"id": "overview", "title": "Overview", "pages": []}]
                },
            }
        )
        _render_report_library_chart_group(
            catalog,
            "kpi_card",
            options,
            selected_tile_key=_tile_option_key(options[0]),
            theme_base="light",
        )

    rendered = AppTest.from_function(app, kwargs={"options": tile_options}).run()

    assert not rendered.exception
    assert not rendered.pills
    assert rendered.selectbox[0].label == "Open KPI card report"


@pytest.mark.unit
def test_stable_catalog_id_is_readable_deterministic_and_collision_safe() -> None:
    dashboard_id = builder.stable_catalog_id("Sales Overview", fallback="dashboard")
    page_id = builder.stable_catalog_id(
        "Engagement",
        fallback="page",
        parent_id=dashboard_id,
    )
    first = builder.stable_catalog_id(
        "CTR Trend",
        fallback="tile",
        parent_id=page_id,
    )
    second = builder.stable_catalog_id(
        "CTR Trend",
        fallback="tile",
        parent_id=page_id,
        existing_ids=[first],
    )

    assert dashboard_id == "dashboard_sales_overview"
    assert page_id == "dashboard_sales_overview_page_engagement"
    assert first.endswith("_tile_ctr_trend")
    assert second == f"{first}_2"


@pytest.mark.unit
def test_stable_catalog_id_preserves_valid_preferred_id_when_title_changes() -> None:
    assert (
        builder.stable_catalog_id(
            "Renamed page",
            fallback="page",
            parent_id="dashboard_sales",
            preferred_id="original_page_id",
        )
        == "original_page_id"
    )


@pytest.mark.unit
def test_metric_mode_options_make_creation_and_editing_explicit() -> None:
    assert config_builder._metric_mode_options(["CTR"]) == [
        "Create Metric",
        "Edit Existing Metric",
    ]
    assert config_builder._metric_mode_options([]) == ["Create Metric"]


@pytest.mark.unit
def test_pending_metric_refresh_opens_written_metric_after_catalog_reload() -> None:
    state: dict[str, object] = {}
    metric_def = {
        "source": "engagement",
        "kind": "approx_distinct_count",
        "state": "Channel_cpc",
    }

    config_builder._queue_metric_refresh(
        state,
        metric_id="UniqueChannels",
        metric_def=metric_def,
        message="Metric written.",
        issues=[],
    )
    feedback = config_builder._consume_pending_metric_refresh(
        state,
        {"UniqueChannels": metric_def},
    )

    assert feedback["metric_id"] == "UniqueChannels"
    assert "builder_metric_pending_refresh" not in state
    assert state["builder_metric_mode"] == "Edit Existing Metric"
    assert state["builder_metric_processor_edit"] == "engagement"
    assert state["builder_metric_kind_edit_engagement"] == "approx_distinct_count"
    assert state["builder_metric_select_engagement_approx_distinct_count"] == "UniqueChannels"


@pytest.mark.unit
def test_metric_filter_helpers_scope_metrics_by_source_and_kind(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)
    metric_defs = {
        "CTR": {"source": "engagement", "kind": "formula"},
        "Dropoff": {"source": "engagement", "kind": "formula"},
        "Reach": {"source": "unknown_profile", "kind": "approx_distinct_count"},
    }

    assert [
        processor.id
        for processor in config_builder._metric_processors_for_definitions(
            catalog.processors.processors, metric_defs
        )
    ] == ["engagement"]
    assert config_builder._metric_kinds_for_source(metric_defs, "engagement") == ["formula"]
    assert config_builder._metric_names_for_source_kind(metric_defs, "engagement", "formula") == [
        "CTR",
        "Dropoff",
    ]


@pytest.mark.unit
def test_metric_choice_label_leads_with_human_name_then_kind_and_source(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    builder.write_metric_definition(
        tmp_path,
        "Dropoff",
        builder.build_formula_metric("engagement", "Count", "Count"),
    )
    catalog = load(tmp_path)

    assert (
        config_builder._metric_choice_label(catalog, "Dropoff")
        == "Dropoff · Formula / state passthrough · Engagement"
    )


@pytest.mark.unit
def test_artifact_identity_formatters_switch_context_without_losing_identity() -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "description": "Interaction history. Additional implementation detail is omitted.",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
        }
    )
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "description": "Customer engagement",
        }
    )

    assert config_builder._source_choice_label_edit(source) == (
        "ih — Interaction history · Parquet"
    )
    assert config_builder._source_choice_label_human(source) == (
        "Interaction history — ih · Parquet"
    )
    assert config_builder._processor_choice_label_edit(processor) == (
        "engagement — Customer engagement · Binary outcome"
    )
    assert config_builder._processor_choice_label_human(processor) == (
        "Customer engagement — engagement · Binary outcome"
    )
    assert config_builder._processor_choice_label("engagement", {"engagement": processor}) == (
        "Customer engagement — engagement · Binary outcome"
    )


@pytest.mark.unit
def test_new_processor_template_uses_selected_source_identity_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        model.Source.model_validate(
            {
                "id": "interactions",
                "reader": {"kind": "parquet", "file_pattern": "interactions/*.parquet"},
                "schema": {"natural_key": ["InteractionID"]},
            }
        ),
        model.Source.model_validate(
            {
                "id": "accounts",
                "reader": {"kind": "parquet", "file_pattern": "accounts/*.parquet"},
                "schema": {"natural_key": ["AccountKey"]},
            }
        ),
    ]
    ctx = SimpleNamespace(
        workspace=Path("."),
        catalog=SimpleNamespace(
            pipelines=SimpleNamespace(sources=sources, defaults=model.WorkspaceDefaults()),
            processors=SimpleNamespace(processors=[]),
        ),
    )
    monkeypatch.setattr(
        config_builder,
        "_source_sample_columns",
        lambda _ctx, source, **_kwargs: [source.schema_.natural_key[0], "Outcome"],
    )

    processor = config_builder._new_processor_template(ctx, "accounts")
    processor_def = builder.processor_to_dict(processor)

    assert processor.source == "accounts"
    assert processor.id == "accounts_processor"
    assert processor_def["entities"] == {"subject": "AccountKey"}
    assert "SubjectID" not in yaml.safe_dump(processor_def)


@pytest.mark.unit
def test_new_processor_template_uses_observed_identity_or_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = model.Source.model_validate(
        {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "events/*.parquet"},
            "schema": {"natural_key": []},
        }
    )
    ctx = SimpleNamespace(
        workspace=Path("."),
        catalog=SimpleNamespace(
            pipelines=SimpleNamespace(sources=[source], defaults=model.WorkspaceDefaults()),
            processors=SimpleNamespace(processors=[]),
        ),
    )

    monkeypatch.setattr(
        config_builder,
        "_source_sample_columns",
        lambda *_args, **_kwargs: ["Outcome", "CustomerToken"],
    )
    observed = builder.processor_to_dict(config_builder._new_processor_template(ctx, "events"))
    assert observed["entities"] == {"subject": "CustomerToken"}

    monkeypatch.setattr(
        config_builder,
        "_source_sample_columns",
        lambda *_args, **_kwargs: ["Outcome", "Channel"],
    )
    empty = builder.processor_to_dict(config_builder._new_processor_template(ctx, "events"))
    assert "entities" not in empty
    assert "SubjectID" not in yaml.safe_dump(empty)


@pytest.mark.unit
def test_new_processor_template_seeds_workspace_common_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = model.Source.model_validate(
        {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "events/*.parquet"},
            "schema": {"natural_key": []},
        }
    )
    ctx = SimpleNamespace(
        workspace=Path("."),
        catalog=SimpleNamespace(
            pipelines=SimpleNamespace(
                sources=[source],
                defaults=model.WorkspaceDefaults(
                    dimensions=["channel", "Issue", "NotInThisSource"]
                ),
            ),
            processors=SimpleNamespace(processors=[]),
        ),
    )
    monkeypatch.setattr(
        config_builder,
        "_source_sample_columns",
        lambda *_args, **_kwargs: ["Channel", "Issue", "Outcome"],
    )

    template = config_builder._new_processor_template(ctx, "events")

    # Case-insensitive match, spelled as the source provides it; dimensions
    # the source cannot provide are skipped instead of breaking the template.
    assert list(template.group_by) == ["Channel", "Issue"]


@pytest.mark.unit
def test_new_processor_template_defaults_to_day_and_month_grains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timed_source = model.Source.model_validate(
        {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "events/*.parquet"},
            "schema": {"timestamp_column": "OutcomeTime"},
        }
    )
    untimed_source = model.Source.model_validate(
        {
            "id": "snapshots",
            "reader": {"kind": "parquet", "file_pattern": "snapshots/*.parquet"},
        }
    )
    ctx = SimpleNamespace(
        workspace=Path("."),
        catalog=SimpleNamespace(
            pipelines=SimpleNamespace(
                sources=[timed_source, untimed_source],
                defaults=model.WorkspaceDefaults(),
            ),
            processors=SimpleNamespace(processors=[]),
        ),
    )
    monkeypatch.setattr(
        config_builder,
        "_source_sample_columns",
        lambda *_args, **_kwargs: ["Outcome"],
    )

    timed = config_builder._new_processor_template(ctx, "events")
    untimed = config_builder._new_processor_template(ctx, "snapshots")

    assert [builder.display_grain(grain) for grain in timed.grains] == ["Day", "Month"]
    assert [builder.display_grain(grain) for grain in untimed.grains] == ["Summary"]


@pytest.mark.unit
def test_processor_kind_guide_covers_every_kind() -> None:
    for kind in forms.PROCESSOR_KIND_OPTIONS:
        guide = forms.processor_kind_guide(kind)
        assert guide is not None, f"missing guide for {kind}"
        assert guide.summary.strip()
        assert guide.purposes.strip()
        assert guide.example_kpis
    assert forms.processor_kind_guide("unknown") is None


def _render_processors_create_step(workspace: str) -> None:
    import streamlit as st  # noqa: PLC0415 - isolated AppTest source

    from valuestream.ui.context import load_context  # noqa: PLC0415
    from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

    st.session_state.setdefault("builder_step", "Processors")
    st.session_state.setdefault("builder_processor_mode", "Create New Processor")
    _builder_steps(load_context(workspace))


@pytest.mark.unit
def test_kind_switch_reseeds_description_and_auto_outputs(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    rendered = AppTest.from_function(
        _render_processors_create_step, kwargs={"workspace": str(tmp_path)}
    ).run()
    assert not rendered.exception

    description = next(item for item in rendered.text_input if item.label == "Description")
    assert description.value == forms.PROCESSOR_KIND_GUIDE["binary_outcome"].summary
    assert any("Example KPIs:" in str(item.value) for item in rendered.caption)
    # The editor mirrors the engine's default outputs, including the
    # unique-subject sketch inferred from the natural key.
    state_rows = rendered.session_state["builder_proc_states_ih_processor"]
    assert [row["State"] for row in state_rows] == [
        "Count",
        "Positives",
        "Negatives",
        "UniqueSubjects_cpc",
    ]
    assert any(item.label == "Dedup Keys" for item in rendered.multiselect)

    kind = next(item for item in rendered.selectbox if item.label == "Kind")
    rendered = kind.set_value("entity_set").run()

    assert not rendered.exception
    description = next(item for item in rendered.text_input if item.label == "Description")
    assert description.value == forms.PROCESSOR_KIND_GUIDE["entity_set"].summary
    state_rows = rendered.session_state["builder_proc_states_ih_processor"]
    assert [row["State"] for row in state_rows] == ["ActiveUsers_cpc", "ActiveUsers_theta"]
    assert all(row["Source Column"] == "InteractionID" for row in state_rows)
    assert any(item.label == "Primary Entity Column" for item in rendered.selectbox)
    assert not any(item.label == "Dedup Keys" for item in rendered.multiselect)


@pytest.mark.unit
def test_kind_state_reseed_preserves_a_disabled_default_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = model.Processors.model_validate(
        {
            "processors": [
                {
                    "id": "engagement",
                    "source": "ih",
                    "kind": "binary_outcome",
                }
            ]
        }
    ).processors[0]
    state_key = "builder_proc_states_engagement"
    defaults = config_builder._state_rows(processor)
    current = copy.deepcopy(defaults)
    next(row for row in current if row["State"] == "Count")["Enabled"] = False
    state = {
        state_key: current,
        f"{state_key}_kind_defaults": defaults,
        f"{state_key}_kind_signature": "previous-kind",
    }
    monkeypatch.setattr(config_builder.st, "session_state", state)

    config_builder._reseed_state_rows_for_kind(
        processor,
        state_key,
        processor.id,
        {},
    )

    count_row = next(row for row in state[state_key] if row["State"] == "Count")
    assert count_row["Enabled"] is False


@pytest.mark.unit
def test_kind_switch_preserves_user_description_and_shows_lifecycle_keys(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)

    rendered = AppTest.from_function(
        _render_processors_create_step, kwargs={"workspace": str(tmp_path)}
    ).run()
    description = next(item for item in rendered.text_input if item.label == "Description")
    rendered = description.set_value("My custom processor").run()

    kind = next(item for item in rendered.selectbox if item.label == "Kind")
    rendered = kind.set_value("entity_lifecycle").run()

    assert not rendered.exception
    description = next(item for item in rendered.text_input if item.label == "Description")
    assert description.value == "My custom processor"
    selectbox_labels = {item.label for item in rendered.selectbox}
    assert {
        "Customer ID Column",
        "Order ID Column",
        "Monetary Column",
        "Purchase Date Column",
    } <= selectbox_labels


@pytest.mark.unit
def test_applicable_dimensions_matches_fields_case_insensitively() -> None:
    assert dimension_profile.applicable_dimensions(
        ["channel", "ISSUE", "Issue", "Missing"],
        ["Channel", "Issue", "Outcome"],
    ) == ["Channel", "Issue"]
    assert dimension_profile.applicable_dimensions([], ["Channel"]) == []


@pytest.mark.unit
def test_write_workspace_dimensions_round_trip(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    builder.write_workspace_dimensions(tmp_path, ["Channel", " Issue ", "Channel", ""])
    catalog = load(tmp_path)
    assert catalog.pipelines.defaults.dimensions == ["Channel", "Issue"]
    assert builder.workspace_dimension_defaults(catalog) == ["Channel", "Issue"]

    # Clearing the selection removes the key so untouched files stay minimal.
    builder.write_workspace_dimensions(tmp_path, [])
    assert load(tmp_path).pipelines.defaults.dimensions == []
    raw = yaml.safe_load((tmp_path / "catalog" / "pipelines.yaml").read_text(encoding="utf-8"))
    assert "defaults" not in raw


def _render_dimensions_step(workspace: str) -> None:
    import streamlit as st  # noqa: PLC0415 - isolated AppTest source

    from valuestream.ui.context import load_context  # noqa: PLC0415
    from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

    st.session_state.setdefault("builder_step", "Dimensions")
    _builder_steps(load_context(workspace))


@pytest.mark.unit
def test_dimensions_step_applies_workspace_dimensions_without_data_run(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    rendered = AppTest.from_function(
        _render_dimensions_step, kwargs={"workspace": str(tmp_path)}
    ).run()
    assert not rendered.exception
    common = next(item for item in rendered.multiselect if item.label == "Common Dimensions")
    # Without explicit defaults the list is restored from the processors'
    # shared group-by, so the step opens pre-populated and clean.
    assert common.value == ["Channel"]
    rendered = common.set_value(["Channel", "Region"]).run()

    # Region is not an ih column and Channel is already engagement's
    # group-by, so nothing recomputes.
    extend = next(
        item for item in rendered.toggle if item.label.startswith("Extend existing processors")
    )
    assert extend.disabled
    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    catalog = load(tmp_path)
    assert catalog.pipelines.defaults.dimensions == ["Channel", "Region"]
    engagement = next(p for p in catalog.processors.processors if p.id == "engagement")
    assert list(engagement.group_by) == ["Channel"]
    outcome = rendered.session_state[config_builder.BUILDER_LAST_OUTCOME_KEY]
    assert outcome["label"] == "Common dimensions"
    assert outcome["action"] != "run_data"


@pytest.mark.unit
def test_dimensions_step_extends_existing_processors_on_apply(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    rendered = AppTest.from_function(
        _render_dimensions_step, kwargs={"workspace": str(tmp_path)}
    ).run()
    common = next(item for item in rendered.multiselect if item.label == "Common Dimensions")
    rendered = common.set_value(["Channel", "InteractionID"]).run()

    extend = next(
        item for item in rendered.toggle if item.label.startswith("Extend existing processors")
    )
    assert not extend.disabled
    rendered = extend.set_value(True).run()
    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    catalog = load(tmp_path)
    assert catalog.pipelines.defaults.dimensions == ["Channel", "InteractionID"]
    engagement = next(p for p in catalog.processors.processors if p.id == "engagement")
    assert list(engagement.group_by) == ["Channel", "InteractionID"]
    outcome = rendered.session_state[config_builder.BUILDER_LAST_OUTCOME_KEY]
    assert outcome["action"] == "run_data"
    assert list(outcome["source_ids"]) == ["ih"]


@pytest.mark.unit
def test_dimensions_step_counts_variant_column_as_covered(tmp_path: Path) -> None:
    """A common dimension persisted as variant_column must not be re-added.

    Validation rejects the variant column inside group_by, so the automatic
    extension counting it as missing produced an apply that always rolled back.
    """
    _write_builder_catalog(tmp_path)
    processors_path = tmp_path / "catalog" / "processors.yaml"
    processors_def = yaml.safe_load(processors_path.read_text(encoding="utf-8"))
    processors_def["processors"][0]["variant_column"] = "InteractionID"
    processors_path.write_text(yaml.safe_dump(processors_def, sort_keys=False), encoding="utf-8")

    rendered = AppTest.from_function(
        _render_dimensions_step, kwargs={"workspace": str(tmp_path)}
    ).run()
    common = next(item for item in rendered.multiselect if item.label == "Common Dimensions")
    rendered = common.set_value(["Channel", "InteractionID"]).run()

    assert not rendered.exception
    extend = next(
        item for item in rendered.toggle if item.label.startswith("Extend existing processors")
    )
    assert extend.disabled
    assert any(
        "Every existing processor already covers its applicable common dimensions" in item.value
        for item in rendered.caption
    )
    rendered = (
        next(button for button in rendered.button if button.label == "Apply to workspace")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert not any("validation failed" in str(item.value) for item in rendered.error)
    catalog = load(tmp_path)
    assert catalog.pipelines.defaults.dimensions == ["Channel", "InteractionID"]
    engagement = next(p for p in catalog.processors.processors if p.id == "engagement")
    assert list(engagement.group_by) == ["Channel"]


@pytest.mark.unit
def test_write_metric_definition_round_trips_valid_catalog(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    builder.write_metric_definition(
        tmp_path,
        "ClickShare",
        builder.build_formula_metric("engagement", "Positives", "Count"),
    )

    ok, issues = builder.validate_workspace(tmp_path)
    assert ok, issues
    catalog = load(tmp_path)
    assert "ClickShare" in catalog.metrics.metrics


@pytest.mark.unit
def test_write_tile_definition_replaces_existing_tile(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    tile = builder.build_tile(
        tile_id="ctr_line",
        title="CTR Line",
        metric_name="CTR",
        chart_kind="line",
        fields={"x": "Day", "y": "CTR", "color": "Channel"},
    )

    builder.write_tile_definition(
        tmp_path,
        dashboard_id="builder_overview",
        dashboard_title="Builder Overview",
        page_id="engagement",
        page_title="Engagement",
        tile=tile,
    )
    builder.write_tile_definition(
        tmp_path,
        dashboard_id="builder_overview",
        dashboard_title="Builder Overview",
        page_id="engagement",
        page_title="Engagement",
        tile={**tile, "title": "CTR Updated"},
    )

    ok, issues = builder.validate_workspace(tmp_path)
    assert ok, issues
    catalog = load(tmp_path)
    tiles = catalog.dashboards.dashboards[0].pages[0].tiles
    assert len(tiles) == 1
    assert tiles[0].title == "CTR Updated"


@pytest.mark.unit
def test_write_tile_definition_updates_dashboard_layout_and_container_titles(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)
    tile = builder.build_tile(
        tile_id="ctr_line",
        title="CTR Line",
        metric_name="CTR",
        chart_kind="line",
        fields={"x": "Day", "y": "CTR", "color": "Channel"},
    )
    builder.write_tile_definition(
        tmp_path,
        dashboard_id="builder_overview",
        dashboard_title="Original dashboard",
        dashboard_layout="tabs",
        page_id="engagement",
        page_title="Original page",
        tile=tile,
    )

    builder.write_tile_definition(
        tmp_path,
        dashboard_id="builder_overview",
        dashboard_title="Updated dashboard",
        dashboard_layout="stacked",
        page_id="engagement",
        page_title="Updated page",
        tile=tile,
    )

    dashboard = load(tmp_path).dashboards.dashboards[0]
    assert dashboard.title == "Updated dashboard"
    assert dashboard.layout == "stacked"
    assert dashboard.pages[0].title == "Updated page"


@pytest.mark.unit
def test_validate_report_candidate_checks_complete_catalog_and_metric_fields(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)
    valid_tile = builder.build_tile(
        tile_id="ctr_line",
        title="CTR Line",
        metric_name="CTR",
        chart_kind="line",
        fields={"x": "Day", "y": "CTR", "color": "Channel"},
    )

    ok, issues = builder.validate_report_candidate(
        catalog,
        dashboard_id="builder_overview",
        dashboard_title="Builder Overview",
        dashboard_layout="grid",
        page_id="engagement",
        page_title="Engagement",
        filters=[],
        time_filter={"default": "all_time", "presets": ["all_time"]},
        tile=valid_tile,
    )
    assert ok, issues

    ok, issues = builder.validate_report_candidate(
        catalog,
        dashboard_id="builder_overview",
        dashboard_title="Builder Overview",
        dashboard_layout="grid",
        page_id="engagement",
        page_title="Engagement",
        filters=[],
        time_filter={"default": "all_time", "presets": ["all_time"]},
        tile={**valid_tile, "y": "CTRR", "time_range": {"last": "30d"}},
    )
    assert not ok
    assert any("unsupported tile field 'time_range'" in issue for issue in issues)
    assert any("field 'CTRR' is not exposed by metric 'CTR'" in issue for issue in issues)


@pytest.mark.unit
def test_every_demo_report_is_a_valid_editor_candidate() -> None:
    workspace = Path(__file__).resolve().parents[2] / "examples" / "demo"
    catalog = load(workspace)

    for dashboard in catalog.dashboards.dashboards:
        for page in dashboard.pages:
            for tile in page.tiles:
                ok, issues = builder.validate_report_candidate(
                    catalog,
                    dashboard_id=dashboard.id,
                    dashboard_title=dashboard.title,
                    dashboard_layout=dashboard.layout,
                    page_id=page.id,
                    page_title=page.title,
                    filters=[
                        item.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for item in page.filters
                    ],
                    time_filter=page.time_filter.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                    tile=tile.model_dump(mode="json", by_alias=True, exclude_none=True),
                )
                assert ok, f"{dashboard.id}/{page.id}/{tile.id}: {issues}"


@pytest.mark.unit
def test_page_settings_writer_preserves_theme_layout_and_tiles(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    builder.write_tile_definition(
        tmp_path,
        dashboard_id="builder_overview",
        dashboard_title="Builder Overview",
        page_id="engagement",
        page_title="Engagement",
        tile=builder.build_tile(
            tile_id="ctr",
            title="CTR",
            metric_name="CTR",
            chart_kind="line",
            fields={"x": "Day", "y": "CTR", "color": "Channel"},
        ),
    )
    dashboard_path = tmp_path / "catalog" / "dashboards.yaml"
    dashboards = yaml.safe_load(dashboard_path.read_text())
    dashboards["theme"] = {"category_colors": {"Channel": {"Web": "#2563EB"}}}
    dashboards["dashboards"][0]["layout"] = "grid"
    dashboard_path.write_text(yaml.safe_dump(dashboards, sort_keys=False))

    builder.write_page_settings(
        tmp_path,
        dashboard_id="builder_overview",
        dashboard_title="Builder Overview",
        page_id="engagement",
        page_title="Engagement",
        filters=[
            {
                "field": "Channel",
                "label": "Channel",
                "display": "primary",
                "scope": "all_tiles",
                "control": "multiselect",
            }
        ],
        time_filter={
            "default": "all_time",
            "presets": ["last_30_days", "all_time"],
        },
    )

    catalog = load(tmp_path)
    dashboard = catalog.dashboards.dashboards[0]
    page = dashboard.pages[0]
    assert catalog.dashboards.theme["category_colors"]["Channel"]["Web"] == "#2563EB"
    assert dashboard.layout == "grid"
    assert page.tiles[0].id == "ctr"
    assert page.filters[0].field == "Channel"
    assert page.time_filter.presets == ["last_30_days", "all_time"]


@pytest.mark.unit
def test_full_dashboard_writer_validates_before_replacing_file(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    dashboard_path = tmp_path / "catalog" / "dashboards.yaml"
    before = dashboard_path.read_text()

    with pytest.raises(ValueError, match="Field required"):
        builder.write_dashboards_definition(
            tmp_path,
            {"theme": {}, "dashboards": [{"id": "broken"}]},
        )

    assert dashboard_path.read_text() == before


@pytest.mark.unit
def test_ensure_minimum_workspace_creates_missing_workspace_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "Fresh Workspace"

    created = builder.ensure_minimum_workspace(workspace)

    assert created == workspace
    assert sorted(path.name for path in (workspace / "catalog").iterdir()) == sorted(
        builder.MINIMUM_CATALOG_FILES
    )
    catalog = load(workspace)
    assert catalog.pipelines.workspace == "fresh_workspace"
    assert catalog.pipelines.sources == []
    ok, issues = builder.validate_workspace(workspace)
    assert ok, issues


@pytest.mark.unit
def test_write_definitions_bootstrap_nonexistent_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "New Studio Workspace"

    builder.write_source_definition(
        workspace,
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "schema": {
                "timestamp_column": "OutcomeTime",
                "natural_key": ["InteractionID"],
            },
        },
    )
    builder.write_processor_definition(
        workspace,
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count"},
            },
        },
    )
    builder.write_metric_definition(
        workspace,
        "CTR",
        builder.build_formula_metric("engagement", "Positives", "Count"),
    )
    builder.write_tile_definition(
        workspace,
        dashboard_id="studio_overview",
        dashboard_title="Studio Overview",
        page_id="engagement",
        page_title="Engagement",
        tile=builder.build_tile(
            tile_id="ctr_line",
            title="CTR",
            metric_name="CTR",
            chart_kind="line",
            fields={"x": "Day", "y": "CTR", "color": "Channel"},
        ),
    )

    ok, issues = builder.validate_workspace(workspace)
    assert ok, issues
    catalog = load(workspace)
    assert catalog.pipelines.workspace == "new_studio_workspace"
    assert catalog.pipelines.sources[0].id == "ih"
    assert catalog.dashboards.dashboards[0].pages[0].tiles[0].metric == "CTR"


@pytest.mark.unit
def test_validate_workspace_accepts_observed_data_only_columns(tmp_path: Path) -> None:
    workspace = tmp_path / "observed_columns_workspace"
    builder.write_source_definition(
        workspace,
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "schema": {
                "timestamp_column": "OutcomeTime",
                "natural_key": ["CustomerID"],
            },
            "transforms": [
                {
                    "kind": "filter",
                    "expression": {"op": "eq", "column": "ActionContext", "value": "Web"},
                }
            ],
        },
    )

    ok, issues = builder.validate_workspace(workspace)
    assert not ok
    assert any("'ActionContext' not found in schema" in issue for issue in issues)

    ok, issues = builder.validate_workspace(
        workspace,
        source_columns_by_id={"ih": ["ActionContext", "CustomerID", "OutcomeTime"]},
    )
    assert ok, issues


@pytest.mark.unit
def test_require_valid_workspace_reports_only_blocking_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_builder_catalog(tmp_path)
    monkeypatch.setattr(
        builder,
        "validate_catalog",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=False,
            issues=[
                SimpleNamespace(
                    location="sources[ih].transforms[0]",
                    message="runtime-only expression warning",
                    severity="warning",
                ),
                SimpleNamespace(
                    location="dashboards[overview]",
                    message="blocking report error",
                    severity="error",
                ),
            ],
        ),
    )

    with pytest.raises(ValueError, match="blocking report error") as captured:
        builder.require_valid_workspace(tmp_path)

    assert "blocking report error" in str(captured.value)
    assert "runtime-only expression warning" not in str(captured.value)


@pytest.mark.unit
def test_write_processor_definition_replaces_in_place_without_reordering(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    for processor_id in ("first", "second", "third"):
        builder.write_processor_definition(
            workspace,
            {"id": processor_id, "source": "ih", "kind": "binary_outcome"},
        )

    builder.write_processor_definition(
        workspace,
        {"id": "second", "source": "ih", "kind": "binary_outcome", "group_by": ["Channel"]},
    )

    raw = yaml.safe_load((workspace / "catalog" / "processors.yaml").read_text())
    assert [item["id"] for item in raw["processors"]] == ["first", "second", "third"]
    assert raw["processors"][1]["group_by"] == ["Channel"]


@pytest.mark.unit
def test_catalog_transaction_restores_all_files_when_a_write_fails(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    builder.write_metric_definition(
        workspace,
        "CTR",
        builder.build_formula_metric("engagement", "Positives", "Count"),
    )
    before = {
        name: (workspace / "catalog" / name).read_text() for name in builder.CATALOG_FILENAMES
    }

    def _failing_two_file_install() -> None:
        builder.write_processor_definition(
            workspace,
            {"id": "engagement", "source": "ih", "kind": "binary_outcome"},
        )
        builder.write_metric_definition(
            workspace,
            "Reach",
            {"source": "engagement", "kind": "approx_distinct_count", "state": "Reach_cpc"},
        )
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"), builder.catalog_transaction(workspace):
        _failing_two_file_install()

    after = {name: (workspace / "catalog" / name).read_text() for name in builder.CATALOG_FILENAMES}
    assert after == before


@pytest.mark.unit
def test_validated_catalog_transaction_rolls_back_all_post_write_changes(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)
    metrics_path = tmp_path / "catalog" / "metrics.yaml"
    metrics_path.write_bytes(metrics_path.read_bytes().replace(b"\n", b"\r\n"))
    before = {
        name: (tmp_path / "catalog" / name).read_bytes() for name in builder.CATALOG_FILENAMES
    }

    def install_invalid_metric() -> None:
        with builder.validated_catalog_transaction(tmp_path):
            builder.write_workspace_settings(
                tmp_path,
                workspace_name="should_roll_back",
                time_zone="Europe/Berlin",
                calendar_grains=["Day", "Summary"],
                week_start="sunday",
                dashboard_theme={"colorway": ["#ff0000"]},
            )
            builder.write_metric_definition(
                tmp_path,
                "BrokenReach",
                {
                    "source": "engagement",
                    "kind": "approx_distinct_count",
                    "state": "Missing_theta",
                },
            )

    with pytest.raises(ValueError, match="changes were rolled back"):
        install_invalid_metric()

    after = {name: (tmp_path / "catalog" / name).read_bytes() for name in builder.CATALOG_FILENAMES}
    assert after == before


@pytest.mark.unit
def test_configuration_builder_never_runs_data_from_apply() -> None:
    source = Path(config_builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "run_source" not in called_names
    assert "run_source" not in imported_names
    assert "Save & Run" not in source
    assert "& Run Source" not in source
    assert "Create & Run" not in source


@pytest.mark.unit
def test_tile_deletion_is_staged_behind_the_step_apply_action() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    tile_builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tile_builder"
    )
    guarded_delete_calls = 0

    class DeleteGuardVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.apply_depth = 0

        def visit_If(self, node: ast.If) -> None:
            guarded = (
                isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Name)
                and node.test.func.id == "_render_editor_primary_action"
            )
            self.apply_depth += int(guarded)
            self.generic_visit(node)
            self.apply_depth -= int(guarded)

        def visit_Call(self, node: ast.Call) -> None:
            nonlocal guarded_delete_calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "delete_tile_definition":
                assert self.apply_depth > 0
                guarded_delete_calls += 1
            self.generic_visit(node)

    DeleteGuardVisitor().visit(tile_builder)

    assert guarded_delete_calls == 1


@pytest.mark.unit
def test_sketch_helper_does_not_preselect_topk_or_avoid_fields() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    sketch_panel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sketch_states_panel"
    )
    calls = [node for node in ast.walk(sketch_panel) if isinstance(node, ast.Call)]
    topk_checkbox = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "checkbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Top frequent values"
    )
    topk_selector = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Top-K Field"
    )
    checkbox_default = next(
        keyword.value for keyword in topk_checkbox.keywords if keyword.arg == "value"
    )
    selector_index = next(
        keyword.value for keyword in topk_selector.keywords if keyword.arg == "index"
    )

    assert isinstance(checkbox_default, ast.Constant)
    assert checkbox_default.value is False
    assert isinstance(selector_index, ast.IfExp)
    assert isinstance(selector_index.orelse, ast.Constant)
    assert selector_index.orelse.value is None


@pytest.mark.unit
def test_configuration_builder_catalog_mutations_use_validated_transaction() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    catalog_mutators = {
        "delete_tile_definition",
        "write_dashboards_definition",
        "write_metric_definition",
        "write_metrics_definition",
        "write_page_settings",
        "write_pipelines_definition",
        "write_processor_definition",
        "write_processors_definition",
        "write_source_definition",
        "write_tile_definition",
        "write_workspace_settings",
    }
    violations: list[tuple[str, int]] = []

    class CatalogMutationVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.validated_transaction_depth = 0

        def visit_With(self, node: ast.With) -> None:
            guarded = any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "builder"
                and item.context_expr.func.attr == "validated_catalog_transaction"
                for item in node.items
            )
            self.validated_transaction_depth += int(guarded)
            self.generic_visit(node)
            self.validated_transaction_depth -= int(guarded)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "builder"
                and func.attr in catalog_mutators
                and self.validated_transaction_depth == 0
            ):
                violations.append((func.attr, node.lineno))
            self.generic_visit(node)

    CatalogMutationVisitor().visit(tree)

    assert violations == []


@pytest.mark.unit
def test_write_workspace_settings_updates_catalog_defaults_and_theme(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    builder.write_workspace_settings(
        tmp_path,
        workspace_name="review_workspace",
        time_zone="Europe/Berlin",
        calendar_grains=["Day", "Month", "Summary"],
        week_start="sunday",
        dashboard_theme={"colorway": ["#0055aa"], "font": {"family": "Inter"}},
    )

    catalog = load(tmp_path)
    assert catalog.pipelines.workspace == "review_workspace"
    assert catalog.pipelines.defaults.time_zone == "Europe/Berlin"
    assert catalog.pipelines.defaults.calendar.grains == ["Day", "Month", "Summary"]
    assert catalog.pipelines.defaults.calendar.week_start == "sunday"
    assert catalog.dashboards.theme == {"colorway": ["#0055aa"], "font": {"family": "Inter"}}
    ok, issues = builder.validate_workspace(tmp_path)
    assert ok, issues


@pytest.mark.unit
def test_settings_editor_keeps_an_explicit_empty_calendar_draft_clean(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    pipelines_path = tmp_path / "catalog" / "pipelines.yaml"
    pipelines = yaml.safe_load(pipelines_path.read_text(encoding="utf-8"))
    pipelines["defaults"] = {
        "time_zone": "UTC",
        "calendar": {"grains": [], "week_start": "monday"},
    }
    pipelines_path.write_text(yaml.safe_dump(pipelines, sort_keys=False), encoding="utf-8")

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Settings").run()

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert any("Select at least one calendar grain" in item.value for item in rendered.warning)


@pytest.mark.unit
def test_chat_metric_rows_include_processor_dimensions(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    rows = config_builder._chat_metric_rows(catalog)

    assert rows == [
        {
            "Metric": "CTR",
            "Kind": "formula",
            "Processor": "engagement",
            "Group By": "Channel",
        }
    ]


@pytest.mark.unit
def test_chat_description_rows_preserve_catalog_and_custom_keys(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    catalog = load(tmp_path)

    rows = config_builder._chat_description_rows(
        [(processor.id, "Processor") for processor in catalog.processors.processors],
        {"engagement": "Catalog processor description.", "legacy_family": "Legacy label."},
    )
    description_map = config_builder._chat_description_map(rows)

    assert rows == [
        {
            "Type": "Processor",
            "Key": "engagement",
            "Description": "Catalog processor description.",
        },
        {"Type": "Custom", "Key": "legacy_family", "Description": "Legacy label."},
    ]
    assert description_map == {
        "engagement": "Catalog processor description.",
        "legacy_family": "Legacy label.",
    }


@pytest.mark.unit
def test_chat_description_rows_preserve_case_variant_custom_keys() -> None:
    rows = config_builder._chat_description_rows(
        [("sample", "Dataset")],
        {"SAMPLE": "Configured legacy spelling."},
    )

    assert config_builder._chat_description_map(rows) == {"SAMPLE": "Configured legacy spelling."}


@pytest.mark.unit
def test_compile_filter_rows_builds_expression_ast() -> None:
    expression = builder.compile_filter_rows(
        [
            {"Field": "Outcome", "Operator": "in", "Value": "Clicked, Conversion", "Enabled": True},
            {"Field": "Channel", "Operator": "contains", "Value": "Web", "Enabled": True},
            {"Field": "Ignored", "Operator": "==", "Value": "x", "Enabled": False},
        ]
    )

    assert expression == {
        "op": "and",
        "args": [
            {"op": "in", "column": "Outcome", "values": ["Clicked", "Conversion"]},
            {"op": "matches", "column": "Channel", "pattern": "Web"},
        ],
    }


@pytest.mark.unit
def test_compile_condition_rows_supports_any_combinator() -> None:
    rows = [
        {"Field": "Channel", "Operator": "==", "Value": "Web", "Enabled": True},
        {"Field": "Channel", "Operator": "==", "Value": "Mobile", "Enabled": True},
    ]

    assert builder.compile_condition_rows(rows, combine="any") == {
        "op": "or",
        "args": [
            {"op": "eq", "column": "Channel", "value": "Web"},
            {"op": "eq", "column": "Channel", "value": "Mobile"},
        ],
    }
    assert builder.compile_condition_rows(rows[:1], combine="any") == {
        "op": "eq",
        "column": "Channel",
        "value": "Web",
    }
    assert builder.compile_condition_rows([], combine="any") is None


@pytest.mark.unit
def test_compile_case_expression_builds_validated_case_ast() -> None:
    expression = builder.compile_case_expression(
        [
            {
                "conditions": [
                    {"Field": "Revenue", "Operator": ">", "Value": "100", "Enabled": True},
                    {"Field": "Channel", "Operator": "==", "Value": "Web", "Enabled": True},
                ],
                "combine": "all",
                "then_kind": "Literal",
                "then_value": "High",
            },
            {
                "conditions": [
                    {"Field": "Revenue", "Operator": ">", "Value": "10", "Enabled": True},
                ],
                "combine": "any",
                "then_kind": "Field",
                "then_value": "Segment",
            },
        ],
        else_kind="Literal",
        else_value="Standard",
    )

    assert expression == {
        "op": "case",
        "when": [
            {
                "cond": {
                    "op": "and",
                    "args": [
                        {"op": "gt", "column": "Revenue", "value": 100},
                        {"op": "eq", "column": "Channel", "value": "Web"},
                    ],
                },
                "then": {"lit": "High"},
            },
            {
                "cond": {"op": "gt", "column": "Revenue", "value": 10},
                "then": {"col": "Segment"},
            },
        ],
        "else": {"lit": "Standard"},
    }
    validation = builder.validate_calculated_expression(
        "AST YAML", builder.expression_yaml(expression)
    )
    assert validation.valid


@pytest.mark.unit
def test_compile_case_expression_reports_branch_level_problems() -> None:
    complete_row = {"Field": "Revenue", "Operator": ">", "Value": "1", "Enabled": True}

    with pytest.raises(ValueError, match="branch 1 needs at least one complete condition row"):
        builder.compile_case_expression(
            [{"conditions": [], "combine": "all", "then_kind": "Literal", "then_value": "x"}],
            else_kind="Literal",
            else_value="",
        )
    with pytest.raises(ValueError, match="branch 1: a field name is required"):
        builder.compile_case_expression(
            [
                {
                    "conditions": [complete_row],
                    "combine": "all",
                    "then_kind": "Field",
                    "then_value": "",
                }
            ],
            else_kind="Literal",
            else_value="",
        )
    with pytest.raises(ValueError, match="else value: a field name is required"):
        builder.compile_case_expression(
            [
                {
                    "conditions": [complete_row],
                    "combine": "all",
                    "then_kind": "Literal",
                    "then_value": "x",
                }
            ],
            else_kind="Field",
            else_value="",
        )
    with pytest.raises(ValueError, match="add at least one branch"):
        builder.compile_case_expression([], else_kind="Literal", else_value="")


@pytest.mark.unit
def test_build_derive_column_transforms_validates_expression_yaml() -> None:
    transforms = builder.build_derive_column_transforms(
        [
            {
                "Name": "ResponseTime",
                "Expression YAML": (
                    "op: date_diff\n"
                    "unit: seconds\n"
                    "end: {col: OutcomeTime}\n"
                    "start: {col: DecisionTime}\n"
                ),
                "Enabled": True,
            }
        ]
    )

    assert transforms == [
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
    ]


@pytest.mark.unit
def test_build_derive_column_transforms_supports_direct_polars_expression() -> None:
    transforms = builder.build_derive_column_transforms(
        [
            {
                "Name": "Margin",
                "Mode": "Polars",
                "Expression": 'pl.col("Revenue") - pl.col("Cost")',
                "Enabled": True,
            }
        ]
    )

    assert transforms == [
        {
            "kind": "derive_column",
            "output": "Margin",
            "expression": {"polars": 'pl.col("Revenue") - pl.col("Cost")'},
        }
    ]


@pytest.mark.unit
def test_build_derive_column_transforms_supports_builder_rows() -> None:
    transforms = builder.build_derive_column_transforms(
        [
            {
                "Name": "Margin",
                "Mode": "Subtract",
                "Left": "Revenue",
                "Right Kind": "Field",
                "Right": "Cost",
                "Enabled": True,
            }
        ]
    )

    assert transforms == [
        {
            "kind": "derive_column",
            "output": "Margin",
            "expression": {
                "op": "sub",
                "args": [{"col": "Revenue"}, {"col": "Cost"}],
            },
        }
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        (float("nan"), True),
        (True, True),
        (False, False),
        ("true", True),
        ("False", False),
        (1, True),
        (0, False),
    ],
)
def test_editor_row_enabled_strictly_normalizes_grid_values(value: object, expected: bool) -> None:
    assert builder.editor_row_enabled(value) is expected


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [None, ""])
def test_grid_added_rows_with_missing_enabled_values_are_included(enabled: object) -> None:
    transforms = builder.build_derive_column_transforms(
        [
            {
                "Name": "Margin",
                "Mode": "AST YAML",
                "Expression": "op: sub\nargs:\n  - {col: Revenue}\n  - {col: Cost}",
                "Enabled": enabled,
            }
        ]
    )

    assert transforms == [
        {
            "kind": "derive_column",
            "output": "Margin",
            "expression": {
                "op": "sub",
                "args": [{"col": "Revenue"}, {"col": "Cost"}],
            },
        }
    ]
    assert builder.build_default_values(
        [{"Field": "Channel", "Default Value": "Unknown", "Enabled": enabled}]
    ) == {"Channel": "Unknown"}
    assert builder.compile_filter_rows(
        [{"Field": "Channel", "Operator": "==", "Value": "Web", "Enabled": enabled}]
    ) == {"op": "eq", "column": "Channel", "value": "Web"}


@pytest.mark.unit
def test_calculated_expression_validation_translates_case_repro_errors() -> None:
    result = builder.validate_calculated_expression(
        "AST YAML",
        """op: case
when:
  - cond: {column: Revenue, value: 100}
    then: {lit: High}
otherwise: {lit: Standard}
""",
    )

    assert not result.valid
    assert result.messages == (
        "`when[0].cond` must be a condition such as `{op: gt, column: Revenue, value: 100}`.",
        "`else` is required for a `case` expression.",
        "`otherwise` is not supported for `case`; use `else:` instead.",
    )
    assert "union_tag_not_found" in result.technical_details
    assert "extra_forbidden" in result.technical_details
    assert all("union_tag" not in message for message in result.messages)


@pytest.mark.unit
def test_calculated_expression_validation_covers_yaml_and_guarded_polars() -> None:
    valid_yaml = builder.validate_calculated_expression(
        "AST YAML",
        builder.calculated_expression_example("AST YAML"),
    )
    invalid_yaml = builder.validate_calculated_expression(
        "AST YAML",
        "op: case\nwhen: [",
    )
    valid_polars = builder.validate_calculated_expression(
        "Polars",
        'pl.col("Revenue") - pl.col("Cost")',
    )
    invalid_polars = builder.validate_calculated_expression(
        "Polars",
        "foo + 1",
    )

    assert valid_yaml.valid
    assert valid_polars.valid
    assert not invalid_yaml.valid
    assert invalid_yaml.messages[0].startswith("Expression YAML is not valid near line")
    assert invalid_yaml.technical_details
    assert not invalid_polars.valid
    assert invalid_polars.messages == (
        "Only the `pl` namespace is available in a Polars expression.",
    )
    assert "unsupported name" in invalid_polars.technical_details


@pytest.mark.unit
def test_calculated_expression_editor_requires_explicit_apply_and_cancel() -> None:
    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui import builder  # noqa: PLC0415
        from valuestream.ui.pages import config_builder as page  # noqa: PLC0415

        calc_key = "builder_source_calcs_test"
        editor_key = "builder_source_calcs_editor_test"
        if calc_key not in st.session_state:
            st.session_state[calc_key] = [
                {
                    "Name": "RevenueBand",
                    "Mode": "AST YAML",
                    "Left": "",
                    "Right Kind": "Field",
                    "Right": "",
                    "Expression": "col: Revenue",
                    "Enabled": True,
                }
            ]
        frame = builder.editor_frame(
            st.session_state[calc_key],
            ["Name", "Enabled", "Mode", "Expression", "Left", "Right Kind", "Right"],
            builder.blank_calculated_row,
        )
        page._render_calculated_rows_editor(calc_key, editor_key, frame, ["Revenue", "Cost"])

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    assert rendered.text_area[0].label == "AST YAML expression direct editor"
    assert rendered.text_area[0].value == "col: Revenue"
    assert {button.label for button in rendered.button} == {
        "Cancel changes",
        "Apply expression",
        "Generate expression",
    }

    invalid = (
        "op: case\n"
        "when:\n"
        "  - cond: {column: Revenue, value: 100}\n"
        "    then: {lit: High}\n"
        "otherwise: {lit: Standard}"
    )
    rendered = rendered.text_area[0].set_value(invalid).run()

    assert not rendered.exception
    assert rendered.session_state["builder_source_calcs_test"][0]["Expression"] == "col: Revenue"
    assert rendered.session_state["builder_source_calcs_test_expression_pending"] is True
    assert any("Working expression is not applied yet" in item.value for item in rendered.warning)
    assert {
        "`when[0].cond` must be a condition such as `{op: gt, column: Revenue, value: 100}`.",
        "`else` is required for a `case` expression.",
        "`otherwise` is not supported for `case`; use `else:` instead.",
    }.issubset({item.value for item in rendered.error})
    apply_button = next(button for button in rendered.button if button.label == "Apply expression")
    assert apply_button.disabled
    assert any("union_tag_not_found" in item.value for item in rendered.code)

    valid = builder.calculated_expression_example("AST YAML")
    rendered = rendered.text_area[0].set_value(valid).run()
    apply_button = next(button for button in rendered.button if button.label == "Apply expression")
    assert not apply_button.disabled
    rendered = apply_button.click().run()

    assert not rendered.exception
    assert rendered.session_state["builder_source_calcs_test"][0]["Expression"] == valid
    assert rendered.session_state["builder_source_calcs_test_expression_pending"] is False
    assert any(
        "Expression applied to the calculated row" in item.value for item in rendered.success
    )

    rendered = rendered.text_area[0].set_value("col: Cost").run()
    rendered = (
        next(button for button in rendered.button if button.label == "Cancel changes").click().run()
    )

    assert not rendered.exception
    assert rendered.text_area[0].value == valid
    assert rendered.session_state["builder_source_calcs_test"][0]["Expression"] == valid
    assert rendered.session_state["builder_source_calcs_test_expression_pending"] is False


@pytest.mark.unit
def test_pending_expression_blocks_source_apply_until_explicit_expression_apply(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        if "qa_expression_seeded" not in st.session_state:
            st.session_state["qa_expression_seeded"] = True
            st.session_state["builder_source_calcs_ih"] = [
                {
                    "Name": "RevenueCopy",
                    "Mode": "AST YAML",
                    "Left": "",
                    "Right Kind": "Field",
                    "Right": "",
                    "Expression": "col: Revenue",
                    "Enabled": True,
                }
            ]
            st.session_state["builder_source_calcs_editor_ih_expression_draft_0"] = "col: Cost"
        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Sources").run()

    assert not rendered.exception
    workspace_apply = next(
        button for button in rendered.button if button.label == "Apply to workspace"
    )
    assert workspace_apply.disabled
    assert any("Working expression is not applied yet" in item.value for item in rendered.warning)

    expression_apply = next(
        button for button in rendered.button if button.label == "Apply expression"
    )
    assert not expression_apply.disabled
    rendered = expression_apply.click().run()

    assert not rendered.exception
    workspace_apply = next(
        button for button in rendered.button if button.label == "Apply to workspace"
    )
    assert not workspace_apply.disabled
    assert rendered.session_state["builder_source_calcs_ih"][0]["Expression"] == "col: Cost"


@pytest.mark.unit
def test_visual_case_builder_generates_yaml_into_focused_editor(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)
    visual_base = "builder_source_calcs_editor_ih_expression_draft_0_visual"

    def app(workspace: str, base: str) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        if "qa_visual_seeded" not in st.session_state:
            st.session_state["qa_visual_seeded"] = True
            st.session_state["builder_source_calcs_ih"] = [
                {
                    "Name": "RevenueBand",
                    "Mode": "AST YAML",
                    "Left": "",
                    "Right Kind": "Field",
                    "Right": "",
                    "Expression": "col: Revenue",
                    "Enabled": True,
                }
            ]
            st.session_state[f"{base}_b0_rows"] = [
                {"Field": "Revenue", "Operator": ">", "Value": "100", "Enabled": True}
            ]
            st.session_state[f"{base}_b0_then_value"] = "High"
            st.session_state[f"{base}_else_value"] = "Standard"
        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(
        app, kwargs={"workspace": str(tmp_path), "base": visual_base}
    ).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Sources").run()

    assert not rendered.exception
    generate = next(button for button in rendered.button if button.label == "Generate expression")
    rendered = generate.click().run()

    assert not rendered.exception
    expected = builder.expression_yaml(
        builder.compile_case_expression(
            [
                {
                    "conditions": [
                        {"Field": "Revenue", "Operator": ">", "Value": "100", "Enabled": True}
                    ],
                    "combine": "all",
                    "then_kind": "Literal",
                    "then_value": "High",
                }
            ],
            else_kind="Literal",
            else_value="Standard",
        )
    )
    editor = next(
        item for item in rendered.text_area if item.label == "AST YAML expression direct editor"
    )
    assert editor.value == expected
    assert any("Generated expression inserted" in item.value for item in rendered.success)

    expression_apply = next(
        button for button in rendered.button if button.label == "Apply expression"
    )
    assert not expression_apply.disabled
    rendered = expression_apply.click().run()

    assert not rendered.exception
    assert rendered.session_state["builder_source_calcs_ih"][0]["Expression"] == expected


@pytest.mark.unit
def test_all_grid_row_enabled_checkboxes_default_true() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    function_names = {
        "_render_default_values_editor",
        "_render_filter_rows_editor",
        "_render_calculated_rows_editor",
        "_render_state_rows_editor",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]

    assert {function.name for function in functions} == function_names
    for function in functions:
        checkbox_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "CheckboxColumn"
        ]
        assert len(checkbox_calls) == 1
        default = next(
            keyword.value for keyword in checkbox_calls[0].keywords if keyword.arg == "default"
        )
        assert isinstance(default, ast.Constant)
        assert default.value is True


@pytest.mark.unit
def test_calculated_rows_from_source_hydrates_polars_mode() -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
            "transforms": [
                {
                    "kind": "derive_column",
                    "output": "Margin",
                    "expression": {"polars": 'pl.col("Revenue") - pl.col("Cost")'},
                }
            ],
        }
    )

    rows = builder.calculated_rows_from_source(source)

    assert rows == [
        {
            "Name": "Margin",
            "Mode": "Polars",
            "Left": "",
            "Right Kind": "Field",
            "Right": "",
            "Expression": 'pl.col("Revenue") - pl.col("Cost")',
            "Enabled": True,
        }
    ]


@pytest.mark.unit
def test_calculated_rows_for_editor_accepts_legacy_yaml_rows() -> None:
    rows = builder.calculated_rows_for_editor(
        [{"Name": "ResponseTime", "Expression YAML": "col: OutcomeTime", "Enabled": True}]
    )

    assert rows == [
        {
            "Name": "ResponseTime",
            "Mode": "AST YAML",
            "Left": "",
            "Right Kind": "Field",
            "Right": "",
            "Expression": "col: OutcomeTime",
            "Enabled": True,
        }
    ]


@pytest.mark.unit
def test_write_source_and_processor_definitions_round_trip(tmp_path: Path) -> None:
    _write_builder_catalog(tmp_path)

    builder.write_source_definition(
        tmp_path,
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet", "streaming": True},
            "schema": {
                "timestamp_column": "OutcomeTime",
                "natural_key": ["InteractionID"],
                "drop_columns": [],
            },
            "defaults": {"Channel": "Unknown"},
            "transforms": [
                {
                    "kind": "filter",
                    "expression": {"op": "not_null", "column": "Channel"},
                },
                {
                    "kind": "derive_column",
                    "output": "ResponseTime",
                    "expression": {
                        "op": "date_diff",
                        "unit": "seconds",
                        "end": {"col": "OutcomeTime"},
                        "start": {"col": "OutcomeTime"},
                    },
                },
            ],
        },
    )
    builder.write_processor_definition(
        tmp_path,
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel", "ResponseTime"],
            "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
            "states": {
                "Count": {"type": "count"},
                "Positives": {"type": "count"},
                "Negatives": {"type": "count"},
            },
            "filter": {"op": "not_null", "column": "Channel"},
        },
    )

    ok, issues = builder.validate_workspace(tmp_path)
    assert ok, issues
    catalog = load(tmp_path)
    assert catalog.pipelines.sources[0].defaults == {"Channel": "Unknown"}
    assert catalog.processors.processors[0].group_by == ["Channel", "ResponseTime"]


@pytest.mark.unit
def test_delete_source_cascade_removes_catalog_and_chat_dependencies(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)
    catalog = load(tmp_path)

    plan = builder.source_cascade_plan(catalog, "holdings")

    assert plan.processor_ids == ("holdings_lifecycle",)
    assert plan.metric_ids == ("HoldingsRate", "HoldingsValue")
    assert plan.tile_locations == (
        "overview/portfolio/holdings_rate",
        "overview/portfolio/holdings_value",
    )
    assert plan.page_filter_locations == ("overview/portfolio/HoldingType",)

    deleted = builder.delete_source_cascade(tmp_path, "holdings")

    assert deleted == plan
    remaining = load(tmp_path)
    assert [source.id for source in remaining.pipelines.sources] == ["ih"]
    assert [processor.id for processor in remaining.processors.processors] == ["engagement"]
    assert list(remaining.metrics.metrics) == ["CTR"]
    page = remaining.dashboards.dashboards[0].pages[0]
    assert [tile.id for tile in page.tiles] == ["ctr"]
    assert [filter_spec.field for filter_spec in page.filters] == ["Channel"]
    ai_config = yaml.safe_load((tmp_path / "ai.yaml").read_text())
    assert ai_config["ai"]["llm"]["model"] == "test-model"
    assert ai_config["chat_with_data"]["dataset_descriptions"] == {"ih": "Interactions"}
    assert ai_config["chat_with_data"]["metric_descriptions"] == {
        "CTR": "Engagement rate",
        "engagement": "Interaction processor",
    }


@pytest.mark.unit
def test_delete_source_cascade_rolls_back_every_configuration_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_source_cascade_catalog(tmp_path)
    paths = [
        *(tmp_path / "catalog" / name for name in builder.CATALOG_FILENAMES),
        tmp_path / "ai.yaml",
    ]
    before = {path: path.read_text(encoding="utf-8") for path in paths}

    def reject_deleted_catalog(_workspace: str | Path) -> None:
        raise ValueError("post-delete validation failed")

    monkeypatch.setattr(builder, "require_valid_workspace", reject_deleted_catalog)

    with pytest.raises(ValueError, match="post-delete validation failed"):
        builder.delete_source_cascade(tmp_path, "holdings")

    assert {path: path.read_text(encoding="utf-8") for path in paths} == before


@pytest.mark.unit
def test_processor_cascade_plan_is_transitive_and_names_exact_report_paths(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)

    plan = builder.processor_cascade_plan(load(tmp_path), "holdings_lifecycle")

    assert plan.processor_id == "holdings_lifecycle"
    assert plan.metric_ids == ("HoldingsRate", "HoldingsValue")
    assert plan.tile_locations == (
        "overview/portfolio/holdings_rate",
        "overview/portfolio/holdings_value",
    )
    assert plan.page_filter_locations == ("overview/portfolio/HoldingType",)


@pytest.mark.unit
def test_rename_processor_retargets_metrics_and_chat_description(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)
    processor = next(
        item for item in load(tmp_path).processors.processors if item.id == "engagement"
    )
    definition = builder.processor_to_dict(processor)
    definition["id"] = "engagement_v2"

    builder.rename_processor_definition(tmp_path, "engagement", definition)

    renamed = load(tmp_path)
    assert [item.id for item in renamed.processors.processors] == [
        "engagement_v2",
        "holdings_lifecycle",
    ]
    assert renamed.metrics.metrics["CTR"].source == "engagement_v2"
    ai_config = yaml.safe_load((tmp_path / "ai.yaml").read_text(encoding="utf-8"))
    descriptions = ai_config["chat_with_data"]["metric_descriptions"]
    assert descriptions["engagement_v2"] == "Interaction processor"
    assert "engagement" not in descriptions


@pytest.mark.unit
def test_delete_processor_cascade_removes_only_named_target_and_keeps_aggregates(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    aggregate = tmp_path / "aggregates" / "holdings" / "holdings_lifecycle" / "part.parquet"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"persisted aggregate")

    deleted = builder.delete_processor_cascade(tmp_path, "holdings_lifecycle")

    assert deleted.metric_ids == ("HoldingsRate", "HoldingsValue")
    remaining = load(tmp_path)
    assert [source.id for source in remaining.pipelines.sources] == ["ih", "holdings"]
    assert [processor.id for processor in remaining.processors.processors] == ["engagement"]
    assert list(remaining.metrics.metrics) == ["CTR"]
    page = remaining.dashboards.dashboards[0].pages[0]
    assert [tile.id for tile in page.tiles] == ["ctr"]
    assert [filter_spec.field for filter_spec in page.filters] == ["Channel"]
    assert aggregate.read_bytes() == b"persisted aggregate"
    ai_config = yaml.safe_load((tmp_path / "ai.yaml").read_text(encoding="utf-8"))
    assert ai_config["chat_with_data"]["dataset_descriptions"] == {
        "ih": "Interactions",
        "holdings": "Product holdings",
    }
    assert ai_config["chat_with_data"]["metric_descriptions"] == {
        "engagement": "Interaction processor",
        "CTR": "Engagement rate",
    }
    ok, issues = builder.validate_workspace(tmp_path)
    assert ok, issues


@pytest.mark.unit
def test_delete_processor_without_dependencies_is_valid(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)
    builder.write_processor_definition(
        tmp_path,
        {
            "id": "unused_processor",
            "source": "ih",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "states": {"Count": {"type": "count"}},
        },
    )
    builder.require_valid_workspace(tmp_path)

    plan = builder.delete_processor_cascade(tmp_path, "unused_processor")

    assert plan.metric_ids == ()
    assert plan.tile_locations == ()
    assert "unused_processor" not in {
        processor.id for processor in load(tmp_path).processors.processors
    }


@pytest.mark.unit
def test_delete_processor_cascade_rolls_back_on_post_write_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    paths = [
        *(tmp_path / "catalog" / name for name in builder.CATALOG_FILENAMES),
        tmp_path / "ai.yaml",
    ]
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(
        builder,
        "require_valid_workspace",
        lambda _workspace, **_kwargs: (_ for _ in ()).throw(
            ValueError("processor validation failed")
        ),
    )

    with pytest.raises(ValueError, match="processor validation failed"):
        builder.delete_processor_cascade(tmp_path, "holdings_lifecycle")

    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.unit
def test_metric_delete_plan_blocks_dependent_metrics_and_names_target_tiles(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    catalog = load(tmp_path)

    blocked = builder.metric_delete_plan(catalog, "HoldingsValue")
    exact = builder.metric_delete_plan(catalog, "CTR")

    assert blocked.dependent_metric_ids == ("HoldingsRate",)
    assert blocked.tile_locations == ("overview/portfolio/holdings_value",)
    assert exact.dependent_metric_ids == ()
    assert exact.tile_locations == ("overview/portfolio/ctr",)
    assert exact.page_filter_locations == ("overview/portfolio/Channel",)
    with pytest.raises(ValueError, match=r"dependent metric.*HoldingsRate"):
        builder.delete_metric_definition(tmp_path, "HoldingsValue", cascade_tiles=True)
    assert list(load(tmp_path).metrics.metrics) == ["CTR", "HoldingsValue", "HoldingsRate"]


@pytest.mark.unit
def test_metric_delete_requires_explicit_tile_cascade_then_deletes_exact_target(
    tmp_path: Path,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    before = {
        path: path.read_bytes()
        for path in [
            tmp_path / "catalog" / "metrics.yaml",
            tmp_path / "catalog" / "dashboards.yaml",
            tmp_path / "ai.yaml",
        ]
    }

    with pytest.raises(ValueError, match="choose the tile cascade explicitly"):
        builder.delete_metric_definition(tmp_path, "CTR", cascade_tiles=False)
    assert {path: path.read_bytes() for path in before} == before

    deleted = builder.delete_metric_definition(tmp_path, "CTR", cascade_tiles=True)

    assert deleted.metric_id == "CTR"
    remaining = load(tmp_path)
    assert list(remaining.metrics.metrics) == ["HoldingsValue", "HoldingsRate"]
    assert [processor.id for processor in remaining.processors.processors] == [
        "engagement",
        "holdings_lifecycle",
    ]
    page = remaining.dashboards.dashboards[0].pages[0]
    assert [tile.id for tile in page.tiles] == ["holdings_value", "holdings_rate"]
    assert [filter_spec.field for filter_spec in page.filters] == ["HoldingType"]
    ai_config = yaml.safe_load((tmp_path / "ai.yaml").read_text(encoding="utf-8"))
    assert "CTR" not in ai_config["chat_with_data"]["metric_descriptions"]
    assert "engagement" in ai_config["chat_with_data"]["metric_descriptions"]
    ok, issues = builder.validate_workspace(tmp_path)
    assert ok, issues


@pytest.mark.unit
def test_delete_metric_without_report_dependencies_is_valid(tmp_path: Path) -> None:
    _write_source_cascade_catalog(tmp_path)
    builder.write_metric_definition(
        tmp_path,
        "Unused",
        {
            "source": "engagement",
            "kind": "formula",
            "expression": {"col": "Count"},
            "display": {"label": "Unused QA metric"},
        },
    )
    builder.require_valid_workspace(tmp_path)

    plan = builder.delete_metric_definition(tmp_path, "Unused", cascade_tiles=False)

    assert plan.tile_locations == ()
    assert "Unused" not in load(tmp_path).metrics.metrics


@pytest.mark.unit
def test_delete_metric_rolls_back_catalog_and_chat_on_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_source_cascade_catalog(tmp_path)
    paths = [
        *(tmp_path / "catalog" / name for name in builder.CATALOG_FILENAMES),
        tmp_path / "ai.yaml",
    ]
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(
        builder,
        "require_valid_workspace",
        lambda _workspace, **_kwargs: (_ for _ in ()).throw(
            ValueError("metric validation failed")
        ),
    )

    with pytest.raises(ValueError, match="metric validation failed"):
        builder.delete_metric_definition(tmp_path, "CTR", cascade_tiles=True)

    assert {path: path.read_bytes() for path in paths} == before


def _write_builder_catalog(workspace: Path) -> None:
    catalog = workspace / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: builder_test
sources:
  - id: ih
    reader:
      kind: parquet
      file_pattern: "data/*.parquet"
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID]
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    group_by: [Channel]
    time:
      column: OutcomeTime
      grains: [Day, Summary]
    states:
      Count: {type: count}
      Positives: {type: count}
      Negatives: {type: count}
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        """
metrics:
  CTR:
    source: engagement
    kind: formula
    expression:
      op: safe_div
      num: {col: Positives}
      den: {col: Count}
""",
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


def _write_source_cascade_catalog(workspace: Path) -> None:
    catalog = workspace / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: cascade_test
sources:
  - id: ih
    reader: {kind: parquet, file_pattern: "ih/*.parquet"}
  - id: holdings
    reader: {kind: parquet, file_pattern: "holdings/*.parquet"}
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    group_by: [Channel]
    states:
      Count: {type: count}
  - id: holdings_lifecycle
    source: holdings
    kind: binary_outcome
    group_by: [HoldingType]
    states:
      Count: {type: count}
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        """
metrics:
  CTR:
    source: engagement
    kind: formula
    expression: {col: Count}
  HoldingsValue:
    source: holdings_lifecycle
    kind: formula
    expression: {col: Count}
  HoldingsRate:
    source: holdings_lifecycle
    kind: formula
    depends_on: [HoldingsValue]
    expression: {col: HoldingsValue}
""",
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text(
        """
dashboards:
  - id: overview
    title: Overview
    pages:
      - id: portfolio
        title: Portfolio
        filters:
          - {field: Channel, scope: compatible_tiles}
          - {field: HoldingType, scope: compatible_tiles}
        tiles:
          - {id: ctr, title: CTR, metric: CTR, chart: kpi_card, value: CTR}
          - {id: holdings_value, title: Holdings value, metric: HoldingsValue, chart: kpi_card, value: HoldingsValue}
          - {id: holdings_rate, title: Holdings rate, metric: HoldingsRate, chart: kpi_card, value: HoldingsRate}
""",
        encoding="utf-8",
    )
    (workspace / "ai.yaml").write_text(
        """
ai:
  llm:
    model: test-model
chat_with_data:
  agent_prompt: Test prompt
  dataset_descriptions:
    ih: Interactions
    holdings: Product holdings
  metric_descriptions:
    engagement: Interaction processor
    holdings_lifecycle: Holdings processor
    CTR: Engagement rate
    HoldingsValue: Holdings value
    HoldingsRate: Holdings rate
""",
        encoding="utf-8",
    )
    builder.require_valid_workspace(workspace)


def _numeric_distribution_catalog() -> model.Catalog:
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "test",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "descriptive",
                        "source": "ih",
                        "kind": "numeric_distribution",
                        "group_by": ["Channel"],
                        "time": {"grains": ["Month", "Summary"]},
                        "properties": ["Propensity", "ResponseTime"],
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "ResponseP50": {
                        "source": "descriptive",
                        "kind": "tdigest_quantile",
                        "state": "ResponseTime_tdigest",
                        "quantile": 0.5,
                    }
                }
            },
            "dashboards": {"dashboards": []},
        }
    )


@pytest.mark.unit
def test_builder_continue_escalates_to_full_app_rerun() -> None:
    """Continue is rendered from inside step fragments.

    An ``on_click`` callback would advance session state but only rerun the
    fragment, so the page would never move. Assigning the instantiated Jump
    widget key inline raises StreamlitAPIException. The inline handler must
    set only the plain step key and escalate with ``st.rerun(scope="app")``.
    """

    import inspect  # noqa: PLC0415 - focused source guard
    import textwrap  # noqa: PLC0415 - focused source guard

    source = textwrap.dedent(inspect.getsource(config_builder._render_continue_primary))
    tree = ast.parse(source)

    button_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "button"
    ]
    assert button_calls
    for call in button_calls:
        assert all(keyword.arg != "on_click" for keyword in call.keywords)

    rerun_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rerun"
    ]
    assert any(
        any(
            keyword.arg == "scope" and getattr(keyword.value, "value", None) == "app"
            for keyword in call.keywords
        )
        for call in rerun_calls
    )

    assigned_keys = [
        target.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
    ]
    assert "builder_step_jump" not in assigned_keys


# ---------------------------------------------------------------------------
# AST reverse mapping: visual case state, simple-mode recognition, between.
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

_CUSTOMER_TYPE_YAML = """
op: when_then
cond:
  op: starts_with
  column: CustomerID
  value: C
then:
  lit: Customers known
else:
  lit: Device/Anonymous
"""


@pytest.mark.unit
def test_visual_case_state_decompiles_when_then_as_single_branch_case() -> None:
    assert builder.visual_case_state_from_expression(_CUSTOMER_TYPE_YAML) == {
        "shape": "case",
        "branches": [
            {
                "rows": [
                    {
                        "Field": "CustomerID",
                        "Operator": "starts with",
                        "Value": "C",
                        "Enabled": True,
                    }
                ],
                "combine": "all",
                "then_kind": "Literal",
                "then_value": "Customers known",
            }
        ],
        "else_kind": "Literal",
        "else_value": "Device/Anonymous",
    }


@pytest.mark.unit
def test_visual_case_state_round_trips_multi_branch_case() -> None:
    expression = {
        "op": "case",
        "when": [
            {
                "cond": {"op": "ne", "column": "PlacementType", "value": ""},
                "then": {"col": "PlacementType"},
            },
            {
                "cond": {
                    "op": "or",
                    "args": [
                        {"op": "starts_with", "column": "Name", "value": "CR"},
                        {"op": "between", "column": "Rank", "low": 1, "high": 3},
                    ],
                },
                "then": {"lit": "Flex"},
            },
        ],
        "else": {"lit": "Hero"},
    }

    state = builder.visual_case_state_from_expression(builder.expression_yaml(expression))

    assert state is not None
    assert state["shape"] == "case"
    assert [branch["combine"] for branch in state["branches"]] == ["all", "any"]
    regenerated = builder.compile_case_expression(
        [
            {
                "conditions": branch["rows"],
                "combine": branch["combine"],
                "then_kind": branch["then_kind"],
                "then_value": branch["then_value"],
            }
            for branch in state["branches"]
        ],
        else_kind=state["else_kind"],
        else_value=state["else_value"],
    )
    assert regenerated == expression


@pytest.mark.unit
def test_visual_case_state_decompiles_boolean_condition_shape() -> None:
    state = builder.visual_case_state_from_expression(
        "op: or\nargs:\n- {op: eq, column: A, value: 1}\n- {op: is_null, column: B}"
    )

    assert state is not None
    assert state["shape"] == "condition"
    assert state["branches"][0]["combine"] == "any"
    assert [row["Operator"] for row in state["branches"][0]["rows"]] == ["==", "is null"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("empty", ""),
        ("invalid yaml", "op: [unclosed"),
        ("non-conditional", "op: date_diff\nunit: seconds\nend: {col: A}\nstart: {col: B}"),
        ("column copy", "col: Treatment"),
        (
            "nested logic",
            "op: and\nargs:\n- op: or\n  args:\n  - {op: eq, column: A, value: 1}\n"
            "  - {op: eq, column: B, value: 2}\n- {op: eq, column: C, value: 3}",
        ),
        ("args-form comparison", "op: eq\nargs:\n- {col: A}\n- {col: B}"),
        (
            "computed then value",
            "op: when_then\ncond: {op: eq, column: A, value: 1}\n"
            "then: {op: add, args: [{col: B}, {lit: 1}]}\nelse: {lit: 0}",
        ),
        (
            "param result",
            "op: when_then\ncond: {op: eq, column: A, value: 1}\nthen: {param: p}\nelse: {lit: 0}",
        ),
        (
            "lossy literal",
            "op: when_then\ncond: {op: eq, column: A, value: 1}\n"
            "then: {lit: '3.5'}\nelse: {lit: x}",
        ),
    ],
)
def test_visual_case_state_rejects_expressions_beyond_the_builder(label: str, text: str) -> None:
    assert builder.visual_case_state_from_expression(text) is None, label


@pytest.mark.unit
def test_visual_case_state_rejects_more_branches_than_the_builder_offers() -> None:
    expression = {
        "op": "case",
        "when": [
            {"cond": {"op": "eq", "column": "A", "value": index}, "then": {"lit": index}}
            for index in range(builder.VISUAL_CASE_MAX_BRANCHES + 1)
        ],
        "else": {"lit": 0},
    }

    assert builder.visual_case_state_from_expression(builder.expression_yaml(expression)) is None


@pytest.mark.unit
def test_between_operator_compiles_and_reverses() -> None:
    row = {"Field": "Score", "Operator": "between", "Value": "5, 10", "Enabled": True}

    compiled = builder.compile_filter_rows([row])

    assert compiled == {"op": "between", "column": "Score", "low": 5, "high": 10}
    assert builder.filter_rows_from_expression(compiled) == [row]
    assert builder.validate_calculated_expression(
        "AST YAML", builder.expression_yaml(compiled)
    ).valid


@pytest.mark.unit
def test_between_operator_requires_two_values() -> None:
    with pytest.raises(ValueError, match="between on 'Score' needs exactly two"):
        builder.compile_filter_rows(
            [{"Field": "Score", "Operator": "between", "Value": "5", "Enabled": True}]
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "right_kind", "right"),
    [
        ("Copy Field", "Field", ""),
        ("Absolute Value", "Field", ""),
        ("Round", "Field", ""),
        ("Round", "Literal", "2"),
        ("Add", "Field", "Cost"),
        ("Subtract", "Literal", "1.5"),
        ("Multiply", "Field", "Cost"),
        ("Divide", "Literal", "100"),
        ("Safe Divide", "Field", "Cost"),
        ("Concat", "Field", "Cost"),
        ("Coalesce", "Literal", "fallback"),
        ("Date Diff Seconds", "Field", "Start"),
        ("Date Diff Days", "Field", "Start"),
        ("Date Part Year", "Field", ""),
        ("Date Part Weekday", "Field", ""),
    ],
)
def test_calculation_mode_recognition_round_trips(mode: str, right_kind: str, right: str) -> None:
    row = {
        "Name": "Derived",
        "Mode": mode,
        "Left": "Revenue",
        "Right Kind": right_kind,
        "Right": right,
        "Expression": "",
        "Enabled": True,
    }

    transforms = builder.build_derive_column_transforms([row])

    assert len(transforms) == 1
    expression = transforms[0]["expression"]
    expr_parser.parse(expression)
    assert builder.calculation_mode_from_expression(expression) == {
        "Mode": mode,
        "Left": "Revenue",
        "Right Kind": right_kind,
        "Right": right,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "expression"),
    [
        ("three args", {"op": "add", "args": [{"col": "A"}, {"col": "B"}, {"col": "C"}]}),
        (
            "nested operand",
            {"op": "add", "args": [{"op": "abs", "arg": {"col": "A"}}, {"col": "B"}]},
        ),
        (
            "custom concat separator",
            {"op": "concat", "args": [{"col": "A"}, {"col": "B"}], "sep": "-"},
        ),
        ("concat without separator", {"op": "concat", "args": [{"col": "A"}, {"col": "B"}]}),
        ("lossy literal", {"op": "add", "args": [{"col": "A"}, {"lit": "3.5"}]}),
        ("blank literal", {"op": "concat", "args": [{"col": "A"}, {"lit": ""}], "sep": ""}),
        ("explicit null ndigits", {"op": "round", "arg": {"col": "A"}, "ndigits": None}),
        ("boolean ndigits", {"op": "round", "arg": {"col": "A"}, "ndigits": True}),
        ("unary not", {"op": "not", "arg": {"col": "A"}}),
        ("polars", {"polars": "pl.col('A')"}),
    ],
)
def test_calculation_mode_recognition_rejects_inexact_shapes(label: str, expression: dict) -> None:
    assert builder.calculation_mode_from_expression(expression) is None, label


@pytest.mark.unit
@pytest.mark.parametrize("workspace", ["demo", "fat", "new", "test_config_editor"])
def test_example_catalog_calculated_rows_round_trip_identically(workspace: str) -> None:
    catalog = load(_EXAMPLES_DIR / workspace)

    for source in catalog.pipelines.sources:
        expected = [
            {
                "kind": "derive_column",
                "output": transform.output,
                "expression": expr_parser.to_dict(transform.expression),
            }
            for transform in source.transforms
            if isinstance(transform, model.DeriveColumn)
        ]
        rows = builder.calculated_rows_from_source(source)
        assert builder.build_derive_column_transforms(rows) == expected


@pytest.mark.unit
def test_fat_example_catalog_rows_surface_recognized_modes() -> None:
    catalog = load(_EXAMPLES_DIR / "fat")
    rows = {
        row["Name"]: row
        for source in catalog.pipelines.sources
        for row in builder.calculated_rows_from_source(source)
    }

    assert rows["ConversionEventID"]["Mode"] == "Copy Field"
    assert rows["ConversionEventID"]["Left"] == "Treatment"
    assert rows["ResponseTime"]["Mode"] == "Date Diff Seconds"
    assert rows["ResponseTime"]["Left"] == "OutcomeTime"
    assert rows["ResponseTime"]["Right"] == "DecisionTime"
    for name in ("CustomerType", "Placement", "Revenue"):
        assert rows[name]["Mode"] == "AST YAML"
        assert builder.visual_case_state_from_expression(rows[name]["Expression"]) is not None


@pytest.mark.unit
def test_visual_builder_seeds_widgets_from_draft_expression() -> None:
    app = AppTest.from_string(
        '''
import streamlit as st
from valuestream.ui.pages import config_builder

st.session_state.setdefault(
    "draft",
    """op: when_then
cond:
  op: starts_with
  column: CustomerID
  value: C
then:
  lit: Customers known
else:
  lit: Device/Anonymous""",
)
config_builder._render_visual_case_builder(
    draft_key="draft",
    input_key="draft_input",
    notice_key="notice",
    field_options=["CustomerID"],
)
'''
    )

    rendered = app.run(timeout=15)

    assert not rendered.exception
    assert rendered.session_state["draft_visual_b0_rows"] == [
        {"Field": "CustomerID", "Operator": "starts with", "Value": "C", "Enabled": True}
    ]
    assert rendered.session_state["draft_visual_branches"] == 1
    assert rendered.session_state["draft_visual_b0_then_kind"] == "Literal"
    assert rendered.session_state["draft_visual_b0_then_value"] == "Customers known"
    assert rendered.session_state["draft_visual_else_kind"] == "Literal"
    assert rendered.session_state["draft_visual_else_value"] == "Device/Anonymous"
    assert not rendered.info


@pytest.mark.unit
def test_visual_builder_flags_expression_beyond_the_builder() -> None:
    app = AppTest.from_string(
        '''
import streamlit as st
from valuestream.ui.pages import config_builder

st.session_state.setdefault(
    "draft",
    """op: date_diff
unit: seconds
end:
  col: OutcomeTime
start:
  col: DecisionTime""",
)
config_builder._render_visual_case_builder(
    draft_key="draft",
    input_key="draft_input",
    notice_key="notice",
    field_options=["OutcomeTime"],
)
'''
    )

    rendered = app.run(timeout=15)

    assert not rendered.exception
    assert rendered.info
    assert "beyond this builder" in rendered.info[0].value


# ---------------------------------------------------------------------------
# Filter logic formulas: E-references with AND / OR / NOT.
# ---------------------------------------------------------------------------

_FORMULA_ROWS = [
    {"Field": "Channel", "Operator": "==", "Value": "Web", "Enabled": True},
    {"Field": "Region", "Operator": "in", "Value": "EU, US", "Enabled": True},
    {"Field": "Score", "Operator": ">", "Value": "5", "Enabled": True},
]


@pytest.mark.unit
def test_condition_formula_compiles_not_and_or_nesting() -> None:
    compiled = builder.compile_condition_formula(_FORMULA_ROWS, "(NOT(E1) AND E2) OR E3")

    assert compiled == {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [
                    {"op": "not", "arg": {"op": "eq", "column": "Channel", "value": "Web"}},
                    {"op": "in", "column": "Region", "values": ["EU", "US"]},
                ],
            },
            {"op": "gt", "column": "Score", "value": 5},
        ],
    }
    expr_parser.parse(compiled)


@pytest.mark.unit
def test_condition_formula_keywords_and_refs_are_case_insensitive() -> None:
    strict = builder.compile_condition_formula(_FORMULA_ROWS, "NOT E1 AND E2")
    relaxed = builder.compile_condition_formula(_FORMULA_ROWS, "not e1 and e2")

    assert strict == relaxed


@pytest.mark.unit
def test_condition_state_round_trips_advanced_shapes() -> None:
    for formula in ["(NOT E1 AND E2) OR E3", "NOT (E1 OR E2)", "(E1 AND E2) AND E3", "NOT NOT E1"]:
        compiled = builder.compile_condition_formula(_FORMULA_ROWS, formula)
        state = builder.condition_state_from_expression(compiled)
        assert state is not None
        assert state["mode"] == "Advanced"
        recompiled = builder.compile_condition_formula(state["rows"], state["formula"])
        assert recompiled == compiled, formula


@pytest.mark.unit
def test_condition_state_classifies_flat_shapes_as_basic() -> None:
    assert builder.condition_state_from_expression(None) == {
        "rows": [],
        "mode": "Basic",
        "combine": "AND",
        "formula": "",
    }

    single = builder.compile_condition_rows(_FORMULA_ROWS[:1], combine="all")
    single_state = builder.condition_state_from_expression(single)
    assert single_state is not None
    assert (single_state["mode"], single_state["combine"]) == ("Basic", "AND")

    for combine, expected in (("all", "AND"), ("any", "OR")):
        flat = builder.compile_condition_rows(_FORMULA_ROWS, combine=combine)
        state = builder.condition_state_from_expression(flat)
        assert state is not None
        assert (state["mode"], state["combine"]) == ("Basic", expected)
        assert [row["Ref"] for row in state["rows"]] == ["E1", "E2", "E3"]
        # The prefill formula matches the basic combine semantics.
        assert builder.compile_condition_formula(state["rows"], state["formula"]) == flat


@pytest.mark.unit
def test_condition_state_rejects_unmappable_leaves() -> None:
    assert (
        builder.condition_state_from_expression({"op": "eq", "args": [{"col": "A"}, {"col": "B"}]})
        is None
    )
    assert (
        builder.condition_state_from_expression(
            {"op": "not", "arg": {"op": "add", "args": [{"col": "A"}, {"lit": 1}]}}
        )
        is None
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("formula", "rows", "match"),
    [
        ("E1 AND E9", _FORMULA_ROWS, "E9 does not exist"),
        ("E1 AND", _FORMULA_ROWS, "not valid"),
        ("E1 XOR E2", _FORMULA_ROWS, "not valid"),
        ("E1 if E2 else E1", _FORMULA_ROWS, "support only condition references"),
        ("Channel AND E1", _FORMULA_ROWS, "unknown token 'Channel'"),
        ("", _FORMULA_ROWS, "enter a logic formula"),
        (
            "E1 AND E2",
            [_FORMULA_ROWS[0], {**_FORMULA_ROWS[1], "Enabled": False}],
            "E2 is disabled",
        ),
        (
            "E1 AND E2",
            [_FORMULA_ROWS[0], {"Field": "", "Operator": "==", "Value": "", "Enabled": True}],
            "E2 is incomplete",
        ),
    ],
)
def test_condition_formula_reports_actionable_errors(
    formula: str, rows: list[dict], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        builder.compile_condition_formula(rows, formula)


@pytest.mark.unit
def test_label_condition_rows_renumbers_in_order() -> None:
    rows = [dict(row, Ref="stale") for row in _FORMULA_ROWS]

    labeled = builder.label_condition_rows(rows)

    assert [row["Ref"] for row in labeled] == ["E1", "E2", "E3"]
    assert [row["Field"] for row in labeled] == ["Channel", "Region", "Score"]


@pytest.mark.unit
@pytest.mark.parametrize("workspace", ["demo", "fat", "new", "test_config_editor"])
def test_example_catalog_filters_stay_editable_and_round_trip(workspace: str) -> None:
    """Every example filter must decompile to rules and recompile identically.

    The example workspaces are live app data, so the mode is not pinned:
    flat conjunctions come back Basic while formula-authored filters come
    back Advanced — both must round-trip byte-for-byte.
    """
    catalog = load(_EXAMPLES_DIR / workspace)

    subjects = [*catalog.pipelines.sources, *catalog.processors.processors]
    for subject in subjects:
        expression = builder.first_filter_expression(subject)
        if expression is None:
            continue
        state = builder.condition_state_from_expression(expression)
        assert state is not None, f"{workspace}: filter should stay editable as rules"
        if state["mode"] == "Basic":
            combine = "any" if state["combine"] == "OR" else "all"
            recompiled = builder.compile_condition_rows(state["rows"], combine=combine)
        else:
            recompiled = builder.compile_condition_formula(state["rows"], state["formula"])
        assert recompiled == expression


@pytest.mark.unit
def test_filter_editor_seeds_advanced_state_and_compiles_formula() -> None:
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import builder
from valuestream.ui.pages import config_builder

state = builder.condition_state_from_expression(
    {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [
                    {"op": "not", "arg": {"op": "eq", "column": "Channel", "value": "Web"}},
                    {"op": "eq", "column": "Region", "value": "EU"},
                ],
            },
            {"op": "gt", "column": "Score", "value": 5},
        ],
    }
)
rows_key = "test_filter_rows"
config_builder._seed_filter_editor_state(rows_key, state)
filter_frame = builder.editor_frame(
    st.session_state[rows_key],
    ["Ref", "Field", "Operator", "Value", "Enabled"],
    builder.blank_filter_row,
)
config_builder._render_filter_rows_editor(
    rows_key,
    "test_filter_editor",
    filter_frame,
    ["Channel", "Region", "Score"],
)
st.session_state["compiled_out"] = config_builder._compiled_filter_expression(rows_key)
"""
    )

    rendered = app.run(timeout=15)

    assert not rendered.exception
    assert rendered.session_state["test_filter_rows_logic_mode"] == "Advanced"
    assert rendered.session_state["test_filter_rows_formula"] == "(NOT E1 AND E2) OR E3"
    assert [row["Ref"] for row in rendered.session_state["test_filter_rows"]] == [
        "E1",
        "E2",
        "E3",
    ]
    assert rendered.session_state["compiled_out"] == {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [
                    {"op": "not", "arg": {"op": "eq", "column": "Channel", "value": "Web"}},
                    {"op": "eq", "column": "Region", "value": "EU"},
                ],
            },
            {"op": "gt", "column": "Score", "value": 5},
        ],
    }


@pytest.mark.unit
def test_filter_editor_reports_formula_errors_inline() -> None:
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import builder
from valuestream.ui.pages import config_builder

rows_key = "test_filter_rows"
st.session_state[rows_key] = [
    {"Ref": "E1", "Field": "Channel", "Operator": "==", "Value": "Web", "Enabled": True}
]
st.session_state[f"{rows_key}_logic_mode"] = "Advanced"
st.session_state[f"{rows_key}_combine"] = "AND"
st.session_state[f"{rows_key}_formula"] = "E1 AND E7"
filter_frame = builder.editor_frame(
    st.session_state[rows_key],
    ["Ref", "Field", "Operator", "Value", "Enabled"],
    builder.blank_filter_row,
)
config_builder._render_filter_rows_editor(
    rows_key,
    "test_filter_editor",
    filter_frame,
    ["Channel"],
)
"""
    )

    rendered = app.run(timeout=15)

    assert not rendered.exception
    assert rendered.error
    assert "E7 does not exist" in rendered.error[0].value


# ---------------------------------------------------------------------------
# Common dimensions restored from processors.
# ---------------------------------------------------------------------------


def _dimension_catalog(processor_dimensions: list[list[str]]) -> model.Catalog:
    processors = [
        {
            "id": f"p{index}",
            "source": "ih",
            "kind": "binary_outcome",
            "dimensions": dimensions,
            "states": {"Count": {"type": "count"}},
        }
        for index, dimensions in enumerate(processor_dimensions)
    ]
    return model.Catalog.model_validate(
        {
            "pipelines": {
                "version": 1,
                "workspace": "dims",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
                        "schema": {"natural_key": ["InteractionID"]},
                    }
                ],
            },
            "processors": {"processors": processors},
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
        }
    )


@pytest.mark.unit
def test_workspace_dimensions_fall_back_to_shared_processor_group_by() -> None:
    catalog = _dimension_catalog(
        [
            ["Channel", "CustomerType", "Placement", "Outcome"],
            ["Channel", "Placement", "AppliedModel", "CustomerType"],
            ["channel", "CustomerType", "Placement", "Issue"],
        ]
    )

    # Intersection in first-processor order; matching is case-insensitive and
    # the first processor's spelling wins.
    assert builder.workspace_dimension_defaults(catalog) == [
        "Channel",
        "CustomerType",
        "Placement",
    ]


@pytest.mark.unit
def test_workspace_dimensions_fallback_skips_processors_without_group_by() -> None:
    catalog = _dimension_catalog([["Channel", "Issue"], [], ["Issue", "Channel"]])

    assert builder.workspace_dimension_defaults(catalog) == ["Channel", "Issue"]


@pytest.mark.unit
def test_workspace_dimensions_fallback_handles_disjoint_and_missing() -> None:
    assert builder.workspace_dimension_defaults(_dimension_catalog([["Channel"], ["Issue"]])) == []
    assert builder.workspace_dimension_defaults(_dimension_catalog([])) == []


@pytest.mark.unit
def test_workspace_dimensions_explicit_defaults_override_fallback() -> None:
    catalog = _dimension_catalog([["Channel", "Issue"], ["Channel", "Issue"]])
    explicit = catalog.model_copy(
        update={
            "pipelines": catalog.pipelines.model_copy(
                update={
                    "defaults": catalog.pipelines.defaults.model_copy(
                        update={"dimensions": ["Region"]}
                    )
                }
            )
        }
    )

    assert builder.workspace_dimension_defaults(explicit) == ["Region"]


@pytest.mark.unit
def test_fat_example_workspace_resolves_common_dimensions() -> None:
    """The live fat workspace always resolves a non-empty common list.

    The workspace is user-editable app data, so the exact list is not pinned:
    explicit ``defaults.dimensions`` win verbatim, and without them the list
    falls back to the dimensions shared by every processor group-by.
    """
    catalog = load(_EXAMPLES_DIR / "fat")

    resolved = builder.workspace_dimension_defaults(catalog)
    assert resolved
    explicit = [str(field).strip() for field in catalog.pipelines.defaults.dimensions]
    explicit = [field for field in explicit if field]
    if explicit:
        assert resolved == builder.dedupe(explicit)
    else:
        shared = {field.casefold() for field in resolved}
        for processor in catalog.processors.processors:
            group_by = {str(field).casefold() for field in processor.group_by}
            if group_by:
                assert shared <= group_by


_RETAINED_SET_APP = """
import streamlit as st
from valuestream.ui import forms

seed = {
    "source": "audience",
    "kind": "set_op",
    "op": "intersection",
    "output": "count",
    "operands": [
        {"state": "Customers_theta", "time_window": {"last": "1d"}},
        {"state": "Customers_theta", "time_window": {"between": ["-30d", "-1d"]}},
    ],
}
ctx = forms.MetricFormContext(state_options=lambda _types: ["Customers_theta"])
st.session_state["result"] = forms.metric_kind_fields(
    "set_op", seed, ctx, key_prefix="retained_set"
)
"""


@pytest.mark.unit
def test_windowed_set_op_is_editable_with_one_theta_state() -> None:
    """Retention-style metrics intersect one state with itself across windows.

    The editor must render editable operand rows seeded from the metric and
    round-trip the definition unchanged — not the misleading "requires at
    least two theta states" warning, which only applies to state-vs-state
    authoring.
    """
    rendered = AppTest.from_string(_RETAINED_SET_APP).run()

    assert not rendered.exception
    assert not rendered.warning
    assert rendered.session_state["result"] == {
        "op": "intersection",
        "operands": [
            {"state": "Customers_theta", "time_window": {"last": "1d"}},
            {"state": "Customers_theta", "time_window": {"between": ["-30d", "-1d"]}},
        ],
        "output": "count",
    }
    assert rendered.session_state["retained_set_set_operand_0_last"] == "1d"
    assert rendered.session_state["retained_set_set_operand_1_from"] == "-30d"
    assert rendered.session_state["retained_set_set_operand_1_to"] == "-1d"


@pytest.mark.unit
def test_windowed_set_op_edits_write_back_and_validate() -> None:
    rendered = AppTest.from_string(_RETAINED_SET_APP).run()
    last_input = next(
        item for item in rendered.text_input if item.key == "retained_set_set_operand_0_last"
    )
    rendered = last_input.set_value("7d").run()

    assert not rendered.exception
    assert rendered.session_state["result"]["operands"][0] == {
        "state": "Customers_theta",
        "time_window": {"last": "7d"},
    }

    bad = next(
        item for item in rendered.text_input if item.key == "retained_set_set_operand_0_last"
    )
    rendered = bad.set_value("-3d").run()

    assert rendered.session_state["result"] is None
    assert any("positive duration" in item.value for item in rendered.warning)


def _write_entity_set_catalog(workspace: Path) -> None:
    catalog = workspace / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: sketch_test
sources:
  - id: ih
    reader:
      kind: parquet
      file_pattern: "data/*.parquet"
    schema:
      timestamp_column: OutcomeTime
      natural_key: [CustomerID]
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: audience
    source: ih
    kind: entity_set
    description: Audience sets.
    entity: CustomerID
    dimensions: [Channel]
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text("metrics: {}\n", encoding="utf-8")
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


@pytest.mark.unit
def test_sketch_helper_appends_states_to_processor_sketches_grid(tmp_path: Path) -> None:
    _write_entity_set_catalog(tmp_path)

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        _builder_steps(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()
    jump = next(item for item in rendered.selectbox if item.label == "Jump to step")
    rendered = jump.set_value("Processors").run()

    assert not rendered.exception
    rendered_titles = " ".join(item.value for item in rendered.markdown)
    assert "Processor Sketches" in rendered_titles
    assert "Sketch Helper" in rendered_titles
    assert "Top-K And Sketch Exploration" not in rendered_titles

    before = rendered.session_state["builder_proc_states_audience"]
    before_names = {row["State"] for row in before}
    assert {"ActiveUsers_cpc", "ActiveUsers_theta"} <= before_names

    add = next(button for button in rendered.button if button.label == "Add to Processor Sketches")
    rendered = add.click().run(timeout=15)

    assert not rendered.exception
    after = rendered.session_state["builder_proc_states_audience"]
    after_names = [row["State"] for row in after]
    assert after_names[: len(before)] == [row["State"] for row in before]
    assert "UniqueCustomerid_cpc" in after_names
    assert "AudienceCustomerid_theta" in after_names
    added = {row["State"]: row for row in after if row["State"] not in before_names}
    assert added["UniqueCustomerid_cpc"]["Type"] == "cpc"
    assert added["UniqueCustomerid_cpc"]["Source Column"] == "CustomerID"
    assert added["AudienceCustomerid_theta"]["Type"] == "theta"


@pytest.mark.unit
def test_set_op_picker_offers_minus_instead_of_diff_alias() -> None:
    assert "diff" not in forms.SET_OP_OPTIONS
    assert set(forms.SET_OP_LABELS) == set(forms.SET_OP_OPTIONS)
    assert forms.SET_OP_LABELS["a_not_b"].startswith("Minus")


@pytest.mark.unit
def test_set_op_diff_seed_edits_as_minus_and_saves_canonically() -> None:
    """`diff` is an engine alias of `a_not_b`; editing normalizes it."""
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import forms

seed = {
    "source": "audience",
    "kind": "set_op",
    "op": "diff",
    "states": ["A_theta", "B_theta"],
}
ctx = forms.MetricFormContext(state_options=lambda _types: ["A_theta", "B_theta"])
st.session_state["result"] = forms.metric_kind_fields(
    "set_op", seed, ctx, key_prefix="diff_set"
)
"""
    ).run()

    assert not app.exception
    assert app.session_state["result"] == {"op": "a_not_b", "states": ["A_theta", "B_theta"]}


class _PolicyWarningCapture(logging.Handler):
    """Capture Streamlit widget-policy warnings at their source logger.

    Streamlit sets ``propagate = False`` on its loggers, so ``caplog`` (which
    listens on the root logger) sees the records only in some import orders.
    Attaching directly to the emitting logger is deterministic.
    """

    LOGGER_NAME = "streamlit.elements.lib.policies"

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> _PolicyWarningCapture:
        from streamlit.elements.lib import policies  # noqa: PLC0415 - test-only reset

        # Streamlit emits this warning once per process; reset the flag so the
        # capture observes it regardless of which tests ran earlier.
        policies._shown_default_value_warning = False
        logging.getLogger(self.LOGGER_NAME).addHandler(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        logging.getLogger(self.LOGGER_NAME).removeHandler(self)

    @property
    def default_clash_messages(self) -> list[str]:
        return [message for message in self.messages if "created with a default value" in message]


@pytest.mark.unit
def test_streamlit_default_plus_session_state_warning_is_capturable() -> None:
    """Meta-check: the policy warning must be observable, or the test below is vacuous."""
    app_code = """
import streamlit as st

st.session_state.setdefault("clash_key", "B")
st.segmented_control("Clash", ["A", "B"], default="A", key="clash_key")
"""
    with _PolicyWarningCapture() as capture:
        AppTest.from_string(app_code).run()

    assert capture.default_clash_messages


@pytest.mark.unit
def test_processor_mode_seeded_by_post_apply_renders_without_policy_warning(
    tmp_path: Path,
) -> None:
    _write_builder_catalog(tmp_path)

    def app(workspace: str) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages.config_builder import _builder_steps  # noqa: PLC0415

        st.session_state.setdefault("builder_step", "Processors")
        # What _consume_builder_post_apply_cleanup does after a processor apply.
        st.session_state.setdefault("builder_processor_mode", "Edit Existing Processor")
        _builder_steps(load_context(workspace))

    with _PolicyWarningCapture() as capture:
        rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()

    assert not rendered.exception
    mode_labels = {
        button.label
        for button in rendered.button
        if button.label in ("Create New Processor", "Edit Existing Processor")
    }
    assert mode_labels == {"Create New Processor", "Edit Existing Processor"}
    assert rendered.session_state["builder_processor_mode"] == "Edit Existing Processor"
    assert not capture.default_clash_messages


# ---------------------------------------------------------------------------
# Distribution metrics: digest quantile without a stored quantile.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tdigest_quantile_model_defaults_to_median_and_bounds_quantile() -> None:
    parsed = model.Metrics.model_validate(
        {
            "metrics": {
                "dist": {
                    "source": "descriptive",
                    "kind": "tdigest_quantile",
                    "state": "Propensity_tdigest",
                }
            }
        }
    ).metrics["dist"]
    assert parsed.quantile == 0.5

    with pytest.raises(ValueError, match="quantile"):
        model.Metrics.model_validate(
            {
                "metrics": {
                    "bad": {
                        "source": "descriptive",
                        "kind": "tdigest_quantile",
                        "state": "Propensity_tdigest",
                        "quantile": 1.5,
                    }
                }
            }
        )


@pytest.mark.unit
def test_quantile_metric_form_supports_distribution_without_quantile() -> None:
    """A digest metric seeded without a quantile edits as a distribution metric."""
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import forms

ctx = forms.MetricFormContext(state_options=lambda _types: ["Propensity_tdigest"])
st.session_state["result"] = forms.metric_kind_fields(
    "tdigest_quantile",
    {"source": "descriptive", "kind": "tdigest_quantile", "state": "Propensity_tdigest"},
    ctx,
    key_prefix="dist_quantile",
)
"""
    ).run()

    assert not app.exception
    checkbox = next(
        item for item in app.checkbox if item.key == "dist_quantile_quantile_distribution"
    )
    assert checkbox.value is True
    assert app.session_state["result"] == {"state": "Propensity_tdigest"}

    rendered = checkbox.uncheck().run()
    assert rendered.session_state["result"] == {
        "state": "Propensity_tdigest",
        "quantile": 0.5,
    }


@pytest.mark.unit
def test_quantile_metric_form_defaults_to_single_quantile_for_new_metrics() -> None:
    app = AppTest.from_string(
        """
import streamlit as st
from valuestream.ui import forms

ctx = forms.MetricFormContext(state_options=lambda _types: ["Propensity_tdigest"])
st.session_state["result"] = forms.metric_kind_fields(
    "tdigest_quantile", {}, ctx, key_prefix="new_quantile"
)
"""
    ).run()

    assert not app.exception
    checkbox = next(
        item for item in app.checkbox if item.key == "new_quantile_quantile_distribution"
    )
    assert checkbox.value is False
    # No state chosen yet: the form stays invalid instead of guessing.
    assert app.session_state["result"] is None

    state_select = next(item for item in app.selectbox if item.key == "new_quantile_quantile_state")
    rendered = state_select.set_value("Propensity_tdigest").run()
    assert rendered.session_state["result"] == {
        "state": "Propensity_tdigest",
        "quantile": 0.5,
    }

    checkbox = next(
        item for item in rendered.checkbox if item.key == "new_quantile_quantile_distribution"
    )
    rendered = checkbox.check().run()
    assert rendered.session_state["result"] == {"state": "Propensity_tdigest"}


@pytest.mark.unit
def test_boxplot_chart_offered_only_for_distribution_metrics() -> None:
    """The metric's digest defines the boxplot; scalar metrics get no box."""
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "version": 1,
                "workspace": "box",
                "sources": [
                    {
                        "id": "ih",
                        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
                        "schema": {"natural_key": ["CustomerID"]},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "descriptive",
                        "source": "ih",
                        "kind": "numeric_distribution",
                        "properties": ["Propensity"],
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "PropensityDistribution": {
                        "source": "descriptive",
                        "kind": "tdigest_quantile",
                        "state": "Propensity_tdigest",
                    },
                    "PropensityCount": {
                        "source": "descriptive",
                        "kind": "formula",
                        "expression": {"col": "Propensity_Count"},
                    },
                }
            },
            "dashboards": {"dashboards": []},
        }
    )

    assert "boxplot" in builder.chart_choices_for_metric(catalog, "PropensityDistribution")
    assert "boxplot" not in builder.chart_choices_for_metric(catalog, "PropensityCount")
    # The axis is optional; the digest property comes from the metric.
    assert builder.chart_field_controls("boxplot") == ("x", "color", "facet_row", "facet_col")


@pytest.mark.unit
def test_descriptive_boxplot_is_retired_from_chart_offering() -> None:
    assert "descriptive_boxplot" not in RECIPES
    assert "boxplot" in RECIPES
    assert "descriptive_boxplot" not in builder.CHART_DISPLAY_LABELS
    assert set(RECIPES) == set(builder.CHART_REQUIRED_FIELDS)
    assert set(RECIPES) == set(config_builder.REPORT_LIBRARY_GROUP_BY_CHART)
    assert set(model.Tile.model_json_schema()["properties"]["chart"]["enum"]) == set(RECIPES)


def _tile_option(dashboard_id: str, page_id: str, tile_id: str, title: str):
    return (dashboard_id, page_id, tile_id, {"id": tile_id, "title": title})


@pytest.mark.unit
def test_report_library_labels_disambiguate_duplicate_pages() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "version": 1,
                "workspace": "labels",
                "sources": [
                    {"id": "ih", "reader": {"kind": "parquet", "file_pattern": "*.parquet"}}
                ],
            },
            "processors": {"processors": []},
            "metrics": {"metrics": {}},
            "dashboards": {
                "dashboards": [
                    {"id": "model_quality", "title": "Model quality", "pages": []},
                    {"id": "experiments", "title": "Experiments", "pages": []},
                ]
            },
        }
    )
    colliding = [
        _tile_option("model_quality", "distributions", "response_time_boxplot", "Response time"),
        _tile_option("experiments", "distributions", "response_time_boxplot", "Response time"),
        _tile_option("model_quality", "distributions", "response_histogram", "Histogram"),
    ]

    labels = config_builder._report_library_tile_labels(catalog, colliding)

    assert labels["model_quality/distributions/response_time_boxplot"] == (
        "Response time · Distributions · Model quality"
    )
    assert labels["experiments/distributions/response_time_boxplot"] == (
        "Response time · Distributions · Experiments"
    )
    # Unique labels stay short.
    assert labels["model_quality/distributions/response_histogram"] == "Histogram · Distributions"
