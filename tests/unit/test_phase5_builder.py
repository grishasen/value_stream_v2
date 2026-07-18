"""Phase 5 Builder helper tests."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import plotly.io as pio  # type: ignore[import-untyped]
import polars as pl
import pytest
import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

from valuestream.config import model
from valuestream.config.loader import load
from valuestream.expr.translator import translate
from valuestream.query import executor
from valuestream.states import kll, topk
from valuestream.ui import builder, dimension_profile, forms
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
    metric_action = next(
        item for item in rendered.segmented_control if item.label == "Metric action"
    )
    rendered = metric_action.set_value("Edit Existing Metric").run()

    assert not rendered.exception
    assert any(button.label == "Continue" for button in rendered.button)
    assert not any(button.label == "Apply to workspace" for button in rendered.button)
    assert not any("Editing draft" in item.value for item in rendered.markdown)


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
def test_set_op_metric_keeps_time_window_operands_read_only() -> None:
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
    assert app.session_state["result"] == {
        "op": "intersection",
        "operands": [
            {
                "state": "Active_theta",
                "time_window": {"start": "-30d", "end": "-1d"},
            },
            {
                "state": "Active_theta",
                "time_window": {"start": "-1d", "end": "now"},
            },
        ],
    }
    assert app.info


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
    assert dimension_profile.dimension_pack_fields(sample.columns)[:3] == [
        "Channel",
        "Issue",
        "CustomerType",
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
def test_exploration_processor_helpers_generate_valid_yaml_shapes() -> None:
    source = model.Source.model_validate(
        {
            "id": "ih",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "schema": {
                "timestamp_column": "Day",
                "natural_key": ["CustomerID"],
            },
        }
    )
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "ih",
            "kind": "binary_outcome",
            "dimensions": ["Channel"],
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )
    sample = pl.DataFrame(
        {
            "Day": ["2026-06-01", "2026-06-02"],
            "Channel": ["Web", "Mobile"],
            "Issue": ["Cards", "Loans"],
            "Campaign": ["Retention", "CrossSell"],
            "CustomerID": ["c1", "c2"],
        }
    )

    temporary = config_builder._temporary_processor_def(
        processor,
        source,
        ["Channel", "Issue"],
        ttl_days=7,
        window_days=30,
        sample=sample,
    )
    sketch, metrics = config_builder._sketch_processor_and_metrics(
        source,
        base_processor=processor,
        dimensions=["Channel"],
        topk_field="Campaign",
        entity_field="CustomerID",
        include_cpc=True,
        include_theta=True,
    )

    assert temporary["dimensions"] == ["Channel", "Issue"]
    assert temporary["exploration"]["temporary"] is True
    assert temporary["exploration"]["ttl_days"] == 7
    assert temporary["filter"]["polars"].startswith("pl.col('Day').cast(pl.String) >=")
    parsed_temporary = model.BinaryOutcomeProcessor.model_validate(temporary)
    assert parsed_temporary.filter is not None
    assert translate(parsed_temporary.filter) is not None
    assert sketch["kind"] == "entity_set"
    assert {state["type"] for state in sketch["states"].values()} == {"topk", "cpc", "theta"}
    assert {metric["kind"] for metric in metrics.values()} == {
        "topk_items",
        "approx_distinct_count",
    }
    model.EntitySetProcessor.model_validate(sketch)
    model.Metrics.model_validate({"metrics": metrics})


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
def test_report_library_plotly_previews_cover_every_supported_chart() -> None:
    for chart_type in builder.CHART_REQUIRED_FIELDS:
        figure = config_builder._chart_library_preview(chart_type, theme_base="light")

        assert figure.data
        assert pio.to_json(figure, validate=True)


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
            "Chart": "Kpi card",
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
        from valuestream.ui.pages.config_builder import (  # noqa: PLC0415
            _render_report_library_chart_group,
            _tile_option_key,
        )

        _render_report_library_chart_group(
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
def test_generated_catalog_id_uses_name_prefix_and_random_suffix() -> None:
    generated = config_builder._generated_catalog_id(
        "Engagement Rate by Offer",
        "0123456789abcdef",
        fallback="tile",
    )

    assert generated == "engagement_rate_by_o_0123456789abcdef"


@pytest.mark.unit
def test_generated_catalog_id_keeps_identifier_letter_prefixed() -> None:
    generated = config_builder._generated_catalog_id(
        "2026 Campaign",
        "0123456789abcdef",
        fallback="tile",
    )

    assert generated == "tile_2026_campaign_0123456789abcdef"


@pytest.mark.unit
def test_stable_random_suffix_uses_eight_random_bytes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_token_hex(byte_count: int) -> str:
        calls.append(byte_count)
        return "feedfacecafebeef"

    monkeypatch.setattr(config_builder.secrets, "token_hex", fake_token_hex)
    state: dict[str, object] = {}

    first = config_builder._stable_random_suffix(state, "suffix")
    second = config_builder._stable_random_suffix(state, "suffix")

    assert first == "feedfacecafebeef"
    assert second == "feedfacecafebeef"
    assert calls == [8]


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
def test_sketch_exploration_does_not_preselect_topk_or_avoid_fields() -> None:
    tree = ast.parse(Path(config_builder.__file__).read_text(encoding="utf-8"))
    sketch_panel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sketch_exploration_panel"
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
