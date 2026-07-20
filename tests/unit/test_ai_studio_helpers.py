"""AI configuration studio helper tests."""

from __future__ import annotations

import copy
import gzip
import json
import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import streamlit as st

import valuestream.ai.studio as ai_studio
from valuestream.ai import (
    AICallSettings,
    call_litellm,
    classify_draft_validation_issues,
    filter_draft_by_selection,
    generate_schema_preview,
    merge_draft_sections,
    parse_ai_yaml_sections,
    prompt_for_config_draft,
    prompt_for_draft_refinement,
    prompt_for_repair,
    prompt_for_report_refresh,
    section_name_diff,
    tile_keys,
    validate_draft_catalog,
    validate_draft_field_contract,
    validation_trace_for_repair,
)
from valuestream.config import model
from valuestream.config.loader import load
from valuestream.readers.discovery import discover
from valuestream.recipes import (
    instantiate_metric,
    instantiate_tile,
    load_builtin_kpi_recipes,
    processor_with_recipe_states,
    recipe_readiness,
)
from valuestream.ui import builder, dimension_profile, forms, recipe_library
from valuestream.ui.pages import ai_config_studio as ai_config_studio_page
from valuestream.ui.pages.ai_config_studio import (
    DETERMINISTIC_STEPS,
    STEPS,
    _apply_field_approval_edits,
    _blank_ai_calculation_row,
    _catalog_approved_fields,
    _deterministic_dashboards_from_metrics,
    _draft_files,
    _draft_funnel_stage_options,
    _draft_metric_choice_label,
    _draft_metric_kinds_for_source,
    _draft_metric_names_for_source_kind,
    _draft_metric_source_ids,
    _draft_state_options,
    _field_approval_editor_rows,
    _generated_processor_id,
    _install_recipe_in_draft,
    _load_ai_settings_config,
    _normalize_studio_step,
    _processor_state_rows,
    _processor_state_specs,
    _processor_states_from_rows,
    _rename_capitalize_mapping,
    _render_selected_step,
    _schema_preview_display_rows,
    _schema_sample,
    _studio_steps,
    _sync_ai_rename_capitalize_state,
    _update_metric_definition,
    _update_processor_definition,
    _with_generated_report_ids,
    _working_sample,
)


@pytest.mark.unit
def test_workspace_sample_files_lists_supported_data_files_only(tmp_path: Path) -> None:
    data = tmp_path / "data"
    nested = data / "ih"
    nested.mkdir(parents=True)
    (nested / "interactions.zip").write_bytes(b"zip")
    (nested / "interactions.json").write_text("{}", encoding="utf-8")
    (nested / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "outside.zip").write_bytes(b"ignore")

    assert ai_config_studio_page._workspace_sample_files(tmp_path) == [
        "data/ih/interactions.json",
        "data/ih/interactions.zip",
    ]


@pytest.mark.unit
def test_sample_capability_registry_drives_upload_and_workspace_contract(tmp_path: Path) -> None:
    advertised = ai_config_studio_page._sample_upload_extensions()
    registered = {
        extension
        for capability in ai_config_studio_page.SAMPLE_FORMAT_CAPABILITIES
        for extension in capability.upload_extensions
    }
    data = tmp_path / "data"
    data.mkdir()
    for extension in advertised:
        (data / f"sample.{extension}").write_bytes(b"")
    (data / "sample.xlsx").write_bytes(b"")

    discovered = ai_config_studio_page._workspace_sample_files(tmp_path)

    assert set(advertised) == registered
    assert "xlsx" not in advertised
    assert len(discovered) == len(advertised)
    assert all(not value.endswith(".xlsx") for value in discovered)
    assert all(
        ai_config_studio_page._sample_format_capability(value) is not None for value in discovered
    )


@pytest.mark.unit
def test_unsupported_sample_format_fails_before_payload_parsing() -> None:
    with pytest.raises(ValueError, match="Unsupported sample format"):
        ai_config_studio_page._read_sample_bytes("sample.xlsx", b"not a workbook")


@pytest.mark.unit
def test_sample_source_plan_matches_preview_and_runtime_format() -> None:
    csv_plan = ai_config_studio_page._sample_source_plan(
        "orders.csv",
        ["OrderID", "Revenue"],
        workspace_relative="data/orders/orders.csv",
    )
    parquet_plan = ai_config_studio_page._sample_source_plan(
        "orders.parquet",
        ["OrderID", "Revenue"],
    )
    generic_json_plan = ai_config_studio_page._sample_source_plan(
        "events.ndjson",
        ["event_id", "event_time"],
        workspace_relative="data/events/events.ndjson",
    )
    demo_plan = ai_config_studio_page._sample_source_plan(
        "value_stream_demo.csv",
        ["CustomerID", "OutcomeTime", "Outcome", "Channel"],
        workspace_relative="data/studio/value_stream_demo.csv",
    )

    assert (csv_plan.reader_kind, csv_plan.root, csv_plan.file_pattern) == (
        "csv",
        "data/orders",
        "**/*.csv",
    )
    assert csv_plan.production_ready
    assert (parquet_plan.reader_kind, parquet_plan.file_pattern) == (
        "parquet",
        "**/*.parquet",
    )
    assert not parquet_plan.production_ready
    assert generic_json_plan.reader_kind == "pega_ds_export"
    assert generic_json_plan.requires_runtime_confirmation
    assert not generic_json_plan.production_ready
    assert not generic_json_plan.group_pattern
    assert not generic_json_plan.timestamp_format
    assert demo_plan.timestamp_format == "%+"

    pega_parquet_plan = ai_config_studio_page._sample_source_plan(
        "interactions.parquet",
        ["pyOutcome", "pxOutcomeTime", "pxInteractionID"],
        workspace_relative="data/Month=08/Day=2024-08-31/interactions.parquet",
    )
    assert pega_parquet_plan.root == "data"
    assert pega_parquet_plan.file_pattern == "**/*.parquet"
    assert pega_parquet_plan.timestamp_format == "%Y%m%dT%H%M%S%.3f %Z"

    uppercase_csv_plan = ai_config_studio_page._sample_source_plan(
        "Orders.CSV",
        ["OrderID", "Revenue"],
    )
    assert uppercase_csv_plan.file_pattern == "**/*.CSV"


@pytest.mark.unit
def test_partitioned_sample_plan_discovers_sibling_partitions(tmp_path: Path) -> None:
    relative_files = (
        Path("data/Month=08/Day=2024-08-30/orders.parquet"),
        Path("data/Month=08/Day=2024-08-31/orders.parquet"),
        Path("data/Month=09/Day=2024-09-01/orders.parquet"),
    )
    for relative in relative_files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"sample")
    plan = ai_config_studio_page._sample_source_plan(
        relative_files[1].name,
        ["OrderID"],
        workspace_relative=relative_files[1].as_posix(),
    )
    source = model.Source.model_validate(
        {
            "id": "orders",
            "reader": {
                "kind": plan.reader_kind,
                "root": plan.root,
                "file_pattern": plan.file_pattern,
            },
        }
    )

    discovered = {
        path.relative_to(tmp_path).as_posix()
        for chunk in discover(tmp_path, source)
        for path in chunk.files
    }

    assert plan.root == "data"
    assert plan.file_pattern == "**/*.parquet"
    assert discovered == {path.as_posix() for path in relative_files}


@pytest.mark.unit
def test_workspace_parquet_preview_is_bounded_without_reading_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"row": range(10_000), "value": ["x"] * 10_000}).write_parquet(
        path,
        row_group_size=250,
    )

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unexpected eager byte read: {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    frame = ai_config_studio_page._read_workspace_sample(
        path,
        limit=123,
        columns=["value"],
    )

    assert frame.shape == (123, 1)
    assert frame.columns == ["value"]


@pytest.mark.unit
def test_workspace_parquet_preview_applies_projection_before_row_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations: list[tuple[str, object]] = []

    class FakeLazyFrame:
        def select(self, columns: list[str]) -> FakeLazyFrame:
            operations.append(("select", columns))
            return self

        def head(self, limit: int) -> FakeLazyFrame:
            operations.append(("head", limit))
            return self

        def collect(self) -> pl.DataFrame:
            operations.append(("collect", None))
            return pl.DataFrame({"value": [1, 2]})

    monkeypatch.setattr(ai_config_studio_page.pl, "scan_parquet", lambda _path: FakeLazyFrame())

    frame = ai_config_studio_page._read_workspace_sample(
        Path("sample.parquet"),
        limit=2,
        columns=["value"],
    )

    assert frame.columns == ["value"]
    assert operations == [("select", ["value"]), ("head", 2), ("collect", None)]


@pytest.mark.unit
def test_outcome_mapping_prefers_pega_outcome_over_outcome_time() -> None:
    sample = pl.DataFrame(
        {
            "pxOutcomeTime": ["20240831T010203.000 GMT", "20240831T010204.000 GMT"],
            "pyOutcome": ["Clicked", "Impression"],
            "OutcomeTime": ["20240831T010203.000 GMT", "20240831T010204.000 GMT"],
        }
    )

    assert ai_config_studio_page._default_outcome_column(sample) == "pyOutcome"


@pytest.mark.unit
def test_observed_outcome_groups_cover_every_fixture_value_without_pending() -> None:
    working = pl.DataFrame(
        {
            "Outcome": [
                "Impression",
                "NoConversion",
                "Clicked",
                "Conversion",
                "Impression",
            ]
        }
    )

    positive, negative = ai_config_studio_page._observed_outcome_groups(working, "Outcome")

    assert positive == ["Clicked"]
    assert negative == ["Conversion", "Impression", "NoConversion"]
    assert "Pending" not in [*positive, *negative]


@pytest.mark.unit
def test_studio_continue_queues_step_before_jump_widget_renders() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import (  # noqa: PLC0415 - isolated AppTest source
            ai_config_studio as page,
        )

        steps = ["Sample", "Required Fields"]
        queued = page._normalize_studio_step(
            st.session_state.pop("ai_studio_next_step", None), steps
        )
        if queued in steps:
            st.session_state["ai_studio_step"] = queued
            st.session_state["ai_studio_jump_step"] = queued
        current = st.session_state.get("ai_studio_step", steps[0])
        current = page._render_studio_step_header(current, steps)
        page._render_studio_step_navigation(current, steps)

    at = AppTest.from_function(app).run()

    next(widget for widget in at.button if widget.label == "Continue").click().run()

    assert not at.exception
    assert at.session_state["ai_studio_step"] == "Required Fields"
    assert at.selectbox[0].value == "Required Fields"


@pytest.mark.unit
def test_schema_contract_review_queues_navigation_without_mutating_rendered_widget() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        steps = page.STEPS
        st.session_state[page.AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY] = True
        queued = page._normalize_studio_step(
            st.session_state.pop("ai_studio_next_step", None), steps
        )
        if queued in steps:
            st.session_state["ai_studio_step"] = queued
            st.session_state["ai_studio_jump_step"] = queued
        current = page._normalize_studio_step(
            st.session_state.get("ai_studio_step", steps[0]), steps
        )
        assert current is not None
        st.session_state["ai_studio_step"] = current
        page._render_studio_step_header(current, steps)
        page._render_schema_contract_notice(steps)

    at = AppTest.from_function(app).run()
    at = (
        next(button for button in at.button if button.label == "Review updated draft").click().run()
    )

    assert not at.exception
    assert at.session_state["ai_studio_step"] == STEPS[6]
    assert at.session_state["ai_studio_jump_step"] == STEPS[6]
    assert at.selectbox[0].value == STEPS[6]


@pytest.mark.unit
def test_phase_rail_migrates_legacy_step_and_preserves_committed_state_on_jump() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        steps = page._studio_steps(ai_calls_enabled=False)
        if "qa_phase_seeded" not in st.session_state:
            st.session_state["qa_phase_seeded"] = True
            st.session_state["ai_studio_step"] = page.STEPS[6]
            st.session_state["qa_committed_draft"] = {"revision": "accepted"}
            st.session_state["ai_studio_reviewed_signature"] = "reviewed"
        queued = page._normalize_studio_step(
            st.session_state.pop("ai_studio_next_step", None), steps
        )
        if queued in steps:
            st.session_state["ai_studio_step"] = queued
            st.session_state["ai_studio_jump_step"] = queued
        current = page._normalize_studio_step(st.session_state["ai_studio_step"], steps)
        assert current is not None
        st.session_state["ai_studio_step"] = current
        page._render_studio_step_header(
            current,
            steps,
            statuses={
                "Data": "complete",
                "Draft": "attention",
                "Review": "empty",
                "Apply": "empty",
            },
        )

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    assert rendered.session_state["ai_studio_step"] == DETERMINISTIC_STEPS[6]
    assert rendered.segmented_control[0].options == [
        "Data · Complete",
        "Draft · Attention",
        "Review · Not started",
        "Apply · Not started",
    ]
    assert rendered.segmented_control[0].value == "Draft"
    assert rendered.selectbox[0].label == "Jump to step in Draft"
    assert rendered.selectbox[0].options == [DETERMINISTIC_STEPS[6]]

    rendered = rendered.segmented_control[0].set_value("Review").run()

    assert not rendered.exception
    assert rendered.session_state["ai_studio_step"] == DETERMINISTIC_STEPS[7]
    assert rendered.selectbox[0].label == "Jump to step in Review"
    assert rendered.selectbox[0].options == DETERMINISTIC_STEPS[7:11]
    assert rendered.session_state["qa_committed_draft"] == {"revision": "accepted"}
    assert rendered.session_state["ai_studio_reviewed_signature"] == "reviewed"


@pytest.mark.unit
def test_required_field_mapping_renders_with_targeted_optional_help() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        from valuestream.ui.pages import (  # noqa: PLC0415 - isolated AppTest source
            ai_config_studio as page,
        )

        sample = page._demo_sample()
        page._set_effective_schema_state(sample)
        page._required_fields(sample, sample)

    at = AppTest.from_function(app).run()

    assert not at.exception
    assert {widget.label for widget in at.selectbox} >= {
        "Subject ID Field",
        "Outcome Field",
        "Outcome Timestamp",
        "Decision Timestamp",
    }


@pytest.mark.unit
def test_read_sample_bytes_rejects_zip_without_supported_members() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("README.txt", "not data")

    with pytest.raises(ValueError, match="JSON or NDJSON"):
        ai_config_studio_page._read_sample_bytes("sample.zip", buffer.getvalue())


@pytest.mark.unit
def test_oversized_upload_is_rejected_before_getvalue() -> None:
    getvalue_called = False

    def fail_getvalue() -> bytes:
        nonlocal getvalue_called
        getvalue_called = True
        raise AssertionError("oversized upload payload was materialized")

    upload = SimpleNamespace(
        size=ai_config_studio_page.AI_STUDIO_UPLOAD_MAX_BYTES + 1,
        getvalue=fail_getvalue,
    )

    with pytest.raises(ai_config_studio_page.SamplePreviewLimitError, match="64 MiB"):
        ai_config_studio_page._uploaded_sample_bytes(upload)

    assert getvalue_called is False


@pytest.mark.unit
@pytest.mark.parametrize("extension", ["json", "ndjson"])
def test_json_preview_stops_before_invalid_rows_after_limit(extension: str) -> None:
    if extension == "json":
        payload = b'[{"row": 1}, {"row": 2}, BROKEN]'
    else:
        payload = b'{"row": 1}\n{"row": 2}\nBROKEN\n'

    frame = ai_config_studio_page._read_sample_bytes(
        f"sample.{extension}",
        payload,
        limit=2,
    )

    assert frame.to_dicts() == [{"row": 1}, {"row": 2}]


@pytest.mark.unit
def test_gzip_preview_stops_before_invalid_rows_after_limit() -> None:
    payload = gzip.compress(b'{"row": 1}\n{"row": 2}\nBROKEN\n')

    frame = ai_config_studio_page._read_sample_bytes("sample.gz", payload, limit=2)

    assert frame.to_dicts() == [{"row": 1}, {"row": 2}]


@pytest.mark.unit
def test_gzip_preview_rejects_expansion_beyond_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_config_studio_page, "AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES", 64)
    payload = gzip.compress(b"x" * 65)

    with pytest.raises(ai_config_studio_page.SamplePreviewLimitError, match="expanded"):
        ai_config_studio_page._read_sample_bytes("sample.gz", payload, limit=2)


@pytest.mark.unit
def test_zip_preview_rejects_declared_expansion_before_member_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_config_studio_page, "AI_STUDIO_ARCHIVE_EXPANDED_MAX_BYTES", 64)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("rows.json", json.dumps([{"value": "x" * 100}]))

    with pytest.raises(ai_config_studio_page.SamplePreviewLimitError, match="expanded"):
        ai_config_studio_page._read_sample_bytes("sample.zip", buffer.getvalue(), limit=2)


@pytest.mark.unit
def test_zip_preview_stops_reading_members_after_row_limit() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.ndjson", '{"row": 1}\n{"row": 2}\n')
        archive.writestr("b.json", "BROKEN")

    frame = ai_config_studio_page._read_sample_bytes(
        "sample.zip",
        buffer.getvalue(),
        limit=2,
    )

    assert frame.to_dicts() == [{"row": 1}, {"row": 2}]


@pytest.mark.unit
def test_workspace_buffered_preview_rejects_size_before_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sample.json"
    path.write_bytes(b"x" * 65)
    monkeypatch.setattr(ai_config_studio_page, "AI_STUDIO_UPLOAD_MAX_BYTES", 64)

    def fail_read_bytes(self: Path) -> bytes:
        raise AssertionError(f"unexpected eager byte read: {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)

    with pytest.raises(ai_config_studio_page.SamplePreviewLimitError, match="64 MiB"):
        ai_config_studio_page._read_workspace_sample(path, limit=2)


@pytest.mark.unit
def test_parse_ai_yaml_sections_accepts_fenced_file_keys() -> None:
    sections = parse_ai_yaml_sections(
        """
```yaml
processors.yaml:
  processors:
    - id: engagement
      source: ih
      kind: binary_outcome
metrics.yaml:
  metrics:
    CTR:
      source: engagement
      kind: formula
      expression: {col: Count}
dashboards.yaml:
  dashboards:
    - id: overview
      title: Overview
      pages: []
chat_with_data:
  agent_prompt: Use CDH business language.
  metric_descriptions:
    engagement: CTR and lift metrics.
```
"""
    )

    assert sections["processors"]["processors"][0]["id"] == "engagement"
    assert "CTR" in sections["metrics"]["metrics"]
    assert sections["dashboards"]["dashboards"][0]["id"] == "overview"
    assert sections["chat_with_data"] == {
        "agent_prompt": "Use CDH business language.",
        "metric_descriptions": {"engagement": "CTR and lift metrics."},
    }


@pytest.mark.unit
def test_generate_validated_candidate_repairs_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(["metrics: {metrics: {}}", "metrics: {metrics: {CTR: {}}}"])
    prompts: list[str] = []

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        return (bool(candidate.get("metrics", {}).get("metrics")), ["missing metric"])

    monkeypatch.setattr(ai_studio, "validate_draft_catalog", validate)
    result = ai_studio.generate_validated_candidate(
        base_draft=_base_draft(),
        prompt="first",
        call=lambda prompt: (prompts.append(prompt), next(responses))[1],
        repair_prompt=lambda _draft, issues, _trace: f"repair: {issues[0]}",
        max_repairs=1,
    )

    assert result.ok
    assert result.attempts == 2
    assert prompts == ["first", "repair: missing metric"]


@pytest.mark.unit
def test_generate_validated_candidate_records_parse_and_success_diagnostics() -> None:
    malformed_response = "metrics:\n  metrics: [\n    - broken"
    valid_response = "metrics: {metrics: {CTR: {}}}"
    responses = iter([malformed_response, valid_response])

    result = ai_studio.generate_validated_candidate(
        base_draft=_base_draft(),
        prompt="generate",
        call=lambda _prompt: next(responses),
        repair_prompt=lambda _draft, _issues, _trace: "repair",
        max_repairs=1,
        validate=lambda _candidate: (True, []),
        operation="catalog_draft",
    )

    assert result.ok
    assert result.attempts == 2
    assert re.fullmatch(r"[0-9a-f]{12}", result.reference)
    assert len(result.attempt_diagnostics) == 2

    parse_diagnostic, success_diagnostic = result.attempt_diagnostics
    assert parse_diagnostic.attempt == 1
    assert parse_diagnostic.role == "generation"
    assert parse_diagnostic.stage == "parse"
    assert parse_diagnostic.issues == (
        "The model response was not valid catalog YAML. Return complete catalog sections.",
    )
    assert parse_diagnostic.issue_count == 1
    assert parse_diagnostic.issue_areas == (("other", 1),)
    assert parse_diagnostic.sections == ()
    assert parse_diagnostic.response_chars == len(malformed_response)
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", parse_diagnostic.error_type)
    assert parse_diagnostic.line == 3
    assert parse_diagnostic.column == 5

    assert success_diagnostic.attempt == 2
    assert success_diagnostic.role == "repair"
    assert success_diagnostic.stage == "validated"
    assert success_diagnostic.issues == ()
    assert success_diagnostic.issue_count == 0
    assert success_diagnostic.issue_areas == ()
    assert success_diagnostic.sections == ("metrics",)
    assert success_diagnostic.response_chars == len(valid_response)
    assert success_diagnostic.error_type == ""
    assert success_diagnostic.line is None
    assert success_diagnostic.column is None


@pytest.mark.unit
def test_generate_validated_candidate_bounds_attempt_diagnostic_issues() -> None:
    long_issue = "miscellaneous: " + ("x" * 700) + "\nPRIVATE-CUSTOMER-42"
    issues = [
        "sources[private-source].path: invalid /Users/alice",
        "processors.processors.0.PRIVATE-ID: invalid",
        "metrics.metrics.PRIVATE-METRIC: invalid",
        "dashboards.dashboards.0: invalid",
        "chat_with_data.agent_prompt: invalid",
        "processor private references stale raw field 'secret_name'",
        long_issue,
        "miscellaneous second issue",
        "miscellaneous third issue",
        "miscellaneous fourth issue",
        "miscellaneous fifth issue",
    ]

    result = ai_studio.generate_validated_candidate(
        base_draft=_base_draft(),
        prompt="generate",
        call=lambda _prompt: "metrics: {metrics: {CTR: {}}}",
        repair_prompt=lambda _draft, _issues, _trace: "repair",
        max_repairs=0,
        validate=lambda _candidate: (False, issues),
    )

    assert not result.ok
    assert re.fullmatch(r"[0-9a-f]{12}", result.reference)
    assert len(result.attempt_diagnostics) == 1
    diagnostic = result.attempt_diagnostics[0]
    assert diagnostic.stage == "validation"
    assert diagnostic.role == "generation"
    assert diagnostic.issue_count == len(issues)
    assert len(diagnostic.issues) == 8
    assert all(len(issue) <= 512 for issue in diagnostic.issues)
    assert all("\n" not in issue for issue in diagnostic.issues)
    assert diagnostic.issues[6].endswith("…")
    assert diagnostic.issue_areas == (
        ("source", 1),
        ("processor", 1),
        ("metric", 1),
        ("report", 1),
        ("chat", 1),
        ("field_contract", 1),
        ("other", 5),
    )
    assert diagnostic.sections == ("metrics",)


@pytest.mark.unit
def test_generate_validated_candidate_repairs_stale_post_rename_processor_field() -> None:
    base = _base_draft()
    source = base["pipelines"]["sources"][0]
    source["schema"] = {
        "timestamp_column": "pxOutcomeTime",
        "natural_key": ["pyCustomerID", "pyChannel"],
    }
    source["transforms"] = [{"kind": "rename_capitalize"}]
    processor_yaml = """
processors:
  processors:
    - id: engagement
      source: ih
      kind: binary_outcome
      dimensions: [{channel}]
      time: {{column: OutcomeTime, grains: [Day, Summary]}}
      outcome:
        column: Outcome
        positive_values: [Clicked]
        negative_values: [Impression]
"""
    responses = iter(
        [
            processor_yaml.format(channel="pyChannel"),
            processor_yaml.format(channel="Channel"),
        ]
    )
    prompts: list[str] = []

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        catalog_ok, catalog_issues = validate_draft_catalog(candidate)
        fields_ok, field_issues = validate_draft_field_contract(
            candidate,
            ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
            source_id="ih",
        )
        return catalog_ok and fields_ok, [*catalog_issues, *field_issues]

    result = ai_studio.generate_validated_candidate(
        base_draft=base,
        prompt="draft",
        call=lambda prompt: (prompts.append(prompt), next(responses))[1],
        repair_prompt=lambda _draft, issues, _trace: f"repair: {issues[0]}",
        max_repairs=1,
        validate=validate,
    )

    assert result.ok
    assert result.attempts == 2
    assert result.draft is not None
    assert result.draft["processors"]["processors"][0]["dimensions"] == ["Channel"]
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_generate_validated_candidate_never_returns_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ai_studio,
        "validate_draft_catalog",
        lambda _candidate: (False, ["metrics.bad: invalid reference"]),
    )
    accepted = _base_draft()

    result = ai_studio.generate_validated_candidate(
        base_draft=accepted,
        prompt="first",
        call=lambda _prompt: "metrics: {metrics: {bad: {source: missing, kind: formula}}}",
        repair_prompt=lambda _draft, _issues, _trace: "repair",
        max_repairs=2,
    )

    assert not result.ok
    assert result.draft is None
    assert result.attempts == 3
    assert result.failure_stage == "validation"
    assert accepted == _base_draft()


@pytest.mark.unit
def test_generate_validated_candidate_bounds_repairs_to_two() -> None:
    calls = 0

    def call(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "not: a catalog section"

    result = ai_studio.generate_validated_candidate(
        base_draft=_base_draft(),
        prompt="first",
        call=call,
        repair_prompt=lambda _draft, _issues, _trace: "repair",
        max_repairs=99,
    )

    assert not result.ok
    assert result.draft is None
    assert result.attempts == 3
    assert calls == 3


@pytest.mark.unit
def test_validate_draft_catalog_hides_unexpected_exception_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "PRIVATE-CUSTOMER-42 /Users/alice/private.yaml"

    def raise_unexpected(_draft: dict) -> dict:
        raise RuntimeError(secret)

    monkeypatch.setattr(ai_studio, "_catalog_sections_for_validation", raise_unexpected)

    ok, issues = ai_studio.validate_draft_catalog(_base_draft())

    assert not ok
    assert issues == ["Catalog validation could not complete (RuntimeError)."]
    assert secret not in str(issues)


@pytest.mark.unit
def test_draft_validation_snapshot_uses_effective_field_contract_and_cache_identity() -> None:
    st.session_state.clear()
    draft = _base_draft()
    source = draft["pipelines"]["sources"][0]
    source["schema"] = {
        "timestamp_column": "pxOutcomeTime",
        "natural_key": ["pyCustomerID", "pyChannel"],
    }
    source["transforms"] = [{"kind": "rename_capitalize"}]
    approved = ["Channel", "CustomerID", "Outcome", "OutcomeTime"]
    st.session_state["ai_studio_draft_source"] = ""
    st.session_state["ai_studio_source_id"] = "ih"
    st.session_state["ai_studio_approved_fields"] = approved
    st.session_state["ai_studio_effective_schema_columns"] = approved
    st.session_state["ai_studio_effective_schema_signature"] = "schema-a"

    stale = copy.deepcopy(draft)
    stale["processors"]["processors"][0]["dimensions"] = ["pyChannel"]
    stale_snapshot = ai_config_studio_page._draft_validation_snapshot(stale)
    valid_snapshot = ai_config_studio_page._draft_validation_snapshot(draft)

    assert not stale_snapshot.ok
    assert any("stale raw field 'pyChannel'" in issue for issue in stale_snapshot.issues)
    assert valid_snapshot.ok
    cache_size = len(st.session_state["ai_studio_validation_cache"])

    st.session_state["ai_studio_approved_fields"] = [
        "CustomerID",
        "Outcome",
        "OutcomeTime",
    ]
    st.session_state["ai_studio_effective_schema_signature"] = "schema-b"
    changed_contract = ai_config_studio_page._draft_validation_snapshot(draft)

    assert not changed_contract.ok
    assert len(st.session_state["ai_studio_validation_cache"]) == cache_size + 1


@pytest.mark.unit
def test_invalid_schema_snapshot_clears_matching_review_on_cache_miss_and_hit() -> None:
    st.session_state.clear()
    draft = _base_draft()
    source = draft["pipelines"]["sources"][0]
    source["schema"] = {
        "timestamp_column": "pxOutcomeTime",
        "natural_key": ["pyCustomerID", "pyChannel"],
    }
    source["transforms"] = [{"kind": "rename_capitalize"}]
    draft["processors"]["processors"][0]["dimensions"] = ["pyChannel"]
    approved = ["Channel", "CustomerID", "Outcome", "OutcomeTime"]
    signature = ai_config_studio_page._draft_signature(draft)
    st.session_state["ai_studio_draft_source"] = ""
    st.session_state["ai_studio_source_id"] = "ih"
    st.session_state["ai_studio_approved_fields"] = approved
    st.session_state["ai_studio_effective_schema_columns"] = approved
    st.session_state["ai_studio_effective_schema_signature"] = "schema-a"
    st.session_state["ai_studio_reviewed_signature"] = signature

    first = ai_config_studio_page._draft_validation_snapshot(draft)

    assert not first.ok
    assert st.session_state["ai_studio_reviewed_signature"] == ""

    st.session_state["ai_studio_reviewed_signature"] = signature
    second = ai_config_studio_page._draft_validation_snapshot(draft)

    assert second == first
    assert st.session_state["ai_studio_reviewed_signature"] == ""


@pytest.mark.unit
def test_validated_candidate_records_timeout_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    events: list[tuple[object, object]] = []

    def capture_event(_session_state: object, **kwargs: object) -> bool:
        events.append((kwargs["event"], kwargs["outcome"]))
        return True

    monkeypatch.setattr(ai_config_studio_page, "record_event", capture_event)
    monkeypatch.setattr(
        ai_config_studio_page,
        "_preflight_ai_operation",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("provider timeout")),
    )

    def app(draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ai import AICallSettings  # noqa: PLC0415
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        try:
            with st.status("operation") as status:
                page._run_validated_ai_candidate(
                    settings=AICallSettings(model="openai/test", api_key="test"),
                    operation="catalog_draft",
                    prompt="draft",
                    base_draft=draft,
                    approved_fields=[],
                    repair_prompt_factory=lambda *_args: "repair",
                    status=status,
                )
        except TimeoutError:
            st.session_state["timeout_caught"] = True

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()

    assert not at.exception
    assert at.session_state["timeout_caught"]
    assert (
        ai_config_studio_page.AuthoringEvent.FAILED,
        ai_config_studio_page.AuthoringOutcome.TIMEOUT,
    ) in events


@pytest.mark.unit
def test_provider_preflight_uses_short_timeout_and_negative_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st.session_state.clear()
    observed_timeouts: list[int] = []

    def timeout_call(settings: AICallSettings, *_args: object, **_kwargs: object) -> str:
        observed_timeouts.append(settings.timeout_seconds)
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(
        ai_config_studio_page,
        "_call_litellm_for_current_sample",
        timeout_call,
    )
    settings = AICallSettings(model="openai/test", api_key="test", timeout_seconds=90)

    with pytest.raises(ai_studio.AIProviderCallError) as first_failure:
        ai_config_studio_page._preflight_ai_operation(
            settings,
            operation="catalog_draft",
            approved_fields=[],
        )
    with pytest.raises(ai_studio.AIProviderCallError) as cached_failure:
        ai_config_studio_page._preflight_ai_operation(
            settings,
            operation="catalog_draft",
            approved_fields=[],
        )

    st.session_state["ai_studio_force_preflight_retry"] = True
    with pytest.raises(ai_studio.AIProviderCallError) as retried_failure:
        ai_config_studio_page._preflight_ai_operation(
            settings,
            operation="catalog_draft",
            approved_fields=[],
        )

    assert first_failure.value.category is ai_studio.AIProviderFailureCategory.TIMEOUT
    assert first_failure.value.retryable is True
    assert cached_failure.value.call_id == first_failure.value.call_id
    assert cached_failure.value.category is ai_studio.AIProviderFailureCategory.TIMEOUT
    assert retried_failure.value.call_id != first_failure.value.call_id
    assert observed_timeouts == [5, 5]


@pytest.mark.unit
def test_provider_preflight_caches_success_for_matching_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st.session_state.clear()
    observed_timeouts: list[int] = []

    def ready_call(settings: AICallSettings, *_args: object, **_kwargs: object) -> str:
        observed_timeouts.append(settings.timeout_seconds)
        return "READY"

    monkeypatch.setattr(
        ai_config_studio_page,
        "_call_litellm_for_current_sample",
        ready_call,
    )
    settings = AICallSettings(model="openai/test", api_key="test", timeout_seconds=120)

    for _ in range(2):
        ai_config_studio_page._preflight_ai_operation(
            settings,
            operation="report_refresh",
            approved_fields=[],
        )

    assert observed_timeouts == [5]


@pytest.mark.unit
def test_merge_and_validate_ai_sections() -> None:
    draft = _base_draft()
    sections = parse_ai_yaml_sections(
        """
metrics:
  CTR:
    source: engagement
    kind: formula
    expression:
      op: safe_div
      num: {col: Positives}
      den: {col: Count}
  Total:
    source: engagement
    kind: formula
    expression: {col: Count}
"""
    )

    merged = merge_draft_sections(draft, sections)
    ok, issues = validate_draft_catalog(merged)

    assert ok, issues
    assert sorted(merged["metrics"]["metrics"]) == ["CTR", "Total"]


@pytest.mark.unit
def test_validate_draft_catalog_ignores_chat_with_data_settings() -> None:
    draft = _base_draft()
    draft["chat_with_data"] = {
        "agent_prompt": "Use Pega CDH terms.",
        "metric_descriptions": {"engagement": "CTR metrics."},
    }

    ok, issues = validate_draft_catalog(draft)

    assert ok, issues


@pytest.mark.unit
def test_draft_files_exports_ai_yaml_when_chat_settings_exist() -> None:
    draft = _base_draft()
    draft["chat_with_data"] = {
        "agent_prompt": "Use Pega CDH terms.",
        "metric_descriptions": {"engagement": "CTR metrics."},
    }

    files = _draft_files(draft)

    assert files["ai.yaml"] == {
        "chat_with_data": {
            "agent_prompt": "Use Pega CDH terms.",
            "metric_descriptions": {"engagement": "CTR metrics."},
        }
    }


@pytest.mark.unit
def test_apply_draft_rolls_back_catalog_and_ai_config_on_post_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = _base_draft()
    draft["chat_with_data"] = {
        "agent_prompt": "Use the reviewed KPI catalog.",
        "metric_descriptions": {"CTR": "Engagement rate."},
    }
    ai_path = tmp_path / "ai.yaml"
    original_ai = "ai:\n  llm:\n    model: ollama/test\n"
    ai_path.write_text(original_ai, encoding="utf-8")

    def fail_validation(_workspace: object, **_kwargs: object) -> None:
        raise ValueError("post-write validation failed")

    monkeypatch.setattr(builder, "require_valid_workspace", fail_validation)

    with pytest.raises(ValueError, match="post-write validation failed"):
        ai_config_studio_page._apply_draft(
            SimpleNamespace(workspace=tmp_path),
            draft,
        )

    assert ai_path.read_text(encoding="utf-8") == original_ai
    assert not any(
        (tmp_path / "catalog" / filename).exists() for filename in builder.CATALOG_FILENAMES
    )


@pytest.mark.unit
def test_apply_draft_persists_reviewed_removals_after_catalog_reload(tmp_path: Path) -> None:
    current = json.loads(json.dumps(_base_draft()))
    legacy_source = json.loads(json.dumps(current["pipelines"]["sources"][0]))
    legacy_source["id"] = "legacy"
    current["pipelines"]["sources"].append(legacy_source)
    legacy_processor = json.loads(json.dumps(current["processors"]["processors"][0]))
    legacy_processor.update({"id": "legacy_engagement", "source": "legacy"})
    current["processors"]["processors"].append(legacy_processor)
    current["metrics"]["metrics"]["LegacyCTR"] = {
        **current["metrics"]["metrics"]["CTR"],
        "source": "legacy_engagement",
    }
    ctx = SimpleNamespace(workspace=tmp_path)
    ai_config_studio_page._apply_draft(ctx, current)

    accepted = _base_draft()
    accepted["dashboards"]["theme"] = {"colorway": ["#275dad"]}
    ai_config_studio_page._apply_draft(ctx, accepted)

    reloaded = load(tmp_path)
    assert [source.id for source in reloaded.pipelines.sources] == ["ih"]
    assert [processor.id for processor in reloaded.processors.processors] == ["engagement"]
    assert sorted(reloaded.metrics.metrics) == ["CTR"]
    assert [dashboard.id for dashboard in reloaded.dashboards.dashboards] == ["overview"]
    assert reloaded.dashboards.theme == {"colorway": ["#275dad"]}


@pytest.mark.unit
def test_filter_draft_by_metric_selection_drops_dependent_tiles() -> None:
    draft = _base_draft()
    draft["metrics"]["metrics"]["Total"] = {
        "source": "engagement",
        "kind": "formula",
        "expression": {"col": "Count"},
    }
    draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"].append(
        {
            "id": "total",
            "title": "Total",
            "metric": "Total",
            "chart": "bar",
            "x": "Channel",
            "y": "Total",
        }
    )

    filtered = filter_draft_by_selection(draft, selected_metrics=["CTR"])

    assert sorted(filtered["metrics"]["metrics"]) == ["CTR"]
    assert tile_keys(filtered) == ["overview/engagement/ctr"]


@pytest.mark.unit
def test_filter_draft_respects_explicit_empty_selection() -> None:
    filtered = filter_draft_by_selection(
        _base_draft(),
        selected_processors=[],
        selected_metrics=[],
        selected_tiles=[],
    )

    assert filtered["processors"]["processors"] == []
    assert filtered["metrics"]["metrics"] == {}
    assert filtered["dashboards"]["dashboards"] == []


@pytest.mark.unit
def test_install_recipe_in_ai_draft_adds_metric_and_report_tile() -> None:
    draft = _base_draft()
    processor = model.Processors.model_validate(draft["processors"]).processors[0]
    recipe = next(
        item
        for item in load_builtin_kpi_recipes().recipes
        if item.id == "engagement.engagement_rate"
    )
    readiness = recipe_readiness(recipe, processor)
    metric_id = "Recipe_Engagement"
    request = recipe_library.RecipeInstallRequest(
        recipe_id=recipe.id,
        recipe_version=recipe.version,
        metric_id=metric_id,
        metric_def=instantiate_metric(
            recipe,
            processor,
            metric_id,
            readiness.resolved_inputs,
        ),
        report_target=recipe_library.ReportPageTarget(
            dashboard_id="overview",
            dashboard_title="Overview",
            page_id="engagement",
            page_title="Engagement",
        ),
        tile_def=instantiate_tile(recipe, metric_id, "recipe_engagement_tile"),
    )

    updated = _install_recipe_in_draft(draft, request)
    ok, issues = validate_draft_catalog(updated)

    assert ok, issues
    assert metric_id not in draft["metrics"]["metrics"]
    assert updated["metrics"]["metrics"][metric_id]["recipe"] == {
        "id": recipe.id,
        "version": recipe.version,
    }
    assert tile_keys(updated)[-1] == "overview/engagement/recipe_engagement_tile"


@pytest.mark.unit
def test_install_recipe_in_ai_draft_adds_processor_state_before_metric() -> None:
    draft = _base_draft()
    processor = model.Processors.model_validate(draft["processors"]).processors[0]
    recipe = next(
        item for item in load_builtin_kpi_recipes().recipes if item.id == "audience.unique_entities"
    )
    state_additions = {
        "Channel_cpc": {
            "type": "cpc",
            "source_column": "Channel",
            "lg_k": 11,
        }
    }
    configured = processor_with_recipe_states(processor, state_additions)
    request = recipe_library.RecipeInstallRequest(
        recipe_id=recipe.id,
        recipe_version=recipe.version,
        metric_id="Unique_Channels",
        metric_def=instantiate_metric(
            recipe,
            configured,
            "Unique_Channels",
            {"cardinality_state": "Channel_cpc"},
        ),
        processor_id=processor.id,
        state_additions=state_additions,
    )

    updated = _install_recipe_in_draft(draft, request)
    ok, issues = validate_draft_catalog(updated)

    assert ok, issues
    assert "states" not in draft["processors"]["processors"][0]
    states = updated["processors"]["processors"][0]["states"]
    assert states["Channel_cpc"] == state_additions["Channel_cpc"]
    assert {"Count", "Positives", "Negatives"} <= set(states)
    assert updated["metrics"]["metrics"]["Unique_Channels"]["state"] == "Channel_cpc"


@pytest.mark.unit
def test_validate_draft_catalog_reports_cross_reference_errors() -> None:
    draft = _base_draft()
    draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]["metric"] = "Missing"

    ok, issues = validate_draft_catalog(draft)

    assert not ok
    assert any("unknown metric 'Missing'" in issue for issue in issues)


@pytest.mark.unit
def test_default_group_by_fields_use_dimension_profile_rules() -> None:
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
            "DecisionTime": [
                "2026-01-01T01:00:00",
                "2026-01-02T01:00:00",
                "2026-01-03T01:00:00",
                "2026-01-04T01:00:00",
                "2026-01-05T01:00:00",
            ],
            "Rank": [1, 1, 2, 2, 3],
            "Propensity": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    group_by = ai_config_studio_page._default_group_by_fields(
        sample,
        list(sample.columns),
        ["Outcome", "OutcomeTime"],
    )

    assert group_by == ["Channel"]


@pytest.mark.unit
def test_default_group_by_fields_include_explicit_calendar_granularities() -> None:
    sample = pl.DataFrame(
        {
            "Channel": ["Web", "Mobile", "Branch", "Web"],
            "Day": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "Month": ["2026-01"] * 4,
            "Quarter": ["2026-Q1"] * 4,
            "Year": [2026] * 4,
            "Issue": ["Cards", "Loans", "Savings", "Cards"],
        }
    )
    calendar_fields = ["Day", "Month", "Quarter", "Year"]

    group_by = ai_config_studio_page._default_group_by_fields(
        sample,
        list(sample.columns),
        calendar_fields,
    )
    rows = {
        row.field: row
        for row in dimension_profile.selection_dimension_profile_rows(
            sample,
            selected_fields=[],
            required_fields=calendar_fields,
        )
    }

    assert group_by == ["Channel", "Day", "Month", "Quarter", "Year"]
    assert {
        field: (rows[field].recommendation, rows[field].reason) for field in calendar_fields
    } == dict.fromkeys(
        calendar_fields,
        (
            "Recommended",
            "Explicit calendar granularity for time-based aggregate breakdowns.",
        ),
    )


@pytest.mark.unit
def test_missing_kind_specific_metric_fields_are_repairable_draft_issues() -> None:
    issues = [
        "metrics.metrics.customer_reach.approx_distinct_count.state: Field required",
        "metrics.metrics.response_p50.tdigest_quantile.state: Field required",
        "metrics.metrics.calibration.calibration_from_digests.positive_state: Field required",
        "metrics.metrics.dropoff.funnel_dropoff.to_stage: Field required",
        "processors.processors.0.binary_outcome.states: Input should be a valid dictionary",
        "metrics.metrics.ModelControl_Contingency.contingency_test.tests.0: Input should be 'chi2', 'g' or 'z'",
        "dashboards[overview].pages[main].tiles[bad].metric: unknown metric 'Missing'",
    ]

    blocking, repairable = classify_draft_validation_issues(issues)

    assert blocking == [
        "dashboards[overview].pages[main].tiles[bad].metric: unknown metric 'Missing'"
    ]
    assert repairable == issues[:6]


@pytest.mark.unit
def test_config_draft_prompt_lists_metric_kind_requirements() -> None:
    prompt = prompt_for_config_draft(
        file_name="sample.csv",
        approved_schema=[
            {"column": "CustomerID", "dtype": "String", "unique": 500},
            {"column": "Channel", "dtype": "String", "unique": 3},
            {"column": "OutcomeTime", "dtype": "Datetime", "unique": 500},
            {"column": "Revenue", "dtype": "Float64", "unique": 120},
            {"column": "ModelControlGroup", "dtype": "String", "unique": 2},
            {"column": "Propensity", "dtype": "Float64", "unique": 500},
        ],
        approved_fields=[
            "CustomerID",
            "Channel",
            "OutcomeTime",
            "Revenue",
            "ModelControlGroup",
            "Propensity",
        ],
        hidden_fields=["InternalSecret"],
        baseline_draft=_base_draft(),
    )

    assert "Application structure dictionary:" in prompt
    assert "Catalog schema dictionary:" in prompt
    assert "Processor kind dictionary:" in prompt
    assert "Metric kind dictionary:" in prompt
    assert "Chart required-field dictionary:" in prompt
    assert "Expression AST dictionary:" in prompt
    assert "Approved field role dictionary:" in prompt
    assert "optional chat_with_data" in prompt
    assert "chat_with_data.agent_prompt" in prompt
    assert "tdigest_quantile:" in prompt
    assert "allowed_tests:" in prompt
    assert "safe_dimension_candidates:" in prompt
    assert "- Channel" in prompt
    assert "time_candidates:" in prompt
    assert "- OutcomeTime" in prompt
    assert "numeric_property_candidates:" in prompt
    assert "- Revenue" in prompt
    assert "avoid_for_group_by_or_filters:" in prompt
    assert "- CustomerID" in prompt
    assert "Hidden field count: 1" in prompt
    assert "InternalSecret" not in prompt
    assert "Do not emit legacy TOML-only settings such as metrics.global_filters" in prompt
    assert "Set sketch_build_mode to bulk" in prompt
    assert (
        "Create as many distinct valid processors as possible from the approved schema "
        "and business requirements."
    ) in prompt
    assert "Do not stop after a minimal baseline" in prompt
    assert "Maximize useful processor coverage while keeping the catalog coherent" in prompt
    assert "Prefer a small useful set over a large speculative catalog" not in prompt
    assert "Every report/dashboard tile metric exists in metrics." in prompt
    assert "Output valid YAML only." in prompt
    assert "Return valid YAML only. Do not wrap the answer in prose or Markdown fences." in prompt


@pytest.mark.unit
def test_expression_prompt_dictionary_covers_the_closed_dsl() -> None:
    expression_ast = ai_studio.catalog_prompt_dictionaries()["expression_ast"]
    prompted_ops = {op for form in expression_ast["operator_forms"].values() for op in form["ops"]}

    assert prompted_ops == {
        "not",
        "neg",
        "abs",
        "sqrt",
        "exp",
        "ceil",
        "floor",
        "log",
        "round",
        "cast",
        "and",
        "or",
        "add",
        "sub",
        "mul",
        "div",
        "safe_div",
        "concat",
        "least",
        "greatest",
        "coalesce",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "in",
        "not_in",
        "between",
        "is_null",
        "not_null",
        "matches",
        "starts_with",
        "ends_with",
        "case",
        "when_then",
        "date_trunc",
        "date_diff",
        "date_part",
        "now",
        "strftime",
        "strptime",
    }
    assert expression_ast["operator_forms"]["concatenation"]["shape"] == {
        "op": "concat",
        "args": ["<string expression>", "<string expression>", "<optional more>"],
        "sep": "<optional string; empty by default>",
    }
    assert expression_ast["examples"]["concatenate_fields"] == {
        "op": "concat",
        "args": [{"col": "Issue"}, {"col": "Group"}, {"col": "Name"}],
        "sep": "/",
    }


@pytest.mark.unit
def test_chart_prompt_dictionary_matches_tile_validation_contract() -> None:
    from typing import get_args  # noqa: PLC0415 - test-only introspection

    from valuestream.config.validate import (  # noqa: PLC0415 - contract under test
        _TILE_REQUIRED_ALTERNATIVES,
    )

    chart_dictionary = ai_studio.catalog_prompt_dictionaries()["chart_required_fields"]
    required_by_chart = chart_dictionary["required_fields_by_chart"]
    chart_kinds = set(get_args(model.Tile.model_fields["chart"].annotation))

    assert set(required_by_chart) == chart_kinds
    for chart, alternatives in _TILE_REQUIRED_ALTERNATIVES.items():
        assert required_by_chart[chart] == ["|".join(group) for group in alternatives], chart

    for chart, example in chart_dictionary["tile_examples"].items():
        example_chart = example["chart"]
        assert example_chart in chart_kinds, chart
        for requirement in required_by_chart[example_chart]:
            assert any(option in example for option in requirement.split("|")), (
                chart,
                requirement,
            )

    state_type_enum = ai_studio.catalog_prompt_dictionaries()["catalog_schema"]["processors.yaml"][
        "state_type_enum"
    ]
    assert set(state_type_enum) == set(get_args(model.StateSpec.model_fields["type"].annotation))


@pytest.mark.unit
def test_config_draft_prompt_includes_business_requirements() -> None:
    prompt = prompt_for_config_draft(
        file_name="sample.csv",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=[],
        baseline_draft=_base_draft(),
        user_goals="Weekly conversion by channel.\nAverage revenue per customer.",
    )

    assert "Business requirements from the user" in prompt
    assert "Weekly conversion by channel." in prompt
    assert "Average revenue per customer." in prompt
    goals_index = prompt.index("Business requirements from the user")
    assert goals_index < prompt.index("Source sample:")


@pytest.mark.unit
def test_config_draft_prompt_omits_requirements_heading_when_goals_empty() -> None:
    prompt = prompt_for_config_draft(
        file_name="sample.csv",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=[],
        baseline_draft=_base_draft(),
        user_goals="   ",
    )

    assert "Business requirements from the user" not in prompt


@pytest.mark.unit
def test_report_refresh_prompt_includes_business_requirements() -> None:
    prompt = prompt_for_report_refresh(
        file_name="sample.csv",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=[],
        current_draft=_base_draft(),
        user_goals="Focus reports on channel-level engagement.",
    )

    assert "Business requirements from the user" in prompt
    assert "Focus reports on channel-level engagement." in prompt
    assert "Refresh only dashboards.yaml" in prompt


@pytest.mark.unit
def test_draft_refinement_prompt_includes_change_request_and_rules() -> None:
    prompt = prompt_for_draft_refinement(
        file_name="sample.csv",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=[],
        current_draft=_base_draft(),
        instruction="Add a KPI card with total orders to the overview page.",
        user_goals="Weekly conversion by channel.",
    )

    assert "Change request from the user:" in prompt
    assert "Add a KPI card with total orders to the overview page." in prompt
    assert "Business requirements from the user" in prompt
    assert "Revise this Value Stream catalog draft" in prompt
    assert "keep unrelated processors, metrics, and tiles unchanged." in prompt
    assert "Return valid YAML only. Do not wrap the answer in prose or Markdown fences." in prompt


@pytest.mark.unit
def test_repair_prompt_includes_validation_errors_and_traceback() -> None:
    prompt = prompt_for_repair(
        file_name="sample.csv",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=["CustomerID"],
        current_draft=_base_draft(),
        validation_issues=[
            "dashboards[overview].pages[engagement].tiles[ctr].metric: unknown metric 'Missing'",
            "CustomerID must not be exposed",
        ],
        validation_trace=(
            "Traceback (most recent call last):\nValidationError: CustomerID caused a bad draft"
        ),
    )

    assert "Validation errors to fix:" in prompt
    assert "unknown metric" in prompt
    assert "Missing" in prompt
    assert "Validation exception traceback, if available:" in prompt
    assert "CustomerID" not in prompt
    assert "<hidden-field>" in prompt
    assert "Traceback (most recent call last):" in prompt
    assert "ValidationError: <hidden-field> caused a bad draft" in prompt


@pytest.mark.unit
def test_redact_hidden_field_mentions_preserves_approved_fields_containing_hidden_names() -> None:
    redacted = ai_studio.redact_hidden_field_mentions(
        {
            "goals": "Weekly OutcomeTime trend split by Outcome and outcome.",
            "schema": [{"column": "OutcomeTime"}, {"column": "Outcome"}],
        },
        ["Outcome"],
        preserve_fields=["OutcomeTime", "Channel"],
    )

    assert (
        redacted["goals"] == "Weekly OutcomeTime trend split by <hidden-field> and <hidden-field>."
    )
    assert redacted["schema"] == [{"column": "OutcomeTime"}, {"column": "<hidden-field>"}]


@pytest.mark.unit
def test_redact_hidden_field_mentions_keeps_ids_derived_from_preserved_fields() -> None:
    redacted = ai_studio.redact_hidden_field_mentions(
        ["overview/engagement/OutcomeTime_tile", "CustomerID_metric", "Outcome_metric"],
        ["Outcome", "CustomerID"],
        preserve_fields=["OutcomeTime"],
    )

    assert redacted == [
        "overview/engagement/OutcomeTime_tile",
        "<hidden-field>_metric",
        "<hidden-field>_metric",
    ]


@pytest.mark.unit
def test_redact_hidden_field_mentions_still_redacts_derived_ids_without_preserve_list() -> None:
    text = "column: Outcome\n- CustomerID_metric\nvalue: 'Outcome'\nother: outcome_time"

    redacted = ai_studio.redact_hidden_field_mentions(text, ["Outcome", "CustomerID"])

    assert redacted == (
        "column: <hidden-field>\n- <hidden-field>_metric\nvalue: '<hidden-field>'\n"
        "other: <hidden-field>_time"
    )


@pytest.mark.unit
def test_repair_prompt_keeps_approved_fields_containing_hidden_names() -> None:
    prompt = prompt_for_repair(
        file_name="sample.csv",
        approved_schema=[
            {"column": "OutcomeTime", "dtype": "Datetime", "unique": 300},
            {"column": "Channel", "dtype": "String", "unique": 3},
        ],
        approved_fields=["OutcomeTime", "Channel"],
        hidden_fields=["Outcome"],
        current_draft=_base_draft(),
        validation_issues=["Outcome must not be exposed; keep OutcomeTime for the time axis"],
        validation_trace="",
    )

    assert "OutcomeTime" in prompt
    assert "<hidden-field-name>Time" not in prompt
    assert "<hidden-field>Time" not in prompt
    assert re.search(r"(?<![A-Za-z0-9_])Outcome(?![A-Za-z0-9_])", prompt) is None
    assert "<hidden-field> must not be exposed" in prompt


@pytest.mark.unit
def test_validation_trace_for_repair_captures_model_validation_traceback() -> None:
    draft = _base_draft()
    draft["processors"]["processors"][0]["kind"] = "unknown_processor_kind"

    trace = validation_trace_for_repair(draft)

    assert "Catalog model validation failed" in trace
    assert "ValidationError" in trace
    assert "unknown_processor_kind" in trace
    assert "Traceback (most recent call last):" in trace


@pytest.mark.unit
def test_metric_editor_state_options_handle_invalid_ai_state_lists() -> None:
    draft = _base_draft()
    draft["processors"]["processors"][0]["states"] = [
        {"name": "UniqueCustomers_hll", "type": "hll"},
        {"name": "Score_tdigest", "type": "tdigest"},
    ]

    assert _draft_state_options(draft, "engagement", state_types={"hll"}) == ["UniqueCustomers_hll"]
    assert "Count" in _draft_state_options(draft, "engagement", state_types={"count"})
    assert _draft_state_options(draft, "engagement", state_types={"tdigest"}) == ["Score_tdigest"]


@pytest.mark.unit
def test_draft_metric_choice_label_includes_id_source_and_kind() -> None:
    draft = _base_draft()
    draft["metrics"]["metrics"]["Dropoff"] = {
        "source": "engagement",
        "kind": "formula",
        "expression": {
            "op": "safe_div",
            "num": {"col": "Count"},
            "den": {"col": "Count"},
        },
    }

    assert (
        _draft_metric_choice_label(draft, "Dropoff")
        == "Dropoff · engagement · Formula / state passthrough"
    )


@pytest.mark.unit
def test_draft_metric_filter_helpers_scope_metrics_by_source_and_kind() -> None:
    draft = _base_draft()
    draft["metrics"]["metrics"]["Dropoff"] = {
        "source": "engagement",
        "kind": "formula",
        "expression": {
            "op": "safe_div",
            "num": {"col": "Count"},
            "den": {"col": "Count"},
        },
    }
    draft["metrics"]["metrics"]["Reach"] = {
        "source": "unknown_profile",
        "kind": "approx_distinct_count",
        "state": "UniqueCustomers_hll",
    }

    assert _draft_metric_source_ids(draft) == ["engagement", "unknown_profile"]
    assert _draft_metric_kinds_for_source(draft, "engagement") == ["formula"]
    assert _draft_metric_names_for_source_kind(draft, "engagement", "formula") == [
        "CTR",
        "Dropoff",
    ]


@pytest.mark.unit
def test_ai_metric_parameter_fields_pass_variant_role_map_to_shared_form(monkeypatch) -> None:
    draft = _base_draft()
    processor = draft["processors"]["processors"][0]
    processor["variant_column"] = "ExperimentGroup"
    processor["variant_role_map"] = {"Test": "Challenger", "Control": "Champion"}
    captured: dict[str, forms.MetricFormContext] = {}

    def fake_metric_kind_fields(
        kind: str,
        seed: dict[str, object],
        ctx: forms.MetricFormContext,
        *,
        key_prefix: str,
    ) -> dict[str, object]:
        captured["ctx"] = ctx
        return {}

    monkeypatch.setattr(forms, "metric_kind_fields", fake_metric_kind_fields)

    ai_config_studio_page._metric_kind_parameter_fields(
        draft,
        "engagement",
        "variant_compare",
        {},
        key_prefix="test_metric",
    )

    assert captured["ctx"].default_variant_column == "ExperimentGroup"
    assert captured["ctx"].variant_roles == {"Test": "Challenger", "Control": "Champion"}


@pytest.mark.unit
def test_metric_editor_update_renames_tile_metric_references() -> None:
    draft = _base_draft()

    updated = _update_metric_definition(
        draft,
        "CTR",
        "ClickThroughRate",
        {
            "source": "engagement",
            "kind": "formula",
            "expression": {"col": "Count"},
        },
    )

    assert "CTR" not in updated["metrics"]["metrics"]
    assert "ClickThroughRate" in updated["metrics"]["metrics"]
    assert (
        updated["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]["metric"]
        == "ClickThroughRate"
    )


@pytest.mark.unit
def test_metric_editor_funnel_stage_options_use_processor_stages() -> None:
    draft = _base_draft()
    draft["processors"]["processors"].append(
        {
            "id": "journey",
            "source": "ih",
            "kind": "funnel",
            "stages": [{"name": "Impression"}, {"name": "Click"}, {"name": "Conversion"}],
        }
    )

    assert _draft_funnel_stage_options(draft, "journey") == [
        "Impression",
        "Click",
        "Conversion",
    ]


@pytest.mark.unit
def test_processor_editor_converts_ai_state_lists_to_state_mapping() -> None:
    processor = {
        "id": "engagement",
        "kind": "binary_outcome",
        "states": [
            {"name": "UniqueCustomers_hll", "type": "hll", "source_column": "CustomerID"},
            {"name": "BadState", "type": "", "enabled": True},
        ],
    }

    rows = _processor_state_rows(processor)
    states = _processor_states_from_rows(rows, builder.state_spec_definitions(processor))

    assert states["UniqueCustomers_hll"] == {
        "type": "hll",
        "source_column": "CustomerID",
    }
    assert states["Count"] == {"type": "count"}
    assert "BadState" not in states


@pytest.mark.unit
def test_processor_editor_state_rows_preserve_kind_specific_extras() -> None:
    processor = {
        "id": "scores",
        "kind": "score_distribution",
        "score_properties": ["Propensity"],
        "states": {
            "Propensity_tdigest_positives": {
                "type": "tdigest",
                "source_column": "Propensity",
                "score_property": "Propensity",
                "outcome": "positive",
                "k": 500,
            },
            "UniqueCustomers_hll": {
                "type": "hll",
                "source_column": "CustomerID",
                "lg_k": 12,
            },
        },
    }

    rows = _processor_state_rows(processor)
    states = _processor_states_from_rows(rows, builder.state_spec_definitions(processor))

    assert states["Propensity_tdigest_positives"] == {
        "type": "tdigest",
        "source_column": "Propensity",
        "score_property": "Propensity",
        "outcome": "positive",
        "k": 500,
    }
    assert states["UniqueCustomers_hll"] == {
        "type": "hll",
        "source_column": "CustomerID",
        "lg_k": 12,
    }


@pytest.mark.unit
def test_processor_editor_inferred_state_rows_preserve_runtime_metadata() -> None:
    processor = {
        "id": "scores",
        "kind": "score_distribution",
        "score_properties": ["ModelScore"],
        "entities": {"subject": "CustomerID"},
    }

    rows = _processor_state_rows(processor)
    states = _processor_states_from_rows(rows, _processor_state_specs(processor))

    assert states["ModelScore_tdigest_positives"] == {
        "type": "tdigest",
        "source_column": "ModelScore",
        "outcome": "positive",
        "score_property": "ModelScore",
        "k": 500,
    }
    assert states["ModelScore_tdigest_negatives"] == {
        "type": "tdigest",
        "source_column": "ModelScore",
        "outcome": "negative",
        "score_property": "ModelScore",
        "k": 500,
    }
    assert states["UniqueCustomers_cpc"] == {
        "type": "cpc",
        "source_column": "CustomerID",
        "lg_k": 11,
    }
    assert states["personalization"] == {"type": "pooled_mean", "weight": "Count"}


@pytest.mark.unit
def test_processor_editor_inferred_binary_cpc_uses_subject_source_column() -> None:
    processor = {
        "id": "engagement",
        "kind": "binary_outcome",
        "entities": {"subject": "CustomerID"},
    }

    rows = _processor_state_rows(processor)
    states = _processor_states_from_rows(rows, _processor_state_specs(processor))

    assert states["UniqueSubjects_cpc"] == {
        "type": "cpc",
        "source_column": "CustomerID",
        "lg_k": 11,
    }


@pytest.mark.unit
def test_state_spec_definitions_handle_ai_state_lists() -> None:
    processor = {
        "states": [
            {
                "name": "UniqueCustomers_hll",
                "type": "hll",
                "source_column": "CustomerID",
                "lg_k": 12,
                "enabled": True,
            }
        ]
    }

    assert builder.state_spec_definitions(processor) == {
        "UniqueCustomers_hll": {
            "type": "hll",
            "source_column": "CustomerID",
            "lg_k": 12,
        }
    }


@pytest.mark.unit
def test_processor_states_from_rows_drop_cleared_source_column() -> None:
    existing = {"Visitors_hll": {"type": "hll", "source_column": "CustomerID", "lg_k": 12}}
    rows = [{"State": "Visitors_hll", "Type": "hll", "Source Column": "", "Enabled": True}]

    states = _processor_states_from_rows(rows, existing)

    assert states == {"Visitors_hll": {"type": "hll", "lg_k": 12}}


@pytest.mark.unit
def test_processor_editor_rename_updates_metric_sources() -> None:
    draft = _base_draft()

    updated = _update_processor_definition(
        draft,
        "engagement",
        "engagement_v2",
        {
            "id": "engagement_v2",
            "source": "ih",
            "kind": "binary_outcome",
            "dimensions": ["Channel"],
            "states": {"Count": {"type": "count"}},
        },
    )

    assert updated["processors"]["processors"][0]["id"] == "engagement_v2"
    assert updated["metrics"]["metrics"]["CTR"]["source"] == "engagement_v2"


@pytest.mark.unit
def test_generate_schema_preview_masks_unselected_examples() -> None:
    frame = pl.DataFrame(
        {
            "Channel": ["Web", "Mobile", "Web"],
            "Sensitive": ["a", "b", "c"],
        }
    )

    preview = generate_schema_preview(
        frame,
        approved_fields=["Channel", "Sensitive"],
        example_fields=["Channel"],
    )

    by_column = {row["column"]: row for row in preview}
    assert by_column["Channel"]["examples"] == ["Web", "Mobile"]
    assert "examples" not in by_column["Sensitive"]


@pytest.mark.unit
def test_zero_example_prompt_sends_profile_counts_but_no_values_or_hidden_names() -> None:
    frame = pl.DataFrame(
        {
            "Channel": ["private-web", "private-mobile"],
            "SubjectID": ["customer-secret-001", "customer-secret-002"],
            "HealthDiagnosis": ["private-a", "private-b"],
        }
    )
    approved_fields = ["Channel", "SubjectID"]
    schema = generate_schema_preview(frame, approved_fields, example_fields=[])

    prompt = prompt_for_config_draft(
        file_name="sample.csv",
        approved_schema=schema,
        approved_fields=approved_fields,
        hidden_fields=["HealthDiagnosis"],
        baseline_draft=_base_draft(),
    )

    assert "nulls:" in prompt
    assert "unique:" in prompt
    assert "Hidden field count: 1" in prompt
    assert "private-web" not in prompt
    assert "customer-secret-001" not in prompt
    assert "HealthDiagnosis" not in prompt


@pytest.mark.unit
def test_schema_preview_display_rows_are_arrow_safe_for_mixed_examples() -> None:
    rows = _schema_preview_display_rows(
        [
            {
                "column": "Revenue",
                "dtype": "Float64",
                "nulls": 0,
                "unique": 2,
                "examples": [1.25, 2.5],
            },
            {
                "column": "Outcome",
                "dtype": "String",
                "nulls": 0,
                "unique": 2,
                "examples": ["", "Accepted"],
            },
        ]
    )

    assert rows[0]["examples"] == "[1.25, 2.5]"
    assert rows[1]["examples"] == "['', 'Accepted']"
    pa.Table.from_pandas(pd.DataFrame(rows))


@pytest.mark.unit
def test_field_approval_editor_rows_include_approval_examples_and_schema() -> None:
    frame = pl.DataFrame(
        {
            "Channel": ["Web", "Mobile", "Web"],
            "SubjectID": ["C1", "C2", "C3"],
            "Sensitive": ["a", "b", "c"],
        }
    )

    rows = _field_approval_editor_rows(
        frame,
        ["Channel", "SubjectID", "Sensitive"],
        required_fields=["SubjectID"],
        approved_fields=["Channel", "SubjectID"],
        example_fields=["Channel"],
    )

    by_field = {row["Column"]: row for row in rows}
    assert by_field["Channel"]["Approve"] is True
    assert by_field["Sensitive"]["Approve"] is False
    assert by_field["Channel"]["Send To AI"] is True
    assert by_field["Channel"]["Most occurring"] == "['Web']"
    assert by_field["Channel"]["Values"] == "['Web', 'Mobile']"
    assert "Required" in by_field["SubjectID"]["Field Tags"]
    assert "Likely ID" in by_field["SubjectID"]["Field Tags"]


@pytest.mark.unit
def test_new_sample_defaults_example_sharing_off_and_prompt_preview_has_no_values() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        frame = pl.DataFrame(
            {
                "Channel": ["Web", "Mobile"],
                "SubjectID": ["customer-001", "customer-002"],
            }
        )
        st.session_state["ai_studio_sample_identity"] = "sample-one"
        page._initialize_state(frame)
        available = sorted(frame.columns, key=str.casefold)
        approved, examples, _ = page._sync_field_approval_state(
            frame,
            available,
            ["SubjectID"],
        )
        st.session_state["approved"] = approved
        st.session_state["examples"] = examples
        st.session_state["schema_preview"] = page._schema_preview_for_ai(frame, approved)
        st.session_state["fresh_editor_rows"] = {
            "defaults": list(st.session_state["ai_studio_defaults"]),
            "filters": list(st.session_state["ai_studio_filter_rows"]),
            "calculations": list(st.session_state["ai_studio_calculations"]),
        }

    at = AppTest.from_function(app).run()

    assert not at.exception
    assert at.session_state["approved"] == ["Channel", "SubjectID"]
    assert at.session_state["examples"] == []
    assert all("examples" not in row for row in at.session_state["schema_preview"])
    assert at.session_state["fresh_editor_rows"] == {
        "defaults": [],
        "filters": [],
        "calculations": [],
    }


@pytest.mark.unit
def test_field_scope_change_immediately_invalidates_ai_sharing_confirmation() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_SHARING_CONFIRMATION_STATE_KEY] = "confirmed"
        stale_widget_key = f"{page.AI_SHARING_CONFIRMATION_WIDGET_PREFIX}old-scope"
        st.session_state[stale_widget_key] = True
        st.session_state["ai_studio_copilot_history"] = [
            {"role": "assistant", "content": "Observed alice@example.com"}
        ]
        st.session_state["ai_studio_copilot_questions"] = [{"question": "Keep it?"}]
        st.session_state["ai_studio_copilot_last_prompt"] = "alice@example.com"
        st.session_state["ai_studio_copilot_queued_message"] = "alice@example.com"
        page._invalidate_ai_sharing_confirmation_if_scope_changed(
            previous_approved_fields=["Channel", "Email"],
            previous_example_fields=["Email"],
            approved_fields=["Channel", "Email"],
            example_fields=[],
        )
        st.session_state["stale_consent_widget_present"] = stale_widget_key in st.session_state
        st.session_state["queued_copilot_message_present"] = (
            "ai_studio_copilot_queued_message" in st.session_state
        )
        st.session_state["post_revoke_prompt"] = page.prompt_for_copilot(
            step="9. Metrics",
            user_message="Add a metric.",
            history=st.session_state["ai_studio_copilot_history"],
            user_goals="",
            approved_schema=[{"column": "Email", "dtype": "String"}],
            approved_fields=["Channel", "Email"],
            hidden_fields=[],
            current_draft={},
        )

    at = AppTest.from_function(app).run()

    assert not at.exception
    assert at.session_state[ai_config_studio_page.AI_SHARING_CONFIRMATION_STATE_KEY] == ""
    assert at.session_state["stale_consent_widget_present"] is False
    assert at.session_state["ai_studio_copilot_history"] == []
    assert at.session_state["ai_studio_copilot_questions"] == []
    assert at.session_state["ai_studio_copilot_last_prompt"] == ""
    assert at.session_state["queued_copilot_message_present"] is False
    assert "alice@example.com" not in at.session_state["post_revoke_prompt"]


@pytest.mark.unit
def test_ai_calls_require_sample_scoped_data_sharing_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    calls: list[str] = []

    def fake_call_litellm(settings: AICallSettings, prompt: str, **kwargs: object) -> str:
        calls.append(prompt)
        return "ok"

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ai import AICallSettings  # noqa: PLC0415
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state.setdefault("ai_studio_sample_identity", "sample-one")
        st.session_state.setdefault("ai_studio_sample_name", "sample.csv")
        st.session_state.setdefault("ai_studio_ai_model", "openai/gpt-test")
        st.session_state.setdefault("ai_studio_ai_provider", "openai")
        st.session_state.setdefault(
            "ai_studio_ai_api_base",
            "https://user:private-password@internal.example/v1?token=secret",
        )
        st.session_state.setdefault("ai_studio_approved_fields", ["Channel", "SubjectID"])
        st.session_state.setdefault("ai_studio_example_fields", [])
        contract = page._ai_sharing_contract([])
        st.session_state["explicit_empty_scope"] = contract["approved_fields"]
        st.session_state["sharing_contract"] = contract

        def switch_sample() -> None:
            current = st.session_state["ai_studio_sample_identity"]
            st.session_state["ai_studio_sample_identity"] = (
                "sample-two" if current == "sample-one" else "sample-one"
            )

        def seed_copilot_history() -> None:
            st.session_state["ai_studio_copilot_history"] = [
                {"role": "assistant", "content": "Echoed PRIVATE-SAMPLE-VALUE"}
            ]

        page._render_ai_data_sharing_confirmation(["Channel", "SubjectID"])
        confirmed = page._ai_data_sharing_confirmed(["Channel", "SubjectID"])
        if st.button("Run model", disabled=not confirmed):
            st.session_state["result"] = page._call_litellm_for_current_sample(
                AICallSettings(model="openai/gpt-test"),
                "Generate a draft",
                approved_fields=["Channel", "SubjectID"],
            )
        st.button("Switch sample", on_click=switch_sample)
        st.button("Seed Copilot history", on_click=seed_copilot_history)

    at = AppTest.from_function(app).run()

    assert not at.exception
    assert at.session_state["explicit_empty_scope"] == []
    assert at.session_state["sharing_contract"]["destination"] == "Custom endpoint configured"
    assert "private-password" not in json.dumps(at.session_state["sharing_contract"])
    assert "token=secret" not in json.dumps(at.session_state["sharing_contract"])
    assert any("Custom endpoint configured" in caption.value for caption in at.caption)
    assert any(
        "AI generation, Copilot, repair, and report refresh remain disabled" in caption.value
        for caption in at.caption
    )
    assert next(button for button in at.button if button.label == "Run model").disabled
    assert calls == []

    consent = next(
        checkbox
        for checkbox in at.checkbox
        if checkbox.label.startswith("Review (changed) sharing scope")
    )
    consent.check().run()
    assert any(
        "report refresh are enabled for this confirmed scope" in caption.value
        for caption in at.caption
    )
    run_button = next(button for button in at.button if button.label == "Run model")
    assert not run_button.disabled
    run_button.click().run()
    assert calls == ["Generate a draft"]
    assert at.session_state["result"] == "ok"

    next(
        button for button in at.button if button.label == "Revoke AI sharing confirmation"
    ).click().run()
    assert not at.exception
    assert next(button for button in at.button if button.label == "Run model").disabled
    next(
        checkbox
        for checkbox in at.checkbox
        if checkbox.label.startswith("Review (changed) sharing scope")
    ).check().run()

    next(button for button in at.button if button.label == "Seed Copilot history").click().run()
    assert "PRIVATE-SAMPLE-VALUE" in str(at.session_state["ai_studio_copilot_history"])

    next(button for button in at.button if button.label == "Switch sample").click().run()
    assert next(button for button in at.button if button.label == "Run model").disabled
    assert calls == ["Generate a draft"]
    assert at.session_state["ai_studio_copilot_history"] == []

    # Returning to a previously confirmed scope does not reactivate stale consent.
    next(button for button in at.button if button.label == "Switch sample").click().run()
    assert next(button for button in at.button if button.label == "Run model").disabled
    assert calls == ["Generate a draft"]


@pytest.mark.unit
def test_ai_studio_routes_every_litellm_call_through_sample_consent_guard() -> None:
    import ast  # noqa: PLC0415 - focused source guard
    import inspect  # noqa: PLC0415 - focused source guard

    class CallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.callers: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "call_litellm":
                self.callers.append(self.function_stack[-1] if self.function_stack else "")
            self.generic_visit(node)

    visitor = CallVisitor()
    visitor.visit(ast.parse(inspect.getsource(ai_config_studio_page)))

    assert visitor.callers == ["_call_litellm_for_current_sample"]


@pytest.mark.unit
def test_apply_field_approval_edits_merges_visible_rows_over_state() -> None:
    approved, examples = _apply_field_approval_edits(
        [
            {"Approve": False, "Send To AI": False, "Column": "Channel"},
            {"Approve": True, "Send To AI": True, "Column": "Plan"},
            {"Approve": False, "Send To AI": True, "Column": "SubjectID"},
        ],
        available_fields=["Channel", "Hidden", "Plan", "SubjectID"],
        required_fields=["SubjectID"],
        approved_fields=["Channel", "Hidden", "SubjectID"],
        example_fields=["Channel", "Hidden"],
    )

    # Hidden rows keep membership, required fields stay approved, and
    # example sharing is limited to approved fields.
    assert approved == ["Hidden", "Plan", "SubjectID"]
    assert examples == ["Hidden", "Plan", "SubjectID"]


@pytest.mark.unit
def test_apply_field_approval_edits_accepts_legacy_share_column() -> None:
    approved, examples = _apply_field_approval_edits(
        [{"Approve": True, "Share Sample Values": True, "Column": "Channel"}],
        available_fields=["Channel"],
        required_fields=[],
        approved_fields=[],
        example_fields=[],
    )

    assert approved == ["Channel"]
    assert examples == ["Channel"]


@pytest.mark.unit
def test_field_approval_editor_commits_one_checkbox_event_on_first_click() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        editor_key = "test_field_approval_editor"
        stale_widget_key = f"{page.AI_SHARING_CONFIRMATION_WIDGET_PREFIX}confirmed"
        if not st.session_state.get("test_field_approval_initialized"):
            st.session_state["test_field_approval_initialized"] = True
            st.session_state[editor_key] = {
                "edited_rows": {
                    0: {"Send To AI": True},
                    1: {"Approve": False},
                }
            }
            st.session_state["ai_studio_approved_fields"] = [
                "Channel",
                "Sensitive",
                "SubjectID",
            ]
            st.session_state["ai_studio_example_fields"] = []
            st.session_state["ai_studio_group_by_fields"] = ["Channel", "Sensitive"]
            st.session_state[page.AI_SHARING_CONFIRMATION_STATE_KEY] = "confirmed-scope"
            st.session_state[stale_widget_key] = True
        st.button(
            "Apply one editor event",
            on_click=page._on_field_approval_editor_change,
            args=(
                editor_key,
                ("Channel", "Sensitive", "SubjectID"),
                ("Channel", "Sensitive", "SubjectID"),
                ("SubjectID",),
            ),
        )
        st.session_state["stale_consent_widget_present"] = stale_widget_key in st.session_state

    rendered = AppTest.from_function(app).run()
    committed = (
        next(button for button in rendered.button if button.label == "Apply one editor event")
        .click()
        .run()
    )

    assert not committed.exception
    assert committed.session_state["ai_studio_approved_fields"] == ["Channel", "SubjectID"]
    assert committed.session_state["ai_studio_example_fields"] == ["Channel"]
    assert committed.session_state["ai_studio_group_by_fields"] == ["Channel"]
    assert committed.session_state[ai_config_studio_page.AI_SHARING_CONFIRMATION_STATE_KEY] == ""
    assert committed.session_state["stale_consent_widget_present"] is False


@pytest.mark.unit
def test_editor_frame_uses_stable_text_and_boolean_columns() -> None:
    frame = builder.editor_frame(
        [
            {
                "Name": "Margin",
                "Mode": "Subtract",
                "Left": "Revenue",
                "Right Kind": "Field",
                "Right": "Cost",
                "Expression": None,
                "Enabled": False,
            }
        ],
        ["Name", "Mode", "Left", "Right Kind", "Right", "Expression", "Enabled"],
        _blank_ai_calculation_row,
    )

    assert frame.to_dicts() == [
        {
            "Name": "Margin",
            "Mode": "Subtract",
            "Left": "Revenue",
            "Right Kind": "Field",
            "Right": "Cost",
            "Expression": "",
            "Enabled": False,
        }
    ]
    assert frame.schema["Enabled"] == pl.Boolean


@pytest.mark.unit
def test_generated_processor_id_is_derived_from_source_id() -> None:
    assert _generated_processor_id("ih") == "ih_engagement"
    assert _generated_processor_id("Customer Events") == "Customer_Events_engagement"


@pytest.mark.unit
def test_deterministic_studio_steps_map_to_ai_studio_stage_indices() -> None:
    assert _studio_steps(ai_calls_enabled=True) == STEPS
    assert _studio_steps(ai_calls_enabled=False) == DETERMINISTIC_STEPS
    assert _normalize_studio_step("7. AI Draft", DETERMINISTIC_STEPS) == "7. Draft"
    assert _normalize_studio_step("10. Reports", STEPS) == "10. AI Reports"


@pytest.mark.unit
def test_deterministic_report_refresh_builds_valid_dashboard_tiles() -> None:
    draft = _base_draft()
    working = pl.DataFrame(
        {
            "Day": ["2026-01-01"],
            "Channel": ["Web"],
            "CTR": [0.12],
        }
    )

    dashboards = _deterministic_dashboards_from_metrics(
        draft,
        working,
        approved_fields=["Day", "Channel"],
    )
    refreshed = {**draft, "dashboards": dashboards}
    ok, issues = validate_draft_catalog(refreshed)

    assert ok, issues
    assert dashboards["dashboards"][0]["id"] == "dashboard_builder_overview"
    assert (
        dashboards["dashboards"][0]["pages"][0]["id"] == "dashboard_builder_overview_page_metrics"
    )
    tiles = dashboards["dashboards"][0]["pages"][0]["tiles"]
    assert tiles == [
        {
            "id": "metrics_tile_ctr",
            "title": "CTR",
            "metric": "CTR",
            "chart": "line",
            "x": "Day",
            "y": "CTR",
            "color": "Channel",
        }
    ]
    assert dashboards == _deterministic_dashboards_from_metrics(
        draft,
        working,
        approved_fields=["Day", "Channel"],
    )


@pytest.mark.unit
def test_generated_report_ids_preserve_valid_existing_ids() -> None:
    dashboards = _with_generated_report_ids(
        {
            "theme": {},
            "dashboards": [
                {
                    "id": "manual_dashboard",
                    "title": "Sales Overview",
                    "pages": [
                        {
                            "id": "manual_page",
                            "title": "Engagement",
                            "tiles": [
                                {
                                    "id": "manual_tile",
                                    "title": "CTR Trend",
                                    "metric": "CTR",
                                    "chart": "line",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )

    dashboard = dashboards["dashboards"][0]
    page = dashboard["pages"][0]
    tile = page["tiles"][0]
    assert dashboard["id"] == "manual_dashboard"
    assert page["id"] == "manual_page"
    assert tile["id"] == "manual_tile"
    assert tile["metric"] == "CTR"


@pytest.mark.unit
def test_generated_report_ids_are_stable_and_resolve_sibling_collisions() -> None:
    proposal = {
        "theme": {},
        "dashboards": [
            {
                "title": "Sales Overview",
                "pages": [
                    {
                        "title": "Engagement",
                        "tiles": [
                            {"title": "CTR Trend", "metric": "CTR", "chart": "line"},
                            {"title": "CTR Trend", "metric": "CTR", "chart": "bar"},
                        ],
                    }
                ],
            }
        ],
    }

    first = _with_generated_report_ids(proposal)
    second = _with_generated_report_ids(proposal)
    tiles = first["dashboards"][0]["pages"][0]["tiles"]

    assert first == second
    assert first["dashboards"][0]["id"] == "dashboard_sales_overview"
    assert first["dashboards"][0]["pages"][0]["id"] == "dashboard_sales_overview_page_engagement"
    assert tiles[0]["id"].endswith("_tile_ctr_trend")
    assert tiles[1]["id"] == f"{tiles[0]['id']}_2"


@pytest.mark.unit
def test_no_provider_report_baseline_has_three_pages_six_tiles_and_stable_ids() -> None:
    first = ai_config_studio_page._studio_baseline_dashboards("Day", ["Channel"])
    second = ai_config_studio_page._studio_baseline_dashboards("Day", ["Channel"])
    pages = first["dashboards"][0]["pages"]
    tiles = [tile for page in pages for tile in page["tiles"]]

    assert first == second
    assert [page["title"] for page in pages] == ["Engagement", "Volume", "Outcomes"]
    assert len(tiles) == 6
    assert {tile["metric"] for tile in tiles} == {
        "Studio_CTR",
        "Studio_Count",
        "Studio_Positive_Outcomes",
        "Studio_Negative_Outcomes",
    }
    assert len({tile["id"] for tile in tiles}) == 6
    assert all(page["time_filter"]["default"] == "all_time" for page in pages)


@pytest.mark.unit
def test_keep_selection_reconciles_new_removed_and_explicitly_rejected_ids() -> None:
    state: dict[str, object] = {}
    key = "keep"

    assert ai_config_studio_page._reconcile_keep_selection(
        state,
        key=key,
        options=["a", "b"],
        revision="one",
    ) == ["a", "b"]

    state[key] = ["a"]
    assert ai_config_studio_page._reconcile_keep_selection(
        state,
        key=key,
        options=["a", "b", "c"],
        revision="two",
    ) == ["a", "c"]
    assert state[f"{key}__rejected_ids"] == ["b"]

    state[key] = ["c"]
    assert ai_config_studio_page._reconcile_keep_selection(
        state,
        key=key,
        options=["b", "c", "d"],
        revision="three",
    ) == ["c", "d"]

    state[key] = ["b", "c", "d"]
    assert ai_config_studio_page._reconcile_keep_selection(
        state,
        key=key,
        options=["b", "c", "d"],
        revision="four",
    ) == ["b", "c", "d"]
    assert state[f"{key}__rejected_ids"] == ["a"]


@pytest.mark.unit
def test_keep_labels_are_human_first_with_stable_identity_context() -> None:
    draft = _base_draft()

    assert ai_config_studio_page._processor_choice_label(draft, "engagement").endswith(
        "— engagement"
    )
    assert ai_config_studio_page._draft_metric_choice_label(draft, "CTR").startswith("CTR ·")
    assert (
        ai_config_studio_page._tile_choice_label(draft, "overview/engagement/ctr")
        == "CTR · Engagement — overview/engagement/ctr"
    )


@pytest.mark.unit
def test_tile_inventory_rows_stay_parallel_to_tile_keys() -> None:
    draft = _base_draft()
    dashboard = draft["dashboards"]["dashboards"][0]
    dashboard["pages"][0]["tiles"].append({"title": "No Id", "metric": "CTR", "chart": "line"})

    keys = ai_studio.tile_keys(draft)
    rows = ai_config_studio_page._tile_inventory_rows(draft)

    assert len(rows) == len(keys)
    assert all(row["Report"] != "No Id" for row in rows)


@pytest.mark.unit
def test_tile_keep_table_selects_all_tiles_by_default_in_editor_state() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        # A stale selection from an earlier session must not uncheck rows:
        # the keep table starts fully checked for every draft revision.
        st.session_state["ai_studio_tiles_to_keep"] = []
        draft = {
            "metrics": {"metrics": {"ctr": {"kind": "formula", "source": "engagement"}}},
            "dashboards": {
                "dashboards": [
                    {
                        "id": "studio",
                        "title": "Studio Overview",
                        "pages": [
                            {
                                "id": "engagement",
                                "title": "Engagement",
                                "tiles": [
                                    {
                                        "id": "ctr_trend",
                                        "title": "CTR Trend",
                                        "metric": "ctr",
                                        "chart": "line",
                                        "x": "Day",
                                        "y": "ctr",
                                    },
                                    {
                                        "id": "ctr_bar",
                                        "title": "CTR By Dimension",
                                        "metric": "ctr",
                                        "chart": "bar",
                                        "x": "Channel",
                                        "y": "ctr",
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
        }
        page._render_tile_keep_table(draft, revision="rev-1")

    at = AppTest.from_function(app).run()

    assert not at.exception
    assert at.session_state["ai_studio_tiles_to_keep"] == [
        "studio/engagement/ctr_trend",
        "studio/engagement/ctr_bar",
    ]
    update = next(button for button in at.button if button.label == "Update Draft: Tile Selection")
    assert not update.disabled


@pytest.mark.unit
def test_catalog_approved_fields_infers_source_processor_and_metric_fields() -> None:
    fields = _catalog_approved_fields(_base_draft())

    assert {"OutcomeTime", "CustomerID", "Channel", "Outcome", "CTR"} <= set(fields)


@pytest.mark.unit
def test_rename_capitalize_mapping_uses_pega_aware_names() -> None:
    mapping = _rename_capitalize_mapping(["pyName", "pxOutcomeTime", "customer_id"])

    assert mapping == {
        "pyName": "Name",
        "pxOutcomeTime": "OutcomeTime",
        "customer_id": "Customer_ID",
    }


@pytest.mark.unit
def test_schema_sample_applies_rename_capitalize_before_later_steps() -> None:
    st.session_state.clear()
    st.session_state["ai_studio_rename_capitalize"] = True
    st.session_state["ai_studio_defaults"] = [
        {"Field": "Revenue", "Default Value": "1.5", "Enabled": True}
    ]
    st.session_state["ai_studio_filter_mode"] = "Rules"
    st.session_state["ai_studio_filter_rows"] = []
    st.session_state["ai_studio_calculations"] = []
    st.session_state["ai_studio_timestamp_format"] = "%Y-%m-%d"

    frame = pl.DataFrame(
        {
            "pyName": ["Offer"],
            "pxOutcomeTime": ["2026-01-01"],
            "pyRevenue": [None],
        }
    )

    schema = _schema_sample(frame)
    working, error = _working_sample(schema)

    assert error is None
    assert schema.columns == ["Name", "OutcomeTime", "Revenue"]
    assert {"Name", "OutcomeTime", "Revenue", "Day"} <= set(working.columns)
    assert {"pyName", "pxOutcomeTime", "pyRevenue"}.isdisjoint(working.columns)
    assert working.get_column("Revenue").to_list() == [1.5]
    assert st.session_state["ai_studio_rename_capitalize_enabled"] is True
    assert "ai_studio_rename_capitalize" not in st.session_state


@pytest.mark.unit
def test_rename_capitalize_toggle_updates_effective_schema_on_same_rerun() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        from pathlib import Path  # noqa: PLC0415 - isolated AppTest source
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import (  # noqa: PLC0415 - isolated AppTest source
            ai_config_studio as page,
        )

        raw = pl.DataFrame({"pyChannel": ["Web"], "pxOutcomeTime": ["2026-01-01"]})
        page._initialize_state(raw)
        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = False
        schema = page._schema_sample(raw)
        page._set_effective_schema_state(schema)
        context = SimpleNamespace(
            workspace=Path("."),
            catalog=SimpleNamespace(
                pipelines=SimpleNamespace(sources=[]),
                processors=SimpleNamespace(processors=[]),
            ),
        )
        page._sample_step(context, raw, schema)

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    assert rendered.session_state["ai_studio_effective_schema_columns"] == [
        "pyChannel",
        "pxOutcomeTime",
    ]
    raw_signature = rendered.session_state["ai_studio_effective_schema_signature"]

    enabled = rendered.toggle(key="ai_studio_rename_capitalize_transform").set_value(True).run()

    assert not enabled.exception
    assert enabled.session_state["ai_studio_rename_capitalize_enabled"] is True
    assert enabled.session_state["ai_studio_effective_schema_columns"] == [
        "Channel",
        "OutcomeTime",
    ]
    assert enabled.session_state["ai_studio_effective_schema_signature"] != raw_signature

    disabled = enabled.toggle(key="ai_studio_rename_capitalize_transform").set_value(False).run()

    assert not disabled.exception
    assert disabled.session_state["ai_studio_rename_capitalize_enabled"] is False
    assert disabled.session_state["ai_studio_effective_schema_columns"] == [
        "pyChannel",
        "pxOutcomeTime",
    ]
    assert disabled.session_state["ai_studio_effective_schema_signature"] == raw_signature


@pytest.mark.unit
def test_rename_capitalize_state_survives_when_sample_widget_is_not_rendered() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import (  # noqa: PLC0415 - isolated AppTest source
            ai_config_studio as page,
        )

        raw = pl.DataFrame({"pyChannel": ["Web"], "pxOutcomeTime": ["2026-01-01"]})
        page._initialize_state(raw)
        screen = st.radio("Screen", ["Sample", "Filters"], key="test_studio_screen")
        schema = page._schema_sample(raw)
        page._set_effective_schema_state(schema)
        if screen == "Sample":
            page._render_rename_capitalize_toggle()
        st.write(",".join(schema.columns))

    rendered = AppTest.from_function(app).run()
    enabled = rendered.toggle(key="ai_studio_rename_capitalize_transform").set_value(True).run()

    assert enabled.session_state["ai_studio_rename_capitalize_enabled"] is True
    assert enabled.session_state["ai_studio_effective_schema_columns"] == [
        "Channel",
        "OutcomeTime",
    ]

    hidden = enabled.radio(key="test_studio_screen").set_value("Filters").run()
    hidden = hidden.run()

    assert not hidden.exception
    assert not hidden.toggle
    assert hidden.session_state["ai_studio_rename_capitalize_enabled"] is True
    assert hidden.session_state["ai_studio_effective_schema_columns"] == [
        "Channel",
        "OutcomeTime",
    ]

    restored = hidden.radio(key="test_studio_screen").set_value("Sample").run()

    assert not restored.exception
    assert restored.toggle(key="ai_studio_rename_capitalize_transform").value is True


@pytest.mark.unit
def test_deterministic_demo_timestamp_plan_preprocesses_without_error() -> None:
    st.session_state.clear()
    st.session_state["ai_studio_defaults"] = []
    st.session_state["ai_studio_filter_mode"] = "Rules"
    st.session_state["ai_studio_filter_rows"] = []
    st.session_state["ai_studio_calculations"] = []
    st.session_state["ai_studio_timestamp_format"] = "%+"

    working, error = _working_sample(ai_config_studio_page._demo_sample())

    assert error is None
    assert working.schema["OutcomeTime"] == pl.Datetime(time_zone="UTC")
    assert {"Day", "Month", "Quarter", "Year"} <= set(working.columns)


@pytest.mark.unit
def test_schema_sample_uses_original_columns_when_rename_capitalize_is_off() -> None:
    st.session_state.clear()
    st.session_state["ai_studio_rename_capitalize"] = False

    frame = pl.DataFrame({"pyName": ["Offer"], "pxOutcomeTime": ["2026-01-01"]})

    schema = _schema_sample(frame)

    assert schema.columns == ["pyName", "pxOutcomeTime"]


@pytest.mark.unit
def test_filters_step_uses_effective_schema_as_field_source(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def capture_filters(frame: pl.DataFrame) -> None:
        captured["columns"] = list(frame.columns)

    monkeypatch.setattr(ai_config_studio_page, "_filters", capture_filters)
    raw = pl.DataFrame({"pyName": ["Offer"]})
    effective = pl.DataFrame({"Name": ["Offer"]})
    working = pl.DataFrame({"Name": ["Offer"], "Day": ["2026-01-01"]})

    _render_selected_step(object(), STEPS[3], raw, effective, working, [], None)

    assert captured["columns"] == ["Name"]


@pytest.mark.unit
def test_rename_capitalize_sync_remaps_filters_without_forced_rerun() -> None:
    st.session_state.clear()
    st.session_state["ai_studio_rename_capitalize"] = True
    st.session_state["ai_studio_rename_capitalize_applied"] = False
    st.session_state["ai_studio_filter_rows"] = [
        {"Field": "pyName", "Operator": "Equals", "Value": "Offer", "Enabled": True}
    ]
    st.session_state["ai_studio_raw_filter"] = "op: eq\ncolumn: pyName\nvalue: Offer\n"
    st.session_state["ai_studio_filter_editor"] = "stale"

    _sync_ai_rename_capitalize_state(pl.DataFrame({"pyName": ["Offer"]}))

    assert st.session_state["ai_studio_filter_rows"][0]["Field"] == "Name"
    assert "column: Name" in st.session_state["ai_studio_raw_filter"]
    assert "ai_studio_filter_editor" not in st.session_state
    assert "ai_studio_next_step" not in st.session_state
    assert st.session_state["ai_studio_rename_capitalize_applied"] is True


@pytest.mark.unit
def test_rename_capitalize_sync_remaps_back_to_original_schema() -> None:
    st.session_state.clear()
    st.session_state["ai_studio_rename_capitalize"] = False
    st.session_state["ai_studio_rename_capitalize_applied"] = True
    st.session_state["ai_studio_filter_rows"] = [
        {"Field": "Name", "Operator": "Equals", "Value": "Offer", "Enabled": True}
    ]
    st.session_state["ai_studio_raw_filter"] = "op: eq\ncolumn: Name\nvalue: Offer\n"

    _sync_ai_rename_capitalize_state(pl.DataFrame({"pyName": ["Offer"]}))

    assert st.session_state["ai_studio_filter_rows"][0]["Field"] == "pyName"
    assert "column: pyName" in st.session_state["ai_studio_raw_filter"]
    assert st.session_state["ai_studio_rename_capitalize_applied"] is False


@pytest.mark.unit
def test_rename_capitalize_rejects_raw_names_in_free_form_preprocessing() -> None:
    st.session_state.clear()
    st.session_state["ai_studio_rename_capitalize_enabled"] = True
    st.session_state["ai_studio_raw_schema_columns"] = ["pyChannel", "pxOutcomeTime"]
    st.session_state["ai_studio_defaults"] = [
        {"Field": "pyChannel", "Default Value": "Web", "Enabled": True}
    ]
    st.session_state["ai_studio_filter_mode"] = "Raw AST"
    st.session_state["ai_studio_raw_filter"] = "op: eq\ncolumn: pyChannel\nvalue: Web\n"
    st.session_state["ai_studio_calculations"] = [
        {
            "Name": "ChannelCopy",
            "Mode": "AST YAML",
            "Expression": "col: pyChannel",
            "Enabled": True,
        }
    ]

    issues = ai_config_studio_page._stale_preprocessing_field_name_issues()
    _working, error = _working_sample(
        pl.DataFrame({"Channel": ["Web"], "OutcomeTime": ["2026-01-01"]})
    )

    assert len(issues) == 3
    assert all("use effective field 'Channel'" in issue for issue in issues)
    assert error is not None
    assert "pyChannel" in error


@pytest.mark.unit
def test_ai_sharing_confirmation_is_not_requested_again_across_unchanged_steps() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state.setdefault("ai_studio_sample_identity", "sample-one")
        st.session_state.setdefault("ai_studio_sample_name", "sample.parquet")
        st.session_state.setdefault("ai_studio_ai_model", "openai/gpt-test")
        st.session_state.setdefault("ai_studio_ai_provider", "openai")
        steps = page.STEPS[:7]
        current = st.session_state.get("ai_studio_step", steps[0])
        queued = page._normalize_studio_step(
            st.session_state.pop("ai_studio_next_step", None), steps
        )
        if queued in steps:
            current = queued
            st.session_state["ai_studio_step"] = queued
            st.session_state["ai_studio_jump_step"] = queued
        page._render_ai_data_sharing_confirmation(["Channel", "OutcomeTime"])
        st.session_state["confirmed"] = page._ai_data_sharing_confirmed(["Channel", "OutcomeTime"])
        page._render_studio_step_navigation(current, steps)

    rendered = AppTest.from_function(app).run()
    consent = next(
        item
        for item in rendered.checkbox
        if item.label.startswith("Review (changed) sharing scope")
    )
    confirmed = consent.check().run()
    signature = confirmed.session_state[ai_config_studio_page.AI_SHARING_CONFIRMATION_STATE_KEY]
    assert all(
        not item.label.startswith("Review (changed) sharing scope") for item in confirmed.checkbox
    )

    continued = confirmed
    for expected_step in ai_config_studio_page.STEPS[1:7]:
        continued = (
            next(item for item in continued.button if item.label == "Continue").click().run()
        )
        assert not continued.exception
        assert continued.session_state["ai_studio_step"] == expected_step
        assert continued.session_state["confirmed"] is True
        assert (
            continued.session_state[ai_config_studio_page.AI_SHARING_CONFIRMATION_STATE_KEY]
            == signature
        )
        assert all(
            not item.label.startswith("Review (changed) sharing scope")
            for item in continued.checkbox
        )


@pytest.mark.unit
def test_load_ai_settings_config_reads_workspace_llm_defaults(tmp_path) -> None:
    config_path = tmp_path / "ai.yaml"
    config_path.write_text(
        """
ai:
  llm:
    provider: openai
    model: openai/gpt-5.5
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    reasoning_effort: high
    verbosity: low
    timeout_seconds: 120
""",
        encoding="utf-8",
    )

    path, config = _load_ai_settings_config(tmp_path)

    assert path == config_path
    assert config["model"] == "openai/gpt-5.5"
    assert config["api_key_env"] == "OPENAI_API_KEY"
    assert config["reasoning_effort"] == "high"
    assert config["verbosity"] == "low"
    assert config["timeout_seconds"] == 120


@pytest.mark.unit
def test_report_refresh_prompt_is_report_only() -> None:
    prompt = prompt_for_report_refresh(
        file_name="sample.csv",
        approved_schema=[{"column": "Channel", "dtype": "String"}],
        approved_fields=["Channel"],
        hidden_fields=["CustomerID"],
        current_draft=_base_draft(),
    )

    assert "Refresh only dashboards.yaml" in prompt
    assert "Do not change pipelines, processors, or metrics." in prompt
    assert "Hidden field count: 1" in prompt
    assert "CustomerID" not in prompt


@pytest.mark.unit
def test_section_name_diff_tracks_changed_tiles() -> None:
    base = _base_draft()
    changed = _base_draft()
    changed["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]["title"] = "CTR Trend"

    diff = section_name_diff(base, changed)

    assert diff["tiles"]["changed"] == ["overview/engagement/ctr"]
    assert diff["metrics"]["unchanged"] == ["CTR"]


@pytest.mark.unit
def test_call_litellm_omits_optional_generation_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "metrics: {}"}}]}

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)

    result = call_litellm(AICallSettings(model="openai/gpt-5.1"), "Return YAML")

    assert result == "metrics: {}"
    assert "temperature" not in captured
    assert "reasoning_effort" not in captured
    assert "verbosity" not in captured


@pytest.mark.unit
def test_call_litellm_sends_provider_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "metrics: {}"}}]}

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)

    result = call_litellm(
        AICallSettings(
            model="ollama/llama3.1",
            api_base="http://localhost:11434",
            custom_llm_provider="ollama",
            temperature=0.0,
            reasoning_effort="low",
            verbosity="high",
            timeout_seconds=12,
        ),
        "Return YAML",
    )

    assert result == "metrics: {}"
    assert captured["model"] == "ollama/llama3.1"
    assert captured["temperature"] == 0.0
    assert captured["reasoning_effort"] == "low"
    assert captured["verbosity"] == "high"
    assert captured["api_base"] == "http://localhost:11434"
    assert captured["custom_llm_provider"] == "ollama"
    assert captured["request_timeout"] == 12
    assert "api_key" not in captured


@pytest.mark.unit
def test_call_litellm_logs_metadata_without_prompts_or_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        return {"choices": [{"message": {"content": "metrics: {}"}}]}

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    result = call_litellm(
        AICallSettings(model="openai/gpt-5.1", api_key="secret-token"),
        "Return YAML with CTR",
        system_prompt="Return valid YAML only",
    )

    assert result == "metrics: {}"
    assert "LLM call started" in caplog.text
    assert "LLM call completed" in caplog.text
    assert "openai/gpt-5.1" in caplog.text
    assert "has_api_key': True" in caplog.text
    assert "Return valid YAML only" not in caplog.text
    assert "Return YAML with CTR" not in caplog.text
    assert "metrics: {}" not in caplog.text
    assert "secret-token" not in caplog.text


@pytest.mark.unit
def test_call_litellm_logs_failure_without_prompt_or_exception_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    with pytest.raises(RuntimeError, match=r"AI provider call failed \(RuntimeError\)") as error:
        call_litellm(AICallSettings(model="ollama/llama3.1"), "Plan CTR query")

    assert error.value.__context__ is None
    assert "provider unavailable" not in str(error.value)
    assert "LLM call started" in caplog.text
    assert "LLM call failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "Plan CTR query" not in caplog.text
    assert "provider unavailable" not in caplog.text


@pytest.mark.unit
def test_ai_refine_panel_holds_revision_in_pending_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    responses = iter(
        [
            "READY",
            """
metrics:
  CTR:
    source: engagement
    kind: formula
    expression:
      op: safe_div
      num: {col: Positives}
      den: {col: Count}
  Total:
    source: engagement
    kind: formula
    expression: {col: Count}
""",
        ]
    )

    def fake_call_litellm(settings: AICallSettings, prompt: str, **kwargs: object) -> str:
        return next(responses)

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_user_goals"] = "Track total volume."
        page._render_ai_refine_panel(draft, None, ["Channel"])

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()

    assert not at.exception
    change_request = next(widget for widget in at.text_area if widget.label == "Change Request")
    change_request.set_value("Add a Total metric with the event count.").run()
    next(widget for widget in at.button if widget.label == "Generate AI Revision").click().run()

    assert not at.exception
    pending = at.session_state["ai_studio_pending_draft"]
    assert sorted(pending["metrics"]["metrics"]) == ["CTR", "Total"]
    assert at.session_state["ai_studio_pending_kind"] == "revision"
    prompt = at.session_state["ai_studio_pending_prompt"]
    assert "Add a Total metric with the event count." in prompt
    assert "Track total volume." in prompt


@pytest.mark.unit
def test_workspace_save_bar_exposes_ready_and_published_states() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict, published: bool) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_reviewed_signature"] = page._draft_signature(draft)
        st.session_state["ai_studio_sample_source_plan"] = page.SampleSourcePlan(
            "CSV",
            "sample",
            "csv",
            "data",
            "sample.csv",
            production_ready=True,
        )
        st.session_state["ai_studio_reader_kind"] = "csv"
        st.session_state["ai_studio_reader_root"] = "data"
        st.session_state["ai_studio_file_pattern"] = "sample.csv"
        st.session_state["ai_studio_published_signature"] = (
            page._draft_signature(draft) if published else ""
        )
        page._render_workspace_save_bar(SimpleNamespace(workspace="."))

    ready = AppTest.from_function(app, kwargs={"draft": _base_draft(), "published": False}).run()
    published = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "published": True},
    ).run()

    assert not ready.exception
    assert ready.button[0].label == "Apply to workspace"
    assert not ready.button[0].disabled
    assert not published.exception
    assert published.button[0].label == "Applied"
    assert published.button[0].disabled


@pytest.mark.unit
def test_workspace_apply_exception_records_bounded_apply_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    events: list[dict[str, object]] = []

    def capture_event(_state: object, **kwargs: object) -> bool:
        events.append(kwargs)
        return True

    def fail_apply(_ctx: object, _draft: object) -> None:
        raise TimeoutError("PRIVATE-CUSTOMER-42")

    monkeypatch.setattr(ai_config_studio_page, "record_event", capture_event)
    monkeypatch.setattr(ai_config_studio_page, "_draft_requires_data_run", lambda *_args: False)
    monkeypatch.setattr(ai_config_studio_page, "_apply_draft", fail_apply)

    def app(draft: dict) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_reviewed_signature"] = page._draft_signature(draft)
        st.session_state["ai_studio_sample_source_plan"] = page.SampleSourcePlan(
            "CSV",
            "sample",
            "csv",
            "data",
            "sample.csv",
            production_ready=True,
        )
        st.session_state["ai_studio_reader_kind"] = "csv"
        st.session_state["ai_studio_reader_root"] = "data"
        st.session_state["ai_studio_file_pattern"] = "sample.csv"
        page._render_workspace_save_bar(SimpleNamespace(workspace="."))

    rendered = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    rendered = rendered.button[0].click().run()

    assert not rendered.exception
    failed = next(event for event in events if event["event"].value == "failed")  # type: ignore[union-attr]
    assert failed["workflow"].value == "ai_studio"  # type: ignore[union-attr]
    assert failed["stage"].value == "apply"  # type: ignore[union-attr]
    assert failed["outcome"].value == "timeout"  # type: ignore[union-attr]
    assert all("PRIVATE-CUSTOMER-42" not in str(value) for value in failed.values())


@pytest.mark.unit
def test_studio_status_panel_does_not_repeat_apply_action_on_early_steps() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_reviewed_signature"] = page._draft_signature(draft)
        page._studio_status_bar(
            SimpleNamespace(workspace="."),
            pl.DataFrame({"Channel": ["Web"]}),
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
            None,
        )

    rendered = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()

    assert not rendered.exception
    assert all(button.label != "Apply to workspace" for button in rendered.button)


def _base_draft() -> dict:
    return {
        "pipelines": {
            "version": 1,
            "workspace": "test",
            "sources": [
                {
                    "id": "ih",
                    "reader": {"kind": "csv", "file_pattern": "*.csv"},
                    "schema": {
                        "timestamp_column": "OutcomeTime",
                        "natural_key": ["CustomerID"],
                    },
                }
            ],
        },
        "processors": {
            "processors": [
                {
                    "id": "engagement",
                    "source": "ih",
                    "kind": "binary_outcome",
                    "dimensions": ["Channel"],
                    "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked"],
                        "negative_values": ["Impression"],
                    },
                }
            ]
        },
        "metrics": {
            "metrics": {
                "CTR": {
                    "source": "engagement",
                    "kind": "formula",
                    "expression": {
                        "op": "safe_div",
                        "num": {"col": "Positives"},
                        "den": {"col": "Count"},
                    },
                }
            }
        },
        "dashboards": {
            "dashboards": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "pages": [
                        {
                            "id": "engagement",
                            "title": "Engagement",
                            "tiles": [
                                {
                                    "id": "ctr",
                                    "title": "CTR",
                                    "metric": "CTR",
                                    "chart": "line",
                                    "x": "Day",
                                    "y": "CTR",
                                    "color": "Channel",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    }


@pytest.mark.unit
def test_deterministic_apply_readiness_is_complete_with_consistent_counts() -> None:
    st.session_state.clear()
    draft = _base_draft()
    signature = ai_config_studio_page._draft_signature(draft)
    st.session_state[ai_config_studio_page.AI_CALLS_ENABLED_STATE_KEY] = False
    st.session_state["ai_studio_draft_source"] = ai_config_studio_page.CATALOG_DRAFT_SOURCE
    st.session_state["ai_studio_reviewed_signature"] = signature
    st.session_state["ai_studio_pending_draft"] = None

    readiness = ai_config_studio_page._studio_apply_readiness(
        SimpleNamespace(catalog=model.Catalog.model_validate(draft)),
        draft,
        ["Channel", "Outcome"],
        None,
        ai_calls_enabled=False,
    )

    assert readiness.apply_ready
    assert readiness.export_ready
    assert readiness.blocker_count == 0
    assert readiness.warning_count == 0
    assert readiness.artifact_counts == {
        "Data": "1 source(s) · 2 approved field(s)",
        "Processor": "1 processor(s)",
        "Metric": "1 metric(s)",
        "Report": "1 dashboard(s) · 1 tile(s)",
        "Provider": "Deterministic mode",
        "Runtime": "1 accepted revision",
    }
    assert all(
        ai_config_studio_page._readiness_area_status(area, readiness) == "Complete"
        for area in ai_config_studio_page.STUDIO_READINESS_AREAS
    )


@pytest.mark.unit
def test_invalid_schema_snapshot_is_not_reported_as_reviewed_or_published() -> None:
    st.session_state.clear()
    draft = _base_draft()
    source = draft["pipelines"]["sources"][0]
    source["schema"] = {
        "timestamp_column": "pxOutcomeTime",
        "natural_key": ["pyCustomerID", "pyChannel"],
    }
    source["transforms"] = [{"kind": "rename_capitalize"}]
    draft["processors"]["processors"][0]["dimensions"] = ["pyChannel"]
    approved = ["Channel", "CustomerID", "Outcome", "OutcomeTime"]
    signature = ai_config_studio_page._draft_signature(draft)
    st.session_state["ai_studio_draft_source"] = ""
    st.session_state["ai_studio_source_id"] = "ih"
    st.session_state["ai_studio_approved_fields"] = approved
    st.session_state["ai_studio_effective_schema_columns"] = approved
    st.session_state["ai_studio_effective_schema_signature"] = "schema-a"
    st.session_state["ai_studio_reviewed_signature"] = signature
    st.session_state["ai_studio_published_signature"] = signature
    st.session_state["ai_studio_pending_draft"] = None

    readiness = ai_config_studio_page._studio_apply_readiness(
        SimpleNamespace(catalog=model.Catalog.model_validate(draft)),
        draft,
        approved,
        None,
        ai_calls_enabled=False,
    )

    assert not readiness.apply_ready
    assert not readiness.export_ready
    assert readiness.last_changes["Runtime"] == f"Accepted revision {signature[:12]}"
    assert "already applied" not in readiness.apply_disabled_reason
    assert st.session_state["ai_studio_reviewed_signature"] == ""


@pytest.mark.unit
def test_status_bar_keeps_invalid_review_and_workspace_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    badges: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ai_config_studio_page.components,
        "status_badge",
        lambda label, status, **_kwargs: badges.append((label, status)),
    )

    def app(draft: dict) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        approved = ["Channel", "CustomerID", "Outcome", "OutcomeTime"]
        signature = page._draft_signature(draft)
        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = False
        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_draft_source"] = ""
        st.session_state["ai_studio_source_id"] = "ih"
        st.session_state["ai_studio_approved_fields"] = approved
        st.session_state["ai_studio_effective_schema_columns"] = approved
        st.session_state["ai_studio_effective_schema_signature"] = "schema-a"
        st.session_state["ai_studio_reviewed_signature"] = signature
        st.session_state["ai_studio_published_signature"] = signature
        st.session_state["ai_studio_pending_draft"] = None
        sample = pl.DataFrame(
            {
                "Channel": ["Web"],
                "CustomerID": ["C-1"],
                "Outcome": ["Clicked"],
                "OutcomeTime": ["2026-07-01T09:00:00Z"],
            }
        )
        page._studio_status_bar(SimpleNamespace(workspace="."), sample, sample, approved, None)

    draft = _base_draft()
    source = draft["pipelines"]["sources"][0]
    source["schema"] = {
        "timestamp_column": "pxOutcomeTime",
        "natural_key": ["pyCustomerID", "pyChannel"],
    }
    source["transforms"] = [{"kind": "rename_capitalize"}]
    draft["processors"]["processors"][0]["dimensions"] = ["pyChannel"]

    rendered = AppTest.from_function(app, kwargs={"draft": draft}).run()

    assert not rendered.exception
    assert dict(badges)["Validation"] == "blocked"
    assert dict(badges)["Review"] == "pending"
    assert dict(badges)["Workspace"] == "pending"
    assert rendered.session_state["ai_studio_reviewed_signature"] == ""


@pytest.mark.unit
def test_provider_readiness_groups_blockers_and_jump_preserves_committed_state() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    invalid_draft = _base_draft()
    invalid_draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]["metric"] = "MissingMetric"

    def app(draft: dict, catalog_payload: dict) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.config import model as config_model  # noqa: PLC0415
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        if "qa_readiness_seeded" not in st.session_state:
            st.session_state["qa_readiness_seeded"] = True
            st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
            st.session_state["ai_studio_draft_source"] = page.CATALOG_DRAFT_SOURCE
            st.session_state["ai_studio_pending_draft"] = {"proposal": "waiting"}
            st.session_state["qa_committed_editor"] = {"saved": True}
        readiness = page._studio_apply_readiness(
            SimpleNamespace(catalog=config_model.Catalog.model_validate(catalog_payload)),
            draft,
            ["Channel"],
            None,
            ai_calls_enabled=True,
        )
        page._render_apply_readiness(readiness)

    rendered = AppTest.from_function(
        app,
        kwargs={"draft": invalid_draft, "catalog_payload": _base_draft()},
    ).run()

    assert not rendered.exception
    assert any("2 blocker(s)" in item.value for item in rendered.markdown)
    assert [expander.label.split(" · ", 1)[0] for expander in rendered.expander] == list(
        ai_config_studio_page.STUDIO_READINESS_AREAS
    )
    report_error = next(item for item in rendered.error if "MissingMetric" in item.value)
    assert "**Object/path:**" in report_error.value
    assert "**Current safe value:**" in report_error.value
    assert "**Expected contract:**" in report_error.value
    assert "**Remediation:**" in report_error.value
    report_jump = next(button for button in rendered.button if button.label == "Jump to fix")
    rendered = report_jump.click().run()

    assert not rendered.exception
    assert rendered.session_state["ai_studio_next_step"] == STEPS[10]
    assert rendered.session_state["ai_studio_step"] == STEPS[10]
    assert rendered.session_state["qa_committed_editor"] == {"saved": True}


@pytest.mark.unit
def test_apply_and_export_disabled_reasons_are_adjacent_when_no_draft() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(catalog_payload: dict) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import polars as pl  # noqa: PLC0415
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.config import model as config_model  # noqa: PLC0415
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = False
        st.session_state["ai_studio_draft"] = None
        st.session_state["ai_studio_pending_draft"] = None
        page._save_export(
            SimpleNamespace(
                catalog=config_model.Catalog.model_validate(catalog_payload),
                validation=SimpleNamespace(issues=[], ok=True),
                workspace=".",
            ),
            pl.DataFrame({"Channel": ["Web"]}),
            [],
            None,
        )

    rendered = AppTest.from_function(app, kwargs={"catalog_payload": _base_draft()}).run()

    assert not rendered.exception
    apply_button = next(
        button for button in rendered.button if button.label == "Apply to workspace"
    )
    assert apply_button.disabled
    assert any("Apply status: Apply is unavailable" in item.value for item in rendered.caption)
    assert any("Export is unavailable" in item.value for item in rendered.warning)
    assert not rendered.get("download_button")


def _catalog_with_extras() -> model.Catalog:
    catalog_payload = _base_draft()
    catalog_payload["pipelines"]["sources"].append(
        {
            "id": "holdings",
            "reader": {"kind": "csv", "file_pattern": "holdings/*.csv"},
            "schema": {"timestamp_column": "OutcomeTime", "natural_key": ["CustomerID"]},
        }
    )
    catalog_payload["metrics"]["metrics"]["Revenue"] = {
        "source": "engagement",
        "kind": "formula",
        "expression": {"col": "Count"},
    }
    return model.Catalog.model_validate(catalog_payload)


@pytest.mark.unit
def test_builder_source_addition_draft_is_additive_and_namespaces_generated_ids() -> None:
    active = _base_draft()
    candidate = copy.deepcopy(_base_draft())
    candidate["pipelines"]["sources"][0]["id"] = "sample"
    candidate["processors"]["processors"][0]["id"] = "sample_engagement"
    candidate["processors"]["processors"][0]["source"] = "sample"
    candidate["metrics"]["metrics"]["CTR"]["source"] = "sample_engagement"

    merged = ai_config_studio_page._builder_source_addition_draft(
        SimpleNamespace(catalog=model.Catalog.model_validate(active)),
        candidate,
    )

    assert [source["id"] for source in merged["pipelines"]["sources"]] == ["ih", "sample"]
    assert [processor["id"] for processor in merged["processors"]["processors"]] == [
        "engagement",
        "sample_engagement",
    ]
    assert "CTR" in merged["metrics"]["metrics"]
    added_metric_ids = set(merged["metrics"]["metrics"]) - {"CTR"}
    assert added_metric_ids == {"sample_metric_ctr"}
    assert merged["metrics"]["metrics"]["sample_metric_ctr"]["source"] == "sample_engagement"
    assert [dashboard["id"] for dashboard in merged["dashboards"]["dashboards"]] == [
        "overview",
        "sample_dashboard_overview",
    ]
    added_tile = merged["dashboards"]["dashboards"][1]["pages"][0]["tiles"][0]
    assert added_tile["metric"] == "sample_metric_ctr"
    assert added_tile["y"] == "sample_metric_ctr"
    ok, issues = validate_draft_catalog(merged)
    assert ok, issues


@pytest.mark.unit
def test_builder_source_addition_rejects_duplicate_source_without_mutating_candidate() -> None:
    active = _base_draft()
    candidate = copy.deepcopy(_base_draft())
    before = copy.deepcopy(candidate)

    with pytest.raises(ValueError, match="never edits an existing source implicitly"):
        ai_config_studio_page._builder_source_addition_draft(
            SimpleNamespace(catalog=model.Catalog.model_validate(active)),
            candidate,
        )

    assert candidate == before


@pytest.mark.unit
def test_builder_source_handoff_is_deterministic_sample_first_and_preserves_journey(
    tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    draft = _base_draft()
    builder.write_pipelines_definition(tmp_path, draft["pipelines"])
    builder.write_processors_definition(tmp_path, draft["processors"])
    builder.write_metrics_definition(tmp_path, draft["metrics"])
    builder.write_dashboards_definition(tmp_path, draft["dashboards"])

    def app(workspace: str) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.instrumentation import (  # noqa: PLC0415
            AuthoringWorkflow,
            start_journey,
        )
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.query_params["mode"] = "deterministic"
        st.query_params["from"] = "configuration_builder"
        st.query_params["intent"] = "add_source"
        st.query_params["return_to"] = "configuration_builder"
        st.session_state["qa_builder_journey"] = start_journey(
            st.session_state,
            workflow=AuthoringWorkflow.BUILDER,
        )
        page.render(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run(timeout=15)

    assert not rendered.exception
    assert (
        rendered.session_state["vs_authoring_journey_id"]
        == rendered.session_state["qa_builder_journey"]
    )
    assert rendered.session_state["vs_authoring_journey_workflow"] == "builder"
    assert rendered.session_state[ai_config_studio_page.AI_CALLS_ENABLED_STATE_KEY] is False
    assert rendered.session_state["ai_studio_active_workspace_name"] == "test"
    links = {item.label: item.url for item in rendered.get("link_button")}
    assert links["Cancel and return to Builder"].endswith("ai_studio_source_cancelled")
    assert not any("review the current catalog draft" in str(item.value) for item in rendered.info)


@pytest.mark.unit
def test_builder_source_receipt_returns_explicitly_after_apply() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.query_params["mode"] = "deterministic"
        st.query_params["from"] = "configuration_builder"
        st.query_params["intent"] = "add_source"
        st.session_state["ai_studio_outcome_receipt"] = {
            "revision": "abc123",
            "applied": True,
            "requires_data_run": True,
            "source_count": 2,
        }
        page._render_outcome_receipt()

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    links = {item.label: item.url for item in rendered.get("link_button")}
    assert links["Return to Configuration Builder"].endswith("ai_studio_source_applied")
    assert any("run its data separately" in str(item.value) for item in rendered.caption)


@pytest.mark.unit
def test_workspace_replacement_impact_lists_objects_the_draft_would_remove() -> None:
    impact = ai_config_studio_page._workspace_replacement_impact(
        _catalog_with_extras(), _base_draft()
    )

    assert impact == {"sources": ["holdings"], "metrics": ["Revenue"]}
    assert ai_config_studio_page._workspace_replacement_impact(None, _base_draft()) == {}
    assert (
        ai_config_studio_page._workspace_replacement_impact(
            model.Catalog.model_validate(_base_draft()), _base_draft()
        )
        == {}
    )


@pytest.mark.unit
def test_workspace_apply_requires_explicit_replacement_consent() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict, catalog_payload: dict, confirm: bool) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.config import model as config_model  # noqa: PLC0415
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_reviewed_signature"] = page._draft_signature(draft)
        st.session_state["ai_studio_sample_source_plan"] = page.SampleSourcePlan(
            "CSV",
            "sample",
            "csv",
            "data",
            "sample.csv",
            production_ready=True,
        )
        st.session_state["ai_studio_reader_kind"] = "csv"
        st.session_state["ai_studio_reader_root"] = "data"
        st.session_state["ai_studio_file_pattern"] = "sample.csv"
        st.session_state["ai_studio_published_signature"] = ""
        if confirm:
            st.session_state[page.AI_REPLACEMENT_CONFIRM_STATE_KEY] = page._draft_signature(draft)
        page._render_workspace_save_bar(
            SimpleNamespace(
                workspace=".",
                catalog=config_model.Catalog.model_validate(catalog_payload),
            )
        )

    catalog_payload = _base_draft()
    catalog_payload["metrics"]["metrics"]["Revenue"] = {
        "source": "engagement",
        "kind": "formula",
        "expression": {"col": "Count"},
    }

    blocked = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "catalog_payload": catalog_payload, "confirm": False},
    ).run()
    confirmed = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "catalog_payload": catalog_payload, "confirm": True},
    ).run()

    assert not blocked.exception
    apply_button = next(b for b in blocked.button if b.label == "Apply to workspace")
    assert apply_button.disabled
    assert any("removes existing" in str(w.value) for w in blocked.warning)
    assert any("Remove these existing objects" in c.label for c in blocked.checkbox)

    assert not confirmed.exception
    apply_button = next(b for b in confirmed.button if b.label == "Apply to workspace")
    assert not apply_button.disabled
