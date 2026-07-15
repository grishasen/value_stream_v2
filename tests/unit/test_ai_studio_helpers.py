"""AI configuration studio helper tests."""

from __future__ import annotations

import logging
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
    validation_trace_for_repair,
)
from valuestream.config import model
from valuestream.recipes import (
    instantiate_metric,
    instantiate_tile,
    load_builtin_kpi_recipes,
    processor_with_recipe_states,
    recipe_readiness,
)
from valuestream.ui import builder, forms, recipe_library
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

    def fail_validation(_workspace: object) -> None:
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
        hidden_fields=["CustomerID"],
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
    assert "Do not emit legacy TOML-only settings such as metrics.global_filters" in prompt
    assert "Every report/dashboard tile metric exists in metrics." in prompt
    assert "Output valid YAML only." in prompt
    assert "Return valid YAML only. Do not wrap the answer in prose or Markdown fences." in prompt


@pytest.mark.unit
def test_expression_prompt_dictionary_covers_the_closed_dsl() -> None:
    expression_ast = ai_studio.catalog_prompt_dictionaries()["expression_ast"]
    prompted_ops = {
        op
        for form in expression_ast["operator_forms"].values()
        for op in form["ops"]
    }

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
        hidden_fields=[],
        current_draft=_base_draft(),
        validation_issues=[
            "dashboards[overview].pages[engagement].tiles[ctr].metric: unknown metric 'Missing'"
        ],
        validation_trace="Traceback (most recent call last):\nValidationError: bad draft",
    )

    assert "Validation errors to fix:" in prompt
    assert "unknown metric" in prompt
    assert "Missing" in prompt
    assert "Validation exception traceback, if available:" in prompt
    assert "Traceback (most recent call last):" in prompt
    assert "ValidationError: bad draft" in prompt


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
def test_deterministic_report_refresh_builds_valid_dashboard_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffixes = iter(
        [
            "1111111111111111",
            "2222222222222222",
            "3333333333333333",
        ]
    )
    calls: list[int] = []

    def fake_token_hex(byte_count: int) -> str:
        calls.append(byte_count)
        return next(suffixes)

    monkeypatch.setattr(builder.secrets, "token_hex", fake_token_hex)
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
    assert dashboards["dashboards"][0]["id"] == "builder_overview_2222222222222222"
    assert dashboards["dashboards"][0]["pages"][0]["id"] == "metrics_3333333333333333"
    tiles = dashboards["dashboards"][0]["pages"][0]["tiles"]
    assert tiles == [
        {
            "id": "ctr_1111111111111111",
            "title": "CTR",
            "metric": "CTR",
            "chart": "line",
            "x": "Day",
            "y": "CTR",
            "color": "Channel",
        }
    ]
    assert calls == [8, 8, 8]


@pytest.mark.unit
def test_generated_report_ids_replace_model_supplied_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    suffixes = iter(
        [
            "aaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbb",
            "cccccccccccccccc",
        ]
    )
    calls: list[int] = []

    def fake_token_hex(byte_count: int) -> str:
        calls.append(byte_count)
        return next(suffixes)

    monkeypatch.setattr(builder.secrets, "token_hex", fake_token_hex)

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
    assert dashboard["id"] == "sales_overview_aaaaaaaaaaaaaaaa"
    assert page["id"] == "engagement_bbbbbbbbbbbbbbbb"
    assert tile["id"] == "ctr_trend_cccccccccccccccc"
    assert tile["metric"] == "CTR"
    assert calls == [8, 8, 8]


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
    assert "CustomerID" in prompt


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
def test_call_litellm_logs_prompts_and_response(
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
    assert "Return valid YAML only" in caplog.text
    assert "Return YAML with CTR" in caplog.text
    assert "LLM call completed" in caplog.text
    assert "metrics: {}" in caplog.text
    assert "has_api_key': True" in caplog.text
    assert "secret-token" not in caplog.text


@pytest.mark.unit
def test_call_litellm_logs_failure_with_prompt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        call_litellm(AICallSettings(model="ollama/llama3.1"), "Plan CTR query")

    assert "LLM call started" in caplog.text
    assert "Plan CTR query" in caplog.text
    assert "LLM call failed" in caplog.text


@pytest.mark.unit
def test_ai_refine_panel_holds_revision_in_pending_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def fake_call_litellm(settings: AICallSettings, prompt: str, **kwargs: object) -> str:
        return """
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

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
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
    assert ready.button[0].label == "Save draft"
    assert not ready.button[0].disabled
    assert not published.exception
    assert published.button[0].label == "Saved"
    assert published.button[0].disabled


@pytest.mark.unit
def test_studio_status_panel_hosts_compact_save_action() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict) -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        page._studio_status_bar(
            SimpleNamespace(workspace="."),
            pl.DataFrame({"Channel": ["Web"]}),
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
            None,
        )

    rendered = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()

    assert not rendered.exception
    assert rendered.button[0].label == "Save draft"
    assert not rendered.button[0].disabled
    assert len(rendered.get("column")) == 2


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
