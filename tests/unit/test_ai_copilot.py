"""Copilot draft-operation, prompt, and coverage tests."""

from __future__ import annotations

import pytest

from valuestream.ai import (
    apply_draft_operations,
    draft_patches,
    merge_selected_draft_patches,
    parse_copilot_response,
    parse_coverage_response,
    prompt_for_copilot,
    prompt_for_coverage,
    run_copilot_tool_loop,
)
from valuestream.ai.copilot import (
    draft_patch_bundles,
    merge_selected_draft_patch_bundles,
)
from valuestream.ui.pages import ai_config_studio as ai_config_studio_page
from valuestream.ui.pages.ai_config_studio import (
    DETERMINISTIC_STEPS,
    STEPS,
    STUDIO_PHASES,
    _phase_for_step,
    _phase_step_options,
)


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
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    }


@pytest.mark.unit
def test_apply_operations_upserts_metric_and_tile() -> None:
    draft = _base_draft()
    operations = [
        {
            "op": "set_metric",
            "name": "Total",
            "metric": {"source": "engagement", "kind": "formula", "expression": {"col": "Count"}},
        },
        {
            "op": "set_tile",
            "dashboard": "overview",
            "page": "engagement",
            "tile": {"id": "total", "title": "Total", "metric": "Total", "chart": "kpi_card"},
        },
    ]

    updated, summaries = apply_draft_operations(draft, operations)

    assert "Total" in updated["metrics"]["metrics"]
    tiles = updated["dashboards"]["dashboards"][0]["pages"][0]["tiles"]
    assert [tile["id"] for tile in tiles] == ["ctr", "total"]
    assert summaries == ["Added metric 'Total'", "Added tile 'overview/engagement/total'"]
    assert "Total" not in draft["metrics"]["metrics"]


@pytest.mark.unit
def test_apply_operations_set_tile_creates_dashboard_and_page() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [
            {
                "op": "set_tile",
                "dashboard": "revenue_watch",
                "page": "weekly",
                "tile": {"id": "orders", "title": "Orders", "metric": "CTR", "chart": "kpi_card"},
            }
        ],
    )

    dashboards = updated["dashboards"]["dashboards"]
    created = next(item for item in dashboards if item["id"] == "revenue_watch")
    assert created["title"] == "Revenue Watch"
    assert created["pages"][0]["id"] == "weekly"
    assert created["pages"][0]["tiles"][0]["id"] == "orders"
    assert summaries == ["Added tile 'revenue_watch/weekly/orders'"]


@pytest.mark.unit
def test_apply_operations_remove_processor_cascades() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [{"op": "remove_processor", "id": "engagement"}],
    )

    assert updated["processors"]["processors"] == []
    assert updated["metrics"]["metrics"] == {}
    assert updated["dashboards"]["dashboards"] == []
    assert summaries == [
        "Removed processor 'engagement' and its dependent metrics and tiles",
    ]


@pytest.mark.unit
def test_apply_operations_remove_metric_cascades_tiles() -> None:
    updated, _ = apply_draft_operations(
        _base_draft(),
        [{"op": "remove_metric", "name": "CTR"}],
    )

    assert updated["metrics"]["metrics"] == {}
    assert updated["dashboards"]["dashboards"] == []


@pytest.mark.unit
def test_apply_operations_remove_tile() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [{"op": "remove_tile", "dashboard": "overview", "page": "engagement", "id": "ctr"}],
    )

    assert updated["dashboards"]["dashboards"][0]["pages"][0]["tiles"] == []
    assert summaries == ["Removed tile 'overview/engagement/ctr'"]


@pytest.mark.unit
def test_apply_operations_sets_and_removes_source_preprocessing() -> None:
    draft = _base_draft()
    updated, summaries = apply_draft_operations(
        draft,
        [
            {
                "op": "set_source_default",
                "source": "ih",
                "field": "ModelControlGroup",
                "value": "Test",
            },
            {
                "op": "set_calculated_field",
                "source": "ih",
                "name": "ChannelCopy",
                "expression": {"col": "Channel"},
            },
        ],
    )

    source = updated["pipelines"]["sources"][0]
    assert source["defaults"] == {"ModelControlGroup": "Test"}
    assert source["transforms"] == [
        {
            "kind": "derive_column",
            "output": "ChannelCopy",
            "expression": {"col": "Channel"},
        }
    ]
    assert summaries == [
        "Added default 'ih/ModelControlGroup'",
        "Added calculated field 'ih/ChannelCopy'",
    ]
    assert ai_config_studio_page.validate_draft_catalog(updated) == (True, [])
    assert "defaults" not in draft["pipelines"]["sources"][0]

    removed, remove_summaries = apply_draft_operations(
        updated,
        [
            {"op": "remove_source_default", "source": "ih", "field": "ModelControlGroup"},
            {"op": "remove_calculated_field", "source": "ih", "name": "ChannelCopy"},
        ],
    )

    assert removed["pipelines"]["sources"][0]["defaults"] == {}
    assert removed["pipelines"]["sources"][0]["transforms"] == []
    assert remove_summaries == [
        "Removed default 'ih/ModelControlGroup'",
        "Removed calculated field 'ih/ChannelCopy'",
    ]


@pytest.mark.unit
def test_apply_operations_sets_concatenated_calculated_field() -> None:
    expression = {
        "op": "concat",
        "args": [{"col": "Channel"}, {"lit": "offer"}],
        "sep": "-",
    }

    updated, summaries = apply_draft_operations(
        _base_draft(),
        [
            {
                "op": "set_calculated_field",
                "source": "ih",
                "name": "ChannelOffer",
                "expression": expression,
            }
        ],
    )

    assert updated["pipelines"]["sources"][0]["transforms"] == [
        {
            "kind": "derive_column",
            "output": "ChannelOffer",
            "expression": expression,
        }
    ]
    assert summaries == ["Added calculated field 'ih/ChannelOffer'"]
    assert ai_config_studio_page.validate_draft_catalog(updated) == (True, [])


@pytest.mark.unit
def test_source_default_operation_uses_transform_after_rename_capitalize() -> None:
    draft = _base_draft()
    draft["pipelines"]["sources"][0]["transforms"] = [{"kind": "rename_capitalize"}]

    updated, _ = apply_draft_operations(
        draft,
        [
            {
                "op": "set_source_default",
                "source": "ih",
                "field": "ModelControlGroup",
                "value": "Test",
            }
        ],
    )

    source = updated["pipelines"]["sources"][0]
    assert source.get("defaults", {}) == {}
    assert source["transforms"] == [
        {"kind": "rename_capitalize"},
        {"kind": "defaults", "values": {"ModelControlGroup": "Test"}},
    ]


@pytest.mark.unit
def test_source_filter_operation_updates_pipeline_without_touching_processor() -> None:
    draft = _base_draft()
    draft["pipelines"]["sources"][0]["transforms"] = [
        {"kind": "rename_capitalize"},
        {
            "kind": "derive_column",
            "output": "SubjectID",
            "expression": {"col": "CustomerID"},
        },
        {
            "kind": "derive_column",
            "output": "Margin",
            "expression": {"col": "Revenue"},
        },
    ]
    processor_before = draft["processors"]["processors"][0].copy()
    expression = {"op": "not_null", "column": "SubjectID"}

    updated, summaries = apply_draft_operations(
        draft,
        [{"op": "set_source_filter", "source": "ih", "expression": expression}],
    )

    transforms = updated["pipelines"]["sources"][0]["transforms"]
    assert [transform["kind"] for transform in transforms] == [
        "rename_capitalize",
        "derive_column",
        "filter",
        "derive_column",
    ]
    assert transforms[2]["expression"] == expression
    assert updated["processors"]["processors"][0] == processor_before
    assert summaries == ["Added source filter 'ih'"]

    removed, remove_summaries = apply_draft_operations(
        updated,
        [{"op": "remove_source_filter", "source": "ih"}],
    )

    assert all(
        transform.get("kind") != "filter"
        for transform in removed["pipelines"]["sources"][0]["transforms"]
    )
    assert remove_summaries == ["Removed source filter 'ih'"]


@pytest.mark.unit
def test_source_preprocessing_operations_reject_invalid_targets_and_values() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        apply_draft_operations(
            _base_draft(),
            [{"op": "set_source_default", "source": "missing", "field": "X", "value": 1}],
        )
    with pytest.raises(ValueError, match="string, number, boolean, or null"):
        apply_draft_operations(
            _base_draft(),
            [{"op": "set_source_default", "source": "ih", "field": "X", "value": []}],
        )
    with pytest.raises(ValueError, match="expression"):
        apply_draft_operations(
            _base_draft(),
            [
                {
                    "op": "set_calculated_field",
                    "source": "ih",
                    "name": "Broken",
                    "expression": {"unknown": "expression"},
                }
            ],
        )
    with pytest.raises(ValueError, match="closed expression AST"):
        apply_draft_operations(
            _base_draft(),
            [
                {
                    "op": "set_calculated_field",
                    "source": "ih",
                    "name": "Unsafe",
                    "expression": {"polars": 'pl.col("Channel")'},
                }
            ],
        )
    with pytest.raises(ValueError, match="closed expression AST"):
        apply_draft_operations(
            _base_draft(),
            [
                {
                    "op": "set_source_filter",
                    "source": "ih",
                    "expression": {"polars": 'pl.col("Outcome") == "Clicked"'},
                }
            ],
        )


@pytest.mark.unit
def test_apply_operations_installs_ready_builtin_recipe() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [
            {
                "op": "install_recipe",
                "recipe_id": "engagement.engagement_rate",
                "processor": "engagement",
                "metric_id": "Recipe_Engagement",
                "dashboard": "overview",
                "page": "engagement",
                "tile_id": "recipe_engagement",
            }
        ],
    )

    metric = updated["metrics"]["metrics"]["Recipe_Engagement"]
    assert metric["recipe"] == {"id": "engagement.engagement_rate", "version": 1}
    assert "overview/engagement/recipe_engagement" in {
        patch.object_id for patch in draft_patches(_base_draft(), updated)
    }
    assert summaries == [
        "Installed recipe 'engagement.engagement_rate' as metric 'Recipe_Engagement'"
    ]


@pytest.mark.unit
def test_apply_operations_rejects_unknown_and_invalid_operations() -> None:
    with pytest.raises(ValueError, match="Unknown operation"):
        apply_draft_operations(_base_draft(), [{"op": "drop_everything"}])
    with pytest.raises(ValueError, match="does not exist"):
        apply_draft_operations(_base_draft(), [{"op": "remove_metric", "name": "Missing"}])
    with pytest.raises(ValueError, match="requires a processor mapping"):
        apply_draft_operations(_base_draft(), [{"op": "set_processor", "processor": {}}])


@pytest.mark.unit
def test_patch_rejection_restores_changed_object_instead_of_deleting_it() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_metric",
                "name": "CTR",
                "metric": {
                    **base["metrics"]["metrics"]["CTR"],
                    "direction": "higher_is_better",
                },
            },
            {
                "op": "set_metric",
                "name": "Total",
                "metric": {
                    "source": "engagement",
                    "kind": "formula",
                    "expression": {"col": "Count"},
                },
            },
        ],
    )

    patches = draft_patches(base, proposed)
    assert {patch.key for patch in patches} == {"metrics:CTR", "metrics:Total"}

    reviewed = merge_selected_draft_patches(base, proposed, {"metrics:Total"})

    assert reviewed["metrics"]["metrics"]["CTR"] == base["metrics"]["metrics"]["CTR"]
    assert "Total" in reviewed["metrics"]["metrics"]


@pytest.mark.unit
def test_source_preprocessing_patches_are_independent_and_selective() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_source_default",
                "source": "ih",
                "field": "ModelControlGroup",
                "value": "Test",
            },
            {
                "op": "set_calculated_field",
                "source": "ih",
                "name": "ChannelCopy",
                "expression": {"col": "Channel"},
            },
        ],
    )

    patches = draft_patches(base, proposed)
    assert {patch.key for patch in patches} == {
        "source_defaults:ih/ModelControlGroup",
        "calculated_fields:ih/ChannelCopy",
    }

    reviewed = merge_selected_draft_patches(
        base,
        proposed,
        {"source_defaults:ih/ModelControlGroup"},
    )
    source = reviewed["pipelines"]["sources"][0]
    assert source["defaults"] == {"ModelControlGroup": "Test"}
    assert source.get("transforms", []) == []


@pytest.mark.unit
def test_source_filter_patch_is_independent_from_processor_definitions() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_source_filter",
                "source": "ih",
                "expression": {"op": "eq", "column": "Outcome", "value": "Clicked"},
            }
        ],
    )

    patches = draft_patches(base, proposed)
    assert [patch.key for patch in patches] == ["source_filters:ih"]

    accepted = merge_selected_draft_patches(base, proposed, {"source_filters:ih"})
    rejected = merge_selected_draft_patches(base, proposed, set())

    assert accepted["pipelines"]["sources"][0]["transforms"][0]["kind"] == "filter"
    assert accepted["processors"] == base["processors"]
    assert rejected == base


@pytest.mark.unit
def test_patch_bundles_close_processor_metric_tile_and_report_dependencies() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_processor",
                "processor": {
                    "id": "satisfaction",
                    "source": "ih",
                    "kind": "binary_outcome",
                    "dimensions": ["Channel"],
                    "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked"],
                        "negative_values": ["Impression"],
                    },
                },
            },
            {
                "op": "set_metric",
                "name": "Satisfaction",
                "metric": {
                    "source": "satisfaction",
                    "kind": "formula",
                    "expression": {
                        "op": "safe_div",
                        "num": {"col": "Positives"},
                        "den": {"col": "Count"},
                    },
                },
            },
            {
                "op": "set_tile",
                "dashboard": "satisfaction",
                "page": "overview",
                "tile": {
                    "id": "satisfaction",
                    "title": "Satisfaction",
                    "metric": "Satisfaction",
                    "chart": "kpi_card",
                    "value": "Satisfaction",
                },
            },
        ],
    )

    bundles = draft_patch_bundles(
        base,
        proposed,
        ai_config_studio_page.validate_draft_catalog,
    )
    repeated = draft_patch_bundles(
        base,
        proposed,
        ai_config_studio_page.validate_draft_catalog,
    )

    assert len(bundles) == 1
    bundle = bundles[0]
    assert set(bundle.patch_keys) == {
        "processors:satisfaction",
        "metrics:Satisfaction",
        "dashboards:structure",
        "tiles:satisfaction/overview/satisfaction",
    }
    assert bundle.key == repeated[0].key
    assert bundle.title == "Add Satisfaction processing flow"
    assert "1 processing flow" in bundle.summary
    assert "1 metric" in bundle.summary
    assert "1 report tile" in bundle.summary
    assert "accepted together" in bundle.consequence
    assert not bundle.is_removal
    assert bundle.is_valid


@pytest.mark.unit
def test_patch_bundle_removals_require_explicit_review() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [{"op": "remove_processor", "id": "engagement"}],
    )

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        return True, []

    bundles = draft_patch_bundles(base, proposed, validate)

    assert len(bundles) == 1
    assert bundles[0].is_removal
    assert "never selected automatically" in bundles[0].consequence

    rejected, rejection_issues = merge_selected_draft_patch_bundles(
        base,
        proposed,
        bundles,
        {bundles[0].key},
        validate,
    )
    accepted, acceptance_issues = merge_selected_draft_patch_bundles(
        base,
        proposed,
        bundles,
        {bundles[0].key},
        validate,
        allow_removals=True,
    )

    assert rejected is None
    assert "requires explicit review" in rejection_issues[0]
    assert accepted is not None
    assert accepted["processors"]["processors"] == []
    assert acceptance_issues == ()


@pytest.mark.unit
def test_patch_bundle_selection_revalidates_the_combination() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [
            {"op": "set_source_default", "source": "ih", "field": "Region", "value": "EU"},
            {"op": "set_source_default", "source": "ih", "field": "Segment", "value": "A"},
        ],
    )

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        defaults = candidate["pipelines"]["sources"][0].get("defaults", {})
        if {"Region", "Segment"}.issubset(defaults):
            return False, ["Region and Segment defaults cannot be accepted together."]
        return True, []

    bundles = draft_patch_bundles(base, proposed, validate)

    assert len(bundles) == 2
    assert all(bundle.is_valid for bundle in bundles)

    candidate, issues = merge_selected_draft_patch_bundles(
        base,
        proposed,
        bundles,
        {bundle.key for bundle in bundles},
        validate,
    )

    assert candidate is None
    assert issues == ("Region and Segment defaults cannot be accepted together.",)


@pytest.mark.unit
def test_invalid_patch_bundle_cannot_be_selected() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [{"op": "set_source_default", "source": "ih", "field": "Blocked", "value": True}],
    )

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        defaults = candidate["pipelines"]["sources"][0].get("defaults", {})
        return (
            (False, ["Blocked defaults are not allowed."]) if "Blocked" in defaults else (True, [])
        )

    bundles = draft_patch_bundles(base, proposed, validate)
    candidate, issues = merge_selected_draft_patch_bundles(
        base,
        proposed,
        bundles,
        {bundles[0].key},
        validate,
    )

    assert not bundles[0].is_valid
    assert bundles[0].validation_issues == ("Blocked defaults are not allowed.",)
    assert candidate is None
    assert issues == (f"'{bundles[0].title}' does not pass draft validation.",)


@pytest.mark.unit
def test_copilot_tool_loop_repairs_invalid_operations_before_pending_review() -> None:
    responses = iter(
        [
            '{"reply":"Adding it.","operations":[{"op":"set_metric","name":"Total",'
            '"metric":{"source":"missing","kind":"formula","expression":{"col":"Count"}}}],'
            '"questions":[]}',
            '{"reply":"Corrected the source.","operations":[{"op":"set_metric",'
            '"name":"Total","metric":{"source":"engagement","kind":"formula",'
            '"expression":{"col":"Count"}}}],"questions":[]}',
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        ok, issues = ai_config_studio_page.validate_draft_catalog(candidate)
        if not ok:
            issues.append("CustomerID remains hidden")
        return ok, issues

    result = run_copilot_tool_loop(
        prompt="Add Total",
        draft=_base_draft(),
        call_model=call_model,
        validate=validate,
        max_iterations=3,
        hidden_fields=["CustomerID"],
    )

    assert result.iterations == 2
    assert result.pending_draft is not None
    assert result.pending_draft["metrics"]["metrics"]["Total"]["source"] == "engagement"
    assert "Validation or operation errors" in prompts[1]
    assert "CustomerID" not in prompts[1]


@pytest.mark.unit
def test_copilot_tool_loop_rejects_processor_edit_on_filters_step() -> None:
    responses = iter(
        [
            '{"reply":"Filtering the processor.","operations":[{"op":"set_processor",'
            '"processor":{"id":"engagement","source":"ih","kind":"binary_outcome",'
            '"dimensions":["Channel"],"time":{"column":"OutcomeTime",'
            '"grains":["Day","Summary"]},"outcome":{"column":"Outcome",'
            '"positive_values":["Clicked"],"negative_values":["Impression"]},'
            '"filter":{"op":"eq","column":"Outcome","value":"Clicked"}}}],'
            '"questions":[]}',
            '{"reply":"Filtering the source pipeline.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"Outcome","value":"Clicked"}}],"questions":[]}',
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    policy = ai_config_studio_page._copilot_operation_policy("4. Filters")
    result = run_copilot_tool_loop(
        prompt="Filter Outcome to Clicked",
        draft=_base_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        operation_policy=policy,
    )

    assert result.iterations == 2
    assert result.pending_draft is not None
    source = result.pending_draft["pipelines"]["sources"][0]
    assert source["transforms"][0]["kind"] == "filter"
    assert "filter" not in result.pending_draft["processors"]["processors"][0]
    assert "before processor fan-out" in prompts[1]
    assert "set_source_filter" in prompts[1]


@pytest.mark.unit
def test_copilot_tool_loop_never_applies_operations_in_read_only_mode() -> None:
    prompts: list[str] = []
    validation_calls = 0

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return (
            '{"reply":"I will add Total.","operations":[{"op":"set_metric",'
            '"name":"Total","metric":{"source":"engagement","kind":"formula",'
            '"expression":{"col":"Count"}}}],"questions":[]}'
        )

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        nonlocal validation_calls
        validation_calls += 1
        return True, []

    result = run_copilot_tool_loop(
        prompt="Add Total",
        draft=_base_draft(),
        call_model=call_model,
        validate=validate,
        read_only=True,
        pending_summary="One new conversion metric is waiting for review.",
    )

    assert result.iterations == 1
    assert result.pending_draft is None
    assert result.turn.operations == []
    assert validation_calls == 0
    assert "PENDING REVIEW MODE" in prompts[0]
    assert "One new conversion metric is waiting for review." in prompts[0]
    assert "No draft change was created" in result.turn.reply
    assert result.validation_issues == (
        "Mutating operations are blocked while a proposal is pending review.",
    )


@pytest.mark.unit
def test_prompt_for_copilot_includes_step_goals_history_and_contract() -> None:
    prompt = prompt_for_copilot(
        step="9. Metrics",
        user_message="Add average revenue per customer.",
        history=[
            {"role": "user", "content": "What does this step do?"},
            {"role": "assistant", "content": "It reviews metric definitions."},
            {"role": "user", "content": "Keep CustomerID hidden."},
        ],
        user_goals="Weekly conversion by channel.",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=["CustomerID"],
        current_draft=_base_draft(),
    )

    assert "'Metrics' studio step" in prompt
    assert "reviewing metric definitions" in prompt
    assert "Business requirements from the user:" in prompt
    assert "Weekly conversion by channel." in prompt
    assert "user: What does this step do?" in prompt
    assert "assistant: It reviews metric definitions." in prompt
    assert "User message:\nAdd average revenue per customer." in prompt
    assert '"reply": str' in prompt
    assert "set_metric" in prompt
    assert "set_source_default" in prompt
    assert "set_calculated_field" in prompt
    assert "Never invent an input field" in prompt
    assert "Return valid JSON only." in prompt
    assert "Hidden field count: 1" in prompt
    assert "CustomerID" not in prompt
    assert "Keep <hidden-field> hidden." in prompt


@pytest.mark.unit
def test_pending_review_prompt_is_read_only_and_includes_redacted_summary() -> None:
    prompt = prompt_for_copilot(
        step="9. Metrics",
        user_message="What will this proposal change?",
        history=[],
        user_goals="",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=["CustomerID"],
        current_draft=_base_draft(),
        read_only=True,
        pending_summary="Adds Channel coverage without exposing CustomerID.",
    )

    assert "PENDING REVIEW MODE" in prompt
    assert "operations: MUST be an empty list" in prompt
    assert "accept or reject the pending proposal first" in prompt
    assert "Pending proposal summary:" in prompt
    assert "Adds Channel coverage" in prompt
    assert "CustomerID" not in prompt
    assert "<hidden-field>" in prompt


@pytest.mark.unit
def test_filters_step_prompt_requires_source_filter_operation() -> None:
    prompt = prompt_for_copilot(
        step="4. Filters",
        user_message="Only keep Clicked outcomes.",
        history=[],
        user_goals="",
        approved_schema=[],
        approved_fields=["Outcome"],
        hidden_fields=[],
        current_draft=_base_draft(),
    )

    assert "set_source_filter" in prompt
    assert "remove_source_filter" in prompt
    assert "before processor fan-out" in prompt
    assert "do not use set_processor for a dataset filter" in prompt


@pytest.mark.unit
def test_calculations_step_prompt_exposes_concat_and_complete_expression_dsl() -> None:
    prompt = prompt_for_copilot(
        step="5. Calculations",
        user_message="Concatenate Issue and Group with a slash.",
        history=[],
        user_goals="",
        approved_schema=[
            {"column": "Issue", "dtype": "String", "unique": 3},
            {"column": "Group", "dtype": "String", "unique": 8},
        ],
        approved_fields=["Issue", "Group"],
        hidden_fields=[],
        current_draft=_base_draft(),
    )

    assert "set_calculated_field" in prompt
    assert "expression_ast catalog dictionary is the complete supported DSL" in prompt
    assert "Do not claim an operation is unavailable" in prompt
    assert '"op":"concat"' in prompt
    assert "concat(...) function-call string" in prompt
    assert "concatenation:" in prompt
    assert "multi_branch_conditional:" in prompt
    assert "datetime_parse:" in prompt


@pytest.mark.unit
def test_parse_copilot_response_reads_operations_and_questions() -> None:
    turn = parse_copilot_response(
        """
```json
{
  "reply": "I can add that metric.",
  "operations": [{"op": "set_metric", "name": "Total", "metric": {"kind": "formula"}}],
  "questions": [{"question": "Which time grain?", "options": ["Day", "Month"]}]
}
```
"""
    )

    assert turn.reply == "I can add that metric."
    assert turn.operations[0]["op"] == "set_metric"
    assert turn.questions[0].question == "Which time grain?"
    assert turn.questions[0].options == ("Day", "Month")


@pytest.mark.unit
def test_parse_copilot_response_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_copilot_response("no json here")
    with pytest.raises(ValueError, match="JSON object"):
        parse_copilot_response("[1, 2, 3]")
    with pytest.raises(ValueError, match="no reply"):
        parse_copilot_response('{"reply": "", "operations": [], "questions": []}')


@pytest.mark.unit
def test_coverage_prompt_and_response_round_trip() -> None:
    prompt = prompt_for_coverage(
        user_goals="Weekly conversion by channel.\nAverage revenue per customer.",
        draft=_base_draft(),
        hidden_fields=["CustomerID"],
    )

    assert "Business requirements:" in prompt
    assert "Weekly conversion by channel." in prompt
    assert "- CTR" in prompt
    assert "overview/engagement/ctr" in prompt
    assert "CustomerID" not in prompt

    rows = parse_coverage_response(
        """
[
  {"requirement": "Weekly conversion by channel", "status": "covered",
   "metrics": ["CTR"], "tiles": ["overview/engagement/ctr"], "note": "CTR trend by day."},
  {"requirement": "Average revenue per customer", "status": "missing",
   "metrics": [], "tiles": [], "note": "No revenue metric exists."}
]
"""
    )

    assert [row.status for row in rows] == ["covered", "missing"]
    assert rows[0].metrics == ("CTR",)
    assert rows[1].note == "No revenue metric exists."


@pytest.mark.unit
def test_coverage_prompt_redacts_hidden_names_from_metric_and_tile_ids() -> None:
    draft = _base_draft()
    metric = draft["metrics"]["metrics"].pop("CTR")
    draft["metrics"]["metrics"]["CustomerID_metric"] = metric
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile["id"] = "CustomerID_tile"
    tile["metric"] = "CustomerID_metric"

    prompt = prompt_for_coverage(
        user_goals="Explain CustomerID coverage.",
        draft=draft,
        hidden_fields=["CustomerID"],
    )

    assert "CustomerID" not in prompt
    assert "<hidden-field>_metric" in prompt
    assert "<hidden-field>_tile" in prompt


@pytest.mark.unit
def test_parse_coverage_response_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        parse_coverage_response('[{"requirement": "X", "status": "unknown"}]')


@pytest.mark.unit
def test_coverage_response_downgrades_hallucinated_references() -> None:
    rows = parse_coverage_response(
        '[{"requirement":"Revenue","status":"covered","metrics":["Missing"],'
        '"tiles":["overview/missing/tile"],"note":"Available."}]',
        draft=_base_draft(),
    )

    assert rows[0].status == "missing"
    assert rows[0].metrics == ()
    assert rows[0].tiles == ()
    assert "Ignored unknown draft references" in rows[0].note


@pytest.mark.unit
def test_phase_helpers_cover_every_step_in_both_step_lists() -> None:
    for steps in (STEPS, DETERMINISTIC_STEPS):
        covered: list[str] = []
        for phase, _ in STUDIO_PHASES:
            options = _phase_step_options(phase, steps)
            covered.extend(options)
            for step in options:
                assert _phase_for_step(step, steps) == phase
        assert covered == steps


@pytest.mark.unit
def test_copilot_panel_holds_operations_in_pending_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def fake_call_litellm(settings: object, prompt: str, **kwargs: object) -> str:
        if prompt == "Reply with READY.":
            return "READY"
        return (
            '{"reply": "Added a total metric.", '
            '"operations": [{"op": "set_metric", "name": "Total", '
            '"metric": {"source": "engagement", "kind": "formula", '
            '"expression": {"col": "Count"}}}], '
            '"questions": []}'
        )

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        working = pl.DataFrame({"Channel": ["Web", "Mobile"]})
        page._render_copilot_panel("9. Metrics", working, ["Channel"])

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()

    assert not at.exception
    at.chat_input[0].set_value("Add a Total metric.").run()

    assert not at.exception
    pending = at.session_state["ai_studio_pending_draft"]
    assert "Total" in pending["metrics"]["metrics"]
    assert at.session_state["ai_studio_pending_kind"] == "copilot"
    history = at.session_state["ai_studio_copilot_history"]
    assert history[-1]["role"] == "assistant"
    assert "Added metric 'Total'" in history[-1]["content"]
    assert at.session_state["ai_studio_copilot_input"] is None


@pytest.mark.unit
def test_copilot_permission_error_names_model_and_preserves_last_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def denied_call(*args: object, **kwargs: object) -> str:
        raise RuntimeError("OpenAIException - You have insufficient permissions for this operation")

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", denied_call)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "gpt-unavailable"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        page._render_copilot_panel(
            "3. Defaults",
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
        )

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    at.chat_input[0].set_value("Set ModelControlGroup to Test.").run()

    assert not at.exception
    error = at.error[0].value
    assert "Couldn't complete Copilot request" in error
    assert "accepted revision was not changed" in error
    assert "insufficient permissions" not in error
    assert "Set ModelControlGroup to Test." in at.session_state["ai_studio_copilot_last_prompt"]


@pytest.mark.unit
def test_accepting_preprocessing_patches_syncs_all_source_editors() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    base = _base_draft()
    pending, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_source_default",
                "source": "ih",
                "field": "ModelControlGroup",
                "value": "Test",
            },
            {
                "op": "set_source_filter",
                "source": "ih",
                "expression": {"op": "eq", "column": "Channel", "value": "Web"},
            },
            {
                "op": "set_calculated_field",
                "source": "ih",
                "name": "ChannelCopy",
                "expression": {"col": "Channel"},
            },
        ],
    )

    def app(base_draft: dict, pending_draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state.setdefault("ai_studio_source_id", "ih")
        st.session_state.setdefault("ai_studio_draft", base_draft)
        st.session_state.setdefault("ai_studio_pending_draft", pending_draft)
        st.session_state.setdefault("ai_studio_pending_base_draft", base_draft)
        st.session_state.setdefault("ai_studio_pending_kind", "copilot")
        page._consume_preprocessing_editor_sync()
        page._render_pending_draft_review()

    at = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()
    at = next(button for button in at.button if button.label == "Review individually").click().run()
    for index, checkbox in enumerate(at.checkbox):
        if checkbox.label == "Accept this complete bundle":
            at = at.checkbox[index].check().run()
    next(button for button in at.button if button.label == "Accept selected bundles").click().run()

    assert not at.exception
    assert at.session_state["ai_studio_pending_draft"] is None
    assert at.session_state["ai_studio_defaults"] == [
        {"Field": "ModelControlGroup", "Default Value": "Test", "Enabled": True}
    ]
    assert at.session_state["ai_studio_filter_mode"] == "Rules"
    assert at.session_state["ai_studio_filter_rows"] == [
        {
            "Field": "Channel",
            "Operator": "==",
            "Value": "Web",
            "Enabled": True,
        }
    ]
    calculation = at.session_state["ai_studio_calculations"][0]
    assert calculation["Name"] == "ChannelCopy"
    assert calculation["Mode"] == "AST YAML"
    assert "col: Channel" in calculation["Expression"]


@pytest.mark.unit
def test_copilot_panel_renders_clarifying_question_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def fake_call_litellm(settings: object, prompt: str, **kwargs: object) -> str:
        if prompt == "Reply with READY.":
            return "READY"
        return (
            '{"reply": "Which grain should the metric use?", "operations": [], '
            '"questions": [{"question": "Which time grain?", "options": ["Day", "Month"]}]}'
        )

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        working = pl.DataFrame({"Channel": ["Web", "Mobile"]})
        page._render_copilot_panel("9. Metrics", working, ["Channel"])

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    at.chat_input[0].set_value("Add a trend metric.").run()

    assert not at.exception
    assert at.session_state["ai_studio_pending_draft"] is None
    labels = [widget.label for widget in at.button]
    assert "Day" in labels
    assert "Month" in labels

    next(widget for widget in at.button if widget.label == "Day").click().run()

    assert not at.exception
    history = at.session_state["ai_studio_copilot_history"]
    assert {"role": "user", "content": "Day"} in history


@pytest.mark.unit
def test_copilot_question_before_first_draft_does_not_accept_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    monkeypatch.setattr(
        ai_config_studio_page,
        "call_litellm",
        lambda *args, **kwargs: (
            '{"reply":"This step reviews metrics.","operations":[],"questions":[]}'
        ),
    )
    monkeypatch.setattr(
        ai_config_studio_page,
        "_build_draft_catalog",
        lambda working, approved_fields: _base_draft(),
    )

    def app() -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state.setdefault("ai_studio_draft", None)
        st.session_state.setdefault("ai_studio_pending_draft", None)
        page._render_copilot_panel(
            "9. Metrics",
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
        )

    at = AppTest.from_function(app).run()
    at.chat_input[0].set_value("What happens on this step?").run()

    assert not at.exception
    assert at.session_state["ai_studio_draft"] is None
    assert at.session_state["ai_studio_pending_draft"] is None


@pytest.mark.unit
def test_pending_review_allows_read_only_copilot_and_preserves_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    prompts: list[str] = []

    def fake_call(settings: object, prompt: str, **kwargs: object) -> str:
        prompts.append(prompt)
        if prompt == "Reply with READY.":
            return "READY"
        return '{"reply":"This proposal adds one metric.","operations":[],"questions":[]}'

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call)
    base = _base_draft()
    pending, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_metric",
                "name": "Total",
                "metric": {
                    "source": "engagement",
                    "kind": "formula",
                    "expression": {"col": "Count"},
                },
            }
        ],
    )

    def app(base_draft: dict, pending_draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_draft"] = base_draft
        st.session_state["ai_studio_pending_draft"] = pending_draft
        st.session_state["ai_studio_pending_base_draft"] = base_draft
        st.session_state["ai_studio_pending_kind"] = "revision"
        page._render_copilot_panel(
            "9. Metrics",
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
        )

    at = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()

    assert not at.exception
    assert not at.chat_input[0].disabled
    at.chat_input[0].set_value("What does this proposal change?").run()

    assert not at.exception
    assert at.session_state["ai_studio_pending_draft"] == pending
    assert len(prompts) == 2
    assert "PENDING REVIEW MODE" in prompts[-1]
    assert (
        "This proposal adds one metric."
        in at.session_state["ai_studio_copilot_history"][-1]["content"]
    )


@pytest.mark.unit
def test_new_sample_identity_resets_draft_and_copilot_with_same_columns() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        frame = pl.DataFrame({"Channel": ["Web"]})
        st.session_state.setdefault("ai_studio_sample_identity", "first-file")
        st.session_state.setdefault("ai_studio_user_goals", "Keep this requirement")
        page._initialize_state(frame)
        if st.button("Switch sample"):
            st.session_state["ai_studio_draft"] = draft
            st.session_state["ai_studio_copilot_history"] = [
                {"role": "user", "content": "Old sample question"}
            ]
            st.session_state["ai_studio_sample_identity"] = "second-file"
            page._initialize_state(frame)

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    next(button for button in at.button if button.label == "Switch sample").click().run()

    assert not at.exception
    assert at.session_state["ai_studio_draft"] is None
    assert at.session_state["ai_studio_copilot_history"] == []
    assert at.session_state["ai_studio_user_goals"] == "Keep this requirement"


@pytest.mark.unit
def test_phase_statuses_require_explicit_review_before_publish() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict, reviewed: bool, published: bool) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_reviewed_signature"] = (
            page._draft_signature(draft) if reviewed else ""
        )
        st.session_state["ai_studio_published_signature"] = (
            page._draft_signature(draft) if published else ""
        )
        st.session_state["phase_statuses"] = page._phase_statuses(["Channel"])

    unreviewed = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "reviewed": False, "published": False},
    ).run()
    reviewed = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "reviewed": True, "published": False},
    ).run()
    published = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "reviewed": True, "published": True},
    ).run()

    assert unreviewed.session_state["phase_statuses"]["Review"] == "attention"
    assert unreviewed.session_state["phase_statuses"]["Apply"] == "empty"
    assert reviewed.session_state["phase_statuses"]["Review"] == "complete"
    assert reviewed.session_state["phase_statuses"]["Apply"] == "attention"
    assert published.session_state["phase_statuses"]["Apply"] == "complete"
