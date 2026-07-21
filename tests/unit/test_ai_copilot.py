"""Copilot draft-operation, prompt, and coverage tests."""

from __future__ import annotations

import copy
import json

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
    validate_draft_field_contract,
)
from valuestream.ai.copilot import (
    draft_patch_bundles,
    merge_selected_draft_patch_bundles,
    remap_operation_field_names,
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


def _rename_capitalize_draft() -> dict:
    draft = _base_draft()
    source = draft["pipelines"]["sources"][0]
    source["schema"] = {
        "timestamp_column": "pxOutcomeTime",
        "natural_key": ["pyCustomerID", "pyChannel"],
    }
    source["transforms"] = [{"kind": "rename_capitalize"}]
    return draft


def _binary_processor(channel_field: str) -> dict:
    return {
        "id": "engagement",
        "source": "ih",
        "kind": "binary_outcome",
        "dimensions": [channel_field],
        "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
        "outcome": {
            "column": "Outcome",
            "positive_values": ["Clicked"],
            "negative_values": ["Impression"],
        },
    }


def _operation_response(reply: str, operation: dict) -> str:
    return json.dumps({"reply": reply, "operations": [operation], "questions": []})


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
def test_source_naming_patch_preserves_complete_proposed_source() -> None:
    base = _base_draft()
    base_source = base["pipelines"]["sources"][0]
    base_source["defaults"] = {"Channel": "Unknown"}
    proposed = copy.deepcopy(base)
    proposed_source = proposed["pipelines"]["sources"][0]
    proposed_source.pop("defaults")
    proposed_source["transforms"] = [
        {"kind": "rename_capitalize"},
        {"kind": "defaults", "values": {"Channel": "Unknown"}},
    ]

    patches = draft_patches(base, proposed)

    assert [patch.key for patch in patches] == ["sources:ih"]
    assert patches[0].before == base_source
    assert patches[0].after == proposed_source
    merged = merge_selected_draft_patches(base, proposed, {"sources:ih"})
    assert merged == proposed


@pytest.mark.unit
def test_source_naming_bundle_accepts_effective_downstream_field_graph() -> None:
    base = _rename_capitalize_draft()
    base_source = base["pipelines"]["sources"][0]
    base_source["transforms"] = [
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "pyChannel", "value": "Web"},
        }
    ]
    base_processor = _binary_processor("pyChannel")
    base_processor["time"]["column"] = "pxOutcomeTime"
    base_processor["outcome"]["column"] = "pyOutcome"
    base["processors"]["processors"] = [base_processor]
    base_tile = base["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    base_tile.update({"chart": "bar", "x": "pyChannel"})

    proposed = copy.deepcopy(base)
    proposed_source = proposed["pipelines"]["sources"][0]
    proposed_source["transforms"] = [
        {"kind": "rename_capitalize"},
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "Channel", "value": "Web"},
        },
        {
            "kind": "derive_column",
            "output": "ChannelCopy",
            "expression": {"col": "Channel"},
        },
    ]
    proposed_processor = _binary_processor("ChannelCopy")
    proposed["processors"]["processors"] = [proposed_processor]
    proposed_tile = proposed["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    proposed_tile["x"] = "ChannelCopy"

    def validate(candidate: dict) -> tuple[bool, list[str]]:
        catalog_ok, catalog_issues = ai_config_studio_page.validate_draft_catalog(candidate)
        fields_ok, field_issues = validate_draft_field_contract(
            candidate,
            ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
            source_id="ih",
            baseline_draft=base,
            expected_rename_capitalize=True,
        )
        return catalog_ok and fields_ok, [*catalog_issues, *field_issues]

    patches = draft_patches(base, proposed)
    bundles = draft_patch_bundles(base, proposed, validate)

    assert {patch.key for patch in patches} == {
        "sources:ih",
        "processors:engagement",
        "tiles:overview/engagement/ctr",
    }
    assert len(bundles) == 1
    assert set(bundles[0].patch_keys) == {patch.key for patch in patches}
    assert bundles[0].is_valid

    accepted, issues = merge_selected_draft_patch_bundles(
        base,
        proposed,
        bundles,
        {bundles[0].key},
        validate,
    )

    assert accepted == proposed
    assert issues == ()


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
def test_copilot_filters_step_rejects_default_workaround() -> None:
    responses = iter(
        [
            '{"reply":"Adding a synthetic field and filtering it.","operations":['
            '{"op":"set_source_default","source":"ih","field":"Channel","value":"Web"},'
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"Channel","value":"Web"}}],"questions":[]}',
            '{"reply":"Filtering the effective field only.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"Channel","value":"Web"}}],"questions":[]}',
        ]
    )

    result = run_copilot_tool_loop(
        prompt="Only keep Web interactions on Channel.",
        draft=_rename_capitalize_draft(),
        call_model=lambda _prompt: next(responses),
        validate=ai_config_studio_page.validate_draft_catalog,
        operation_policy=ai_config_studio_page._copilot_operation_policy("4. Filters"),
        approved_fields=["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        field_contract_source_id="ih",
        expected_rename_capitalize=True,
    )

    assert result.iterations == 2
    assert result.pending_draft is not None
    source = result.pending_draft["pipelines"]["sources"][0]
    assert source.get("defaults", {}) == {}
    assert source["transforms"][-1]["expression"]["column"] == "Channel"


@pytest.mark.unit
def test_copilot_prompt_marks_effective_names_as_authoritative() -> None:
    prompt = prompt_for_copilot(
        step="4. Filters",
        user_message="Filter Channel to Web.",
        history=[],
        user_goals="",
        approved_schema=[{"column": "Channel", "dtype": "String"}],
        approved_fields=["Channel"],
        hidden_fields=[],
        current_draft=_rename_capitalize_draft(),
        rename_capitalize_enabled=True,
        approved_field_name_mapping={"pyChannel": "Channel"},
    )

    assert "Approved fields below are the authoritative post-transform names" in prompt
    assert "pyChannel: Channel" in prompt
    assert "never use them in downstream operations" in prompt


@pytest.mark.unit
def test_field_contract_requires_active_source_naming_transform() -> None:
    ok, issues = validate_draft_field_contract(
        _base_draft(),
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
        expected_rename_capitalize=True,
    )

    assert not ok
    assert any("must include exactly one rename_capitalize" in issue for issue in issues)


@pytest.mark.unit
def test_copilot_filter_uses_effective_field_after_rename_capitalize() -> None:
    result = run_copilot_tool_loop(
        prompt="Only keep Web interactions on Channel.",
        draft=_rename_capitalize_draft(),
        call_model=lambda _prompt: (
            '{"reply":"Filtering Channel.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"Channel","value":"Web"}}],"questions":[]}'
        ),
        validate=ai_config_studio_page.validate_draft_catalog,
        operation_policy=ai_config_studio_page._copilot_operation_policy("4. Filters"),
        approved_fields=["Channel"],
    )

    assert result.iterations == 1
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    transforms = result.pending_draft["pipelines"]["sources"][0]["transforms"]
    assert transforms == [
        {"kind": "rename_capitalize"},
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "Channel", "value": "Web"},
        },
    ]


@pytest.mark.unit
def test_copilot_filter_normalizes_raw_field_without_provider_repair() -> None:
    calls = 0

    def call_model(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return (
            '{"reply":"Filtering the source field.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"pyChannel","value":"Web"}}],"questions":[]}'
        )

    result = run_copilot_tool_loop(
        prompt="Only keep Web interactions on Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        operation_policy=ai_config_studio_page._copilot_operation_policy("4. Filters"),
        approved_fields=["Channel"],
        field_name_mapping={"pyChannel": "Channel"},
    )

    assert calls == 1
    assert result.iterations == 1
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    expression = result.pending_draft["pipelines"]["sources"][0]["transforms"][1]["expression"]
    assert expression == {"op": "eq", "column": "Channel", "value": "Web"}


@pytest.mark.unit
def test_operation_field_normalization_covers_schema_slots_without_changing_values() -> None:
    operations = [
        {
            "op": "set_source_filter",
            "source": "ih",
            "expression": {
                "op": "and",
                "args": [
                    {"op": "eq", "column": "pyChannel", "value": "pyChannel"},
                    {"op": "is_not_null", "arg": {"col": "pyCustomerID"}},
                ],
            },
        },
        {
            "op": "set_calculated_field",
            "source": "ih",
            "name": "pyChannel",
            "expression": {"col": "pyChannel"},
        },
        {
            "op": "set_processor",
            "processor": {
                "id": "engagement",
                "source": "ih",
                "kind": "binary_outcome",
                "dimensions": ["pyChannel"],
                "time": {"column": "pxOutcomeTime", "grains": ["Summary"]},
                "outcome": {
                    "column": "pyOutcome",
                    "positive_values": ["pyChannel"],
                    "negative_values": ["Impression"],
                },
                "filter": {"op": "eq", "column": "pyChannel", "value": "pyChannel"},
                "states": {"ChannelState": {"type": "value_sum", "source_column": "pyChannel"}},
            },
        },
        {
            "op": "set_metric",
            "name": "ChannelLift",
            "metric": {
                "source": "engagement",
                "kind": "variant_compare",
                "variant_column": "pyChannel",
                "expression": {"col": "pyChannel"},
            },
        },
        {
            "op": "set_tile",
            "dashboard": "overview",
            "page": "engagement",
            "tile": {
                "id": "by_channel",
                "title": "pyChannel performance",
                "metric": "CTR",
                "chart": "bar",
                "x": "pyChannel",
                "filters": {"pyChannel": "pyChannel"},
                "labels": {"pyChannel": "pyChannel label"},
            },
        },
        {
            "op": "set_dashboards",
            "dashboards": {
                "dashboards": [
                    {
                        "id": "overview",
                        "pages": [
                            {
                                "id": "engagement",
                                "filters": [{"field": "pyChannel", "default": "pyChannel"}],
                                "tiles": [
                                    {
                                        "id": "by_channel",
                                        "metric": "CTR",
                                        "chart": "bar",
                                        "x": "pyChannel",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
        },
    ]

    normalized = remap_operation_field_names(
        operations,
        {
            "pyChannel": "Channel",
            "pyCustomerID": "CustomerID",
            "pxOutcomeTime": "OutcomeTime",
            "pyOutcome": "Outcome",
        },
    )

    filter_expression = normalized[0]["expression"]
    assert filter_expression["args"][0] == {
        "op": "eq",
        "column": "Channel",
        "value": "pyChannel",
    }
    assert filter_expression["args"][1]["arg"]["col"] == "CustomerID"
    assert normalized[1]["name"] == "pyChannel"
    assert normalized[1]["expression"]["col"] == "Channel"
    processor = normalized[2]["processor"]
    assert processor["dimensions"] == ["Channel"]
    assert processor["time"]["column"] == "OutcomeTime"
    assert processor["outcome"]["column"] == "Outcome"
    assert processor["outcome"]["positive_values"] == ["pyChannel"]
    assert processor["filter"]["column"] == "Channel"
    assert processor["filter"]["value"] == "pyChannel"
    assert processor["states"]["ChannelState"]["source_column"] == "Channel"
    assert normalized[3]["metric"]["variant_column"] == "Channel"
    assert normalized[3]["metric"]["expression"]["col"] == "pyChannel"
    tile = normalized[4]["tile"]
    assert tile["title"] == "pyChannel performance"
    assert tile["x"] == "Channel"
    assert tile["filters"] == {"Channel": "pyChannel"}
    assert tile["labels"] == {"Channel": "pyChannel label"}
    page = normalized[5]["dashboards"]["dashboards"][0]["pages"][0]
    assert page["filters"] == [{"field": "Channel", "default": "pyChannel"}]
    assert page["tiles"][0]["x"] == "Channel"


@pytest.mark.unit
def test_copilot_filter_repairs_stale_raw_field_after_rename_capitalize() -> None:
    responses = iter(
        [
            '{"reply":"Filtering the raw field.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"pyChannel","value":"Web"}}],"questions":[]}',
            '{"reply":"Filtering the effective field.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"Channel","value":"Web"}}],"questions":[]}',
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Only keep Web interactions on Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        operation_policy=ai_config_studio_page._copilot_operation_policy("4. Filters"),
        approved_fields=["Channel"],
    )

    assert result.iterations == 2
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    expression = result.pending_draft["pipelines"]["sources"][0]["transforms"][1]["expression"]
    assert expression == {"op": "eq", "column": "Channel", "value": "Web"}
    assert "Validation or operation errors:" in prompts[1]
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]
    assert "rename_capitalize" in prompts[1]


@pytest.mark.unit
def test_copilot_calculation_repairs_stale_raw_field_after_rename_capitalize() -> None:
    responses = iter(
        [
            _operation_response(
                "Copying the raw field.",
                {
                    "op": "set_calculated_field",
                    "source": "ih",
                    "name": "ChannelCopy",
                    "expression": {"col": "pyChannel"},
                },
            ),
            _operation_response(
                "Copying the effective field.",
                {
                    "op": "set_calculated_field",
                    "source": "ih",
                    "name": "ChannelCopy",
                    "expression": {"col": "Channel"},
                },
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Create ChannelCopy from Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel"],
    )

    assert result.iterations == 2
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    transform = result.pending_draft["pipelines"]["sources"][0]["transforms"][1]
    assert transform["expression"] == {"col": "Channel"}
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_copilot_default_repairs_stale_post_rename_output_name() -> None:
    responses = iter(
        [
            _operation_response(
                "Defaulting the raw field.",
                {
                    "op": "set_source_default",
                    "source": "ih",
                    "field": "pyChannel",
                    "value": "Web",
                },
            ),
            json.dumps(
                {
                    "reply": "Removing the raw default and using the effective field.",
                    "operations": [
                        {
                            "op": "remove_source_default",
                            "source": "ih",
                            "field": "pyChannel",
                        },
                        {
                            "op": "set_source_default",
                            "source": "ih",
                            "field": "Channel",
                            "value": "Web",
                        },
                    ],
                    "questions": [],
                }
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Default missing Channel values to Web.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel"],
    )

    assert result.iterations == 2
    assert result.pending_draft is not None
    transforms = result.pending_draft["pipelines"]["sources"][0]["transforms"]
    assert transforms[1] == {"kind": "defaults", "values": {"Channel": "Web"}}
    assert "creates stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_copilot_calculation_repairs_stale_post_rename_output_name() -> None:
    responses = iter(
        [
            _operation_response(
                "Writing a colliding raw output.",
                {
                    "op": "set_calculated_field",
                    "source": "ih",
                    "name": "pyChannel",
                    "expression": {"col": "Channel"},
                },
            ),
            json.dumps(
                {
                    "reply": "Removing the collision and writing a new output.",
                    "operations": [
                        {
                            "op": "remove_calculated_field",
                            "source": "ih",
                            "name": "pyChannel",
                        },
                        {
                            "op": "set_calculated_field",
                            "source": "ih",
                            "name": "ChannelCopy",
                            "expression": {"col": "Channel"},
                        },
                    ],
                    "questions": [],
                }
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Create a copy of Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel"],
    )

    assert result.iterations == 2
    assert result.pending_draft is not None
    transform = result.pending_draft["pipelines"]["sources"][0]["transforms"][1]
    assert transform["output"] == "ChannelCopy"
    assert "creates stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_copilot_processor_repairs_stale_raw_field_after_rename_capitalize() -> None:
    responses = iter(
        [
            _operation_response(
                "Grouping by the raw field.",
                {"op": "set_processor", "processor": _binary_processor("pyChannel")},
            ),
            _operation_response(
                "Grouping by the effective field.",
                {"op": "set_processor", "processor": _binary_processor("Channel")},
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Group the engagement processor by Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel", "Outcome", "OutcomeTime"],
    )

    assert result.iterations == 2
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    processor = result.pending_draft["processors"]["processors"][0]
    assert processor["dimensions"] == ["Channel"]
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_copilot_cumulative_repair_cannot_leave_a_stale_processor() -> None:
    responses = iter(
        [
            _operation_response(
                "Grouping by the raw field.",
                {"op": "set_processor", "processor": _binary_processor("pyChannel")},
            ),
            _operation_response(
                "Adding a valid metric.",
                {
                    "op": "set_metric",
                    "name": "Total",
                    "metric": {
                        "source": "engagement",
                        "kind": "formula",
                        "expression": {"col": "Count"},
                    },
                },
            ),
        ]
    )

    result = run_copilot_tool_loop(
        prompt="Group by Channel and add a total metric.",
        draft=_rename_capitalize_draft(),
        call_model=lambda _prompt: next(responses),
        validate=ai_config_studio_page.validate_draft_catalog,
        max_iterations=2,
        approved_fields=["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        field_contract_source_id="ih",
    )

    assert result.pending_draft is None
    assert any("pyChannel" in issue for issue in result.validation_issues)
    assert any("processor 'engagement'" in issue for issue in result.validation_issues)


@pytest.mark.unit
def test_copilot_metric_repairs_stale_variant_column_after_rename_capitalize() -> None:
    def metric(variant_column: str) -> dict:
        return {
            "source": "engagement",
            "kind": "variant_compare",
            "variant_column": variant_column,
            "test_role": "Test",
            "control_role": "Control",
        }

    responses = iter(
        [
            _operation_response(
                "Comparing the raw variant field.",
                {
                    "op": "set_metric",
                    "name": "ChannelLift",
                    "metric": metric("pyChannel"),
                },
            ),
            _operation_response(
                "Comparing the effective variant field.",
                {
                    "op": "set_metric",
                    "name": "ChannelLift",
                    "metric": metric("Channel"),
                },
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Compare test and control outcomes by Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        field_contract_source_id="ih",
    )

    assert result.iterations == 2
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    metric_definition = result.pending_draft["metrics"]["metrics"]["ChannelLift"]
    assert metric_definition["variant_column"] == "Channel"
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_copilot_processor_contract_accepts_score_column_rows_and_state_ids() -> None:
    processor = {
        "id": "engagement",
        "source": "ih",
        "kind": "score_distribution",
        "dimensions": ["Channel"],
        "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
        "outcome_column": "Outcome",
        "score_properties": ["Propensity", "FinalPropensity"],
        "score_columns": [
            {"column": "Propensity", "state": "Propensity_tdigest"},
            {"column": "FinalPropensity", "state": "FinalPropensity_tdigest"},
        ],
        "states": {
            "Count": {"type": "count"},
            "Positives": {"type": "count"},
            "Negatives": {"type": "count"},
            "Propensity_tdigest_positives": {
                "type": "tdigest",
                "source_column": "Propensity",
            },
        },
    }
    result = run_copilot_tool_loop(
        prompt="Use the approved score fields.",
        draft=_rename_capitalize_draft(),
        call_model=lambda _prompt: _operation_response(
            "Updating the score processor.",
            {"op": "set_processor", "processor": processor},
        ),
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=[
            "Channel",
            "Outcome",
            "OutcomeTime",
            "Propensity",
            "FinalPropensity",
        ],
    )

    assert result.iterations == 1
    assert result.validation_issues == ()
    assert result.pending_draft is not None


@pytest.mark.unit
def test_copilot_tile_repairs_stale_raw_field_and_allows_metric_results() -> None:
    responses = iter(
        [
            _operation_response(
                "Using the raw field.",
                {
                    "op": "set_tile",
                    "dashboard": "overview",
                    "page": "engagement",
                    "tile": {
                        "id": "channel_ctr",
                        "title": "Channel CTR",
                        "metric": "CTR",
                        "chart": "bar",
                        "x": "pyChannel",
                        "y": "Count",
                        "error_y": "Positives",
                    },
                },
            ),
            _operation_response(
                "Using the effective field.",
                {
                    "op": "set_tile",
                    "dashboard": "overview",
                    "page": "engagement",
                    "tile": {
                        "id": "channel_ctr",
                        "title": "Channel CTR",
                        "metric": "CTR",
                        "chart": "bar",
                        "x": "Channel",
                        "y": "Count",
                        "error_y": "Positives",
                    },
                },
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Add Channel CTR with count error bars.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel"],
    )

    assert result.iterations == 2
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    tiles = result.pending_draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"]
    tile = next(item for item in tiles if item["id"] == "channel_ctr")
    assert tile["x"] == "Channel"
    assert tile["y"] == "Count"
    assert tile["error_y"] == "Positives"
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "tile_fields",
    [
        pytest.param({"facets": {"row": "pyChannel"}}, id="facets-mapping"),
        pytest.param({"error_y_plus": "pyChannel"}, id="error-y-plus"),
        pytest.param({"error_y_minus": "pyChannel"}, id="error-y-minus"),
        pytest.param({"hover_name": "pyChannel"}, id="hover-name"),
        pytest.param({"measure": "pyChannel"}, id="measure"),
        pytest.param({"z": "pyChannel"}, id="z"),
        pytest.param({"fallback_property": "pyChannel"}, id="fallback-property"),
        pytest.param({"labels": {"pyChannel": "Channel"}}, id="labels-mapping"),
    ],
)
def test_draft_field_contract_rejects_stale_extended_tile_fields(
    tile_fields: dict,
) -> None:
    draft = _rename_capitalize_draft()
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile.update(tile_fields)

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("pyChannel" in issue for issue in issues)
    assert any("set_dashboards tile" in issue for issue in issues)


@pytest.mark.unit
@pytest.mark.parametrize("reference_key", ["reference", "delta_reference"])
def test_draft_field_contract_accepts_numeric_string_tile_reference(
    reference_key: str,
) -> None:
    draft = _rename_capitalize_draft()
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile.clear()
    tile.update(
        {
            "id": "ctr",
            "title": "CTR",
            "metric": "CTR",
            "chart": "kpi_card",
            "value": "CTR",
            reference_key: "0.25",
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "calendar_field",
    ["day", "week", "month", "quarter", "year", "as_of_date"],
)
def test_draft_field_contract_accepts_lowercase_calendar_tile_fields(
    calendar_field: str,
) -> None:
    draft = _rename_capitalize_draft()
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile["x"] = calendar_field

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []


@pytest.mark.unit
def test_draft_field_contract_validates_funnel_stage_expressions_not_stage_names() -> None:
    draft = _rename_capitalize_draft()
    draft["processors"]["processors"][0] = {
        "id": "engagement",
        "source": "ih",
        "kind": "funnel",
        "dimensions": ["Channel"],
        "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
        "stages": [
            {
                "name": "Impression",
                "when": {"op": "eq", "column": "Outcome", "value": "Impression"},
            },
            {
                "name": "Clicked",
                "when": {"op": "eq", "column": "Outcome", "value": "Clicked"},
            },
        ],
    }
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile.update(
        {
            "chart": "funnel",
            "stages": ["Impression", "Clicked"],
            "color": "Channel",
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []

    tile["stages"] = ["pyChannel"]
    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("stale raw field 'pyChannel' in funnel stages" in issue for issue in issues)

    tile["stages"] = ["Impression", "Clicked"]
    draft["processors"]["processors"][0]["stages"][1]["when"] = {
        "op": "eq",
        "column": "pyChannel",
        "value": "Web",
    }
    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("stale raw field 'pyChannel'" in issue for issue in issues)
    assert all("'Impression'" not in issue and "'Clicked'" not in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_rejects_stale_processor_and_accepts_effective_field() -> None:
    draft = _rename_capitalize_draft()

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []

    stale = copy.deepcopy(draft)
    stale["processors"]["processors"][0]["dimensions"] = ["pyChannel"]

    ok, issues = validate_draft_field_contract(
        stale,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("stale raw field 'pyChannel'" in issue for issue in issues)
    assert any("Use 'Channel'" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_rejects_implicit_scalar_state_source_column() -> None:
    draft = _rename_capitalize_draft()
    draft["processors"]["processors"][0]["states"] = {
        "Count": {"type": "count"},
        "Positives": {"type": "count"},
        "Negatives": {"type": "count"},
        "pyRevenue": {"type": "value_sum"},
    }

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime", "Revenue"],
        source_id="ih",
    )

    assert not ok
    assert any("processor 'engagement'" in issue and "pyRevenue" in issue for issue in issues)
    assert any("Use 'Revenue'" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_revalidates_baseline_objects_when_rename_changes() -> None:
    processor_baseline = _rename_capitalize_draft()
    processor_baseline["pipelines"]["sources"][0]["transforms"] = []
    processor_baseline["processors"]["processors"][0]["dimensions"] = ["pyChannel"]
    processor_candidate = copy.deepcopy(processor_baseline)
    processor_candidate["pipelines"]["sources"][0]["transforms"] = [{"kind": "rename_capitalize"}]

    processor_ok, processor_issues = validate_draft_field_contract(
        processor_candidate,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
        baseline_draft=processor_baseline,
    )

    assert not processor_ok
    assert any(
        "processor 'engagement'" in issue and "pyChannel" in issue for issue in processor_issues
    )

    dashboard_baseline = _rename_capitalize_draft()
    dashboard_baseline["pipelines"]["sources"][0]["transforms"] = []
    dashboard_baseline["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]["x"] = "pyChannel"
    dashboard_candidate = copy.deepcopy(dashboard_baseline)
    dashboard_candidate["pipelines"]["sources"][0]["transforms"] = [{"kind": "rename_capitalize"}]

    dashboard_ok, dashboard_issues = validate_draft_field_contract(
        dashboard_candidate,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
        baseline_draft=dashboard_baseline,
    )

    assert not dashboard_ok
    assert any(
        "set_dashboards tile" in issue and "pyChannel" in issue for issue in dashboard_issues
    )


@pytest.mark.unit
def test_draft_field_contract_rejects_inactive_source_mutation_from_baseline() -> None:
    baseline = _rename_capitalize_draft()
    baseline["pipelines"]["sources"].append(
        {
            "id": "legacy",
            "reader": {"kind": "csv", "file_pattern": "legacy/*.csv"},
            "schema": {
                "timestamp_column": "LegacyTime",
                "natural_key": ["LegacyID"],
            },
        }
    )
    candidate = copy.deepcopy(baseline)
    candidate["pipelines"]["sources"][1]["transforms"] = [
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "pyChannel", "value": "Web"},
        }
    ]

    ok, issues = validate_draft_field_contract(
        candidate,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
        baseline_draft=baseline,
    )

    assert not ok
    assert any(
        "outside the active sampled-source contract" in issue and "non-active sources" in issue
        for issue in issues
    )


@pytest.mark.unit
def test_draft_field_contract_rejects_inactive_dependency_mutations() -> None:
    baseline = _rename_capitalize_draft()
    baseline["pipelines"]["sources"].append(
        {
            "id": "legacy",
            "reader": {"kind": "csv", "file_pattern": "legacy/*.csv"},
            "schema": {"timestamp_column": "LegacyTime", "natural_key": ["LegacyID"]},
        }
    )
    baseline["processors"]["processors"].append(
        {
            "id": "legacy_engagement",
            "source": "legacy",
            "kind": "binary_outcome",
            "dimensions": ["LegacyChannel"],
            "time": {"column": "LegacyTime", "grains": ["Day", "Summary"]},
            "outcome": {
                "column": "LegacyOutcome",
                "positive_values": ["Yes"],
                "negative_values": ["No"],
            },
        }
    )
    baseline["metrics"]["metrics"]["LegacyCTR"] = {
        "source": "legacy_engagement",
        "kind": "formula",
        "expression": {"col": "Count"},
    }
    baseline["dashboards"]["dashboards"][0]["pages"][0]["tiles"].append(
        {
            "id": "legacy_ctr",
            "title": "Legacy CTR",
            "metric": "LegacyCTR",
            "chart": "line",
            "x": "Day",
            "y": "LegacyCTR",
        }
    )

    processor_candidate = copy.deepcopy(baseline)
    processor_candidate["processors"]["processors"][1]["dimensions"] = ["inventedField"]
    metric_candidate = copy.deepcopy(baseline)
    metric_candidate["metrics"]["metrics"]["LegacyCTR"]["expression"] = {"col": "Positives"}
    report_candidate = copy.deepcopy(baseline)
    report_candidate["dashboards"]["dashboards"][0]["pages"][0]["tiles"][1]["x"] = "inventedField"

    for candidate, artifact in (
        (processor_candidate, "processor"),
        (metric_candidate, "metric"),
        (report_candidate, "report artifact"),
    ):
        ok, issues = validate_draft_field_contract(
            candidate,
            ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
            source_id="ih",
            baseline_draft=baseline,
        )

        assert not ok
        assert any(f"changes a {artifact} outside" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_rejects_stale_polars_processor_filter() -> None:
    draft = _rename_capitalize_draft()
    draft["processors"]["processors"][0]["filter"] = {"polars": "pl.col('pyChannel') == 'Web'"}

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("stale raw field 'pyChannel'" in issue for issue in issues)
    assert any("processor 'engagement'" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_rejects_stale_post_rename_outputs() -> None:
    draft = _rename_capitalize_draft()
    draft["pipelines"]["sources"][0]["transforms"].extend(
        [
            {"kind": "defaults", "values": {"pyChannel": "Web"}},
            {
                "kind": "derive_column",
                "output": "pyCustomerID",
                "expression": {"col": "CustomerID"},
            },
        ]
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("creates stale raw field 'pyChannel'" in issue for issue in issues)
    assert any("creates stale raw field 'pyCustomerID'" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_allows_new_calculated_output_downstream() -> None:
    draft = _rename_capitalize_draft()
    draft["pipelines"]["sources"][0]["transforms"].append(
        {
            "kind": "derive_column",
            "output": "ChannelCopy",
            "expression": {"col": "Channel"},
        }
    )
    draft["processors"]["processors"][0]["dimensions"] = ["ChannelCopy"]

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []


@pytest.mark.unit
def test_draft_field_contract_allows_hidden_local_filter_without_exposing_it_downstream() -> None:
    draft = _rename_capitalize_draft()
    draft["pipelines"]["sources"][0]["transforms"].append(
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "HiddenFlag", "value": True},
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
        source_fields=[
            "Channel",
            "CustomerID",
            "HiddenFlag",
            "Outcome",
            "OutcomeTime",
        ],
    )

    assert ok
    assert issues == []

    draft["processors"]["processors"][0]["dimensions"] = ["HiddenFlag"]
    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
        source_fields=[
            "Channel",
            "CustomerID",
            "HiddenFlag",
            "Outcome",
            "OutcomeTime",
        ],
    )

    assert not ok
    assert any("processor 'engagement'" in issue and "HiddenFlag" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_blocks_missing_explicit_active_source() -> None:
    draft = _rename_capitalize_draft()
    draft["pipelines"]["sources"][0]["id"] = "renamed_by_candidate"

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("active field-contract source 'ih' does not exist" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_rejects_source_field_not_persisted_by_processor() -> None:
    draft = _rename_capitalize_draft()
    draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]["x"] = "Outcome"

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("Outcome" in issue and "queryable fields" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_respects_lifecycle_metric_output_subset() -> None:
    draft = _rename_capitalize_draft()
    draft["processors"]["processors"] = [
        {
            "id": "engagement",
            "source": "ih",
            "kind": "entity_lifecycle",
            "group_by": ["Channel"],
            "time": {"column": "OutcomeTime", "grains": ["Summary"]},
            "keys": {
                "customer_id": "CustomerID",
                "order_id": "Outcome",
                "monetary": "Outcome",
                "purchase_date": "OutcomeTime",
            },
        }
    ]
    draft["metrics"]["metrics"] = {
        "Lifecycle": {
            "source": "engagement",
            "kind": "lifecycle_summary",
            "outputs": ["frequency"],
        }
    }
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile.update(
        {
            "metric": "Lifecycle",
            "chart": "bar",
            "x": "Channel",
            "y": "lifetime_value",
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("lifetime_value" in issue and "queryable fields" in issue for issue in issues)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outputs", "metric_extra", "tile_field"),
    [
        pytest.param(
            ["frequency", "bogus"],
            {},
            "bogus",
            id="unknown-configured-output",
        ),
        pytest.param(
            ["frequency"],
            {"output": "lifetime_value"},
            "lifetime_value",
            id="ignored-singular-output",
        ),
        pytest.param(
            ["frequency"],
            {},
            "Lifecycle",
            id="metric-name",
        ),
    ],
)
def test_draft_field_contract_rejects_non_queryable_lifecycle_fields(
    outputs: list[str],
    metric_extra: dict[str, str],
    tile_field: str,
) -> None:
    draft = _rename_capitalize_draft()
    draft["processors"]["processors"] = [
        {
            "id": "engagement",
            "source": "ih",
            "kind": "entity_lifecycle",
            "group_by": ["Channel"],
            "time": {"column": "OutcomeTime", "grains": ["Summary"]},
            "keys": {
                "customer_id": "CustomerID",
                "order_id": "Outcome",
                "monetary": "Outcome",
                "purchase_date": "OutcomeTime",
            },
        }
    ]
    draft["metrics"]["metrics"] = {
        "Lifecycle": {
            "source": "engagement",
            "kind": "lifecycle_summary",
            "outputs": outputs,
            **metric_extra,
        }
    }
    tile = draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"][0]
    tile.update(
        {
            "metric": "Lifecycle",
            "chart": "bar",
            "x": "Channel",
            "y": tile_field,
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any(tile_field in issue and "queryable fields" in issue for issue in issues)


@pytest.mark.unit
def test_draft_field_contract_scopes_validation_to_active_source() -> None:
    draft = _rename_capitalize_draft()
    draft["pipelines"]["sources"].append(
        {
            "id": "legacy",
            "reader": {"kind": "csv", "file_pattern": "legacy/*.csv"},
            "schema": {
                "timestamp_column": "LegacyTime",
                "natural_key": ["LegacyID"],
            },
        }
    )
    draft["processors"]["processors"].append(
        {
            "id": "legacy_engagement",
            "source": "legacy",
            "kind": "binary_outcome",
            "dimensions": ["pyChannel"],
            "time": {"column": "LegacyTime", "grains": ["Day", "Summary"]},
            "outcome": {
                "column": "LegacyOutcome",
                "positive_values": ["Yes"],
                "negative_values": ["No"],
            },
        }
    )
    draft["metrics"]["metrics"]["LegacyCTR"] = {
        "source": "legacy_engagement",
        "kind": "formula",
        "expression": {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {"col": "Count"},
        },
    }
    page = draft["dashboards"]["dashboards"][0]["pages"][0]
    page["filters"] = [
        {"field": "Channel", "label": "Channel"},
        {
            "field": "pyChannel",
            "label": "Legacy channel",
            "scope": "compatible_tiles",
        },
    ]
    page["tiles"].append(
        {
            "id": "legacy_ctr",
            "title": "Legacy CTR",
            "metric": "LegacyCTR",
            "chart": "bar",
            "x": "pyChannel",
            "y": "LegacyCTR",
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []

    draft["processors"]["processors"][0]["dimensions"] = ["pyChannel"]
    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert not ok
    assert any("pyChannel" in issue for issue in issues)
    assert all(
        "pyChannel" not in issue for issue in issues if "processor 'engagement'" not in issue
    )
    assert all("LegacyTime" not in issue for issue in issues)


@pytest.mark.unit
def test_copilot_set_dashboards_repairs_stale_filter_and_tile_field() -> None:
    def dashboards(channel_field: str) -> dict:
        return {
            "dashboards": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "pages": [
                        {
                            "id": "engagement",
                            "title": "Engagement",
                            "filters": [{"field": channel_field, "label": "Channel"}],
                            "tiles": [
                                {
                                    "id": "ctr",
                                    "title": "CTR",
                                    "metric": "CTR",
                                    "chart": "bar",
                                    "x": channel_field,
                                    "y": "CTR",
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    responses = iter(
        [
            _operation_response(
                "Using the raw dashboard field.",
                {"op": "set_dashboards", "dashboards": dashboards("pyChannel")},
            ),
            _operation_response(
                "Using the effective dashboard field.",
                {"op": "set_dashboards", "dashboards": dashboards("Channel")},
            ),
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Replace the report and filter by Channel.",
        draft=_rename_capitalize_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        approved_fields=["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        field_contract_source_id="ih",
    )

    assert result.iterations == 2
    assert result.validation_issues == ()
    assert result.pending_draft is not None
    page = result.pending_draft["dashboards"]["dashboards"][0]["pages"][0]
    assert page["filters"][0]["field"] == "Channel"
    assert page["tiles"][0]["x"] == "Channel"
    assert "stale raw field" in prompts[1]
    assert "pyChannel" in prompts[1]


@pytest.mark.unit
def test_draft_field_contract_accepts_authoritative_odds_ratio_result_column() -> None:
    draft = _rename_capitalize_draft()
    draft["metrics"]["metrics"]["Experiment"] = {
        "source": "engagement",
        "kind": "contingency_test",
        "variant_column": "Channel",
        "tests": ["chi2", "g", "z"],
    }
    draft["dashboards"]["dashboards"][0]["pages"][0]["tiles"].append(
        {
            "id": "experiment_odds",
            "title": "Experiment odds ratio",
            "metric": "Experiment",
            "chart": "experiment_odds_ratio",
            "x": "g_odds_ratio_stat",
            "y": "g_odds_ratio_ci_high",
            "error_x": "g_odds_ratio_ci_low",
        }
    )

    ok, issues = validate_draft_field_contract(
        draft,
        ["Channel", "CustomerID", "Outcome", "OutcomeTime"],
        source_id="ih",
    )

    assert ok
    assert issues == []


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
@pytest.mark.parametrize(
    ("step", "heading"),
    [
        ("2. Required Fields", "Configure step 2 · Required Fields with AI"),
        ("7. AI Draft", "Configure step 7 · AI Draft"),
        ("10. AI Reports", "Configure step 10 · AI Reports"),
    ],
)
def test_copilot_heading_names_step_without_repeating_ai(step: str, heading: str) -> None:
    assert ai_config_studio_page._copilot_step_heading(step) == heading


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
    headings = [str(markdown.value) for markdown in at.markdown]
    assert "### Configure step 9 · Metrics with AI" in headings
    assert "### Configure this step with AI" not in headings
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
def test_pending_copilot_details_include_change_table_and_yaml() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

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
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_draft"] = base_draft
        st.session_state["ai_studio_pending_draft"] = pending_draft
        st.session_state["ai_studio_pending_base_draft"] = base_draft
        st.session_state["ai_studio_pending_kind"] = "copilot"
        page._render_pending_draft_review()

    at = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()

    assert not at.exception
    assert len(at.dataframe) == 1
    change_table = at.dataframe[0].value
    assert change_table.columns.to_list() == ["Change", "Configuration", "Item", "Result"]
    assert change_table.to_dict(orient="records") == [
        {
            "Change": "Add",
            "Configuration": "Metric",
            "Item": "Total",
            "Result": "New configuration will be added.",
        }
    ]
    yaml_views = [str(code.value) for code in at.code if "changes:" in str(code.value)]
    assert len(yaml_views) == 1
    assert "action: added" in yaml_views[0]
    assert "configuration: metric" in yaml_views[0]
    assert "item: Total" in yaml_views[0]
    assert "before: null" in yaml_views[0]
    assert "after:" in yaml_views[0]


@pytest.mark.unit
def test_copilot_accepts_effective_filter_after_rename_capitalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    calls: list[str] = []

    def fake_call_litellm(settings: object, prompt: str, **kwargs: object) -> str:
        calls.append(prompt)
        if prompt == "Reply with READY.":
            return "READY"
        return (
            '{"reply":"Filtered web rows.","operations":['
            '{"op":"set_source_filter","source":"ih","expression":'
            '{"op":"eq","column":"Channel","value":"Web"}}],"questions":[]}'
        )

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        raw_fields = ["pyChannel", "pyCustomerID", "pxOutcomeTime", "pyOutcome"]
        effective_fields = ["Channel", "CustomerID", "OutcomeTime", "Outcome"]
        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state[page.AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = True
        st.session_state[page.AI_STUDIO_RENAME_CAPITALIZE_LEGACY_KEY] = True
        st.session_state["ai_studio_rename_capitalize_applied"] = True
        st.session_state["ai_studio_source_id"] = "ih"
        st.session_state["ai_studio_raw_schema_columns"] = raw_fields
        st.session_state["ai_studio_effective_schema_columns"] = effective_fields
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        page._render_copilot_panel(
            "4. Filters",
            pl.DataFrame({field: ["value"] for field in effective_fields}),
            effective_fields,
        )

    at = AppTest.from_function(app, kwargs={"draft": _rename_capitalize_draft()}).run()
    at.chat_input[0].set_value("Filter Channel to Web.").run()

    assert not at.exception
    assert len(calls) == 2
    pending = at.session_state["ai_studio_pending_draft"]
    filters = [
        transform
        for transform in pending["pipelines"]["sources"][0]["transforms"]
        if transform.get("kind") == "filter"
    ]
    assert filters == [
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "Channel", "value": "Web"},
        }
    ]
    assert "pyChannel" not in str(filters)


@pytest.mark.unit
def test_copilot_blocks_provider_call_for_stale_source_naming_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    calls: list[str] = []

    def fake_call(_settings: object, prompt: str, **_kwargs: object) -> str:
        calls.append(prompt)
        return '{"reply":"unexpected","operations":[],"questions":[]}'

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state[page.AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = True
        st.session_state["ai_studio_source_id"] = "ih"
        st.session_state["ai_studio_effective_schema_columns"] = ["Channel", "OutcomeTime"]
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_api_key"] = "test-key"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        page._render_copilot_panel(
            "4. Filters",
            pl.DataFrame({"Channel": ["Web"], "OutcomeTime": ["2026-01-01"]}),
            ["Channel", "OutcomeTime"],
        )

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    at.chat_input[0].set_value("Filter Channel to Web.").run()

    assert not at.exception
    assert calls == []
    assert at.session_state[ai_config_studio_page.AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY]
    assert at.session_state["ai_studio_pending_draft"] is None
    assert "did not call the model" in at.session_state["ai_studio_copilot_history"][-1]["content"]


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
    # A bare column copy now loads as the recognized Copy Field mode.
    assert calculation["Mode"] == "Copy Field"
    assert calculation["Left"] == "Channel"
    assert calculation["Expression"] == ""


@pytest.mark.unit
def test_accepting_source_naming_bundle_syncs_all_preprocessing_editors() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    base = _base_draft()
    pending = copy.deepcopy(base)
    pending_source = pending["pipelines"]["sources"][0]
    pending_source["transforms"] = [
        {"kind": "rename_capitalize"},
        {"kind": "defaults", "values": {"Channel": "Unknown"}},
        {
            "kind": "filter",
            "expression": {"op": "eq", "column": "Channel", "value": "Web"},
        },
        {
            "kind": "derive_column",
            "output": "ChannelCopy",
            "expression": {"col": "Channel"},
        },
    ]

    def app(base_draft: dict, pending_draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state[page.AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = True
        st.session_state.setdefault("ai_studio_source_id", "ih")
        effective_fields = ["Channel", "CustomerID", "Outcome", "OutcomeTime"]
        st.session_state.setdefault("ai_studio_effective_schema_columns", effective_fields)
        st.session_state.setdefault("ai_studio_approved_fields", effective_fields)
        st.session_state.setdefault("ai_studio_draft", base_draft)
        st.session_state.setdefault("ai_studio_pending_draft", pending_draft)
        st.session_state.setdefault("ai_studio_pending_base_draft", base_draft)
        st.session_state.setdefault("ai_studio_pending_kind", "deterministic draft")
        page._consume_preprocessing_editor_sync()
        if st.session_state.get("ai_studio_pending_draft") is not None:
            page._render_pending_draft_review()

    at = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()
    at = next(button for button in at.button if button.label == "Review individually").click().run()
    bundle_checkbox = next(
        checkbox for checkbox in at.checkbox if checkbox.label == "Accept this complete bundle"
    )
    at = bundle_checkbox.check().run()
    at = (
        next(button for button in at.button if button.label == "Accept selected bundles")
        .click()
        .run()
    )

    assert not at.exception
    assert at.session_state["ai_studio_pending_draft"] is None
    assert at.session_state[ai_config_studio_page.AI_STUDIO_SCHEMA_CONTRACT_STALE_KEY] is False
    assert at.session_state["ai_studio_defaults"] == [
        {"Field": "Channel", "Default Value": "Unknown", "Enabled": True}
    ]
    assert at.session_state["ai_studio_filter_rows"][0]["Field"] == "Channel"
    assert at.session_state["ai_studio_calculations"][0]["Name"] == "ChannelCopy"
    assert at.session_state["ai_studio_draft"]["pipelines"]["sources"][0]["transforms"][0] == {
        "kind": "rename_capitalize"
    }


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
