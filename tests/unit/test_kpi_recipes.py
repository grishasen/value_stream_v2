"""Reusable KPI recipe artifact tests."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

from valuestream.charts.recipes import RECIPES as CHART_RECIPES
from valuestream.config import model
from valuestream.recipes import (
    instantiate_metric,
    instantiate_tile,
    load_builtin_kpi_recipes,
    processor_recipe_fields,
    processor_with_recipe_states,
    recipe_binding_options,
    recipe_readiness,
    unique_artifact_id,
)
from valuestream.recipes._schema_gen import generate_schema
from valuestream.ui import recipe_library


@pytest.mark.unit
def test_builtin_recipe_library_is_versioned_and_unique() -> None:
    library = load_builtin_kpi_recipes()
    repo_root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (repo_root / "src" / "valuestream" / "recipes" / "kpis.yaml").read_text()
    )
    Draft202012Validator(generate_schema()).validate(payload)

    assert library.schema_version == 1
    assert len(library.recipes) == 22
    assert len({recipe.id for recipe in library.recipes}) == len(library.recipes)
    assert {recipe.domain for recipe in library.recipes} >= {
        "Audience",
        "Contact policy",
        "Distribution",
        "Engagement",
        "Funnel",
    }
    documented = (repo_root / "docs" / "reference" / "kpi-recipes.md").read_text()
    for recipe in library.recipes:
        assert f"`{recipe.id}`" in documented


@pytest.mark.unit
def test_contact_policy_recipes_require_exact_frequency_response_states() -> None:
    processor = _frequency_response_processor()
    expected = {
        "contact_policy.frequency_marginal_ctr": (
            {"clicks": "Clicks", "contacts": "Contacts"},
            {
                "op": "safe_div",
                "num": {"col": "Clicks"},
                "den": {"col": "Contacts"},
            },
        ),
        "contact_policy.frequency_comparable_ctr": (
            {
                "comparable_clicks": "ComparableClicks",
                "comparable_contacts": "ComparableContacts",
            },
            {
                "op": "safe_div",
                "num": {"col": "ComparableClicks"},
                "den": {"col": "ComparableContacts"},
            },
        ),
        "contact_policy.runner_up_expected_ctr": (
            {
                "runner_propensity_sum": "RunnerPropensitySum",
                "comparable_contacts": "ComparableContacts",
            },
            {
                "op": "safe_div",
                "num": {"col": "RunnerPropensitySum"},
                "den": {"col": "ComparableContacts"},
            },
        ),
        "contact_policy.runner_up_coverage": (
            {
                "comparable_contacts": "ComparableContacts",
                "contacts": "Contacts",
            },
            {
                "op": "safe_div",
                "num": {"col": "ComparableContacts"},
                "den": {"col": "Contacts"},
            },
        ),
        "contact_policy.response_opportunity_margin": (
            {
                "comparable_clicks": "ComparableClicks",
                "runner_propensity_sum": "RunnerPropensitySum",
                "comparable_contacts": "ComparableContacts",
            },
            {
                "op": "safe_div",
                "num": {
                    "op": "sub",
                    "args": [
                        {"col": "ComparableClicks"},
                        {"col": "RunnerPropensitySum"},
                    ],
                },
                "den": {"col": "ComparableContacts"},
            },
        ),
        "contact_policy.priority_opportunity_gap": (
            {
                "focal_priority_comparable_sum": "FocalPriorityComparableSum",
                "runner_priority_comparable_sum": "RunnerPriorityComparableSum",
                "priority_comparable_contacts": "PriorityComparableContacts",
            },
            {
                "op": "safe_div",
                "num": {
                    "op": "sub",
                    "args": [
                        {"col": "FocalPriorityComparableSum"},
                        {"col": "RunnerPriorityComparableSum"},
                    ],
                },
                "den": {"col": "PriorityComparableContacts"},
            },
        ),
    }

    for recipe_id, (bindings, expression) in expected.items():
        recipe = _recipe(recipe_id)
        readiness = recipe_readiness(recipe, processor)

        assert recipe.maturity == "reviewed"
        assert recipe.processor_kinds == ("frequency_response",)
        assert recipe.metric.kind == "formula"
        assert readiness.status == "ready"
        assert readiness.resolved_inputs == bindings
        assert {
            item.role: item.preferred_names for item in recipe.inputs
        } == {role: (state,) for role, state in bindings.items()}
        assert all(item.require_preferred for item in recipe.inputs)
        assert all(item.selection == "automatic" for item in recipe.inputs)
        assert all(not item.state_template and not item.proposed_name for item in recipe.inputs)

        metric = instantiate_metric(
            recipe,
            processor,
            recipe.default_metric_id,
            readiness.resolved_inputs,
        )
        assert metric["expression"] == expression
        caveat = recipe.method.caveat.casefold()
        assert "fixed-window approximation" in caveat
        assert "exposurebucket" in caveat
        assert "configured impression proxies" in caveat
        assert "not measured viewability" in caveat
        assert "no dismiss telemetry" in caveat
        assert "rank 2" in caveat
        assert "next recorded rank" in caveat
        assert "raw propensity" in caveat
        assert "response probability" in caveat
        assert "arbitration diagnostic" in caveat
        assert "never ctr" in caveat

    priority = _recipe("contact_policy.priority_opportunity_gap")
    assert priority.metric.display.unit == "index"
    assert priority.metric.display.value_format == "number"
    assert priority.metric.display.direction == "neutral"
    assert all(
        _recipe(recipe_id).metric.display.unit == "percent"
        for recipe_id in set(expected) - {priority.id}
    )


@pytest.mark.unit
def test_contact_policy_recipes_do_not_propose_generic_replacement_states() -> None:
    recipe = _recipe("contact_policy.frequency_marginal_ctr")
    processor = SimpleNamespace(
        id="frequency",
        kind="frequency_response",
        states={
            "GenericCount": TypeAdapter(model.StateSpec).validate_python({"type": "count"})
        },
    )
    readiness = recipe_readiness(recipe, processor)

    assert readiness.status == "backfill_required"
    assert readiness.input_options == {"clicks": (), "contacts": ()}
    for item in recipe.inputs:
        assert recipe_binding_options(
            item,
            processor,
            readiness.input_options[item.role],
            proposal_fields=["AnyField"],
        ) == []


@pytest.mark.unit
def test_frequency_response_supports_aggregate_recipe_charts() -> None:
    for chart in ("line", "bar", "kpi_card", "combo", "table"):
        assert "frequency_response" in CHART_RECIPES[chart].allowed_processor_kinds

    recipe = _recipe("contact_policy.frequency_marginal_ctr")
    processor = _frequency_response_processor()
    tile = instantiate_tile(recipe, processor, recipe.default_metric_id, "frequency_curve")
    assert tile["chart"] == "line"
    assert tile["x"] == "ExposureBucket"


@pytest.mark.unit
def test_new_diagnostic_recipes_state_their_population_and_interpretation_limits() -> None:
    latency = _recipe("engagement.decision_to_outcome_latency_p95")
    upward = _recipe("decisioning.material_upward_exploration_rate")
    evidence = _recipe("model_quality.implied_evidence_index")
    relative_variance = _recipe("model_quality.relative_exploration_variance")

    assert latency.metric.display.unit == "value"
    assert "source field's unit" in latency.method.caveat
    assert "not system serving latency" in latency.method.caveat

    assert upward.title == "Material upward exploration rate"
    assert upward.domain == "Adaptive decisioning"
    assert upward.processor_kinds == ("numeric_distribution",)
    assert upward.parameters[0].name == "minimum_relative_increase"
    assert upward.parameters[0].default == pytest.approx(0.10)
    assert upward.parameters[0].unit == "percent"
    assert upward.inputs[0].proposed_name == "MaterialExploredUp_Count"
    assert upward.inputs[0].state_template
    assert upward.inputs[0].requires_where
    assert upward.inputs[1].preferred_names == ("Explore_Count",)
    assert upward.inputs[1].require_preferred
    assert "default threshold is 10%" in upward.method.caveat
    assert "holdout" in upward.method.caveat.casefold()

    assert evidence.maturity == "draft"
    assert evidence.metric.display.unit == "index"
    assert "not a response count" in evidence.method.caveat
    assert "not centred exactly" in evidence.method.caveat

    assert relative_variance.maturity == "draft"
    assert "not bounded" in relative_variance.method.caveat
    assert "not a posterior uncertainty probability" in relative_variance.method.caveat


@pytest.mark.unit
def test_material_upward_exploration_requires_dedicated_relative_numeric_states() -> None:
    recipe = _recipe("decisioning.material_upward_exploration_rate")
    states = {
        "Explore_Count": {"type": "count", "source_column": "ExplorationSq"},
        "MaterialExploredUp_Count": _material_upward_state(0.10),
    }
    binary = _binary_processor(states)
    exploration = model.NumericDistributionProcessor.model_validate(
        {
            "id": "exploration",
            "source": "events",
            "kind": "numeric_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "properties": ["Propensity", "ExplorationSq", "ExplorationDelta"],
            "states": states,
        }
    )

    assert recipe_readiness(recipe, binary).status == "incompatible"

    readiness = recipe_readiness(recipe, exploration)
    assert readiness.status == "ready"
    assert readiness.resolved_inputs == {
        "explored": "MaterialExploredUp_Count",
        "observations": "Explore_Count",
    }


@pytest.mark.unit
def test_material_upward_exploration_proposes_state_for_custom_threshold() -> None:
    recipe = _recipe("decisioning.material_upward_exploration_rate")
    processor = model.NumericDistributionProcessor.model_validate(
        {
            "id": "exploration",
            "source": "events",
            "kind": "numeric_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "properties": ["Propensity", "ExplorationSq", "ExplorationDelta"],
            "states": {
                "Explore_Count": {"type": "count", "source_column": "ExplorationSq"},
                "MaterialExploredUp_Count": _material_upward_state(0.10),
            },
        }
    )

    parameter_values = {"minimum_relative_increase": 0.20}
    readiness = recipe_readiness(
        recipe,
        processor,
        parameter_values=parameter_values,
    )
    options = recipe_binding_options(
        recipe.inputs[0],
        processor,
        readiness.input_options["explored"],
        parameter_values=parameter_values,
    )

    assert readiness.status == "backfill_required"
    assert readiness.input_options["explored"] == ()
    assert len(options) == 1
    assert not options[0].configured
    assert options[0].value == "MaterialExploredUp_Count_2"
    assert options[0].state_definition == _material_upward_state(0.20)

    configured = processor_with_recipe_states(
        processor,
        {options[0].value: options[0].state_definition},
    )
    metric = instantiate_metric(
        recipe,
        configured,
        "Material_Upward_20pct",
        {
            "explored": options[0].value,
            "observations": "Explore_Count",
        },
        parameter_values=parameter_values,
    )

    assert metric["recipe"]["parameters"] == parameter_values
    assert metric["expression"]["num"]["col"] == "MaterialExploredUp_Count_2"


@pytest.mark.unit
def test_material_upward_threshold_is_editable_as_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _recipe("decisioning.material_upward_exploration_rate")
    captured: dict[str, object] = {}

    monkeypatch.setattr(recipe_library.st, "container", lambda **_: nullcontext())
    monkeypatch.setattr(recipe_library.st, "write", lambda *_args, **_kwargs: None)

    def number_input(label: str, **kwargs: object) -> float:
        captured["label"] = label
        captured.update(kwargs)
        return 25.0

    monkeypatch.setattr(recipe_library.st, "number_input", number_input)

    values = recipe_library._render_recipe_parameters(recipe, key_prefix="material")

    assert captured["label"] == "Minimum increase over raw score (%)"
    assert captured["value"] == pytest.approx(10.0)
    assert captured["step"] == pytest.approx(1.0)
    assert values == {"minimum_relative_increase": 0.25}


@pytest.mark.unit
def test_unique_entities_recipe_prefers_cpc_but_accepts_hll_and_theta() -> None:
    recipe = _recipe("audience.unique_entities")
    cpc_processor = _binary_processor(
        {"UniqueCustomers_cpc": {"type": "cpc", "source_column": "CustomerID", "lg_k": 11}}
    )
    hll_processor = _binary_processor(
        {"LegacyAudience_hll": {"type": "hll", "source_column": "CustomerID", "lg_k": 12}}
    )
    theta_processor = _binary_processor(
        {
            "ReusableAudience_theta": {
                "type": "theta",
                "source_column": "CustomerID",
                "lg_k": 12,
            }
        }
    )

    cpc = recipe_readiness(recipe, cpc_processor)
    hll = recipe_readiness(recipe, hll_processor)
    theta = recipe_readiness(recipe, theta_processor)

    assert cpc.status == "ready"
    assert cpc.resolved_inputs == {"cardinality_state": "UniqueCustomers_cpc"}
    assert hll.status == "ready"
    assert hll.resolved_inputs == {"cardinality_state": "LegacyAudience_hll"}
    assert theta.status == "ready"
    assert theta.resolved_inputs == {"cardinality_state": "ReusableAudience_theta"}


@pytest.mark.unit
def test_sketch_binding_options_expose_business_field_and_algorithm() -> None:
    recipe = _recipe("audience.unique_entities")
    item = recipe.inputs[0]
    processor = _binary_processor(
        {
            "UniqueSubjects_hll": {
                "type": "hll",
                "source_column": "SubjectID",
                "lg_k": 12,
            },
            "UniqueInteractions_hll": {
                "type": "hll",
                "source_column": "InteractionID",
                "lg_k": 12,
            },
        }
    )

    options = recipe_binding_options(item, processor)

    assert [(option.field, option.algorithm, option.label) for option in options] == [
        ("SubjectID", "HLL", "SubjectID · HLL"),
        ("InteractionID", "HLL", "InteractionID · HLL"),
    ]
    assert options[0].value == "UniqueSubjects_hll"
    assert "UniqueSubjects_hll" in options[0].technical_detail
    assert "lg_k=12" in options[0].technical_detail


@pytest.mark.unit
def test_sketch_binding_options_propose_every_grouping_field_and_algorithm() -> None:
    recipe = _recipe("audience.unique_entities")
    item = recipe.inputs[0]
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "events",
            "kind": "binary_outcome",
            "group_by": ["Channel", "Placement"],
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "states": {
                "UniqueSubjects_hll": {
                    "type": "hll",
                    "source_column": "SubjectID",
                    "lg_k": 12,
                }
            },
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )

    options = recipe_binding_options(
        item,
        processor,
        recipe_readiness(recipe, processor).input_options[item.role],
        proposal_fields=processor_recipe_fields(processor),
    )

    assert {(option.field, option.state_type) for option in options} >= {
        ("Channel", "cpc"),
        ("Channel", "hll"),
        ("Channel", "theta"),
        ("Placement", "cpc"),
        ("Placement", "hll"),
        ("Placement", "theta"),
        ("SubjectID", "cpc"),
        ("SubjectID", "hll"),
        ("SubjectID", "theta"),
    }
    channel_cpc = next(
        option for option in options if option.field == "Channel" and option.state_type == "cpc"
    )
    assert not channel_cpc.configured
    assert channel_cpc.state_definition == {
        "type": "cpc",
        "source_column": "Channel",
        "lg_k": 11,
    }

    configured = processor_with_recipe_states(
        processor,
        {channel_cpc.value: channel_cpc.state_definition},
    )
    metric = instantiate_metric(
        recipe,
        configured,
        "Unique_Channels",
        {item.role: channel_cpc.value},
    )

    assert configured.states[channel_cpc.value].type == "cpc"
    assert metric["state"] == channel_cpc.value


@pytest.mark.unit
def test_binding_options_reconcile_stale_generated_state_after_catalog_rerun() -> None:
    recipe = _recipe("audience.unique_entities")
    item = recipe.inputs[0]
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "events",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "states": {"Count": {"type": "count"}},
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )

    options = recipe_binding_options(
        item,
        processor,
        ("Channel_cpc",),
        proposal_fields=["Channel"],
    )

    channel_cpc = next(option for option in options if option.value == "Channel_cpc")
    assert not channel_cpc.configured
    assert channel_cpc.state_definition["source_column"] == "Channel"


@pytest.mark.unit
def test_distribution_binding_options_separate_field_from_algorithm() -> None:
    recipe = _recipe("distribution.median")
    processor = model.NumericDistributionProcessor.model_validate(
        {
            "id": "latency",
            "source": "events",
            "kind": "numeric_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "properties": ["Latency"],
            "states": {
                "Latency_tdigest": {
                    "type": "tdigest",
                    "source_column": "Latency",
                    "k": 500,
                },
                "Latency_kll": {"type": "kll", "source_column": "Latency", "k": 200},
            },
        }
    )

    options = recipe_binding_options(recipe.inputs[0], processor)

    assert {option.field for option in options} == {"Latency"}
    assert {option.algorithm for option in options} == {"t-digest", "KLL"}


@pytest.mark.unit
def test_distribution_recipe_excludes_outcome_conditioned_digests() -> None:
    recipe = _recipe("distribution.p95")
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "scores",
            "source": "events",
            "kind": "score_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "score_properties": [{"column": "Propensity", "role": "primary"}],
            "states": {
                "Propensity_tdigest": {
                    "type": "tdigest",
                    "source_column": "Propensity",
                },
                "Propensity_positive": {
                    "type": "tdigest",
                    "source_column": "Propensity",
                    "score_property": "Propensity",
                    "outcome": "positive",
                },
            },
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )

    readiness = recipe_readiness(recipe, processor)

    assert readiness.input_options["digest_state"] == ("Propensity_tdigest",)


@pytest.mark.unit
def test_exact_outcome_recipe_does_not_offer_arbitrary_count_states() -> None:
    recipe = _recipe("engagement.engagement_rate")
    processor = _binary_processor(
        {
            "Count": {"type": "count"},
            "Accepted": {"type": "count"},
            "Rejected": {"type": "count"},
        }
    )

    readiness = recipe_readiness(recipe, processor)

    assert readiness.status == "backfill_required"
    assert readiness.input_options == {"positives": (), "negatives": ()}


@pytest.mark.unit
def test_recipe_binding_ui_uses_field_and_algorithm_not_state_id() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.config import model as config_model  # noqa: PLC0415
        from valuestream.recipes import (  # noqa: PLC0415
            load_builtin_kpi_recipes as load_recipes,
        )
        from valuestream.recipes import (  # noqa: PLC0415
            recipe_readiness as resolve_readiness,
        )
        from valuestream.ui.recipe_library import _render_recipe_bindings  # noqa: PLC0415

        recipe = next(
            item for item in load_recipes().recipes if item.id == "audience.unique_entities"
        )
        processor = config_model.BinaryOutcomeProcessor.model_validate(
            {
                "id": "engagement",
                    "source": "events",
                    "kind": "binary_outcome",
                    "group_by": ["Channel", "Placement"],
                    "time": {"property": "OutcomeTime", "grain": "daily"},
                    "states": {
                    "UniqueSubjects_hll": {
                        "type": "hll",
                        "source_column": "SubjectID",
                        "lg_k": 12,
                    },
                    "UniqueInteractions_hll": {
                        "type": "hll",
                        "source_column": "InteractionID",
                        "lg_k": 12,
                        },
                    },
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked"],
                        "negative_values": ["Impression"],
                    },
                }
        )
        st.session_state["bindings"] = _render_recipe_bindings(
            recipe,
            processor,
            resolve_readiness(recipe, processor),
            key_prefix="recipe_test",
        )

    at = AppTest.from_function(app).run()

    assert not at.exception
    field = next(widget for widget in at.selectbox if widget.label == "Entity field")
    visible_fields = [
        option
        for option in field.options
        if option in {"Channel", "InteractionID", "Placement", "SubjectID"}
    ]
    assert visible_fields == ["Channel", "InteractionID", "Placement", "SubjectID"]
    assert "UniqueSubjects_hll" not in field.options

    field.set_value("SubjectID").run()

    algorithm = next(widget for widget in at.segmented_control if widget.label == "Algorithm")
    assert algorithm.options == ["CPC", "HLL", "Theta"]
    assert algorithm.value == "CPC"
    selection = at.session_state["bindings"]
    assert selection.bindings == {"cardinality_state": "SubjectID_cpc"}
    assert selection.state_additions["SubjectID_cpc"] == {
        "type": "cpc",
        "source_column": "SubjectID",
        "lg_k": 11,
    }


@pytest.mark.unit
def test_recipe_library_requires_preview_before_returning_install_request() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app() -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.config import model as config_model  # noqa: PLC0415
        from valuestream.ui.recipe_library import (  # noqa: PLC0415
            render_recipe_library as render_library,
        )

        catalog = config_model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "preview",
                    "sources": [
                        {
                            "id": "events",
                            "reader": {"kind": "csv", "file_pattern": "*.csv"},
                        }
                    ],
                },
                "processors": {
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "events",
                            "kind": "binary_outcome",
                            "group_by": ["Channel"],
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {
                                "Count": {"type": "count"},
                                "Positives": {"type": "count", "outcome": "positive"},
                                "Negatives": {"type": "count", "outcome": "negative"},
                            },
                            "outcome": {
                                "column": "Outcome",
                                "positive_values": ["Clicked"],
                                "negative_values": ["Impression"],
                            },
                        }
                    ]
                },
                "metrics": {"catalog_version": 2, "metrics": {}},
                "dashboards": {
                    "catalog_version": 2,
                    "dashboards": [
                        {
                            "id": "overview",
                            "title": "Overview",
                            "pages": [
                                {
                                    "id": "engagement",
                                    "title": "Engagement",
                                    "tiles": [],
                                }
                            ],
                        }
                    ]
                },
            }
        )
        request = render_library(
            catalog=catalog,
            key_prefix="preview_test",
            submit_label="Add recipe to catalog",
            expanded=True,
        )
        if request is not None:
            st.session_state["installed_request"] = request

    at = AppTest.from_function(app).run()

    assert not at.exception
    assert "installed_request" not in at.session_state
    recipe_options = next(widget for widget in at.selectbox if widget.label == "Recipe").options
    assert recipe_options == sorted(recipe_options, key=lambda value: (value.casefold(), value))
    review = next(button for button in at.button if button.label == "Review changes")
    review.click().run()

    assert not at.exception
    assert any(code.language == "yaml" for code in at.code)
    confirm = next(button for button in at.button if button.label == "Add recipe to catalog")
    confirm.click().run()

    assert not at.exception
    assert at.session_state["installed_request"].metric_id == "VS_Engagement_Rate"


@pytest.mark.unit
def test_recipe_readiness_distinguishes_mapping_from_backfill() -> None:
    median = _recipe("distribution.median")
    processor = model.NumericDistributionProcessor.model_validate(
        {
            "id": "latency",
            "source": "events",
            "kind": "numeric_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "properties": ["Latency", "Cost"],
            "states": {
                "Latency_tdigest": {"type": "tdigest", "source_column": "Latency"},
                "Cost_tdigest": {"type": "tdigest", "source_column": "Cost"},
            },
        }
    )
    unique = _recipe("audience.unique_entities")

    assert recipe_readiness(median, processor).status == "mapping_required"
    assert recipe_readiness(unique, processor).status == "backfill_required"


@pytest.mark.unit
def test_recipe_instantiation_materializes_valid_metric_and_tile_with_provenance() -> None:
    recipe = _recipe("engagement.engagement_rate")
    processor = _binary_processor({})
    readiness = recipe_readiness(recipe, processor)

    metric = instantiate_metric(recipe, processor, "Test_Engagement", readiness.resolved_inputs)
    tile = instantiate_tile(recipe, processor, "Test_Engagement", "test_engagement_tile")

    assert metric["expression"]["den"]["op"] == "add"
    assert metric["recipe"] == {"id": recipe.id, "version": recipe.version}
    assert "depends_on" not in metric
    assert tile["metric"] == "Test_Engagement"
    assert tile["placement"] == "kpi_strip"
    assert tile["chart"] == "kpi_card"


@pytest.mark.unit
def test_install_preview_contains_exact_yaml_patch_and_materialization_plan() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "catalog_version": 2,
                "workspace": "preview",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "csv", "file_pattern": "*.csv"},
                        "schema": {"natural_key": ["CustomerID"]},
                    }
                ],
        },
        "processors": {
            "catalog_version": 2,
                "processors": [
                    {
                        "id": "engagement",
                    "source": "events",
                    "kind": "binary_outcome",
                    "group_by": ["Channel"],
                    "time": {"property": "OutcomeTime", "grain": "daily"},
                    "states": {"Count": {"type": "count"}},
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked"],
                        "negative_values": ["Impression"],
                    },
                }
            ]
        },
        "metrics": {"catalog_version": 2, "metrics": {}},
        "dashboards": {
            "catalog_version": 2,
                "dashboards": [
                    {
                        "id": "overview",
                        "title": "Overview",
                        "pages": [{"id": "audience", "title": "Audience", "tiles": []}],
                    }
                ]
            },
        }
    )
    recipe = _recipe("audience.unique_entities")
    processor = catalog.processors.processors[0]
    target = recipe_library.ReportPageTarget(
        dashboard_id="overview",
        dashboard_title="Overview",
        page_id="audience",
        page_title="Audience",
    )
    state_additions = {
        "CustomerID_theta": {
            "type": "theta",
            "source_column": "CustomerID",
            "lg_k": 12,
        }
    }

    request = recipe_library.build_recipe_install_request(
        catalog=catalog,
        recipe=recipe,
        processor=processor,
        metric_id="Unique_Customers",
        bindings={"cardinality_state": "CustomerID_theta"},
        state_additions=state_additions,
        report_target=target,
        tile_id="unique_customers_tile",
    )
    files = recipe_library.recipe_install_preview_files(request)

    assert list(files) == ["processors.yaml", "metrics.yaml", "dashboards.yaml"]
    assert "CustomerID_theta" in files["processors.yaml"]
    assert "Unique_Customers" in files["metrics.yaml"]
    assert "unique_customers_tile" in files["dashboards.yaml"]
    assert request.materialization is not None
    assert request.materialization.source_id == "events"
    assert request.materialization.state_names == ("CustomerID_theta",)
    assert request.materialization.source_fields == ("CustomerID",)
    assert (
        request.materialization.current_computation_hash
        != request.materialization.proposed_computation_hash
    )
    assert recipe_library.recipe_install_fingerprint(request) != (
        recipe_library.recipe_install_fingerprint(
            replace(request, metric_id="Unique_Customers_Changed")
        )
    )


@pytest.mark.unit
def test_roc_recipe_rejects_mismatched_score_digest_pair() -> None:
    recipe = _recipe("model_quality.roc_auc")
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "scores",
            "source": "events",
            "kind": "score_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "score_properties": [
                {"column": "Propensity", "role": "primary"},
                {"column": "Priority", "role": "auxiliary"},
            ],
            "states": {
                "Propensity_positive": {
                    "type": "tdigest",
                    "source_column": "Propensity",
                    "outcome": "positive",
                    "score_property": "Propensity",
                },
                "Priority_negative": {
                    "type": "tdigest",
                    "source_column": "Priority",
                    "outcome": "negative",
                    "score_property": "Priority",
                },
            },
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )

    with pytest.raises(ValueError, match="same score property"):
        instantiate_metric(
            recipe,
            processor,
            "Bad_AUC",
            {
                "positive_digest": "Propensity_positive",
                "negative_digest": "Priority_negative",
            },
        )


@pytest.mark.unit
def test_roc_recipe_can_propose_a_matched_digest_pair_for_a_new_score_field() -> None:
    recipe = _recipe("model_quality.roc_auc")
    processor = model.ScoreDistributionProcessor.model_validate(
        {
            "id": "scores",
            "source": "events",
            "kind": "score_distribution",
            "group_by": ["NewScore"],
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "score_properties": [{"column": "Propensity", "role": "primary"}],
            "states": {"Count": {"type": "count"}},
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )
    readiness = recipe_readiness(recipe, processor)
    positive_item, negative_item = recipe.inputs
    positive = next(
        option
        for option in recipe_binding_options(
            positive_item,
            processor,
            readiness.input_options[positive_item.role],
            proposal_fields=["NewScore"],
        )
        if option.field == "NewScore"
    )
    with_positive = processor_with_recipe_states(
        processor,
        {positive.value: positive.state_definition},
    )
    negative = next(
        option
        for option in recipe_binding_options(
            negative_item,
            with_positive,
            readiness.input_options[negative_item.role],
            proposal_fields=["NewScore"],
        )
        if option.field == "NewScore"
    )
    configured = processor_with_recipe_states(
        with_positive,
        {negative.value: negative.state_definition},
    )

    metric = instantiate_metric(
        recipe,
        configured,
        "New_Score_AUC",
        {
            positive_item.role: positive.value,
            negative_item.role: negative.value,
        },
    )

    assert positive.state_definition["outcome"] == "positive"
    assert negative.state_definition["outcome"] == "negative"
    assert positive.state_definition["score_property"] == "NewScore"
    assert metric["positive_state"] == "NewScore_tdigest_positives"
    assert metric["negative_state"] == "NewScore_tdigest_negatives"


@pytest.mark.unit
def test_funnel_recipe_rejects_identical_start_and_completion() -> None:
    recipe = _recipe("funnel.conversion_rate")
    processor = model.FunnelProcessor.model_validate(
        {
            "id": "funnel",
            "source": "events",
            "kind": "funnel",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "stages": [
                {"name": "Started", "when": {"col": "Started"}},
                {"name": "Completed", "when": {"col": "Completed"}},
            ],
            "states": {
                "Started_Count": {"type": "count", "stage": "Started"},
                "Completed_Count": {"type": "count", "stage": "Completed"},
            },
        }
    )

    with pytest.raises(ValueError, match="must be different"):
        instantiate_metric(
            recipe,
            processor,
            "Bad_Conversion",
            {
                "start_count": "Started_Count",
                "completion_count": "Started_Count",
            },
        )


@pytest.mark.unit
def test_unique_artifact_id_uses_stable_numeric_suffixes() -> None:
    assert unique_artifact_id("VS_Reach", {"VS_Reach", "VS_Reach_2"}) == "VS_Reach_3"


@pytest.mark.unit
def test_recipe_json_schema_matches_checked_in_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    on_disk = json.loads((repo_root / "schemas" / "kpi-recipes.json").read_text())

    assert on_disk == generate_schema(), (
        "schemas/kpi-recipes.json is out of sync; regenerate with: "
        "uv run python -m valuestream.recipes._schema_gen"
    )


def _recipe(recipe_id: str):
    return next(recipe for recipe in load_builtin_kpi_recipes().recipes if recipe.id == recipe_id)


def _material_upward_state(threshold: float) -> dict[str, object]:
    return {
        "type": "count",
        "source_column": "ExplorationDelta",
        "where": {
            "op": "and",
            "args": [
                {"op": "gt", "column": "Propensity", "value": 0.0},
                {
                    "op": "gt",
                    "args": [
                        {"col": "ExplorationDelta"},
                        {
                            "op": "mul",
                            "args": [
                                {"col": "Propensity"},
                                {"lit": threshold},
                            ],
                        },
                    ],
                },
            ],
        },
    }


def _frequency_response_processor() -> SimpleNamespace:
    state_adapter = TypeAdapter(model.StateSpec)
    definitions = {
        "Clicks": {"type": "count"},
        "Contacts": {"type": "count"},
        "ComparableClicks": {"type": "count"},
        "ComparableContacts": {"type": "count"},
        "RunnerPropensitySum": {
            "type": "value_sum",
            "source_column": "RunnerPropensity",
        },
        "FocalPriorityComparableSum": {
            "type": "value_sum",
            "source_column": "FocalPriority",
        },
        "RunnerPriorityComparableSum": {
            "type": "value_sum",
            "source_column": "RunnerPriority",
        },
        "PriorityComparableContacts": {"type": "count"},
    }
    return SimpleNamespace(
        id="frequency",
        kind="frequency_response",
        states={
            name: state_adapter.validate_python(definition)
            for name, definition in definitions.items()
        },
    )


def _binary_processor(states: dict[str, dict[str, object]]) -> model.BinaryOutcomeProcessor:
    if not states:
        states = {
            "Count": {"type": "count"},
            "Positives": {"type": "count", "outcome": "positive"},
            "Negatives": {"type": "count", "outcome": "negative"},
        }
    return model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "events",
            "kind": "binary_outcome",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "states": states,
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
    )


@pytest.mark.unit
def test_distribution_boxplot_recipe_materializes_quantile_free_metric_and_tile() -> None:
    recipe = _recipe("distribution.boxplot")
    processor = model.NumericDistributionProcessor.model_validate(
        {
            "id": "descriptive",
            "source": "ih",
            "kind": "numeric_distribution",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "properties": ["Propensity"],
            "states": {
                "Propensity_tdigest": {
                    "type": "tdigest",
                    "source_column": "Propensity",
                }
            },
        }
    )

    metric_def = instantiate_metric(
        recipe, processor, "PropensityDistribution", {"digest_state": "Propensity_tdigest"}
    )
    assert metric_def["kind"] == "distribution"
    # A distribution metric stores no single quantile; boxplot tiles pull the
    # full quantile suite from the selected metric.
    assert "quantile" not in metric_def
    parsed = model.Metrics.model_validate(
        {"catalog_version": 2, "metrics": {"PropensityDistribution": metric_def}}
    ).metrics["PropensityDistribution"]
    assert isinstance(parsed, model.DistributionMetric)
    assert parsed.state == "Propensity_tdigest"

    tile_def = instantiate_tile(
        recipe,
        processor,
        "PropensityDistribution",
        "tile_dist",
        {"digest_state": "Propensity_tdigest"},
    )
    assert tile_def["chart"] == "boxplot"
    assert tile_def["metric"] == "PropensityDistribution"
    assert "property" not in tile_def
