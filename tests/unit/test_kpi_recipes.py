"""Reusable KPI recipe artifact tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

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
    assert len(library.recipes) == 11
    assert len({recipe.id for recipe in library.recipes}) == len(library.recipes)
    assert {recipe.domain for recipe in library.recipes} >= {
        "Audience",
        "Distribution",
        "Engagement",
        "Funnel",
    }


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
