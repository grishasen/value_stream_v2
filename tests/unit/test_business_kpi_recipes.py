"""Business definitions retain billing populations and experiment semantics."""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from valuestream.config import model
from valuestream.processors.binary_outcome import BinaryOutcomeProcessor
from valuestream.processors.context import ChunkContext
from valuestream.query.executor import _derive_metric
from valuestream.recipes.kpi import (
    RecipeInput,
    instantiate_metric,
    instantiate_tile,
    load_builtin_kpi_recipes,
    recipe_binding_options,
    recipe_readiness,
)

pytestmark = pytest.mark.unit


def _recipe(recipe_id: str):
    return next(r for r in load_builtin_kpi_recipes().recipes if r.id == recipe_id)


def _processor(states, *, group_by=None, dedup=None, positive=None, negative=None, filter_=None):
    return model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "business",
            "source": "ih",
            "kind": "binary_outcome",
            "time": {"property": "DecisionTime", "grain": "daily"},
            "group_by": group_by or ["Channel"],
            "states": states,
            "dedup_keys": dedup or [],
            "outcome": {
                "column": "Outcome",
                "positive_values": positive or ["Conversion"],
                "negative_values": ["NoConversion"] if negative is None else negative,
            },
            "filter": filter_,
        }
    )


def _metric(recipe_id, processor, bindings=None):
    recipe = _recipe(recipe_id)
    selected = {**recipe_readiness(recipe, processor).resolved_inputs, **(bindings or {})}
    raw = instantiate_metric(recipe, processor, "Result", selected)
    return model.validate_metric(raw)


def _evaluate(metric, frame, processor, group_columns=None):
    return _derive_metric(
        frame,
        "Result",
        metric,
        {"Result": metric},
        state_specs=processor.states,
        group_columns=group_columns or [],
    )


def _cost_processor(basis):
    noun = "Impression" if basis == "impressions" else "Interaction"
    keys = ["CustomerID", "InteractionID"]
    if basis == "impressions":
        keys += ["ActionID", "Placement", "Rank"]
    return _processor(
        {
            noun + "s": {"type": "count"},
            "Costed" + noun + "s": {"type": "count", "where": {"op": "not_null", "column": "Cost"}},
            noun + "Cost": {"type": "value_sum", "source_column": "Cost"},
        },
        dedup=keys,
        positive=["Impression"],
        negative=[],
        filter_={"op": "eq", "column": "Outcome", "value": "Impression"},
    )


def _billing_rows(last_cost=3.0):
    # Two actions in interaction i1, one in i2, a duplicate impression and a click callback.
    rows = [
        ("i1", "a1", 2.0, "Impression"),
        ("i1", "a1", 2.0, "Impression"),
        ("i1", "a2", 2.0, "Impression"),
        ("i2", "a1", last_cost, "Impression"),
        ("i1", "a1", 99.0, "Clicked"),
    ]
    return pl.DataFrame(
        [
            {
                "CustomerID": "c1",
                "InteractionID": interaction,
                "ActionID": action,
                "Placement": "Hero",
                "Rank": 1,
                "Channel": "Web",
                "Outcome": outcome,
                "DecisionTime": dt.datetime(2026, 9, 15),
                "Cost": cost,
            }
            for interaction, action, cost, outcome in rows
        ],
        schema_overrides={"Cost": pl.Float64},
    )


@pytest.mark.parametrize(
    ("basis", "total", "units"), [("impressions", 7.0, 3), ("interactions", 5.0, 2)]
)
def test_marketing_cost_respects_billing_grain_and_deduplicates_callbacks(basis, total, units):
    config = _cost_processor(basis)
    processor = BinaryOutcomeProcessor(config)
    ctx = ChunkContext("test", "2026-09-15", dt.datetime(2026, 9, 15, tzinfo=dt.UTC))
    aggregate = processor.chunk_aggregate(_billing_rows().lazy(), ctx)
    merged = processor.merge_for_query(aggregate, ["Channel"])
    total_metric = _metric("finance.marketing_cost_" + basis, config)
    average_metric = _metric("finance.cost_per_" + basis[:-1], config)
    assert _evaluate(total_metric, merged, config)["Result"][0] == pytest.approx(total)
    assert _evaluate(average_metric, merged, config)["Result"][0] == pytest.approx(total / units)
    assert "CustomerID" not in merged.columns
    assert "InteractionID" not in merged.columns


@pytest.mark.parametrize("basis", ["impressions", "interactions"])
def test_missing_cost_is_not_reported_as_zero_or_partial_total(basis):
    config = _cost_processor(basis)
    processor = BinaryOutcomeProcessor(config)
    ctx = ChunkContext("test", "2026-09-15", dt.datetime(2026, 9, 15, tzinfo=dt.UTC))
    aggregate = processor.merge_for_query(
        processor.chunk_aggregate(_billing_rows(None).lazy(), ctx), []
    )
    for rid in ["finance.marketing_cost_" + basis, "finance.cost_per_" + basis[:-1]]:
        assert _evaluate(_metric(rid, config), aggregate, config)["Result"][0] is None
    coverage = _evaluate(_metric("finance.cost_coverage", config), aggregate, config)["Result"][0]
    assert coverage == pytest.approx(2 / 3 if basis == "impressions" else 1 / 2)
    zero_rows = _billing_rows().with_columns(pl.lit(0.0).alias("Cost"))
    zero = processor.merge_for_query(processor.chunk_aggregate(zero_rows.lazy(), ctx), [])
    assert (
        _evaluate(_metric("finance.marketing_cost_" + basis, config), zero, config)["Result"][0]
        == 0.0
    )
    assert _evaluate(_metric("finance.cost_coverage", config), zero, config)["Result"][0] == 1.0


def test_conversion_revenue_excludes_other_outcomes_and_merges_chunks():
    config = _processor(
        {
            "Revenue": {
                "type": "value_sum",
                "source_column": "Revenue",
                "where": {"op": "eq", "column": "Outcome", "value": "Conversion"},
            },
            "Positives": {"type": "count", "outcome": "positive"},
            "Negatives": {"type": "count", "outcome": "negative"},
        },
        group_by=["Channel", "ProductGroup"],
    )
    processor = BinaryOutcomeProcessor(config)
    frames = []
    for day, rows in [
        (14, [("Cards", "Conversion", 20.0), ("Cards", "NoConversion", 999.0)]),
        (15, [("Cards", "Conversion", 30.0), ("Savings", "Conversion", 40.0)]),
    ]:
        raw = pl.DataFrame(
            [
                {
                    "Channel": "Web",
                    "ProductGroup": product,
                    "Outcome": outcome,
                    "Revenue": revenue,
                    "DecisionTime": dt.datetime(2026, 9, day),
                }
                for product, outcome, revenue in rows
            ]
        )
        frames.append(
            processor.chunk_aggregate(
                raw.lazy(), ChunkContext("test", str(day), dt.datetime(2026, 9, day, tzinfo=dt.UTC))
            )
        )
    merged = processor.merge_for_query(pl.concat(frames), [])
    expected = {
        "finance.revenue": 90.0,
        "finance.revenue_per_conversion": 30.0,
        "finance.revenue_per_opportunity": 22.5,
        "conversion.conversions": 3.0,
        "conversion.conversion_rate": 0.75,
    }
    for rid, result in expected.items():
        assert _evaluate(_metric(rid, config), merged, config)["Result"][0] == pytest.approx(result)
    recipe = _recipe("products.revenue_mix")
    bindings = {"revenue": "Revenue", "product": "ProductGroup"}
    tile = instantiate_tile(recipe, config, "ProductRevenue", "mix", bindings)
    assert tile["x"] == "ProductGroup"
    grouped = processor.merge_for_query(pl.concat(frames), ["ProductGroup"])
    result = _evaluate(_metric(recipe.id, config, bindings), grouped, config)
    assert dict(zip(result["ProductGroup"], result["Result"], strict=True)) == {
        "Cards": 50.0,
        "Savings": 40.0,
    }
    with pytest.raises(ValueError, match="valid binding"):
        _metric(recipe.id, config, {**bindings, "product": "UnstoredProduct"})
    with pytest.raises(ValueError, match="configured group-by"):
        instantiate_tile(
            recipe, config, "ProductRevenue", "mix", {**bindings, "product": "UnstoredProduct"}
        )


def test_revenue_recipe_requires_conversion_filtered_state():
    config = _processor({"Revenue": {"type": "value_sum", "source_column": "Revenue"}})
    recipe = _recipe("finance.revenue")
    readiness = recipe_readiness(recipe, config)
    assert readiness.status == "backfill_required"
    options = recipe_binding_options(recipe.inputs[0], config)
    assert len(options) == 1
    assert not options[0].configured
    assert options[0].state_definition["where"] == {
        "op": "eq",
        "column": "Outcome",
        "value": "Conversion",
    }


def test_experiment_tests_use_correct_arms_effect_sizes_and_grouping():
    config = _processor(
        {
            "Positives": {"type": "count", "outcome": "positive"},
            "Negatives": {"type": "count", "outcome": "negative"},
        },
        group_by=["ExperimentName", "ExperimentGroup"],
        positive=["Clicked"],
        negative=["Impression"],
    )
    rows = pl.DataFrame(
        {
            "ExperimentName": ["E1"] * 3 + ["E2"] * 2,
            "ExperimentGroup": ["Test", "Control", "NBA", "Test", "Control"],
            "Positives": [400, 300, 800, 100, 100],
            "Negatives": [9600, 9700, 9200, 9900, 9900],
        }
    )
    compare = _evaluate(
        _metric("experiments.test_control_comparison", config), rows, config, ["ExperimentName"]
    ).sort("ExperimentName")
    assert compare["AbsoluteRateDifference"].to_list() == pytest.approx([0.01, 0.0])
    assert compare["Lift"].to_list() == pytest.approx([1 / 3, 0.0])
    assert compare["TestSampleSize"].to_list() == [10000, 10000]
    assert (
        compare["AbsoluteRateDifference_CI_Low"][0]
        < 0.01
        < compare["AbsoluteRateDifference_CI_High"][0]
    )
    ztest = _evaluate(_metric("experiments.z_test", config), rows, config, ["ExperimentName"]).sort(
        "ExperimentName"
    )
    expected_z = 0.01 / math.sqrt(0.035 * 0.965 * (2 / 10000))
    assert ztest["Count"].to_list() == [20000, 20000]
    assert ztest["z_score"][0] == pytest.approx(expected_z)
    assert ztest["z_p_val"][0] == pytest.approx(math.erfc(expected_z / math.sqrt(2)))
    all_variants = _evaluate(
        _metric("experiments.chi_square_test", config), rows, config, ["ExperimentName"]
    ).sort("ExperimentName")
    assert all_variants["Count"].to_list() == [30000, 20000]
    assert all_variants["chi2_dof"].to_list() == [2, 1]


def test_statistical_recipes_reject_missing_fixed_states_and_unknown_dimensions():
    config = _processor({"GenericCount": {"type": "count"}}, group_by=["ExperimentGroup"])
    recipe = _recipe("experiments.z_test")
    assert recipe_readiness(recipe, config).status == "incompatible"
    with pytest.raises(ValueError, match="Positives"):
        instantiate_metric(recipe, config, "ZTest", {"variant": "ExperimentGroup"})
    with pytest.raises(ValidationError, match="dimension inputs"):
        RecipeInput(role="product", label="Product", source="dimension", state_types=["count"])


def test_documented_recipe_tables_cover_the_library_once_with_business_examples():
    text = (Path(__file__).parents[2] / "docs/reference/kpi-recipes.md").read_text()
    section = text.split("## Built-in Recipes", 1)[1].split("## Versioning and Governance", 1)[0]
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| Business KPI |")
    ]
    cells = [[part.strip() for part in line.strip("|").split("|")] for line in rows]
    assert all(len(row) == 6 and "**Example:**" in row[1] for row in cells)
    assert Counter(row[2].strip("`") for row in cells) == Counter(
        r.id for r in load_builtin_kpi_recipes().recipes
    )


@pytest.mark.parametrize(
    ("recipe_id", "role", "selected"),
    [
        ("products.revenue_mix", "product", "ProductGroup"),
        ("experiments.z_test", "variant", "ModelControlGroup"),
    ],
)
def test_installer_ui_keeps_selected_business_grouping(recipe_id, role, selected):
    config = _processor(
        {
            "Revenue": {
                "type": "value_sum",
                "source_column": "Revenue",
                "where": {"op": "eq", "column": "Outcome", "value": "Conversion"},
            },
            "Positives": {"type": "count", "outcome": "positive"},
            "Negatives": {"type": "count", "outcome": "negative"},
        },
        group_by=["Group", "ProductGroup", "ExperimentGroup", "ModelControlGroup"],
    )
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "catalog_version": 2,
                "workspace": "business_ui",
                "sources": [{"id": "ih", "reader": {"kind": "csv", "file_pattern": "*.csv"}}],
            },
            "processors": {"catalog_version": 2, "processors": [config]},
            "metrics": {"catalog_version": 2, "metrics": {}},
            "dashboards": {"catalog_version": 2, "dashboards": []},
        }
    )
    app = AppTest.from_string(f"""
import streamlit as st
from valuestream.config import model
from valuestream.recipes import load_builtin_kpi_recipes, recipe_readiness
from valuestream.ui.recipe_library import (
    _render_recipe_bindings, build_recipe_install_request, ReportPageTarget,
)
catalog = model.Catalog.model_validate_json({catalog.model_dump_json()!r})
processor = catalog.processors.processors[0]
recipe = next(r for r in load_builtin_kpi_recipes().recipes if r.id == {recipe_id!r})
selection = _render_recipe_bindings(
    recipe, processor, recipe_readiness(recipe, processor), key_prefix="business",
)
if st.button("Prepare recipe"):
    st.session_state["request"] = build_recipe_install_request(
        catalog=catalog, recipe=recipe, processor=processor, metric_id="BusinessKPI",
        bindings=selection.bindings, state_additions=selection.state_additions,
        report_target=ReportPageTarget(
            dashboard_id="overview", dashboard_title="Overview",
            page_id="business", page_title="Business",
        ),
        tile_id="business_tile",
    )
""").run()
    assert not app.exception
    selector = app.selectbox(key=f"business_{role}_choice")
    option = next(value for value in selector.proto.options if value.replace(" ", "") == selected)
    selector.select(option).run()
    app.button[0].click().run()
    assert not app.exception
    request = app.session_state["request"]
    assert request.recipe_id == recipe_id
    if role == "product":
        assert request.tile_def["x"] == selected
    else:
        assert model.validate_metric(request.metric_def).variant_column == selected
