"""Contract tests for the intentionally non-compatible catalog-v2 schema."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from valuestream.config import model
from valuestream.config.loader import load
from valuestream.processors.grain_levels import aggregate_grain_candidates
from valuestream.recipes import (
    instantiate_metric,
    load_builtin_kpi_recipes,
    processor_with_recipe_states,
    recipe_binding_options,
)
from valuestream.ui import builder

_STATE_ADAPTER = TypeAdapter(model.StateSpec)


@pytest.mark.unit
def test_catalog_version_two_is_required_and_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        model.Pipelines.model_validate({"workspace": "demo", "sources": []})
    with pytest.raises(ValidationError):
        model.Pipelines.model_validate(
            {
                "catalog_version": 2,
                "workspace": "demo",
                "sources": [],
                "legacy_option": True,
            }
        )


@pytest.mark.unit
def test_metrics_use_processor_and_explicit_distribution_kinds() -> None:
    with pytest.raises(ValidationError):
        model.FormulaMetric.model_validate(
            {
                "source": "engagement",
                "kind": "formula",
                "expression": {"lit": 1},
            }
        )

    distribution = model.DistributionMetric.model_validate(
        {
            "processor": "scores",
            "kind": "distribution",
            "state": "Propensity_tdigest",
        }
    )
    quantile = model.QuantileMetric.model_validate(
        {
            "processor": "scores",
            "kind": "quantile",
            "state": "Propensity_tdigest",
            "quantile": 0.95,
        }
    )

    assert distribution.kind == "distribution"
    assert quantile.quantile == pytest.approx(0.95)


@pytest.mark.unit
def test_set_operation_uses_minus_not_legacy_difference_names() -> None:
    payload = {
        "processor": "customers",
        "kind": "set_op",
        "operands": [{"state": "Customers_theta"}, {"state": "Customers_theta"}],
    }
    with pytest.raises(ValidationError):
        model.SetOpMetric.model_validate({**payload, "operation": "a_not_b"})

    metric = model.SetOpMetric.model_validate({**payload, "operation": "minus"})
    assert metric.operation == "minus"


@pytest.mark.unit
def test_chart_contract_rejects_retired_aliases_and_metric_owned_y() -> None:
    base = {"id": "engagement", "title": "Engagement", "metric": "CTR"}
    with pytest.raises(ValidationError):
        model.validate_tile({**base, "chart": "calendar_heatmap", "x": "Day"})
    with pytest.raises(ValidationError):
        model.validate_tile({**base, "chart": "bar", "x": "Channel", "y": "Count"})
    with pytest.raises(ValidationError):
        model.validate_tile({**base, "chart": "heatmap", "x": "Day", "property": "Score"})

    tile = model.validate_tile({**base, "chart": "bar", "x": "Channel"})
    assert tile.chart == "bar"


@pytest.mark.unit
def test_line_tile_accepts_dual_style_dimensions_only_for_lines() -> None:
    base = {"id": "engagement", "title": "Engagement", "metric": "CTR"}

    tile = model.validate_tile(
        {
            **base,
            "chart": "line",
            "x": "Day",
            "color": "Channel",
            "line_dash": "CustomerType",
            "symbol": "CustomerType",
        }
    )

    assert tile.line_dash == "CustomerType"
    assert tile.symbol == "CustomerType"
    with pytest.raises(ValidationError, match="line_dash"):
        model.validate_tile(
            {
                **base,
                "chart": "bar",
                "x": "Channel",
                "line_dash": "CustomerType",
            }
        )


@pytest.mark.unit
def test_boxplot_owns_its_distribution_through_the_selected_metric() -> None:
    base = {
        "id": "scores",
        "title": "Score distribution",
        "metric": "Score_Distribution",
        "chart": "boxplot",
    }

    tile = model.validate_tile(base)
    assert isinstance(tile, model.DistributionTile)
    with pytest.raises(ValidationError, match="property"):
        model.validate_tile({**base, "property": "Score"})


@pytest.mark.unit
def test_processor_write_automatically_persists_digest_metrics(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "pipelines.yaml").write_text(
        yaml.safe_dump(
            {
                "catalog_version": 2,
                "workspace": "automatic_distributions",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (catalog_dir / "processors.yaml").write_text(
        yaml.safe_dump({"catalog_version": 2, "processors": []}, sort_keys=False),
        encoding="utf-8",
    )
    (catalog_dir / "metrics.yaml").write_text(
        yaml.safe_dump({"catalog_version": 2, "metrics": {}}, sort_keys=False),
        encoding="utf-8",
    )
    (catalog_dir / "dashboards.yaml").write_text(
        yaml.safe_dump({"catalog_version": 2, "dashboards": []}, sort_keys=False),
        encoding="utf-8",
    )

    created = builder.write_processor_definition(
        tmp_path,
        {
            "id": "scores",
            "source": "events",
            "kind": "numeric_distribution",
            "group_by": ["Channel"],
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "properties": ["Score", "Priority"],
            "states": {
                "Score_tdigest": {
                    "type": "tdigest",
                    "source_column": "Score",
                    "k": 500,
                },
                "Priority_kll": {
                    "type": "kll",
                    "source_column": "Priority",
                    "k": 200,
                },
            },
        },
    )

    assert created == [
        "scores_metric_score_distribution",
        "scores_metric_priority_distribution",
    ]
    catalog = load(tmp_path)
    assert {
        name: metric.state
        for name, metric in catalog.metrics.metrics.items()
        if isinstance(metric, model.DistributionMetric)
    } == {
        "scores_metric_score_distribution": "Score_tdigest",
        "scores_metric_priority_distribution": "Priority_kll",
    }
    assert builder.write_processor_definition(
        tmp_path,
        catalog.processors.processors[0].model_dump(mode="json", exclude_none=True),
    ) == []


@pytest.mark.unit
def test_state_contract_never_infers_selectors_or_companions_from_names() -> None:
    with pytest.raises(ValidationError):
        _STATE_ADAPTER.validate_python({"type": "count", "distinct": True})
    with pytest.raises(ValidationError):
        _STATE_ADAPTER.validate_python(
            {"type": "count", "source_column": "CustomerID", "outcome": "positive"}
        )
    with pytest.raises(ValidationError):
        _STATE_ADAPTER.validate_python(
            {"type": "pooled_variance", "source_column": "Score"}
        )

    state = _STATE_ADAPTER.validate_python(
        {"type": "count", "source_column": "CustomerID", "distinct": True}
    )
    assert isinstance(state, model.CountState)
    assert state.distinct is True


@pytest.mark.unit
def test_lifecycle_metric_requires_explicit_state_roles() -> None:
    base = {
        "processor": "lifecycle",
        "kind": "lifecycle_summary",
        "entity_column": "CustomerID",
    }
    with pytest.raises(ValidationError):
        model.LifecycleSummaryMetric.model_validate(base)

    metric = model.LifecycleSummaryMetric.model_validate(
        {
            **base,
            "holdings_state": "Orders",
            "monetary_state": "Revenue",
            "first_purchase_state": "FirstPurchase",
            "last_purchase_state": "LastPurchase",
        }
    )
    assert metric.last_purchase_state == "LastPurchase"


@pytest.mark.unit
def test_recipe_proposed_states_validate_through_discriminated_union() -> None:
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "events",
            "kind": "binary_outcome",
            "group_by": ["Channel"],
            "time": {"property": "OutcomeDate", "grain": "daily"},
            "states": {"Count": {"type": "count"}},
            "outcome": {
                "column": "Outcome",
                "positive_values": [True],
                "negative_values": [False],
            },
        }
    )
    recipe = next(
        item
        for item in load_builtin_kpi_recipes().recipes
        if item.id == "audience.unique_entities"
    )
    proposed = next(
        option
        for option in recipe_binding_options(
            recipe.inputs[0],
            processor,
            proposal_fields=["Channel"],
        )
        if option.field == "Channel" and option.state_type == "cpc"
    )

    configured = processor_with_recipe_states(
        processor,
        {proposed.value: proposed.state_definition},
    )
    metric = instantiate_metric(
        recipe,
        configured,
        "Unique_Channels",
        {recipe.inputs[0].role: proposed.value},
    )

    assert isinstance(configured.states[proposed.value], model.CpcState)
    assert metric["display"]["label"] == "Unique Channels"


@pytest.mark.unit
def test_metric_output_contract_is_read_only_and_derived_from_metric_kind() -> None:
    outputs = builder.metric_outputs_from_definition(
        "ExperimentProportionTest",
        {
            "processor": "experiment",
            "kind": "proportion_test",
            "variant_column": "ExperimentGroup",
            "test_role": "Test",
            "control_role": "Control",
        },
    )

    assert outputs == ["Count", "Positives", "Negatives", "z_score", "z_p_val"]


@pytest.mark.unit
def test_metric_owned_chart_controls_select_only_the_metric_output() -> None:
    metric = model.ProportionTestMetric.model_validate(
        {
            "processor": "experiment",
            "kind": "proportion_test",
            "variant_column": "ExperimentGroup",
        }
    )
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "catalog_version": 2,
                "workspace": "outputs",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
                    }
                ],
            },
            "processors": {
                "catalog_version": 2,
                "processors": [
                    {
                        "id": "experiment",
                        "source": "events",
                        "kind": "binary_outcome",
                        "group_by": ["Day", "Channel"],
                        "time": {"property": "OutcomeDate", "grain": "daily"},
                        "states": {
                            "Count": {"type": "count"},
                            "Positives": {"type": "count"},
                            "Negatives": {"type": "count"},
                        },
                        "outcome": {
                            "column": "Outcome",
                            "positive_values": [True],
                            "negative_values": [False],
                        },
                        "variant_column": "ExperimentGroup",
                    }
                ],
            },
            "metrics": {
                "catalog_version": 2,
                "metrics": {"ExperimentProportionTest": metric},
            },
            "dashboards": {"catalog_version": 2, "dashboards": []},
        }
    )

    assert builder.chart_field_controls("bar")[0] == "metric_output"
    assert builder.metric_output_options(catalog, "ExperimentProportionTest") == [
        "Count",
        "Positives",
        "Negatives",
        "z_score",
        "z_p_val",
    ]
    assert builder.default_tile_fields(
        catalog,
        "ExperimentProportionTest",
        "bar",
    )["metric_output"] == "Count"


@pytest.mark.unit
def test_processor_materializes_only_its_base_grain() -> None:
    processor = model.BinaryOutcomeProcessor.model_validate(
        {
            "id": "engagement",
            "source": "events",
            "kind": "binary_outcome",
            "group_by": ["Day", "Month", "Quarter", "Year", "Channel"],
            "time": {"property": "OutcomeDate", "grain": "daily"},
            "states": {"Count": {"type": "count"}},
            "outcome": {
                "column": "Outcome",
                "positive_values": [True],
                "negative_values": [False],
            },
        }
    )

    assert processor.grains == ["daily"]
    assert aggregate_grain_candidates(processor, "daily") == ["daily"]
    assert aggregate_grain_candidates(processor, "monthly") == ["daily"]
    assert aggregate_grain_candidates(processor, "quarterly") == ["daily"]
    assert aggregate_grain_candidates(processor, "summary") == ["daily"]
