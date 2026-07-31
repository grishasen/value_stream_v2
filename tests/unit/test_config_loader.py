"""Loader tests against the demo workspace and synthetic error cases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from valuestream.config import model
from valuestream.config._schema_gen import generate_all
from valuestream.config.loader import CatalogLoadError, load
from valuestream.config.validate import validate_catalog
from valuestream.expr import ast as expr_ast
from valuestream.recipes import load_builtin_kpi_recipes, recipe_readiness

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_WS = REPO_ROOT / "examples" / "demo"
FAT_WS = REPO_ROOT / "examples" / "fat"


@pytest.mark.unit
class TestDemoWorkspace:
    def test_loads(self) -> None:
        catalog = load(DEMO_WS)
        assert isinstance(catalog, model.Catalog)

    def test_sources(self) -> None:
        catalog = load(DEMO_WS)
        ids = [s.id for s in catalog.pipelines.sources]
        assert ids == ["ih"]

    def test_processor_kinds(self) -> None:
        catalog = load(DEMO_WS)
        kinds = {p.kind for p in catalog.processors.processors}
        # The demo exercises the four interaction-history processor kinds; all
        # must be members of the supported processor-kind set.
        assert kinds == {
            "binary_outcome",
            "numeric_distribution",
            "score_distribution",
            "funnel",
        }

    def test_metric_kinds(self) -> None:
        catalog = load(DEMO_WS)
        kinds = {m.kind for m in catalog.metrics.metrics.values()}
        expected = {
            "formula",
            "approx_distinct_count",
            "distribution",
            "quantile",
            "variant_compare",
            "curve_from_digests",
            "calibration_from_digests",
            "contingency_test",
            "funnel_dropoff",
        }
        assert expected <= kinds

    def test_engagement_processor_states(self) -> None:
        catalog = load(DEMO_WS)
        eng = next(p for p in catalog.processors.processors if p.id == "ih_engagement")
        assert set(eng.states.keys()) == {
            "Count",
            "Positives",
            "Negatives",
            "UniqueCustomers_cpc",
            "UniqueInteractions_cpc",
        }
        assert eng.states["Count"].type == "count"
        assert eng.states["UniqueCustomers_cpc"].type == "cpc"

    def test_filter_transform_carries_typed_expression(self) -> None:
        catalog = load(DEMO_WS)
        ih = next(s for s in catalog.pipelines.sources if s.id == "ih")
        filters = [t for t in ih.transforms if isinstance(t, model.FilterTransform)]
        assert len(filters) == 1
        # The expression should be a parsed AST node, not a raw dict.
        assert isinstance(filters[0].expression, expr_ast.NullCheck)

    def test_dashboard_tile_metric_references_resolve(self) -> None:
        catalog = load(DEMO_WS)
        defined = set(catalog.metrics.metrics.keys())
        for dash in catalog.dashboards.dashboards:
            for page in dash.pages:
                for tile in page.tiles:
                    assert tile.metric in defined, (
                        f"unknown metric on tile {tile.id}: {tile.metric}"
                    )


@pytest.mark.unit
class TestFatWorkspace:
    def test_loads_and_validates(self) -> None:
        catalog = load(FAT_WS)

        result = validate_catalog(catalog)

        assert result.ok, [f"{issue.location}: {issue.message}" for issue in result.issues]
        assert {source.id for source in catalog.pipelines.sources} == {"ih"}

    def test_covers_supported_processor_and_metric_kinds(self) -> None:
        catalog = load(FAT_WS)

        assert {processor.kind for processor in catalog.processors.processors} == {
            "binary_outcome",
            "frequency_response",
            "numeric_distribution",
            "score_distribution",
            "entity_set",
            "funnel",
        }
        assert {metric.kind for metric in catalog.metrics.metrics.values()} == {
            "formula",
            "approx_distinct_count",
            "topk_items",
            "distribution",
            "quantile",
            "variant_compare",
            "curve_from_digests",
            "calibration_from_digests",
            "contingency_test",
            "proportion_test",
            "set_op",
            "funnel_dropoff",
        }

    def test_filtered_state_kpis_are_wired_to_their_backing_states(self) -> None:
        catalog = load(FAT_WS)
        processors = {processor.id: processor for processor in catalog.processors.processors}
        metrics = catalog.metrics.metrics

        # Coverage and repeat-clicker KPIs are only meaningful if their states
        # actually carry the outcome filter that narrows them.
        clicked_actions = processors["engagement"].states["ClickedActions_cpc"]
        clicked_customers = processors["audience"].states["ClickedCustomers_theta"]
        assert clicked_actions.where is not None
        assert clicked_customers.where is not None
        assert metrics["ActionsClicked"].state == "ClickedActions_cpc"
        assert metrics["ActionsDelivered"].state == "DeliveredActions_cpc"
        assert {operand.state for operand in metrics["RepeatClickers30d"].operands} == {
            "ClickedCustomers_theta"
        }

        # Conversion latency is correct without a tile filter only because the
        # processor itself is restricted to converted decisions.
        latency = processors["conversion_latency"]
        assert latency.filter is not None
        assert metrics["ConversionLatencyMedian"].processor == "conversion_latency"

    def test_upward_exploration_recipe_targets_the_dedicated_processor(self) -> None:
        catalog = load(FAT_WS)
        processors = {processor.id: processor for processor in catalog.processors.processors}
        recipe = next(
            recipe
            for recipe in load_builtin_kpi_recipes().recipes
            if recipe.id == "decisioning.material_upward_exploration_rate"
        )

        assert recipe_readiness(recipe, processors["engagement"]).status == "incompatible"

        readiness = recipe_readiness(recipe, processors["exploration"])
        assert readiness.status == "ready"
        assert readiness.resolved_inputs == {
            "explored": "MaterialExploredUp_Count",
            "observations": "Explore_Count",
        }
        material = processors["exploration"].states["MaterialExploredUp_Count"]
        assert material.where is not None
        assert material.where.model_dump(mode="json", by_alias=True)["args"][1]["args"][1][
            "args"
        ][1] == {"lit": 0.1}

    def test_business_report_pages_are_present(self) -> None:
        catalog = load(FAT_WS)

        pages_by_dashboard = {
            dashboard.id: [page.id for page in dashboard.pages]
            for dashboard in catalog.dashboards.dashboards
        }

        assert pages_by_dashboard == {
            "fat_engagement": [
                "engagement_overview",
                "frequency_response",
                "engagement_breakdowns",
                "engagement_lift",
                "engagement_actions",
                "reach_and_frequency",
                "response_time_health",
            ],
            "fat_machine_learning": [
                "model_quality",
                "model_comparison",
                "recommendation_quality",
                "exploration_and_evidence",
                "distributions",
            ],
            "fat_experiments": [
                "experiment_readout",
                "experiment_significance",
            ],
            "fat_conversions": [
                "conversion_overview",
                "products_and_revenue_mix",
                "unit_economics",
                "outcome_funnel",
            ],
        }

    def test_adaptive_diagnostics_are_fixed_to_valid_populations(self) -> None:
        catalog = load(FAT_WS)
        pages = {
            page.id: page
            for dashboard in catalog.dashboards.dashboards
            for page in dashboard.pages
        }
        adaptive = pages["exploration_and_evidence"]
        tiles = {tile.id: tile for tile in adaptive.tiles}

        test_tiles = {
            "exploration_rate_kpi",
            "model_maturity_kpi",
            "exploration_delta_median_kpi",
            "exploration_delta_p95_kpi",
            "exploration_samples_kpi",
            "band_calibration_final",
            "remaining_uncertainty_kpi",
            "convergence_curve_uncertainty",
            "convergence_curve_maturity",
            "convergence_curve_exploration_rate",
            "convergence_curve_volume",
            "exploration_rate_trend",
            "model_maturity_trend",
        }
        assert all(
            tiles[tile_id].filters == {"ModelControlGroup": ["Test"]}
            for tile_id in test_tiles
        )
        assert tiles["band_calibration_model"].filters == {
            "ModelControlGroup": ["Control"]
        }
        assert {filter_.field for filter_ in adaptive.filters} == {
            "Channel",
            "Issue",
            "Treatment",
            "CustomerType",
        }

    def test_exploration_uses_treatment_level_grain_and_age(self) -> None:
        catalog = load(FAT_WS)
        source = catalog.pipelines.sources[0]
        processors = {processor.id: processor for processor in catalog.processors.processors}

        treatment_age = next(
            transform
            for transform in source.transforms
            if isinstance(transform, model.DeriveColumn)
            and transform.output == "TreatmentAgeDays"
        )
        assert isinstance(treatment_age.expression, expr_ast.OpCast)
        assert not any(
            isinstance(transform, model.DeriveColumn)
            and transform.output == "ActionAgeDays"
            for transform in source.transforms
        )

        treatment_hierarchy = {"Issue", "Group", "Name", "Treatment"}
        assert treatment_hierarchy <= set(processors["exploration"].group_by)
        assert treatment_hierarchy | {"TreatmentAgeDays"} <= set(
            processors["exploration_by_treatment_age"].group_by
        )

        treatment_age_metrics = {
            "RelativeExplorationVarianceByTreatmentAge",
            "ImpliedEvidenceIndexByTreatmentAge",
            "MaterialUpwardExplorationRateByTreatmentAge",
            "ExplorationSamplesByTreatmentAge",
        }
        assert all(
            catalog.metrics.metrics[name].processor == "exploration_by_treatment_age"
            for name in treatment_age_metrics
        )
        adaptive = next(
            page
            for dashboard in catalog.dashboards.dashboards
            for page in dashboard.pages
            if page.id == "exploration_and_evidence"
        )
        convergence_tiles = [
            tile for tile in adaptive.tiles if tile.id.startswith("convergence_curve_")
        ]
        assert {tile.x for tile in convergence_tiles} == {"TreatmentAgeDays"}

    def test_corrected_business_names_and_action_metric_are_used(self) -> None:
        catalog = load(FAT_WS)
        pages = {
            page.id: page
            for dashboard in catalog.dashboards.dashboards
            for page in dashboard.pages
        }

        actions = pages["engagement_actions"]
        action_tiles = {tile.id: tile for tile in actions.tiles}
        assert action_tiles["actions_delivered"].metric == "ActionsDelivered"
        assert "engaged_action_coverage" not in action_tiles
        assert action_tiles["action_coverage_ratio"].title == "Clicked-action coverage"

        assert pages["response_time_health"].title == "Decision-to-outcome latency"
        adaptive_metrics = {tile.metric for tile in pages["exploration_and_evidence"].tiles}
        assert {
            "MaterialUpwardExplorationRate",
            "ImpliedEvidenceIndex",
            "RelativeExplorationVariance",
        } <= adaptive_metrics


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorPaths:
    def test_missing_workspace(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogLoadError, match="catalog directory"):
            load(tmp_path / "no-such-workspace")

    def test_missing_file(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        # Create only one of the four required files.
        (catalog_dir / "pipelines.yaml").write_text(
            "catalog_version: 2\nworkspace: foo\nsources: []\n"
        )
        with pytest.raises(CatalogLoadError, match="missing catalog file"):
            load(tmp_path)

    def test_yaml_parse_error(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        for name in (
            "pipelines.yaml",
            "processors.yaml",
            "metrics.yaml",
            "dashboards.yaml",
        ):
            (catalog_dir / name).write_text(": :\n: invalid yaml")
        with pytest.raises(CatalogLoadError, match="YAML parse error"):
            load(tmp_path)

    def test_empty_file(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        for name in ("processors.yaml", "metrics.yaml", "dashboards.yaml"):
            (catalog_dir / name).write_text(
                "processors: []\n" if name == "processors.yaml" else "{}"
            )
        (catalog_dir / "pipelines.yaml").write_text("")
        (catalog_dir / "metrics.yaml").write_text("metrics: {}\n")
        (catalog_dir / "dashboards.yaml").write_text("dashboards: []\n")
        with pytest.raises(CatalogLoadError, match="empty catalog file"):
            load(tmp_path)

    def test_top_level_not_mapping(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "pipelines.yaml").write_text("- a\n- b\n")
        for name in ("processors.yaml", "metrics.yaml", "dashboards.yaml"):
            (catalog_dir / name).write_text("{}")
        with pytest.raises(CatalogLoadError, match="must be a mapping"):
            load(tmp_path)

    def test_bad_processor_kind(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "pipelines.yaml").write_text(
            yaml.safe_dump(
                {
                    "catalog_version": 2,
                    "workspace": "x",
                    "sources": [
                        {
                            "id": "s",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                        }
                    ],
                }
            )
        )
        (catalog_dir / "processors.yaml").write_text(
            yaml.safe_dump(
                {
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "p",
                            "source": "s",
                            "kind": "wibble",
                            "states": {},
                        }
                    ]
                }
            )
        )
        (catalog_dir / "metrics.yaml").write_text("catalog_version: 2\nmetrics: {}\n")
        (catalog_dir / "dashboards.yaml").write_text(
            "catalog_version: 2\ndashboards: []\n"
        )
        with pytest.raises(CatalogLoadError):
            load(tmp_path)

    def test_legacy_processor_grouping_fields_are_rejected(self) -> None:
        base = {
            "id": "p",
            "source": "s",
            "kind": "binary_outcome",
            "time": {"property": "OutcomeTime", "grain": "daily"},
            "states": {"Count": {"type": "count"}},
            "outcome": {
                "column": "Outcome",
                "positive_values": ["Clicked"],
                "negative_values": ["Impression"],
            },
        }
        for field in ("grains", "extra_dimensions"):
            with pytest.raises(ValueError, match="Extra inputs are not permitted"):
                model.BinaryOutcomeProcessor.model_validate({**base, field: ["Channel"]})

    def test_semantic_binary_processor_validates_formula_metrics(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "catalog_version": 2,
                    "workspace": "semantic",
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
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "ih",
                            "kind": "binary_outcome",
                            "group_by": ["Channel"],
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {
                                "Count": {"type": "count"},
                                "Positives": {"type": "count", "outcome": "positive"},
                                "Negatives": {"type": "count", "outcome": "negative"},
                                "UniqueSubjects_cpc": {
                                    "type": "cpc",
                                    "source_column": "CustomerID",
                                },
                            },
                            "entities": {"subject": "CustomerID"},
                            "outcome": {
                                "column": "Outcome",
                                "positive_values": [1],
                                "negative_values": [0],
                            },
                        }
                    ]
                },
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "CTR": {
                            "processor": "engagement",
                            "kind": "formula",
                            "expression": {
                                "op": "safe_div",
                                "num": {"col": "Positives"},
                                "den": {"col": "Count"},
                            },
                        }
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        assert validate_catalog(catalog).ok

    def test_semantic_metric_validates_approx_distinct_state(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "semantic",
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
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "ih",
                            "kind": "binary_outcome",
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {
                                "Count": {"type": "count"},
                                "UniqueCustomers_hll": {
                                    "type": "hll",
                                    "source_column": "CustomerID",
                                },
                            },
                            "outcome": {
                                "column": "Outcome",
                                "positive_values": ["Clicked"],
                                "negative_values": ["Impression"],
                            },
                        }
                    ]
                },
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "UniqueCustomers": {
                            "processor": "engagement",
                            "kind": "approx_distinct_count",
                            "state": "Missing_hll",
                        }
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "metrics.UniqueCustomers.state"
            and "unknown state 'Missing_hll'" in issue.message
            for issue in result.issues
        )

    def test_semantic_metric_accepts_theta_for_approx_distinct(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "semantic",
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
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "audience",
                            "source": "ih",
                            "kind": "entity_set",
                            "entity": "CustomerID",
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {
                                "Audience_theta": {
                                    "type": "theta",
                                    "source_column": "CustomerID",
                                }
                            },
                        }
                    ]
                },
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "UniqueCustomers": {
                            "processor": "audience",
                            "kind": "approx_distinct_count",
                            "state": "Audience_theta",
                        }
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert result.ok, result.issues

    def test_semantic_metric_validates_quantile_state_type(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "semantic",
                    "sources": [
                        {
                            "id": "ih",
                            "reader": {"kind": "csv", "file_pattern": "*.csv"},
                            "schema": {"timestamp_column": "OutcomeTime"},
                        }
                    ],
                },
                "processors": {
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "response_time",
                            "source": "ih",
                            "kind": "numeric_distribution",
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "properties": ["ResponseTime"],
                            "states": {
                                "ResponseTime_Count": {
                                    "type": "count",
                                    "source_column": "ResponseTime",
                                }
                            },
                        }
                    ]
                },
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "ResponseTimeP95": {
                            "processor": "response_time",
                            "kind": "quantile",
                            "state": "ResponseTime_Count",
                            "quantile": 0.95,
                        }
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "metrics.ResponseTimeP95.state"
            and "must have type" in issue.message
            and "got 'count'" in issue.message
            for issue in result.issues
        )

    def test_semantic_funnel_dropoff_validates_stage_names(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "semantic",
                    "sources": [
                        {
                            "id": "ih",
                            "reader": {"kind": "csv", "file_pattern": "*.csv"},
                            "schema": {"timestamp_column": "OutcomeTime"},
                            "transforms": [
                                {"kind": "defaults", "values": {"Outcome": "Impression"}}
                            ],
                        }
                    ],
                },
                "processors": {
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "outcome_funnel",
                            "source": "ih",
                            "kind": "funnel",
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {
                                "Impression": {"type": "count", "stage": "Impression"},
                                "Clicked": {"type": "count", "stage": "Clicked"},
                            },
                            "stages": [
                                {
                                    "name": "Impression",
                                    "when": {
                                        "op": "eq",
                                        "column": "Outcome",
                                        "value": "Impression",
                                    },
                                },
                                {
                                    "name": "Clicked",
                                    "when": {
                                        "op": "eq",
                                        "column": "Outcome",
                                        "value": "Clicked",
                                    },
                                },
                            ],
                        }
                    ]
                },
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "Dropoff": {
                            "processor": "outcome_funnel",
                            "kind": "funnel_dropoff",
                            "from_state": "Impression",
                            "to_state": "Conversion",
                        }
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "metrics.Dropoff.to_state"
            and "unknown funnel stage-count state 'Conversion'" in issue.message
            for issue in result.issues
        )

    def test_validator_applies_rename_capitalize_to_source_schema(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "renamed",
                    "sources": [
                        {
                            "id": "ih",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                            "schema": {"timestamp_column": "pxDecisionTime"},
                            "transforms": [
                                {"kind": "rename_capitalize"},
                                {
                                    "kind": "filter",
                                    "expression": {"op": "not_null", "column": "DecisionTime"},
                                },
                            ],
                        }
                    ],
                },
                "processors": {"catalog_version": 2, "processors": []},
                "metrics": {"catalog_version": 2, "metrics": {}},
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        assert validate_catalog(catalog).ok

    def test_source_filter_accepts_field_declared_by_bound_processor(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "source_filter",
                    "sources": [
                        {
                            "id": "ih",
                            "reader": {"kind": "csv", "file_pattern": "*.csv"},
                            "schema": {"timestamp_column": "OutcomeTime"},
                            "transforms": [
                                {
                                    "kind": "filter",
                                    "expression": {
                                        "op": "eq",
                                        "column": "Outcome",
                                        "value": "Clicked",
                                    },
                                }
                            ],
                        }
                    ],
                },
                "processors": {
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "ih",
                            "kind": "binary_outcome",
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
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        assert validate_catalog(catalog).ok

    def test_loader_rejects_duplicate_yaml_keys(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "pipelines.yaml").write_text(
            "catalog_version: 2\nworkspace: first\nworkspace: second\nsources: []\n",
            encoding="utf-8",
        )

        with pytest.raises(CatalogLoadError, match="duplicate key 'workspace'"):
            load(tmp_path)

    def test_validator_rejects_duplicate_catalog_ids(self) -> None:
        source = {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
        }
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "duplicates",
                    "sources": [source, source],
                },
                "processors": {"catalog_version": 2, "processors": []},
                "metrics": {"catalog_version": 2, "metrics": {}},
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "pipelines.sources" and "duplicate id 'events'" in issue.message
            for issue in result.issues
        )

    def test_validator_rejects_metric_cycles_and_cross_processor_dependencies(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "dependencies",
                    "sources": [
                        {
                            "id": "events",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                        }
                    ],
                },
                "processors": {
                    "catalog_version": 2,
                    "processors": [
                        {
                            "id": "first",
                            "source": "events",
                            "kind": "binary_outcome",
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {"Count": {"type": "count"}},
                            "outcome": {
                                "column": "Outcome",
                                "positive_values": ["Clicked"],
                                "negative_values": ["Impression"],
                            },
                        },
                        {
                            "id": "second",
                            "source": "events",
                            "kind": "binary_outcome",
                            "time": {"property": "OutcomeTime", "grain": "daily"},
                            "states": {"Count": {"type": "count"}},
                            "outcome": {
                                "column": "Outcome",
                                "positive_values": ["Clicked"],
                                "negative_values": ["Impression"],
                            },
                        },
                    ]
                },
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "A": {
                            "processor": "first",
                            "kind": "formula",
                            "depends_on": ["B"],
                            "expression": {"col": "B"},
                        },
                        "B": {
                            "processor": "second",
                            "kind": "formula",
                            "depends_on": ["A"],
                            "expression": {"col": "A"},
                        },
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any("dependency cycle" in issue.message for issue in result.issues)
        assert any(
            "uses processor 'second'; expected 'first'" in issue.message for issue in result.issues
        )

    def test_validator_requires_experiment_dimension_in_processor_output(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "catalog_version": 2,
                    "workspace": "experiment",
                    "sources": [
                        {
                            "id": "events",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
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
                "metrics": {
                    "catalog_version": 2,
                    "metrics": {
                        "Lift": {
                            "processor": "engagement",
                            "kind": "variant_compare",
                            "variant_column": "ModelControlGroup",
                            "test_role": "Test",
                            "control_role": "Control",
                        }
                    }
                },
                "dashboards": {"catalog_version": 2, "dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "metrics.Lift.variant_column" and "not persisted" in issue.message
            for issue in result.issues
        )

# ---------------------------------------------------------------------------
# Schema-on-disk parity.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_accepts_explicit_variant_column_in_group_by() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "catalog_version": 2,
                "catalog_version": 2,
                "workspace": "duplicate_variant",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
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
                        "group_by": ["Channel", "ModelControlGroup"],
                        "time": {"property": "OutcomeTime", "grain": "daily"},
                        "states": {"Count": {"type": "count"}},
                        "outcome": {
                            "column": "Outcome",
                            "positive_values": ["Clicked"],
                            "negative_values": ["Impression"],
                        },
                        "variant_column": "ModelControlGroup",
                    }
                ]
            },
            "metrics": {"catalog_version": 2, "metrics": {}},
            "dashboards": {"catalog_version": 2, "dashboards": []},
        }
    )

    result = validate_catalog(catalog)

    assert result.ok, result.issues


@pytest.mark.unit
def test_validator_evolves_observed_source_columns_through_rename_capitalize() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "catalog_version": 2,
                "workspace": "sample_backed",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "**/*.parquet"},
                        "transforms": [
                            {"kind": "rename_capitalize"},
                            {
                                "kind": "filter",
                                "expression": {
                                    "op": "eq",
                                    "column": "Channel",
                                    "value": "Web",
                                },
                            },
                        ],
                    }
                ],
            },
            "processors": {"catalog_version": 2, "processors": []},
            "metrics": {"catalog_version": 2, "metrics": {}},
            "dashboards": {"catalog_version": 2, "dashboards": []},
        }
    )

    assert not validate_catalog(catalog).ok
    result = validate_catalog(catalog, source_columns_by_id={"events": ["pyChannel"]})

    assert result.ok
    assert result.issues == []


@pytest.mark.unit
class TestSchemaParity:
    def test_disk_matches_models(self) -> None:
        on_disk = {
            name: json.loads((REPO_ROOT / "schemas" / name).read_text())
            for name in (
                "pipelines.json",
                "processors.json",
                "metrics.json",
                "dashboards.json",
                "catalog.json",
            )
        }
        from_models = generate_all()
        assert on_disk == from_models, (
            "schemas/*.json out of sync with valuestream.config.model — "
            "regenerate with: uv run python -m valuestream.config._schema_gen"
        )


@pytest.mark.unit
def test_validator_accepts_boxplot_without_axes_for_distribution_metrics() -> None:
    """A digest metric can render one overall box; scalar metrics are incompatible."""
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "catalog_version": 2,
                "workspace": "charts",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "catalog_version": 2,
                "processors": [
                    {
                        "id": "descriptive",
                        "source": "events",
                        "kind": "numeric_distribution",
                        "group_by": ["Year"],
                        "time": {"property": "OutcomeTime", "grain": "daily"},
                        "properties": ["Propensity"],
                        "states": {
                            "Propensity_Count": {
                                "type": "count",
                                "source_column": "Propensity",
                            },
                            "Propensity_tdigest": {
                                "type": "tdigest",
                                "source_column": "Propensity",
                            },
                        },
                    }
                ]
            },
            "metrics": {
                "catalog_version": 2,
                "metrics": {
                    "PropensityDistribution": {
                        "processor": "descriptive",
                        "kind": "distribution",
                        "state": "Propensity_tdigest",
                    },
                    "PropensityCount": {
                        "processor": "descriptive",
                        "kind": "formula",
                        "expression": {"col": "Propensity_Count"},
                    },
                }
            },
            "dashboards": {
                "catalog_version": 2,
                "dashboards": [
                    {
                        "id": "overview",
                        "title": "Overview",
                        "pages": [
                            {
                                "id": "main",
                                "title": "Main",
                                "tiles": [
                                    {
                                        "id": "distribution_box",
                                        "title": "Distribution",
                                        "metric": "PropensityDistribution",
                                        "chart": "boxplot",
                                    },
                                    {
                                        "id": "scalar_box",
                                        "title": "Scalar box",
                                        "metric": "PropensityCount",
                                        "chart": "boxplot",
                                        "x": "Year",
                                    },
                                ],
                            }
                        ],
                    }
                ]
            },
        }
    )

    result = validate_catalog(catalog)

    assert not any("distribution_box" in issue.location for issue in result.issues)
    assert any(
        "scalar_box" in issue.location and "requires metric kind 'distribution'" in issue.message
        for issue in result.issues
    )
