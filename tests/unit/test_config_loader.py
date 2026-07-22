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
            "tdigest_quantile",
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
        assert {source.id for source in catalog.pipelines.sources} == {"ih", "holdings"}

    def test_covers_viable_legacy_processor_and_metric_kinds(self) -> None:
        catalog = load(FAT_WS)

        assert {processor.kind for processor in catalog.processors.processors} == {
            "binary_outcome",
            "numeric_distribution",
            "score_distribution",
            "entity_lifecycle",
            "entity_set",
            "funnel",
        }
        assert {metric.kind for metric in catalog.metrics.metrics.values()} == {
            "formula",
            "approx_distinct_count",
            "topk_items",
            "tdigest_quantile",
            "variant_compare",
            "curve_from_digests",
            "calibration_from_digests",
            "contingency_test",
            "proportion_test",
            "lifecycle_summary",
            "set_op",
            "funnel_dropoff",
        }

    def test_business_report_pages_are_present(self) -> None:
        catalog = load(FAT_WS)

        pages_by_dashboard = {
            dashboard.id: [page.id for page in dashboard.pages]
            for dashboard in catalog.dashboards.dashboards
        }

        assert pages_by_dashboard == {
            "fat_business_value": [
                "executive_overview",
                "engagement",
                "reach_and_frequency",
                "conversion_and_revenue",
                "outcome_funnel",
            ],
            "fat_model_quality": ["model_quality", "distributions"],
            "fat_experiments": ["experiments", "distributions"],
            "fat_clv": ["customer_lifecycle"],
        }

    def test_enables_bulk_sketch_build_only_for_quantile_processors(self) -> None:
        catalog = load(FAT_WS)

        quantile_processors = {
            processor.id: processor.sketch_build_mode
            for processor in catalog.processors.processors
            if isinstance(
                processor,
                model.NumericDistributionProcessor | model.ScoreDistributionProcessor,
            )
        }

        assert quantile_processors == {
            "descriptive": "bulk",
            "model_ml_scores": "bulk",
        }


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
        (catalog_dir / "pipelines.yaml").write_text("version: 1\nworkspace: foo\nsources: []\n")
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
                    "version": 1,
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
        (catalog_dir / "metrics.yaml").write_text("metrics: {}\n")
        (catalog_dir / "dashboards.yaml").write_text("dashboards: []\n")
        with pytest.raises(CatalogLoadError):
            load(tmp_path)

    def test_legacy_processor_grouping_fields_are_rejected(self) -> None:
        base = {"id": "p", "source": "s", "kind": "binary_outcome"}
        for field in ("grains", "extra_dimensions"):
            with pytest.raises(ValueError, match="legacy processor field"):
                model.BinaryOutcomeProcessor.model_validate({**base, field: ["Channel"]})

    def test_dimensions_alias_maps_to_group_by(self) -> None:
        processor = model.BinaryOutcomeProcessor.model_validate(
            {
                "id": "p",
                "source": "s",
                "kind": "binary_outcome",
                "dimensions": ["Channel", "Issue"],
            }
        )

        assert processor.group_by == ["Channel", "Issue"]

    def test_binary_processor_derives_default_states_without_yaml_states(self) -> None:
        processor = model.BinaryOutcomeProcessor.model_validate(
            {
                "id": "p",
                "source": "s",
                "kind": "binary_outcome",
                "entities": {"subject": "CustomerID"},
            }
        )

        assert processor.states == {}
        assert set(model.effective_processor_states(processor)) == {
            "Count",
            "Positives",
            "Negatives",
            "UniqueSubjects_cpc",
        }

    def test_numeric_processor_explicit_states_overlay_defaults(self) -> None:
        processor = model.NumericDistributionProcessor.model_validate(
            {
                "id": "p",
                "source": "s",
                "kind": "numeric_distribution",
                "properties": ["ResponseTime"],
                "states": {"ResponseTime_tdigest": {"type": "tdigest"}},
            }
        )

        states = model.effective_processor_states(processor)

        assert set(states) >= {
            "ResponseTime_Count",
            "ResponseTime_Sum",
            "ResponseTime_Mean",
            "ResponseTime_Var",
            "ResponseTime_Min",
            "ResponseTime_Max",
            "ResponseTime_tdigest",
        }

    @pytest.mark.parametrize(
        ("processor_type", "kind"),
        [
            (model.NumericDistributionProcessor, "numeric_distribution"),
            (model.ScoreDistributionProcessor, "score_distribution"),
        ],
    )
    def test_quantile_processor_sketch_build_mode_is_typed(
        self,
        processor_type: type[model.NumericDistributionProcessor | model.ScoreDistributionProcessor],
        kind: str,
    ) -> None:
        base = {"id": "p", "source": "s", "kind": kind}

        assert processor_type.model_validate(base).sketch_build_mode == "bulk"
        assert (
            processor_type.model_validate({**base, "sketch_build_mode": "bulk"}).sketch_build_mode
            == "bulk"
        )
        assert (
            processor_type.model_validate({**base, "sketch_build_mode": "legacy"}).sketch_build_mode
            == "legacy"
        )
        with pytest.raises(ValueError, match=r"legacy.*bulk"):
            processor_type.model_validate({**base, "sketch_build_mode": "adaptive"})

    def test_semantic_binary_processor_validates_formula_metrics(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "ih",
                            "kind": "binary_outcome",
                            "dimensions": ["Channel"],
                            "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
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
                "dashboards": {"dashboards": []},
            }
        )

        assert validate_catalog(catalog).ok

    def test_semantic_metric_validates_approx_distinct_state(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "ih",
                            "kind": "binary_outcome",
                            "states": {
                                "Count": {"type": "count"},
                                "UniqueCustomers_hll": {"type": "hll"},
                            },
                        }
                    ]
                },
                "metrics": {
                    "metrics": {
                        "UniqueCustomers": {
                            "source": "engagement",
                            "kind": "approx_distinct_count",
                            "state": "Missing_hll",
                        }
                    }
                },
                "dashboards": {"dashboards": []},
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
                    "processors": [
                        {
                            "id": "audience",
                            "source": "ih",
                            "kind": "entity_set",
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
                    "metrics": {
                        "UniqueCustomers": {
                            "source": "audience",
                            "kind": "approx_distinct_count",
                            "state": "Audience_theta",
                        }
                    }
                },
                "dashboards": {"dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert result.ok, result.issues

    def test_semantic_metric_validates_quantile_state_type(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                    "processors": [
                        {
                            "id": "response_time",
                            "source": "ih",
                            "kind": "numeric_distribution",
                            "properties": ["ResponseTime"],
                        }
                    ]
                },
                "metrics": {
                    "metrics": {
                        "ResponseTimeP95": {
                            "source": "response_time",
                            "kind": "tdigest_quantile",
                            "state": "ResponseTime_Count",
                            "quantile": 0.95,
                        }
                    }
                },
                "dashboards": {"dashboards": []},
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

    def test_semantic_funnel_requires_stage_conditions(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                    "processors": [
                        {
                            "id": "outcome_funnel",
                            "source": "ih",
                            "kind": "funnel",
                            "stages": [{"name": "Impression"}],
                        }
                    ]
                },
                "metrics": {"metrics": {}},
                "dashboards": {"dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "processors[outcome_funnel].stages[0].when"
            and issue.message == "field required"
            for issue in result.issues
        )

    def test_semantic_funnel_dropoff_validates_stage_names(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                    "processors": [
                        {
                            "id": "outcome_funnel",
                            "source": "ih",
                            "kind": "funnel",
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
                    "metrics": {
                        "Dropoff": {
                            "source": "outcome_funnel",
                            "kind": "funnel_dropoff",
                            "from_stage": "Impression",
                            "to_stage": "Conversion",
                        }
                    }
                },
                "dashboards": {"dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "metrics.Dropoff.to_stage"
            and "unknown funnel stage 'Conversion'" in issue.message
            for issue in result.issues
        )

    def test_validator_applies_rename_capitalize_to_source_schema(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                "processors": {"processors": []},
                "metrics": {"metrics": {}},
                "dashboards": {"dashboards": []},
            }
        )

        assert validate_catalog(catalog).ok

    def test_source_filter_accepts_field_declared_by_bound_processor(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
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
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "ih",
                            "kind": "binary_outcome",
                            "time": {"column": "OutcomeTime"},
                            "outcome": {
                                "column": "Outcome",
                                "positive_values": ["Clicked"],
                                "negative_values": ["Impression"],
                            },
                        }
                    ]
                },
                "metrics": {"metrics": {}},
                "dashboards": {"dashboards": []},
            }
        )

        assert validate_catalog(catalog).ok

    def test_score_processor_accepts_scores_alias(self) -> None:
        processor = model.ScoreDistributionProcessor.model_validate(
            {
                "id": "scores",
                "source": "ih",
                "kind": "score_distribution",
                "scores": {"primary": "final_propensity"},
            }
        )

        states = model.effective_processor_states(processor)
        assert processor.model_extra["score_columns"]["primary"] == "final_propensity"
        assert processor.model_extra["score_properties"] == ["final_propensity"]
        assert (
            states["final_propensity_tdigest_positives"].model_extra["source_column"]
            == "final_propensity"
        )

    def test_processor_time_aggregation_levels_are_normalized(self) -> None:
        processor = model.BinaryOutcomeProcessor.model_validate(
            {
                "id": "p",
                "source": "s",
                "kind": "binary_outcome",
                "time": {
                    "grains": ["Day", "Month", "Summary"],
                    "aggregation_levels": {"Summary": "Quarter", "Month": "Day"},
                },
            }
        )

        assert processor.aggregation_level_for("Summary") == "quarterly"
        assert processor.aggregation_level_for("Month") == "daily"
        assert processor.aggregation_level_for("Day") == "daily"

    def test_processor_time_rejects_invalid_aggregation_level(self) -> None:
        with pytest.raises(ValueError, match="not valid for grain"):
            model.BinaryOutcomeProcessor.model_validate(
                {
                    "id": "p",
                    "source": "s",
                    "kind": "binary_outcome",
                    "time": {"aggregation_levels": {"Summary": "Day"}},
                }
            )

    def test_loader_rejects_duplicate_yaml_keys(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        (catalog_dir / "pipelines.yaml").write_text(
            "version: 1\nworkspace: first\nworkspace: second\nsources: []\n",
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
                    "workspace": "duplicates",
                    "sources": [source, source],
                },
                "processors": {"processors": []},
                "metrics": {"metrics": {}},
                "dashboards": {"dashboards": []},
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
                    "workspace": "dependencies",
                    "sources": [
                        {
                            "id": "events",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                        }
                    ],
                },
                "processors": {
                    "processors": [
                        {"id": "first", "source": "events", "kind": "binary_outcome"},
                        {"id": "second", "source": "events", "kind": "binary_outcome"},
                    ]
                },
                "metrics": {
                    "metrics": {
                        "A": {
                            "source": "first",
                            "kind": "formula",
                            "depends_on": ["B"],
                            "expression": {"col": "B"},
                        },
                        "B": {
                            "source": "second",
                            "kind": "formula",
                            "depends_on": ["A"],
                            "expression": {"col": "A"},
                        },
                    }
                },
                "dashboards": {"dashboards": []},
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
                    "workspace": "experiment",
                    "sources": [
                        {
                            "id": "events",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                        }
                    ],
                },
                "processors": {
                    "processors": [
                        {
                            "id": "engagement",
                            "source": "events",
                            "kind": "binary_outcome",
                            "group_by": ["Channel"],
                        }
                    ]
                },
                "metrics": {
                    "metrics": {
                        "Lift": {
                            "source": "engagement",
                            "kind": "variant_compare",
                            "variant_column": "ModelControlGroup",
                            "test_role": "Test",
                            "control_role": "Control",
                        }
                    }
                },
                "dashboards": {"dashboards": []},
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location == "metrics.Lift.variant_column" and "not persisted" in issue.message
            for issue in result.issues
        )

    def test_validator_requires_chart_role_fields(self) -> None:
        catalog = model.Catalog.model_validate(
            {
                "pipelines": {
                    "workspace": "charts",
                    "sources": [
                        {
                            "id": "events",
                            "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                        }
                    ],
                },
                "processors": {
                    "processors": [
                        {"id": "engagement", "source": "events", "kind": "binary_outcome"}
                    ]
                },
                "metrics": {
                    "metrics": {
                        "Count": {
                            "source": "engagement",
                            "kind": "formula",
                            "expression": {"col": "Count"},
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
                                    "id": "main",
                                    "title": "Main",
                                    "tiles": [
                                        {
                                            "id": "broken_line",
                                            "title": "Broken",
                                            "metric": "Count",
                                            "chart": "line",
                                            "x": "Day",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
            }
        )

        result = validate_catalog(catalog)

        assert not result.ok
        assert any(
            issue.location.endswith("tiles[broken_line].y")
            and "chart 'line' requires 'y'" in issue.message
            for issue in result.issues
        )


# ---------------------------------------------------------------------------
# Schema-on-disk parity.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_rejects_variant_column_duplicated_in_group_by() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "duplicate_variant",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "engagement",
                        "source": "events",
                        "kind": "binary_outcome",
                        "group_by": ["Channel", "ModelControlGroup"],
                        "variant_column": "ModelControlGroup",
                    }
                ]
            },
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
        }
    )

    result = validate_catalog(catalog)

    assert not result.ok
    assert any(
        issue.location == "processors[engagement].variant_column"
        and "already present in group_by" in issue.message
        for issue in result.issues
    )


@pytest.mark.unit
def test_validator_evolves_observed_source_columns_through_rename_capitalize() -> None:
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
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
            "processors": {"processors": []},
            "metrics": {"metrics": {}},
            "dashboards": {"dashboards": []},
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
def test_validator_accepts_boxplot_without_y_for_distribution_metrics() -> None:
    """The digest metric implies the boxplot property; y stays required elsewhere."""
    catalog = model.Catalog.model_validate(
        {
            "pipelines": {
                "workspace": "charts",
                "sources": [
                    {
                        "id": "events",
                        "reader": {"kind": "parquet", "file_pattern": "*.parquet"},
                    }
                ],
            },
            "processors": {
                "processors": [
                    {
                        "id": "descriptive",
                        "source": "events",
                        "kind": "numeric_distribution",
                        "properties": ["Propensity"],
                    }
                ]
            },
            "metrics": {
                "metrics": {
                    "PropensityDistribution": {
                        "source": "descriptive",
                        "kind": "tdigest_quantile",
                        "state": "Propensity_tdigest",
                    },
                    "PropensityCount": {
                        "source": "descriptive",
                        "kind": "formula",
                        "expression": {"col": "Propensity_Count"},
                    },
                }
            },
            "dashboards": {
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
                                        "x": "Year",
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
        "scalar_box" in issue.location and "requires 'y' or 'property'" in issue.message
        for issue in result.issues
    )
