"""Legacy TOML to Value Stream catalog migration."""

from __future__ import annotations

import datetime as dt
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from valuestream.config.loader import load
from valuestream.config.validate import validate_catalog
from valuestream.utils.timer import timed

_TIME_GROUPS = {
    "Day",
    "day",
    "Week",
    "week",
    "Month",
    "month",
    "Quarter",
    "quarter",
    "Year",
    "year",
}
_PROCESSOR_KIND_BY_FAMILY = {
    "engagement": "binary_outcome",
    "conversion": "binary_outcome",
    "experiment": "binary_outcome",
    "descriptive": "numeric_distribution",
    "model_ml_scores": "score_distribution",
    "clv": "entity_lifecycle",
    "cohort": "entity_set",
    "funnel": "funnel",
    "snapshot": "snapshot",
}
_LEGACY_REPORT_FIELD_MAP = {
    "animation_frame": "animation_frame",
    "animation_group": "animation_group",
    "color": "color",
    "description": "title",
    "facet_row": "facet_row",
    "group_by": "group_by",
    "log_x": "log_x",
    "log_y": "log_y",
    "property": "property",
    "r": "r",
    "score": "score",
    "showlegend": "showlegend",
    "size": "size",
    "stages": "stages",
    "theta": "theta",
    "title": "title",
    "value": "value",
    "x": "x",
    "y": "y",
}
_STATE_SCORE_NAMES = {"Count", "Positives", "Negatives", "Revenue"}


@dataclass
class MigrationReport:
    """Structured migration report."""

    source: Path
    target: Path
    mappings: list[dict[str, str]] = field(default_factory=list)
    gaps: list[dict[str, str]] = field(default_factory=list)
    generated_files: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.gaps

    def mapped(self, legacy: str, target: str, note: str = "") -> None:
        self.mappings.append({"legacy": legacy, "target": target, "note": note})

    def gap(self, legacy: str, reason: str, target: str = "manual migration") -> None:
        item = {"legacy": legacy, "target": target, "reason": reason}
        if item not in self.gaps:
            self.gaps.append(item)

    def markdown(self) -> str:
        """Render the report as Markdown."""
        lines = [
            "# Migration Report",
            "",
            f"- Source: `{self.source}`",
            f"- Target catalog: `{self.target}`",
            f"- Generated at: `{dt.datetime.now(dt.UTC).isoformat()}`",
            f"- Status: `{'ok' if self.ok else 'needs review'}`",
            "",
            "## Generated Files",
            "",
        ]
        lines.extend(f"- `{path}`" for path in self.generated_files)
        lines.extend(["", "## Field Mapping", ""])
        if self.mappings:
            lines.extend(["| Legacy field | Target | Note |", "|---|---|---|"])
            lines.extend(
                f"| `{item['legacy']}` | `{item['target']}` | {item.get('note', '')} |"
                for item in self.mappings
            )
        else:
            lines.append("_No fields mapped._")
        lines.extend(["", "## Gaps", ""])
        if self.gaps:
            lines.extend(["| Legacy field | Target | Reason |", "|---|---|---|"])
            lines.extend(
                f"| `{item['legacy']}` | `{item['target']}` | {item['reason']} |"
                for item in self.gaps
            )
        else:
            lines.append("_No gaps detected._")
        return "\n".join(lines) + "\n"


@timed
def migrate_toml(source: str | Path, target_catalog: str | Path) -> MigrationReport:
    """Translate a legacy TOML config into Value Stream catalog YAML."""
    source_path = Path(source)
    target = Path(target_catalog)
    raw = tomllib.loads(source_path.read_text(encoding="utf-8"))
    report = MigrationReport(source=source_path, target=target)

    metrics_block = _dict(raw.get("metrics"))
    source_ids = _infer_sources(metrics_block)
    pipelines = _build_pipelines(raw, source_ids, report)
    processors, metrics = _build_processors_and_metrics(metrics_block, report)
    dashboards = _build_dashboards(raw, metrics, report)
    _record_metrics_container_mappings(metrics_block, report)
    _record_top_level_mappings(raw, report)

    target.mkdir(parents=True, exist_ok=True)
    files = {
        "pipelines.yaml": pipelines,
        "processors.yaml": {"processors": processors},
        "metrics.yaml": {"metrics": metrics},
        "dashboards.yaml": dashboards,
    }
    for name, body in files.items():
        path = target / name
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        report.generated_files.append(path)

    catalog = load(target.parent)
    validation = validate_catalog(catalog)
    if not validation.ok:
        for issue in validation.issues:
            report.gap(issue.location, issue.message, "generated catalog validation")

    report_path = target / "migration_report.md"
    report.generated_files.append(report_path)
    report_path.write_text(report.markdown(), encoding="utf-8")
    return report


def _record_top_level_mappings(raw: dict[str, Any], report: MigrationReport) -> None:
    known = {
        "workspace": "pipelines.workspace",
        "variant": "pipelines.workspace",
        "variants": "pipelines.workspace",
        "sources": "pipelines.sources",
        "ih": "pipelines.sources[ih]",
        "holdings": "pipelines.sources[holdings]",
        "metrics": "processors + metrics",
        "reports": "dashboards",
    }
    for key in sorted(raw):
        target = known.get(key)
        if target is not None:
            report.mapped(key, target)
        else:
            report.gap(key, "unknown legacy top-level field")


def _build_pipelines(
    raw: dict[str, Any],
    source_ids: set[str],
    report: MigrationReport,
) -> dict[str, Any]:
    workspace = _workspace_name(raw)
    sources: list[dict[str, Any]] = []
    for source_id in sorted(source_ids or {"ih"}):
        cfg = _legacy_source_config(raw, source_id)
        if source_id == "holdings":
            timestamp = str(cfg.get("timestamp_column", "PurchasedDateTime"))
            reader_kind = str(cfg.get("reader", cfg.get("file_type", "parquet")))
            file_pattern = str(cfg.get("file_pattern", "holdings/*.parquet"))
            default_values = _legacy_default_values(cfg)
            transforms = [
                {"kind": "rename_capitalize"},
                {
                    "kind": "parse_datetime",
                    "columns": [timestamp],
                    "format": str(cfg.get("datetime_format", "%Y-%m-%d %H:%M:%S")),
                },
            ]
            if default_values:
                transforms.append({"kind": "defaults", "values": default_values})
        else:
            timestamp = str(cfg.get("timestamp_column", "OutcomeTime"))
            reader_kind = str(cfg.get("reader", cfg.get("file_type", "pega_ds_export")))
            file_pattern = str(cfg.get("file_pattern", "**/*.zip"))
            default_values = {
                "ModelControlGroup": "Test",
                "PlacementType": "N/A",
                "ExperimentGroup": "N/A",
                "FinalPropensity": 0.0,
                "Revenue": 0.0,
                **_legacy_default_values(cfg),
            }
            datetime_columns = [timestamp]
            if "DecisionTime" not in datetime_columns:
                datetime_columns.append("DecisionTime")
            transforms = [
                {"kind": "rename_capitalize"},
                {
                    "kind": "parse_datetime",
                    "columns": datetime_columns,
                    "format": str(cfg.get("datetime_format", "%Y%m%dT%H%M%S%.3f %Z")),
                },
                {"kind": "derive_calendar", "from": timestamp, "outputs": ["Day", "Month"]},
                {"kind": "derive_action_id", "parts": ["Issue", "Group", "Name"], "sep": "/"},
                {"kind": "defaults", "values": default_values},
            ]
        group_by_filename = _legacy_group_by_filename(source_id, cfg)
        reader: dict[str, Any] = {
            "kind": reader_kind,
            "file_pattern": file_pattern,
            "group_by_filename": group_by_filename,
        }
        if "streaming" in cfg:
            reader["streaming"] = _boolish(cfg["streaming"])
        if "hive_partitioning" in cfg:
            reader["hive_partitioning"] = _boolish(cfg["hive_partitioning"])
        source = {
            "id": source_id,
            "reader": reader,
            "schema": {
                "timestamp_column": timestamp,
                "natural_key": list(cfg.get("natural_key", ["InteractionID", "ActionID", "Rank"])),
            },
            "transforms": transforms,
        }
        sources.append(source)
        legacy_location = source_id if source_id in raw else f"sources.{source_id}"
        report.mapped(legacy_location, f"pipelines.sources[{source_id}]")
        _record_source_field_mappings(source_id, cfg, report)
    return {"version": 1, "workspace": workspace, "sources": sources}


def _record_source_field_mappings(
    source_id: str,
    cfg: dict[str, Any],
    report: MigrationReport,
) -> None:
    source = f"pipelines.sources[{source_id}]"
    known = {
        "timestamp_column": f"{source}.schema.timestamp_column",
        "reader": f"{source}.reader.kind",
        "file_type": f"{source}.reader.kind",
        "file_pattern": f"{source}.reader.file_pattern",
        "group_by_filename": f"{source}.reader.group_by_filename",
        "ih_group_pattern": f"{source}.reader.group_by_filename",
        "file_group_pattern": f"{source}.reader.group_by_filename",
        "datetime_format": f"{source}.transforms.parse_datetime.format",
        "natural_key": f"{source}.schema.natural_key",
        "streaming": f"{source}.reader.streaming",
        "hive_partitioning": f"{source}.reader.hive_partitioning",
        "background": f"{source}.reader",
    }
    for key in sorted(cfg):
        target = known.get(key)
        if target is not None:
            report.mapped(f"sources.{source_id}.{key}", target)
        elif key == "extensions":
            _record_extension_field_mappings(source_id, _dict(cfg.get("extensions")), report)
        else:
            report.gap(f"sources.{source_id}.{key}", "unknown legacy source field")


def _build_processors_and_metrics(
    metrics_block: dict[str, Any],
    report: MigrationReport,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    processors: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    for family, cfg in _metric_family_configs(metrics_block):
        kind = _PROCESSOR_KIND_BY_FAMILY.get(family)
        if kind is None:  # Defensive; _metric_family_configs already filters this.
            continue
        processor = _processor_for_family(family, kind, cfg, report)
        processors.append(processor)
        metrics.update(_metrics_for_family(family, kind, cfg, report))
        _record_metric_field_mappings(family, kind, cfg, report)
        report.mapped(f"metrics.{family}", f"processors.{processor['id']}", f"kind={kind}")
    return processors, metrics


def _metric_family_configs(metrics_block: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (family, cfg)
        for family, cfg in metrics_block.items()
        if family in _PROCESSOR_KIND_BY_FAMILY and isinstance(cfg, dict)
    ]


def _record_metrics_container_mappings(
    metrics_block: dict[str, Any],
    report: MigrationReport,
) -> None:
    for key, value in sorted(metrics_block.items()):
        if key in _PROCESSOR_KIND_BY_FAMILY:
            if not isinstance(value, dict):
                report.gap(f"metrics.{key}", "metric family must be a TOML table")
            continue
        if key == "global_filters":
            report.gap(
                "metrics.global_filters",
                "legacy global filters need manual dashboard filter wiring",
                "dashboards filters",
            )
        else:
            report.gap(f"metrics.{key}", "unknown legacy metrics-level setting")


def _processor_for_family(
    family: str,
    kind: str,
    cfg: dict[str, Any],
    report: MigrationReport,
) -> dict[str, Any]:
    source = "holdings" if kind == "entity_lifecycle" else "ih"
    processor: dict[str, Any] = {
        "id": family,
        "source": source,
        "kind": kind,
        "group_by": [column for column in _group_by(cfg) if column not in _TIME_GROUPS],
        "time": {
            "column": str(
                cfg.get(
                    "timestamp_column",
                    "PurchasedDateTime" if source == "holdings" else "OutcomeTime",
                )
            ),
            "grains": list(cfg.get("grains", ["Day", "Month", "Summary"])),
        },
    }
    if family == "experiment":
        experiment_name = str(cfg.get("experiment_name", "ExperimentName"))
        if experiment_name not in processor["group_by"]:
            processor["group_by"].append(experiment_name)
    filter_expr = _translate_filter(str(cfg.get("filter", "")), report, f"metrics.{family}.filter")
    if filter_expr is not None:
        processor["filter"] = filter_expr

    if kind == "binary_outcome":
        scores = [str(score) for score in cfg.get("scores", [])]
        states = {
            "Count": {"type": "count"},
            "Positives": {"type": "count"},
            "Negatives": {"type": "count"},
        }
        if "Revenue" in scores:
            states["Revenue"] = {"type": "value_sum", "source_column": "Revenue"}
        processor.update(
            {
                "outcome": {
                    "column": str(cfg.get("outcome_column", "Outcome")),
                    "positive_values": list(cfg.get("positive_model_response", ["Clicked"])),
                    "negative_values": list(
                        cfg.get("negative_model_response", ["Impression", "Pending"])
                    ),
                },
                "states": states,
            }
        )
        variant_column = _variant_column_for_family(family, cfg)
        if variant_column is not None:
            processor["variant_column"] = variant_column
    elif kind == "numeric_distribution":
        properties = _numeric_properties(cfg)
        processor["properties"] = properties
        processor["quantile_engine"] = (
            "tdigest" if _boolish(cfg.get("use_t_digest", True)) else "kll"
        )
        processor["states"] = _numeric_state_specs(properties, processor["quantile_engine"])
    elif kind == "score_distribution":
        score_properties = _score_properties_for_cfg(cfg)
        processor.update(
            {
                "score_properties": score_properties,
                "outcome": {
                    "column": str(cfg.get("outcome_column", "Outcome")),
                    "positive_values": list(cfg.get("positive_model_response", ["Clicked"])),
                    "negative_values": list(cfg.get("negative_model_response", ["Impression"])),
                },
                "dedup_keys": list(cfg.get("dedup_keys", ["InteractionID", "ActionID", "Rank"])),
                "states": _score_state_specs(score_properties),
            }
        )
    elif kind == "entity_lifecycle":
        processor["keys"] = {
            "customer_id": str(cfg.get("customer_id", "CustomerID")),
            "order_id": str(cfg.get("order_id", "HoldingID")),
            "monetary": str(cfg.get("monetary", "LifetimeValue")),
            "purchase_date": str(cfg.get("purchase_date", "PurchasedDateTime")),
        }
        processor["time"] = {"column": processor["time"]["column"], "grains": ["Summary"]}
        processor["states"] = {
            "unique_holdings": {"type": "count"},
            "lifetime_value": {"type": "value_sum", "source_column": processor["keys"]["monetary"]},
            "MinPurchasedDate": {
                "type": "min",
                "source_column": processor["keys"]["purchase_date"],
            },
            "MaxPurchasedDate": {
                "type": "max",
                "source_column": processor["keys"]["purchase_date"],
            },
            "UniquePurchasers_cpc": {
                "type": "cpc",
                "source_column": processor["keys"]["customer_id"],
                "lg_k": 11,
            },
        }
    elif kind == "entity_set":
        entity = str(cfg.get("entity", "CustomerID"))
        processor["states"] = {
            "ActiveUsers_cpc": {"type": "cpc", "source_column": entity, "lg_k": 11},
            "ActiveUsers_theta": {"type": "theta", "source_column": entity, "lg_k": 12},
        }
    elif kind == "funnel":
        stages = cfg.get("stages", ["Impression", "Clicked", "Conversion"])
        processor["stages"] = [
            {"name": str(stage), "when": {"op": "eq", "column": "Outcome", "value": str(stage)}}
            for stage in stages
        ]
    elif kind == "snapshot":
        processor["snapshot_kind"] = str(cfg.get("snapshot_kind", "periodic"))
        processor["cadence"] = str(cfg.get("cadence", "daily"))
        processor["states"] = {"Count": {"type": "count"}}
    return processor


def _metrics_for_family(  # noqa: PLR0912
    family: str,
    kind: str,
    cfg: dict[str, Any],
    report: MigrationReport,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    scores = [str(score) for score in cfg.get("scores", [])]
    if kind == "binary_outcome":
        rate_name = _binary_rate_metric_name(family, scores)
        if rate_name is not None:
            metrics[rate_name] = {
                "source": family,
                "kind": "formula",
                "expression": _binary_rate_expr(),
            }
        for score in scores:
            if score in {rate_name, "CTR"}:
                continue
            if score in _STATE_SCORE_NAMES:
                metrics.setdefault(score, _passthrough_metric(family, score))
                continue
            if score.startswith("Lift"):
                outputs = _variant_outputs(scores)
                metrics.setdefault(
                    "Lift",
                    {
                        "source": family,
                        "kind": "variant_compare",
                        "variant_column": str(
                            _variant_column_for_family(family, cfg) or "ModelControlGroup"
                        ),
                        "test_role": str(cfg.get("test_role", "Test")),
                        "control_role": str(cfg.get("control_role", "Control")),
                        "outputs": outputs,
                    },
                )
            elif score in _experiment_score_names():
                metrics.setdefault(
                    "Experiment_Significance",
                    {
                        "source": family,
                        "kind": "contingency_test",
                        "variant_column": str(
                            _variant_column_for_family(family, cfg) or "ExperimentGroup"
                        ),
                        "tests": ["chi2", "g", "z"],
                        "outputs": scores,
                    },
                )
            else:
                report.gap(f"metrics.{family}.scores.{score}", "unknown binary score")
    elif kind == "numeric_distribution":
        for prop in _numeric_properties(cfg):
            for score in scores or ["Median"]:
                metric_name, metric_cfg = _numeric_metric_for_score(family, prop, score)
                if metric_cfg is None:
                    report.gap(
                        f"metrics.{family}.scores.{score}", "unknown numeric distribution score"
                    )
                    continue
                metrics[metric_name] = metric_cfg
    elif kind == "score_distribution":
        requested = scores or ["roc_auc", "average_precision"]
        score_properties = _score_properties_for_cfg(cfg)
        curve_property = score_properties[0]
        calibration_property = _calibration_score_property(cfg, score_properties)
        for score in requested:
            if score in {"roc_auc", "ROC_AUC"}:
                metrics[score] = _curve_metric(family, "roc_auc", curve_property)
                metrics.setdefault("ROC_AUC", _curve_metric(family, "roc_auc", curve_property))
            elif score in {"average_precision", "AvgPrecision"}:
                metrics[score] = _curve_metric(family, "average_precision", curve_property)
                metrics.setdefault(
                    "AvgPrecision",
                    _curve_metric(family, "average_precision", curve_property),
                )
            elif score in {"personalization", "novelty", "Count"}:
                metrics[score] = _passthrough_metric(family, score)
            else:
                report.gap(f"metrics.{family}.scores.{score}", "unknown score distribution score")
        metrics.setdefault(
            "Calibration",
            {
                "source": family,
                "kind": "calibration_from_digests",
                "positive_state": f"{calibration_property}_tdigest_positives",
                "negative_state": f"{calibration_property}_tdigest_negatives",
            },
        )
    elif kind == "entity_lifecycle":
        metrics["CLV_Summary"] = {
            "source": family,
            "kind": "lifecycle_summary",
            "outputs": ["recency", "frequency", "monetary_value", "lifetime_value", "rfm_segment"],
        }
    elif kind == "entity_set":
        metrics["ActiveUsers"] = {
            "source": family,
            "kind": "approx_distinct_count",
            "state": "ActiveUsers_cpc",
        }
    return metrics


def _record_metric_field_mappings(
    family: str,
    kind: str,
    cfg: dict[str, Any],
    report: MigrationReport,
) -> None:
    processor = f"processors.{family}"
    metric_target = "metrics"
    known = {
        "group_by": f"{processor}.group_by + {processor}.time.grains",
        "filter": f"{processor}.filter",
        "grains": f"{processor}.time.grains",
        "scores": metric_target,
        "positive_model_response": f"{processor}.outcome.positive_values",
        "negative_model_response": f"{processor}.outcome.negative_values",
        "outcome_column": f"{processor}.outcome.column",
        "variant_column": f"{processor}.variant_column",
        "test_role": f"{metric_target}.Lift.test_role",
        "control_role": f"{metric_target}.Lift.control_role",
        "properties": f"{processor}.properties",
        "value_columns": f"{processor}.properties",
        "columns": f"{processor}.properties",
        "use_t_digest": f"{processor}.quantile_engine",
        "score_column": f"{processor}.score_properties",
        "calibrated_score_column": f"{processor}.score_properties",
        "dedup_keys": f"{processor}.dedup_keys",
        "experiment_name": f"{processor}.group_by",
        "experiment_group": f"{processor}.variant_column",
        "customer_id": f"{processor}.keys.customer_id",
        "customer_id_col": f"{processor}.keys.customer_id",
        "order_id": f"{processor}.keys.order_id",
        "order_id_col": f"{processor}.keys.order_id",
        "monetary": f"{processor}.keys.monetary",
        "monetary_value_col": f"{processor}.keys.monetary",
        "purchase_date": f"{processor}.keys.purchase_date",
        "purchase_date_col": f"{processor}.keys.purchase_date",
        "model": f"{processor}.model",
        "recurring_period": f"{processor}.recurring_period",
        "recurring_cost": f"{processor}.recurring_cost",
        "lifespan": f"{processor}.lifespan",
        "rfm_segment_config": f"{metric_target}.CLV_Summary.segment_preset",
        "entity": f"{processor}.states",
        "stages": f"{processor}.stages",
        "snapshot_kind": f"{processor}.snapshot_kind",
        "cadence": f"{processor}.cadence",
    }
    for key in sorted(cfg):
        target = known.get(key)
        if target is not None:
            report.mapped(f"metrics.{family}.{key}", target, f"kind={kind}")
        else:
            report.gap(f"metrics.{family}.{key}", "unknown legacy metric field")


def _build_dashboards(
    raw: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    report: MigrationReport,
) -> dict[str, Any]:
    reports = _dict(raw.get("reports"))
    pages: list[dict[str, Any]] = []
    if reports:
        for name, cfg in reports.items():
            report_cfg = _dict(cfg)
            page_id = _snake(name)
            title = _report_title(name, report_cfg)
            pages.append(
                {
                    "id": page_id,
                    "title": title,
                    "tiles": [_tile_for_report(page_id, title, report_cfg, metrics, report)],
                }
            )
            report.mapped(f"reports.{name}", f"dashboards.migrated.pages.{page_id}")
            _record_report_field_mappings(name, report_cfg, report)
    else:
        pages.append(
            {
                "id": "overview",
                "title": "Overview",
                "tiles": [
                    _tile_for_metric(_snake(metric), metric, metrics)
                    for metric in list(metrics)[:8]
                ],
            }
        )
    return {
        "dashboards": [
            {
                "id": "migrated_overview",
                "title": f"{_workspace_name(raw).replace('_', ' ').title()} Reports",
                "layout": "tabs",
                "pages": pages,
            }
        ]
    }


def _record_report_field_mappings(
    name: str,
    cfg: dict[str, Any],
    report: MigrationReport,
) -> None:
    page = f"dashboards.migrated.pages.{_snake(name)}"
    known = {
        "metric": f"{page}.tiles[0].metric",
        "title": f"{page}.title",
        "description": f"{page}.title",
        "chart": f"{page}.tiles[0].chart",
        "type": f"{page}.tiles[0].chart",
        "reference": f"{page}.tiles[0].references",
        "facet_column": f"{page}.tiles[0].facet_col",
    }
    for legacy, target in _LEGACY_REPORT_FIELD_MAP.items():
        known.setdefault(legacy, f"{page}.tiles[0].{target}")
    for key in sorted(cfg):
        mapped_target = known.get(key)
        if mapped_target is not None:
            report.mapped(f"reports.{name}.{key}", mapped_target)
        else:
            report.gap(f"reports.{name}.{key}", "unknown legacy report field")


def _tile_for_metric(
    tile_id: str, metric_name: str, metrics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    metric = metrics.get(metric_name, {})
    chart = "line"
    tile: dict[str, Any] = {
        "id": tile_id,
        "title": metric_name.replace("_", " "),
        "metric": metric_name,
        "chart": chart,
        "grain": "Day",
        "x": "Day",
        "y": metric_name,
    }
    if metric.get("kind") == "lifecycle_summary":
        tile = {"id": tile_id, "title": metric_name, "metric": metric_name, "chart": "rfm_density"}
    elif metric.get("kind") == "calibration_from_digests":
        tile = {
            "id": tile_id,
            "title": metric_name,
            "metric": metric_name,
            "chart": "calibration_curve",
        }
    return tile


def _tile_for_report(  # noqa: PLR0912
    tile_id: str,
    title: str,
    cfg: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    report: MigrationReport,
) -> dict[str, Any]:
    metric_name = _metric_for_report(cfg, metrics)
    chart = _chart_for_report(cfg)
    tile: dict[str, Any] = {
        "id": tile_id,
        "title": title,
        "metric": metric_name,
        "chart": chart,
    }
    for key, value in cfg.items():
        if key in {"metric", "title", "description", "type"}:
            continue
        if key == "facet_column":
            tile["facet_col"] = value
        elif key == "reference":
            tile["references"] = value
        elif key in {"log_x", "log_y", "showlegend"}:
            tile[key] = _boolish(value)
        elif key in _LEGACY_REPORT_FIELD_MAP:
            tile[_LEGACY_REPORT_FIELD_MAP[key]] = value

    if "group_by" in tile and chart == "treemap" and "path" not in tile:
        tile["path"] = list(tile["group_by"])
    if "x" not in tile and chart in {"line", "bar", "descriptive_line"}:
        tile["x"] = _default_x_for_report(cfg)
    if "y" not in tile and chart in {"line", "bar", "scatter"}:
        tile["y"] = metric_name
    if chart.startswith("descriptive_") and "property" not in tile:
        tile["property"] = str(cfg.get("y") or cfg.get("x") or "ResponseTime")
    if chart.startswith("descriptive_") and "score" not in tile:
        tile["score"] = str(cfg.get("score", "Mean"))
    if chart == "gauge" and "value" not in tile:
        tile["value"] = metric_name
    if chart == "bar_polar":
        tile.setdefault("r", metric_name)
    if chart == "calibration_curve":
        tile.pop("x", None)
        tile.pop("y", None)
    if metric_name not in metrics:
        report.gap(
            f"reports.{tile_id}.metric",
            f"could not resolve legacy report metric to a generated metric: {metric_name}",
            "dashboards tile metric",
        )
    return tile


def _workspace_name(raw: dict[str, Any]) -> str:
    variants = _dict(raw.get("variants"))
    return str(raw.get("workspace") or raw.get("variant") or variants.get("name") or "migrated")


def _legacy_source_config(raw: dict[str, Any], source_id: str) -> dict[str, Any]:
    cfg = dict(_dict(_dict(raw.get("sources")).get(source_id)))
    top_level = _dict(raw.get(source_id))
    for key, value in top_level.items():
        cfg.setdefault(key, value)
    return cfg


def _legacy_group_by_filename(source_id: str, cfg: dict[str, Any]) -> str:
    if "group_by_filename" in cfg:
        return str(cfg["group_by_filename"])
    if source_id == "ih" and "ih_group_pattern" in cfg:
        return str(cfg["ih_group_pattern"])
    if "file_group_pattern" in cfg:
        return str(cfg["file_group_pattern"])
    return r"(\d{8})"


def _legacy_default_values(cfg: dict[str, Any]) -> dict[str, Any]:
    extensions = _dict(cfg.get("extensions"))
    defaults = _dict(extensions.get("default_values"))
    return dict(defaults)


def _record_extension_field_mappings(
    source_id: str,
    extensions: dict[str, Any],
    report: MigrationReport,
) -> None:
    source = f"pipelines.sources[{source_id}]"
    for key in sorted(extensions):
        if key == "default_values":
            report.mapped(
                f"sources.{source_id}.extensions.default_values",
                f"{source}.transforms.defaults.values",
            )
        elif key == "filter":
            if str(extensions.get(key, "")).strip():
                report.gap(
                    f"sources.{source_id}.extensions.filter",
                    "source extension filters need manual AST translation",
                    f"{source}.transforms.filter",
                )
            else:
                report.mapped(f"sources.{source_id}.extensions.filter", f"{source}.transforms")
        elif key == "columns":
            if extensions.get(key):
                report.gap(
                    f"sources.{source_id}.extensions.columns",
                    "source extension column expressions need manual transform translation",
                    f"{source}.transforms.derive_column",
                )
            else:
                report.mapped(f"sources.{source_id}.extensions.columns", f"{source}.transforms")
        else:
            report.gap(f"sources.{source_id}.extensions.{key}", "unknown legacy source extension")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _variant_column_for_family(family: str, cfg: dict[str, Any]) -> str | None:
    if "variant_column" in cfg:
        return str(cfg["variant_column"])
    if family == "experiment":
        return str(cfg.get("experiment_group", "ExperimentGroup"))
    scores = {str(score) for score in cfg.get("scores", [])}
    if any(score.startswith("Lift") for score in scores):
        group_by = set(_group_by(cfg))
        if "HoldOutActionGroup" in group_by:
            return "HoldOutActionGroup"
        return "ModelControlGroup"
    return None


def _numeric_properties(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("properties", cfg.get("value_columns", cfg.get("columns", ["ResponseTime"])))
    if not isinstance(raw, list):
        return ["ResponseTime"]
    return [str(item) for item in raw]


def _numeric_state_specs(properties: list[str], engine: str) -> dict[str, dict[str, Any]]:
    sketch = "kll" if engine == "kll" else "tdigest"
    states: dict[str, dict[str, Any]] = {}
    for prop in properties:
        states[f"{prop}_Count"] = {"type": "count", "per_property": True}
        states[f"{prop}_Sum"] = {"type": "value_sum", "per_property": True}
        states[f"{prop}_Mean"] = {
            "type": "pooled_mean",
            "per_property": True,
            "weight": f"{prop}_Count",
        }
        states[f"{prop}_Var"] = {"type": "pooled_variance", "per_property": True}
        states[f"{prop}_Min"] = {"type": "min", "per_property": True}
        states[f"{prop}_Max"] = {"type": "max", "per_property": True}
        states[f"{prop}_{sketch}"] = {"type": sketch, "per_property": True}
    return states


def _score_state_specs(properties: list[str]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {
        "Count": {"type": "count"},
        "personalization": {"type": "pooled_mean", "weight": "Count"},
        "novelty": {"type": "pooled_mean", "weight": "Count"},
        "UniqueCustomers_cpc": {"type": "cpc", "source_column": "CustomerID", "lg_k": 11},
    }
    for prop in properties:
        states[f"{prop}_tdigest_positives"] = {
            "type": "tdigest",
            "source_column": prop,
            "score_property": prop,
            "outcome": "positive",
            "k": 500,
        }
        states[f"{prop}_tdigest_negatives"] = {
            "type": "tdigest",
            "source_column": prop,
            "score_property": prop,
            "outcome": "negative",
            "k": 500,
        }
    return states


def _score_properties_for_cfg(cfg: dict[str, Any]) -> list[str]:
    configured = cfg.get("score_properties")
    if isinstance(configured, list):
        properties = [str(item) for item in configured if str(item).strip()]
    elif isinstance(configured, str):
        properties = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        properties = []
    if not properties:
        properties = [
            str(cfg.get("score_column", "Propensity")),
            str(cfg.get("calibrated_score_column", "FinalPropensity")),
        ]
    return list(dict.fromkeys(prop for prop in properties if prop))


def _calibration_score_property(cfg: dict[str, Any], properties: list[str]) -> str:
    legacy_calibrated = str(cfg.get("calibrated_score_column", "") or "")
    if legacy_calibrated and legacy_calibrated in properties:
        return legacy_calibrated
    return properties[0] if properties else "Propensity"


def _passthrough_metric(source: str, column: str) -> dict[str, Any]:
    return {"source": source, "kind": "formula", "expression": {"col": column}}


def _binary_rate_metric_name(family: str, scores: list[str]) -> str | None:
    if family == "engagement":
        return "CTR" if not scores or "CTR" in scores else None
    if family == "conversion":
        return "ConversionRate" if not scores or "ConversionRate" in scores else None
    if "CTR" in scores:
        return "CTR"
    return None


def _variant_outputs(scores: list[str]) -> list[str]:
    outputs = ["TestCTR", "ControlCTR"]
    outputs.extend(score for score in scores if score.startswith("Lift"))
    outputs.append("StdErr")
    return list(dict.fromkeys(outputs))


def _experiment_score_names() -> set[str]:
    return {
        "z_score",
        "z_p_val",
        "g_stat",
        "g_p_val",
        "chi2_stat",
        "chi2_p_val",
        "g_odds_ratio_stat",
        "g_odds_ratio_ci_low",
        "g_odds_ratio_ci_high",
        "chi2_odds_ratio_stat",
        "chi2_odds_ratio_ci_low",
        "chi2_odds_ratio_ci_high",
    }


def _curve_metric(source: str, output: str, score_property: str = "Propensity") -> dict[str, Any]:
    return {
        "source": source,
        "kind": "curve_from_digests",
        "positive_state": f"{score_property}_tdigest_positives",
        "negative_state": f"{score_property}_tdigest_negatives",
        "output": output,
    }


def _numeric_metric_for_score(
    family: str,
    prop: str,
    score: str,
) -> tuple[str, dict[str, Any] | None]:
    normalized = str(score)
    state_metric = f"{prop}_{normalized}"
    if normalized in {"Count", "Sum", "Mean", "Var", "Min", "Max"}:
        return state_metric, _passthrough_metric(family, state_metric)
    if normalized == "Std":
        return state_metric, {
            "source": family,
            "kind": "formula",
            "expression": {"op": "sqrt", "arg": {"col": f"{prop}_Var"}},
        }
    quantile = {
        "Median": 0.5,
        "p25": 0.25,
        "p75": 0.75,
        "p90": 0.9,
        "p95": 0.95,
    }.get(normalized)
    if quantile is not None:
        return state_metric, {
            "source": family,
            "kind": "tdigest_quantile",
            "state": f"{prop}_tdigest",
            "quantile": quantile,
        }
    return state_metric, None


def _report_title(name: str, cfg: dict[str, Any]) -> str:
    return str(cfg.get("description") or cfg.get("title") or name.replace("_", " ").title()).strip()


def _metric_for_report(  # noqa: PLR0911, PLR0912
    cfg: dict[str, Any], metrics: dict[str, dict[str, Any]]
) -> str:
    raw_metric = str(cfg.get("metric", next(iter(metrics), "CTR")))
    if raw_metric in metrics:
        return raw_metric
    family = raw_metric
    if family == "descriptive":
        prop = str(cfg.get("property") or cfg.get("y") or cfg.get("x") or "ResponseTime")
        score = str(cfg.get("score", "Mean"))
        if str(cfg.get("type", "")).casefold() == "boxplot":
            score = "Median"
        metric = f"{prop}_{score}"
        if metric in metrics:
            return metric
    for key in ("value", "y", "color", "r", "x"):
        candidate = str(cfg.get(key, ""))
        if candidate in metrics:
            return candidate
        if candidate.startswith("Lift") and "Lift" in metrics:
            return "Lift"
    if family == "engagement" and "CTR" in metrics:
        return "CTR"
    if family == "conversion" and "ConversionRate" in metrics:
        return "ConversionRate"
    if family == "model_ml_scores":
        for candidate in ("roc_auc", "ROC_AUC", "average_precision", "AvgPrecision"):
            if candidate in metrics:
                return candidate
    if family == "experiment" and "Experiment_Significance" in metrics:
        return "Experiment_Significance"
    if family == "clv" and "CLV_Summary" in metrics:
        return "CLV_Summary"
    return next(iter(metrics), raw_metric)


def _chart_for_report(cfg: dict[str, Any]) -> str:
    legacy = str(cfg.get("type", cfg.get("chart", "line"))).strip() or "line"
    family = str(cfg.get("metric", ""))
    if legacy == "generic":
        return "line"
    if family == "descriptive":
        mapped = {
            "line": "descriptive_line",
            "boxplot": "descriptive_boxplot",
            "histogram": "descriptive_histogram",
            "heatmap": "descriptive_heatmap",
            "funnel": "descriptive_funnel",
        }
        return mapped.get(legacy, legacy)
    if family == "experiment":
        x_y = f"{cfg.get('x', '')} {cfg.get('y', '')}"
        if "odds_ratio" in x_y:
            return "experiment_odds_ratio"
        if legacy == "line":
            return "experiment_z_score"
    if family == "clv" and legacy == "treemap":
        return "clv_treemap"
    return legacy


def _default_x_for_report(cfg: dict[str, Any]) -> str:
    for candidate in _group_by(cfg):
        if candidate in _TIME_GROUPS:
            return candidate
    return "Day"


def _translate_filter(raw: str, report: MigrationReport, location: str) -> dict[str, Any] | None:
    stripped = raw.strip()
    if not stripped:
        return None
    candidate = stripped
    if candidate.startswith("(") and candidate.endswith(")"):
        candidate = candidate[1:-1].strip()
    match = re.fullmatch(
        r"""(?:pl\.)?col\(["'](?P<column>[^"']+)["']\)\.is_in\(\[(?P<values>.*)]\)""",
        candidate,
    )
    if match:
        values = [
            value.strip().strip("'\"")
            for value in match.group("values").split(",")
            if value.strip()
        ]
        return {"op": "in", "column": match.group("column"), "values": values}
    report.gap(location, "filter expression needs manual AST translation")
    return None


def _group_by(cfg: dict[str, Any]) -> list[str]:
    raw = cfg.get("group_by", [])
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _infer_sources(metrics_block: dict[str, Any]) -> set[str]:
    out = {"ih"} if metrics_block else set()
    if any(_PROCESSOR_KIND_BY_FAMILY.get(family) == "entity_lifecycle" for family in metrics_block):
        out.add("holdings")
    return out


def _binary_rate_expr() -> dict[str, Any]:
    return {
        "op": "safe_div",
        "num": {"col": "Positives"},
        "den": {"op": "add", "args": [{"col": "Positives"}, {"col": "Negatives"}]},
    }


def _snake(value: str) -> str:
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(value))
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.strip("_").lower() or "field"


def _title_id(value: str) -> str:
    return "".join(part.title() for part in _snake(value).split("_"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["MigrationReport", "migrate_toml"]
