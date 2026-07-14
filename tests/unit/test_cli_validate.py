"""Integration tests for ``valuestream validate``.

Covers the demo workspace happy path plus several deliberately-broken
catalogs that should exit non-zero with a structured error message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from valuestream.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_WS = REPO_ROOT / "examples" / "demo"
# Demo metric names exercised by the error-path tests below.
FORMULA_METRIC = "VS_Engagement_Rate"
CURVE_METRIC = "ih_propensity_scores_roc_auc"


def _seed_workspace(ws: Path) -> None:
    """Copy the demo catalog into ``ws/catalog`` so we can mutate it per test."""
    src = DEMO_WS / "catalog"
    dst = ws / "catalog"
    dst.mkdir(parents=True)
    for f in src.iterdir():
        (dst / f.name).write_text(f.read_text())


@pytest.mark.unit
class TestHappyPath:
    def test_demo_validates_clean(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(DEMO_WS)])
        assert result.exit_code == 0, result.output
        # Rich may wrap the line on narrow terminals; collapse whitespace.
        flattened = " ".join(result.output.split())
        assert "validates clean" in flattened

    def test_summary_table_rendered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(DEMO_WS)])
        assert result.exit_code == 0
        assert "sources" in result.output
        assert "processors" in result.output
        assert "metrics" in result.output


@pytest.mark.unit
class TestErrorPaths:
    def test_missing_workspace(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path / "no-such")])
        # click's path-existence check fires before our handler.
        assert result.exit_code != 0

    def test_missing_catalog_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert result.exit_code == 1
        assert "catalog directory" in result.output

    def test_unknown_metric_in_tile(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        # Rewrite dashboards.yaml with a tile referencing an undefined metric.
        dash_path = ws / "catalog" / "dashboards.yaml"
        dash = yaml.safe_load(dash_path.read_text())
        dash["dashboards"][0]["pages"][0]["tiles"][0]["metric"] = "DoesNotExist"
        dash_path.write_text(yaml.safe_dump(dash))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "DoesNotExist" in result.output
        assert "failed" in result.output

    def test_unknown_processor_in_metric(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        metrics_path = ws / "catalog" / "metrics.yaml"
        metrics = yaml.safe_load(metrics_path.read_text())
        metrics["metrics"][FORMULA_METRIC]["source"] = "no_such_processor"
        metrics_path.write_text(yaml.safe_dump(metrics))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "no_such_processor" in result.output

    def test_metric_formula_with_unknown_state(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        metrics_path = ws / "catalog" / "metrics.yaml"
        metrics = yaml.safe_load(metrics_path.read_text())
        # The formula's denominator references an unknown state.
        metrics["metrics"][FORMULA_METRIC]["expression"] = {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {"col": "TotallyUnknown"},
        }
        metrics_path.write_text(yaml.safe_dump(metrics))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "TotallyUnknown" in result.output

    def test_curve_metric_with_unknown_digest_state(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        metrics_path = ws / "catalog" / "metrics.yaml"
        metrics = yaml.safe_load(metrics_path.read_text())
        metrics["metrics"][CURVE_METRIC]["positive_state"] = "MissingDigest"
        metrics_path.write_text(yaml.safe_dump(metrics))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "MissingDigest" in result.output

    def test_curve_metric_with_non_digest_state(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        metrics_path = ws / "catalog" / "metrics.yaml"
        metrics = yaml.safe_load(metrics_path.read_text())
        metrics["metrics"][CURVE_METRIC]["positive_state"] = "Count"
        metrics_path.write_text(yaml.safe_dump(metrics))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "'tdigest', got 'count'" in result.output

    def test_filter_transform_with_unknown_column(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        pipelines_path = ws / "catalog" / "pipelines.yaml"
        pipelines = yaml.safe_load(pipelines_path.read_text())
        ih = pipelines["sources"][0]
        filter_transform = next(t for t in ih["transforms"] if t["kind"] == "filter")
        filter_transform["expression"] = {"op": "not_null", "column": "Channnel"}
        pipelines_path.write_text(yaml.safe_dump(pipelines))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "Channnel" in result.output

    def test_processor_filter_with_unknown_column(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        proc_path = ws / "catalog" / "processors.yaml"
        proc = yaml.safe_load(proc_path.read_text())
        proc["processors"][0]["filter"] = {"op": "eq", "column": "Outcomme", "value": "Clicked"}
        proc_path.write_text(yaml.safe_dump(proc))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "Outcomme" in result.output

    def test_group_by_columns_are_trusted_processor_config(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        proc_path = ws / "catalog" / "processors.yaml"
        proc = yaml.safe_load(proc_path.read_text())
        proc["processors"][0]["dimensions"].append("GhostColumn")
        proc_path.write_text(yaml.safe_dump(proc))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 0
        assert "validates clean" in " ".join(result.output.split())

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        (ws / "catalog" / "pipelines.yaml").write_text(": :\n: invalid")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
        assert "YAML parse error" in result.output

    def test_unknown_processor_kind(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        _seed_workspace(ws)
        proc_path = ws / "catalog" / "processors.yaml"
        proc = yaml.safe_load(proc_path.read_text())
        proc["processors"][0]["kind"] = "wibble"
        proc_path.write_text(yaml.safe_dump(proc))
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(ws)])
        assert result.exit_code == 1
