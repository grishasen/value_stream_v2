"""Chat intent planning helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import valuestream.ai.chat as chat_module
from valuestream.ai.chat import (
    ChartIntent,
    ChatIntent,
    allowed_chart_kinds,
    catalog_chat_manifest,
    chart_intent_from_parameters,
    chart_tile_from_intent,
    chat_pin_tile,
    chat_starter_questions,
    deterministic_chat_starters,
    execute_chat_intent,
    execute_deterministic_chat_query,
    parse_chat_intent,
    prompt_for_chat_intent,
)
from valuestream.config.loader import load


@pytest.mark.unit
def test_parse_chart_intent_adds_chart_dimensions() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "chart",
        "group_by": ["Customer Segment"],
        "filters": {},
        "grain": "daily",
        "start": None,
        "end": None,
        "chart": {
            "kind": "line",
            "x": "Day",
            "y": "VS_Engagement_Rate",
            "color": "channel",
            "facet_col": "CustomerSegment",
        },
        "limit": 100,
    }

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="Plot daily engagement rate",
    )

    assert intent.metric == "VS_Engagement_Rate"
    assert intent.response == "chart"
    assert intent.grain == "daily"
    assert intent.group_by == ["CustomerSegment", "Channel"]
    assert intent.chart is not None
    assert intent.chart.x == "Day"
    assert intent.chart.color == "Channel"


@pytest.mark.unit
def test_parse_daily_chart_replaces_metric_x_axis_with_day() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "chart",
        "group_by": ["Channel", "PropensitySource"],
        "filters": {},
        "grain": "daily",
        "start": None,
        "end": None,
        "chart": {
            "kind": "line",
            "x": "VS_Engagement_Rate",
            "y": "VS_Engagement_Rate",
            "color": "Channel",
            "facet_col": "PropensitySource",
        },
        "limit": 100,
    }

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="Plot daily CTR by channel and propensity source.",
    )

    assert intent.chart is not None
    assert intent.chart.x == "Day"
    assert intent.chart.y == "VS_Engagement_Rate"
    assert intent.chart.color == "Channel"
    assert intent.chart.facet_col == "PropensitySource"
    assert intent.group_by == ["Channel", "PropensitySource"]


@pytest.mark.unit
def test_parse_daily_chart_preserves_dimension_x_axis() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "chart",
        "group_by": ["Issue", "Channel"],
        "filters": {},
        "grain": "daily",
        "start": None,
        "end": None,
        "chart": {
            "kind": "bar",
            "x": "Issue",
            "y": "VS_Engagement_Rate",
            "color": "Channel",
            "facet_col": None,
        },
        "limit": 100,
    }

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="Plot daily engagement rate by issue and channel.",
    )

    assert intent.chart is not None
    assert intent.chart.x == "Issue"
    assert intent.chart.y == "VS_Engagement_Rate"
    assert intent.chart.color == "Channel"
    assert intent.group_by == ["Issue", "Channel"]


@pytest.mark.unit
def test_parse_summary_chart_replaces_metric_x_axis_with_group_dimension() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "chart",
        "group_by": ["Channel"],
        "filters": {},
        "grain": "summary",
        "start": None,
        "end": None,
        "chart": {
            "kind": "bar",
            "x": "VS_Engagement_Rate",
            "y": "VS_Engagement_Rate",
            "color": None,
            "facet_col": None,
        },
        "limit": 100,
    }

    intent = parse_chat_intent(json.dumps(payload), catalog)

    assert intent.chart is not None
    assert intent.chart.x == "Channel"
    assert intent.chart.y == "VS_Engagement_Rate"


@pytest.mark.unit
def test_parse_time_trend_defaults_missing_color_to_grouped_dimension() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "chart",
        "group_by": ["Channel"],
        "filters": {},
        "grain": "monthly",
        "start": None,
        "end": None,
        "chart": {
            "kind": "line",
            "x": "Month",
            "y": "VS_Engagement_Rate",
            "color": None,
            "facet_col": None,
        },
        "limit": 100,
    }

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="Show engagement rate monthly by channel.",
    )

    assert intent.chart is not None
    assert intent.chart.x == "Month"
    assert intent.chart.color == "Channel"
    assert intent.group_by == ["Channel"]


@pytest.mark.unit
def test_parse_intent_ignores_llm_grain_and_infers_month_from_question() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "clicks_e4cbbaaa71be3fb4",
        "response": "text",
        "group_by": [],
        "filters": {},
        "grain": "monthly",
        "start": None,
        "end": None,
        "chart": None,
        "limit": 100,
    }

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="Show Clicks by month.",
    )

    assert intent.metric == "clicks_e4cbbaaa71be3fb4"
    assert intent.grain == "monthly"
    assert intent.group_by == []


@pytest.mark.unit
def test_execute_monthly_clicks_uses_demo_daily_aggregate(demo_workspace: Path) -> None:
    catalog = load(demo_workspace)
    intent = parse_chat_intent(
        json.dumps(
            {
                "metric": "clicks_e4cbbaaa71be3fb4",
                "response": "text",
                "group_by": [],
                "filters": {},
                "time_axis": "Month",
                "start": None,
                "end": None,
                "chart": None,
                "limit": 100,
            }
        ),
        catalog,
        question="Show Clicks by month.",
    )

    result = execute_chat_intent(demo_workspace, catalog, intent)

    assert result.intent.grain == "monthly"
    assert result.rows.columns == ["Month", "clicks_e4cbbaaa71be3fb4"]
    # The synthetic dataset spans four months (see tests/conftest.py).
    assert result.rows.height == 4


@pytest.mark.unit
def test_chart_intent_from_parameters_validates_explicit_mcp_chart_fields() -> None:
    catalog = load(Path("examples/demo"))

    intent = chart_intent_from_parameters(
        catalog,
        metric="VS_Engagement_Rate",
        chart_kind="line",
        x="Day",
        y="VS_Engagement_Rate",
        group_by=["Channel", "PropensitySource"],
        grain="daily",
        color="Channel",
        facet_col="PropensitySource",
    )

    assert intent.chart is not None
    assert intent.chart.x == "Day"
    assert intent.chart.y == "VS_Engagement_Rate"
    assert intent.group_by == ["Channel", "PropensitySource"]


@pytest.mark.unit
def test_allowed_chart_kinds_are_metric_aware() -> None:
    catalog = load(Path("examples/demo"))

    formula = allowed_chart_kinds(catalog, "VS_Engagement_Rate")
    assert {"line", "bar", "table", "kpi_card"} <= set(formula)
    assert "roc_curve" not in formula
    assert "calibration_curve" not in formula

    roc = allowed_chart_kinds(catalog, "ih_propensity_scores_roc_auc")
    assert "roc_curve" in roc
    assert "calibration_curve" not in roc

    calibration = allowed_chart_kinds(catalog, "VS_FinalPropensity_Calibration")
    assert "calibration_curve" in calibration

    assert allowed_chart_kinds(catalog, "does_not_exist") == ["line", "bar", "table", "kpi_card"]


@pytest.mark.unit
def test_parse_intent_gates_chart_kind_to_metric_allowlist() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "chart",
        "group_by": ["Channel"],
        "filters": {},
        "chart": {
            "kind": "roc_curve",
            "x": "Channel",
            "y": "VS_Engagement_Rate",
            "value_format": "percent",
        },
    }

    intent = parse_chat_intent(json.dumps(payload), catalog, question="engagement rate by channel")

    assert intent.chart is not None
    # roc_curve is not allowed for a formula metric; falls back to a bar comparison.
    assert intent.chart.kind == "bar"
    assert intent.chart.value_format == "percent"


@pytest.mark.unit
def test_chart_tile_from_intent_maps_kind_specific_fields() -> None:
    donut = chart_tile_from_intent(
        ChatIntent(
            question="",
            metric="VS_Interactions",
            response="chart",
            group_by=["Channel"],
            filters={},
            grain="summary",
            chart=ChartIntent(kind="donut", x="Channel", y="VS_Interactions"),
        )
    )
    assert donut["chart"] == "donut"
    assert donut["names"] == "Channel"
    assert donut["values"] == "VS_Interactions"

    heatmap = chart_tile_from_intent(
        ChatIntent(
            question="",
            metric="VS_Engagement_Rate",
            response="chart",
            group_by=["Channel", "Issue"],
            filters={},
            grain="summary",
            chart=ChartIntent(kind="heatmap", x="Channel", y="VS_Engagement_Rate", color="Issue"),
        )
    )
    assert heatmap["x"] == "Channel"
    assert heatmap["y"] == "Issue"
    assert heatmap["color"] == "VS_Engagement_Rate"

    kpi = chart_tile_from_intent(
        ChatIntent(
            question="",
            metric="VS_Engagement_Rate",
            response="chart",
            group_by=[],
            filters={},
            grain="summary",
            chart=ChartIntent(kind="kpi_card", y="VS_Engagement_Rate", value_format="percent"),
        )
    )
    assert kpi["chart"] == "kpi_card"
    assert kpi["value"] == "VS_Engagement_Rate"
    assert kpi["value_format"] == "percent"

    roc = chart_tile_from_intent(
        ChatIntent(
            question="",
            metric="ih_propensity_scores_roc_auc",
            response="chart",
            group_by=["Channel"],
            filters={},
            grain="summary",
            chart=ChartIntent(kind="roc_curve", color="Channel"),
        )
    )
    assert roc["chart"] == "roc_curve"
    assert roc["color"] == "Channel"
    assert "x" not in roc
    assert "y" not in roc


@pytest.mark.unit
def test_chat_starter_questions_are_metric_grounded() -> None:
    catalog = load(Path("examples/demo"))

    questions = chat_starter_questions(catalog, limit=3)

    assert 1 <= len(questions) <= 3
    assert all(isinstance(question, str) and question for question in questions)
    # Starters reference real dimensions or time axes, not raw metric ids.
    assert any("by" in question.lower() or "overall" in question.lower() for question in questions)


@pytest.mark.unit
def test_deterministic_starters_cover_supported_test_ai_studio_contracts() -> None:
    catalog = load(Path("examples/test_ai_studio"))

    starters = deterministic_chat_starters(catalog)
    by_key = {starter.key: starter for starter in starters}

    assert list(by_key) == ["count", "rate", "unique", "channel", "date_range"]
    assert by_key["count"].intent.metric == "Studio_Count"
    assert by_key["rate"].intent.metric in {"Studio_CTR", "VS_Engagement_Rate"}
    assert by_key["unique"].intent.metric == "VS_Unique_Entities"
    assert by_key["unique"].intent.metric in catalog.metrics.metrics
    assert by_key["channel"].intent.group_by == ["Channel"]
    assert by_key["channel"].intent.chart is not None
    assert by_key["channel"].intent.chart.x == "Channel"
    assert by_key["date_range"].intent.grain == "daily"


@pytest.mark.unit
def test_deterministic_starters_execute_through_governed_demo_aggregates(
    demo_workspace: Path,
) -> None:
    catalog = load(demo_workspace)
    results = {
        starter.key: execute_deterministic_chat_query(demo_workspace, catalog, starter)
        for starter in deterministic_chat_starters(catalog)
    }

    assert set(results) == {"count", "rate", "unique", "channel", "date_range"}
    assert all(result.query_summary.startswith("query_metric(") for result in results.values())
    assert all(result.freshness for result in results.values())
    assert results["count"].rows.height == 1
    assert results["rate"].rows.height == 1
    assert results["unique"].rows.height == 1
    assert "Channel" in results["channel"].rows.columns
    assert results["channel"].rows.height == 3
    assert results["date_range"].rows.to_dicts() == [
        {
            "Available from": "2026-01-05",
            "Available through": "2026-04-20",
            "Grain": "daily",
        }
    ]


@pytest.mark.unit
def test_date_range_template_reads_only_query_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = load(Path("examples/test_ai_studio"))
    starter = next(
        item for item in deterministic_chat_starters(catalog) if item.key == "date_range"
    )
    calls: list[tuple[object, str, dict[str, object]]] = []

    def fake_query(workspace: object, metric: str, **kwargs: object) -> pl.DataFrame:
        calls.append((workspace, metric, kwargs))
        return pl.DataFrame(
            {
                "Day": ["2024-08-31", "2024-08-30"],
                metric: [20, 10],
            }
        )

    monkeypatch.setattr(chat_module, "query_metric", fake_query)
    monkeypatch.setattr(chat_module, "metric_freshness", lambda *args, **kwargs: object())
    monkeypatch.setattr(chat_module, "freshness_label", lambda value: "fresh")

    result = execute_deterministic_chat_query("/tmp/aggregate-only", catalog, starter)

    assert calls == [
        (
            "/tmp/aggregate-only",
            "Studio_Count",
            {
                "group_by": [],
                "filters": {},
                "grain": "daily",
                "include_curve_columns": True,
            },
        )
    ]
    assert result.rows.to_dicts() == [
        {
            "Available from": "2024-08-30",
            "Available through": "2024-08-31",
            "Grain": "daily",
        }
    ]


@pytest.mark.unit
def test_chat_pin_tile_reuses_chart_mapping_and_forces_id() -> None:
    chart_intent = ChatIntent(
        question="",
        metric="VS_Engagement_Rate",
        response="chart",
        group_by=["Channel"],
        filters={},
        grain="summary",
        chart=ChartIntent(kind="bar", x="Channel", y="VS_Engagement_Rate"),
    )
    chart_tile = chat_pin_tile(chart_intent, tile_id="pin_123")
    assert chart_tile["id"] == "pin_123"
    assert chart_tile["chart"] == "bar"
    assert chart_tile["metric"] == "VS_Engagement_Rate"

    text_intent = ChatIntent(
        question="",
        metric="VS_Interactions",
        response="text",
        group_by=[],
        filters={},
        grain="summary",
        chart=None,
    )
    text_tile = chat_pin_tile(text_intent, tile_id="pin_456")
    assert text_tile == {
        "id": "pin_456",
        "title": "VS_Interactions",
        "metric": "VS_Interactions",
        "chart": "table",
    }


@pytest.mark.unit
def test_manifest_exposes_per_metric_chart_kinds() -> None:
    catalog = load(Path("examples/demo"))

    manifest = catalog_chat_manifest(catalog)
    engagement = next(item for item in manifest["metrics"] if item["name"] == "VS_Engagement_Rate")

    assert "chart_kinds" in engagement
    assert "kpi_card" in engagement["chart_kinds"]
    assert set(manifest["supported_charts"]) >= {"line", "bar", "heatmap", "donut", "roc_curve"}


@pytest.mark.unit
def test_parse_intent_rejects_unknown_dimension() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "text",
        "group_by": ["CustomerID"],
        "filters": {},
        "grain": "summary",
        "start": None,
        "end": None,
        "chart": None,
        "limit": 100,
    }

    with pytest.raises(ValueError, match="CustomerID"):
        parse_chat_intent(json.dumps(payload), catalog)


@pytest.mark.unit
def test_parse_intent_accepts_having_order_top_n_and_compare() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "table",
        "group_by": ["Channel"],
        "filters": {"Channel": {"op": "not_in", "values": ["Unknown"]}},
        "having": {"vs engagement rate": {"op": ">", "value": 0.05}},
        "order_by": ["-VS_Engagement_Rate", "Channel"],
        "top_n": 5,
        "top_n_by": "VS_Engagement_Rate",
        "compare": "previous_period",
        "quantiles": False,
        "time_axis": "Month",
        "limit": 50,
    }

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="Top channels by engagement rate change month over month",
    )

    assert intent.grain == "monthly"
    assert intent.filters == {"Channel": {"op": "not_in", "values": ["Unknown"]}}
    assert intent.having == {"VS_Engagement_Rate": {"op": ">", "value": 0.05}}
    assert intent.order_by == ["-VS_Engagement_Rate", "Channel"]
    assert intent.top_n == 5
    assert intent.top_n_by == "VS_Engagement_Rate"
    assert intent.compare == "prior_period"


@pytest.mark.unit
def test_parse_intent_compare_requires_time_axis() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "metric": "VS_Engagement_Rate",
        "response": "text",
        "group_by": [],
        "filters": {},
        "compare": "prior_period",
    }

    with pytest.raises(ValueError, match="needs a time axis"):
        parse_chat_intent(json.dumps(payload), catalog, question="How did engagement change?")


@pytest.mark.unit
def test_parse_intent_returns_clarify_without_metric_resolution() -> None:
    catalog = load(Path("examples/demo"))
    payload = {
        "response": "clarify",
        "clarify": "Do you mean engagement rate or unique customers?",
    }

    intent = parse_chat_intent(json.dumps(payload), catalog, question="How are we doing?")

    assert intent.response == "clarify"
    assert intent.metric == ""
    assert intent.clarify == "Do you mean engagement rate or unique customers?"


@pytest.mark.unit
def test_parse_intent_gates_sql_responses() -> None:
    catalog = load(Path("examples/demo"))
    payload = {"response": "sql", "sql": "SELECT 1"}

    with pytest.raises(ValueError, match="SQL answers are not enabled"):
        parse_chat_intent(json.dumps(payload), catalog, question="join two metrics")

    intent = parse_chat_intent(
        json.dumps(payload),
        catalog,
        question="join two metrics",
        allow_sql=True,
    )

    assert intent.response == "sql"
    assert intent.sql == "SELECT 1"


@pytest.mark.unit
def test_prompt_advertises_query_features_and_optional_sql() -> None:
    catalog = load(Path("examples/demo"))

    prompt = prompt_for_chat_intent(catalog, "top channels by CTR")
    assert '"having"' in prompt
    assert '"top_n"' in prompt
    assert '"compare"' in prompt
    assert "clarify" in prompt
    assert "Governed SQL tables" not in prompt

    sql_prompt = prompt_for_chat_intent(
        catalog,
        "top channels by CTR",
        sql_schema='- metrics_summary."CTR" (table): Channel VARCHAR, CTR DOUBLE',
    )
    assert "Governed SQL tables" in sql_prompt
    assert '"sql": null' in sql_prompt


@pytest.mark.unit
def test_execute_intent_supports_top_n_and_order(demo_workspace: Path) -> None:
    catalog = load(demo_workspace)
    payload = {
        "metric": "VS_Interactions",
        "response": "table",
        "group_by": ["Channel"],
        "filters": {},
        "order_by": ["-VS_Interactions"],
        "top_n": 2,
        "top_n_by": "VS_Interactions",
    }
    intent = parse_chat_intent(
        json.dumps(payload), catalog, question="Top 2 channels by interactions"
    )

    result = execute_chat_intent(demo_workspace, catalog, intent)

    assert result.rows.height == 2
    values = result.rows.get_column("VS_Interactions").to_list()
    assert values == sorted(values, reverse=True)
    assert "top_n=2" in result.query_summary


@pytest.mark.unit
def test_catalog_manifest_exposes_metric_query_shape() -> None:
    catalog = load(Path("examples/demo"))

    manifest = catalog_chat_manifest(catalog)
    engagement = next(item for item in manifest["metrics"] if item["name"] == "VS_Engagement_Rate")
    dataset = next(item for item in manifest["datasets"] if item["id"] == "ih")
    processor = next(item for item in manifest["processors"] if item["id"] == "ih_engagement")

    assert dataset["reader"]["kind"] == "parquet"
    assert dataset["timestamp_column"] == "OutcomeTime"
    assert any(
        transform.get("kind") == "derive_column" and transform.get("output") == "ResponseTime"
        for transform in dataset["transforms"]
    )
    assert "root" not in dataset["reader"]
    assert processor["dataset"] == "ih"
    assert processor["kind_explanation"]
    assert processor["configuration"]["outcome"]["positive_values"] == ["Clicked", "Conversion"]
    assert "UniqueCustomers_cpc" in processor["available_query_fields"]["state_columns"]
    assert "Channel" in engagement["dimensions"]
    assert "CustomerSegment" in engagement["dimensions"]
    assert "Month" in engagement["time_axes"]
    assert engagement["outputs"] == ["VS_Engagement_Rate"]
    assert engagement["kind_explanation"]
    assert engagement["dataset"] == "ih"
    assert engagement["processor_kind"] == "binary_outcome"
    assert engagement["configuration"]["expression"]["op"] == "safe_div"


@pytest.mark.unit
def test_catalog_manifest_applies_chat_only_descriptions() -> None:
    catalog = load(Path("examples/demo"))
    chat_config = {
        "dataset_descriptions": {"ih": "Pega CDH interaction history aggregate dataset."},
        "metric_descriptions": {"ih_engagement": "Use this family for CTR and lift."},
    }

    manifest = catalog_chat_manifest(catalog, chat_config=chat_config)
    dataset = next(item for item in manifest["datasets"] if item["id"] == "ih")
    processor = next(item for item in manifest["processors"] if item["id"] == "ih_engagement")
    metric = next(item for item in manifest["metrics"] if item["name"] == "VS_Engagement_Rate")

    assert dataset["chat_description"] == "Pega CDH interaction history aggregate dataset."
    assert processor["chat_description"] == "Use this family for CTR and lift."
    assert metric["chat_description"] == "Use this family for CTR and lift."
    assert metric["dataset_chat_description"] == "Pega CDH interaction history aggregate dataset."


@pytest.mark.unit
def test_prompt_explains_datasets_processors_and_metric_configuration() -> None:
    catalog = load(Path("examples/demo"))

    prompt = prompt_for_chat_intent(
        catalog,
        "Which datasets and engagement metrics are available?",
    )

    assert '"datasets"' in prompt
    assert '"processors"' in prompt
    assert '"dataset": "ih"' in prompt
    assert '"kind_explanation"' in prompt
    assert '"expression"' in prompt
    assert '"positive_values"' in prompt
    assert '"grain"' not in prompt
    assert "Datasets/sources are not queryable directly" in prompt
    assert "Do not choose or return an aggregate grain" in prompt
    assert "Use the chat descriptions, metric descriptions, kind explanations" in prompt
    assert "include that dimension in group_by and chart.color" in prompt


@pytest.mark.unit
def test_prompt_includes_chat_agent_prompt() -> None:
    catalog = load(Path("examples/demo"))

    prompt = prompt_for_chat_intent(
        catalog,
        "Which engagement metrics are available?",
        chat_config={"agent_prompt": "Use the user's CDH business vocabulary."},
    )

    assert "Workspace chat guidance:" in prompt
    assert "Use the user's CDH business vocabulary." in prompt
    assert "The workspace chat guidance and chat descriptions help choose metrics" in prompt
