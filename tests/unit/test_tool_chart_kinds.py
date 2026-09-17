"""The chart kinds an explicit tool caller can request, end to end.

Each case builds a chart intent the way the MCP and HTTP chart tools build
one, runs it, maps it to a tile, and renders it through the real chart
factory. A kind that validates but cannot be drawn is worse than one that is
not offered, so the render is the assertion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from valuestream.ai.chat import (
    CHAT_CHART_KINDS,
    TOOL_CHART_KINDS,
    allowed_tool_chart_kinds,
    chart_intent_from_parameters,
    chart_tile_from_intent,
    execute_chat_intent,
)
from valuestream.charts import render_chart
from valuestream.config.loader import load

DEMO_WS = Path("examples/demo")
METRIC = "VS_Engagement_Rate"

EXTENDED_CASES: list[tuple[str, dict[str, Any]]] = [
    (
        "combo",
        {
            "chart_kind": "combo",
            "x": "Channel",
            "y": METRIC,
            "group_by": ["Channel"],
            "chart_fields": {
                "secondary_metric": "VS_Interactions",
                "primary_mark": "line",
                "shared_y_axis": True,
            },
        },
    ),
    (
        "treemap",
        {
            "chart_kind": "treemap",
            "x": "Channel",
            "y": METRIC,
            "group_by": ["Channel", "Issue"],
            "chart_fields": {"path": ["Channel", "Issue"]},
        },
    ),
    (
        "sankey",
        {
            "chart_kind": "sankey",
            "x": "Channel",
            "y": METRIC,
            "group_by": ["Channel", "Issue"],
            "chart_fields": {"path": ["Channel", "Issue"]},
        },
    ),
    (
        "pareto",
        {"chart_kind": "pareto", "x": "Channel", "y": METRIC, "group_by": ["Channel"]},
    ),
    (
        "waterfall",
        {"chart_kind": "waterfall", "x": "Channel", "y": METRIC, "group_by": ["Channel"]},
    ),
    (
        "gauge",
        {"chart_kind": "gauge", "x": "Channel", "y": METRIC, "group_by": ["Channel"]},
    ),
    (
        "bar_polar",
        {
            "chart_kind": "bar_polar",
            "x": "Channel",
            "y": METRIC,
            "group_by": ["Channel"],
            "chart_fields": {"theta": "Channel"},
        },
    ),
    (
        "line_with_facet_row_and_goal",
        {
            "chart_kind": "line",
            "x": "Day",
            "y": METRIC,
            "group_by": ["Channel", "Issue"],
            "color": "Channel",
            "grain": "daily",
            "chart_fields": {"facet_row": "Issue", "goal_line": 0.5},
        },
    ),
]


@pytest.mark.parametrize(("label", "kwargs"), EXTENDED_CASES, ids=[c[0] for c in EXTENDED_CASES])
@pytest.mark.unit
def test_tool_chart_kinds_render(
    demo_workspace: Path, label: str, kwargs: dict[str, Any]
) -> None:
    catalog = load(demo_workspace)
    assert kwargs["chart_kind"] in allowed_tool_chart_kinds(catalog, METRIC), label

    intent = chart_intent_from_parameters(catalog, metric=METRIC, **kwargs)
    result = execute_chat_intent(demo_workspace, catalog, intent)
    figure = render_chart(result.rows, chart_tile_from_intent(result.intent))

    assert len(figure.data) >= 1, f"{label} produced an empty figure"


@pytest.mark.unit
def test_tool_kinds_are_a_superset_of_the_planner_kinds() -> None:
    # The planner keeps the always-renderable subset; widening the tool set
    # must never widen what the LLM is asked to produce.
    assert set(CHAT_CHART_KINDS) < set(TOOL_CHART_KINDS)


@pytest.mark.unit
def test_missing_kind_specific_fields_name_what_is_missing() -> None:
    catalog = load(DEMO_WS)

    with pytest.raises(ValueError, match="requires secondary_metric"):
        chart_intent_from_parameters(
            catalog,
            metric=METRIC,
            chart_kind="combo",
            x="Channel",
            y=METRIC,
            group_by=["Channel"],
        )
    with pytest.raises(ValueError, match=r"requires .*lower_output"):
        chart_intent_from_parameters(
            catalog,
            metric=METRIC,
            chart_kind="interval",
            x="Channel",
            y=METRIC,
            group_by=["Channel"],
        )


@pytest.mark.unit
def test_a_path_needs_at_least_two_dimensions() -> None:
    catalog = load(DEMO_WS)

    with pytest.raises(ValueError, match="path of at least two"):
        chart_intent_from_parameters(
            catalog,
            metric=METRIC,
            chart_kind="treemap",
            x="Channel",
            y=METRIC,
            group_by=["Channel"],
            chart_fields={"path": ["Channel"]},
        )


@pytest.mark.unit
def test_primary_mark_is_restricted_to_marks_a_combo_can_draw() -> None:
    catalog = load(DEMO_WS)

    with pytest.raises(ValueError, match="primary_mark must be one of"):
        chart_intent_from_parameters(
            catalog,
            metric=METRIC,
            chart_kind="combo",
            x="Channel",
            y=METRIC,
            group_by=["Channel"],
            chart_fields={"secondary_metric": "VS_Interactions", "primary_mark": "pie"},
        )


@pytest.mark.unit
def test_fields_a_kind_ignores_do_not_reach_its_tile() -> None:
    # A bar chart given funnel stages should not echo them back as if they
    # were used; the returned spec is what a client renders from.
    catalog = load(DEMO_WS)

    intent = chart_intent_from_parameters(
        catalog,
        metric=METRIC,
        chart_kind="bar",
        x="Channel",
        y=METRIC,
        group_by=["Channel"],
        chart_fields={"stages": ["a", "b"], "secondary_metric": "VS_Interactions"},
    )

    assert intent.chart is not None
    assert intent.chart.stages == ()
    assert intent.chart.secondary_metric is None
