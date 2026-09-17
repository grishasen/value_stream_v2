"""Tool-surface tests that build the real MCP server object.

``tests/unit/test_cli_serve.py`` mocks ``run_stdio`` wholesale, so it never
touches the SDK. These tests construct the server the CLI would construct,
which is what caught the FastMCP/MCPServer rename going unnoticed.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from valuestream.mcp import server as mcp_server
from valuestream.query import AggregateNotReadyError
from valuestream.reporting import render as render_module

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_WS = REPO_ROOT / "examples" / "demo"

BASE_TOOLS = {
    "metric_list",
    "metric_query",
    "metric_chart_query",
    "dashboard_list",
    "tile_query",
    "kpi_query",
    "chat",
    "workspace_status_tool",
    "dimension_values_tool",
    "freshness_get",
}
SQL_TOOLS = {"sql_schema", "sql_query"}


def _build(workspace: Path, *, enable_sql: bool = False, render_dir: Path | None = None) -> Any:
    """Run the server builder, capturing the instance instead of serving it."""

    captured: dict[str, Any] = {}
    server_cls = mcp_server._server_class()

    class Captured(server_cls):  # type: ignore[misc, valid-type]
        def run(self, *args: Any, **kwargs: Any) -> None:
            captured["server"] = self

    original = mcp_server._server_class
    mcp_server._server_class = lambda: Captured  # type: ignore[assignment]
    try:
        mcp_server.run_stdio(workspace, enable_sql=enable_sql, render_dir=render_dir)
    finally:
        mcp_server._server_class = original  # type: ignore[assignment]
    return captured["server"]


def _first_tile_ids(server: Any) -> tuple[str, str, str]:
    """Return the first dashboard/page/tile id trio the server reports."""

    manifest = server._tool_manager.get_tool("dashboard_list").fn()
    dashboard = manifest["dashboards"][0]
    page = dashboard["pages"][0]
    return dashboard["id"], page["id"], page["tiles"][0]["id"]


def _tool_names(server: Any) -> set[str]:
    return {tool.name for tool in server._tool_manager.list_tools()}


@pytest.mark.unit
def test_server_class_resolves_against_installed_sdk() -> None:
    server_cls = mcp_server._server_class()

    # Both SDK majors keep this shape; the body of run_stdio relies on it.
    assert callable(server_cls)
    instance = server_cls("Value Stream")
    assert callable(instance.tool)
    assert callable(instance.run)


@pytest.mark.unit
def test_registers_read_only_tool_surface() -> None:
    server = _build(DEMO_WS)

    assert _tool_names(server) == BASE_TOOLS


@pytest.mark.unit
def test_sql_tools_are_opt_in() -> None:
    assert _tool_names(_build(DEMO_WS)).isdisjoint(SQL_TOOLS)
    assert _tool_names(_build(DEMO_WS, enable_sql=True)) >= SQL_TOOLS


@pytest.mark.unit
def test_metric_query_defaults_keep_curve_arrays_out() -> None:
    server = _build(DEMO_WS)
    schema = server._tool_manager.get_tool("metric_query").parameters["properties"]

    assert schema["include_curves"]["default"] is False
    assert schema["provenance"]["default"] == "summary"


@pytest.mark.unit
def test_invalid_metric_returns_structured_error_not_exception() -> None:
    server = _build(DEMO_WS)
    tool = server._tool_manager.get_tool("metric_query")

    result = tool.fn(metric="NoSuchMetric")

    assert result["error"]["kind"] == "invalid_request"
    assert "NoSuchMetric" in result["error"]["message"]
    assert result["error"]["remediation"]


@pytest.mark.unit
def test_stale_aggregates_are_reported_with_backfill_remediation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    shutil.copytree(DEMO_WS, workspace)
    server = _build(workspace)
    tool = server._tool_manager.get_tool("metric_query")

    def raise_stale(*args: Any, **kwargs: Any) -> Any:
        raise AggregateNotReadyError("aggregate data for ih/engagement/summary is not ready")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mcp_server, "query_metric_result", raise_stale)
        result = tool.fn(metric="CTR")

    assert result["error"]["kind"] == "aggregate_not_ready"
    assert "backfill" in result["error"]["remediation"].lower()


@pytest.mark.unit
def test_provenance_summary_drops_id_lists_but_keeps_counts() -> None:
    class Provenance:
        @staticmethod
        def to_dict() -> dict[str, Any]:
            return {
                "metric": "CTR",
                "chunk_ids": ["a", "b", "c"],
                "pipeline_run_ids": ["r1"],
                "aggregate_rows_scanned": 10,
            }

    summary = mcp_server._provenance_payload(Provenance(), "summary")
    full = mcp_server._provenance_payload(Provenance(), "full")

    assert "chunk_ids" not in summary
    assert summary["chunk_ids_count"] == 3
    assert summary["pipeline_run_ids_count"] == 1
    assert summary["aggregate_rows_scanned"] == 10
    assert full["chunk_ids"] == ["a", "b", "c"]


@pytest.mark.unit
def test_sql_failures_point_at_the_sql_schema_not_the_metric_list() -> None:
    # Governed SQL rejects statements with a plain ValueError, so the
    # remediation cannot be chosen from the exception type alone.
    payload = mcp_server._error_payload(
        "sql_query",
        ValueError("governed SQL rejected; read_parquet is not allowed"),
    )

    assert payload["error"]["kind"] == "sql_rejected"
    assert "sql_schema" in payload["error"]["remediation"]

    metric = mcp_server._error_payload("metric_query", ValueError("unknown metric 'X'"))
    assert metric["error"]["kind"] == "invalid_request"
    assert "metric_list" in metric["error"]["remediation"]


@pytest.mark.unit
def test_dashboard_list_returns_ids_the_tile_tools_accept(demo_workspace: Path) -> None:
    # Querying a tile needs real aggregates, so this uses the ingested fixture
    # workspace rather than the catalog-only example directory.
    server = _build(demo_workspace)

    manifest = server._tool_manager.get_tool("dashboard_list").fn()
    dashboard = manifest["dashboards"][0]
    page = dashboard["pages"][0]
    tile = page["tiles"][0]

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard["id"],
        page_id=page["id"],
        tile_id=tile["id"],
    )

    assert "error" not in result, result
    assert result["tile_id"] == tile["id"]
    assert result["metric"] == tile["metric"]


@pytest.mark.unit
def test_tile_query_reports_filters_the_tile_cannot_apply(demo_workspace: Path) -> None:
    server = _build(demo_workspace)
    manifest = server._tool_manager.get_tool("dashboard_list").fn()
    dashboard = manifest["dashboards"][0]
    page = dashboard["pages"][0]
    tile = page["tiles"][0]

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard["id"],
        page_id=page["id"],
        tile_id=tile["id"],
        filters={"NotAColumn": "x"},
    )

    assert result["ignored_filters"] == ["NotAColumn"]
    assert result["applied_filters"] == {}


@pytest.mark.unit
def test_unknown_tile_error_lists_the_valid_tile_ids() -> None:
    server = _build(DEMO_WS)
    manifest = server._tool_manager.get_tool("dashboard_list").fn()
    dashboard = manifest["dashboards"][0]
    page = dashboard["pages"][0]

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard["id"],
        page_id=page["id"],
        tile_id="nope",
    )

    assert result["error"]["kind"] == "invalid_request"
    assert page["tiles"][0]["id"] in result["error"]["message"]


@pytest.mark.unit
def test_state_blob_columns_never_reach_the_client() -> None:
    # Stateful charts ask the query layer for sketch columns. They are raw
    # binary: unserializable, and governed out of the SQL surface already.
    rows = pl.DataFrame(
        {
            "Channel": ["Web"],
            "Impression_Count": [10],
            "Propensity_tdigest": [b"\x00\x01"],
            "config_hash": ["abc"],
        }
    )

    clipped, masked = mcp_server._without_state_blobs(rows)

    assert masked == ["Propensity_tdigest"]
    assert clipped.columns == ["Channel", "Impression_Count"]


@pytest.mark.unit
def test_date_bounds_reject_non_iso_input_with_a_readable_message() -> None:
    assert mcp_server._as_date(None, "start") is None
    assert mcp_server._as_date("", "start") is None
    assert mcp_server._as_date("2024-05-15", "start") == dt.date(2024, 5, 15)

    with pytest.raises(ValueError, match="start must be an ISO date"):
        mcp_server._as_date("last tuesday", "start")


@pytest.mark.unit
def test_resources_and_prompts_are_registered() -> None:
    server = _build(DEMO_WS)

    resources = {str(resource.uri) for resource in server._resource_manager.list_resources()}
    prompts = {prompt.name for prompt in server._prompt_manager.list_prompts()}

    assert "valuestream://catalog" in resources
    assert "valuestream://dashboards" in resources
    assert {"explore_workspace", "starter_questions"} <= prompts


@pytest.mark.unit
def test_metric_list_defaults_to_the_compact_manifest() -> None:
    server = _build(DEMO_WS)
    tool = server._tool_manager.get_tool("metric_list")

    compact = tool.fn()
    full = tool.fn(detail="full", limit=1)

    assert compact["detail"] == "compact"
    assert compact["metrics"], "compact manifest still lists metrics"
    assert "configuration" not in compact["metrics"][0]
    # The full manifest carries the prose and configuration blocks.
    assert "configuration" in full["metrics"][0]
    assert len(json.dumps(compact["metrics"][0])) < len(json.dumps(full["metrics"][0]))


@pytest.mark.unit
def test_metric_list_search_narrows_by_name_and_description() -> None:
    server = _build(DEMO_WS)
    tool = server._tool_manager.get_tool("metric_list")

    everything = tool.fn()
    narrowed = tool.fn(search="engagement")

    assert narrowed["matched_metrics"] < everything["matched_metrics"]
    assert narrowed["metrics"]
    assert all(
        "engagement" in entry["name"].casefold()
        or "engagement" in str(entry.get("description") or "").casefold()
        for entry in narrowed["metrics"]
    )


@pytest.mark.unit
def test_metric_list_restricts_to_one_dashboard_or_processor() -> None:
    server = _build(DEMO_WS)
    tool = server._tool_manager.get_tool("metric_list")
    manifest = server._tool_manager.get_tool("dashboard_list").fn()
    dashboard_id = manifest["dashboards"][0]["id"]
    used = {
        tile["metric"]
        for page in manifest["dashboards"][0]["pages"]
        for tile in page["tiles"]
    }

    by_dashboard = tool.fn(dashboard=dashboard_id)

    assert {entry["name"] for entry in by_dashboard["metrics"]} <= used

    unknown = tool.fn(dashboard="nope")
    assert unknown["error"]["kind"] == "invalid_request"


@pytest.mark.unit
def test_metric_list_reports_truncation_rather_than_silently_cutting() -> None:
    server = _build(DEMO_WS)
    tool = server._tool_manager.get_tool("metric_list")

    limited = tool.fn(limit=2)

    assert limited["returned_metrics"] == 2
    assert limited["truncated"] is True
    assert limited["matched_metrics"] > 2


@pytest.mark.unit
def test_chat_explains_itself_when_no_model_is_configured(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    shutil.copytree(DEMO_WS, workspace)
    (workspace / "ai.yaml").write_text("ai:\n  llm:\n    model: ''\n")
    server = _build(workspace)

    result = server._tool_manager.get_tool("chat").fn(question="what is the click rate?")

    assert result["error"]["kind"] == "invalid_request"
    assert "ai.yaml" in result["error"]["message"]


@pytest.mark.unit
def test_workspace_status_agrees_with_what_a_query_would_do(demo_workspace: Path) -> None:
    server = _build(demo_workspace)

    status = server._tool_manager.get_tool("workspace_status_tool").fn()

    assert status["queryable"] is True
    assert status["processors_ready"] == len(status["processors"])
    assert all(item["state"] == "ready" for item in status["processors"])
    assert "ready" in status["summary"]


@pytest.mark.unit
def test_workspace_status_explains_an_unqueryable_workspace(tmp_path: Path) -> None:
    # A catalog with no aggregates at all is the simplest unready workspace.
    workspace = tmp_path / "ws"
    shutil.copytree(DEMO_WS, workspace)
    server = _build(workspace)

    status = server._tool_manager.get_tool("workspace_status_tool").fn()

    assert status["queryable"] is False
    assert status["processors_ready"] == 0
    assert all(item["state"] != "ready" for item in status["processors"])
    assert all(item["detail"] for item in status["processors"])
    assert "cannot be queried" in status["summary"]


@pytest.mark.unit
def test_tile_query_render_defaults_to_rows_only(demo_workspace: Path) -> None:
    server = _build(demo_workspace)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile
    )

    assert isinstance(result, dict)
    assert "render" not in result


@pytest.mark.unit
def test_tile_query_rejects_an_unknown_render_mode(demo_workspace: Path) -> None:
    server = _build(demo_workspace)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="gif"
    )

    assert result["error"]["kind"] == "invalid_request"
    assert "none, png, html, html_cdn, spec" in result["error"]["message"]


@pytest.mark.unit
def test_tile_query_html_writes_a_self_contained_file(
    demo_workspace: Path, tmp_path: Path
) -> None:
    # Interactive HTML only helps in a browser, so the response carries a path
    # rather than spending itself on markup the model cannot render. Because it
    # is a file, Plotly is embedded and the page opens offline.
    renders = tmp_path / "renders"
    server = _build(demo_workspace, render_dir=renders)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="html"
    )

    assert result["render"]["mode"] == "html"
    assert result["render"]["self_contained"] is True
    written = Path(result["render"]["path"])
    assert written.parent == renders
    assert 'src="https://cdn.plot.ly' not in written.read_text()


@pytest.mark.unit
def test_tile_query_html_cdn_trades_offline_use_for_a_small_file(
    demo_workspace: Path, tmp_path: Path
) -> None:
    renders = tmp_path / "renders"
    server = _build(demo_workspace, render_dir=renders)
    dashboard, page, tile = _first_tile_ids(server)

    embedded = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="html"
    )
    cdn = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="html_cdn"
    )

    assert cdn["render"]["self_contained"] is False
    assert cdn["render"]["bytes"] < embedded["render"]["bytes"] / 10
    assert 'src="https://cdn.plot.ly' in Path(cdn["render"]["path"]).read_text()


@pytest.mark.unit
def test_render_never_writes_into_the_workspace(demo_workspace: Path, tmp_path: Path) -> None:
    renders = tmp_path / "renders"
    server = _build(demo_workspace, render_dir=renders)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="html"
    )

    assert demo_workspace not in Path(result["render"]["path"]).parents


@pytest.mark.unit
def test_render_spec_returns_a_drawable_figure(demo_workspace: Path) -> None:
    server = _build(demo_workspace)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="spec"
    )

    assert result["render"]["mode"] == "spec"
    assert set(result["render"]["spec"]) >= {"data", "layout"}


@pytest.mark.unit
def test_render_is_skipped_with_a_reason_when_there_is_nothing_to_plot(
    demo_workspace: Path,
) -> None:
    server = _build(demo_workspace)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard,
        page_id=page,
        tile_id=tile,
        filters={"Channel": "NoSuchChannel"},
        render="spec",
    )

    assert result["render"]["skipped"]


@pytest.mark.unit
def test_png_without_the_viz_extra_is_a_readable_dependency_error(
    demo_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Most installs will not have kaleido, so the missing extra has to reach
    # the client as an instruction rather than as a crash.
    monkeypatch.setattr(render_module, "png_available", lambda: False)
    server = _build(demo_workspace)
    dashboard, page, tile = _first_tile_ids(server)

    result = server._tool_manager.get_tool("tile_query").fn(
        dashboard_id=dashboard, page_id=page, tile_id=tile, render="png"
    )

    assert result["error"]["kind"] == "dependency_missing"
    assert "uv sync --extra viz" in result["error"]["remediation"]
    assert 'render="html"' in result["error"]["message"]
