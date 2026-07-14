"""Minimal MCP server for read-only Value Stream aggregate tools."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from valuestream.ai.chat import (
    catalog_chat_manifest,
    chart_intent_from_parameters,
    dimension_values,
    execute_chat_intent,
)
from valuestream.ai.sql_tool import list_sql_tables, run_sql_query
from valuestream.config.watch import CatalogCache
from valuestream.query import query_metric_result
from valuestream.ui.freshness import freshness_label, metric_freshness
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)


def run_stdio(workspace_path: str | Path, *, enable_sql: bool = False) -> None:
    """Run the Value Stream MCP server over stdio for one workspace."""

    try:
        fast_mcp_cls = import_module("mcp.server.fastmcp").FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The MCP server requires the optional `ai` dependencies. "
            "Install them with `uv sync --extra ai` or `uv sync --all-extras`."
        ) from exc

    workspace = Path(workspace_path).resolve()
    catalog_cache = CatalogCache(workspace)
    catalog_cache.get()  # fail fast if the catalog is invalid at startup
    mcp = fast_mcp_cls("Value Stream")
    logger.info("Starting Value Stream MCP server: workspace=%s", workspace)

    @mcp.tool()
    def metric_list() -> dict[str, Any]:
        """List metrics, dimensions, query time axes, and supported chart kinds."""

        manifest = catalog_chat_manifest(catalog_cache.get())
        logger.debug(
            "MCP metric_list: workspace=%s metrics=%s",
            workspace,
            len(manifest.get("metrics", [])),
        )
        return manifest

    @mcp.tool()
    def metric_query(
        metric: str,
        group_by: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        grain: str = "summary",
        start: str | None = None,
        end: str | None = None,
        having: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        top_n: int | None = None,
        top_n_by: str | None = None,
        compare: str | None = None,
        include_quantile_suite: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query metric rows through the governed aggregate query layer.

        Filter values may be scalars, lists, or operator objects such as
        {"op": ">=", "value": 3} or {"op": "not_in", "values": [...]}.
        `having` applies the same operator objects to metric output columns
        after aggregation. `order_by` accepts column names with an optional
        "-" prefix for descending. `top_n` keeps the largest rows by
        `top_n_by`. `compare="prior_period"` adds *_prev/*_delta/*_pct_change
        columns over the time axis. Use metric_chart_query for chart requests
        so chart parameters are explicit.
        """

        logger.info(
            "MCP metric_query: metric=%s grain=%s group_by=%s filters=%s start=%s end=%s "
            "having=%s order_by=%s top_n=%s compare=%s limit=%s",
            metric,
            grain,
            group_by or [],
            list((filters or {}).keys()),
            start,
            end,
            list((having or {}).keys()),
            order_by or [],
            top_n,
            compare,
            limit,
        )
        result = query_metric_result(
            workspace,
            metric,
            group_by=group_by or [],
            filters=filters or {},
            grain=grain,
            start=start,
            end=end,
            having=having or {},
            order_by=order_by or [],
            top_n=top_n,
            top_n_by=top_n_by,
            compare=compare,
            include_quantile_suite=include_quantile_suite,
            include_curve_columns=True,
        )
        frame = result.rows
        clipped = frame.head(max(1, min(int(limit), 500)))
        logger.info(
            "MCP metric_query completed: metric=%s grain=%s rows=%s returned=%s columns=%s",
            metric,
            grain,
            frame.height,
            clipped.height,
            clipped.columns,
        )
        return {
            "metric": metric,
            "grain": grain,
            "group_by": group_by or [],
            "filters": filters or {},
            "having": having or {},
            "order_by": order_by or [],
            "top_n": top_n,
            "compare": compare,
            "row_count": frame.height,
            "provenance": result.provenance.to_dict(),
            "rows": clipped.to_dicts(),
        }

    @mcp.tool()
    def metric_chart_query(
        metric: str,
        chart_kind: str,
        x: str,
        y: str,
        group_by: list[str],
        filters: dict[str, Any] | None = None,
        grain: str = "summary",
        start: str | None = None,
        end: str | None = None,
        color: str | None = None,
        facet_col: str | None = None,
        having: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        top_n: int | None = None,
        top_n_by: str | None = None,
        compare: str | None = None,
        value_format: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query metric rows and return an explicit validated chart spec.

        The model must provide chart_kind, x, y, group_by, color, and facet_col
        explicitly. Use null for optional color/facet_col values. Filter and
        having values may be operator objects such as {"op": ">=", "value": 3}.
        chart_kind must be one of the metric's chart_kinds from metric_list;
        value_format may be percent/integer/number/currency.
        """

        catalog = catalog_cache.get()
        logger.info(
            "MCP metric_chart_query: metric=%s kind=%s x=%s y=%s color=%s facet_col=%s grain=%s "
            "group_by=%s filters=%s having=%s order_by=%s top_n=%s compare=%s limit=%s",
            metric,
            chart_kind,
            x,
            y,
            color,
            facet_col,
            grain,
            group_by,
            list((filters or {}).keys()),
            list((having or {}).keys()),
            order_by or [],
            top_n,
            compare,
            limit,
        )
        intent = chart_intent_from_parameters(
            catalog,
            metric=metric,
            chart_kind=chart_kind,
            x=x,
            y=y,
            group_by=group_by,
            filters=filters or {},
            grain=grain,
            start=start,
            end=end,
            color=color,
            facet_col=facet_col,
            having=having or {},
            order_by=order_by or [],
            top_n=top_n,
            top_n_by=top_n_by,
            compare=compare,
            value_format=value_format,
            limit=limit,
        )
        result = execute_chat_intent(workspace, catalog, intent)
        chart = result.intent.chart
        chart_spec = {
            "kind": chart.kind if chart else chart_kind,
            "x": chart.x if chart else x,
            "y": chart.y if chart else y,
            "color": chart.color if chart else color,
            "facet_col": chart.facet_col if chart else facet_col,
            "value_format": chart.value_format if chart else value_format,
        }
        logger.info(
            "MCP metric_chart_query completed: metric=%s grain=%s rows=%s chart=%s",
            result.intent.metric,
            result.intent.grain,
            result.rows.height,
            chart_spec,
        )
        return {
            "metric": result.intent.metric,
            "grain": result.intent.grain,
            "group_by": result.intent.group_by,
            "filters": result.intent.filters,
            "row_count": result.rows.height,
            "columns": result.rows.columns,
            "chart": chart_spec,
            "query": result.query_summary,
            "freshness": result.freshness,
            "rows": result.rows.to_dicts(),
            "rendering_instruction": "Render using chart.x, chart.y, chart.color, and chart.facet_col exactly as returned.",
        }

    @mcp.tool()
    def dimension_values_tool(
        metric: str,
        column: str,
        grain: str = "summary",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return aggregate-backed distinct values for one metric dimension."""

        logger.info(
            "MCP dimension_values_tool: metric=%s column=%s grain=%s limit=%s",
            metric,
            column,
            grain,
            limit,
        )
        return {
            "metric": metric,
            "column": column,
            "values": dimension_values(
                workspace,
                catalog_cache.get(),
                metric,
                column,
                grain=grain,
                limit=limit,
            ),
        }

    if enable_sql:

        @mcp.tool()
        def sql_schema() -> dict[str, Any]:
            """List governed DuckDB tables/views available to sql_query.

            Names are fully qualified; sketch/state blob columns are hidden.
            """

            logger.info("MCP sql_schema: workspace=%s", workspace)
            tables = list_sql_tables(workspace, catalog_cache.get())
            return {
                "tables": [
                    {
                        "name": table.name,
                        "kind": table.kind,
                        "columns": [{"name": name, "type": dtype} for name, dtype in table.columns],
                    }
                    for table in tables
                ],
                "notes": [
                    "Only single read-only SELECT (or WITH ... SELECT) statements are accepted.",
                    "Use the fully qualified table names exactly as listed.",
                    "Row counts are capped; sketch state columns are masked from results.",
                ],
            }

        @mcp.tool()
        def sql_query(sql: str, limit: int = 200) -> dict[str, Any]:
            """Run one governed read-only SELECT over the aggregate DuckDB views.

            Use sql_schema first to discover table names and columns. DDL/DML,
            multiple statements, comments, and file/catalog functions are rejected.
            """

            logger.info("MCP sql_query: workspace=%s limit=%s", workspace, limit)
            result = run_sql_query(workspace, sql, catalog=catalog_cache.get(), limit=limit)
            return {
                "sql": result.sql,
                "row_count": result.row_count,
                "truncated": result.truncated,
                "masked_columns": result.masked_columns,
                "columns": result.rows.columns,
                "rows": result.rows.to_dicts(),
            }

    @mcp.tool()
    def freshness_get(metric: str, grain: str = "summary") -> dict[str, Any]:
        """Return freshness metadata for a metric and grain."""

        logger.info("MCP freshness_get: metric=%s grain=%s", metric, grain)
        fresh = metric_freshness(workspace, catalog_cache.get(), metric, grain=grain)
        return {
            "metric": metric,
            "grain": grain,
            "latest_period": fresh.latest_period,
            "last_created_at": fresh.last_created_at.isoformat() if fresh.last_created_at else None,
            "last_run_finished_at": fresh.last_run_finished_at.isoformat()
            if fresh.last_run_finished_at
            else None,
            "status": fresh.status,
            "label": freshness_label(fresh),
        }

    mcp.run()


__all__ = ["run_stdio"]
