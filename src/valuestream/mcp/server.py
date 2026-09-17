"""Minimal MCP server for read-only Value Stream aggregate tools."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar

import duckdb
import polars as pl

from valuestream.ai.chat import (
    catalog_chat_manifest,
    chart_intent_from_parameters,
    chart_spec_warnings,
    chart_tile_from_intent,
    deterministic_chat_starters,
    dimension_values,
    execute_chat_intent,
    narrate_chat_result,
    plan_chat_intent,
)
from valuestream.ai.settings import (
    configured_api_key,
    load_chat_with_data_config,
    load_llm_settings_config,
)
from valuestream.ai.sql_tool import list_sql_tables, run_sql_query
from valuestream.ai.studio import AICallSettings
from valuestream.config.watch import CatalogCache
from valuestream.query import AggregateNotReadyError, query_metric_result
from valuestream.reporting.kpi import kpi_bundle
from valuestream.reporting.manifest import dashboard_manifest, resolve_tile
from valuestream.reporting.render import (
    RENDER_MODES,
    figure_for_tile,
    figure_html,
    figure_png,
    figure_spec,
    png_available,
    write_render,
)
from valuestream.reporting.status import workspace_status
from valuestream.reporting.tiles import (
    available_filter_columns_for_tile,
    grain_for_tile,
    query_tile,
    tile_to_dict,
)
from valuestream.ui.freshness import freshness_label, metric_freshness
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_MISSING_SDK_HINT = (
    "The MCP server requires the optional `ai` dependencies. "
    "Install them with `uv sync --extra ai` or `uv sync --all-extras`."
)

_BACKFILL_REMEDIATION = (
    "Run ingestion for this workspace (`valuestream run <workspace>`), or backfill/reprocess "
    "the affected source if the catalog changed since the last run."
)


def _server_class() -> Any:
    """Return the MCP server class for the installed SDK.

    The Python MCP SDK renamed ``FastMCP`` to ``MCPServer`` in 2.0; both keep
    the ``name`` constructor, the ``@tool()`` decorator, and ``run()``, so the
    server body below is version-neutral. Import 2.x first, fall back to 1.x,
    and only claim the extra is missing when the SDK itself is absent.
    """

    try:
        return import_module("mcp.server.mcpserver").MCPServer
    except ImportError:
        pass
    try:
        return import_module("mcp.server.fastmcp").FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        try:
            import_module("mcp")
        except ImportError:
            raise RuntimeError(_MISSING_SDK_HINT) from exc
        raise RuntimeError(
            f"The installed `mcp` SDK exposes neither MCPServer nor FastMCP: {exc}"
        ) from exc


_SQL_TOOLS = frozenset({"sql_query", "sql_schema"})

_SQL_REMEDIATION = "Call sql_schema for the exact table names and the governed-SQL rules."
_METRIC_REMEDIATION = "Call metric_list to check the metric name, dimensions, and grains."


def _image_class() -> Any:
    """Return the SDK's image-content helper, whichever major is installed."""

    try:
        return import_module("mcp.server.mcpserver").Image
    except ImportError:  # pragma: no cover - depends on the installed SDK major
        return import_module("mcp.server.fastmcp").Image


def _error_payload(tool: str, exc: Exception) -> dict[str, Any]:
    """Describe a governed-layer failure in terms the calling model can act on.

    The governed SQL layer rejects statements with a plain ValueError, so the
    remediation follows the tool that failed rather than the exception type.
    """

    kind = "invalid_request"
    remediation = _SQL_REMEDIATION if tool in _SQL_TOOLS else _METRIC_REMEDIATION
    if isinstance(exc, AggregateNotReadyError):
        kind = "aggregate_not_ready"
        remediation = _BACKFILL_REMEDIATION
    elif isinstance(exc, FileNotFoundError):
        kind = "aggregate_missing"
        remediation = _BACKFILL_REMEDIATION
    elif isinstance(exc, TimeoutError):
        kind = "timeout"
        remediation = (
            "Narrow the query with filters, a date range, or a coarser grain."
            if tool not in _SQL_TOOLS
            else "Narrow the SELECT, or add a WHERE clause; long queries are interrupted."
        )
    elif isinstance(exc, RuntimeError):
        kind = "dependency_missing"
        remediation = str(exc)
    elif isinstance(exc, duckdb.Error) or tool in _SQL_TOOLS:
        kind = "sql_rejected"
    return {
        "error": {
            "kind": kind,
            "type": type(exc).__name__,
            "message": str(exc),
            "remediation": remediation,
        }
    }


def _guarded(tool: str, call: Callable[[], T]) -> T | dict[str, Any]:
    """Return a tool result, or a structured error the client can read.

    Letting the exception escape gives MCP clients a bare "Error executing
    tool <name>" and drops the actionable message, so every tool answers with
    a payload instead.
    """

    try:
        return call()
    except (ValueError, FileNotFoundError, TimeoutError, RuntimeError, duckdb.Error) as exc:
        logger.warning("MCP %s failed: %s: %s", tool, type(exc).__name__, exc)
        return _error_payload(tool, exc)


def run_stdio(  # noqa: PLR0915 - one function body per registered tool
    workspace_path: str | Path,
    *,
    enable_sql: bool = False,
    render_dir: str | Path | None = None,
) -> None:
    """Run the Value Stream MCP server over stdio for one workspace.

    ``render_dir`` is where interactive chart HTML is written. It defaults to a
    ``valuestream-renders`` folder under the system temp directory; the server
    never writes to the workspace itself.
    """

    server_cls = _server_class()

    workspace = Path(workspace_path).resolve()
    renders = Path(render_dir) if render_dir else Path(tempfile.gettempdir()) / "valuestream-renders"
    catalog_cache = CatalogCache(workspace)
    catalog_cache.get()  # fail fast if the catalog is invalid at startup
    mcp = server_cls("Value Stream")
    logger.info(
        "Starting Value Stream MCP server: workspace=%s render_dir=%s png=%s",
        workspace,
        renders,
        png_available(),
    )

    @mcp.tool()
    def metric_list(
        search: str | None = None,
        processor: str | None = None,
        dashboard: str | None = None,
        detail: str = "compact",
        limit: int = 60,
    ) -> dict[str, Any]:
        """List metrics, dimensions, query time axes, and supported chart kinds.

        A large catalog has hundreds of metrics, so narrow the list: `search`
        matches metric names and descriptions, `processor` and `dashboard`
        restrict to one processor or to the metrics a dashboard actually uses.
        `detail="full"` adds each metric's prose and configuration block and is
        much larger — ask for it only for a named metric or two.
        """

        logger.info(
            "MCP metric_list: search=%s processor=%s dashboard=%s detail=%s limit=%s",
            search,
            processor,
            dashboard,
            detail,
            limit,
        )

        def run() -> dict[str, Any]:
            catalog = catalog_cache.get()
            names = _select_metric_names(
                catalog,
                search=search,
                processor=processor,
                dashboard=dashboard,
            )
            capped = names[: max(1, min(int(limit), 500))]
            manifest = catalog_chat_manifest(
                catalog,
                detail=detail,
                metric_names=capped,
            )
            manifest["matched_metrics"] = len(names)
            manifest["returned_metrics"] = len(capped)
            if len(capped) < len(names):
                manifest["truncated"] = True
            return manifest

        return _guarded("metric_list", run)

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
        include_curves: bool = False,
        provenance: str = "summary",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query metric rows through the governed aggregate query layer.

        Filter values may be scalars, lists, or operator objects such as
        {"op": ">=", "value": 3} or {"op": "not_in", "values": [...]}.
        `having` applies the same operator objects to metric output columns
        after aggregation. `order_by` accepts column names with an optional
        "-" prefix for descending. `top_n` keeps the largest rows by
        `top_n_by`. `compare="prior_period"` adds *_prev/*_delta/*_pct_change
        columns over the time axis. Set `include_curves` only when you need
        the full roc/pr point arrays — they are large. Set
        `provenance="full"` to include every contributing chunk id. Use
        metric_chart_query for chart requests so chart parameters are explicit.
        """

        logger.info(
            "MCP metric_query: metric=%s grain=%s group_by=%s filters=%s start=%s end=%s "
            "having=%s order_by=%s top_n=%s compare=%s curves=%s limit=%s",
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
            include_curves,
            limit,
        )

        def run() -> dict[str, Any]:
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
                include_curve_columns=include_curves,
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
                "columns": clipped.columns,
                "provenance": _provenance_payload(result.provenance, provenance),
                "rows": clipped.to_dicts(),
            }

        return _guarded("metric_query", run)

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
        chart_fields: dict[str, Any] | None = None,
        render: str = "none",
        limit: int = 100,
    ) -> Any:
        """Query metric rows and return an explicit validated chart spec.

        The model must provide chart_kind, x, y, group_by, color, and facet_col
        explicitly. Use null for optional color/facet_col values. Filter and
        having values may be operator objects such as {"op": ">=", "value": 3}.
        chart_kind must be one of the metric's chart_kinds from metric_list,
        which is rejected with the available list rather than substituted;
        value_format may be percent/integer/number/currency. When a valid kind
        needs an axis or colour adjusted, the change is reported in `warnings`
        — compare `chart` with `requested` before describing the result.

        Kinds beyond the plain x/y set take their extra inputs in
        `chart_fields`: funnel needs `stages`, combo needs `secondary_metric`
        (and optionally `primary_mark`, `shared_y_axis`), interval needs
        `lower_output` and `upper_output`, treemap and sankey need a `path` of
        at least two dimensions, boxplot and histogram need `property`,
        bar_polar needs `theta`. `facet_row`, `line_dash`, `goal_line`,
        `x_axis_title`, and `y_axis_title` work for any kind. Missing fields
        are named in the error.

        Set `render` to see the chart itself: "png" attaches the rendered
        image, "html" writes a self-contained interactive page and returns its
        path, "html_cdn" writes the lighter CDN-backed variant, "spec" returns
        the Plotly figure spec.
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

        def run() -> Any:
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
                chart_fields=chart_fields,
                limit=limit,
            )
            warnings = chart_spec_warnings(
                intent,
                catalog,
                chart_kind=chart_kind,
                x=x,
                y=y,
                group_by=group_by,
                color=color,
                facet_col=facet_col,
                value_format=value_format,
                grain=grain,
                compare=compare,
            )
            result = execute_chat_intent(workspace, catalog, intent)
            rows, masked = _without_state_blobs(result.rows)
            chart = result.intent.chart
            chart_spec: dict[str, Any] = {
                "kind": chart.kind if chart else chart_kind,
                "x": chart.x if chart else x,
                "y": chart.y if chart else y,
                "color": chart.color if chart else color,
                "facet_col": chart.facet_col if chart else facet_col,
                "value_format": chart.value_format if chart else value_format,
            }
            if chart is not None:
                chart_spec.update(_resolved_chart_fields(chart))
            logger.info(
                "MCP metric_chart_query completed: metric=%s grain=%s rows=%s chart=%s warnings=%s",
                result.intent.metric,
                result.intent.grain,
                result.rows.height,
                chart_spec,
                len(warnings),
            )
            payload = {
                "metric": result.intent.metric,
                "grain": result.intent.grain,
                "group_by": result.intent.group_by,
                "filters": result.intent.filters,
                "row_count": result.rows.height,
                "columns": rows.columns,
                "masked_columns": masked,
                "chart": chart_spec,
                "requested": {
                    "kind": chart_kind,
                    "x": x,
                    "y": y,
                    "color": color,
                    "facet_col": facet_col,
                    "value_format": value_format,
                    "chart_fields": chart_fields or {},
                },
                "warnings": warnings,
                "query": result.query_summary,
                "freshness": result.freshness,
                "rows": rows.to_dicts(),
                "rendering_instruction": (
                    "Render using chart.x, chart.y, chart.color, and chart.facet_col exactly as "
                    "returned. When warnings is non-empty the spec differs from what was "
                    "requested; say so rather than describing the requested chart."
                ),
            }
            return _with_render(
                payload,
                rows=rows,
                tile=chart_tile_from_intent(result.intent),
                render=render,
                stem=f"{result.intent.metric}-{chart_spec['kind']}",
                renders=renders,
            )

        return _guarded("metric_chart_query", run)

    @mcp.tool()
    def dashboard_list(
        dashboard_id: str | None = None,
        include_tiles: bool = True,
    ) -> dict[str, Any]:
        """List authored dashboards, pages, page filters, and tiles.

        Use this before answering a question about "the report" or a named
        chart: it returns the tile ids that tile_query and kpi_query take,
        each tile's chart kind and chart fields, and the filter columns the
        page exposes. Pass dashboard_id to fetch one dashboard, or
        include_tiles=false for a compact index of pages only.
        """

        logger.info(
            "MCP dashboard_list: dashboard_id=%s include_tiles=%s", dashboard_id, include_tiles
        )
        return _guarded(
            "dashboard_list",
            lambda: dashboard_manifest(
                catalog_cache.get(),
                dashboard_id=dashboard_id,
                include_tiles=include_tiles,
            ),
        )

    @mcp.tool()
    def tile_query(
        dashboard_id: str,
        page_id: str,
        tile_id: str,
        *,
        filters: dict[str, Any] | None = None,
        start: str | None = None,
        end: str | None = None,
        render: str = "none",
        limit: int = 100,
    ) -> Any:
        """Return the rows behind one authored dashboard tile.

        This runs the tile exactly as the report page runs it, including its
        authored filters, inferred grain and group-by, combo secondary metric
        and histogram property remap, so the numbers match what a user sees.
        Page filters are applied where the tile supports them and reported in
        `ignored_filters` where it does not. Get ids from dashboard_list.

        Set `render` to see the chart, not just its rows: "png" attaches the
        rendered image, "html" writes a self-contained interactive page and
        returns its path, "html_cdn" writes the same page with Plotly loaded
        from its CDN instead (about 13 KB rather than 4 MB, but it needs a
        network), "spec" returns the Plotly figure spec. PNG needs the optional
        `viz` extra; the error says so when it is missing.
        """

        logger.info(
            "MCP tile_query: dashboard=%s page=%s tile=%s filters=%s start=%s end=%s",
            dashboard_id,
            page_id,
            tile_id,
            list((filters or {}).keys()),
            start,
            end,
        )

        def run() -> Any:
            catalog = catalog_cache.get()
            _dashboard, _page, tile = resolve_tile(
                catalog,
                dashboard_id=dashboard_id,
                page_id=page_id,
                tile_id=tile_id,
            )
            applied, ignored = _partition_tile_filters(catalog, tile, filters)
            rows = query_tile(
                workspace,
                catalog,
                tile,
                filters=applied,
                start=_as_date(start, "start"),
                end=_as_date(end, "end"),
            )
            clipped, masked = _without_state_blobs(rows.head(max(1, min(int(limit), 500))))
            tile_dict = tile_to_dict(tile)
            payload = {
                "dashboard_id": dashboard_id,
                "page_id": page_id,
                "tile_id": tile_id,
                "title": tile.title,
                "metric": tile.metric,
                "chart": tile.chart,
                "chart_fields": {
                    key: value
                    for key, value in tile_dict.items()
                    if key
                    in {
                        "x",
                        "y",
                        "color",
                        "facet_col",
                        "facet_row",
                        "line_dash",
                        "stages",
                        "secondary_metric",
                        "metric_output",
                        "lower_output",
                        "upper_output",
                        "value_format",
                        "goal_line",
                    }
                },
                "grain": grain_for_tile(tile_dict),
                "applied_filters": applied,
                "ignored_filters": list(ignored),
                "row_count": rows.height,
                "columns": clipped.columns,
                "masked_columns": masked,
                "rows": clipped.to_dicts(),
                "freshness": freshness_label(
                    metric_freshness(
                        workspace, catalog, tile.metric, grain=grain_for_tile(tile_dict)
                    )
                ),
            }
            return _with_render(
                payload,
                rows=rows,
                tile=tile_dict,
                render=render,
                stem=f"{dashboard_id}-{tile_id}",
                renders=renders,
            )

        return _guarded("tile_query", run)

    @mcp.tool()
    def kpi_query(
        dashboard_id: str,
        page_id: str,
        tile_id: str,
        *,
        filters: dict[str, Any] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Return a KPI card's value, comparison delta, and sparkline.

        A KPI tile is more than a metric query: it resolves the tile's output
        column and derives the comparison window from the authored kpi spec.
        Use this instead of metric_query when a user asks what a KPI card
        shows, so the period and delta match the report.
        """

        logger.info(
            "MCP kpi_query: dashboard=%s page=%s tile=%s", dashboard_id, page_id, tile_id
        )

        def run() -> dict[str, Any]:
            catalog = catalog_cache.get()
            _dashboard, _page, tile = resolve_tile(
                catalog,
                dashboard_id=dashboard_id,
                page_id=page_id,
                tile_id=tile_id,
            )
            applied, ignored = _partition_tile_filters(catalog, tile, filters)
            bundle = kpi_bundle(
                workspace,
                catalog,
                tile,
                filters=applied,
                start=_as_date(start, "start"),
                end=_as_date(end, "end"),
            )
            return {
                "dashboard_id": dashboard_id,
                "page_id": page_id,
                "tile_id": tile_id,
                "title": tile.title,
                "metric": tile.metric,
                "value_column": bundle.value_column,
                "value": bundle.value,
                "delta": bundle.delta,
                "delta_description": bundle.delta_description,
                "period": bundle.period_description,
                "sparkline": list(bundle.sparkline) if bundle.sparkline else None,
                "applied_filters": applied,
                "ignored_filters": list(ignored),
            }

        return _guarded("kpi_query", run)

    @mcp.tool()
    def chat(question: str, narrate: bool = False) -> dict[str, Any]:
        """Answer a natural-language question by planning a governed query.

        This is the same planner the app's Chat page and the HTTP API use: it
        picks the metric, dimensions, grain, and chart, then runs the query.
        It needs an LLM model configured in the workspace ai.yaml. Prefer the
        explicit tools when you already know the metric or tile.
        """

        logger.info("MCP chat: question_length=%s narrate=%s", len(question), narrate)

        def run() -> dict[str, Any]:
            settings = _settings_from_workspace(workspace)
            if settings is None:
                raise ValueError(
                    "no LLM model is configured for this workspace; set ai.llm.model in its "
                    "ai.yaml, or use metric_query/tile_query directly"
                )
            catalog = catalog_cache.get()
            _config_path, chat_config = load_chat_with_data_config(workspace)
            intent, _raw = plan_chat_intent(
                settings,
                catalog,
                question,
                chat_config=chat_config,
            )
            if intent.response == "clarify":
                return {"response": "clarify", "clarify": intent.clarify}
            if intent.response == "sql":
                raise ValueError(
                    "the planner asked for governed SQL; call sql_query directly instead"
                )
            result = execute_chat_intent(workspace, catalog, intent)
            rows, masked = _without_state_blobs(result.rows.head(100))
            payload: dict[str, Any] = {
                "response": intent.response,
                "question": question,
                "metric": intent.metric,
                "grain": intent.grain,
                "group_by": intent.group_by,
                "filters": intent.filters,
                "query": result.query_summary,
                "freshness": result.freshness,
                "row_count": result.rows.height,
                "columns": rows.columns,
                "masked_columns": masked,
                "rows": rows.to_dicts(),
            }
            if intent.chart is not None:
                payload["chart"] = {
                    "kind": intent.chart.kind,
                    "x": intent.chart.x,
                    "y": intent.chart.y,
                    "color": intent.chart.color,
                    "facet_col": intent.chart.facet_col,
                    "value_format": intent.chart.value_format,
                }
            if narrate and not result.rows.is_empty():
                payload["narrative"] = narrate_chat_result(settings, result)
            return payload

        return _guarded("chat", run)

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
        return _guarded(
            "dimension_values_tool",
            lambda: {
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
            },
        )

    if enable_sql:

        @mcp.tool()
        def sql_schema() -> dict[str, Any]:
            """List governed DuckDB tables/views available to sql_query.

            Names are fully qualified; sketch/state blob columns are hidden.
            """

            logger.info("MCP sql_schema: workspace=%s", workspace)

            def run() -> dict[str, Any]:
                tables = list_sql_tables(workspace, catalog_cache.get())
                return {
                    "tables": [
                        {
                            "name": table.name,
                            "kind": table.kind,
                            "columns": [
                                {"name": name, "type": dtype} for name, dtype in table.columns
                            ],
                        }
                        for table in tables
                    ],
                    "notes": [
                        "Only single read-only SELECT (or WITH ... SELECT) statements are "
                        "accepted.",
                        "Use the fully qualified table names exactly as listed.",
                        "Row counts are capped; sketch state columns are masked from results.",
                    ],
                }

            return _guarded("sql_schema", run)

        @mcp.tool()
        def sql_query(sql: str, limit: int = 200) -> dict[str, Any]:
            """Run one governed read-only SELECT over the aggregate DuckDB views.

            Use sql_schema first to discover table names and columns. DDL/DML,
            multiple statements, comments, and file/catalog functions are rejected.
            """

            logger.info("MCP sql_query: workspace=%s limit=%s", workspace, limit)

            def run() -> dict[str, Any]:
                result = run_sql_query(workspace, sql, catalog=catalog_cache.get(), limit=limit)
                return {
                    "sql": result.sql,
                    "row_count": result.row_count,
                    "truncated": result.truncated,
                    "masked_columns": result.masked_columns,
                    "columns": result.rows.columns,
                    "rows": result.rows.to_dicts(),
                }

            return _guarded("sql_query", run)

    @mcp.tool()
    def workspace_status_tool(run_limit: int = 5) -> dict[str, Any]:
        """Report which processors can answer a query, and why others cannot.

        Call this first when a metric or tile fails with an aggregate error,
        or before trusting a whole workspace: it reports per-processor
        readiness (ready / stale / unpublished / missing), recent pipeline
        runs, and runs that never finished. Readiness comes from the same load
        a query performs, so it agrees with what the other tools will do.
        """

        logger.info("MCP workspace_status: workspace=%s run_limit=%s", workspace, run_limit)
        return _guarded(
            "workspace_status",
            lambda: workspace_status(
                workspace,
                catalog_cache.get(),
                run_limit=max(1, min(int(run_limit), 50)),
            ),
        )

    @mcp.tool()
    def freshness_get(metric: str, grain: str = "summary") -> dict[str, Any]:
        """Return freshness metadata for a metric and grain."""

        logger.info("MCP freshness_get: metric=%s grain=%s", metric, grain)

        def run() -> dict[str, Any]:
            fresh = metric_freshness(workspace, catalog_cache.get(), metric, grain=grain)
            return {
                "metric": metric,
                "grain": grain,
                "latest_period": fresh.latest_period,
                "last_created_at": fresh.last_created_at.isoformat()
                if fresh.last_created_at
                else None,
                "last_run_finished_at": fresh.last_run_finished_at.isoformat()
                if fresh.last_run_finished_at
                else None,
                "status": fresh.status,
                "label": freshness_label(fresh),
            }

        return _guarded("freshness_get", run)

    @mcp.resource(
        "valuestream://catalog",
        name="Workspace catalog",
        description="The workspace's pipelines, processors, metrics, and dashboards as YAML.",
        mime_type="text/yaml",
    )
    def catalog_resource() -> str:
        """Return the catalog YAML files concatenated, newest read from disk."""

        parts: list[str] = []
        for name in ("pipelines", "processors", "metrics", "dashboards"):
            path = workspace / "catalog" / f"{name}.yaml"
            if path.exists():
                parts.append(f"# ---- {name}.yaml ----\n{path.read_text()}")
        return "\n\n".join(parts)

    @mcp.resource(
        "valuestream://dashboards",
        name="Dashboards",
        description="Authored dashboards, pages, and tiles with their chart specs.",
        mime_type="application/json",
    )
    def dashboards_resource() -> str:
        """Return the dashboard manifest as JSON."""

        return json.dumps(dashboard_manifest(catalog_cache.get()), indent=1, default=str)

    @mcp.prompt(
        name="explore_workspace",
        description="Open-ended starting point for exploring what this workspace measures.",
    )
    def explore_workspace_prompt() -> str:
        """Return an orientation prompt grounded in this workspace's catalog."""

        catalog = catalog_cache.get()
        dashboards = ", ".join(item.id for item in catalog.dashboards.dashboards) or "none"
        return (
            f"This Value Stream workspace has {len(catalog.metrics.metrics)} metrics across "
            f"these dashboards: {dashboards}. Start with dashboard_list to see the authored "
            "pages and tiles, then use tile_query or kpi_query to read a specific chart, or "
            "metric_list(search=...) to find a metric by name. Report the freshness that comes "
            "back with the numbers."
        )

    @mcp.prompt(
        name="starter_questions",
        description="Catalog-grounded questions this workspace can answer without an LLM plan.",
    )
    def starter_questions_prompt() -> str:
        """Return deterministic starter questions with the tool call for each."""

        catalog = catalog_cache.get()
        lines = ["Questions this workspace can answer, with the call that answers each:"]
        for starter in deterministic_chat_starters(catalog):
            lines.append(
                f"- {starter.question}  ->  metric_query(metric={starter.intent.metric!r}, "
                f"grain={starter.intent.grain!r})"
            )
        return "\n".join(lines)

    mcp.run()


_EXTENDED_CHART_FIELDS = (
    "facet_row",
    "line_dash",
    "stages",
    "path",
    "secondary_metric",
    "primary_mark",
    "shared_y_axis",
    "lower_output",
    "upper_output",
    "property_name",
    "theta",
    "goal_line",
    "x_axis_title",
    "y_axis_title",
)


def _with_render(
    payload: dict[str, Any],
    *,
    rows: pl.DataFrame,
    tile: dict[str, Any],
    render: str,
    stem: str,
    renders: Path,
) -> Any:
    """Attach a rendered chart to a tool payload, in the form asked for.

    A PNG comes back as an image block the caller can actually look at. HTML is
    written to a file and referenced by path instead: it is interactive, which
    only helps in a browser, and inlining the markup would spend the response
    on something the model cannot render.
    """

    mode = str(render or "none").strip().lower()
    if mode in ("", "none"):
        return payload
    if mode not in RENDER_MODES:
        raise ValueError(f"render must be one of {', '.join(RENDER_MODES)}, got {render!r}")
    if rows.is_empty():
        payload["render"] = {"mode": mode, "skipped": "the query returned no rows to plot"}
        return payload

    figure = figure_for_tile(rows, tile)
    if mode == "spec":
        payload["render"] = {"mode": "spec", "spec": figure_spec(figure)}
        return payload
    if mode in {"html", "html_cdn"}:
        embed = mode == "html"
        path = write_render(
            figure_html(figure, title=str(tile.get("title") or stem), embed_plotlyjs=embed),
            directory=renders,
            stem=stem,
            suffix="html",
        )
        payload["render"] = {
            "mode": mode,
            "path": str(path),
            "bytes": path.stat().st_size,
            "self_contained": embed,
            "note": (
                "Open this file in a browser. Plotly's JavaScript is embedded, so it works "
                "offline."
                if embed
                else "Open this file in a browser; Plotly's JavaScript loads from its CDN, so "
                "it needs a network connection."
            ),
        }
        return payload

    image = figure_png(figure)
    payload["render"] = {"mode": "png", "bytes": len(image), "attached": True}
    return [payload, _image_class()(data=image, format="png")]


def _resolved_chart_fields(chart: Any) -> dict[str, Any]:
    """Echo the kind-specific fields the validated chart actually kept."""

    resolved: dict[str, Any] = {}
    for name in _EXTENDED_CHART_FIELDS:
        value = getattr(chart, name, None)
        if value in (None, "", (), [], False):
            continue
        resolved["property" if name == "property_name" else name] = (
            list(value) if isinstance(value, tuple) else value
        )
    return resolved


def _select_metric_names(
    catalog: Any,
    *,
    search: str | None,
    processor: str | None,
    dashboard: str | None,
) -> list[str]:
    """Return catalog metric names narrowed by the metric_list filters."""

    names = sorted(catalog.metrics.metrics)
    if dashboard is not None:
        matched = next(
            (item for item in catalog.dashboards.dashboards if item.id == dashboard), None
        )
        if matched is None:
            known = ", ".join(item.id for item in catalog.dashboards.dashboards)
            raise ValueError(f"unknown dashboard {dashboard!r}; available: {known}")
        used = {tile.metric for page in matched.pages for tile in page.tiles}
        names = [name for name in names if name in used]
    if processor is not None:
        names = [
            name for name in names if catalog.metrics.metrics[name].processor == processor
        ]
    if search:
        needle = search.casefold()
        names = [
            name
            for name in names
            if needle in name.casefold()
            or needle in str(catalog.metrics.metrics[name].description or "").casefold()
        ]
    return names


def _settings_from_workspace(workspace: Path) -> AICallSettings | None:
    """Build LLM call settings from the workspace ai.yaml, or None if unset."""

    _path, config = load_llm_settings_config(workspace)
    model_name = str(config.get("model") or "").strip()
    if not model_name:
        return None
    temperature = config.get("temperature")
    return AICallSettings(
        model=model_name,
        api_key=configured_api_key(config),
        api_base=str(config.get("api_base") or ""),
        custom_llm_provider=str(
            config.get("custom_provider") or config.get("custom_llm_provider") or ""
        ),
        temperature=float(temperature) if temperature is not None else None,
        reasoning_effort=str(config.get("reasoning_effort") or ""),
        verbosity=str(config.get("verbosity") or ""),
        timeout_seconds=int(config.get("timeout_seconds") or 90),
    )


def _without_state_blobs(rows: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Drop sketch/state blob columns before a tile's rows are serialized.

    Stateful charts (funnel, boxplot, histogram, descriptive) ask the query
    layer for state columns so the UI can render from the sketches. Those are
    raw binary and must not cross the tool boundary — the governed SQL path
    masks them for the same reason, and JSON cannot carry them regardless.
    """

    masked = [
        name for name, dtype in zip(rows.columns, rows.dtypes, strict=True) if dtype == pl.Binary
    ]
    # config_hash is lineage bookkeeping, already reported in provenance.
    internal = [name for name in ("config_hash",) if name in rows.columns]
    dropped = [*masked, *internal]
    return (rows.drop(dropped) if dropped else rows), masked


def _as_date(value: str | None, label: str) -> dt.date | None:
    """Parse an ISO date bound, naming the field when it is malformed."""

    if value in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}") from exc


def _partition_tile_filters(
    catalog: Any,
    tile: Any,
    filters: dict[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Split requested filters into ones this tile supports and ones it cannot."""

    supported = available_filter_columns_for_tile(catalog, tile)
    applied: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in (filters or {}).items():
        if str(key) not in supported:
            ignored.append(str(key))
        elif value not in (None, "", []):
            applied[str(key)] = value
    return applied, tuple(ignored)


def _provenance_payload(provenance: Any, mode: str) -> dict[str, Any]:
    """Return the provenance envelope, dropping chunk ids unless asked for.

    A workspace contributes hundreds of chunk ids to every query, which costs
    more response budget than the rows for a one-row answer, so the id lists
    are opt-in.
    """

    payload = dict(provenance.to_dict())
    if str(mode).strip().lower() == "full":
        return payload
    for key in ("chunk_ids", "pipeline_run_ids"):
        ids = payload.pop(key, None) or []
        payload[f"{key}_count"] = len(ids)
    return payload


__all__ = ["run_stdio"]
