"""Home page for the Value Stream Streamlit app."""

from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st

from valuestream.ui import components
from valuestream.ui.context import ValueStreamContext, catalog_counts, recent_runs_frame


def render(ctx: ValueStreamContext) -> None:
    """Render the app home page."""
    components.render_page_header(
        "Value Stream Dashboard",
        "Monitor workspace health, report coverage, and pipeline activity from one place.",
        status="ok" if ctx.validation.ok else "warning",
        status_label="Catalog OK" if ctx.validation.ok else "Needs attention",
    )

    _render_kpi_cards(_dashboard_kpis(ctx))

    activity_col, health_col = st.columns([0.62, 0.38], gap="large")
    with (
        activity_col,
        components.bordered_panel(
            "Recent Runs",
            "Latest source activity and rows kept for this workspace.",
        ),
    ):
        runs = recent_runs_frame(ctx, limit=8)
        if runs.is_empty():
            components.status_badge("No runs recorded", "pending")
        else:
            display_runs = _recent_runs_display(runs)
            components.dataframe_with_search(
                display_runs,
                key="home_recent_runs",
                height="auto",
                static=True,
                column_order=display_runs.columns,
                column_config={
                    "Run": st.column_config.TextColumn("Run", width="small"),
                    "Source": st.column_config.TextColumn("Source", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Rows kept": st.column_config.TextColumn("Rows kept", width="medium"),
                    "Finished": st.column_config.DatetimeColumn(
                        "Finished", format="YYYY-MM-DD HH:mm", width="medium"
                    ),
                },
            )

    with health_col:
        _render_workspace_summary(ctx)
        _render_workflow_cards(ctx)


def _dashboard_kpis(ctx: ValueStreamContext) -> list[dict[str, Any]]:
    counts = catalog_counts(ctx)
    dashboard_pages, dashboard_tiles = _dashboard_inventory(ctx)
    latest = recent_runs_frame(ctx, limit=1)
    latest_row = latest.row(0, named=True) if not latest.is_empty() else {}

    latest_status = str(latest_row.get("status") or "not run")
    latest_finished = components.format_timestamp(latest_row.get("finished_at"))
    latest_rows = components.format_compact_number(latest_row.get("rows_kept"))

    return [
        {
            "label": "Sources",
            "value": counts["Sources"],
            "delta": f"{counts['Processors']} processor(s)",
            "help": "Configured data sources in the active workspace.",
        },
        {
            "label": "Metrics",
            "value": counts["Metrics"],
            "delta": f"{dashboard_tiles} report tile(s)",
            "help": "Configured semantic metrics available to reports and chat.",
        },
        {
            "label": "Report Pages",
            "value": dashboard_pages,
            "delta": f"{counts['Dashboards']} dashboard(s)",
            "help": "Configured report pages across all dashboards.",
        },
        {
            "label": "Last Run",
            "value": latest_status,
            "delta": latest_finished,
            "help": "Most recent pipeline run status.",
        },
        {
            "label": "Rows Kept",
            "value": latest_rows,
            "delta": "latest run",
            "help": "Rows retained by the latest pipeline run.",
        },
    ]


def _render_kpi_cards(items: list[dict[str, Any]]) -> None:
    components.metric_cards(items, columns=len(items), key="home_summary")


def _render_workspace_summary(ctx: ValueStreamContext) -> None:
    issue_counts = _validation_issue_counts(ctx)
    with components.bordered_panel(
        "Workspace",
        "Active workspace and catalog revision.",
    ):
        st.write(f"**{ctx.catalog.pipelines.workspace}**")
        st.caption(f"Catalog revision `{ctx.catalog_hash}`")
        if issue_counts["errors"] or issue_counts["warnings"]:
            st.warning(f"{issue_counts['errors']} error(s) · {issue_counts['warnings']} warning(s)")
        else:
            st.caption("No catalog issues.")


def _render_workflow_cards(ctx: ValueStreamContext) -> None:
    counts = catalog_counts(ctx)
    cards = [
        (
            "Reports",
            "Explore configured dashboard pages and KPI tiles.",
            "ready" if counts["Dashboards"] else "pending",
        ),
        (
            "Chat With Data",
            "Ask questions over aggregate metrics.",
            "ready" if counts["Metrics"] else "pending",
        ),
        (
            "Data Integration",
            "Load source files and inspect pipeline runs.",
            "ready" if counts["Sources"] else "pending",
        ),
        (
            "Settings",
            "Review catalog, config builders, and AI-assisted drafts.",
            "ok" if ctx.validation.ok else "warning",
        ),
    ]
    with components.bordered_panel(
        "Workspace Flow",
        "Primary areas in the sidebar navigation.",
    ):
        for title, description, status in cards:
            top = st.columns([0.62, 0.38], vertical_alignment="center")
            top[0].write(f"**{title}**")
            with top[1]:
                components.status_badge("OK" if status == "ok" else status.title(), status)
            st.caption(description)


def _recent_runs_display(runs: pl.DataFrame) -> pl.DataFrame:
    """Prioritize operational run fields and keep the UUID as a short reference."""
    expressions: list[pl.Expr] = []
    if "source_id" in runs.columns:
        expressions.append(pl.col("source_id").alias("Source"))
    if "status" in runs.columns:
        expressions.append(pl.col("status").str.to_titlecase().alias("Status"))
    if "rows_kept" in runs.columns:
        expressions.append(
            pl.col("rows_kept")
            .map_elements(components.format_count, return_dtype=pl.String)
            .alias("Rows kept")
        )
    if "finished_at" in runs.columns:
        expressions.append(pl.col("finished_at").alias("Finished"))
    if "id" in runs.columns:
        expressions.append(pl.col("id").cast(pl.String).str.slice(0, 8).alias("Run"))
    return runs.select(expressions) if expressions else runs


def _dashboard_inventory(ctx: ValueStreamContext) -> tuple[int, int]:
    page_count = 0
    tile_count = 0
    for dashboard in ctx.catalog.dashboards.dashboards:
        page_count += len(dashboard.pages)
        tile_count += sum(len(page.tiles) for page in dashboard.pages)
    return page_count, tile_count


def _validation_issue_counts(ctx: ValueStreamContext) -> dict[str, int]:
    errors = [issue for issue in ctx.validation.issues if issue.severity == "error"]
    warnings = [issue for issue in ctx.validation.issues if issue.severity != "error"]
    return {"errors": len(errors), "warnings": len(warnings)}
