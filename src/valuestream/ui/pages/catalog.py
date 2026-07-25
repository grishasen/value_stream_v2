"""Catalog inventory page."""

from __future__ import annotations

import streamlit as st
import yaml

from valuestream.ui import components
from valuestream.ui.context import ValueStreamContext, metrics_for_processor, processors_for_source


def render(ctx: ValueStreamContext) -> None:
    """Render catalog inventory and validation panels."""
    components.render_page_header(
        "Catalog",
        "Search configured sources, processors, metrics, and dashboards.",
        status="ok" if ctx.validation.ok else "warning",
        status_label="Catalog OK" if ctx.validation.ok else "Needs review",
    )

    tabs = st.tabs(["Sources", "Processors", "Metrics", "Dashboards", "Validation"])
    with tabs[0]:
        _sources(ctx)
    with tabs[1]:
        _processors(ctx)
    with tabs[2]:
        _metrics(ctx)
    with tabs[3]:
        _dashboards(ctx)
    with tabs[4]:
        components.render_validation_summary(ctx.validation.issues, ok=ctx.validation.ok)


def _sources(ctx: ValueStreamContext) -> None:
    rows = [
        {
            "id": source.id,
            "reader": source.reader.kind,
            "pattern": source.reader.file_pattern,
            "processors": len(processors_for_source(ctx, source.id)),
            "transforms": len(source.transforms),
            "description": source.description,
        }
        for source in ctx.catalog.pipelines.sources
    ]
    components.dataframe_with_search(rows, key="catalog_sources")


def _processors(ctx: ValueStreamContext) -> None:
    rows = [
        {
            "id": processor.id,
            "source": processor.source,
            "kind": processor.kind,
            "group_by": ", ".join(processor.group_by),
            "grains": ", ".join(processor.grains),
            "states": len(processor.states),
            "metrics": ", ".join(metrics_for_processor(ctx, processor.id)),
        }
        for processor in ctx.catalog.processors.processors
    ]
    components.dataframe_with_search(rows, key="catalog_processors")


def _metrics(ctx: ValueStreamContext) -> None:
    rows = [
        {
            "id": name,
            "source": metric.processor,
            "kind": metric.kind,
            "depends_on": ", ".join(metric.depends_on),
            "description": metric.description,
        }
        for name, metric in ctx.catalog.metrics.metrics.items()
    ]
    components.dataframe_with_search(rows, key="catalog_metrics")


def _dashboards(ctx: ValueStreamContext) -> None:
    rows = []
    for dashboard in ctx.catalog.dashboards.dashboards:
        for page in dashboard.pages:
            rows.append(
                {
                    "dashboard": dashboard.id,
                    "dashboard_title": dashboard.title,
                    "page": page.id,
                    "page_title": page.title,
                    "tiles": len(page.tiles),
                    "tile_ids": ", ".join(tile.id for tile in page.tiles),
                }
            )
    components.dataframe_with_search(rows, key="catalog_dashboards")
    with st.expander("Dashboard YAML", expanded=False):
        st.code(
            yaml.safe_dump(ctx.catalog.dashboards.model_dump(), sort_keys=False), language="yaml"
        )
