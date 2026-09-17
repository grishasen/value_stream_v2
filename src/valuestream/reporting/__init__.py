"""Report-shaped query services shared by the UI, HTTP API, and MCP server."""

from valuestream.reporting.kpi import KpiBundle, kpi_bundle
from valuestream.reporting.manifest import dashboard_manifest, resolve_tile
from valuestream.reporting.render import figure_html, figure_png, figure_spec, png_available
from valuestream.reporting.status import workspace_status
from valuestream.reporting.tiles import query_tile, tile_to_dict

__all__ = [
    "KpiBundle",
    "dashboard_manifest",
    "figure_html",
    "figure_png",
    "figure_spec",
    "kpi_bundle",
    "png_available",
    "query_tile",
    "resolve_tile",
    "tile_to_dict",
    "workspace_status",
]
