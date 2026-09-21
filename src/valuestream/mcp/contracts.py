"""Public response schemas for the MCP query tools (no optional SDK imports)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolPayload(BaseModel):
    """Keep catalog-dependent fields while validating the common contract."""

    model_config = ConfigDict(extra="allow")


class MetricListPayload(ToolPayload):
    metrics: list[dict[str, Any]]
    matched_metrics: int
    returned_metrics: int
    offset: int
    next_offset: int | None
    truncated: bool


class RowPayload(ToolPayload):
    row_count: int
    returned_rows: int
    offset: int
    next_offset: int | None
    truncated: bool
    columns: list[str]
    rows: list[dict[str, Any]]


class MetricPayload(RowPayload):
    metric: str
    grain: str
    provenance: dict[str, Any]


class ChartPayload(RowPayload):
    metric: str
    chart: dict[str, Any]
    warnings: list[str]
    freshness: str
    masked_columns: list[str]


class TilePayload(RowPayload):
    dashboard_id: str
    page_id: str
    tile_id: str
    chart: str
    applied_filters: dict[str, Any]
    ignored_filters: list[str]
    overridden_filters: dict[str, Any]
    provenance: list[dict[str, Any]]
    freshness: str


class KpiPayload(ToolPayload):
    dashboard_id: str
    page_id: str
    tile_id: str
    value: float | int | str
    value_column: str
    delta: float | int | None
    period: str
    applied_filters: dict[str, Any]
    ignored_filters: list[str]
    overridden_filters: dict[str, Any]
    provenance: list[dict[str, Any]]
    freshness: str
