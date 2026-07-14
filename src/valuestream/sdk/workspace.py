"""Small Phase 1 Python SDK."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from valuestream.engine import PipelineRunResult, WorkspaceRunResult, run_source, run_workspace
from valuestream.query import MetricQueryResult, query_metric, query_metric_result


class Workspace:
    """Convenience wrapper around a Value Stream workspace directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def run_source(self, source_id: str, *, force: bool = False) -> PipelineRunResult:
        """Run ingestion for one source."""
        return run_source(self.path, source_id, force=force)

    def run_all(self, *, force: bool = False) -> WorkspaceRunResult:
        """Run ingestion for every source in the workspace."""
        return run_workspace(self.path, force=force)

    def metric(self, name: str) -> MetricQuery:
        """Start a fluent metric query."""
        return MetricQuery(self, name)


class MetricQuery:
    """Fluent query builder used by ``Workspace.metric(...)``."""

    def __init__(self, workspace: Workspace, name: str) -> None:
        self._workspace = workspace
        self._name = name
        self._group_by: list[str] = []
        self._filters: dict[str, Any] = {}
        self._grain = "daily"
        self._start: dt.date | dt.datetime | str | None = None
        self._end: dt.date | dt.datetime | str | None = None
        self._include_state_columns = False

    def by(self, *columns: str) -> MetricQuery:
        self._group_by.extend(columns)
        return self

    def where(self, **filters: Any) -> MetricQuery:
        self._filters.update(filters)
        return self

    def grain(self, grain: str) -> MetricQuery:
        self._grain = grain
        return self

    def between(
        self,
        start: dt.date | dt.datetime | str | None,
        end: dt.date | dt.datetime | str | None,
    ) -> MetricQuery:
        """Restrict a metric query to an inclusive date range."""
        self._start = start
        self._end = end
        return self

    def raw(self, enabled: bool = True) -> MetricQuery:
        """Include underlying aggregate state columns in the query result."""
        self._include_state_columns = enabled
        return self

    def to_polars(self) -> pl.DataFrame:
        """Execute and return a Polars DataFrame."""
        return query_metric(
            self._workspace.path,
            self._name,
            group_by=self._group_by,
            filters=self._filters,
            grain=self._grain,
            start=self._start,
            end=self._end,
            include_state_columns=self._include_state_columns,
        )

    def to_result(self) -> MetricQueryResult:
        """Execute and return rows together with aggregate/config provenance."""
        return query_metric_result(
            self._workspace.path,
            self._name,
            group_by=self._group_by,
            filters=self._filters,
            grain=self._grain,
            start=self._start,
            end=self._end,
            include_state_columns=self._include_state_columns,
        )


__all__ = ["Workspace"]
