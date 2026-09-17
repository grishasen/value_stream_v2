"""Workspace readiness: which processors can actually answer a query.

A metric fails with "aggregate data is not ready" when the processor's
computation hash no longer matches what is on disk, or when its rows belong to
a run that never published. That diagnosis is only visible in the app's Ops
page today, so a tool client sees an unexplained failure instead of "this
workspace needs a backfill". This module renders the same picture.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from valuestream.config import model
from valuestream.config.canonical import processor_computation_hash
from valuestream.processors import grain_levels
from valuestream.query import aggregate_readiness
from valuestream.ui.freshness import recent_runs

_READY = "ready"
_STALE = "stale"
_UNPUBLISHED = "unpublished"
_MISSING = "missing"

_DETAILS = {
    _READY: "aggregates match the current processor configuration",
    _STALE: (
        "aggregates were written under a different processor configuration; the processor "
        "changed since the last run, so queries fail until the source is backfilled or "
        "reprocessed"
    ),
    _UNPUBLISHED: (
        "aggregate rows are registered for the current configuration but were never "
        "published, which usually means a run did not finish; re-run ingestion for this source"
    ),
    _MISSING: "no aggregates have been written for this processor yet; run ingestion first",
}


@dataclass(frozen=True)
class ProcessorStatus:
    """Whether one processor's aggregates match its current configuration."""

    processor_id: str
    source_id: str
    state: str
    grain: str
    current_hash: str
    detail: str


def workspace_status(
    workspace_path: str | Path,
    catalog: model.Catalog,
    *,
    run_limit: int = 5,
) -> dict[str, Any]:
    """Return run history plus per-processor aggregate readiness."""

    workspace = Path(workspace_path)
    statuses = [
        processor_status(workspace, catalog, processor)
        for processor in catalog.processors.processors
    ]
    counts = {
        state: sum(1 for item in statuses if item.state == state)
        for state in (_READY, _STALE, _UNPUBLISHED, _MISSING)
    }
    runs = _recent_runs(workspace, limit=run_limit)
    unfinished = [run for run in runs if run.get("finished_at") in (None, "")]
    return {
        "workspace": str(workspace),
        "queryable": counts[_READY] > 0,
        "processors_ready": counts[_READY],
        "processors_stale": counts[_STALE],
        "processors_unpublished": counts[_UNPUBLISHED],
        "processors_missing": counts[_MISSING],
        "processors": [
            {
                "id": item.processor_id,
                "source": item.source_id,
                "state": item.state,
                "grain": item.grain,
                "current_config_hash": item.current_hash,
                "detail": item.detail,
            }
            for item in statuses
        ],
        "recent_runs": runs,
        "unfinished_runs": len(unfinished),
        "summary": _summary(counts, unfinished_runs=len(unfinished)),
    }


def processor_status(
    workspace_path: str | Path,
    catalog: model.Catalog,
    processor: model.Processor,
) -> ProcessorStatus:
    """Report whether one processor can answer a query, and why not.

    Readiness comes from the query path itself rather than from the lineage
    table: a run that registered lineage and then died leaves the current
    config hash recorded with no published rows, which lineage alone reads as
    healthy.
    """

    grain = _stored_grain(processor)
    state = aggregate_readiness(workspace_path, catalog, processor, grain="summary")
    return ProcessorStatus(
        processor_id=processor.id,
        source_id=processor.source,
        state=state,
        grain=grain,
        current_hash=processor_computation_hash(catalog, processor),
        detail=_DETAILS.get(state, ""),
    )


def _stored_grain(processor: model.Processor) -> str:
    """Return the physical grain this processor actually materializes."""

    candidates = grain_levels.aggregate_grain_candidates(processor, "summary")
    return str(candidates[0]) if candidates else "daily"


def _recent_runs(workspace_path: str | Path, *, limit: int) -> list[dict[str, Any]]:
    frame = recent_runs(workspace_path, limit=limit)
    if frame.is_empty():
        return []
    return [
        {key: _jsonable(value) for key, value in row.items()} for row in frame.to_dicts()
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _summary(counts: dict[str, int], *, unfinished_runs: int) -> str:
    parts = [
        f"{counts[_READY]} processor(s) ready",
        f"{counts[_STALE]} stale",
        f"{counts[_UNPUBLISHED]} unpublished",
        f"{counts[_MISSING]} with no aggregates",
    ]
    blocked = counts[_STALE] + counts[_UNPUBLISHED] + counts[_MISSING]
    if blocked:
        parts.append(
            f"{blocked} processor(s) cannot be queried until ingestion is re-run or the "
            "source is backfilled"
        )
    if unfinished_runs:
        parts.append(
            f"{unfinished_runs} run(s) have no finish time, so freshness reads as running"
        )
    return "; ".join(parts)


__all__ = ["ProcessorStatus", "processor_status", "workspace_status"]
