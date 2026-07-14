"""Shared chunk context and provenance column names for processors.

Kept dependency-free so both ``processors_helper`` and the individual
processors can import it without creating an import cycle.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

PROVENANCE_COLUMNS = ["pipeline_run_id", "chunk_id", "period", "created_at", "config_hash"]


@dataclass(frozen=True)
class ChunkContext:
    """Context added to every aggregate row."""

    pipeline_run_id: str
    chunk_id: str
    created_at: dt.datetime


__all__ = ["PROVENANCE_COLUMNS", "ChunkContext"]
