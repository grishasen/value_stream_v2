"""Source probe helpers for Phase 1 ingestion setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from valuestream.config import model
from valuestream.config.loader import load
from valuestream.config.validate import validate_catalog
from valuestream.readers import discover, read
from valuestream.transforms import apply_transforms
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class SourceProbe:
    """A lightweight profile of one source after configured transforms."""

    source_id: str
    chunk_count: int
    file_count: int
    schema: tuple[tuple[str, str], ...]
    calendar_columns: tuple[str, ...]
    sample: pl.DataFrame


@timed
def probe_source(
    workspace_path: str | Path,
    source_id: str,
    *,
    limit: int = 10,
) -> SourceProbe:
    """Discover source chunks and sample the first transformed chunk."""
    workspace = Path(workspace_path)
    catalog = load(workspace)
    validation = validate_catalog(catalog)
    if not validation.ok:
        messages = "; ".join(f"{i.location}: {i.message}" for i in validation.issues)
        raise ValueError(f"catalog does not validate: {messages}")
    source = next(
        (candidate for candidate in catalog.pipelines.sources if candidate.id == source_id), None
    )
    if source is None:
        raise ValueError(f"unknown source {source_id!r}")

    chunks = tuple(discover(workspace, source))
    if not chunks:
        return SourceProbe(
            source_id=source_id,
            chunk_count=0,
            file_count=0,
            schema=(),
            calendar_columns=(),
            sample=pl.DataFrame(),
        )

    transformed = apply_transforms(read(source.reader, chunks[0].files), source)
    schema = tuple((name, str(dtype)) for name, dtype in transformed.collect_schema().items())
    sample = transformed.limit(limit).collect()
    schema_names = {name for name, _ in schema}
    return SourceProbe(
        source_id=source_id,
        chunk_count=len(chunks),
        file_count=sum(len(chunk.files) for chunk in chunks),
        schema=schema,
        calendar_columns=tuple(_calendar_outputs(source, schema_names)),
        sample=sample,
    )


def _calendar_outputs(source: model.Source, schema_names: set[str]) -> list[str]:
    columns: list[str] = []
    for transform in source.transforms:
        if isinstance(transform, model.DeriveCalendar):
            columns.extend(output for output in transform.outputs if output in schema_names)
    return columns


__all__ = ["SourceProbe", "probe_source"]
