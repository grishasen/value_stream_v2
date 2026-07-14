"""File discovery and chunk grouping for Phase 1 ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from valuestream.config import model
from valuestream.utils.timer import timed


@dataclass(frozen=True)
class Chunk:
    """One idempotent unit of source work."""

    chunk_id: str
    files: tuple[Path, ...]


@timed
def discover(workspace_path: str | Path, source: model.Source) -> list[Chunk]:
    """Discover files for ``source`` and group them into sorted chunks."""
    root = _reader_root(workspace_path, source)
    pattern = source.reader.file_pattern
    files = sorted(p for p in root.glob(pattern) if p.is_file())

    if not files and source.reader.kind == "pega_ds_export":
        files = sorted(p for p in root.glob("**/*.json") if p.is_file())

    groups: dict[str, set[Path]] = {}
    regex = source.reader.group_by_filename
    for file_path in files:
        unit = (
            file_path.parent
            if bool(_reader_extra(source.reader).get("hive_partitioning"))
            else file_path
        )
        chunk_id = _chunk_id(file_path, regex)
        groups.setdefault(chunk_id, set()).add(unit)

    return [
        Chunk(chunk_id=chunk_id, files=tuple(sorted(paths)))
        for chunk_id, paths in sorted(groups.items(), key=lambda item: item[0], reverse=True)
    ]


def _chunk_id(path: Path, regex: str | None) -> str:
    if regex:
        match = re.findall(regex, str(path.resolve()))
        if match:
            first = match[0]
            if isinstance(first, tuple):
                return "-".join(str(part) for part in first)
            return str(first)
    return path.name


def _reader_root(workspace_path: str | Path, source: model.Source) -> Path:
    extra = _reader_extra(source.reader)
    raw = extra.get("root") or extra.get("base_dir") or "."
    root = Path(str(raw))
    if not root.is_absolute():
        root = Path(workspace_path) / root
    return root


def _reader_extra(reader: model.Reader) -> dict[str, object]:
    return dict(reader.model_extra or {})


__all__ = ["Chunk", "discover"]
