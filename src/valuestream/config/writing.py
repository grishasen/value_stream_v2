"""Shared atomic writes and rollback boundaries for catalog authoring."""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

CATALOG_FILENAMES = ("pipelines.yaml", "processors.yaml", "metrics.yaml", "dashboards.yaml")


@contextmanager
def workspace_configuration_transaction(workspace: str | Path) -> Iterator[None]:
    """Restore catalog and workspace AI config when a complete apply fails."""

    root = Path(workspace)
    paths = [
        *(root / "catalog" / filename for filename in CATALOG_FILENAMES),
        root / "ai.yaml",
    ]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".catalog-authoring.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            with _configuration_file_transaction(paths):
                yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _configuration_file_transaction(paths: Iterable[Path]) -> Iterator[None]:
    """Restore ``paths`` to their exact pre-write contents on any exception."""

    unique_paths = list(dict.fromkeys(paths))
    snapshots = {path: path.read_bytes() if path.exists() else None for path in unique_paths}
    try:
        yield
    except BaseException:
        for path, content in snapshots.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace one YAML file after serializing it completely."""

    payload = yaml.safe_dump(data, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
