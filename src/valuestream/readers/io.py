"""Built-in source readers."""

from __future__ import annotations

import atexit
import gzip
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from valuestream.config import model
from valuestream.utils.timer import timed

_TEMP_DIRS: list[Path] = []


@timed
def read(reader: model.Reader, files: list[Path] | tuple[Path, ...]) -> pl.LazyFrame:
    """Read ``files`` according to ``reader.kind``."""
    paths = [str(p) for p in files]
    if reader.kind == "parquet":
        return pl.scan_parquet(
            paths,
            hive_partitioning=bool(_extra(reader).get("hive_partitioning", False)),
            missing_columns="insert",
            extra_columns="ignore",
        )
    if reader.kind == "csv":
        return pl.scan_csv(
            paths,
            separator=str(
                _extra(reader).get("separator") or getattr(reader, "delimiter", None) or ","
            ),
            infer_schema_length=_int_extra(reader, "infer_schema_length", 10_000),
            try_parse_dates=bool(_extra(reader).get("try_parse_dates", True)),
        )
    if reader.kind == "xlsx":
        frames = [
            pl.read_excel(path, sheet_name=getattr(reader, "sheet", None)).lazy() for path in paths
        ]
        return pl.concat(frames) if len(frames) > 1 else frames[0]
    if reader.kind == "pega_ds_export":
        ndjson = _normalize_pega_export(files, reader)
        return pl.scan_ndjson(
            str(ndjson),
            infer_schema_length=_int_extra(reader, "infer_schema_length", 100_000),
        )
    raise ValueError(f"unsupported reader kind: {reader.kind}")


def _normalize_pega_export(files: list[Path] | tuple[Path, ...], reader: model.Reader) -> Path:
    base = _extra(reader).get("archive_temp_dir")
    tmp = Path(tempfile.mkdtemp(prefix="dataset_export_", dir=str(base) if base else None))
    _TEMP_DIRS.append(tmp)
    out = tmp / "normalized.ndjson"
    with out.open("wb") as sink:
        for path in files:
            for payload in _payloads(path):
                _write_records(sink, payload)
    return out


def _payloads(path: Path) -> list[bytes]:
    suffixes = "".join(path.suffixes)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith((".json", ".ndjson")))
            return [zf.read(name) for name in names]
    if suffixes.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as tf:
            members = sorted(
                (m for m in tf.getmembers() if m.name.endswith((".json", ".ndjson"))),
                key=lambda m: m.name,
            )
            out: list[bytes] = []
            for member in members:
                extracted = tf.extractfile(member)
                if extracted is not None:
                    out.append(extracted.read())
            return out
    if path.suffix in {".gz", ".gzip"}:
        with gzip.open(path, "rb") as fh:
            return [fh.read()]
    return [path.read_bytes()]


def _write_records(sink: Any, payload: bytes) -> None:
    text = payload.decode("utf-8").strip()
    if not text:
        return
    if text.startswith("["):
        rows = json.loads(text)
        for row in rows:
            sink.write(json.dumps(row, separators=(",", ":")).encode("utf-8"))
            sink.write(b"\n")
        return
    sink.write(payload)
    if not payload.endswith(b"\n"):
        sink.write(b"\n")


def _cleanup() -> None:
    for tmp in _TEMP_DIRS:
        shutil.rmtree(tmp, ignore_errors=True)
    _TEMP_DIRS.clear()


def cleanup_temporaries() -> None:
    """Remove temporary reader files after the lazy plans using them have collected."""
    _cleanup()


def _extra(reader: model.Reader) -> dict[str, object]:
    return dict(reader.model_extra or {})


def _int_extra(reader: model.Reader, key: str, default: int) -> int:
    value = _extra(reader).get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return int(value)
    return default


atexit.register(_cleanup)

__all__ = ["cleanup_temporaries", "read"]
