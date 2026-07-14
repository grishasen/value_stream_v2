"""Pega interaction-history dummy data generator."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import random
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

CUSTOMER_SEGMENTS = ("VIP", "Premium", "CLVHigh", "CLVMedium", "CLVLow")
_UPDATE_TIME_COLUMN = "pxUpdateDateTime"
_CONVERSION_SHARE_OF_POSITIVES = 0.03
_UTC = dt.UTC


@dataclass(frozen=True)
class PegaDummyGenerationConfig:
    """Configuration for generating synthetic Pega-shaped interaction rows."""

    source_path: Path
    output_dir: Path
    start_date: dt.date
    end_date: dt.date
    rows_per_day: int = 1_000_000
    batch_size: int = 100_000
    seed: int = 13
    customer_count: int = 250_000
    positive_rate: float = 0.12
    file_prefix: str = "pega_interactions"
    compression: str = "zstd"
    overwrite: bool = False


@dataclass(frozen=True)
class PegaDummyGenerationReport:
    """Summary of generated Parquet output."""

    output_dir: Path
    files: list[Path]
    rows: int
    columns: list[str]
    start_date: dt.date
    end_date: dt.date
    rows_per_day: int


def generate_pega_dummy_data(config: PegaDummyGenerationConfig) -> PegaDummyGenerationReport:
    """Generate one Parquet file per day from source export value distributions."""

    _validate_config(config)
    rows, columns = _load_source_rows(config.source_path)
    source = pl.DataFrame({column: [row.get(column) for row in rows] for column in columns})
    rng = random.Random(config.seed)
    files: list[Path] = []
    total_rows = 0

    config.output_dir.mkdir(parents=True, exist_ok=True)
    for day_index, day in enumerate(_days(config.start_date, config.end_date)):
        target = config.output_dir / f"{config.file_prefix}_{day:%Y%m%d}.parquet"
        if target.exists() and not config.overwrite:
            raise FileExistsError(f"{target} already exists; pass overwrite=True to replace it")
        _write_day(source, target, day, day_index, config, rng)
        files.append(target)
        total_rows += config.rows_per_day

    return PegaDummyGenerationReport(
        output_dir=config.output_dir,
        files=files,
        rows=total_rows,
        columns=[*columns, "IsProspect", "CustomerSegment"],
        start_date=config.start_date,
        end_date=config.end_date,
        rows_per_day=config.rows_per_day,
    )


def _validate_config(config: PegaDummyGenerationConfig) -> None:
    if not config.source_path.is_file():
        raise FileNotFoundError(config.source_path)
    if config.end_date < config.start_date:
        raise ValueError("end_date must be on or after start_date")
    if config.rows_per_day <= 0:
        raise ValueError("rows_per_day must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.customer_count <= 0:
        raise ValueError("customer_count must be positive")
    if not 0.0 < config.positive_rate < 1.0:
        raise ValueError("positive_rate must be greater than 0.0 and less than 1.0")
    if not config.file_prefix:
        raise ValueError("file_prefix must not be empty")


def _load_source_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    seen: set[str] = set()
    for payload in _payloads(path):
        for row in _records(payload):
            rows.append(row)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    if not rows:
        raise ValueError(f"{path} did not contain any JSON records")
    return rows, columns


def _payloads(path: Path) -> list[bytes]:
    suffixes = "".join(path.suffixes)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name for name in archive.namelist() if name.endswith((".json", ".ndjson"))
            )
            return [archive.read(name) for name in names]
    if suffixes.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path) as archive:
            members = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.name.endswith((".json", ".ndjson"))
                ),
                key=lambda member: member.name,
            )
            payloads: list[bytes] = []
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is not None:
                    payloads.append(extracted.read())
            return payloads
    if path.suffix in {".gz", ".gzip"}:
        with gzip.open(path, "rb") as handle:
            return [handle.read()]
    return [path.read_bytes()]


def _records(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("JSON array source must contain objects")
        return [_validate_record(record) for record in records]
    return [_validate_record(json.loads(line)) for line in text.splitlines() if line.strip()]


def _validate_record(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("source records must be JSON objects")
    return record


def _days(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=offset) for offset in range((end - start).days + 1)]


def _write_day(
    source: pl.DataFrame,
    target: Path,
    day: dt.date,
    day_index: int,
    config: PegaDummyGenerationConfig,
    rng: random.Random,
) -> None:
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    source_indices = range(source.height)
    try:
        while rows_written < config.rows_per_day:
            batch_rows = min(config.batch_size, config.rows_per_day - rows_written)
            global_start = day_index * config.rows_per_day + rows_written
            indices = rng.choices(source_indices, k=batch_rows)
            batch = _enrich_batch(
                source[indices],
                day=day,
                global_start=global_start,
                customer_count=config.customer_count,
                positive_rate=config.positive_rate,
                rng=rng,
            )
            table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp,
                    table.schema,
                    compression=config.compression,
                    use_dictionary=True,
                )
            writer.write_table(table, row_group_size=batch_rows)
            rows_written += batch_rows
    except Exception:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    if writer is None:
        raise ValueError("no rows were generated")
    writer.close()
    tmp.replace(target)


def _enrich_batch(
    batch: pl.DataFrame,
    *,
    day: dt.date,
    global_start: int,
    customer_count: int,
    positive_rate: float,
    rng: random.Random,
) -> pl.DataFrame:
    count = batch.height
    customer_numbers = [rng.randrange(1, customer_count + 1) for _ in range(count)]
    customer_ids = [f"C-{number}" for number in customer_numbers]
    outcome_ms = _outcome_epoch_ms(day, count, rng)
    day_start_ms = int(dt.datetime.combine(day, dt.time(), tzinfo=_UTC).timestamp() * 1_000)
    decision_ms = [
        max(outcome_ms[index] - rng.randrange(0, 2_001), day_start_ms) for index in range(count)
    ]
    positive_responses = [rng.random() < positive_rate for _ in range(count)]
    propensity_scores = [_propensity_score(is_positive, rng) for is_positive in positive_responses]

    columns: list[pl.Series] = []
    if "pxInteractionID" in batch.columns:
        columns.append(
            pl.Series(
                "pxInteractionID",
                [str(-1_000_000_000_000_000_000 - global_start - index) for index in range(count)],
            )
        )
    if "CustomerID" in batch.columns:
        columns.append(pl.Series("CustomerID", customer_ids))
    if "pySubjectID" in batch.columns:
        columns.append(pl.Series("pySubjectID", customer_ids))
    if "pxStreamPosition" in batch.columns:
        columns.append(
            pl.Series("pxStreamPosition", [str(global_start + index) for index in range(count)])
        )
    if "pxStreamPartition" in batch.columns:
        columns.append(
            pl.Series(
                "pxStreamPartition", [str((global_start + index) % 30) for index in range(count)]
            )
        )
    if "pyOutcome" in batch.columns:
        columns.append(pl.Series("pyOutcome", _outcomes(batch, positive_responses, rng)))
    for column in _numeric_propensity_columns(batch):
        columns.append(pl.Series(column, propensity_scores))
    if "pxOutcomeTime" in batch.columns:
        columns.append(
            pl.Series("pxOutcomeTime", [_format_pega_time(value) for value in outcome_ms])
        )
    if "pxDecisionTime" in batch.columns:
        columns.append(
            pl.Series("pxDecisionTime", [_format_pega_time(value) for value in decision_ms])
        )
    if _UPDATE_TIME_COLUMN in batch.columns:
        columns.append(pl.Series(_UPDATE_TIME_COLUMN, outcome_ms))
    columns.extend(
        [
            pl.Series("IsProspect", [_is_prospect(number) for number in customer_numbers]),
            pl.Series("CustomerSegment", [_segment(number) for number in customer_numbers]),
        ]
    )
    return batch.with_columns(columns)


def _outcome_epoch_ms(day: dt.date, count: int, rng: random.Random) -> list[int]:
    day_start = dt.datetime.combine(day, dt.time(), tzinfo=_UTC)
    start_ms = int(day_start.timestamp() * 1_000)
    return [start_ms + rng.randrange(86_400_000) for _ in range(count)]


def _format_pega_time(epoch_ms: int) -> str:
    value = dt.datetime.fromtimestamp(epoch_ms / 1_000, tz=_UTC)
    return f"{value:%Y%m%dT%H%M%S}.{value.microsecond // 1_000:03d} GMT"


def _outcomes(batch: pl.DataFrame, positive_responses: list[bool], rng: random.Random) -> list[str]:
    directions = (
        [str(value) for value in batch["pyDirection"].to_list()]
        if "pyDirection" in batch.columns
        else ["Inbound"] * len(positive_responses)
    )
    outcomes: list[str] = []
    for is_positive, direction in zip(positive_responses, directions, strict=True):
        if is_positive:
            outcome = "Conversion" if rng.random() < _CONVERSION_SHARE_OF_POSITIVES else "Clicked"
        else:
            outcome = "Pending" if direction.casefold() == "outbound" else "Impression"
        outcomes.append(outcome)
    return outcomes


def _numeric_propensity_columns(batch: pl.DataFrame) -> list[str]:
    return [
        column
        for column, dtype in batch.schema.items()
        if "propensity" in column.lower() and _is_numeric_dtype(dtype)
    ]


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }


def _propensity_score(is_positive: bool, rng: random.Random) -> float:
    score = rng.betavariate(8.0, 2.0) if is_positive else rng.betavariate(2.0, 8.0)
    return min(max(score, 0.001), 0.999)


def _is_prospect(customer_number: int) -> bool:
    return customer_number % 10 == 0


def _segment(customer_number: int) -> str:
    bucket = customer_number % 100
    if bucket < 5:
        return "VIP"
    if bucket < 20:
        return "Premium"
    if bucket < 40:
        return "CLVHigh"
    if bucket < 70:
        return "CLVMedium"
    return "CLVLow"
