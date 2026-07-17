"""One isolated ingestion benchmark sample.

The parent runner launches this module in a fresh process so ``ru_maxrss`` and
CPU counters belong to exactly one sample.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import math
import resource
import struct
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from valuestream.config.canonical import catalog_config_hash, source_computation_hash
from valuestream.config.loader import load
from valuestream.config.model import effective_processor_states
from valuestream.engine import run_source
from valuestream.states import cpc, hll, kll, tdigest, theta, topk

_UNSTABLE_OUTPUT_COLUMNS = {"pipeline_run_id", "created_at"}
_APPROXIMATE_STATE_TYPES = {"cpc", "hll", "kll", "tdigest", "theta", "topk"}
_APPROXIMATE_SAMPLE_LIMIT = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    catalog = load(args.workspace)
    source = next(item for item in catalog.pipelines.sources if item.id == args.source)
    cpu_before = _cpu_seconds()
    wall_started = time.perf_counter()
    result = run_source(args.workspace, args.source, parallel=args.parallel)
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = _cpu_seconds() - cpu_before
    peak_rss_bytes = _peak_rss_bytes()
    rows = result.rows_in
    chunk_seconds = sorted(chunk.elapsed_ms / 1000 for chunk in result.chunks if chunk.elapsed_ms)
    state_types = {
        processor.id: {
            name: spec.type for name, spec in effective_processor_states(processor).items()
        }
        for processor in catalog.processors.processors
        if processor.source == args.source
    }
    output = _output_contract(args.workspace, args.source, state_types=state_types)
    payload: dict[str, Any] = {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "average_cores": cpu_seconds / wall_seconds if wall_seconds else 0.0,
        "rows_per_second": rows / wall_seconds if wall_seconds else 0.0,
        "cpu_seconds_per_million_rows": cpu_seconds * 1_000_000 / rows if rows else 0.0,
        "peak_rss_bytes": peak_rss_bytes,
        "rows_in": rows,
        "rows_kept": result.rows_kept,
        "chunks_total": result.chunks_total,
        "chunks_ok": result.chunks_ok,
        "chunks_failed": result.chunks_failed,
        "chunk_seconds_p50": _percentile(chunk_seconds, 0.50),
        "chunk_seconds_p95": _percentile(chunk_seconds, 0.95),
        "chunk_pipeline_seconds_sum": sum(chunk_seconds),
        "orchestration_wall_seconds": (
            max(0.0, wall_seconds - sum(chunk_seconds)) if args.parallel == 1 else None
        ),
        "aggregate_files": output["files"],
        "aggregate_bytes": output["bytes"],
        # ``output_digest`` remains as a compatibility alias for the exact,
        # non-binary contract introduced in contract v2.
        "output_digest": output["exact_digest"],
        "exact_output_digest": output["exact_digest"],
        "representation_digest": output["representation_digest"],
        "approximate_state_probes": output["approximate_state_probes"],
        "unclassified_binary_states": output["unclassified_binary_states"],
        "catalog_hash": catalog_config_hash(catalog),
        "source_computation_hash": source_computation_hash(catalog, args.source),
        "streaming": source.reader.streaming,
        "materialize_transforms": source.materialize_transforms,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(len(values) * q) - 1))
    return values[index]


def _output_contract(
    workspace: Path,
    source_id: str,
    *,
    state_types: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Return exact and approximate output contracts for one fresh run.

    DataSketches payloads are opaque algorithm state, not canonical result
    bytes.  The exact digest therefore excludes every binary column.  Known
    sketch columns are checked separately through a bounded, deterministic set
    of decoded semantic probes; the full byte representation remains available
    as a diagnostic digest only.
    """

    root = workspace / "aggregates" / source_id
    paths = sorted(path for path in root.glob("*/*/period=*/*.parquet") if path.is_file())
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        logical_partition = str(path.relative_to(root).parent)
        grouped.setdefault(logical_partition, []).append(path)

    expected_types = state_types or {}
    approximate = _empty_approximate_contract(expected_types)
    unclassified_binary_states: set[str] = set()
    exact_digest = hashlib.sha256()
    representation_digest = hashlib.sha256()
    for logical_partition, partition_paths in sorted(grouped.items()):
        frames = []
        for path in partition_paths:
            frame = pl.read_parquet(path)
            frames.append(
                frame.drop(
                    [column for column in _UNSTABLE_OUTPUT_COLUMNS if column in frame.columns]
                )
            )
        frame = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
        columns = sorted(frame.columns)
        frame = frame.select(columns)
        sortable = [
            name
            for name, dtype in frame.schema.items()
            if dtype != pl.Binary and not dtype.is_nested()
        ]
        if sortable:
            frame = frame.sort(sortable, nulls_last=True)
        _update_frame_digest(representation_digest, logical_partition, frame)

        processor_id = logical_partition.split("/", 1)[0]
        processor_state_types = expected_types.get(processor_id, {})
        binary_columns = [name for name, dtype in frame.schema.items() if dtype == pl.Binary]
        exact_frame = _normalize_exact_frame(
            frame.drop(binary_columns),
            processor_state_types=processor_state_types,
        )
        _update_exact_frame_digest(exact_digest, logical_partition, exact_frame)

        for column in binary_columns:
            state_type = processor_state_types.get(column)
            if state_type not in _APPROXIMATE_STATE_TYPES:
                unclassified_binary_states.add(f"{processor_id}.{column}")
                continue
            _collect_approximate_probes(
                approximate[f"{processor_id}.{column}"],
                frame[column],
                logical_partition=logical_partition,
                state_type=state_type,
            )
    return {
        "files": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "exact_digest": exact_digest.hexdigest(),
        "representation_digest": representation_digest.hexdigest(),
        "approximate_state_probes": approximate,
        "unclassified_binary_states": sorted(unclassified_binary_states),
    }


def _empty_approximate_contract(
    state_types: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    return {
        f"{processor_id}.{state_name}": {
            "type": state_type,
            "rows": 0,
            "payloads": 0,
            "nulls": 0,
            "present_partitions": 0,
            "decode_errors": 0,
            "decode_error_rows": [],
            "samples": [],
        }
        for processor_id, states in sorted(state_types.items())
        for state_name, state_type in sorted(states.items())
        if state_type in _APPROXIMATE_STATE_TYPES
    }


def _update_frame_digest(
    digest: Any,
    logical_partition: str,
    frame: pl.DataFrame,
) -> None:
    buffer = io.BytesIO()
    frame.write_ipc(buffer, compression="uncompressed")
    digest.update(logical_partition.encode("utf-8"))
    digest.update(b"\0")
    digest.update(buffer.getvalue())


def _update_exact_frame_digest(
    digest: Any,
    logical_partition: str,
    frame: pl.DataFrame,
) -> None:
    """Hash logical scalar values, independent of Arrow buffer layout."""

    _update_length_prefixed(digest, b"P", logical_partition.encode("utf-8"))
    digest.update(struct.pack(">Q", frame.width))
    for name, dtype in frame.schema.items():
        _update_length_prefixed(digest, b"C", name.encode("utf-8"))
        _update_length_prefixed(digest, b"T", str(dtype).encode("utf-8"))
    digest.update(struct.pack(">Q", frame.height))
    for row in frame.iter_rows(named=False):
        digest.update(b"R")
        for value in row:
            _update_exact_value(digest, value)


def _update_exact_value(digest: Any, value: Any) -> None:  # noqa: PLR0911, PLR0912
    if value is None:
        digest.update(b"N")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        _update_length_prefixed(digest, b"I", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if math.isnan(value):
            digest.update(b"FNAN")
        elif math.isinf(value):
            digest.update(b"F+INF" if value > 0 else b"F-INF")
        else:
            digest.update(b"F" + struct.pack(">d", value))
        return
    if isinstance(value, str):
        _update_length_prefixed(digest, b"S", value.encode("utf-8"))
        return
    if isinstance(value, bytes | bytearray | memoryview):
        _update_length_prefixed(digest, b"Y", bytes(value))
        return
    if isinstance(value, dt.datetime):
        _update_length_prefixed(digest, b"Z", value.isoformat().encode("ascii"))
        return
    if isinstance(value, dt.date):
        _update_length_prefixed(digest, b"D", value.isoformat().encode("ascii"))
        return
    if isinstance(value, dt.time):
        _update_length_prefixed(digest, b"M", value.isoformat().encode("ascii"))
        return
    if isinstance(value, list | tuple):
        digest.update(b"L" + struct.pack(">Q", len(value)))
        for item in value:
            _update_exact_value(digest, item)
        return
    if isinstance(value, dict):
        digest.update(b"O" + struct.pack(">Q", len(value)))
        for key in sorted(value, key=repr):
            _update_exact_value(digest, key)
            _update_exact_value(digest, value[key])
        return
    payload = f"{type(value).__module__}.{type(value).__qualname__}:{value!r}".encode()
    _update_length_prefixed(digest, b"X", payload)


def _update_length_prefixed(digest: Any, tag: bytes, payload: bytes) -> None:
    digest.update(tag)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _normalize_exact_frame(
    frame: pl.DataFrame,
    *,
    processor_state_types: Mapping[str, str],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for name, dtype in frame.schema.items():
        if dtype not in {pl.Float32, pl.Float64}:
            continue
        significant_digits = 10 if processor_state_types.get(name) == "pooled_variance" else 12
        rounded = pl.col(name).round_sig_figs(significant_digits)
        expressions.append(pl.when(rounded == 0.0).then(pl.lit(0.0)).otherwise(rounded).alias(name))
    return frame.with_columns(expressions) if expressions else frame


def _collect_approximate_probes(
    contract: dict[str, Any],
    series: pl.Series,
    *,
    logical_partition: str,
    state_type: str,
) -> None:
    contract["rows"] += len(series)
    contract["nulls"] += series.null_count()
    contract["payloads"] += len(series) - series.null_count()
    contract["present_partitions"] += 1
    samples = contract["samples"]
    if len(samples) >= _APPROXIMATE_SAMPLE_LIMIT:
        return
    for row_index, payload in enumerate(series):
        if payload is None:
            continue
        row_id = f"{logical_partition}#{row_index}"
        try:
            probes = _sketch_probes(state_type, payload)
        except Exception:  # pragma: no cover - defensive contract diagnostics
            contract["decode_errors"] += 1
            error_rows = contract["decode_error_rows"]
            if len(error_rows) < 8:
                error_rows.append(row_id)
        else:
            samples.append({"row": row_id, "probes": probes})
        if len(samples) >= _APPROXIMATE_SAMPLE_LIMIT:
            break


def _sketch_probes(
    state_type: str,
    payload: bytes | bytearray | memoryview,
) -> dict[str, float | int]:
    if state_type == "tdigest":
        return {
            "weight": tdigest.weight(payload),
            "q01": tdigest.quantile(payload, 0.01),
            "q50": tdigest.quantile(payload, 0.50),
            "q99": tdigest.quantile(payload, 0.99),
        }
    if state_type == "kll":
        return {
            "count": kll.count(payload),
            "q01": kll.quantile(payload, 0.01),
            "q50": kll.quantile(payload, 0.50),
            "q99": kll.quantile(payload, 0.99),
        }
    if state_type == "cpc":
        lower, upper = cpc.bounds(payload)
        return {"estimate": cpc.estimate(payload), "lower": lower, "upper": upper}
    if state_type == "hll":
        lower, upper = hll.bounds(payload)
        return {"estimate": hll.estimate(payload), "lower": lower, "upper": upper}
    if state_type == "theta":
        lower, upper = theta.bounds(payload)
        return {"estimate": theta.estimate(payload), "lower": lower, "upper": upper}
    if state_type == "topk":
        items = topk.frequent_items(payload)
        return {
            "weight": topk.weight(payload),
            "items": len(items),
            "estimate_sum": sum(item["estimate"] for item in items),
            "lower_sum": sum(item["lower_bound"] for item in items),
            "upper_sum": sum(item["upper_bound"] for item in items),
        }
    raise ValueError(f"unsupported approximate state type: {state_type}")


if __name__ == "__main__":
    main()
