"""Internal persistent state for bounded frequency-response processing.

The checkpoint namespace is deliberately separate from ``aggregates/`` and
the metadata ledger.  A checkpoint is a versioned, identity-addressed cache of
prepared contact candidates; it is never a query-visible business result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from urllib.parse import quote
from uuid import uuid4

import polars as pl

CHECKPOINT_SCHEMA_REVISION = 3
SHARD_HASH_REVISION = 1
SHARD_HASH_ALGORITHM = "polars.Expr.hash"
SHARD_HASH_SEEDS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
)
SHARD_COLUMN = "__valuestream_checkpoint_shard"
MANIFEST_FILENAME = "manifest.json"

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_FrameT = TypeVar("_FrameT", pl.DataFrame, pl.LazyFrame)


class CheckpointValidationError(ValueError):
    """Raised when a persisted checkpoint is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class CheckpointShard:
    """One non-empty physical shard described by a checkpoint manifest."""

    shard_id: int
    filename: str
    rows: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Validated metadata for one immutable frequency-response checkpoint."""

    path: Path
    schema_revision: int
    shard_hash_revision: int
    shard_hash_algorithm: str
    shard_hash_seeds: tuple[int, int, int, int]
    polars_version: str
    source_id: str
    processor_id: str
    config_hash: str
    layout_hash: str
    chunk_id: str
    raw_fingerprint: str
    customer_column: str
    customer_dtype: str
    shard_count: int
    rows: int
    shards: tuple[CheckpointShard, ...]

    @property
    def directory(self) -> Path:
        """Directory containing this manifest and its Parquet shards."""

        return self.path.parent

    @property
    def nonempty_shard_ids(self) -> tuple[int, ...]:
        """Sorted IDs of shards that have at least one contact."""

        return tuple(shard.shard_id for shard in self.shards)

    def shard(self, shard_id: int) -> CheckpointShard:
        """Return metadata for ``shard_id`` or raise ``KeyError`` when empty."""

        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)


def checkpoint_path(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
    layout_hash: str,
    chunk_id: str,
    raw_fingerprint: str,
) -> Path:
    """Return the immutable directory for one checkpoint identity.

    Format and Polars versions participate in the path because native Polars
    hash compatibility is only promised within the recorded runtime version.
    Upgrading Polars therefore builds a new cache generation instead of mixing
    differently routed customer shards.
    """

    _validate_identity(
        source_id=source_id,
        processor_id=processor_id,
        config_hash=config_hash,
        layout_hash=layout_hash,
        chunk_id=chunk_id,
        raw_fingerprint=raw_fingerprint,
    )
    return (
        Path(workspace_path)
        / ".valuestream"
        / "state"
        / "frequency_response"
        / f"schema={CHECKPOINT_SCHEMA_REVISION}"
        / f"hash={SHARD_HASH_REVISION}"
        / f"polars={_component(pl.__version__)}"
        / f"source={_component(source_id)}"
        / f"processor={_component(processor_id)}"
        / f"config={config_hash}"
        / f"layout={layout_hash}"
        / f"chunk={_component(chunk_id)}"
        / f"raw={raw_fingerprint}"
    )


def checkpoint_manifest_path(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
    layout_hash: str,
    chunk_id: str,
    raw_fingerprint: str,
) -> Path:
    """Return the manifest path for one checkpoint identity."""

    return (
        checkpoint_path(
            workspace_path,
            source_id=source_id,
            processor_id=processor_id,
            config_hash=config_hash,
            layout_hash=layout_hash,
            chunk_id=chunk_id,
            raw_fingerprint=raw_fingerprint,
        )
        / MANIFEST_FILENAME
    )


def assign_customer_shard(
    frame: _FrameT,
    *,
    customer_column: str,
    shard_count: int,
) -> _FrameT:
    """Add the deterministic internal shard column with native Polars hashing."""

    _validate_shard_request(frame, customer_column=customer_column, shard_count=shard_count)
    shard = (
        pl.col(customer_column).hash(*SHARD_HASH_SEEDS) % pl.lit(shard_count, dtype=pl.UInt64)
    ).cast(pl.UInt32)
    return frame.with_columns(shard.alias(SHARD_COLUMN))


def write_checkpoint(
    frame: pl.DataFrame | pl.LazyFrame,
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
    layout_hash: str,
    chunk_id: str,
    raw_fingerprint: str,
    customer_column: str,
    shard_count: int,
    engine: Literal["auto", "streaming"] = "auto",
    replace_existing: bool = False,
) -> CheckpointManifest:
    """Atomically publish or reuse a sharded prepared-contact checkpoint.

    A valid directory at the state address is immutable and reused without
    touching any files.  A corrupt directory raises instead of being silently
    overwritten, preserving a visible failure boundary for cache repair.
    """

    final_directory = checkpoint_path(
        workspace_path,
        source_id=source_id,
        processor_id=processor_id,
        config_hash=config_hash,
        layout_hash=layout_hash,
        chunk_id=chunk_id,
        raw_fingerprint=raw_fingerprint,
    )
    _validate_shard_request(frame, customer_column=customer_column, shard_count=shard_count)
    schema = frame.schema if isinstance(frame, pl.DataFrame) else frame.collect_schema()
    customer_dtype = str(schema[customer_column])
    if not replace_existing:
        existing = load_manifest(
            workspace_path,
            source_id=source_id,
            processor_id=processor_id,
            config_hash=config_hash,
            layout_hash=layout_hash,
            chunk_id=chunk_id,
            raw_fingerprint=raw_fingerprint,
        )
        if existing is not None:
            _validate_manifest_request(
                existing,
                customer_column=customer_column,
                customer_dtype=customer_dtype,
                shard_count=shard_count,
            )
            return existing

    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    sharded = assign_customer_shard(
        lazy,
        customer_column=customer_column,
        shard_count=shard_count,
    )
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            dir=final_directory.parent,
            prefix=f".{final_directory.name}.",
            suffix=".tmp",
        )
    )
    try:
        _sink_shards(sharded, temporary_directory, engine=engine)
        shards = _inspect_written_shards(
            temporary_directory,
            customer_column=customer_column,
            customer_dtype=customer_dtype,
        )
        payload = _manifest_payload(
            source_id=source_id,
            processor_id=processor_id,
            config_hash=config_hash,
            layout_hash=layout_hash,
            chunk_id=chunk_id,
            raw_fingerprint=raw_fingerprint,
            customer_column=customer_column,
            customer_dtype=customer_dtype,
            shard_count=shard_count,
            rows=sum(shard.rows for shard in shards),
            shards=shards,
        )
        temporary_manifest = temporary_directory / MANIFEST_FILENAME
        _write_manifest(temporary_manifest, payload)
        _parse_manifest(
            temporary_manifest,
            expected={
                "source_id": source_id,
                "processor_id": processor_id,
                "config_hash": config_hash,
                "layout_hash": layout_hash,
                "chunk_id": chunk_id,
                "raw_fingerprint": raw_fingerprint,
            },
            validate_files=True,
        )
        _fsync_directory(temporary_directory)

        # A source-level run lock normally makes this uncontended.  The second
        # existence check still prevents replacing a valid artifact if another
        # writer won a race between the first load and this publication step.
        if final_directory.exists():
            if replace_existing:
                _replace_directory(temporary_directory, final_directory)
                return _parse_manifest(
                    final_directory / MANIFEST_FILENAME,
                    expected={
                        "source_id": source_id,
                        "processor_id": processor_id,
                        "config_hash": config_hash,
                        "layout_hash": layout_hash,
                        "chunk_id": chunk_id,
                        "raw_fingerprint": raw_fingerprint,
                    },
                    validate_files=False,
                )
            raced = load_manifest(
                workspace_path,
                source_id=source_id,
                processor_id=processor_id,
                config_hash=config_hash,
                layout_hash=layout_hash,
                chunk_id=chunk_id,
                raw_fingerprint=raw_fingerprint,
            )
            if raced is None:  # pragma: no cover - guarded by exists above
                raise CheckpointValidationError(
                    f"checkpoint directory {final_directory} appeared without a manifest"
                )
            _validate_manifest_request(
                raced,
                customer_column=customer_column,
                customer_dtype=customer_dtype,
                shard_count=shard_count,
            )
            return raced

        try:
            os.rename(temporary_directory, final_directory)
        except OSError as exc:
            # If another writer published the same state address first, use
            # its fully validated immutable result.  Otherwise preserve the
            # original rename failure.
            if not final_directory.exists():
                raise
            raced = load_manifest(
                workspace_path,
                source_id=source_id,
                processor_id=processor_id,
                config_hash=config_hash,
                layout_hash=layout_hash,
                chunk_id=chunk_id,
                raw_fingerprint=raw_fingerprint,
            )
            if raced is None:  # pragma: no cover - guarded by exists above
                raise CheckpointValidationError(
                    f"checkpoint directory {final_directory} appeared without a manifest"
                ) from exc
            _validate_manifest_request(
                raced,
                customer_column=customer_column,
                customer_dtype=customer_dtype,
                shard_count=shard_count,
            )
            return raced
        _fsync_directory(final_directory.parent)
        return _parse_manifest(
            final_directory / MANIFEST_FILENAME,
            expected={
                "source_id": source_id,
                "processor_id": processor_id,
                "config_hash": config_hash,
                "layout_hash": layout_hash,
                "chunk_id": chunk_id,
                "raw_fingerprint": raw_fingerprint,
            },
            validate_files=False,
        )
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)


def load_manifest(
    workspace_path: str | Path,
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
    layout_hash: str,
    chunk_id: str,
    raw_fingerprint: str,
    validate: bool = True,
) -> CheckpointManifest | None:
    """Load one checkpoint manifest, returning ``None`` when it is absent."""

    path = checkpoint_manifest_path(
        workspace_path,
        source_id=source_id,
        processor_id=processor_id,
        config_hash=config_hash,
        layout_hash=layout_hash,
        chunk_id=chunk_id,
        raw_fingerprint=raw_fingerprint,
    )
    directory = path.parent
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise CheckpointValidationError(f"checkpoint path is not a directory: {directory}")
    if not path.is_file():
        raise CheckpointValidationError(f"checkpoint manifest is missing: {path}")
    return _parse_manifest(
        path,
        expected={
            "source_id": source_id,
            "processor_id": processor_id,
            "config_hash": config_hash,
            "layout_hash": layout_hash,
            "chunk_id": chunk_id,
            "raw_fingerprint": raw_fingerprint,
        },
        validate_files=validate,
    )


def scan_shard(manifest: CheckpointManifest, shard_id: int) -> pl.LazyFrame:
    """Lazily scan one non-empty shard from a validated manifest."""

    shard = manifest.shard(shard_id)
    return pl.scan_parquet(manifest.directory / shard.filename, glob=False)


def _sink_shards(
    frame: pl.LazyFrame,
    directory: Path,
    *,
    engine: Literal["auto", "streaming"],
) -> None:
    """Stream a prepared frame into one deterministic file per non-empty shard."""

    def shard_path(args: Any) -> str:
        if args.index_in_partition != 0:
            raise RuntimeError("checkpoint partition unexpectedly split across files")
        return _shard_filename(int(args.partition_keys.item(0, 0)))

    frame.sink_parquet(
        pl.PartitionBy(
            directory,
            key=SHARD_COLUMN,
            include_key=False,
            file_path_provider=shard_path,
            max_rows_per_file=None,
            approximate_bytes_per_file=None,
        ),
        maintain_order=True,
        row_group_size=8192,
        sync_on_close="all",
        mkdir=True,
        engine=engine,
    )


def _inspect_written_shards(
    directory: Path,
    *,
    customer_column: str,
    customer_dtype: str,
) -> tuple[CheckpointShard, ...]:
    """Build manifest entries without materializing checkpoint rows."""

    shards: list[CheckpointShard] = []
    for path in sorted(directory.glob("*.parquet")):
        shard_id = _shard_id_from_filename(path.name)
        scan = pl.scan_parquet(path, glob=False)
        schema = scan.collect_schema()
        if customer_column not in schema.names():
            raise CheckpointValidationError(
                f"checkpoint shard {shard_id} is missing customer column {customer_column!r}"
            )
        actual_dtype = str(schema[customer_column])
        if actual_dtype != customer_dtype:
            raise CheckpointValidationError(
                f"checkpoint shard {shard_id} customer dtype {actual_dtype!r} does not "
                f"match {customer_dtype!r}"
            )
        summary = scan.select(
            pl.len().alias("rows"),
            pl.col(customer_column).null_count().alias("customer_nulls"),
        ).collect()
        rows = int(summary.item(0, "rows"))
        if int(summary.item(0, "customer_nulls")):
            raise ValueError(f"checkpoint customer column {customer_column!r} cannot contain nulls")
        if rows < 1:
            raise CheckpointValidationError(
                f"checkpoint shard {shard_id} must contain at least one row"
            )
        size_bytes = path.stat().st_size
        shards.append(
            CheckpointShard(
                shard_id=shard_id,
                filename=path.name,
                rows=rows,
                size_bytes=size_bytes,
                sha256=_sha256_file(path),
            )
        )
    return tuple(shards)


def _manifest_payload(
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
    layout_hash: str,
    chunk_id: str,
    raw_fingerprint: str,
    customer_column: str,
    customer_dtype: str,
    shard_count: int,
    rows: int,
    shards: tuple[CheckpointShard, ...],
) -> dict[str, Any]:
    return {
        "schema_revision": CHECKPOINT_SCHEMA_REVISION,
        "shard_hash_revision": SHARD_HASH_REVISION,
        "shard_hash_algorithm": SHARD_HASH_ALGORITHM,
        "shard_hash_seeds": list(SHARD_HASH_SEEDS),
        "polars_version": pl.__version__,
        "source_id": source_id,
        "processor_id": processor_id,
        "config_hash": config_hash,
        "layout_hash": layout_hash,
        "chunk_id": chunk_id,
        "raw_fingerprint": raw_fingerprint,
        "customer_column": customer_column,
        "customer_dtype": customer_dtype,
        "shard_count": shard_count,
        "rows": rows,
        "shards": [
            {
                "shard_id": shard.shard_id,
                "filename": shard.filename,
                "rows": shard.rows,
                "size_bytes": shard.size_bytes,
                "sha256": shard.sha256,
            }
            for shard in shards
        ],
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_manifest(
    path: Path,
    *,
    expected: Mapping[str, str],
    validate_files: bool,
) -> CheckpointManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"cannot read checkpoint manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"checkpoint manifest {path} must contain an object")

    schema_revision, hash_revision, algorithm, seeds, polars_version = _parse_hash_contract(payload)
    identity = {name: _manifest_string(payload, name) for name in expected}
    for name, expected_value in expected.items():
        if identity[name] != expected_value:
            raise CheckpointValidationError(
                f"checkpoint {name} {identity[name]!r} does not match {expected_value!r}"
            )

    customer_column = _manifest_string(payload, "customer_column")
    customer_dtype = _manifest_string(payload, "customer_dtype")
    shard_count = _manifest_int(payload, "shard_count", minimum=1)
    rows = _manifest_int(payload, "rows", minimum=0)
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list):
        raise CheckpointValidationError("checkpoint shards must be a list")
    shards = tuple(_parse_shard(value, shard_count=shard_count) for value in raw_shards)
    shard_ids = [shard.shard_id for shard in shards]
    if shard_ids != sorted(set(shard_ids)):
        raise CheckpointValidationError("checkpoint shards must have unique, sorted IDs")
    if sum(shard.rows for shard in shards) != rows:
        raise CheckpointValidationError("checkpoint shard row counts do not match total rows")

    manifest = CheckpointManifest(
        path=path,
        schema_revision=schema_revision,
        shard_hash_revision=hash_revision,
        shard_hash_algorithm=algorithm,
        shard_hash_seeds=seeds,
        polars_version=polars_version,
        source_id=identity["source_id"],
        processor_id=identity["processor_id"],
        config_hash=identity["config_hash"],
        layout_hash=identity["layout_hash"],
        chunk_id=identity["chunk_id"],
        raw_fingerprint=identity["raw_fingerprint"],
        customer_column=customer_column,
        customer_dtype=customer_dtype,
        shard_count=shard_count,
        rows=rows,
        shards=shards,
    )
    if validate_files:
        _validate_shard_files(manifest)
    return manifest


def _parse_hash_contract(
    payload: Mapping[str, object],
) -> tuple[int, int, str, tuple[int, int, int, int], str]:
    schema_revision = _manifest_int(payload, "schema_revision", minimum=1)
    hash_revision = _manifest_int(payload, "shard_hash_revision", minimum=1)
    algorithm = _manifest_string(payload, "shard_hash_algorithm")
    polars_version = _manifest_string(payload, "polars_version")
    if schema_revision != CHECKPOINT_SCHEMA_REVISION:
        raise CheckpointValidationError(
            f"checkpoint schema revision {schema_revision} is unsupported; "
            f"expected {CHECKPOINT_SCHEMA_REVISION}"
        )
    if hash_revision != SHARD_HASH_REVISION:
        raise CheckpointValidationError(
            f"checkpoint shard hash revision {hash_revision} is unsupported; "
            f"expected {SHARD_HASH_REVISION}"
        )
    if algorithm != SHARD_HASH_ALGORITHM:
        raise CheckpointValidationError(
            f"checkpoint shard hash algorithm {algorithm!r} is unsupported"
        )
    if polars_version != pl.__version__:
        raise CheckpointValidationError(
            f"checkpoint Polars version {polars_version!r} does not match {pl.__version__!r}"
        )

    seeds_value = payload.get("shard_hash_seeds")
    if (
        not isinstance(seeds_value, list)
        or len(seeds_value) != 4
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_value)
    ):
        raise CheckpointValidationError("checkpoint shard_hash_seeds must contain four integers")
    seeds = cast(tuple[int, int, int, int], tuple(seeds_value))
    if seeds != SHARD_HASH_SEEDS:
        raise CheckpointValidationError("checkpoint shard hash seeds do not match this runtime")
    return schema_revision, hash_revision, algorithm, seeds, polars_version


def _parse_shard(value: object, *, shard_count: int) -> CheckpointShard:
    if not isinstance(value, dict):
        raise CheckpointValidationError("checkpoint shard entries must be objects")
    shard_id = _manifest_int(value, "shard_id", minimum=0)
    if shard_id >= shard_count:
        raise CheckpointValidationError(
            f"checkpoint shard {shard_id} is outside configured count {shard_count}"
        )
    filename = _manifest_string(value, "filename")
    if filename != _shard_filename(shard_id):
        raise CheckpointValidationError(
            f"checkpoint shard {shard_id} has unexpected filename {filename!r}"
        )
    rows = _manifest_int(value, "rows", minimum=1)
    size_bytes = _manifest_int(value, "size_bytes", minimum=1)
    sha256 = _manifest_string(value, "sha256")
    if _HASH_PATTERN.fullmatch(sha256) is None:
        raise CheckpointValidationError(
            f"checkpoint shard {shard_id} must have a full lowercase sha256 digest"
        )
    return CheckpointShard(
        shard_id=shard_id,
        filename=filename,
        rows=rows,
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _validate_shard_files(manifest: CheckpointManifest) -> None:
    declared = {shard.filename for shard in manifest.shards}
    actual = {path.name for path in manifest.directory.glob("*.parquet") if path.is_file()}
    if actual != declared:
        raise CheckpointValidationError(
            f"checkpoint Parquet files do not match manifest: declared={sorted(declared)}, "
            f"actual={sorted(actual)}"
        )
    for shard in manifest.shards:
        path = manifest.directory / shard.filename
        try:
            if path.stat().st_size != shard.size_bytes:
                raise CheckpointValidationError(
                    f"checkpoint shard {shard.shard_id} size does not match manifest"
                )
            if _sha256_file(path) != shard.sha256:
                raise CheckpointValidationError(
                    f"checkpoint shard {shard.shard_id} sha256 does not match manifest"
                )
            scan = pl.scan_parquet(path, glob=False)
            schema = scan.collect_schema()
            if manifest.customer_column not in schema.names():
                raise CheckpointValidationError(
                    f"checkpoint shard {shard.shard_id} is missing customer column "
                    f"{manifest.customer_column!r}"
                )
            if str(schema[manifest.customer_column]) != manifest.customer_dtype:
                raise CheckpointValidationError(
                    f"checkpoint shard {shard.shard_id} customer dtype does not match manifest"
                )
            actual_rows = int(scan.select(pl.len()).collect().item())
            if actual_rows != shard.rows:
                raise CheckpointValidationError(
                    f"checkpoint shard {shard.shard_id} row count does not match manifest"
                )
        except CheckpointValidationError:
            raise
        except (OSError, pl.exceptions.PolarsError) as exc:
            raise CheckpointValidationError(
                f"cannot validate checkpoint shard {shard.shard_id}: {exc}"
            ) from exc


def _validate_manifest_request(
    manifest: CheckpointManifest,
    *,
    customer_column: str,
    customer_dtype: str,
    shard_count: int,
) -> None:
    if manifest.customer_column != customer_column:
        raise CheckpointValidationError(
            f"checkpoint customer column {manifest.customer_column!r} does not match "
            f"{customer_column!r}"
        )
    if manifest.customer_dtype != customer_dtype:
        raise CheckpointValidationError(
            f"checkpoint customer dtype {manifest.customer_dtype!r} does not match "
            f"{customer_dtype!r}"
        )
    if manifest.shard_count != shard_count:
        raise CheckpointValidationError(
            f"checkpoint shard count {manifest.shard_count} does not match {shard_count}"
        )


def _validate_shard_request(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    customer_column: str,
    shard_count: int,
) -> None:
    if not customer_column:
        raise ValueError("checkpoint customer column cannot be empty")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or not 1 <= shard_count <= 4096
    ):
        raise ValueError("checkpoint shard count must be a positive integer no greater than 4096")
    columns = frame.columns if isinstance(frame, pl.DataFrame) else frame.collect_schema().names()
    if customer_column not in columns:
        raise ValueError(f"checkpoint frame is missing customer column {customer_column!r}")
    if SHARD_COLUMN in columns:
        raise ValueError(f"checkpoint frame uses reserved column {SHARD_COLUMN!r}")


def _validate_identity(
    *,
    source_id: str,
    processor_id: str,
    config_hash: str,
    layout_hash: str,
    chunk_id: str,
    raw_fingerprint: str,
) -> None:
    for name, value in {
        "source_id": source_id,
        "processor_id": processor_id,
        "chunk_id": chunk_id,
    }.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"checkpoint {name} must be a non-empty string")
    for name, value in {
        "config_hash": config_hash,
        "layout_hash": layout_hash,
        "raw_fingerprint": raw_fingerprint,
    }.items():
        if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
            raise ValueError(f"checkpoint {name} must be a full lowercase sha256 digest")


def _manifest_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise CheckpointValidationError(f"checkpoint {name} must be a non-empty string")
    return value


def _manifest_int(payload: Mapping[str, object], name: str, *, minimum: int) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CheckpointValidationError(
            f"checkpoint {name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _component(value: str) -> str:
    return quote(value, safe="-._~")


def _shard_filename(shard_id: int) -> str:
    return f"shard={shard_id:05d}.parquet"


def _shard_id_from_filename(filename: str) -> int:
    match = re.fullmatch(r"shard=(\d{5})\.parquet", filename)
    if match is None:
        raise CheckpointValidationError(f"unexpected checkpoint shard filename {filename!r}")
    return int(match.group(1))


def _replace_directory(temporary: Path, final: Path) -> None:
    """Replace one checkpoint under the source lock, restoring it on failure."""

    backup = final.with_name(f".{final.name}.{uuid4().hex}.replaced.tmp")
    os.rename(final, backup)
    try:
        os.rename(temporary, final)
    except Exception:
        os.rename(backup, final)
        raise
    _fsync_directory(final.parent)
    shutil.rmtree(backup)
    _fsync_directory(final.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA_REVISION",
    "MANIFEST_FILENAME",
    "SHARD_COLUMN",
    "SHARD_HASH_ALGORITHM",
    "SHARD_HASH_REVISION",
    "SHARD_HASH_SEEDS",
    "CheckpointManifest",
    "CheckpointShard",
    "CheckpointValidationError",
    "assign_customer_shard",
    "checkpoint_manifest_path",
    "checkpoint_path",
    "load_manifest",
    "scan_shard",
    "write_checkpoint",
]
