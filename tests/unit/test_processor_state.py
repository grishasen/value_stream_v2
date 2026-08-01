"""Persistent sharded processor-state store tests."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from valuestream.store.processor_state import (
    CHECKPOINT_SCHEMA_REVISION,
    MANIFEST_FILENAME,
    SHARD_COLUMN,
    SHARD_HASH_ALGORITHM,
    SHARD_HASH_REVISION,
    SHARD_HASH_SEEDS,
    CheckpointValidationError,
    assign_customer_shard,
    checkpoint_manifest_path,
    checkpoint_path,
    load_manifest,
    scan_shard,
    write_checkpoint,
)

CONFIG_HASH = "a" * 64
LAYOUT_HASH = "c" * 64
RAW_FINGERPRINT = "b" * 64
IDENTITY = {
    "source_id": "ih/source",
    "processor_id": "frequency response",
    "config_hash": CONFIG_HASH,
    "layout_hash": LAYOUT_HASH,
    "chunk_id": "2026-07-31",
    "raw_fingerprint": RAW_FINGERPRINT,
}


def _contacts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "CustomerID": ["customer-a", "customer-b", "customer-a", "customer-c"],
            "ActionName": ["A", "A", "B", "C"],
            "DecisionTime": [1, 2, 3, 4],
        }
    )


def _write(tmp_path: Path, frame: pl.DataFrame | None = None, *, shards: int = 8):
    return write_checkpoint(
        _contacts() if frame is None else frame,
        tmp_path,
        **IDENTITY,
        customer_column="CustomerID",
        shard_count=shards,
    )


@pytest.mark.unit
def test_checkpoint_path_is_versioned_and_outside_query_visible_aggregates(
    tmp_path: Path,
) -> None:
    directory = checkpoint_path(tmp_path, **IDENTITY)
    manifest = checkpoint_manifest_path(tmp_path, **IDENTITY)

    assert directory.is_relative_to(tmp_path / ".valuestream" / "state" / "frequency_response")
    assert f"schema={CHECKPOINT_SCHEMA_REVISION}" in directory.parts
    assert f"hash={SHARD_HASH_REVISION}" in directory.parts
    assert "source=ih%2Fsource" in directory.parts
    assert f"config={CONFIG_HASH}" in directory.parts
    assert f"layout={LAYOUT_HASH}" in directory.parts
    assert manifest == directory / MANIFEST_FILENAME

    _write(tmp_path)
    assert not (tmp_path / "aggregates").exists()
    assert not (tmp_path / "meta").exists()


@pytest.mark.unit
def test_native_polars_shards_are_deterministic_for_eager_and_lazy_frames() -> None:
    frame = _contacts()
    eager = assign_customer_shard(
        frame,
        customer_column="CustomerID",
        shard_count=8,
    )
    repeated = assign_customer_shard(
        frame,
        customer_column="CustomerID",
        shard_count=8,
    )
    lazy = assign_customer_shard(
        frame.lazy(),
        customer_column="CustomerID",
        shard_count=8,
    ).collect()

    assert eager.get_column(SHARD_COLUMN).to_list() == repeated.get_column(SHARD_COLUMN).to_list()
    assert eager.get_column(SHARD_COLUMN).to_list() == lazy.get_column(SHARD_COLUMN).to_list()
    assert (
        eager.filter(pl.col("CustomerID") == "customer-a").get_column(SHARD_COLUMN).n_unique() == 1
    )
    assert eager.get_column(SHARD_COLUMN).min() >= 0
    assert eager.get_column(SHARD_COLUMN).max() < 8


@pytest.mark.unit
def test_write_load_and_scan_round_trip_manifest_and_contacts(tmp_path: Path) -> None:
    manifest = _write(tmp_path)

    assert manifest.schema_revision == CHECKPOINT_SCHEMA_REVISION
    assert manifest.shard_hash_revision == SHARD_HASH_REVISION
    assert manifest.shard_hash_algorithm == SHARD_HASH_ALGORITHM
    assert manifest.shard_hash_seeds == SHARD_HASH_SEEDS
    assert manifest.polars_version == pl.__version__
    assert manifest.source_id == IDENTITY["source_id"]
    assert manifest.processor_id == IDENTITY["processor_id"]
    assert manifest.config_hash == CONFIG_HASH
    assert manifest.layout_hash == LAYOUT_HASH
    assert manifest.chunk_id == IDENTITY["chunk_id"]
    assert manifest.raw_fingerprint == RAW_FINGERPRINT
    assert manifest.customer_column == "CustomerID"
    assert manifest.customer_dtype == "String"
    assert manifest.shard_count == 8
    assert manifest.rows == 4
    assert manifest.nonempty_shard_ids == tuple(sorted(manifest.nonempty_shard_ids))
    assert all(shard.rows > 0 for shard in manifest.shards)
    assert all(shard.size_bytes > 0 for shard in manifest.shards)
    assert all(len(shard.sha256) == 64 for shard in manifest.shards)

    loaded = load_manifest(tmp_path, **IDENTITY)
    assert loaded == manifest

    recovered = pl.concat(
        [scan_shard(manifest, shard_id).collect() for shard_id in manifest.nonempty_shard_ids]
    ).sort(["CustomerID", "ActionName", "DecisionTime"])
    expected = _contacts().sort(["CustomerID", "ActionName", "DecisionTime"])
    assert recovered.equals(expected)
    assert SHARD_COLUMN not in recovered.columns


@pytest.mark.unit
def test_valid_content_address_is_reused_without_rewriting_files(tmp_path: Path) -> None:
    first = _write(tmp_path)
    mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in [first.path, *(first.directory / item.filename for item in first.shards)]
    }

    second = _write(
        tmp_path,
        pl.DataFrame(
            {
                "CustomerID": ["different-input-is-not-read-at-a-valid-content-address"],
                "ActionName": ["Z"],
                "DecisionTime": [99],
            }
        ),
    )

    assert second == first
    assert {
        path.name: path.stat().st_mtime_ns
        for path in [second.path, *(second.directory / item.filename for item in second.shards)]
    } == mtimes


@pytest.mark.unit
def test_checkpoint_layout_identity_separates_shard_tuning(tmp_path: Path) -> None:
    first = _write(tmp_path, shards=8)
    tuned_identity = {**IDENTITY, "layout_hash": "d" * 64}
    tuned = write_checkpoint(
        _contacts(),
        tmp_path,
        **tuned_identity,
        customer_column="CustomerID",
        shard_count=16,
    )

    assert first.config_hash == tuned.config_hash
    assert first.layout_hash != tuned.layout_hash
    assert first.directory != tuned.directory
    assert first.directory.is_dir()
    assert tuned.directory.is_dir()


@pytest.mark.unit
def test_checkpoint_streams_lazy_input_without_eager_partitioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_partition(*args: object, **kwargs: object) -> None:
        raise AssertionError("eager DataFrame.partition_by must not be used")

    monkeypatch.setattr(pl.DataFrame, "partition_by", fail_partition)

    manifest = write_checkpoint(
        _contacts().lazy(),
        tmp_path,
        **IDENTITY,
        customer_column="CustomerID",
        shard_count=8,
        engine="streaming",
    )

    assert manifest.rows == 4
    assert manifest.shards


@pytest.mark.unit
def test_force_replacement_rebuilds_same_weak_source_address(tmp_path: Path) -> None:
    first = _write(tmp_path)
    replacement = pl.DataFrame(
        {
            "CustomerID": ["customer-z"],
            "ActionName": ["Z"],
            "DecisionTime": [99],
        }
    )

    rebuilt = write_checkpoint(
        replacement.lazy(),
        tmp_path,
        **IDENTITY,
        customer_column="CustomerID",
        shard_count=8,
        engine="streaming",
        replace_existing=True,
    )

    assert rebuilt.path == first.path
    recovered = pl.concat(
        [scan_shard(rebuilt, shard_id).collect() for shard_id in rebuilt.nonempty_shard_ids]
    )
    assert recovered.to_dict(as_series=False) == replacement.to_dict(as_series=False)


@pytest.mark.unit
def test_corrupt_shard_is_rejected_instead_of_reused(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    shard_path = manifest.directory / manifest.shards[0].filename
    with shard_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(CheckpointValidationError, match="size does not match"):
        load_manifest(tmp_path, **IDENTITY)
    with pytest.raises(CheckpointValidationError, match="size does not match"):
        _write(tmp_path)


@pytest.mark.unit
def test_same_size_shard_corruption_is_detected_by_sha256(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    shard_path = manifest.directory / manifest.shards[0].filename
    with shard_path.open("r+b") as handle:
        handle.seek(16)
        original = handle.read(1)
        handle.seek(16)
        handle.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(CheckpointValidationError, match="sha256 does not match"):
        load_manifest(tmp_path, **IDENTITY)


@pytest.mark.unit
def test_manifest_row_counts_are_checked_against_parquet_metadata(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["shards"][0]["rows"] += 1
    payload["rows"] += 1
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointValidationError, match="row count does not match"):
        load_manifest(tmp_path, **IDENTITY)


@pytest.mark.unit
def test_manifest_identity_and_hash_contract_are_validated(tmp_path: Path) -> None:
    manifest = _write(tmp_path)
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["shard_hash_seeds"][0] += 1
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointValidationError, match="hash seeds"):
        load_manifest(tmp_path, **IDENTITY)


@pytest.mark.unit
def test_missing_manifest_in_published_directory_is_invalid(tmp_path: Path) -> None:
    directory = checkpoint_path(tmp_path, **IDENTITY)
    directory.mkdir(parents=True)

    with pytest.raises(CheckpointValidationError, match="manifest is missing"):
        load_manifest(tmp_path, **IDENTITY)


@pytest.mark.unit
def test_empty_checkpoint_has_manifest_and_no_physical_shards(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        schema={
            "CustomerID": pl.String,
            "ActionName": pl.String,
            "DecisionTime": pl.Int64,
        }
    )
    manifest = _write(tmp_path, frame)

    assert manifest.rows == 0
    assert manifest.shards == ()
    assert manifest.nonempty_shard_ids == ()
    assert list(manifest.directory.glob("*.parquet")) == []


@pytest.mark.unit
def test_failed_shard_write_leaves_no_published_or_temporary_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated full disk")

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_write)

    with pytest.raises(OSError, match="simulated full disk"):
        _write(tmp_path)

    final = checkpoint_path(tmp_path, **IDENTITY)
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*.tmp"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("frame", "customer_column", "shards", "message"),
    [
        (pl.DataFrame({"Other": ["x"]}), "CustomerID", 8, "missing customer"),
        (_contacts(), "CustomerID", 0, "positive integer"),
        (
            _contacts().with_columns(pl.lit(0).alias(SHARD_COLUMN)),
            "CustomerID",
            8,
            "reserved column",
        ),
        (
            pl.DataFrame({"CustomerID": [None], "ActionName": ["A"]}),
            "CustomerID",
            8,
            "cannot contain nulls",
        ),
    ],
)
def test_checkpoint_write_rejects_invalid_shard_inputs(
    tmp_path: Path,
    frame: pl.DataFrame,
    customer_column: str,
    shards: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        write_checkpoint(
            frame,
            tmp_path,
            **IDENTITY,
            customer_column=customer_column,
            shard_count=shards,
        )


@pytest.mark.unit
def test_content_hashes_must_be_full_lowercase_sha256(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config_hash"):
        checkpoint_path(tmp_path, **{**IDENTITY, "config_hash": "short"})
    with pytest.raises(ValueError, match="layout_hash"):
        checkpoint_path(tmp_path, **{**IDENTITY, "layout_hash": "short"})
    with pytest.raises(ValueError, match="raw_fingerprint"):
        checkpoint_path(tmp_path, **{**IDENTITY, "raw_fingerprint": "B" * 64})
