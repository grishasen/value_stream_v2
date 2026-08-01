"""Bounded-lookback runner contracts for ``frequency_response`` processors."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import duckdb
import polars as pl
import pytest

from valuestream.config import model
from valuestream.engine import ledger, runner
from valuestream.processors.context import ChunkContext
from valuestream.processors.frequency_response import FrequencyResponseProcessor
from valuestream.readers.discovery import Chunk
from valuestream.store.processor_state import CheckpointPayload


def _source() -> model.Source:
    return model.Source.model_validate(
        {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "defaults": {"Transformed": True},
        }
    )


def _processor(
    *,
    frequency: bool,
    window_hours: int = 168,
    partition_lag_hours: int = 0,
) -> Any:
    return SimpleNamespace(
        id="frequency" if frequency else "normal",
        config_hash="processor-config",
        is_frequency=frequency,
        config=(
            model.FrequencyResponseProcessor.model_validate(
                {
                    "id": "frequency",
                    "source": "events",
                    "kind": "frequency_response",
                    "group_by": ["Placement", "ExposureBucket"],
                    "time": {"property": "DecisionTime", "grain": "daily"},
                    "states": {
                        "clicked_contacts": {
                            "type": "count",
                            "source_column": "ClickedContact",
                        },
                        "runner_propensity_sum": {
                            "type": "value_sum",
                            "source_column": "RunnerPropensity",
                        },
                    },
                    "columns": {
                        "customer": "CustomerID",
                        "interaction": "InteractionID",
                        "action": "ActionID",
                        "placement": "Placement",
                        "rank": "Rank",
                        "outcome": "Outcome",
                        "propensity": "Propensity",
                    },
                    "alternative_group_by": ["Placement"],
                    "positive_values": ["Clicked"],
                    "exposure_values": ["Impression"],
                    "candidate_values": ["Impression", "Clicked"],
                    "window_hours": window_hours,
                    "partition_lag_hours": partition_lag_hours,
                    "frequency_column": "ExposureBucket",
                }
            )
            if frequency
            else SimpleNamespace(grains=[])
        ),
    )


def _frequency_row(label: str, when: dt.datetime) -> dict[str, Any]:
    return {
        "CustomerID": label,
        "InteractionID": label,
        "ActionID": "A",
        "Placement": "Hero",
        "Rank": 1,
        "Outcome": "Impression",
        "Propensity": 0.1,
        "DecisionTime": when,
        "Row": label,
    }


@pytest.mark.unit
def test_normal_processor_stays_current_only_while_frequency_gets_marked_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    history_row = _frequency_row("history", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    history_row["LegacyOnly"] = "legacy"
    current_row = _frequency_row("current", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))
    current_row["CurrentOnly"] = "current-era"
    frames = {
        history_file: pl.DataFrame([history_row]).lazy(),
        current_file: pl.DataFrame([current_row]).lazy(),
    }

    def fake_read(_reader: object, files: tuple[Path, ...]) -> pl.LazyFrame:
        assert len(files) == 1
        return frames[files[0]]

    monkeypatch.setattr(runner, "read", fake_read)
    normal = _processor(frequency=False)
    frequency = _processor(frequency=True)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )

    raw, current, bounded = runner._prepare_chunk_frames(
        _source(),
        [normal, frequency],
        plan,
    )
    assert raw.collect().get_column("Row").to_list() == ["current"]
    assert bounded is not None
    inputs = runner._processor_input_frames([normal, frequency], current, bounded)
    normal_rows, frequency_rows = pl.collect_all([frame for _, frame in inputs])

    assert normal_rows.select("Row", "CurrentOnly", "Transformed").to_dict(as_series=False) == {
        "Row": ["current"],
        "CurrentOnly": ["current-era"],
        "Transformed": [True],
    }
    assert frequency_rows.get_column("CustomerID").to_list() == ["history", "current"]
    assert frequency_rows.get_column("Row").to_list() == [None, "current"]
    assert "LegacyOnly" not in frequency_rows.columns
    assert frequency_rows.get_column("CurrentOnly").to_list() == [None, "current-era"]
    assert frequency_rows.get_column("Transformed").to_list() == [None, True]
    assert frequency_rows.get_column(runner.TARGET_CHUNK_COLUMN).to_list() == [
        False,
        True,
    ]


@pytest.mark.unit
def test_frequency_history_preserves_chunk_scoped_source_dedup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    history_row = _frequency_row("history", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    history_row["Key"] = "same"
    current_row = _frequency_row("current", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))
    current_row["Key"] = "same"
    frames = {
        history_file: pl.DataFrame([history_row]).lazy(),
        current_file: pl.DataFrame([current_row]).lazy(),
    }

    def fake_read(_reader: object, files: tuple[Path, ...]) -> pl.LazyFrame:
        assert len(files) == 1
        return frames[files[0]]

    monkeypatch.setattr(runner, "read", fake_read)
    source = model.Source.model_validate(
        {
            "id": "events",
            "reader": {"kind": "parquet", "file_pattern": "data/*.parquet"},
            "transforms": [{"kind": "dedup", "keys": ["Key"]}],
        }
    )
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )

    _raw, _current, bounded = runner._prepare_chunk_frames(
        source,
        [_processor(frequency=True)],
        plan,
    )

    assert bounded is not None
    assert bounded.collect().select("CustomerID", runner.TARGET_CHUNK_COLUMN).rows() == [
        ("history", False),
        ("current", True),
    ]


@pytest.mark.unit
def test_persistent_frequency_preparation_does_not_read_raw_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    frames = {
        history_file: pl.DataFrame(
            [_frequency_row("history", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))]
        ).lazy(),
        current_file: pl.DataFrame(
            [_frequency_row("current", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))]
        ).lazy(),
    }
    read_files: list[Path] = []

    def fake_read(_reader: object, files: tuple[Path, ...]) -> pl.LazyFrame:
        read_files.extend(files)
        return frames[files[0]]

    monkeypatch.setattr(runner, "read", fake_read)
    config = _processor(frequency=True).config.model_copy(
        update={
            "checkpoint": model.FrequencyResponseCheckpoint(
                mode="persistent_sharded",
                shards=8,
            )
        }
    )
    processor = FrequencyResponseProcessor(config, computation_hash="a" * 64)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )

    _raw, _current, bounded = runner._prepare_chunk_frames(
        _source(),
        [processor],
        plan,
    )

    assert bounded is None
    assert read_files == [current_file]


@pytest.mark.unit
def test_persistent_history_aggregation_scans_only_narrow_history_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    history_file.write_bytes(b"history")
    current_file.write_bytes(b"current")
    history_row = _frequency_row("history", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    history_row["CustomerID"] = "customer"
    current_row = _frequency_row("current", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))
    current_row["CustomerID"] = "customer"
    frames = {
        history_file: pl.DataFrame([history_row]).lazy(),
        current_file: pl.DataFrame([current_row]).lazy(),
    }
    monkeypatch.setattr(runner, "read", lambda _reader, files: frames[files[0]])
    config = _processor(frequency=True).config.model_copy(
        update={
            "checkpoint": model.FrequencyResponseCheckpoint(
                mode="persistent_sharded",
                shards=8,
            )
        }
    )
    processor = FrequencyResponseProcessor(config, computation_hash="a" * 64)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )
    assert (
        runner._ensure_persistent_frequency_checkpoints(
            tmp_path,
            _source(),
            [processor],
            [plan],
        )
        == {}
    )

    validations: list[tuple[str, object]] = []
    original_load = runner.load_processor_state_manifest

    def track_manifest_load(*args: Any, **kwargs: Any) -> runner.CheckpointManifest | None:
        validations.append((kwargs["chunk_id"], kwargs["validate"]))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(runner, "load_processor_state_manifest", track_manifest_load)
    assert (
        runner._ensure_persistent_frequency_checkpoints(
            tmp_path,
            _source(),
            [processor],
            [plan],
        )
        == {}
    )
    assert set(validations) == {
        ("2024-01-01", "history"),
        ("2024-01-02", "target"),
    }

    scans: list[tuple[str, str]] = []
    original_scan = runner.scan_processor_state_shard

    def track_scan(
        manifest: runner.CheckpointManifest,
        shard_id: int,
        *,
        payload: CheckpointPayload = "target",
    ) -> pl.LazyFrame:
        scans.append((manifest.chunk_id, payload))
        return original_scan(manifest, shard_id, payload=payload)

    monkeypatch.setattr(runner, "scan_processor_state_shard", track_scan)
    ctx = ChunkContext(
        pipeline_run_id="run",
        chunk_id="2024-01-02",
        created_at=dt.datetime(2024, 1, 2, 13, tzinfo=dt.UTC),
    )

    runner._collect_persistent_frequency_frames(
        tmp_path,
        _source(),
        [processor],
        plan,
        ctx,
    )

    assert ("2024-01-02", "target") in scans
    assert ("2024-01-01", "history") in scans
    assert ("2024-01-01", "target") not in scans


@pytest.mark.unit
def test_persistent_checkpoint_prepass_streams_and_rejects_customer_dtype_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    history_file.write_bytes(b"history")
    current_file.write_bytes(b"current")
    history_row = _frequency_row("1", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    history_row["CustomerID"] = 1
    current_row = _frequency_row("1", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))
    frames = {
        history_file: pl.DataFrame([history_row]).lazy(),
        current_file: pl.DataFrame([current_row]).lazy(),
    }
    monkeypatch.setattr(runner, "read", lambda _reader, files: frames[files[0]])

    def fail_collect_all(*args: object, **kwargs: object) -> None:
        raise AssertionError("checkpoint prepass must not collect full daily frames")

    monkeypatch.setattr(runner.pl, "collect_all", fail_collect_all)
    config = _processor(frequency=True).config.model_copy(
        update={
            "checkpoint": model.FrequencyResponseCheckpoint(
                mode="persistent_sharded",
                shards=8,
            )
        }
    )
    processor = FrequencyResponseProcessor(config, computation_hash="a" * 64)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )

    failures = runner._ensure_persistent_frequency_checkpoints(
        tmp_path,
        _source(),
        [processor],
        [plan],
    )

    assert "2024-01-02" in failures
    assert "customer dtype drift" in failures["2024-01-02"]
    assert "Int64" in failures["2024-01-02"]
    assert "String" in failures["2024-01-02"]


@pytest.mark.unit
def test_empty_checkpoint_does_not_create_false_customer_dtype_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    history_file.write_bytes(b"history")
    current_file.write_bytes(b"current")
    history_row = _frequency_row("empty", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    history_row["CustomerID"] = None
    current_row = _frequency_row("customer", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))
    frames = {
        history_file: pl.DataFrame([history_row]).lazy(),
        current_file: pl.DataFrame([current_row]).lazy(),
    }
    monkeypatch.setattr(runner, "read", lambda _reader, files: frames[files[0]])
    config = _processor(frequency=True).config.model_copy(
        update={
            "checkpoint": model.FrequencyResponseCheckpoint(
                mode="persistent_sharded",
                shards=8,
            )
        }
    )
    processor = FrequencyResponseProcessor(config, computation_hash="a" * 64)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )

    assert (
        runner._ensure_persistent_frequency_checkpoints(tmp_path, _source(), [processor], [plan])
        == {}
    )


@pytest.mark.unit
def test_checkpoint_integrity_failure_is_isolated_and_force_rebuilds_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_file = tmp_path / "2024-01-01.parquet"
    current_file.write_bytes(b"source")
    original = pl.DataFrame(
        [_frequency_row("customer", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))]
    ).lazy()
    monkeypatch.setattr(runner, "read", lambda _reader, _files: original)
    config = _processor(frequency=True).config.model_copy(
        update={
            "checkpoint": model.FrequencyResponseCheckpoint(
                mode="persistent_sharded",
                shards=8,
            )
        }
    )
    processor = FrequencyResponseProcessor(config, computation_hash="a" * 64)
    plan = runner._ChunkPlan(Chunk("2024-01-01", (current_file,)))

    assert (
        runner._ensure_persistent_frequency_checkpoints(tmp_path, _source(), [processor], [plan])
        == {}
    )
    shard = next(
        (tmp_path / ".valuestream" / "state" / "frequency_response").glob(
            "**/target-shard=*.parquet"
        )
    )
    pl.read_parquet(shard).with_columns(pl.lit("TAMPERED").alias("ActionID")).write_parquet(shard)

    failed = runner._ensure_persistent_frequency_checkpoints(
        tmp_path, _source(), [processor], [plan]
    )
    assert "does not match manifest" in failed["2024-01-01"]

    assert (
        runner._ensure_persistent_frequency_checkpoints(
            tmp_path,
            _source(),
            [processor],
            [plan],
            force=True,
        )
        == {}
    )
    assert "TAMPERED" not in pl.read_parquet(shard).get_column("ActionID").to_list()


@pytest.mark.unit
def test_each_history_chunk_must_supply_frequency_keys_before_relaxed_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    history_row = _frequency_row("history", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))
    history_row.pop("Placement")
    frames = {
        history_file: pl.DataFrame([history_row]).lazy(),
        current_file: pl.DataFrame(
            [_frequency_row("current", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))]
        ).lazy(),
    }

    monkeypatch.setattr(runner, "read", lambda _reader, files: frames[files[0]])
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )

    with pytest.raises(
        ValueError,
        match=r"history chunk '2024-01-01'.*missing Placement",
    ):
        runner._prepare_chunk_frames(
            _source(),
            [_processor(frequency=True)],
            plan,
        )


@pytest.mark.unit
def test_frequency_input_validation_requires_raw_bindings_and_marker_not_derived_columns() -> None:
    processor = _processor(frequency=True)
    schema_fields: list[tuple[str, pl.DataType | type[pl.DataType]]] = [
        ("CustomerID", pl.String),
        ("InteractionID", pl.String),
        ("ActionID", pl.String),
        ("Placement", pl.String),
        ("Rank", pl.Int64),
        ("Outcome", pl.String),
        ("Propensity", pl.Float64),
        ("DecisionTime", pl.Datetime("us")),
        (runner.TARGET_CHUNK_COLUMN, pl.Boolean),
    ]
    schema = pl.Schema(schema_fields)

    # ExposureBucket, Day, ClickedContact, and RunnerPropensity are all
    # processor-derived and therefore must not be required on raw input.
    runner._validate_processor_input_columns([processor], schema)

    missing_marker = pl.Schema(
        {name: dtype for name, dtype in schema.items() if name != runner.TARGET_CHUNK_COLUMN}
    )
    with pytest.raises(ValueError, match=runner.TARGET_CHUNK_COLUMN):
        runner._validate_processor_input_columns([processor], missing_marker)


@pytest.mark.unit
def test_target_schema_gap_cannot_be_masked_by_a_history_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    frames = {
        history_file: pl.DataFrame(
            {
                "CustomerID": ["history"],
                "InteractionID": ["history"],
                "ActionID": ["A"],
                "Placement": ["Hero"],
                "Rank": [1],
                "Outcome": ["Impression"],
                "Propensity": [0.1],
                "DecisionTime": [dt.datetime(2024, 1, 1, tzinfo=dt.UTC)],
            }
        ).lazy(),
        current_file: pl.DataFrame(
            {
                "CustomerID": ["current"],
                "InteractionID": ["current"],
                "ActionID": ["A"],
                "Placement": ["Hero"],
                "Rank": [1],
                "Outcome": ["Impression"],
                "DecisionTime": [dt.datetime(2024, 1, 2, tzinfo=dt.UTC)],
            }
        ).lazy(),
    }

    monkeypatch.setattr(runner, "read", lambda _reader, files: frames[files[0]])
    processor = _processor(frequency=True)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )
    _raw, current, bounded = runner._prepare_chunk_frames(
        _source(),
        [processor],
        plan,
    )

    assert bounded is not None
    # Narrow history projection prevents a history-only Propensity field from
    # masking the target-day schema gap in the relaxed union.
    assert "Propensity" not in bounded.collect_schema()
    assert "Propensity" not in current.collect_schema()
    with pytest.raises(ValueError, match=r"target chunk.*Propensity"):
        runner._validate_frequency_current_input_columns(
            [processor],
            current.collect_schema(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        (
            "DecisionTime",
            dt.date(2024, 1, 2),
            r"target chunk.*datetime decision.*got Date",
        ),
        ("Rank", None, r"target chunk.*integer rank.*got Null"),
    ],
)
def test_history_dtype_promotion_cannot_mask_invalid_target_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    invalid_value: object,
    message: str,
) -> None:
    history_file = tmp_path / "2024-01-01.parquet"
    current_file = tmp_path / "2024-01-02.parquet"
    current_row = _frequency_row("current", dt.datetime(2024, 1, 2, 12, tzinfo=dt.UTC))
    current_row[column] = invalid_value
    frames = {
        history_file: pl.DataFrame(
            [_frequency_row("history", dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC))]
        ).lazy(),
        current_file: pl.DataFrame([current_row]).lazy(),
    }

    monkeypatch.setattr(runner, "read", lambda _reader, files: frames[files[0]])
    processor = _processor(frequency=True)
    plan = runner._ChunkPlan(
        Chunk("2024-01-02", (current_file,)),
        (Chunk("2024-01-01", (history_file,)),),
    )
    _raw, current, bounded = runner._prepare_chunk_frames(
        _source(),
        [processor],
        plan,
    )

    assert bounded is not None
    current_dtype = current.collect_schema()[column]
    bounded_dtype = bounded.collect_schema()[column]
    if column == "DecisionTime":
        assert current_dtype == pl.Date
        assert bounded_dtype.base_type() == pl.Datetime
    else:
        assert current_dtype == pl.Null
        assert bounded_dtype.is_integer()
    with pytest.raises(ValueError, match=message):
        runner._validate_frequency_current_input_columns(
            [processor],
            current.collect_schema(),
        )


@pytest.mark.unit
def test_lookback_uses_calendar_days_without_substituting_for_missing_days(
    tmp_path: Path,
) -> None:
    chunks = [
        Chunk(day, (tmp_path / f"{day}.parquet",))
        for day in ("2024-01-01", "2024-01-03", "2024-01-08", "2024-01-09")
    ]

    plans = runner._plan_source_chunks(
        _source(),
        chunks,
        [_processor(frequency=True, window_hours=168)],
    )
    history_by_target = {
        plan.chunk_id: [chunk.chunk_id for chunk in plan.history_chunks] for plan in plans
    }

    assert history_by_target == {
        "2024-01-01": [],
        "2024-01-03": ["2024-01-01"],
        "2024-01-08": ["2024-01-01", "2024-01-03"],
        # Jan 1 is eight calendar days behind Jan 9 and must not be substituted
        # merely because several intervening source days are absent.
        "2024-01-09": ["2024-01-03", "2024-01-08"],
    }


@pytest.mark.unit
def test_partition_lag_padding_extends_dependency_days_not_window_semantics(
    tmp_path: Path,
) -> None:
    chunks = [
        Chunk(day, (tmp_path / f"{day}.parquet",))
        for day in ("2024-01-01", "2024-01-02", "2024-01-09")
    ]

    plans = runner._plan_source_chunks(
        _source(),
        chunks,
        [
            _processor(
                frequency=True,
                window_hours=168,
                partition_lag_hours=24,
            )
        ],
    )

    jan_nine = next(plan for plan in plans if plan.chunk_id == "2024-01-09")
    assert [chunk.chunk_id for chunk in jan_nine.history_chunks] == [
        "2024-01-01",
        "2024-01-02",
    ]


@pytest.mark.unit
def test_frequency_chunk_id_parse_failure_precedes_run_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    catalog = SimpleNamespace(
        pipelines=SimpleNamespace(sources=[source], workspace="test"),
    )
    monkeypatch.setattr(runner, "load", lambda _workspace: catalog)
    monkeypatch.setattr(
        runner,
        "validate_catalog",
        lambda _catalog, **_kwargs: SimpleNamespace(ok=True, issues=[]),
    )
    monkeypatch.setattr(
        runner,
        "_processors_for_source",
        lambda _catalog, _source_id: [_processor(frequency=True)],
    )
    monkeypatch.setattr(
        runner,
        "discover",
        lambda _workspace, _source: [Chunk("20240101", (tmp_path / "day.parquet",))],
    )

    def unexpected_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("planning must reject the chunk ID before run metadata writes")

    monkeypatch.setattr(runner, "_record_config_versions", unexpected_write)
    monkeypatch.setattr(runner.ledger, "start_run", unexpected_write)

    with pytest.raises(ValueError, match=r"ISO YYYY-MM-DD.*20240101"):
        runner._run_source_locked(tmp_path, "events")


@pytest.mark.unit
def test_dependency_fingerprints_are_shared_by_recovery_and_done_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    chunks = []
    for offset in range(2):
        day = dt.date(2024, 1, 1) + dt.timedelta(days=offset)
        path = tmp_path / f"{day.isoformat()}.parquet"
        path.write_text(day.isoformat(), encoding="utf-8")
        chunks.append(Chunk(day.isoformat(), (path,)))
    catalog = SimpleNamespace(
        pipelines=SimpleNamespace(sources=[source], workspace="test"),
    )
    processor = _processor(frequency=True)
    observed: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(runner, "load", lambda _workspace: catalog)
    monkeypatch.setattr(
        runner,
        "validate_catalog",
        lambda _catalog, **_kwargs: SimpleNamespace(ok=True, issues=[]),
    )
    monkeypatch.setattr(
        runner,
        "_processors_for_source",
        lambda _catalog, _source_id: [processor],
    )
    monkeypatch.setattr(runner, "discover", lambda _workspace, _source: chunks)
    monkeypatch.setattr(runner, "source_computation_hash", lambda *_args: "config")
    monkeypatch.setattr(runner, "_record_config_versions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "refresh_aggregate_views", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_SUPPORTED_TARGET_GRAINS",
        set(),
    )
    monkeypatch.setattr(runner.ledger, "ensure", lambda *_args, **_kwargs: None)

    def recover(*_args: object, **kwargs: object) -> tuple[()]:
        observed["recovery"] = dict(cast(dict[str, str], kwargs["file_hashes"]))
        return ()

    def done(*_args: object, **kwargs: object) -> set[str]:
        observed["done"] = dict(cast(dict[str, str], kwargs["file_hashes"]))
        return set(observed["done"])

    monkeypatch.setattr(runner.ledger, "recover_stale_runs", recover)
    monkeypatch.setattr(runner.ledger, "done_chunk_ids", done)
    monkeypatch.setattr(runner.ledger, "start_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.ledger, "finalize_run", lambda *_args, **_kwargs: None)

    result = runner._run_source_locked(tmp_path, "events")

    expected_plans = runner._plan_source_chunks(source, chunks, [processor])
    expected = {
        plan.chunk_id: ledger.file_fingerprint(plan.dependency_files) for plan in expected_plans
    }
    assert result.chunks_skipped == 2
    assert observed == {"recovery": expected, "done": expected}


@pytest.mark.unit
def test_changed_day_invalidates_itself_and_next_seven_days_but_not_day_eight(
    tmp_path: Path,
) -> None:
    start = dt.date(2024, 1, 1)
    chunks: list[Chunk] = []
    for offset in range(9):
        day = start + dt.timedelta(days=offset)
        path = tmp_path / f"{day.isoformat()}.parquet"
        path.write_text("original", encoding="utf-8")
        chunks.append(Chunk(day.isoformat(), (path,)))

    processor = _processor(frequency=True, window_hours=168)
    plans = runner._plan_source_chunks(_source(), chunks, [processor])
    initial_hashes = {
        plan.chunk_id: ledger.file_fingerprint(plan.dependency_files) for plan in plans
    }
    ledger.ensure(tmp_path)
    ledger.start_run(
        tmp_path,
        run_id="00000000-0000-0000-0000-000000000001",
        workspace="test",
        source_id="events",
        config_hash="config",
        started_at=dt.datetime.now(dt.UTC),
        chunks_total=len(plans),
    )
    for plan in plans:
        ledger.insert_chunk(
            tmp_path,
            source_id="events",
            chunk_id=plan.chunk_id,
            files=plan.dependency_files,
            rows_in=1,
            rows_kept=1,
            started_at=dt.datetime.now(dt.UTC),
            finished_at=dt.datetime.now(dt.UTC),
            status="ok",
            error=None,
            pipeline_run_id="00000000-0000-0000-0000-000000000001",
        )
    ledger.finalize_run(
        tmp_path,
        run_id="00000000-0000-0000-0000-000000000001",
        finished_at=dt.datetime.now(dt.UTC),
        status="ok",
        rows_in=len(plans),
        rows_kept=len(plans),
        chunks_total=len(plans),
        chunks_ok=len(plans),
        chunks_failed=0,
    )

    assert ledger.done_chunk_ids(
        tmp_path,
        source_id="events",
        config_hash="config",
        file_hashes=initial_hashes,
    ) == set(initial_hashes)
    chunks[0].files[0].write_text("changed-and-longer", encoding="utf-8")
    changed_hashes = {
        plan.chunk_id: ledger.file_fingerprint(plan.dependency_files) for plan in plans
    }
    still_done = ledger.done_chunk_ids(
        tmp_path,
        source_id="events",
        config_hash="config",
        file_hashes=changed_hashes,
    )

    assert still_done == {(start + dt.timedelta(days=8)).isoformat()}
    assert {
        chunk_id
        for chunk_id in initial_hashes
        if initial_hashes[chunk_id] != changed_hashes[chunk_id]
    } == {(start + dt.timedelta(days=offset)).isoformat() for offset in range(8)}

    with duckdb.connect(str(tmp_path / "meta" / "chunks.duckdb"), read_only=True) as conn:
        raw_files = conn.execute(
            "SELECT files FROM chunks WHERE chunk_id = ?",
            ((start + dt.timedelta(days=7)).isoformat(),),
        ).fetchone()
    assert raw_files is not None
    assert set(json.loads(str(raw_files[0]))) == {str(chunk.files[0]) for chunk in chunks[:8]}
