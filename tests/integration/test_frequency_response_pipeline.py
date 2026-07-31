"""End-to-end contracts for bounded-lookback frequency-response ingestion."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import polars as pl
import pytest

from valuestream.engine import run_source
from valuestream.query import query_metric_result
from valuestream.store.parquet import scan_aggregate

_START = dt.date(2024, 1, 1)
_DAYS = tuple((_START + dt.timedelta(days=offset)).isoformat() for offset in range(9))
_CLICK_DAYS = frozenset({_DAYS[1], _DAYS[3], _DAYS[6], _DAYS[8]})


def _write_catalog(workspace: Path) -> None:
    catalog = workspace / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
catalog_version: 2
workspace: frequency_response_e2e
sources:
  - id: ih
    reader:
      kind: parquet
      file_pattern: "data/*.parquet"
      group_by_filename: 'day_(\\d{4}-\\d{2}-\\d{2})'
    schema:
      timestamp_column: DecisionTime
      natural_key: [InteractionID, ActionID, Rank]
    transforms:
      - kind: derive_column
        output: Placement
        expression: {col: RawPlacement}
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
catalog_version: 2
processors:
  - id: frequency_response
    source: ih
    kind: frequency_response
    group_by: [Placement, ExposureBucket]
    time:
      property: DecisionTime
      grain: daily
      calendar: {timezone: UTC}
    columns:
      customer: CustomerID
      interaction: InteractionID
      action: ActionID
      placement: Placement
      rank: Rank
      outcome: Outcome
      propensity: Propensity
      priority: Priority
    positive_values: [Clicked]
    exposure_values: [Impression]
    candidate_values: [Pending, Impression, Clicked]
    window_hours: 168
    partition_lag_hours: 0
    max_frequency: 7
    frequency_column: ExposureBucket
    checkpoint: {mode: persistent_sharded, shards: 4}
    states:
      Contacts: {type: count}
      Clicks: {type: count, source_column: ClickedContact}
      ComparableContacts: {type: count, source_column: ComparableContact}
      ComparableClicks: {type: count, source_column: ComparableClick}
      RunnerPropensitySum: {type: value_sum, source_column: RunnerPropensity}
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        """
catalog_version: 2
metrics:
  FrequencyMarginalCTR:
    processor: frequency_response
    kind: formula
    expression:
      op: safe_div
      num: {col: Clicks}
      den: {col: Contacts}
""",
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text(
        "catalog_version: 2\ndashboards: []\n",
        encoding="utf-8",
    )


def _write_day(workspace: Path, day: str, *, focal_outcome: str) -> Path:
    timestamp = dt.datetime.combine(
        dt.date.fromisoformat(day),
        dt.time(hour=12),
        tzinfo=dt.UTC,
    )
    interaction = f"interaction-{day}"
    rows = [
        {
            "DecisionTime": timestamp,
            "CustomerID": "customer-1",
            "InteractionID": interaction,
            "ActionID": "focal-action",
            "RawPlacement": "Hero",
            "Rank": 1,
            "Outcome": focal_outcome,
            "Propensity": 0.8,
            "Priority": 80.0,
        },
        {
            "DecisionTime": timestamp,
            "CustomerID": "customer-1",
            "InteractionID": interaction,
            "ActionID": "runner-action",
            # Runner selection is interaction-wide, so it need not share the
            # focal placement.
            "RawPlacement": "Tile",
            "Rank": 2,
            "Outcome": "Pending",
            "Propensity": 0.25,
            "Priority": 20.0,
        },
    ]
    data = workspace / "data"
    data.mkdir(exist_ok=True)
    path = data / f"day_{day}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _seed_workspace(workspace: Path) -> None:
    _write_catalog(workspace)
    for day in _DAYS:
        outcome = "Clicked" if day in _CLICK_DAYS else "Impression"
        _write_day(workspace, day, focal_outcome=outcome)


def _frequency_rows(workspace: Path) -> dict[int, tuple[int, int, float]]:
    result = query_metric_result(
        workspace,
        "FrequencyMarginalCTR",
        group_by=["ExposureBucket"],
        grain="summary",
        include_state_columns=True,
    )
    return {
        int(row["ExposureBucket"]): (
            int(row["Contacts"]),
            int(row["Clicks"]),
            float(row["FrequencyMarginalCTR"]),
        )
        for row in result.rows.iter_rows(named=True)
    }


@pytest.mark.integration
def test_frequency_response_pipeline_replays_bounded_dependencies_and_hides_empty_rewrite(
    tmp_path: Path,
) -> None:
    """A corrected dependency republishes only its seven-day closure.

    The first run deliberately uses the process pool. The correction then
    changes the oldest target from a non-empty aggregate to an empty one, so
    the assertions also guard against an obsolete partial remaining visible.
    """

    _seed_workspace(tmp_path)

    initial = run_source(tmp_path, "ih", parallel=2)

    assert initial.status == "ok"
    assert (initial.chunks_ok, initial.chunks_skipped, initial.chunks_failed) == (9, 0, 0)
    # Every physical day contains two current rows. Dependency rereads must
    # not inflate source-run or per-chunk accounting.
    assert (initial.rows_in, initial.rows_kept) == (18, 18)
    assert {
        chunk.chunk_id: (chunk.rows_in, chunk.rows_kept) for chunk in initial.chunks
    } == dict.fromkeys(_DAYS, (2, 2))

    physical = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="frequency_response",
        grain="daily",
    ).collect()
    assert physical.height == 9
    assert set(physical["chunk_id"].to_list()) == set(_DAYS)
    with duckdb.connect(str(tmp_path / "meta" / "lineage.duckdb"), read_only=True) as conn:
        lineage = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT chunk_id), SUM(rows)
            FROM lineage
            WHERE source_id = 'ih' AND processor_id = 'frequency_response'
            """
        ).fetchone()
    assert lineage == (9, 9, 9)
    checkpoint_manifests = sorted(
        (tmp_path / ".valuestream" / "state" / "frequency_response").glob("**/manifest.json")
    )
    # Default retention is exactly seven history partitions plus current.
    assert len(checkpoint_manifests) == 8
    assert all(list(path.parent.glob("shard=*.parquet")) for path in checkpoint_manifests)

    before = query_metric_result(
        tmp_path,
        "FrequencyMarginalCTR",
        group_by=["ExposureBucket"],
        grain="summary",
        include_state_columns=True,
    )
    assert set(before.provenance.chunk_ids) == set(_DAYS)
    assert before.provenance.pipeline_run_ids == (initial.run_id,)
    assert _frequency_rows(tmp_path) == {
        1: (1, 0, 0.0),
        2: (1, 1, 1.0),
        3: (1, 0, 0.0),
        4: (1, 1, 1.0),
        5: (1, 0, 0.0),
        6: (1, 0, 0.0),
        7: (3, 2, pytest.approx(2 / 3)),
    }

    # Day 1 no longer contains a focal exposure. Its replacement processor
    # output is empty, while its dependency fingerprint changes for targets
    # through day 8. Day 9 begins exactly 192 hours later and is unaffected.
    _write_day(tmp_path, _DAYS[0], focal_outcome="Pending")
    corrected = run_source(tmp_path, "ih")

    assert corrected.status == "ok"
    assert (corrected.chunks_ok, corrected.chunks_skipped, corrected.chunks_failed) == (8, 1, 0)
    assert (corrected.rows_in, corrected.rows_kept) == (16, 16)
    statuses = {chunk.chunk_id: chunk.status for chunk in corrected.chunks}
    assert statuses == {
        **dict.fromkeys(_DAYS[:8], "ok"),
        _DAYS[8]: "skipped",
    }
    assert {
        chunk.chunk_id: (chunk.rows_in, chunk.rows_kept)
        for chunk in corrected.chunks
        if chunk.status == "ok"
    } == dict.fromkeys(_DAYS[:8], (2, 2))

    after = query_metric_result(
        tmp_path,
        "FrequencyMarginalCTR",
        group_by=["ExposureBucket"],
        grain="summary",
        include_state_columns=True,
    )
    assert set(after.provenance.chunk_ids) == set(_DAYS[1:])
    assert set(after.provenance.pipeline_run_ids) == {initial.run_id, corrected.run_id}
    assert _frequency_rows(tmp_path) == {
        1: (1, 1, 1.0),
        2: (1, 0, 0.0),
        3: (1, 1, 1.0),
        4: (1, 0, 0.0),
        5: (1, 0, 0.0),
        6: (1, 1, 1.0),
        7: (2, 1, 0.5),
    }

    # Old Parquet remains immutable on disk, but the current lineage/query
    # projection must select the newest successful key for each target. Day 1
    # has no replacement partial and therefore disappears rather than leaking
    # its obsolete bucket-1 aggregate.
    all_physical = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="frequency_response",
        grain="daily",
    ).collect()
    assert all_physical.height == 16
    assert all_physical.filter(pl.col("chunk_id") == _DAYS[0]).height == 1
    with duckdb.connect(str(tmp_path / "meta" / "lineage.duckdb"), read_only=True) as conn:
        lineage = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT chunk_id), SUM(rows)
            FROM lineage
            WHERE source_id = 'ih' AND processor_id = 'frequency_response'
            """
        ).fetchone()
    assert lineage == (16, 9, 16)
    # The corrected raw day is rebuilt for replay, then falls outside the
    # active eight-partition retention horizon. Unchanged retained checkpoints
    # are reused by all replayed targets.
    assert (
        len(
            list(
                (tmp_path / ".valuestream" / "state" / "frequency_response").glob(
                    "**/manifest.json"
                )
            )
        )
        == 8
    )


@pytest.mark.integration
def test_checkpoint_failure_only_fails_targets_that_depend_on_it(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    initial = run_source(tmp_path, "ih")
    assert initial.status == "ok"

    corrupted = next(
        (tmp_path / ".valuestream" / "state" / "frequency_response").glob(
            f"**/chunk={_DAYS[1]}/**/shard=*.parquet"
        )
    )
    with corrupted.open("ab") as handle:
        handle.write(b"corrupt")

    # Changing day 1 schedules targets 1..8. Only target 1 is independent of
    # the corrupt day-2 checkpoint; target 9 remains idempotently skipped.
    _write_day(tmp_path, _DAYS[0], focal_outcome="Pending")
    replay = run_source(tmp_path, "ih")

    assert replay.status == "failed"
    assert (replay.chunks_ok, replay.chunks_failed, replay.chunks_skipped) == (1, 7, 1)
    statuses = {chunk.chunk_id: chunk.status for chunk in replay.chunks}
    assert statuses[_DAYS[0]] == "ok"
    assert all(statuses[day] == "failed" for day in _DAYS[1:8])
    assert statuses[_DAYS[8]] == "skipped"
    assert all(
        "does not match manifest" in (chunk.error or "")
        for chunk in replay.chunks
        if chunk.status == "failed"
    )
