"""Phase 1 vertical-slice tests: parquet source -> binary aggregate -> query."""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

import valuestream.query.executor as query_executor
from valuestream.ai import sql_tool
from valuestream.cli import main
from valuestream.config.loader import load
from valuestream.engine import CleanRebuildError, clean_rebuild, run_source, run_workspace
from valuestream.query import AggregateNotReadyError, query_metric, query_metric_result
from valuestream.sdk import Workspace
from valuestream.store.duckdb_export import metric_export_db_path
from valuestream.store.duckdb_views import aggregate_view_name, views_db_path
from valuestream.store.parquet import aggregate_dir, scan_aggregate
from valuestream.store.vacuum import vacuum_workspace


def _write_catalog(ws: Path) -> None:
    catalog = ws / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: phase1_test
sources:
  - id: ih
    reader:
      kind: parquet
      file_pattern: "data/*.parquet"
      group_by_filename: '(\\d{8})'
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID, ActionID, Rank]
    transforms:
      - kind: derive_calendar
        from: OutcomeTime
        outputs: [Day, Month]
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: engagement
    source: ih
    kind: binary_outcome
    group_by: [Channel, Group]
    time:
      column: OutcomeTime
      grains: [Day, Month, Summary]
    outcome:
      column: Outcome
      positive_values: [Clicked]
      negative_values: [Impression, Pending]
    states:
      Count: {type: count}
      Positives: {type: count}
      Negatives: {type: count}
      UniqueCustomers_hll:
        type: hll
        source_column: CustomerID
        lg_k: 12
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        """
metrics:
  CTR:
    source: engagement
    kind: formula
    expression:
      op: safe_div
      num: {col: Positives}
      den:
        op: add
        args: [{col: Positives}, {col: Negatives}]
  UniqueCustomers:
    source: engagement
    kind: approx_distinct_count
    state: UniqueCustomers_hll
""",
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


def _write_data(ws: Path, filename: str, rows: list[dict[str, object]]) -> None:
    data_dir = ws / "data"
    data_dir.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(data_dir / filename)


def _seed_workspace(ws: Path) -> None:
    _write_catalog(ws)
    _write_data(
        ws,
        "ih_20240101000000.parquet",
        [
            _row("2024-01-01", "Web", "Cards", "c1", "Impression", "i1"),
            _row("2024-01-01", "Web", "Cards", "c1", "Clicked", "i2"),
            _row("2024-01-01", "Mobile", "Loans", "c2", "Impression", "i3"),
        ],
    )
    _write_data(
        ws,
        "ih_20240102000000.parquet",
        [
            _row("2024-01-02", "Web", "Cards", "c3", "Clicked", "i4"),
            _row("2024-01-02", "Web", "Cards", "c4", "Clicked", "i5"),
            _row("2024-01-02", "Web", "Cards", "c4", "Impression", "i6"),
        ],
    )


def _row(
    day: str,
    channel: str,
    group: str,
    customer: str,
    outcome: str,
    interaction: str,
) -> dict[str, object]:
    return {
        "OutcomeTime": dt.datetime.fromisoformat(day + "T10:00:00"),
        "Channel": channel,
        "Group": group,
        "CustomerID": customer,
        "Outcome": outcome,
        "InteractionID": interaction,
        "ActionID": "action",
        "Rank": 1,
    }


@pytest.mark.integration
def test_run_source_writes_provenance_and_queryable_ctr(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)

    result = run_source(tmp_path, "ih")

    assert result.status == "ok"
    assert result.chunks_ok == 2
    assert result.chunks_skipped == 0
    daily = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
    ).collect()
    assert {"pipeline_run_id", "chunk_id", "period", "created_at", "config_hash"} <= set(
        daily.columns
    )
    assert daily["period"].to_list() == ["2024-01", "2024-01", "2024-01"]

    ctr = query_metric(tmp_path, "CTR", group_by=["Channel", "Group"], grain="daily")
    assert ctr.columns == ["Day", "Channel", "Group", "CTR"]
    web_cards = ctr.filter((pl.col("Channel") == "Web") & (pl.col("Group") == "Cards")).sort("Day")
    assert web_cards["CTR"].to_list() == [0.5, pytest.approx(2 / 3)]
    raw_ctr = query_metric(
        tmp_path,
        "CTR",
        group_by=["Channel", "Group"],
        grain="daily",
        include_state_columns=True,
    )
    assert {"Count", "Positives", "Negatives", "UniqueCustomers_hll"} <= set(raw_ctr.columns)


@pytest.mark.integration
def test_query_metric_supports_operator_filters_having_order_and_compare(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")

    not_mobile = query_metric(
        tmp_path,
        "CTR",
        group_by=["Channel"],
        grain="daily",
        filters={"Channel": {"op": "ne", "value": "Mobile"}},
    )
    assert set(not_mobile.get_column("Channel").to_list()) == {"Web"}

    thresholded = query_metric(
        tmp_path,
        "CTR",
        group_by=["Channel", "Group"],
        grain="daily",
        having={"CTR": {"op": ">", "value": 0.55}},
    )
    assert thresholded.height == 1
    assert thresholded.get_column("CTR").to_list() == [pytest.approx(2 / 3)]

    ordered = query_metric(
        tmp_path,
        "CTR",
        group_by=["Channel", "Group"],
        grain="daily",
        order_by=["-CTR"],
    )
    ctr_values = ordered.get_column("CTR").to_list()
    assert ctr_values == sorted(ctr_values, reverse=True)

    top = query_metric(
        tmp_path,
        "CTR",
        group_by=["Channel", "Group"],
        grain="daily",
        top_n=1,
        top_n_by="CTR",
    )
    assert top.height == 1
    assert top.get_column("CTR").to_list() == [pytest.approx(2 / 3)]

    compared = query_metric(
        tmp_path,
        "CTR",
        group_by=["Channel", "Group"],
        grain="daily",
        compare="prior_period",
    )
    web_cards = compared.filter((pl.col("Channel") == "Web") & (pl.col("Group") == "Cards")).sort(
        "Day"
    )
    assert web_cards.get_column("CTR_prev").to_list() == [None, 0.5]
    assert web_cards.get_column("CTR_delta").to_list()[1] == pytest.approx(2 / 3 - 0.5)


@pytest.mark.integration
def test_governed_sql_queries_aggregate_views_and_masks_sketches(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    catalog = load(tmp_path)

    schema = sql_tool.sql_schema_summary(tmp_path, catalog)
    assert "aggregate_ih_engagement_daily" in schema
    assert "UniqueCustomers_hll" not in schema

    result = sql_tool.run_sql_query(
        tmp_path,
        """
        SELECT Channel, SUM(Positives) AS positives, SUM(Count) AS total
        FROM aggregate_ih_engagement_daily
        GROUP BY Channel
        ORDER BY positives DESC
        """,
        catalog=catalog,
    )

    assert result.rows.get_column("Channel").to_list() == ["Web", "Mobile"]
    assert result.rows.get_column("positives").to_list() == [3, 0]
    assert result.truncated is False
    assert "UniqueCustomers_hll" not in result.rows.columns


@pytest.mark.integration
def test_run_source_reports_chunk_progress_with_name_and_order(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    events = []

    result = run_source(tmp_path, "ih", progress_callback=events.append)

    assert result.status == "ok"
    assert [
        (event.chunk_name, event.chunk_order, event.chunks_total, event.status) for event in events
    ] == [
        ("20240102", 1, 2, "processing"),
        ("20240101", 2, 2, "processing"),
    ]
    assert all(event.chunk_name == event.chunk_id for event in events)
    assert all(event.files for event in events)

    skipped_events = []
    skipped = run_source(tmp_path, "ih", progress_callback=skipped_events.append)

    assert skipped.chunks_skipped == 2
    assert [
        (event.chunk_name, event.chunk_order, event.chunks_total, event.status)
        for event in skipped_events
    ] == [
        ("20240102", 1, 2, "skipped"),
        ("20240101", 2, 2, "skipped"),
    ]


@pytest.mark.integration
def test_run_source_debugging_logs_chunk_schema_and_rows(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_workspace(tmp_path)
    pipelines = tmp_path / "catalog" / "pipelines.yaml"
    pipelines.write_text(
        pipelines.read_text(encoding="utf-8").replace(
            "  - id: ih\n",
            "  - id: ih\n    debugging: true\n",
            1,
        ),
        encoding="utf-8",
    )

    caplog.set_level(logging.DEBUG, logger="valuestream.engine.runner")
    result = run_source(tmp_path, "ih")

    assert result.status == "ok"
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Chunk schema: source=ih" in message
        and "stage=raw" in message
        and "OutcomeTime:Datetime" in message
        for message in messages
    )
    assert any(
        "Chunk schema: source=ih" in message
        and "stage=transformed" in message
        and "Day:Date" in message
        for message in messages
    )
    assert any(
        "Chunk rows: source=ih" in message and "rows_in=3" in message and "rows_kept=3" in message
        for message in messages
    )
    assert any(
        "Processor frame: source=ih" in message
        and "processor=engagement" in message
        and "stage=base" in message
        and "rows=" in message
        for message in messages
    )
    assert any(
        "Processor frame: source=ih" in message
        and "processor=engagement" in message
        and "stage=daily" in message
        and "period_nulls=0" in message
        for message in messages
    )


@pytest.mark.integration
def test_summary_query_without_group_by_collapses_all_rows_and_sketches(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")

    summary = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="summary",
    ).collect()
    ctr = query_metric(tmp_path, "CTR", grain="summary")
    unique = query_metric(tmp_path, "UniqueCustomers", grain="summary")

    assert summary["period"].unique().to_list() == ["2024-01"]
    assert ctr.height == 1
    assert ctr["CTR"][0] == pytest.approx(0.5)
    assert unique.height == 1
    assert unique["UniqueCustomers"][0] == pytest.approx(4.0, rel=0.02)


@pytest.mark.integration
def test_summary_physical_aggregation_level_is_configurable(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    processors = tmp_path / "catalog" / "processors.yaml"
    processors.write_text(
        processors.read_text(encoding="utf-8").replace(
            "      grains: [Day, Month, Summary]\n",
            "      grains: [Day, Month, Summary]\n"
            "      aggregation_levels:\n"
            "        Summary: Quarter\n",
        ),
        encoding="utf-8",
    )

    run_source(tmp_path, "ih")

    summary = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="summary",
    ).collect()
    ctr = query_metric(tmp_path, "CTR", grain="summary")

    assert summary["period"].unique().to_list() == ["2024_Q1"]
    assert ctr["CTR"][0] == pytest.approx(0.5)


@pytest.mark.integration
def test_quarterly_query_falls_back_to_monthly_aggregate(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")

    ctr = query_metric(tmp_path, "CTR", grain="quarterly")

    assert ctr.columns == ["Quarter", "CTR"]
    assert ctr["Quarter"].to_list() == ["2024_Q1"]
    assert ctr["CTR"][0] == pytest.approx(0.5)


@pytest.mark.integration
def test_optional_quarterly_and_yearly_aggregates_are_materialized(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    pipelines = tmp_path / "catalog" / "pipelines.yaml"
    pipelines.write_text(
        pipelines.read_text(encoding="utf-8").replace(
            "        outputs: [Day, Month]\n",
            "        outputs: [Day, Month, Quarter, Year]\n",
        ),
        encoding="utf-8",
    )
    processors = tmp_path / "catalog" / "processors.yaml"
    processors.write_text(
        processors.read_text(encoding="utf-8").replace(
            "      grains: [Day, Month, Summary]\n",
            "      grains: [Day, Month, Quarter, Year, Summary]\n",
        ),
        encoding="utf-8",
    )

    run_source(tmp_path, "ih")

    quarterly = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="quarterly",
    ).collect()
    yearly = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="yearly",
    ).collect()
    assert quarterly["Quarter"].unique().to_list() == ["2024_Q1"]
    assert yearly["Year"].unique().to_list() == [2024]


@pytest.mark.integration
def test_variant_and_contingency_metrics_emit_complete_outputs(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    processors = tmp_path / "catalog" / "processors.yaml"
    processors.write_text(
        processors.read_text(encoding="utf-8").replace(
            "    states:\n",
            "    variant_column: ModelControlGroup\n    states:\n",
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "catalog" / "metrics.yaml"
    metrics.write_text(
        metrics.read_text(encoding="utf-8")
        + """
  Lift:
    source: engagement
    kind: variant_compare
    variant_column: ModelControlGroup
    test_role: Test
    control_role: Control
    outputs: [TestCTR, ControlCTR, Lift, Lift_Z_Score, Lift_P_Val, StdErr]
  Significance:
    source: engagement
    kind: contingency_test
    variant_column: ModelControlGroup
    tests: [chi2, g, z]
    outputs: [chi2_stat, g_stat, z_score]
""",
        encoding="utf-8",
    )
    rows = [
        {
            **_row("2024-01-01", "Web", "Cards", f"c{i}", outcome, f"i{i}"),
            "ModelControlGroup": role,
        }
        for i, (role, outcome) in enumerate(
            [
                *(("Control", "Clicked") for _ in range(2)),
                *(("Control", "Impression") for _ in range(8)),
                *(("Test", "Clicked") for _ in range(3)),
                *(("Test", "Impression") for _ in range(7)),
            ]
        )
    ]
    _write_data(tmp_path, "ih_20240101000000.parquet", rows)

    run_source(tmp_path, "ih")

    lift = query_metric(tmp_path, "Lift", group_by=["Channel"], grain="summary")
    significance = query_metric(tmp_path, "Significance", group_by=["Channel"], grain="summary")
    assert lift["Lift"][0] == pytest.approx(0.5)
    assert {"CTR", "Lift_Z_Score", "Lift_P_Val", "StdErr"} <= set(lift.columns)
    assert {"chi2_stat", "g_stat", "z_score"} <= set(significance.columns)


@pytest.mark.integration
def test_time_range_applies_when_query_falls_back_to_coarser_grain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Quarterly requests fall back to the stored monthly aggregate. A date range
    # must still filter by the stored grain's calendar column (Month), not be
    # silently dropped because the requested grain has no Day column.
    _write_catalog(tmp_path)
    _write_data(
        tmp_path,
        "ih_20240101000000.parquet",
        [
            _row("2024-01-10", "Web", "Cards", "c1", "Impression", "i1"),
            _row("2024-01-11", "Web", "Cards", "c2", "Impression", "i2"),
        ],
    )
    _write_data(
        tmp_path,
        "ih_20240201000000.parquet",
        [
            _row("2024-02-10", "Web", "Cards", "c3", "Clicked", "i3"),
            _row("2024-02-11", "Web", "Cards", "c4", "Impression", "i4"),
        ],
    )

    run_source(tmp_path, "ih")
    original_collect_all = pl.collect_all
    collect_all_calls = 0

    def counted_collect_all(lazy_frames: list[pl.LazyFrame]) -> list[pl.DataFrame]:
        nonlocal collect_all_calls
        collect_all_calls += 1
        return original_collect_all(lazy_frames)

    monkeypatch.setattr(query_executor.pl, "collect_all", counted_collect_all)

    # February only: CTR = 1 click / 2 = 0.5. Without the fix, January's zeros
    # would be merged into the same quarter, giving 1/4 = 0.25.
    ctr = query_metric(tmp_path, "CTR", grain="quarterly", start="2024-02-01")
    assert collect_all_calls == 1
    assert ctr.columns == ["Quarter", "CTR"]
    assert ctr["Quarter"].to_list() == ["2024_Q1"]
    assert ctr["CTR"][0] == pytest.approx(0.5)


@pytest.mark.integration
def test_time_range_without_rows_is_not_mistaken_for_unpublished_data(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")

    ctr = query_metric(tmp_path, "CTR", grain="daily", start="2030-01-01")

    assert ctr.is_empty()
    assert ctr.columns == ["Day", "CTR"]


@pytest.mark.integration
def test_query_rejects_unknown_group_by_instead_of_dropping_it(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")

    with pytest.raises(ValueError, match="group_by column 'missing_column'"):
        query_metric(tmp_path, "CTR", group_by=["missing_column"], grain="summary")


@pytest.mark.integration
def test_query_reports_stale_aggregates_after_processor_config_changes(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    processors = tmp_path / "catalog" / "processors.yaml"
    processors.write_text(
        processors.read_text(encoding="utf-8").replace(
            "      positive_values: [Clicked]\n",
            "      positive_values: [Conversion]\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AggregateNotReadyError, match="current processor configuration"):
        query_metric(tmp_path, "CTR", grain="summary")


@pytest.mark.integration
def test_query_uses_current_lineage_across_processor_state_schema_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ids = iter(
        [
            "ffffffff-ffff-4fff-8fff-ffffffffffff",
            "00000000-0000-4000-8000-000000000001",
        ]
    )
    monkeypatch.setattr(
        "valuestream.engine.runner.new_pipeline_run_id",
        lambda: next(run_ids),
    )
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    processors = tmp_path / "catalog" / "processors.yaml"
    processors.write_text(
        processors.read_text(encoding="utf-8").replace(
            "      UniqueCustomers_hll:\n",
            "      Channel_cpc:\n"
            "        type: cpc\n"
            "        source_column: Channel\n"
            "        lg_k: 11\n"
            "      UniqueCustomers_hll:\n",
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "catalog" / "metrics.yaml"
    metrics.write_text(
        metrics.read_text(encoding="utf-8")
        + """
  UniqueChannels:
    source: engagement
    kind: approx_distinct_count
    state: Channel_cpc
""",
        encoding="utf-8",
    )

    with pytest.raises(AggregateNotReadyError, match="backfill/reprocess"):
        query_metric(tmp_path, "UniqueChannels", grain="summary")

    run_source(tmp_path, "ih")
    result = query_metric(tmp_path, "UniqueChannels", grain="summary")

    assert result["UniqueChannels"].to_list() == [pytest.approx(2.0, abs=0.1)]


@pytest.mark.integration
def test_query_reports_stale_aggregates_after_source_behavior_changes(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    pipelines = tmp_path / "catalog" / "pipelines.yaml"
    pipelines.write_text(
        pipelines.read_text(encoding="utf-8").replace(
            "    transforms:\n",
            "    defaults: {NewBehaviorColumn: changed}\n    transforms:\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AggregateNotReadyError, match="current processor configuration"):
        query_metric(tmp_path, "CTR", grain="summary")


@pytest.mark.integration
def test_presentation_only_changes_do_not_reprocess_source(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    first = run_source(tmp_path, "ih")
    metrics = tmp_path / "catalog" / "metrics.yaml"
    metrics.write_text(
        metrics.read_text(encoding="utf-8").replace(
            "  CTR:\n",
            "  CTR:\n    description: presentation only\n",
        ),
        encoding="utf-8",
    )
    dashboards = tmp_path / "catalog" / "dashboards.yaml"
    dashboards.write_text(
        "dashboards:\n  - id: overview\n    title: New title\n    pages: []\n",
        encoding="utf-8",
    )

    second = run_source(tmp_path, "ih")

    assert first.chunks_ok == 2
    assert second.chunks_skipped == 2
    assert second.chunks_ok == 0


@pytest.mark.integration
def test_failed_chunk_writes_are_not_visible_to_metric_queries(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    processors = tmp_path / "catalog" / "processors.yaml"
    processors.write_text(
        processors.read_text(encoding="utf-8")
        + """
  - id: broken
    source: ih
    kind: binary_outcome
    group_by: [Channel]
    time:
      grains: [Summary]
    outcome:
      column: MissingOutcome
      positive_values: [Clicked]
      negative_values: [Impression]
    states:
      Count: {type: count}
      Positives: {type: count}
      Negatives: {type: count}
""",
        encoding="utf-8",
    )
    _write_data(
        tmp_path,
        "ih_20240101000000.parquet",
        [
            _row("2024-01-01", "Web", "Cards", "c1", "Clicked", "i1"),
            _row("2024-01-01", "Web", "Cards", "c2", "Impression", "i2"),
        ],
    )

    result = run_source(tmp_path, "ih")
    view = aggregate_view_name("ih", "engagement", "summary")
    with duckdb.connect(str(views_db_path(tmp_path)), read_only=True) as conn:
        view_exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.views WHERE table_name = ?",
            (view,),
        ).fetchone()

    assert result.status == "failed"
    assert result.chunks_failed == 1
    assert result.chunks[0].error is not None
    assert "processor input columns are missing: broken: MissingOutcome" in result.chunks[0].error
    with pytest.raises(FileNotFoundError, match="run ingestion first"):
        query_metric(tmp_path, "CTR", group_by=["Channel"], grain="summary")
    assert view_exists == (0,)


@pytest.mark.integration
def test_failed_replacement_keeps_last_successful_chunk_visible(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    first = run_source(tmp_path, "ih")
    before = query_metric(tmp_path, "CTR", grain="summary")
    changed_path = tmp_path / "data" / "ih_20240101000000.parquet"
    pl.read_parquet(changed_path).drop("Group").write_parquet(changed_path)

    failed = run_source(tmp_path, "ih")
    after = query_metric(tmp_path, "CTR", grain="summary")

    assert first.status == "ok"
    assert failed.status == "failed"
    assert failed.chunks_failed == 1
    assert failed.chunks_skipped == 1
    assert before["CTR"].to_list() == after["CTR"].to_list()


@pytest.mark.integration
def test_run_records_config_history_and_file_lineage(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    _write_data(
        tmp_path,
        "ih_20240101000000.parquet",
        [_row("2024-01-01", "Web", "Cards", "c1", "Clicked", "i1")],
    )

    result = run_source(tmp_path, "ih")
    with duckdb.connect(str(tmp_path / "meta" / "config_versions.duckdb"), read_only=True) as conn:
        versions = conn.execute(
            "SELECT config_hash, yaml FROM config_versions ORDER BY config_hash"
        ).fetchall()
    with duckdb.connect(str(tmp_path / "meta" / "lineage.duckdb"), read_only=True) as conn:
        lineage = conn.execute(
            "SELECT pipeline_run_id, chunk_id, partial_path, config_hash, rows FROM lineage"
        ).fetchall()

    assert result.status == "ok"
    assert len(versions) == 3
    assert all(config_hash and yaml.startswith("{") for config_hash, yaml in versions)
    assert len(lineage) == 3
    assert all(str(run_id) == result.run_id for run_id, *_ in lineage)
    assert all(Path(path).is_file() for _, _, path, _, _ in lineage)
    assert all(config_hash and rows > 0 for _, _, _, config_hash, rows in lineage)


@pytest.mark.integration
def test_metric_query_result_exposes_stable_provenance(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run = run_source(tmp_path, "ih")

    result = query_metric_result(tmp_path, "CTR", grain="summary")
    provenance = result.provenance

    assert result.rows.height == 1
    assert provenance.metric == "CTR"
    assert provenance.source_id == "ih"
    assert provenance.processor_id == "engagement"
    assert provenance.requested_grain == "summary"
    assert provenance.stored_grain == "summary"
    assert provenance.pipeline_run_ids == (run.run_id,)
    assert provenance.chunk_ids == ("20240101", "20240102")
    assert provenance.aggregate_rows_scanned == 3
    assert len(provenance.catalog_hash) == 64
    assert len(provenance.computation_hash) == 64
    assert provenance.latest_created_at is not None


@pytest.mark.integration
def test_ledger_records_input_rows_before_source_filters(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    pipelines = tmp_path / "catalog" / "pipelines.yaml"
    pipelines.write_text(
        pipelines.read_text(encoding="utf-8")
        + """
      - kind: filter
        expression: {op: eq, column: Channel, value: Web}
""",
        encoding="utf-8",
    )
    _write_data(
        tmp_path,
        "ih_20240101000000.parquet",
        [
            _row("2024-01-01", "Web", "Cards", "c1", "Clicked", "i1"),
            _row("2024-01-01", "Mobile", "Loans", "c2", "Impression", "i2"),
        ],
    )

    result = run_source(tmp_path, "ih")

    assert result.rows_in == 2
    assert result.rows_kept == 1


@pytest.mark.integration
def test_run_source_can_materialize_transforms_before_processor_fanout(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    pipelines = tmp_path / "catalog" / "pipelines.yaml"
    pipelines.write_text(
        pipelines.read_text(encoding="utf-8").replace(
            "    transforms:\n",
            "    materialize_transforms: true\n    transforms:\n",
        ),
        encoding="utf-8",
    )

    result = run_source(tmp_path, "ih")
    ctr = query_metric(tmp_path, "CTR", group_by=["Channel", "Group"], grain="daily")

    assert result.status == "ok"
    assert result.rows_in == 6
    assert result.rows_kept == 6
    web_cards = ctr.filter((pl.col("Channel") == "Web") & (pl.col("Group") == "Cards")).sort("Day")
    assert web_cards["CTR"].to_list() == [0.5, pytest.approx(2 / 3)]


@pytest.mark.integration
def test_run_source_refreshes_duckdb_aggregate_views(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)

    run_source(tmp_path, "ih")

    view = aggregate_view_name("ih", "engagement", "daily")
    with duckdb.connect(str(views_db_path(tmp_path)), read_only=True) as conn:
        rows = conn.execute(f'SELECT COUNT(*) FROM "{view}"').fetchone()
    assert rows == (3,)


@pytest.mark.integration
def test_export_duckdb_cli_writes_one_table_per_metric_for_selected_grain(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")

    runner = CliRunner()
    result = runner.invoke(main, ["export-duckdb", str(tmp_path), "--grain", "Summary"])

    assert result.exit_code == 0, result.output
    export_path = metric_export_db_path(tmp_path, "summary")
    assert export_path.exists()
    with duckdb.connect(str(export_path), read_only=True) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        ctr_rows = conn.execute(
            'SELECT Channel, "Group", CTR, _valuestream_grain '
            'FROM metric_ctr_summary ORDER BY Channel, "Group"'
        ).fetchall()
        manifest_rows = conn.execute(
            "SELECT metric_name, table_name, rows, status "
            "FROM valuestream_metric_export_manifest ORDER BY metric_name"
        ).fetchall()

    assert {"metric_ctr_summary", "metric_uniquecustomers_summary"} <= tables
    assert ctr_rows == [
        ("Mobile", "Loans", 0.0, "summary"),
        ("Web", "Cards", pytest.approx(0.6), "summary"),
    ]
    assert manifest_rows == [
        ("CTR", "metric_ctr_summary", 2, "exported"),
        ("UniqueCustomers", "metric_uniquecustomers_summary", 2, "exported"),
    ]


@pytest.mark.integration
def test_rerun_skips_done_chunks_and_new_file_processes_once(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)

    first = run_source(tmp_path, "ih")
    second = run_source(tmp_path, "ih")
    _write_data(
        tmp_path,
        "ih_20240103000000.parquet",
        [_row("2024-01-03", "Web", "Cards", "c5", "Clicked", "i7")],
    )
    third = run_source(tmp_path, "ih")
    _write_data(
        tmp_path,
        "ih_20240101000000.parquet",
        [_row("2024-01-01", "Web", "Cards", "c9", "Clicked", "i9")],
    )
    fourth = run_source(tmp_path, "ih")

    assert first.chunks_ok == 2
    assert second.chunks_ok == 0
    assert second.chunks_skipped == 2
    assert third.chunks_ok == 1
    assert third.chunks_skipped == 2
    assert fourth.chunks_ok == 1
    assert fourth.chunks_skipped == 2

    ctr = query_metric(tmp_path, "CTR", group_by=["Channel", "Group"], grain="summary")
    web_cards = ctr.filter((pl.col("Channel") == "Web") & (pl.col("Group") == "Cards"))
    assert web_cards["CTR"][0] == pytest.approx(4 / 5)


@pytest.mark.integration
def test_vacuum_removes_old_config_aggregate_files(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    stale_dir = (
        aggregate_dir(
            tmp_path,
            source_id="ih",
            processor_id="engagement",
            grain="daily",
        )
        / "period=1999-01"
    )
    stale_dir.mkdir(parents=True)
    stale = stale_dir / "part-stale.parquet"
    pl.DataFrame(
        {
            "chunk_id": ["stale"],
            "period": ["1999-01"],
            "config_hash": ["old-config"],
        }
    ).write_parquet(stale)

    result = vacuum_workspace(tmp_path, load(tmp_path), include_tmp=False)

    assert result.files_deleted == 1
    assert not stale.exists()


@pytest.mark.integration
def test_vacuum_removes_superseded_successful_partials_after_forced_run(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    before = query_metric(tmp_path, "CTR", grain="summary")
    run_source(tmp_path, "ih", force=True)
    aggregate_files = list((tmp_path / "aggregates").glob("*/*/*/period=*/*.parquet"))

    result = vacuum_workspace(tmp_path, load(tmp_path), include_tmp=False)
    after_files = list((tmp_path / "aggregates").glob("*/*/*/period=*/*.parquet"))
    after = query_metric(tmp_path, "CTR", grain="summary")

    assert len(aggregate_files) == 12
    assert result.files_deleted == 6
    assert len(after_files) == 6
    assert before["CTR"].to_list() == after["CTR"].to_list()


@pytest.mark.integration
def test_clean_rebuild_replaces_aggregate_files_and_retains_audit_history(
    tmp_path: Path,
) -> None:
    _seed_workspace(tmp_path)
    first = run_source(tmp_path, "ih")
    before = set((tmp_path / "aggregates").glob("*/*/*/period=*/*.parquet"))
    before_metric = query_metric(tmp_path, "CTR", grain="summary")

    result = clean_rebuild(tmp_path, source_ids=["ih"])

    after = set((tmp_path / "aggregates").glob("*/*/*/period=*/*.parquet"))
    after_metric = query_metric(tmp_path, "CTR", grain="summary")
    assert result.source_ids == ("ih",)
    assert result.chunks_rebuilt == 2
    assert result.vacuum.files_deleted == len(before)
    assert len(after) == len(before)
    assert before.isdisjoint(after)
    assert before_metric["CTR"].to_list() == after_metric["CTR"].to_list()
    assert scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
    ).select("pipeline_run_id").unique().collect().get_column("pipeline_run_id").to_list() == [
        result.runs[0].run_id
    ]
    with duckdb.connect(str(tmp_path / "meta" / "pipeline_runs.duckdb"), read_only=True) as conn:
        run_ids = {
            str(row[0])
            for row in conn.execute(
                "SELECT id FROM pipeline_runs WHERE source_id = 'ih'"
            ).fetchall()
        }
    assert {first.run_id, result.runs[0].run_id} <= run_ids


@pytest.mark.integration
def test_clean_rebuild_preserves_old_aggregates_when_source_discovers_no_chunks(
    tmp_path: Path,
) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    before = set((tmp_path / "aggregates").glob("*/*/*/period=*/*.parquet"))
    for path in (tmp_path / "data").glob("*.parquet"):
        path.unlink()

    with pytest.raises(CleanRebuildError, match="discovered no chunks"):
        clean_rebuild(tmp_path, source_ids=["ih"])

    assert set((tmp_path / "aggregates").glob("*/*/*/period=*/*.parquet")) == before


@pytest.mark.integration
def test_clean_rebuild_removes_aggregate_partials_for_inputs_no_longer_discovered(
    tmp_path: Path,
) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    (tmp_path / "data" / "ih_20240102000000.parquet").unlink()

    result = clean_rebuild(tmp_path, source_ids=["ih"])

    daily = scan_aggregate(
        tmp_path,
        source_id="ih",
        processor_id="engagement",
        grain="daily",
    ).collect()
    assert result.chunks_rebuilt == 1
    assert daily.get_column("chunk_id").unique().to_list() == ["20240101"]


@pytest.mark.integration
def test_vacuum_removes_malformed_orphan_aggregate_file(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    orphan_dir = (
        aggregate_dir(
            tmp_path,
            source_id="ih",
            processor_id="engagement",
            grain="daily",
        )
        / "period=2024-01"
    )
    orphan = orphan_dir / "part-orphan.parquet"
    pl.DataFrame(
        {
            "chunk_id": ["orphan"],
            "period": ["2024-01"],
            "config_hash": [
                scan_aggregate(
                    tmp_path,
                    source_id="ih",
                    processor_id="engagement",
                    grain="daily",
                )
                .select("config_hash")
                .first()
                .collect()
                .item()
            ],
        }
    ).write_parquet(orphan)

    result = vacuum_workspace(tmp_path, load(tmp_path), include_tmp=False)

    assert orphan in result.paths
    assert not orphan.exists()


@pytest.mark.integration
def test_hll_distinct_query_is_close_to_exact(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    rows = [
        _row("2024-01-01", "Web", "Cards", f"cust-{i}", "Clicked", f"i{i}") for i in range(1_000)
    ]
    _write_data(tmp_path, "ih_20240101000000.parquet", rows)

    run_source(tmp_path, "ih")
    result = query_metric(tmp_path, "UniqueCustomers", group_by=["Channel"], grain="summary")

    estimate = result.filter(pl.col("Channel") == "Web")["UniqueCustomers"][0]
    assert abs(estimate - 1_000) / 1_000 < 0.04


@pytest.mark.slow
@pytest.mark.integration
def test_hll_distinct_query_is_within_target_rse_on_one_million_rows(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    rows = {
        "OutcomeTime": [dt.datetime(2024, 1, 1, 10, 0)] * 1_000_000,
        "Channel": ["Web"] * 1_000_000,
        "Group": ["Cards"] * 1_000_000,
        "CustomerID": [f"cust-{i}" for i in range(1_000_000)],
        "Outcome": ["Clicked"] * 1_000_000,
        "InteractionID": [f"i{i}" for i in range(1_000_000)],
        "ActionID": ["action"] * 1_000_000,
        "Rank": [1] * 1_000_000,
    }
    _write_data(tmp_path, "ih_20240101000000.parquet", rows)

    run_source(tmp_path, "ih")
    result = query_metric(tmp_path, "UniqueCustomers", group_by=["Channel"], grain="summary")

    estimate = result.filter(pl.col("Channel") == "Web")["UniqueCustomers"][0]
    assert abs(estimate - 1_000_000) / 1_000_000 <= 0.016


@pytest.mark.integration
def test_workspace_sdk_runs_and_queries(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    workspace = Workspace(tmp_path)

    run = workspace.run_source("ih")
    frame = workspace.metric("CTR").by("Channel").grain("summary").to_polars()

    assert run.status == "ok"
    assert "CTR" in frame.columns


@pytest.mark.integration
def test_probe_cli_reports_transformed_schema_and_chunks(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["probe", str(tmp_path), "ih", "--limit", "1"])

    assert result.exit_code == 0
    assert "2 chunk(s), 2 file(s)" in result.output
    assert "calendar columns: Day, Month" in result.output
    assert "OutcomeTime" in result.output


@pytest.mark.integration
def test_query_cli_filters_with_where_clause(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "query",
            str(tmp_path),
            "CTR",
            "--by",
            "Channel",
            "--grain",
            "summary",
            "--where",
            "Channel=Web",
        ],
    )

    assert result.exit_code == 0
    assert "Web" in result.output
    assert "Mobile" not in result.output


@pytest.mark.integration
def test_query_cli_raw_includes_state_columns(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    run_source(tmp_path, "ih")
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "query",
            str(tmp_path),
            "CTR",
            "--by",
            "Channel",
            "--grain",
            "summary",
            "--raw",
        ],
    )

    assert result.exit_code == 0
    assert "Positives" in result.output
    assert "UniqueCustomers_hll" in result.output


@pytest.mark.integration
def test_run_source_parallel_matches_sequential_results(tmp_path: Path) -> None:
    seq_ws = tmp_path / "seq"
    par_ws = tmp_path / "par"
    for ws in (seq_ws, par_ws):
        ws.mkdir()
        _seed_workspace(ws)

    sequential = run_source(seq_ws, "ih")
    parallel = run_source(par_ws, "ih", parallel=2)

    assert sequential.status == "ok"
    assert parallel.status == "ok"
    assert parallel.chunks_ok == sequential.chunks_ok == 2
    seq_ctr = query_metric(seq_ws, "CTR", group_by=["Channel", "Group"], grain="summary").sort(
        ["Channel", "Group"]
    )
    par_ctr = query_metric(par_ws, "CTR", group_by=["Channel", "Group"], grain="summary").sort(
        ["Channel", "Group"]
    )
    assert seq_ctr.equals(par_ctr)

    # Parallel reruns skip chunks recorded by the parent-side ledger writes.
    rerun = run_source(par_ws, "ih", parallel=2)
    assert rerun.chunks_skipped == 2


@pytest.mark.integration
def test_run_workspace_runs_all_sources(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    result = run_workspace(tmp_path)

    assert result.status == "ok"
    assert result.sources_total == 1
    assert result.results[0].source_id == "ih"


@pytest.mark.integration
def test_run_cli_without_source_runs_workspace(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["run", str(tmp_path)])

    assert result.exit_code == 0
    assert "workspace run" in result.output
    assert "1 source(s)" in result.output


@pytest.mark.integration
def test_run_cli_reports_chunk_times_in_summary_table(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["run", str(tmp_path), "ih"])

    assert result.exit_code == 0
    for column in ("chunk", "status", "rows", "written", "time"):
        assert column in result.output
    assert re.search(r"\d+\.\d{3}ms", result.output)


@pytest.mark.integration
def test_workspace_sdk_filters_by_inclusive_date_range(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    workspace = Workspace(tmp_path)

    workspace.run_source("ih")
    frame = workspace.metric("CTR").by("Channel").between("2024-01-02", "2024-01-02").to_polars()

    assert frame["Day"].to_list() == [dt.date(2024, 1, 2)]
    assert frame.filter(pl.col("Channel") == "Web")["CTR"][0] == pytest.approx(2 / 3)


@pytest.mark.integration
def test_workspace_sdk_raw_query_includes_state_columns(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    workspace = Workspace(tmp_path)

    workspace.run_all()
    frame = workspace.metric("CTR").by("Channel").grain("summary").raw().to_polars()

    assert {"Count", "Positives", "Negatives", "CTR"} <= set(frame.columns)
