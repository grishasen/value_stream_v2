"""Phase 3 vertical slice: lifecycle, sets, funnels, and snapshots."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from valuestream.engine import run_workspace
from valuestream.query import query_metric


def _write_catalog(ws: Path) -> None:
    catalog = ws / "catalog"
    catalog.mkdir(parents=True)
    (catalog / "pipelines.yaml").write_text(
        """
version: 1
workspace: phase3_test
sources:
  - id: events
    reader:
      kind: parquet
      file_pattern: "data/events*.parquet"
      group_by_filename: '(\\d{8})'
    schema:
      timestamp_column: EventTime
      natural_key: [EventID]
    transforms:
      - kind: derive_calendar
        from: EventTime
        outputs: [Day, Month]
      - kind: cast
        columns:
          Channel: String
          Outcome: String
          CustomerID: String
          OrderID: String
          Amount: Float64
  - id: subscriptions
    reader:
      kind: parquet
      file_pattern: "data/subscriptions*.parquet"
      group_by_filename: '(\\d{8})'
    schema:
      timestamp_column: SnapshotTime
      natural_key: [SubscriptionID, SnapshotTime]
    transforms:
      - kind: derive_calendar
        from: SnapshotTime
        outputs: [Day, Month]
      - kind: cast
        columns:
          Plan: String
          Status: String
          MRR: Float64
""",
        encoding="utf-8",
    )
    (catalog / "processors.yaml").write_text(
        """
processors:
  - id: unique_users
    source: events
    kind: entity_set
    group_by: [Channel]
    time:
      column: EventTime
      grains: [Day, Month, Summary]
    states:
      ActiveUsers_hll: {type: hll, source_column: CustomerID, lg_k: 12}
      Visitors_theta: {type: theta, source_column: CustomerID, lg_k: 12}
      Clickers_theta:
        type: theta
        source_column: CustomerID
        lg_k: 12
        where: {op: eq, column: Outcome, value: Clicked}
      Converters_theta:
        type: theta
        source_column: CustomerID
        lg_k: 12
        where: {op: eq, column: Outcome, value: Conversion}
  - id: action_funnel
    source: events
    kind: funnel
    group_by: [Channel]
    time:
      column: EventTime
      grains: [Day, Month, Summary]
    entity: CustomerID
    stages:
      - {name: Impression, when: {op: eq, column: Outcome, value: Impression}}
      - {name: Clicked, when: {op: eq, column: Outcome, value: Clicked}}
      - {name: Conversion, when: {op: eq, column: Outcome, value: Conversion}}
  - id: customer_lifecycle
    source: events
    kind: entity_lifecycle
    group_by: [Channel]
    time:
      column: EventTime
      grains: [Summary]
    filter: {op: eq, column: Outcome, value: Conversion}
    keys:
      customer_id: CustomerID
      order_id: OrderID
      monetary: Amount
      purchase_date: EventTime
  - id: subscription_state
    source: subscriptions
    kind: snapshot
    snapshot_kind: periodic
    cadence: daily
    group_by: [Plan]
    time:
      column: SnapshotTime
      grains: [Day, Month, Summary]
    as_of_column: Day
    states:
      ActiveSubs:
        type: count
        where: {op: eq, column: Status, value: active}
      MRR:
        type: value_sum
        source_column: MRR
        where: {op: eq, column: Status, value: active}
      ChurnedSubs:
        type: count
        where: {op: eq, column: Status, value: churned}
""",
        encoding="utf-8",
    )
    (catalog / "metrics.yaml").write_text(
        """
metrics:
  ActiveUsers:
    source: unique_users
    kind: approx_distinct_count
    state: ActiveUsers_hll
  ClickedAndConverted:
    source: unique_users
    kind: set_op
    op: intersection
    states: [Clickers_theta, Converters_theta]
  ClickedNotConverted:
    source: unique_users
    kind: set_op
    op: a_not_b
    states: [Clickers_theta, Converters_theta]
  ClickDropoff:
    source: action_funnel
    kind: funnel_dropoff
    from_stage: Impression
    to_stage: Clicked
  LifecycleSummary:
    source: customer_lifecycle
    kind: lifecycle_summary
    outputs: [frequency, monetary_value, rfm_seg, rfm_segment, rfm_score]
  ActiveMRR:
    source: subscription_state
    kind: formula
    expression: {col: MRR}
  ActiveSubCount:
    source: subscription_state
    kind: formula
    expression: {col: ActiveSubs}
""",
        encoding="utf-8",
    )
    (catalog / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


def _write_data(ws: Path) -> None:
    data = ws / "data"
    data.mkdir()
    pl.DataFrame(
        [
            _event("e1", "2024-01-01", "Web", "Impression", "c1", None, 0.0),
            _event("e2", "2024-01-01", "Web", "Clicked", "c1", None, 0.0),
            _event("e3", "2024-01-01", "Web", "Conversion", "c1", "o1", 50.0),
            _event("e4", "2024-01-01", "Web", "Impression", "c2", None, 0.0),
            _event("e5", "2024-01-01", "Web", "Clicked", "c2", None, 0.0),
            _event("e6", "2024-01-02", "Web", "Impression", "c3", None, 0.0),
            _event("e7", "2024-01-02", "Web", "Conversion", "c3", "o2", 40.0),
            _event("e8", "2024-01-02", "Web", "Conversion", "c1", "o4", 70.0),
            _event("e9", "2024-01-02", "Mobile", "Impression", "c4", None, 0.0),
            _event("e10", "2024-01-02", "Mobile", "Clicked", "c4", None, 0.0),
            _event("e11", "2024-01-02", "Mobile", "Conversion", "c4", "o3", 90.0),
        ]
    ).write_parquet(data / "events_20240101000000.parquet")
    pl.DataFrame(
        [
            _subscription("s1", "2024-01-01", "Basic", "active", 10.0),
            _subscription("s2", "2024-01-01", "Pro", "active", 20.0),
            _subscription("s1", "2024-01-02", "Basic", "active", 10.0),
            _subscription("s2", "2024-01-02", "Pro", "churned", 0.0),
            _subscription("s3", "2024-01-02", "Pro", "active", 30.0),
            _subscription("s1", "2024-01-03", "Basic", "churned", 0.0),
            _subscription("s4", "2024-01-03", "Basic", "active", 12.0),
            _subscription("s3", "2024-01-03", "Pro", "active", 30.0),
        ]
    ).write_parquet(data / "subscriptions_20240101000000.parquet")


def _event(
    event_id: str,
    day: str,
    channel: str,
    outcome: str,
    customer: str,
    order: str | None,
    amount: float,
) -> dict[str, object]:
    return {
        "EventID": event_id,
        "EventTime": dt.datetime.fromisoformat(day + "T10:00:00"),
        "Channel": channel,
        "Outcome": outcome,
        "CustomerID": customer,
        "OrderID": order,
        "Amount": amount,
    }


def _subscription(
    subscription_id: str,
    day: str,
    plan: str,
    status: str,
    mrr: float,
) -> dict[str, object]:
    return {
        "SubscriptionID": subscription_id,
        "SnapshotTime": dt.datetime.fromisoformat(day + "T00:00:00"),
        "Plan": plan,
        "Status": status,
        "MRR": mrr,
    }


@pytest.mark.integration
def test_phase3_processors_and_metrics(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    _write_data(tmp_path)

    run = run_workspace(tmp_path)
    active = query_metric(tmp_path, "ActiveUsers", group_by=["Channel"], grain="summary")
    retained = query_metric(tmp_path, "ClickedAndConverted", group_by=["Channel"], grain="summary")
    new = query_metric(tmp_path, "ClickedNotConverted", group_by=["Channel"], grain="summary")
    dropoff = query_metric(tmp_path, "ClickDropoff", group_by=["Channel"], grain="summary")
    lifecycle = query_metric(tmp_path, "LifecycleSummary", group_by=["Channel"], grain="summary")
    mrr = query_metric(tmp_path, "ActiveMRR", group_by=["Plan"], grain="summary")
    active_subs = query_metric(tmp_path, "ActiveSubCount", group_by=["Plan"], grain="summary")

    assert run.status == "ok"
    assert active.filter(pl.col("Channel") == "Web")["ActiveUsers"][0] == pytest.approx(3.0)
    assert retained.filter(pl.col("Channel") == "Web")["ClickedAndConverted"][0] == pytest.approx(
        1.0
    )
    assert new.filter(pl.col("Channel") == "Web")["ClickedNotConverted"][0] == pytest.approx(1.0)
    assert dropoff.filter(pl.col("Channel") == "Web")["ClickDropoff"][0] == pytest.approx(1 / 3)

    c1 = lifecycle.filter(pl.col("CustomerID") == "c1").row(0, named=True)
    assert c1["Channel"] == "Web"
    assert c1["frequency"] == 1
    assert c1["monetary_value"] == pytest.approx(60.0)
    assert c1["rfm_score"] >= 1.0

    assert mrr.filter(pl.col("Plan") == "Basic")["ActiveMRR"][0] == pytest.approx(12.0)
    assert mrr.filter(pl.col("Plan") == "Pro")["ActiveMRR"][0] == pytest.approx(30.0)
    assert active_subs.filter(pl.col("Plan") == "Basic")["ActiveSubCount"][0] == 1


@pytest.mark.integration
def test_set_op_operands_apply_relative_time_windows(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    metrics = tmp_path / "catalog" / "metrics.yaml"
    metrics.write_text(
        metrics.read_text(encoding="utf-8")
        + """
  RetainedFromPriorDay:
    source: unique_users
    kind: set_op
    op: intersection
    operands:
      - state: Visitors_theta
        time_window: {last: 1d}
      - state: Visitors_theta
        time_window: {between: [-1d, -1d]}
""",
        encoding="utf-8",
    )
    _write_data(tmp_path)

    run_workspace(tmp_path)
    retained = query_metric(
        tmp_path,
        "RetainedFromPriorDay",
        group_by=["Channel"],
        grain="summary",
    )

    assert retained.filter(pl.col("Channel") == "Web")["RetainedFromPriorDay"][0] == pytest.approx(
        1.0
    )
    assert retained.filter(pl.col("Channel") == "Mobile")["RetainedFromPriorDay"][
        0
    ] == pytest.approx(0.0)
