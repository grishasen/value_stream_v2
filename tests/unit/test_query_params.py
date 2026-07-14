"""Unit tests for query executor filter/having/order/top-n/compare helpers."""

from __future__ import annotations

import polars as pl
import pytest

from valuestream.config import model
from valuestream.query import executor


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Channel": ["Web", "Web", "Mobile", "Store"],
            "Group": ["Cards", "Loans", "Cards", "Cards"],
            "CTR": [0.5, 0.2, 0.4, None],
            "Count": [10, 5, 8, 2],
        }
    )


@pytest.mark.unit
def test_apply_filters_supports_scalars_lists_and_operator_specs() -> None:
    frame = _frame()

    assert executor._apply_filters(frame, {"Channel": "Web"}).height == 2
    assert executor._apply_filters(frame, {"Channel": ["Web", "Mobile"]}).height == 3
    assert executor._apply_filters(frame, {"Channel": {"op": "ne", "value": "Web"}}).height == 2
    assert executor._apply_filters(frame, {"Count": {"op": ">=", "value": 8}}).height == 2
    assert (
        executor._apply_filters(frame, {"Channel": {"op": "not_in", "values": ["Web"]}}).height == 2
    )
    assert (
        executor._apply_filters(frame, {"Channel": {"op": "contains", "value": "eb"}}).height == 2
    )
    assert executor._apply_filters(frame, {"CTR": {"op": "is_null"}}).height == 1
    assert executor._apply_filters(frame, {"CTR": {"op": "not_null"}}).height == 3


@pytest.mark.unit
def test_apply_filters_rejects_unknown_columns_and_operators() -> None:
    frame = _frame()

    with pytest.raises(ValueError, match="filter column 'Missing'"):
        executor._apply_filters(frame, {"Missing": "x"})
    with pytest.raises(ValueError, match="unsupported filter operator"):
        executor._apply_filters(frame, {"Channel": {"op": "between", "value": 1}})
    with pytest.raises(ValueError, match="requires a value"):
        executor._apply_filters(frame, {"Channel": {"op": ">"}})


@pytest.mark.unit
def test_apply_having_filters_metric_outputs() -> None:
    frame = _frame()

    kept = executor._apply_having(frame, {"CTR": {"op": ">", "value": 0.3}})

    assert kept.get_column("Channel").to_list() == ["Web", "Mobile"]
    with pytest.raises(ValueError, match="having column 'Missing'"):
        executor._apply_having(frame, {"Missing": 1})


@pytest.mark.unit
def test_order_by_specs_parse_descending_prefix_and_validate() -> None:
    assert executor._order_by_specs(["-CTR", "Channel"], ["CTR", "Channel"]) == [
        ("CTR", True),
        ("Channel", False),
    ]
    with pytest.raises(ValueError, match="order_by column 'Missing'"):
        executor._order_by_specs(["Missing"], ["CTR"])


@pytest.mark.unit
def test_apply_top_n_keeps_largest_rows_globally_and_per_group() -> None:
    frame = _frame()

    top = executor._apply_top_n(frame, 2, "Count", None, group_columns=["Channel", "Group"])
    assert top.get_column("Count").to_list() == [10, 8]

    per_group = executor._apply_top_n(
        frame, 1, "Count", ["Group"], group_columns=["Channel", "Group"]
    )
    assert sorted(per_group.get_column("Count").to_list()) == [5, 10]

    defaulted = executor._apply_top_n(frame, 1, None, None, group_columns=["Channel", "Group"])
    assert defaulted.get_column("CTR").to_list() == [0.5]

    with pytest.raises(ValueError, match="top_n_by column 'Missing'"):
        executor._apply_top_n(frame, 1, "Missing", None, group_columns=[])
    with pytest.raises(ValueError, match="top_n_within column"):
        executor._apply_top_n(frame, 1, "Count", ["Missing"], group_columns=[])


@pytest.mark.unit
def test_prior_period_comparison_adds_lag_columns_per_dimension() -> None:
    frame = pl.DataFrame(
        {
            "Day": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "Channel": ["Web", "Web", "Mobile", "Mobile"],
            "CTR": [0.5, 0.6, 0.2, 0.1],
        }
    )

    out = executor._with_prior_period_comparison(
        frame,
        "prior_period",
        grain="daily",
        group_columns=["Day", "Channel"],
    )

    web = out.filter(pl.col("Channel") == "Web").sort("Day")
    assert web.get_column("CTR_prev").to_list() == [None, 0.5]
    assert web.get_column("CTR_delta").to_list() == [None, pytest.approx(0.1)]
    assert web.get_column("CTR_pct_change").to_list() == [None, pytest.approx(0.2)]
    mobile = out.filter(pl.col("Channel") == "Mobile").sort("Day")
    assert mobile.get_column("CTR_prev").to_list() == [None, 0.2]


@pytest.mark.unit
def test_prior_period_comparison_requires_time_axis_and_known_mode() -> None:
    frame = pl.DataFrame({"Channel": ["Web"], "CTR": [0.5]})

    with pytest.raises(ValueError, match="requires a time-bucketed query"):
        executor._with_prior_period_comparison(
            frame, "prior_period", grain="summary", group_columns=["Channel"]
        )
    with pytest.raises(ValueError, match="unsupported compare mode"):
        executor._with_prior_period_comparison(
            frame, "wow", grain="daily", group_columns=["Channel"]
        )


@pytest.mark.unit
def test_variant_compare_uses_only_the_configured_test_and_control_roles() -> None:
    frame = pl.DataFrame(
        {
            "Arm": ["Control", "Test", "Holdout"],
            "Positives": [10, 20, 100],
            "Negatives": [90, 80, 0],
        }
    )
    metric = model.VariantCompareMetric.model_validate(
        {
            "source": "engagement",
            "kind": "variant_compare",
            "variant_column": "Arm",
            "test_role": "Test",
            "control_role": "Control",
        }
    )

    out = executor._derive_variant_compare(frame, metric, [])

    assert out["TestCTR"][0] == pytest.approx(0.2)
    assert out["ControlCTR"][0] == pytest.approx(0.1)


@pytest.mark.unit
def test_proportion_test_executes_for_configured_roles() -> None:
    frame = pl.DataFrame(
        {
            "Arm": ["Control", "Test", "Holdout"],
            "Positives": [10, 20, 100],
            "Negatives": [90, 80, 0],
        }
    )
    metric = model.ProportionTestMetric.model_validate(
        {
            "source": "engagement",
            "kind": "proportion_test",
            "variant_column": "Arm",
            "test_role": "Test",
            "control_role": "Control",
        }
    )

    out = executor._derive_proportion_test(frame, metric, [])

    assert out["Count"][0] == 200
    assert out["Positives"][0] == 30
    assert out["z_score"][0] > 0
    assert 0 < out["z_p_val"][0] < 1
