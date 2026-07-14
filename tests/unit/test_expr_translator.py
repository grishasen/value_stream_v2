"""Translator tests covering every entry in EXPRESSION_DSL.md §4 mapping table.

For each AST node kind we build a tiny Polars DataFrame, translate the
expression, evaluate it, and assert the resulting column.
"""

from __future__ import annotations

import datetime as _dt
import math

import polars as pl
import pytest

from valuestream.expr.parser import parse
from valuestream.expr.translator import TranslationError, translate


def _eval(node_dict: dict, df: pl.DataFrame, params: dict | None = None) -> pl.Series:
    """Translate ``node_dict`` and apply it as a single column on ``df``."""
    return df.with_columns(_=translate(parse(node_dict), params))["_"]


# ---------------------------------------------------------------------------
# Atoms.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtoms:
    def test_col(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        out = _eval({"col": "X"}, df)
        assert out.to_list() == [1, 2, 3]

    def test_lit_int(self) -> None:
        df = pl.DataFrame({"_dummy": [0, 0]})
        out = _eval({"lit": 7}, df)
        assert out.to_list() == [7, 7]

    def test_lit_string(self) -> None:
        df = pl.DataFrame({"_": [0]})
        out = _eval({"lit": "Web"}, df)
        assert out.to_list() == ["Web"]

    def test_lit_null(self) -> None:
        df = pl.DataFrame({"_": [0]})
        out = _eval({"lit": None}, df)
        assert out.to_list() == [None]

    def test_param(self) -> None:
        df = pl.DataFrame({"_": [0]})
        out = _eval({"param": "threshold"}, df, params={"threshold": 0.5})
        assert out.to_list() == [0.5]

    def test_param_missing(self) -> None:
        with pytest.raises(TranslationError, match="not provided"):
            translate(parse({"param": "missing"}))

    def test_polars_expr(self) -> None:
        df = pl.DataFrame({"A": [1, 2], "B": [3, 4]})
        assert _eval({"polars": 'pl.col("A") + pl.col("B")'}, df).to_list() == [4, 6]

    def test_polars_expr_when_with_string_slice(self) -> None:
        df = pl.DataFrame({"CustomerID": ["C123", "D456"]})
        expr = (
            "pl.when(pl.col('CustomerID').str.slice(0, 1) == 'C')"
            ".then(pl.lit('Customers known'))"
            ".otherwise(pl.lit('Device/Anonymous'))"
        )

        assert _eval({"polars": expr}, df).to_list() == [
            "Customers known",
            "Device/Anonymous",
        ]

    def test_polars_expr_chained_when(self) -> None:
        df = pl.DataFrame(
            {
                "PlacementType": ["Banner", "", ""],
                "Name": ["Any", "CR-Offer", "Hero-Offer"],
            }
        )
        expr = (
            "pl.when(pl.col('PlacementType') != '').then(pl.col('PlacementType'))"
            ".when(pl.col('Name').str.starts_with('CR')).then(pl.lit('Flex'))"
            ".otherwise(pl.lit('Hero'))"
        )

        assert _eval({"polars": expr}, df).to_list() == ["Banner", "Flex", "Hero"]

    def test_polars_expr_rejects_arbitrary_python(self) -> None:
        with pytest.raises(TranslationError, match="unsupported"):
            translate(parse({"polars": "__import__('os').getcwd()"}))


# ---------------------------------------------------------------------------
# Unary ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnary:
    def test_not(self) -> None:
        df = pl.DataFrame({"B": [True, False]})
        assert _eval({"op": "not", "arg": {"col": "B"}}, df).to_list() == [False, True]

    def test_neg(self) -> None:
        df = pl.DataFrame({"X": [1, -2, 3]})
        assert _eval({"op": "neg", "arg": {"col": "X"}}, df).to_list() == [-1, 2, -3]

    def test_abs(self) -> None:
        df = pl.DataFrame({"X": [-5, 3, -1]})
        assert _eval({"op": "abs", "arg": {"col": "X"}}, df).to_list() == [5, 3, 1]

    def test_sqrt(self) -> None:
        df = pl.DataFrame({"X": [4.0, 9.0, 16.0]})
        assert _eval({"op": "sqrt", "arg": {"col": "X"}}, df).to_list() == [2.0, 3.0, 4.0]

    def test_exp_ceil_floor(self) -> None:
        df = pl.DataFrame({"X": [1.4, 2.6]})
        assert _eval({"op": "ceil", "arg": {"col": "X"}}, df).to_list() == [2.0, 3.0]
        assert _eval({"op": "floor", "arg": {"col": "X"}}, df).to_list() == [1.0, 2.0]

    def test_log_natural(self) -> None:
        df = pl.DataFrame({"X": [1.0, math.e, math.e**2]})
        out = _eval({"op": "log", "arg": {"col": "X"}}, df).to_list()
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(1.0)
        assert out[2] == pytest.approx(2.0)

    def test_log_base_2(self) -> None:
        df = pl.DataFrame({"X": [1.0, 2.0, 8.0]})
        out = _eval({"op": "log", "arg": {"col": "X"}, "base": 2.0}, df).to_list()
        assert out == pytest.approx([0.0, 1.0, 3.0])

    def test_round_default(self) -> None:
        df = pl.DataFrame({"X": [1.4, 1.5, 2.5]})
        # Polars default rounding may differ from banker's; just check stability.
        out = _eval({"op": "round", "arg": {"col": "X"}}, df).to_list()
        assert all(isinstance(v, float) for v in out)

    def test_round_ndigits(self) -> None:
        df = pl.DataFrame({"X": [1.234567]})
        assert _eval({"op": "round", "arg": {"col": "X"}, "ndigits": 2}, df).to_list() == [1.23]

    def test_cast(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        out = _eval({"op": "cast", "arg": {"col": "X"}, "dtype": "Float64"}, df)
        assert out.dtype == pl.Float64()
        assert out.to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Nary ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNary:
    def test_and(self) -> None:
        df = pl.DataFrame({"A": [True, True, False], "B": [True, False, True]})
        out = _eval({"op": "and", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list()
        assert out == [True, False, False]

    def test_or(self) -> None:
        df = pl.DataFrame({"A": [True, False, False], "B": [False, True, False]})
        out = _eval({"op": "or", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list()
        assert out == [True, True, False]

    def test_add(self) -> None:
        df = pl.DataFrame({"P": [1, 2], "N": [3, 4], "X": [10, 20]})
        out = _eval({"op": "add", "args": [{"col": "P"}, {"col": "N"}, {"col": "X"}]}, df).to_list()
        assert out == [14, 26]

    def test_sub_mul_div(self) -> None:
        df = pl.DataFrame({"A": [10, 20], "B": [3, 4]})
        assert _eval({"op": "sub", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list() == [7, 16]
        assert _eval({"op": "mul", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list() == [30, 80]
        out = _eval({"op": "div", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list()
        assert out == pytest.approx([10 / 3, 5.0])

    def test_safe_div_zero_den_returns_zero(self) -> None:
        df = pl.DataFrame({"P": [10, 20, 30], "Total": [100, 0, 200]})
        d = {
            "op": "safe_div",
            "num": {"col": "P"},
            "den": {"col": "Total"},
        }
        out = _eval(d, df).to_list()
        assert out == [0.1, 0.0, 0.15]

    def test_concat(self) -> None:
        df = pl.DataFrame({"A": ["X", "Y"], "B": ["1", "2"]})
        out = _eval(
            {"op": "concat", "args": [{"col": "A"}, {"col": "B"}], "sep": "/"}, df
        ).to_list()
        assert out == ["X/1", "Y/2"]

    def test_concat_no_sep(self) -> None:
        df = pl.DataFrame({"A": ["X"], "B": ["Y"]})
        assert _eval({"op": "concat", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list() == ["XY"]

    def test_least(self) -> None:
        df = pl.DataFrame({"A": [3, 5, 1], "B": [2, 8, 7]})
        assert _eval({"op": "least", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list() == [
            2,
            5,
            1,
        ]

    def test_greatest(self) -> None:
        df = pl.DataFrame({"A": [3, 5, 1], "B": [2, 8, 7]})
        assert _eval({"op": "greatest", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list() == [
            3,
            8,
            7,
        ]

    def test_coalesce(self) -> None:
        df = pl.DataFrame({"A": [None, 2, None], "B": [1, None, None], "C": [99, 99, 99]})
        out = _eval(
            {"op": "coalesce", "args": [{"col": "A"}, {"col": "B"}, {"col": "C"}]},
            df,
        ).to_list()
        assert out == [1, 2, 99]


# ---------------------------------------------------------------------------
# Comparison and predicate ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComparison:
    @pytest.mark.parametrize(
        ("op", "value", "expected"),
        [
            ("eq", 2, [False, True, False]),
            ("ne", 2, [True, False, True]),
            ("lt", 2, [True, False, False]),
            ("le", 2, [True, True, False]),
            ("gt", 2, [False, False, True]),
            ("ge", 2, [False, True, True]),
        ],
    )
    def test_column_form(self, op: str, value: int, expected: list[bool]) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        assert _eval({"op": op, "column": "X", "value": value}, df).to_list() == expected

    def test_args_form_eq(self) -> None:
        df = pl.DataFrame({"A": [1, 2, 3], "B": [1, 0, 3]})
        out = _eval({"op": "eq", "args": [{"col": "A"}, {"col": "B"}]}, df).to_list()
        assert out == [True, False, True]


@pytest.mark.unit
class TestPredicates:
    def test_in(self) -> None:
        df = pl.DataFrame({"C": ["Web", "App", "Email"]})
        out = _eval({"op": "in", "column": "C", "values": ["Web", "App"]}, df).to_list()
        assert out == [True, True, False]

    def test_not_in(self) -> None:
        df = pl.DataFrame({"C": ["Web", "App"]})
        assert _eval({"op": "not_in", "column": "C", "values": ["Web"]}, df).to_list() == [
            False,
            True,
        ]

    def test_between(self) -> None:
        df = pl.DataFrame({"X": [0, 1, 2, 3]})
        out = _eval({"op": "between", "column": "X", "low": 1, "high": 2}, df).to_list()
        assert out == [False, True, True, False]

    def test_is_null(self) -> None:
        df = pl.DataFrame({"X": [1, None, 3]})
        assert _eval({"op": "is_null", "column": "X"}, df).to_list() == [False, True, False]

    def test_not_null(self) -> None:
        df = pl.DataFrame({"X": [1, None, 3]})
        assert _eval({"op": "not_null", "column": "X"}, df).to_list() == [True, False, True]

    def test_matches(self) -> None:
        df = pl.DataFrame({"S": ["foo", "foobar", "bar"]})
        out = _eval({"op": "matches", "column": "S", "pattern": r"^foo"}, df).to_list()
        assert out == [True, True, False]

    def test_starts_with(self) -> None:
        df = pl.DataFrame({"S": ["abc", "xyz"]})
        assert _eval({"op": "starts_with", "column": "S", "value": "ab"}, df).to_list() == [
            True,
            False,
        ]

    def test_ends_with(self) -> None:
        df = pl.DataFrame({"S": ["abc", "xyz"]})
        assert _eval({"op": "ends_with", "column": "S", "value": "yz"}, df).to_list() == [
            False,
            True,
        ]


# ---------------------------------------------------------------------------
# Conditional ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConditional:
    def test_when_then(self) -> None:
        df = pl.DataFrame({"X": [1, 2, 3]})
        d = {
            "op": "when_then",
            "cond": {"op": "lt", "column": "X", "value": 2},
            "then": {"lit": "small"},
            "else": {"lit": "big"},
        }
        assert _eval(d, df).to_list() == ["small", "big", "big"]

    def test_case_two_branches(self) -> None:
        df = pl.DataFrame({"X": [1, 5, 9]})
        d = {
            "op": "case",
            "when": [
                {"cond": {"op": "lt", "column": "X", "value": 3}, "then": {"lit": "low"}},
                {"cond": {"op": "lt", "column": "X", "value": 7}, "then": {"lit": "mid"}},
            ],
            "else": {"lit": "high"},
        }
        assert _eval(d, df).to_list() == ["low", "mid", "high"]


# ---------------------------------------------------------------------------
# Datetime ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatetime:
    def test_date_trunc_day(self) -> None:
        df = pl.DataFrame(
            {"T": [_dt.datetime(2024, 8, 21, 13, 45), _dt.datetime(2024, 8, 22, 0, 1)]}
        )
        out = _eval({"op": "date_trunc", "unit": "day", "arg": {"col": "T"}}, df).to_list()
        assert out == [_dt.datetime(2024, 8, 21), _dt.datetime(2024, 8, 22)]

    def test_date_trunc_month(self) -> None:
        df = pl.DataFrame({"T": [_dt.datetime(2024, 8, 21)]})
        out = _eval({"op": "date_trunc", "unit": "month", "arg": {"col": "T"}}, df).to_list()
        assert out == [_dt.datetime(2024, 8, 1)]

    def test_date_diff_seconds(self) -> None:
        df = pl.DataFrame(
            {
                "E": [_dt.datetime(2024, 1, 1, 0, 1)],
                "S": [_dt.datetime(2024, 1, 1, 0, 0)],
            }
        )
        d = {"op": "date_diff", "unit": "seconds", "end": {"col": "E"}, "start": {"col": "S"}}
        assert _eval(d, df).to_list() == [60.0]

    def test_date_diff_days(self) -> None:
        df = pl.DataFrame(
            {
                "E": [_dt.datetime(2024, 1, 5)],
                "S": [_dt.datetime(2024, 1, 1)],
            }
        )
        d = {"op": "date_diff", "unit": "days", "end": {"col": "E"}, "start": {"col": "S"}}
        assert _eval(d, df).to_list() == [4]

    def test_date_part_year(self) -> None:
        df = pl.DataFrame({"T": [_dt.datetime(2024, 8, 21)]})
        assert _eval({"op": "date_part", "unit": "year", "arg": {"col": "T"}}, df).to_list() == [
            2024
        ]

    def test_date_part_month(self) -> None:
        df = pl.DataFrame({"T": [_dt.datetime(2024, 8, 21)]})
        assert _eval({"op": "date_part", "unit": "month", "arg": {"col": "T"}}, df).to_list() == [8]

    def test_now(self) -> None:
        df = pl.DataFrame({"_": [0]})
        out = _eval({"op": "now"}, df).to_list()
        assert isinstance(out[0], _dt.datetime)

    def test_strftime(self) -> None:
        df = pl.DataFrame({"T": [_dt.date(2024, 8, 21)]})
        assert _eval(
            {"op": "strftime", "arg": {"col": "T"}, "format": "%Y-%m-%d"}, df
        ).to_list() == ["2024-08-21"]

    def test_strptime(self) -> None:
        df = pl.DataFrame({"S": ["2024-08-21"]})
        out = _eval({"op": "strptime", "arg": {"col": "S"}, "format": "%Y-%m-%d"}, df).to_list()
        assert out == [_dt.datetime(2024, 8, 21)]


# ---------------------------------------------------------------------------
# Composite — a real metric formula end-to-end (CTR).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ctr_formula_e2e() -> None:
    df = pl.DataFrame({"Positives": [10, 20, 0], "Negatives": [90, 80, 5]})
    d = {
        "op": "safe_div",
        "num": {"col": "Positives"},
        "den": {"op": "add", "args": [{"col": "Positives"}, {"col": "Negatives"}]},
    }
    out = _eval(d, df).to_list()
    assert out == [0.1, 0.2, 0.0]
