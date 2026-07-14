"""Parse round-trip and validation tests for every operator in EXPRESSION_DSL.md §2.

For each AST node kind we assert:

1. ``parse(d)`` returns the expected typed model.
2. ``to_dict(parse(d)) == d`` (round-trip is identity for canonical inputs).
3. Selected malformed inputs raise :class:`ParseError` with a useful message.

The schema-on-disk parity test guards against drift between
``schemas/expr.json`` and the AST model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from valuestream.expr import ast
from valuestream.expr._schema_gen import generate_schema
from valuestream.expr.parser import ParseError, parse, parse_yaml, to_dict

# ---------------------------------------------------------------------------
# Atom round-trips.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtoms:
    def test_col(self) -> None:
        node = parse({"col": "Channel"})
        assert isinstance(node, ast.Col)
        assert node.col == "Channel"
        assert to_dict(node) == {"col": "Channel"}

    def test_lit_int(self) -> None:
        node = parse({"lit": 42})
        assert isinstance(node, ast.Lit)
        assert node.lit == 42
        assert to_dict(node) == {"lit": 42}

    def test_lit_float(self) -> None:
        assert parse({"lit": 1.5}).lit == 1.5  # type: ignore[union-attr]

    def test_lit_string(self) -> None:
        assert parse({"lit": "Web"}).lit == "Web"  # type: ignore[union-attr]

    def test_lit_bool(self) -> None:
        assert parse({"lit": True}).lit is True  # type: ignore[union-attr]

    def test_lit_null(self) -> None:
        assert parse({"lit": None}).lit is None  # type: ignore[union-attr]

    def test_lit_array(self) -> None:
        assert parse({"lit": [1, 2, 3]}).lit == [1, 2, 3]  # type: ignore[union-attr]

    def test_lit_nested_array(self) -> None:
        # Recursive grammar in §2 — array<scalar> can nest.
        assert parse({"lit": [[1, 2], [3]]}).lit == [[1, 2], [3]]  # type: ignore[union-attr]

    def test_lit_rejects_non_scalar(self) -> None:
        # An AST node inside a `lit` is not a scalar.
        with pytest.raises(ParseError):
            parse({"lit": {"col": "X"}})

    def test_param(self) -> None:
        node = parse({"param": "now"})
        assert isinstance(node, ast.Param)
        assert node.param == "now"

    def test_polars_expr(self) -> None:
        node = parse({"polars": 'pl.col("A") + pl.col("B")'})
        assert isinstance(node, ast.PolarsExpr)
        assert node.polars == 'pl.col("A") + pl.col("B")'
        assert to_dict(node) == {"polars": 'pl.col("A") + pl.col("B")'}


# ---------------------------------------------------------------------------
# Unary ops.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnary:
    @pytest.mark.parametrize("op", ["not", "neg", "abs", "sqrt", "exp", "ceil", "floor"])
    def test_unary_arg_ops_round_trip(self, op: str) -> None:
        d = {"op": op, "arg": {"col": "X"}}
        node = parse(d)
        assert isinstance(node, ast.UnaryArg)
        assert node.op == op
        assert to_dict(node) == d

    def test_log_default_base(self) -> None:
        node = parse({"op": "log", "arg": {"col": "X"}})
        assert isinstance(node, ast.OpLog)
        assert node.base is None
        assert to_dict(node) == {"op": "log", "arg": {"col": "X"}}

    def test_log_with_base(self) -> None:
        node = parse({"op": "log", "arg": {"col": "X"}, "base": 2.0})
        assert isinstance(node, ast.OpLog)
        assert node.base == 2.0

    def test_round_default_ndigits(self) -> None:
        assert parse({"op": "round", "arg": {"col": "X"}}).ndigits is None  # type: ignore[union-attr]

    def test_round_with_ndigits(self) -> None:
        assert parse({"op": "round", "arg": {"col": "X"}, "ndigits": 3}).ndigits == 3  # type: ignore[union-attr]

    def test_cast(self) -> None:
        node = parse({"op": "cast", "arg": {"col": "X"}, "dtype": "Float64"})
        assert isinstance(node, ast.OpCast)
        assert node.dtype == "Float64"

    def test_cast_unknown_dtype_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "cast", "arg": {"col": "X"}, "dtype": "Bigint"})


# ---------------------------------------------------------------------------
# Nary arithmetic / logical / set / coalesce / concat / minmax.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNary:
    @pytest.mark.parametrize("op", ["and", "or"])
    def test_logical(self, op: str) -> None:
        d = {"op": op, "args": [{"col": "A"}, {"col": "B"}]}
        node = parse(d)
        assert isinstance(node, ast.LogicalNary)
        assert node.op == op
        assert to_dict(node) == d

    def test_logical_requires_at_least_two_args(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "and", "args": [{"col": "A"}]})

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_arithmetic(self, op: str) -> None:
        d = {"op": op, "args": [{"col": "A"}, {"col": "B"}, {"col": "C"}]}
        node = parse(d)
        assert isinstance(node, ast.ArithmeticNary)
        assert node.op == op
        assert len(node.args) == 3

    def test_safe_div(self) -> None:
        d = {"op": "safe_div", "num": {"col": "P"}, "den": {"col": "N"}}
        node = parse(d)
        assert isinstance(node, ast.OpSafeDiv)
        assert to_dict(node) == d

    def test_concat_default_sep(self) -> None:
        node = parse({"op": "concat", "args": [{"col": "A"}, {"col": "B"}]})
        assert isinstance(node, ast.OpConcat)
        assert node.sep is None

    def test_concat_with_sep(self) -> None:
        node = parse({"op": "concat", "args": [{"col": "A"}, {"col": "B"}], "sep": "/"})
        assert isinstance(node, ast.OpConcat)
        assert node.sep == "/"

    @pytest.mark.parametrize("op", ["least", "greatest"])
    def test_minmax(self, op: str) -> None:
        node = parse({"op": op, "args": [{"col": "A"}, {"col": "B"}]})
        assert isinstance(node, ast.MinMaxNary)
        assert node.op == op

    def test_coalesce(self) -> None:
        node = parse({"op": "coalesce", "args": [{"col": "A"}, {"col": "B"}, {"lit": 0}]})
        assert isinstance(node, ast.OpCoalesce)
        assert len(node.args) == 3


# ---------------------------------------------------------------------------
# Predicate / comparison nodes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComparison:
    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
    def test_column_form(self, op: str) -> None:
        d = {"op": op, "column": "X", "value": 1}
        node = parse(d)
        assert isinstance(node, ast.ComparisonColumn)
        assert node.op == op
        assert to_dict(node) == d

    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
    def test_args_form(self, op: str) -> None:
        d = {"op": op, "args": [{"col": "A"}, {"col": "B"}]}
        node = parse(d)
        assert isinstance(node, ast.ComparisonArgs)
        assert node.op == op

    def test_args_form_rejects_three(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "eq", "args": [{"col": "A"}, {"col": "B"}, {"col": "C"}]})


@pytest.mark.unit
class TestPredicates:
    def test_in(self) -> None:
        node = parse({"op": "in", "column": "Channel", "values": ["Web", "Mobile"]})
        assert isinstance(node, ast.OpIn)
        assert node.values == ["Web", "Mobile"]

    def test_not_in(self) -> None:
        node = parse({"op": "not_in", "column": "Channel", "values": [1, 2, 3]})
        assert isinstance(node, ast.OpNotIn)

    def test_between(self) -> None:
        node = parse({"op": "between", "column": "Score", "low": 0.0, "high": 1.0})
        assert isinstance(node, ast.OpBetween)
        assert node.low == 0.0
        assert node.high == 1.0

    @pytest.mark.parametrize("op", ["is_null", "not_null"])
    def test_null_check(self, op: str) -> None:
        node = parse({"op": op, "column": "X"})
        assert isinstance(node, ast.NullCheck)
        assert node.op == op

    def test_matches(self) -> None:
        node = parse({"op": "matches", "column": "Name", "pattern": r"^foo.*"})
        assert isinstance(node, ast.OpMatches)
        assert node.pattern == r"^foo.*"

    @pytest.mark.parametrize("op", ["starts_with", "ends_with"])
    def test_str_affix(self, op: str) -> None:
        node = parse({"op": op, "column": "Name", "value": "pre"})
        assert isinstance(node, ast.StrPrefixSuffix)
        assert node.op == op


# ---------------------------------------------------------------------------
# Conditional nodes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConditional:
    def test_case(self) -> None:
        d = {
            "op": "case",
            "when": [
                {"cond": {"op": "is_null", "column": "X"}, "then": {"lit": 0}},
                {"cond": {"op": "lt", "column": "X", "value": 10}, "then": {"lit": 1}},
            ],
            "else": {"col": "X"},
        }
        node = parse(d)
        assert isinstance(node, ast.OpCase)
        assert len(node.when) == 2
        assert isinstance(node.else_, ast.Col)
        assert to_dict(node) == d

    def test_when_then(self) -> None:
        d = {
            "op": "when_then",
            "cond": {"op": "is_null", "column": "X"},
            "then": {"lit": 0},
            "else": {"col": "X"},
        }
        node = parse(d)
        assert isinstance(node, ast.OpWhenThen)
        assert to_dict(node) == d

    def test_case_requires_branch(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "case", "when": [], "else": {"lit": 0}})


# ---------------------------------------------------------------------------
# Datetime nodes.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDatetime:
    @pytest.mark.parametrize("unit", ["day", "month", "quarter", "year", "hour", "week_iso"])
    def test_date_trunc(self, unit: str) -> None:
        d = {"op": "date_trunc", "unit": unit, "arg": {"col": "OutcomeTime"}}
        node = parse(d)
        assert isinstance(node, ast.OpDateTrunc)
        assert node.unit == unit

    @pytest.mark.parametrize("unit", ["seconds", "minutes", "hours", "days", "months", "years"])
    def test_date_diff(self, unit: str) -> None:
        d = {
            "op": "date_diff",
            "unit": unit,
            "end": {"col": "OutcomeTime"},
            "start": {"col": "DecisionTime"},
        }
        node = parse(d)
        assert isinstance(node, ast.OpDateDiff)
        assert node.unit == unit

    @pytest.mark.parametrize("unit", ["year", "month", "day", "quarter", "hour", "weekday"])
    def test_date_part(self, unit: str) -> None:
        node = parse({"op": "date_part", "unit": unit, "arg": {"col": "Day"}})
        assert isinstance(node, ast.OpDatePart)
        assert node.unit == unit

    def test_now(self) -> None:
        node = parse({"op": "now"})
        assert isinstance(node, ast.OpNow)
        assert to_dict(node) == {"op": "now"}

    def test_strftime(self) -> None:
        node = parse({"op": "strftime", "arg": {"col": "Day"}, "format": "%Y-%m-%d"})
        assert isinstance(node, ast.OpStrftime)
        assert node.format == "%Y-%m-%d"

    def test_strptime(self) -> None:
        node = parse({"op": "strptime", "arg": {"col": "RawDate"}, "format": "%Y-%m-%dT%H:%M:%S"})
        assert isinstance(node, ast.OpStrptime)


# ---------------------------------------------------------------------------
# YAML entry point and unrecognized-shape errors.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestYamlAndErrors:
    def test_parse_yaml(self) -> None:
        text = """
        op: safe_div
        num: {col: Positives}
        den:
          op: add
          args:
            - {col: Positives}
            - {col: Negatives}
        """
        node = parse_yaml(text)
        assert isinstance(node, ast.OpSafeDiv)

    def test_unknown_op_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "wibble", "arg": {"col": "X"}})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse({"col": "X", "extra": "no"})

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "safe_div", "num": {"col": "P"}})  # missing `den`

    def test_atom_with_op_key_rejected(self) -> None:
        # ambiguity guard: an atom must not also have an op key.
        with pytest.raises(ParseError):
            parse({"col": "X", "op": "neg"})

    def test_non_dict_top_level_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse(123)

    def test_op_not_a_string_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": 123, "arg": {"col": "X"}})

    def test_comparison_without_column_or_args_rejected(self) -> None:
        with pytest.raises(ParseError):
            parse({"op": "eq"})

    def test_lit_rejects_dict_inside_list(self) -> None:
        # Pydantic accepts list[Any], but _check_scalar rejects non-scalar leaves.
        with pytest.raises(ParseError):
            parse({"lit": [1, {"col": "X"}, 3]})


# ---------------------------------------------------------------------------
# Schema-on-disk parity.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSchemaParity:
    def test_disk_matches_model(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        on_disk = json.loads((repo_root / "schemas" / "expr.json").read_text())
        from_model = generate_schema()
        assert on_disk == from_model, (
            "schemas/expr.json is out of sync with valuestream.expr.ast — "
            "regenerate with: uv run python -m valuestream.expr._schema_gen"
        )
