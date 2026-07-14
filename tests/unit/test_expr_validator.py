"""Validator tests covering ``docs/EXPRESSION_DSL.md`` §3 type rules and §6 checks.

Each test names which §3 row or §6 rule it exercises.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from valuestream.expr import ast
from valuestream.expr.parser import parse
from valuestream.expr.validator import Issue, ValidationError, validate

# A reusable column schema covering every dtype family the validator cares about.
SCHEMA: Mapping[str, ast.Dtype] = {
    "Channel": "String",
    "Outcome": "String",
    "Day": "Date",
    "OutcomeTime": "Datetime",
    "DecisionTime": "Datetime",
    "Positives": "Int64",
    "Negatives": "Int64",
    "Score": "Float64",
    "Cost": "Float32",
    "IsTest": "Boolean",
    "Counter": "Int8",
}


def _validate(d: dict, schema: Mapping[str, ast.Dtype] = SCHEMA) -> ast.Dtype:
    """Parse ``d`` and validate against ``schema``, returning the inferred dtype."""
    return validate(parse(d), schema).dtype


# ---------------------------------------------------------------------------
# §3 type inference — happy paths.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHappyPath:
    def test_col(self) -> None:
        assert _validate({"col": "Channel"}) == "String"

    def test_lit_int(self) -> None:
        assert _validate({"lit": 42}) == "Int64"

    def test_lit_float(self) -> None:
        assert _validate({"lit": 1.5}) == "Float64"

    def test_lit_bool(self) -> None:
        assert _validate({"lit": True}) == "Boolean"

    def test_lit_string(self) -> None:
        assert _validate({"lit": "Web"}) == "String"

    def test_unary_not(self) -> None:
        assert _validate({"op": "not", "arg": {"col": "IsTest"}}) == "Boolean"

    def test_unary_neg_preserves(self) -> None:
        assert _validate({"op": "neg", "arg": {"col": "Positives"}}) == "Int64"

    def test_unary_sqrt_promotes(self) -> None:
        assert _validate({"op": "sqrt", "arg": {"col": "Counter"}}) == "Float64"

    def test_log(self) -> None:
        assert _validate({"op": "log", "arg": {"col": "Score"}}) == "Float64"

    def test_cast(self) -> None:
        assert (
            _validate({"op": "cast", "arg": {"col": "Positives"}, "dtype": "Float32"}) == "Float32"
        )

    def test_arith_widens(self) -> None:
        # Int8 + Int64 + Float32 → Float64 widening
        assert (
            _validate(
                {
                    "op": "add",
                    "args": [
                        {"col": "Counter"},
                        {"col": "Positives"},
                        {"col": "Cost"},
                    ],
                }
            )
            == "Float64"
        )

    def test_div_returns_float64(self) -> None:
        assert (
            _validate({"op": "div", "args": [{"col": "Positives"}, {"col": "Counter"}]})
            == "Float64"
        )

    def test_safe_div(self) -> None:
        d = {
            "op": "safe_div",
            "num": {"col": "Positives"},
            "den": {
                "op": "add",
                "args": [{"col": "Positives"}, {"col": "Negatives"}],
            },
        }
        assert _validate(d) == "Float64"

    def test_concat(self) -> None:
        assert (
            _validate({"op": "concat", "args": [{"col": "Channel"}, {"col": "Outcome"}]})
            == "String"
        )

    def test_logical(self) -> None:
        d = {"op": "and", "args": [{"col": "IsTest"}, {"op": "not_null", "column": "Channel"}]}
        assert _validate(d) == "Boolean"

    def test_least_greatest(self) -> None:
        assert (
            _validate({"op": "least", "args": [{"col": "Positives"}, {"col": "Negatives"}]})
            == "Int64"
        )

    def test_coalesce_widens(self) -> None:
        d = {"op": "coalesce", "args": [{"col": "Score"}, {"col": "Counter"}]}
        assert _validate(d) == "Float64"

    def test_comparison_column_form(self) -> None:
        assert _validate({"op": "eq", "column": "Channel", "value": "Web"}) == "Boolean"

    def test_comparison_args_form(self) -> None:
        d = {"op": "lt", "args": [{"col": "Positives"}, {"col": "Negatives"}]}
        assert _validate(d) == "Boolean"

    def test_in(self) -> None:
        assert (
            _validate({"op": "in", "column": "Channel", "values": ["Web", "Mobile"]}) == "Boolean"
        )

    def test_between(self) -> None:
        assert _validate({"op": "between", "column": "Score", "low": 0.0, "high": 1.0}) == "Boolean"

    def test_null_check(self) -> None:
        assert _validate({"op": "is_null", "column": "Channel"}) == "Boolean"

    def test_matches(self) -> None:
        assert _validate({"op": "matches", "column": "Channel", "pattern": "^W.*"}) == "Boolean"

    def test_starts_with(self) -> None:
        assert _validate({"op": "starts_with", "column": "Channel", "value": "W"}) == "Boolean"

    def test_case(self) -> None:
        d = {
            "op": "case",
            "when": [
                {
                    "cond": {"op": "is_null", "column": "Score"},
                    "then": {"lit": 0.0},
                },
            ],
            "else": {"col": "Score"},
        }
        assert _validate(d) == "Float64"

    def test_when_then(self) -> None:
        d = {
            "op": "when_then",
            "cond": {"op": "is_null", "column": "Score"},
            "then": {"lit": 0.0},
            "else": {"col": "Score"},
        }
        assert _validate(d) == "Float64"

    def test_date_trunc(self) -> None:
        assert (
            _validate({"op": "date_trunc", "unit": "day", "arg": {"col": "OutcomeTime"}})
            == "Datetime"
        )

    def test_date_diff(self) -> None:
        d = {
            "op": "date_diff",
            "unit": "seconds",
            "end": {"col": "OutcomeTime"},
            "start": {"col": "DecisionTime"},
        }
        assert _validate(d) == "Int64"

    def test_date_part(self) -> None:
        assert _validate({"op": "date_part", "unit": "year", "arg": {"col": "Day"}}) == "Int64"

    def test_strftime(self) -> None:
        assert _validate({"op": "strftime", "arg": {"col": "Day"}, "format": "%Y"}) == "String"

    def test_strptime(self) -> None:
        assert (
            _validate({"op": "strptime", "arg": {"col": "Channel"}, "format": "%Y"}) == "Datetime"
        )

    def test_polars_expr_warns_and_uses_placeholder_dtype(self) -> None:
        result = validate(parse({"polars": 'pl.col("Positives") + pl.col("Negatives")'}), SCHEMA)
        assert result.dtype == "String"
        assert result.warnings


# ---------------------------------------------------------------------------
# §6.2 column-existence — every ``col`` reference must resolve.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestColumnExistence:
    def test_unknown_col(self) -> None:
        with pytest.raises(ValidationError) as ei:
            _validate({"col": "DoesNotExist"})
        assert "column 'DoesNotExist' not found" in str(ei.value)

    def test_unknown_col_in_predicate(self) -> None:
        with pytest.raises(ValidationError):
            _validate({"op": "is_null", "column": "DoesNotExist"})

    def test_unknown_col_in_in(self) -> None:
        with pytest.raises(ValidationError):
            _validate({"op": "in", "column": "Missing", "values": ["a"]})

    def test_unknown_col_in_between(self) -> None:
        with pytest.raises(ValidationError):
            _validate({"op": "between", "column": "Missing", "low": 0, "high": 1})

    def test_unknown_param(self) -> None:
        with pytest.raises(ValidationError):
            _validate({"param": "missing"})

    def test_known_param(self) -> None:
        result = validate(parse({"param": "now"}), SCHEMA, params={"now": "Datetime"})
        assert result.dtype == "Datetime"


# ---------------------------------------------------------------------------
# §6.3 type checks — incompatible types are flagged.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypeChecks:
    def test_not_on_string(self) -> None:
        with pytest.raises(ValidationError, match="requires Boolean"):
            _validate({"op": "not", "arg": {"col": "Channel"}})

    def test_neg_on_string(self) -> None:
        with pytest.raises(ValidationError, match="requires numeric"):
            _validate({"op": "neg", "arg": {"col": "Channel"}})

    def test_add_string_to_int(self) -> None:
        with pytest.raises(ValidationError, match="requires numeric"):
            _validate({"op": "add", "args": [{"col": "Channel"}, {"col": "Positives"}]})

    def test_concat_int(self) -> None:
        with pytest.raises(ValidationError, match="requires String"):
            _validate({"op": "concat", "args": [{"col": "Positives"}, {"col": "Channel"}]})

    def test_comparison_args_form_incompatible(self) -> None:
        with pytest.raises(ValidationError, match="incompatible types"):
            _validate({"op": "eq", "args": [{"col": "Channel"}, {"col": "Positives"}]})

    def test_comparison_column_form_incompatible(self) -> None:
        with pytest.raises(ValidationError):
            _validate({"op": "eq", "column": "Positives", "value": "Web"})

    def test_in_value_incompatible(self) -> None:
        with pytest.raises(ValidationError):
            _validate({"op": "in", "column": "Positives", "values": ["a", "b"]})

    def test_between_on_string_column(self) -> None:
        with pytest.raises(ValidationError, match="numeric or date column"):
            _validate({"op": "between", "column": "Channel", "low": 0, "high": 1})

    def test_matches_on_int_column(self) -> None:
        with pytest.raises(ValidationError, match="requires String column"):
            _validate({"op": "matches", "column": "Positives", "pattern": "x"})

    def test_starts_with_on_int_column(self) -> None:
        with pytest.raises(ValidationError, match="requires String column"):
            _validate({"op": "starts_with", "column": "Positives", "value": "x"})

    def test_case_branches_must_agree(self) -> None:
        d = {
            "op": "case",
            "when": [
                {
                    "cond": {"op": "is_null", "column": "Channel"},
                    "then": {"col": "Channel"},
                },
            ],
            "else": {"col": "Positives"},
        }
        with pytest.raises(ValidationError, match="incompatible types"):
            _validate(d)

    def test_when_then_branches_must_agree(self) -> None:
        d = {
            "op": "when_then",
            "cond": {"col": "IsTest"},
            "then": {"col": "Channel"},
            "else": {"col": "Positives"},
        }
        with pytest.raises(ValidationError, match="incompatible types"):
            _validate(d)

    def test_date_trunc_on_string(self) -> None:
        with pytest.raises(ValidationError, match="requires Date/Datetime"):
            _validate({"op": "date_trunc", "unit": "day", "arg": {"col": "Channel"}})

    def test_date_diff_on_int(self) -> None:
        with pytest.raises(ValidationError):
            _validate(
                {
                    "op": "date_diff",
                    "unit": "seconds",
                    "end": {"col": "Positives"},
                    "start": {"col": "OutcomeTime"},
                }
            )

    def test_strptime_on_int(self) -> None:
        with pytest.raises(ValidationError, match="requires String"):
            _validate({"op": "strptime", "arg": {"col": "Positives"}, "format": "%Y"})


# ---------------------------------------------------------------------------
# §6.4 constant-folding sanity.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstantFolding:
    def test_log_of_zero_literal(self) -> None:
        with pytest.raises(ValidationError, match="non-positive literal"):
            _validate({"op": "log", "arg": {"lit": 0}})

    def test_log_of_negative_literal(self) -> None:
        with pytest.raises(ValidationError, match="non-positive literal"):
            _validate({"op": "log", "arg": {"lit": -1.0}})

    def test_log_negative_base(self) -> None:
        with pytest.raises(ValidationError, match="log base must be positive"):
            _validate({"op": "log", "arg": {"col": "Score"}, "base": -2.0})

    def test_safe_div_literal_zero_denominator_warns(self) -> None:
        # Returns successfully (0.0 by spec) but emits a warning.
        result = validate(
            parse({"op": "safe_div", "num": {"col": "Positives"}, "den": {"lit": 0}}),
            SCHEMA,
        )
        assert result.dtype == "Float64"
        assert any("literal zero" in w.message for w in result.warnings)


# ---------------------------------------------------------------------------
# §6.5 pure check — now() emits a warning.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPureCheck:
    def test_now_warns(self) -> None:
        result = validate(parse({"op": "now"}), SCHEMA)
        assert result.dtype == "Datetime"
        assert any("time-dependent" in w.message for w in result.warnings)


# ---------------------------------------------------------------------------
# Multi-error reporting — the validator collects every issue, not just the first.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiError:
    def test_multiple_errors_collected(self) -> None:
        d = {
            "op": "and",
            "args": [
                {"col": "Missing1"},
                {"op": "concat", "args": [{"col": "Positives"}, {"col": "Missing2"}]},
            ],
        }
        with pytest.raises(ValidationError) as ei:
            _validate(d)
        # Three distinct findings: two missing columns + one concat-type error.
        codes = [it.message for it in ei.value.issues]
        assert any("Missing1" in m for m in codes)
        assert any("Missing2" in m for m in codes)
        assert any("requires String" in m for m in codes)


# ---------------------------------------------------------------------------
# Path rendering.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPathRendering:
    def test_args_path_segment(self) -> None:
        d = {"op": "add", "args": [{"col": "Positives"}, {"col": "Channel"}]}
        with pytest.raises(ValidationError) as ei:
            _validate(d)
        paths = [it.path for it in ei.value.issues if it.severity == "error"]
        assert any("args[1]" in p for p in paths)


# ---------------------------------------------------------------------------
# Issue is value-equal — used by tests that want to check the warning shape.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_issue_value_equality() -> None:
    a = Issue(path="a", message="b", severity="warning")
    b = Issue(path="a", message="b", severity="warning")
    assert a == b
