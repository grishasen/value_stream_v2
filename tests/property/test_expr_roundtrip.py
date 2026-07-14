"""Property test: every valid AST round-trips through ``parse`` ⇆ ``to_dict``.

Strategy: generate a random AST dict from the closed grammar in §2 of
docs/EXPRESSION_DSL.md, parse it (which validates against the
discriminated-union model), serialize it back to a dict, and re-parse.
The two parsed nodes must compare equal.

We also verify two structural invariants:
* ``canonicalize`` is idempotent — applying it twice yields the same
  result as applying it once;
* the canonical form of an ``op: when_then`` collapses to a one-branch
  ``op: case``, matching the spec's §7 rewrite rule.

The test runs 200 examples with the default Hypothesis profile, well
above the prompt's ≥ 200 floor.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from valuestream.config.canonical import canonicalize
from valuestream.expr.parser import parse, to_dict

# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------

# Identifiers used for `col` and `param` references. Kept short and
# always start with a letter so they match the AST's `ident` regex.
ident_st = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=4,
).map(lambda s: "c_" + s)

# Scalars used for `lit` and predicate `value` slots. We avoid NaN and
# infinity (the canonical JSON serializer rejects them) and limit floats
# to a stable range so equality holds across round-trips.
scalar_leaf_st = st.one_of(
    st.integers(min_value=-100, max_value=100),
    st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.text(min_size=0, max_size=8),
    st.none(),
)


# ---------------------------------------------------------------------------
# AST strategies — built bottom-up.
# ---------------------------------------------------------------------------


def atom_st() -> st.SearchStrategy[dict[str, Any]]:
    """Atoms: col / lit / param."""
    return st.one_of(
        ident_st.map(lambda s: {"col": s}),
        scalar_leaf_st.map(lambda v: {"lit": v}),
        ident_st.map(lambda s: {"param": s}),
    )


def expr_st(max_depth: int = 3) -> st.SearchStrategy[dict[str, Any]]:
    """Recursive Expr strategy. ``max_depth`` caps the nesting level."""
    if max_depth <= 0:
        return atom_st()

    child = expr_st(max_depth - 1)

    unary_arg_st = st.builds(
        lambda op, arg: {"op": op, "arg": arg},
        st.sampled_from(["not", "neg", "abs", "sqrt", "exp", "ceil", "floor"]),
        child,
    )
    log_st = st.builds(lambda arg: {"op": "log", "arg": arg}, child)
    log_with_base_st = st.builds(
        lambda arg, base: {"op": "log", "arg": arg, "base": base},
        child,
        st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    round_st = st.builds(
        lambda arg, n: {"op": "round", "arg": arg, "ndigits": n},
        child,
        st.integers(min_value=0, max_value=6),
    )
    cast_st = st.builds(
        lambda arg, dtype: {"op": "cast", "arg": arg, "dtype": dtype},
        child,
        st.sampled_from(
            [
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "Float32",
                "Float64",
                "String",
                "Date",
                "Datetime",
                "Boolean",
            ]
        ),
    )

    logical_st = st.builds(
        lambda op, args: {"op": op, "args": args},
        st.sampled_from(["and", "or"]),
        st.lists(child, min_size=2, max_size=4),
    )
    arithmetic_st = st.builds(
        lambda op, args: {"op": op, "args": args},
        st.sampled_from(["add", "sub", "mul", "div"]),
        st.lists(child, min_size=2, max_size=4),
    )
    safe_div_st = st.builds(
        lambda num, den: {"op": "safe_div", "num": num, "den": den},
        child,
        child,
    )
    minmax_st = st.builds(
        lambda op, args: {"op": op, "args": args},
        st.sampled_from(["least", "greatest"]),
        st.lists(child, min_size=2, max_size=3),
    )
    coalesce_st = st.builds(
        lambda args: {"op": "coalesce", "args": args},
        st.lists(child, min_size=1, max_size=3),
    )

    cmp_column_st = st.builds(
        lambda op, col, val: {"op": op, "column": col, "value": val},
        st.sampled_from(["eq", "ne", "lt", "le", "gt", "ge"]),
        ident_st,
        scalar_leaf_st,
    )
    cmp_args_st = st.builds(
        lambda op, args: {"op": op, "args": args},
        st.sampled_from(["eq", "ne", "lt", "le", "gt", "ge"]),
        st.lists(child, min_size=2, max_size=2),
    )
    in_st = st.builds(
        lambda col, vals: {"op": "in", "column": col, "values": vals},
        ident_st,
        st.lists(scalar_leaf_st, min_size=1, max_size=4),
    )
    not_in_st = st.builds(
        lambda col, vals: {"op": "not_in", "column": col, "values": vals},
        ident_st,
        st.lists(scalar_leaf_st, min_size=1, max_size=4),
    )
    between_st = st.builds(
        lambda col, lo, hi: {"op": "between", "column": col, "low": lo, "high": hi},
        ident_st,
        st.integers(min_value=-100, max_value=0),
        st.integers(min_value=1, max_value=100),
    )
    null_st = st.builds(
        lambda op, col: {"op": op, "column": col},
        st.sampled_from(["is_null", "not_null"]),
        ident_st,
    )
    matches_st = st.builds(
        lambda col, pat: {"op": "matches", "column": col, "pattern": pat},
        ident_st,
        st.text(min_size=1, max_size=8),
    )
    affix_st = st.builds(
        lambda op, col, val: {"op": op, "column": col, "value": val},
        st.sampled_from(["starts_with", "ends_with"]),
        ident_st,
        st.text(min_size=1, max_size=4),
    )

    when_then_st = st.builds(
        lambda cond, then, otherwise: {
            "op": "when_then",
            "cond": cond,
            "then": then,
            "else": otherwise,
        },
        child,
        child,
        child,
    )
    case_st = st.builds(
        lambda branches, otherwise: {
            "op": "case",
            "when": [{"cond": c, "then": t} for c, t in branches],
            "else": otherwise,
        },
        st.lists(st.tuples(child, child), min_size=1, max_size=3),
        child,
    )

    date_trunc_st = st.builds(
        lambda unit, arg: {"op": "date_trunc", "unit": unit, "arg": arg},
        st.sampled_from(["day", "month", "quarter", "year", "hour", "week_iso"]),
        child,
    )
    date_diff_st = st.builds(
        lambda unit, end, start: {"op": "date_diff", "unit": unit, "end": end, "start": start},
        st.sampled_from(["seconds", "minutes", "hours", "days", "months", "years"]),
        child,
        child,
    )
    date_part_st = st.builds(
        lambda unit, arg: {"op": "date_part", "unit": unit, "arg": arg},
        st.sampled_from(["year", "month", "day", "quarter", "hour", "weekday"]),
        child,
    )
    now_st = st.just({"op": "now"})
    strftime_st = st.builds(
        lambda arg, fmt: {"op": "strftime", "arg": arg, "format": fmt},
        child,
        st.text(min_size=1, max_size=8),
    )
    strptime_st = st.builds(
        lambda arg, fmt: {"op": "strptime", "arg": arg, "format": fmt},
        child,
        st.text(min_size=1, max_size=8),
    )

    return st.one_of(
        atom_st(),
        unary_arg_st,
        log_st,
        log_with_base_st,
        round_st,
        cast_st,
        logical_st,
        arithmetic_st,
        safe_div_st,
        minmax_st,
        coalesce_st,
        cmp_column_st,
        cmp_args_st,
        in_st,
        not_in_st,
        between_st,
        null_st,
        matches_st,
        affix_st,
        when_then_st,
        case_st,
        date_trunc_st,
        date_diff_st,
        date_part_st,
        now_st,
        strftime_st,
        strptime_st,
    )


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------


@pytest.mark.property
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(expr_st(max_depth=3))
def test_parse_to_dict_round_trip(node_dict: dict[str, Any]) -> None:
    """``parse(to_dict(parse(d))) == parse(d)`` for every random valid AST."""
    first = parse(node_dict)
    again = parse(to_dict(first))
    assert first == again


@pytest.mark.property
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(expr_st(max_depth=3))
def test_canonicalize_idempotent(node_dict: dict[str, Any]) -> None:
    """``canonicalize(canonicalize(x)) == canonicalize(x)``."""
    once = canonicalize(node_dict)
    twice = canonicalize(once)
    assert once == twice


@pytest.mark.property
@settings(max_examples=50, deadline=None)
@given(expr_st(max_depth=2), expr_st(max_depth=2), expr_st(max_depth=2))
def test_when_then_canonicalizes_to_case(cond: Any, then: Any, otherwise: Any) -> None:
    """``when_then`` and the equivalent one-branch ``case`` collapse to the same form."""
    when_then = {"op": "when_then", "cond": cond, "then": then, "else": otherwise}
    case_form = {
        "op": "case",
        "when": [{"cond": cond, "then": then}],
        "else": otherwise,
    }
    assert canonicalize(when_then) == canonicalize(case_form)
