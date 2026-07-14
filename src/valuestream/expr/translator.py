"""Translate an :class:`~valuestream.expr.ast.Expr` into a single ``polars.Expr``.

Implements the full mapping table in ``docs/EXPRESSION_DSL.md`` §4. Each AST
node compiles to exactly one Polars expression — there is no ``eval``, no
Python callback, no string-built fragment.

The translator is purely structural: it does not consult a column schema
(types are the validator's concern) and it does not execute the expression.
The caller composes the returned ``pl.Expr`` into a ``with_columns``,
``filter``, or ``group_by + agg`` pipeline as appropriate.
"""

from __future__ import annotations

import ast as py_ast
import datetime as _dt
import operator
from collections.abc import Mapping
from functools import reduce
from typing import Any

import polars as pl

from valuestream.expr import ast

_DTYPE_MAP: dict[str, pl.DataType] = {
    "Int8": pl.Int8(),
    "Int16": pl.Int16(),
    "Int32": pl.Int32(),
    "Int64": pl.Int64(),
    "Float32": pl.Float32(),
    "Float64": pl.Float64(),
    "String": pl.String(),
    "Date": pl.Date(),
    "Datetime": pl.Datetime("us"),
    "Boolean": pl.Boolean(),
}

_DATE_TRUNC_EVERY: dict[str, str] = {
    "day": "1d",
    "month": "1mo",
    "quarter": "1q",
    "year": "1y",
    "hour": "1h",
    "week_iso": "1w",
}

_ALLOWED_PL_CALLS: frozenset[str] = frozenset(
    (
        "col",
        "lit",
        "when",
        "coalesce",
        "concat_str",
        "min_horizontal",
        "max_horizontal",
        "len",
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
    )
)
_ALLOWED_PL_ATTRS: frozenset[str] = _ALLOWED_PL_CALLS
_ALLOWED_EXPR_METHODS: frozenset[str] = frozenset(
    (
        "abs",
        "alias",
        "cast",
        "ceil",
        "clip",
        "contains",
        "date",
        "day",
        "ends_with",
        "exp",
        "fill_null",
        "floor",
        "is_between",
        "is_in",
        "is_not_null",
        "is_null",
        "log",
        "month",
        "otherwise",
        "over",
        "quarter",
        "replace",
        "round",
        "sqrt",
        "startswith",
        "starts_with",
        "strftime",
        "strptime",
        "then",
        "total_days",
        "total_hours",
        "total_minutes",
        "total_seconds",
        "slice",
        "when",
        "year",
    )
)
_ALLOWED_NODES: tuple[type[py_ast.AST], ...] = (
    py_ast.Expression,
    py_ast.BinOp,
    py_ast.UnaryOp,
    py_ast.Compare,
    py_ast.Call,
    py_ast.Attribute,
    py_ast.Name,
    py_ast.Constant,
    py_ast.Load,
    py_ast.List,
    py_ast.Tuple,
    py_ast.keyword,
)
_ALLOWED_OPERATORS: tuple[type[py_ast.AST], ...] = (
    py_ast.Add,
    py_ast.Sub,
    py_ast.Mult,
    py_ast.Div,
    py_ast.FloorDiv,
    py_ast.Mod,
    py_ast.Pow,
    py_ast.BitAnd,
    py_ast.BitOr,
    py_ast.Eq,
    py_ast.NotEq,
    py_ast.Lt,
    py_ast.LtE,
    py_ast.Gt,
    py_ast.GtE,
    py_ast.Invert,
    py_ast.USub,
    py_ast.UAdd,
)


class TranslationError(ValueError):
    """Raised when an AST node references a missing param or an unsupported shape."""


def translate(node: ast.Expr, params: Mapping[str, Any] | None = None) -> pl.Expr:
    """Compile ``node`` into a single :class:`polars.Expr`.

    ``params`` supplies values for ``{param: name}`` atoms. A missing param
    raises :class:`TranslationError`.
    """
    return _translate(node, params or {})


def _translate(node: ast.Expr, params: Mapping[str, Any]) -> pl.Expr:
    match node:
        # --- atoms -----------------------------------------------------
        case ast.Col(col=name):
            return pl.col(name)
        case ast.Lit(lit=value):
            return pl.lit(value)
        case ast.Param(param=name):
            if name not in params:
                raise TranslationError(f"parameter {name!r} not provided to translate()")
            return pl.lit(params[name])
        case ast.PolarsExpr(polars=text):
            return compile_polars_expr(text)

        # --- unary -----------------------------------------------------
        case ast.UnaryArg(op=op, arg=arg):
            inner = _translate(arg, params)
            if op == "not":
                return ~inner
            if op == "neg":
                return -inner
            if op == "abs":
                return inner.abs()
            if op == "sqrt":
                return inner.sqrt()
            if op == "exp":
                return inner.exp()
            if op == "ceil":
                return inner.ceil()
            if op == "floor":
                return inner.floor()
            raise TranslationError(f"unhandled UnaryArg op {op!r}")  # pragma: no cover

        case ast.OpLog(arg=arg, base=base):
            inner = _translate(arg, params)
            if base is None:
                return inner.log()
            return inner.log(base=base)

        case ast.OpRound(arg=arg, ndigits=n):
            inner = _translate(arg, params)
            return inner.round(n if n is not None else 0)

        case ast.OpCast(arg=arg, dtype=dtype):
            return _translate(arg, params).cast(_DTYPE_MAP[dtype])

        # --- nary ------------------------------------------------------
        case ast.LogicalNary(op=op, args=args):
            exprs = [_translate(a, params) for a in args]
            combine = operator.and_ if op == "and" else operator.or_
            return reduce(combine, exprs)

        case ast.ArithmeticNary(op=op, args=args):
            exprs = [_translate(a, params) for a in args]
            ops = {
                "add": operator.add,
                "sub": operator.sub,
                "mul": operator.mul,
                "div": operator.truediv,
            }
            return reduce(ops[op], exprs)

        case ast.OpSafeDiv(num=num, den=den):
            num_e = _translate(num, params)
            den_e = _translate(den, params)
            return pl.when(den_e == 0).then(pl.lit(0.0)).otherwise(num_e / den_e)

        case ast.OpConcat(args=args, sep=sep):
            exprs = [_translate(a, params) for a in args]
            return pl.concat_str(exprs, separator=sep if sep is not None else "")

        case ast.MinMaxNary(op=op, args=args):
            exprs = [_translate(a, params) for a in args]
            return pl.min_horizontal(*exprs) if op == "least" else pl.max_horizontal(*exprs)

        case ast.OpCoalesce(args=args):
            exprs = [_translate(a, params) for a in args]
            return pl.coalesce(exprs)

        # --- predicates -----------------------------------------------
        case ast.ComparisonColumn(op=op, column=col, value=value):
            col_e = pl.col(col)
            lit_e = pl.lit(value)
            return _apply_cmp(op, col_e, lit_e)

        case ast.ComparisonArgs(op=op, args=args):
            a_e = _translate(args[0], params)
            b_e = _translate(args[1], params)
            return _apply_cmp(op, a_e, b_e)

        case ast.OpIn(column=col, values=values):
            return pl.col(col).is_in(values)

        case ast.OpNotIn(column=col, values=values):
            return ~pl.col(col).is_in(values)

        case ast.OpBetween(column=col, low=low, high=high):
            return pl.col(col).is_between(low, high, closed="both")

        case ast.NullCheck(op=op, column=col):
            return pl.col(col).is_null() if op == "is_null" else pl.col(col).is_not_null()

        case ast.OpMatches(column=col, pattern=pattern):
            return pl.col(col).str.contains(pattern, literal=False)

        case ast.StrPrefixSuffix(op=op, column=col, value=value):
            return (
                pl.col(col).str.starts_with(value)
                if op == "starts_with"
                else pl.col(col).str.ends_with(value)
            )

        # --- conditional ----------------------------------------------
        case ast.OpCase(when=branches, else_=else_):
            else_e = _translate(else_, params)
            # Build nested ``pl.when(...).then(...).otherwise(...)`` from the back.
            chain: pl.Expr = else_e
            for br in reversed(branches):
                cond_e = _translate(br.cond, params)
                then_e = _translate(br.then, params)
                chain = pl.when(cond_e).then(then_e).otherwise(chain)
            return chain

        case ast.OpWhenThen(cond=cond, then=then, else_=else_):
            return (
                pl.when(_translate(cond, params))
                .then(_translate(then, params))
                .otherwise(_translate(else_, params))
            )

        # --- datetime --------------------------------------------------
        case ast.OpDateTrunc(unit=unit, arg=arg):
            return _translate(arg, params).dt.truncate(_DATE_TRUNC_EVERY[unit])

        case ast.OpDateDiff(unit=unit, end=end, start=start):
            end_e = _translate(end, params)
            start_e = _translate(start, params)
            diff = end_e - start_e
            if unit == "seconds":
                return diff.dt.total_seconds()
            if unit == "minutes":
                return diff.dt.total_minutes()
            if unit == "hours":
                return diff.dt.total_hours()
            if unit == "days":
                return diff.dt.total_days()
            # ``months`` and ``years`` have no direct timedelta accessor;
            # approximate via day counts. The validator already type-checks
            # both operands as date-like.
            if unit == "months":
                return (diff.dt.total_days() // 30).cast(pl.Int64())
            if unit == "years":
                return (diff.dt.total_days() // 365).cast(pl.Int64())
            raise TranslationError(f"unhandled date_diff unit {unit!r}")  # pragma: no cover

        case ast.OpDatePart(unit=unit, arg=arg):
            base = _translate(arg, params)
            if unit == "year":
                return base.dt.year()
            if unit == "month":
                return base.dt.month()
            if unit == "day":
                return base.dt.day()
            if unit == "quarter":
                return base.dt.quarter()
            if unit == "hour":
                return base.dt.hour()
            if unit == "weekday":
                return base.dt.weekday()
            raise TranslationError(f"unhandled date_part unit {unit!r}")  # pragma: no cover

        case ast.OpNow():
            # §6.5 pure check: snapshot once at translate time so the
            # resulting expression is reproducible across rows in the run.
            return pl.lit(_dt.datetime.now(_dt.UTC))

        case ast.OpStrftime(arg=arg, format=fmt):
            return _translate(arg, params).dt.strftime(fmt)

        case ast.OpStrptime(arg=arg, format=fmt):
            return _translate(arg, params).str.strptime(pl.Datetime("us"), fmt)

    raise TranslationError(  # pragma: no cover
        f"unhandled AST node: {type(node).__name__}"
    )


def _apply_cmp(op: str, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    if op == "eq":
        return a == b
    if op == "ne":
        return a != b
    if op == "lt":
        return a < b
    if op == "le":
        return a <= b
    if op == "gt":
        return a > b
    if op == "ge":
        return a >= b
    raise TranslationError(f"unhandled comparison op {op!r}")  # pragma: no cover


def compile_polars_expr(text: str) -> pl.Expr:
    """Compile guarded Polars expression text into a :class:`polars.Expr`."""
    try:
        parsed = py_ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise TranslationError(f"invalid Polars expression syntax: {exc.msg}") from exc
    _validate_polars_ast(parsed)
    try:
        result = eval(
            compile(parsed, "<valuestream-polars-expression>", "eval"),
            {"__builtins__": {}},
            {"pl": pl},
        )
    except Exception as exc:
        raise TranslationError(f"could not compile Polars expression: {exc}") from exc
    if not isinstance(result, pl.Expr):
        raise TranslationError("Polars expression must evaluate to a polars.Expr")
    return result


def _validate_polars_ast(node: py_ast.AST) -> None:
    for child in py_ast.walk(node):
        if isinstance(child, _ALLOWED_OPERATORS):
            continue
        if not isinstance(child, _ALLOWED_NODES):
            raise TranslationError(f"unsupported Polars expression syntax: {type(child).__name__}")
        if isinstance(child, py_ast.Name) and child.id != "pl":
            raise TranslationError(f"unsupported name in Polars expression: {child.id!r}")
        if isinstance(child, py_ast.Attribute):
            if child.attr.startswith("_"):
                raise TranslationError("private attributes are not allowed in Polars expressions")
            if (
                isinstance(child.value, py_ast.Name)
                and child.value.id == "pl"
                and child.attr not in _ALLOWED_PL_ATTRS
            ):
                raise TranslationError(f"unsupported Polars function or attribute: pl.{child.attr}")
        if isinstance(child, py_ast.Call):
            _validate_polars_call(child)


def _validate_polars_call(node: py_ast.Call) -> None:
    if isinstance(node.func, py_ast.Attribute):
        attr = node.func.attr
        if attr.startswith("_"):
            raise TranslationError("private methods are not allowed in Polars expressions")
        if isinstance(node.func.value, py_ast.Name) and node.func.value.id == "pl":
            if attr not in _ALLOWED_PL_CALLS:
                raise TranslationError(f"unsupported Polars function: pl.{attr}")
            return
        if attr not in _ALLOWED_EXPR_METHODS:
            raise TranslationError(f"unsupported Polars expression method: {attr}")
        return
    raise TranslationError("only Polars function and expression method calls are allowed")


__all__ = ["TranslationError", "translate"]
