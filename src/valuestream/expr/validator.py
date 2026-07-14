"""Type-check an :class:`~valuestream.expr.ast.Expr` against a column schema.

Implements ``docs/EXPRESSION_DSL.md`` §3 (type rules) and the static checks
from §6:

* §6.1 schema check — already done by Pydantic at parse time;
* §6.2 column-existence — every ``col`` reference must be in the supplied
  schema (or in the running set of columns produced by earlier transforms,
  which is the caller's responsibility to maintain);
* §6.3 type check — every node has a known result type, checked at compile;
* §6.4 constant-folding sanity — ``log(0)`` is rejected when the argument
  is a literal zero, ``safe_div`` with a literal zero divisor is flagged;
* §6.5 pure check — ``now()`` is allowed but emits a warning so callers can
  surface the time-dependence of ``config_hash``.

Errors carry an AST path string (``metrics.CTR.expression.den.args[1]``)
which downstream consumers (CLI, REST API, Builder UI) render verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from valuestream.expr import ast

# ---------------------------------------------------------------------------
# Public types.
# ---------------------------------------------------------------------------
Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """A single validator finding."""

    path: str
    message: str
    severity: Severity = "error"


class ValidationError(ValueError):
    """Raised when validation finds at least one error.

    ``issues`` carries the full list (errors and warnings) so callers can
    render structured output. The string form is human-readable and tagged
    with each issue's AST path.
    """

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__(self._format())

    def _format(self) -> str:
        lines = []
        for it in self.issues:
            lines.append(f"  {it.severity.upper()} at {it.path}: {it.message}")
        return "expression validation failed:\n" + "\n".join(lines)


@dataclass
class ValidationResult:
    """Outcome of a successful validation pass.

    A successful pass returns the inferred result :class:`~valuestream.expr.ast.Dtype`
    of the root expression plus any warnings (e.g. ``now()`` usage).
    """

    dtype: ast.Dtype
    warnings: list[Issue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

# Internal "types" extend the public ``Dtype`` set with ``Null`` (for
# ``lit None``) and ``Unknown`` (for unresolved errors). ``Null`` is
# compatible with every other type via ``_join``.
_InternalDtype = Literal[
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
    "Null",
    "Unknown",
]

_INTEGER: frozenset[str] = frozenset(("Int8", "Int16", "Int32", "Int64"))
_FLOAT: frozenset[str] = frozenset(("Float32", "Float64"))
_NUMERIC: frozenset[str] = _INTEGER | _FLOAT
_DATE_LIKE: frozenset[str] = frozenset(("Date", "Datetime"))
_PUBLIC_DTYPES: frozenset[str] = _NUMERIC | _DATE_LIKE | frozenset(("String", "Boolean"))

_NUMERIC_RANK: dict[str, int] = {
    "Int8": 0,
    "Int16": 1,
    "Int32": 2,
    "Int64": 3,
    "Float32": 4,
    "Float64": 5,
}


def _is_numeric(t: _InternalDtype) -> bool:
    return t in _NUMERIC


def _is_integer(t: _InternalDtype) -> bool:
    return t in _INTEGER


def _is_date_like(t: _InternalDtype) -> bool:
    return t in _DATE_LIKE


def _widen_numeric(a: _InternalDtype, b: _InternalDtype) -> _InternalDtype:
    """Return the wider of two numeric types.

    Int x Int picks the wider Int. Float x Float picks the wider Float.
    Int x Float promotes to Float64 — the safe supertype that preserves the
    full integer range; this matches Polars' default arithmetic supertype.
    """
    if _is_integer(a) and _is_integer(b):
        return max(a, b, key=lambda t: _NUMERIC_RANK[t])
    if a in _FLOAT and b in _FLOAT:
        return max(a, b, key=lambda t: _NUMERIC_RANK[t])
    return "Float64"


def _join(a: _InternalDtype, b: _InternalDtype) -> _InternalDtype | None:
    """Return the common type for ``a`` and ``b``, or ``None`` if incompatible.

    ``Null`` joins with everything (returns the other side). ``Unknown``
    propagates (returns ``Unknown``). Numeric types widen.
    """
    if a == "Unknown" or b == "Unknown":
        return "Unknown"
    if a == "Null":
        return b
    if b == "Null":
        return a
    if a == b:
        return a
    if _is_numeric(a) and _is_numeric(b):
        return _widen_numeric(a, b)
    return None


def _infer_lit(value: object) -> _InternalDtype:
    """Infer the dtype of a literal scalar."""
    # Python ``bool`` is a subclass of ``int`` — check it first.
    if isinstance(value, bool):
        return "Boolean"
    if value is None:
        return "Null"
    if isinstance(value, int):
        return "Int64"
    if isinstance(value, float):
        return "Float64"
    if isinstance(value, str):
        return "String"
    if isinstance(value, list):
        # Lists of scalars only appear in the ``in``/``not_in`` ``values``
        # slot, where each item is type-checked separately. A literal-list
        # ``Lit`` here doesn't have a useful single dtype, so return Unknown.
        return "Unknown"
    return "Unknown"


# ---------------------------------------------------------------------------
# Validator state and dispatch.
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    schema: Mapping[str, ast.Dtype]
    params: Mapping[str, ast.Dtype]
    issues: list[Issue] = field(default_factory=list)
    path: list[str] = field(default_factory=list)

    def at(self, segment: str) -> _Ctx:
        return _Ctx(
            schema=self.schema,
            params=self.params,
            issues=self.issues,
            path=[*self.path, segment],
        )

    def error(self, message: str) -> None:
        self.issues.append(Issue(self._render_path(), message, "error"))

    def warn(self, message: str) -> None:
        self.issues.append(Issue(self._render_path(), message, "warning"))

    def _render_path(self) -> str:
        if not self.path:
            return "<root>"
        out = ""
        for seg in self.path:
            if seg.startswith("[") or out == "":
                out += seg
            else:
                out += "." + seg
        return out


def validate(
    node: ast.Expr,
    schema: Mapping[str, ast.Dtype],
    params: Mapping[str, ast.Dtype] | None = None,
) -> ValidationResult:
    """Type-check ``node`` against ``schema`` (column → dtype) and ``params``.

    On success returns a :class:`ValidationResult` carrying the inferred
    root dtype and any warnings. On failure raises
    :class:`ValidationError` containing every error and warning collected
    during the walk (best-effort multi-error reporting).
    """
    ctx = _Ctx(schema=schema, params=params or {})
    dtype = _infer(node, ctx)

    errors = [i for i in ctx.issues if i.severity == "error"]
    warnings = [i for i in ctx.issues if i.severity == "warning"]
    if errors:
        raise ValidationError(ctx.issues)

    public = _publicize(dtype)
    if public is None:
        # Should not happen on a successful walk; defensive.
        raise ValidationError([Issue("<root>", f"unresolved type {dtype!r}")])
    return ValidationResult(dtype=public, warnings=warnings)


def _publicize(t: _InternalDtype) -> ast.Dtype | None:
    """Project an internal dtype to the public ``Dtype`` literal."""
    if t in _PUBLIC_DTYPES:
        # mypy can't narrow a frozenset membership check to a Literal — cast.
        return t  # type: ignore[return-value]
    if t == "Null":
        # A pure Null at the root is unusual but legal (e.g. ``lit None`` as
        # a metric placeholder). Surface as String to be conservative.
        return "String"
    return None


def _infer(node: ast.Expr, ctx: _Ctx) -> _InternalDtype:
    """Recursive type-inference dispatch."""
    match node:
        # --- atoms -----------------------------------------------------
        case ast.Col(col=name):
            t_col = ctx.schema.get(name)
            if t_col is None:
                ctx.error(f"column {name!r} not found in schema")
                return "Unknown"
            return t_col

        case ast.Lit(lit=value):
            return _infer_lit(value)

        case ast.Param(param=name):
            t_param = ctx.params.get(name)
            if t_param is None:
                ctx.error(f"parameter {name!r} not declared in workspace")
                return "Unknown"
            return t_param

        case ast.PolarsExpr():
            ctx.warn(
                "direct Polars expression is checked at runtime; static dtype inference "
                "uses String as a conservative placeholder"
            )
            return "String"

        # --- unary -----------------------------------------------------
        case ast.UnaryArg(op=op, arg=arg):
            inner = _infer(arg, ctx.at("arg"))
            if op == "not":
                if inner not in {"Boolean", "Null", "Unknown"}:
                    ctx.error(f"`not` requires Boolean, got {inner}")
                return "Boolean"
            # neg / abs / ceil / floor preserve numeric type; sqrt/exp -> Float64
            if not (_is_numeric(inner) or inner in {"Null", "Unknown"}):
                ctx.error(f"`{op}` requires numeric, got {inner}")
                return "Unknown"
            if op in {"sqrt", "exp"}:
                return "Float64"
            return inner

        case ast.OpLog(arg=arg, base=base):
            inner = _infer(arg, ctx.at("arg"))
            if not (_is_numeric(inner) or inner in {"Null", "Unknown"}):
                ctx.error(f"`log` requires numeric arg, got {inner}")
            # §6.4 constant-folding sanity: log(0) over a literal is invalid.
            if isinstance(arg, ast.Lit) and isinstance(arg.lit, (int, float)) and arg.lit <= 0:
                ctx.error(f"log of non-positive literal {arg.lit} is undefined")
            if base is not None and base <= 0:
                ctx.error(f"log base must be positive, got {base}")
            return "Float64"

        case ast.OpRound(arg=arg):
            inner = _infer(arg, ctx.at("arg"))
            if not (_is_numeric(inner) or inner in {"Null", "Unknown"}):
                ctx.error(f"`round` requires numeric arg, got {inner}")
                return "Unknown"
            return inner

        case ast.OpCast(arg=arg, dtype=dtype):
            _infer(arg, ctx.at("arg"))  # walk for column existence
            return dtype

        # --- nary ------------------------------------------------------
        case ast.LogicalNary(op=op, args=args):
            for i, a in enumerate(args):
                t = _infer(a, ctx.at(f"args[{i}]"))
                if t not in {"Boolean", "Null", "Unknown"}:
                    ctx.at(f"args[{i}]").error(f"`{op}` requires Boolean, got {t}")
            return "Boolean"

        case ast.ArithmeticNary(op=op, args=args):
            inferred: list[_InternalDtype] = []
            for i, a in enumerate(args):
                t = _infer(a, ctx.at(f"args[{i}]"))
                if not (_is_numeric(t) or t in {"Null", "Unknown"}):
                    ctx.at(f"args[{i}]").error(f"`{op}` requires numeric, got {t}")
                    inferred.append("Unknown")
                else:
                    inferred.append(t)
            if op == "div":
                return "Float64"
            non_null = [t for t in inferred if t not in {"Null", "Unknown"}]
            if not non_null:
                return "Unknown"
            result: _InternalDtype = non_null[0]
            for t in non_null[1:]:
                result = _widen_numeric(result, t)
            return result

        case ast.OpSafeDiv(num=num, den=den):
            t_num = _infer(num, ctx.at("num"))
            t_den = _infer(den, ctx.at("den"))
            if not (_is_numeric(t_num) or t_num in {"Null", "Unknown"}):
                ctx.at("num").error(f"`safe_div` num must be numeric, got {t_num}")
            if not (_is_numeric(t_den) or t_den in {"Null", "Unknown"}):
                ctx.at("den").error(f"`safe_div` den must be numeric, got {t_den}")
            # §6.4 sanity: literal zero divisor.
            if isinstance(den, ast.Lit) and isinstance(den.lit, (int, float)) and den.lit == 0:
                ctx.at("den").warn("`safe_div` denominator is literal zero; result will be 0.0")
            return "Float64"

        case ast.OpConcat(args=args):
            for i, a in enumerate(args):
                t = _infer(a, ctx.at(f"args[{i}]"))
                if t not in {"String", "Null", "Unknown"}:
                    ctx.at(f"args[{i}]").error(f"`concat` requires String, got {t}")
            return "String"

        case ast.MinMaxNary(op=op, args=args):
            inferred = [_infer(a, ctx.at(f"args[{i}]")) for i, a in enumerate(args)]
            joined: _InternalDtype = inferred[0]
            for t in inferred[1:]:
                j = _join(joined, t)
                if j is None:
                    ctx.error(f"`{op}` args have incompatible types {joined} vs {t}")
                    joined = "Unknown"
                else:
                    joined = j
            return joined

        case ast.OpCoalesce(args=args):
            inferred = [_infer(a, ctx.at(f"args[{i}]")) for i, a in enumerate(args)]
            joined = inferred[0]
            for t in inferred[1:]:
                j = _join(joined, t)
                if j is None:
                    ctx.error(f"`coalesce` args have incompatible types {joined} vs {t}")
                    joined = "Unknown"
                else:
                    joined = j
            return joined

        # --- predicates -----------------------------------------------
        case ast.ComparisonColumn(op=op, column=col, value=value):
            t_col = ctx.schema.get(col)
            if t_col is None:
                ctx.at(f"column={col}").error(f"column {col!r} not found in schema")
                return "Boolean"
            t_val = _infer_lit(value)
            if _join(t_col, t_val) is None:
                ctx.error(f"`{op}` on column {col!r} ({t_col}) compared to {t_val} value {value!r}")
            return "Boolean"

        case ast.ComparisonArgs(op=op, args=args):
            t0 = _infer(args[0], ctx.at("args[0]"))
            t1 = _infer(args[1], ctx.at("args[1]"))
            if _join(t0, t1) is None:
                ctx.error(f"`{op}` args have incompatible types {t0} vs {t1}")
            return "Boolean"

        case ast.OpIn(column=col, values=values) | ast.OpNotIn(column=col, values=values):
            t_col = ctx.schema.get(col)
            if t_col is None:
                ctx.error(f"column {col!r} not found in schema")
                return "Boolean"
            for i, v in enumerate(values):
                t_v = _infer_lit(v)
                if _join(t_col, t_v) is None:
                    ctx.at(f"values[{i}]").error(
                        f"value {v!r} ({t_v}) incompatible with column {col!r} ({t_col})"
                    )
            return "Boolean"

        case ast.OpBetween(column=col, low=low, high=high):
            t_col = ctx.schema.get(col)
            if t_col is None:
                ctx.error(f"column {col!r} not found in schema")
                return "Boolean"
            if not (_is_numeric(t_col) or _is_date_like(t_col)):
                ctx.error(f"`between` requires numeric or date column, got {t_col}")
            for slot, val in (("low", low), ("high", high)):
                t = _infer_lit(val)
                if _join(t_col, t) is None:
                    ctx.at(slot).error(
                        f"{slot} {val!r} ({t}) incompatible with column {col!r} ({t_col})"
                    )
            return "Boolean"

        case ast.NullCheck(column=col):
            if col not in ctx.schema:
                ctx.error(f"column {col!r} not found in schema")
            return "Boolean"

        case ast.OpMatches(column=col):
            t_col = ctx.schema.get(col)
            if t_col is None:
                ctx.error(f"column {col!r} not found in schema")
            elif t_col != "String":
                ctx.error(f"`matches` requires String column, got {t_col}")
            return "Boolean"

        case ast.StrPrefixSuffix(op=op, column=col):
            t_col = ctx.schema.get(col)
            if t_col is None:
                ctx.error(f"column {col!r} not found in schema")
            elif t_col != "String":
                ctx.error(f"`{op}` requires String column, got {t_col}")
            return "Boolean"

        # --- conditional ----------------------------------------------
        case ast.OpCase(when=branches, else_=else_):
            for i, br in enumerate(branches):
                t_cond = _infer(br.cond, ctx.at(f"when[{i}].cond"))
                if t_cond not in {"Boolean", "Null", "Unknown"}:
                    ctx.at(f"when[{i}].cond").error(
                        f"`case` branch cond must be Boolean, got {t_cond}"
                    )
            then_types = [
                _infer(br.then, ctx.at(f"when[{i}].then")) for i, br in enumerate(branches)
            ]
            t_else = _infer(else_, ctx.at("else"))
            joined = t_else
            for t in then_types:
                j = _join(joined, t)
                if j is None:
                    ctx.error(f"`case` branches have incompatible types {joined} vs {t}")
                    joined = "Unknown"
                else:
                    joined = j
            return joined

        case ast.OpWhenThen(cond=cond, then=then, else_=else_):
            t_cond = _infer(cond, ctx.at("cond"))
            if t_cond not in {"Boolean", "Null", "Unknown"}:
                ctx.at("cond").error(f"`when_then` cond must be Boolean, got {t_cond}")
            t_then = _infer(then, ctx.at("then"))
            t_else = _infer(else_, ctx.at("else"))
            joined_or_none = _join(t_then, t_else)
            if joined_or_none is None:
                ctx.error(f"`when_then` branches have incompatible types {t_then} vs {t_else}")
                return "Unknown"
            return joined_or_none

        # --- datetime --------------------------------------------------
        case ast.OpDateTrunc(arg=arg):
            t = _infer(arg, ctx.at("arg"))
            if not (_is_date_like(t) or t in {"Null", "Unknown"}):
                ctx.error(f"`date_trunc` requires Date/Datetime, got {t}")
                return "Unknown"
            return t

        case ast.OpDateDiff(end=end, start=start):
            te = _infer(end, ctx.at("end"))
            ts = _infer(start, ctx.at("start"))
            if not (_is_date_like(te) or te in {"Null", "Unknown"}):
                ctx.at("end").error(f"`date_diff` end must be date-like, got {te}")
            if not (_is_date_like(ts) or ts in {"Null", "Unknown"}):
                ctx.at("start").error(f"`date_diff` start must be date-like, got {ts}")
            return "Int64"

        case ast.OpDatePart(arg=arg):
            t = _infer(arg, ctx.at("arg"))
            if not (_is_date_like(t) or t in {"Null", "Unknown"}):
                ctx.error(f"`date_part` requires Date/Datetime, got {t}")
            return "Int64"

        case ast.OpNow():
            ctx.warn(
                "`now()` makes the resulting value time-dependent; the engine "
                "snapshots it once per run instead of evaluating per row"
            )
            return "Datetime"

        case ast.OpStrftime(arg=arg):
            t = _infer(arg, ctx.at("arg"))
            if not (_is_date_like(t) or t in {"Null", "Unknown"}):
                ctx.error(f"`strftime` requires Date/Datetime, got {t}")
            return "String"

        case ast.OpStrptime(arg=arg):
            t = _infer(arg, ctx.at("arg"))
            if t not in {"String", "Null", "Unknown"}:
                ctx.error(f"`strptime` requires String, got {t}")
            return "Datetime"

    # Should be unreachable — every Expr variant is matched above.
    raise AssertionError(f"unhandled node type: {type(node).__name__}")  # pragma: no cover


__all__ = ["Issue", "ValidationError", "ValidationResult", "validate"]
