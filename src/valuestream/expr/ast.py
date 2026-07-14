"""Closed expression AST.

Implements the grammar in ``docs/EXPRESSION_DSL.md`` §2 as a Pydantic v2
discriminated union. Every input dict is routed to exactly one node class
based on its shape (the ``op`` string for op-having nodes, or the presence
of ``col`` / ``lit`` / ``param`` for atoms).

Ops with identical field shapes are grouped into one class — ``UnaryArg``
covers ``not``/``neg``/``abs``/``sqrt``/``exp``/``ceil``/``floor`` because
each takes a single ``arg`` and nothing else. The translator pattern-matches
on ``(type(node), node.op)`` to dispatch.

Recursive references in atoms' ``Scalar`` and across node types use
``from __future__ import annotations`` plus ``model_rebuild()`` at the end.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator

# ---------------------------------------------------------------------------
# Scalar values for ``lit`` atoms and predicate ``value(s)`` slots.
# ---------------------------------------------------------------------------
# Per ``docs/EXPRESSION_DSL.md`` §2:
#   ``scalar ::= number | string | bool | null | array<scalar>``  (recursive)
#
# Pydantic 2.x can't materialize a fully self-referential type alias without
# blowing the schema generator, so we type the surface as one level of
# list-nesting (which covers every real-world use) and enforce the full
# recursive grammar in ``_check_scalar`` at validate time.
ScalarLeaf: TypeAlias = int | float | str | bool | None
ScalarValue: TypeAlias = ScalarLeaf | list[Any]


def _check_scalar(value: Any) -> None:
    """Recursively assert ``value`` is a JSON scalar or list of scalars.

    Raises ``ValueError`` on non-scalar leaves; otherwise returns ``None``.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            _check_scalar(item)
        return
    raise ValueError(
        f"not a scalar: {value!r} (type={type(value).__name__}); "
        "scalars must be number / string / bool / null / array<scalar>"
    )


# ---------------------------------------------------------------------------
# dtype enumeration (per EXPRESSION_DSL.md §2 ``dtype`` production).
# ---------------------------------------------------------------------------
Dtype = Literal[
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

# Unit enumerations for date helpers.
DateTruncUnit = Literal["day", "month", "quarter", "year", "hour", "week_iso"]
DateDiffUnit = Literal["seconds", "minutes", "hours", "days", "months", "years"]
DatePartUnit = Literal["year", "month", "day", "quarter", "hour", "weekday"]

# Unary-arg ops that all share the same shape (single ``arg``, no params).
UnaryArgOp = Literal["not", "neg", "abs", "sqrt", "exp", "ceil", "floor"]

# Logical and arithmetic n-ary op groupings.
LogicalOp = Literal["and", "or"]
ArithmeticOp = Literal["add", "sub", "mul", "div"]
MinMaxOp = Literal["least", "greatest"]
ComparisonOp = Literal["eq", "ne", "lt", "le", "gt", "ge"]
NullCheckOp = Literal["is_null", "not_null"]
StrPrefixSuffixOp = Literal["starts_with", "ends_with"]


# ---------------------------------------------------------------------------
# Common base.
# ---------------------------------------------------------------------------
class _Node(BaseModel):
    """Base for AST nodes.

    ``extra='forbid'`` prevents drift — if the YAML carries a key the AST
    doesn't recognize, validation fails loudly.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)


# ---------------------------------------------------------------------------
# Atoms.
# ---------------------------------------------------------------------------
class Col(_Node):
    """``{col: <ident>}`` — reference to a column by name."""

    col: str = Field(min_length=1)


class Lit(_Node):
    """``{lit: <scalar>}`` — literal value."""

    lit: ScalarValue

    @field_validator("lit", mode="after")
    @classmethod
    def _validate(cls, v: Any) -> Any:
        _check_scalar(v)
        return v


class Param(_Node):
    """``{param: <ident>}`` — workspace-level parameter reference."""

    param: str = Field(min_length=1)


class PolarsExpr(_Node):
    """``{polars: <expr>}`` — direct Polars expression text for advanced transforms."""

    polars: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Unary nodes.
# ---------------------------------------------------------------------------
class UnaryArg(_Node):
    """``not`` / ``neg`` / ``abs`` / ``sqrt`` / ``exp`` / ``ceil`` / ``floor``."""

    op: UnaryArgOp
    arg: Expr


class OpLog(_Node):
    """``{op: log, arg: ..., base?: number}`` — natural log unless ``base`` set."""

    op: Literal["log"]
    arg: Expr
    base: float | None = None


class OpRound(_Node):
    """``{op: round, arg: ..., ndigits?: int}``."""

    op: Literal["round"]
    arg: Expr
    ndigits: int | None = None


class OpCast(_Node):
    """``{op: cast, arg: ..., dtype: <Dtype>}``."""

    op: Literal["cast"]
    arg: Expr
    dtype: Dtype


# ---------------------------------------------------------------------------
# Nary nodes.
# ---------------------------------------------------------------------------
class LogicalNary(_Node):
    """``{op: and|or, args: [...]}`` — short-circuit logical n-ary."""

    op: LogicalOp
    args: list[Expr] = Field(min_length=2)


class ArithmeticNary(_Node):
    """``{op: add|sub|mul|div, args: [...]}``."""

    op: ArithmeticOp
    args: list[Expr] = Field(min_length=2)


class OpSafeDiv(_Node):
    """``{op: safe_div, num: ..., den: ...}`` — returns 0.0 when ``den == 0``."""

    op: Literal["safe_div"]
    num: Expr
    den: Expr


class OpConcat(_Node):
    """``{op: concat, args: [...], sep?: <string>}``."""

    op: Literal["concat"]
    args: list[Expr] = Field(min_length=1)
    sep: str | None = None


class MinMaxNary(_Node):
    """``{op: least|greatest, args: [...]}``."""

    op: MinMaxOp
    args: list[Expr] = Field(min_length=2)


class OpCoalesce(_Node):
    """``{op: coalesce, args: [...]}`` — first non-null wins."""

    op: Literal["coalesce"]
    args: list[Expr] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Predicate nodes — comparisons (column-form vs args-form).
# ---------------------------------------------------------------------------
class ComparisonColumn(_Node):
    """Column-form ``eq``/``ne``/``lt``/``le``/``gt``/``ge`` against a scalar."""

    op: ComparisonOp
    column: str = Field(min_length=1)
    value: ScalarValue

    @field_validator("value", mode="after")
    @classmethod
    def _validate(cls, v: Any) -> Any:
        _check_scalar(v)
        return v


class ComparisonArgs(_Node):
    """Args-form ``eq``/``ne``/``lt``/``le``/``gt``/``ge`` between two expressions."""

    op: ComparisonOp
    args: list[Expr] = Field(min_length=2, max_length=2)


# ---------------------------------------------------------------------------
# Predicate nodes — set / range / null / regex / string-affix.
# ---------------------------------------------------------------------------
class OpIn(_Node):
    op: Literal["in"]
    column: str = Field(min_length=1)
    values: list[ScalarValue]

    @field_validator("values", mode="after")
    @classmethod
    def _validate(cls, v: list[Any]) -> list[Any]:
        for item in v:
            _check_scalar(item)
        return v


class OpNotIn(_Node):
    op: Literal["not_in"]
    column: str = Field(min_length=1)
    values: list[ScalarValue]

    @field_validator("values", mode="after")
    @classmethod
    def _validate(cls, v: list[Any]) -> list[Any]:
        for item in v:
            _check_scalar(item)
        return v


class OpBetween(_Node):
    op: Literal["between"]
    column: str = Field(min_length=1)
    low: ScalarValue
    high: ScalarValue

    @field_validator("low", "high", mode="after")
    @classmethod
    def _validate(cls, v: Any) -> Any:
        _check_scalar(v)
        return v


class NullCheck(_Node):
    """``is_null`` / ``not_null`` over a column."""

    op: NullCheckOp
    column: str = Field(min_length=1)


class OpMatches(_Node):
    op: Literal["matches"]
    column: str = Field(min_length=1)
    pattern: str


class StrPrefixSuffix(_Node):
    """``starts_with`` / ``ends_with`` against a literal string."""

    op: StrPrefixSuffixOp
    column: str = Field(min_length=1)
    value: str


# ---------------------------------------------------------------------------
# Conditional nodes.
# ---------------------------------------------------------------------------
class CaseBranch(_Node):
    """One branch of an ``op: case`` node."""

    cond: Expr
    then: Expr


class OpCase(_Node):
    """``{op: case, when: [...], else: ...}`` — n-ary conditional."""

    op: Literal["case"]
    when: list[CaseBranch] = Field(min_length=1)
    # ``else`` is a Python keyword; expose as ``else_`` with alias ``else``.
    else_: Expr = Field(alias="else")


class OpWhenThen(_Node):
    """``{op: when_then, cond: ..., then: ..., else: ...}`` — binary conditional."""

    op: Literal["when_then"]
    cond: Expr
    then: Expr
    else_: Expr = Field(alias="else")


# ---------------------------------------------------------------------------
# Datetime nodes.
# ---------------------------------------------------------------------------
class OpDateTrunc(_Node):
    op: Literal["date_trunc"]
    unit: DateTruncUnit
    arg: Expr


class OpDateDiff(_Node):
    op: Literal["date_diff"]
    unit: DateDiffUnit
    end: Expr
    start: Expr


class OpDatePart(_Node):
    op: Literal["date_part"]
    unit: DatePartUnit
    arg: Expr


class OpNow(_Node):
    op: Literal["now"]


class OpStrftime(_Node):
    op: Literal["strftime"]
    arg: Expr
    format: str


class OpStrptime(_Node):
    op: Literal["strptime"]
    arg: Expr
    format: str


# ---------------------------------------------------------------------------
# Discriminator and discriminated union.
# ---------------------------------------------------------------------------

# Set membership for the callable discriminator (kept in sync with the
# Literal types above; tested in ``tests/unit/test_expr_parser.py``).
_UNARY_ARG_OPS: frozenset[str] = frozenset(("not", "neg", "abs", "sqrt", "exp", "ceil", "floor"))
_LOGICAL_OPS: frozenset[str] = frozenset(("and", "or"))
_ARITHMETIC_OPS: frozenset[str] = frozenset(("add", "sub", "mul", "div"))
_MINMAX_OPS: frozenset[str] = frozenset(("least", "greatest"))
_COMPARISON_OPS: frozenset[str] = frozenset(("eq", "ne", "lt", "le", "gt", "ge"))
_NULLCHECK_OPS: frozenset[str] = frozenset(("is_null", "not_null"))
_STR_AFFIX_OPS: frozenset[str] = frozenset(("starts_with", "ends_with"))


def _discriminator(value: Any) -> str | None:
    """Route a value to a discriminator tag.

    Called during both validation (dict input) and serialization (model
    input). Returns ``None`` for unrecognized shapes; Pydantic then raises a
    clean "no matching variant" error pointing at the offending node.
    """
    # Serialization path: Pydantic hands us already-instantiated models.
    if isinstance(value, _Node):
        tag = _MODEL_TO_TAG.get(type(value))
        if tag is not None:
            return tag

    if not isinstance(value, dict):
        return None

    if "col" in value and "op" not in value:
        return "col"
    if "lit" in value and "op" not in value:
        return "lit"
    if "param" in value and "op" not in value:
        return "param"
    if "polars" in value and "op" not in value:
        return "polars"

    op = value.get("op")
    if not isinstance(op, str):
        return None

    if op in _UNARY_ARG_OPS:
        return "unary_arg"
    if op in _LOGICAL_OPS:
        return "logical"
    if op in _ARITHMETIC_OPS:
        return "arithmetic"
    if op in _MINMAX_OPS:
        return "minmax"
    if op in _COMPARISON_OPS:
        if "column" in value:
            return "cmp_column"
        if "args" in value:
            return "cmp_args"
        return None
    if op in _NULLCHECK_OPS:
        return "nullcheck"
    if op in _STR_AFFIX_OPS:
        return "str_affix"

    # Single-tag ops — the tag equals the op string.
    if op in _SINGLE_TAG_OPS:
        return op

    return None


_SINGLE_TAG_OPS: frozenset[str] = frozenset(
    (
        "log",
        "round",
        "cast",
        "safe_div",
        "concat",
        "coalesce",
        "in",
        "not_in",
        "between",
        "matches",
        "case",
        "when_then",
        "date_trunc",
        "date_diff",
        "date_part",
        "now",
        "strftime",
        "strptime",
    )
)


# Inverse map for the serialization path of the discriminator. Populated
# below after every node class is defined, then frozen.
_MODEL_TO_TAG: dict[type[_Node], str] = {}


# The single ``Expr`` type — every value flowing through parser / validator /
# translator is one of these.
Expr: TypeAlias = Annotated[
    Annotated[Col, Tag("col")]
    | Annotated[Lit, Tag("lit")]
    | Annotated[Param, Tag("param")]
    | Annotated[PolarsExpr, Tag("polars")]
    | Annotated[UnaryArg, Tag("unary_arg")]
    | Annotated[OpLog, Tag("log")]
    | Annotated[OpRound, Tag("round")]
    | Annotated[OpCast, Tag("cast")]
    | Annotated[LogicalNary, Tag("logical")]
    | Annotated[ArithmeticNary, Tag("arithmetic")]
    | Annotated[OpSafeDiv, Tag("safe_div")]
    | Annotated[OpConcat, Tag("concat")]
    | Annotated[MinMaxNary, Tag("minmax")]
    | Annotated[OpCoalesce, Tag("coalesce")]
    | Annotated[ComparisonColumn, Tag("cmp_column")]
    | Annotated[ComparisonArgs, Tag("cmp_args")]
    | Annotated[OpIn, Tag("in")]
    | Annotated[OpNotIn, Tag("not_in")]
    | Annotated[OpBetween, Tag("between")]
    | Annotated[NullCheck, Tag("nullcheck")]
    | Annotated[OpMatches, Tag("matches")]
    | Annotated[StrPrefixSuffix, Tag("str_affix")]
    | Annotated[OpCase, Tag("case")]
    | Annotated[OpWhenThen, Tag("when_then")]
    | Annotated[OpDateTrunc, Tag("date_trunc")]
    | Annotated[OpDateDiff, Tag("date_diff")]
    | Annotated[OpDatePart, Tag("date_part")]
    | Annotated[OpNow, Tag("now")]
    | Annotated[OpStrftime, Tag("strftime")]
    | Annotated[OpStrptime, Tag("strptime")],
    Discriminator(_discriminator),
]


# Populate the model-to-tag inverse table for the serialization path.
_MODEL_TO_TAG.update(
    {
        Col: "col",
        Lit: "lit",
        Param: "param",
        PolarsExpr: "polars",
        UnaryArg: "unary_arg",
        OpLog: "log",
        OpRound: "round",
        OpCast: "cast",
        LogicalNary: "logical",
        ArithmeticNary: "arithmetic",
        OpSafeDiv: "safe_div",
        OpConcat: "concat",
        MinMaxNary: "minmax",
        OpCoalesce: "coalesce",
        ComparisonColumn: "cmp_column",
        ComparisonArgs: "cmp_args",
        OpIn: "in",
        OpNotIn: "not_in",
        OpBetween: "between",
        NullCheck: "nullcheck",
        OpMatches: "matches",
        StrPrefixSuffix: "str_affix",
        OpCase: "case",
        OpWhenThen: "when_then",
        OpDateTrunc: "date_trunc",
        OpDateDiff: "date_diff",
        OpDatePart: "date_part",
        OpNow: "now",
        OpStrftime: "strftime",
        OpStrptime: "strptime",
    }
)


# Resolve forward references inside every model now that Expr is bound.
_models_to_rebuild: tuple[type[_Node], ...] = (*_MODEL_TO_TAG, CaseBranch)
for _model in _models_to_rebuild:
    _model.model_rebuild()


__all__ = [
    "ArithmeticNary",
    "CaseBranch",
    "Col",
    "ComparisonArgs",
    "ComparisonColumn",
    "Dtype",
    "Expr",
    "Lit",
    "LogicalNary",
    "MinMaxNary",
    "NullCheck",
    "OpBetween",
    "OpCase",
    "OpCast",
    "OpCoalesce",
    "OpConcat",
    "OpDateDiff",
    "OpDatePart",
    "OpDateTrunc",
    "OpIn",
    "OpLog",
    "OpMatches",
    "OpNotIn",
    "OpNow",
    "OpRound",
    "OpSafeDiv",
    "OpStrftime",
    "OpStrptime",
    "OpWhenThen",
    "Param",
    "PolarsExpr",
    "ScalarValue",
    "StrPrefixSuffix",
    "UnaryArg",
]
