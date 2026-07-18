"""Apply configured source transforms to a Polars LazyFrame."""

from __future__ import annotations

import polars as pl

from valuestream.config import model
from valuestream.expr.translator import translate
from valuestream.utils.names import capitalize_fields
from valuestream.utils.timer import timed


@timed
def apply_transforms(frame: pl.LazyFrame, source: model.Source) -> pl.LazyFrame:
    """Apply source defaults and ordered transforms."""
    out = _apply_defaults(frame, source.defaults)
    for transform in source.transforms:
        out = _apply_one(out, transform)
    return out


def _apply_one(frame: pl.LazyFrame, transform: model.Transform) -> pl.LazyFrame:  # noqa: PLR0911, PLR0912
    if isinstance(transform, model.RenameCapitalize):
        names = frame.collect_schema().names()
        return frame.rename(dict(zip(names, capitalize_fields(names), strict=False)))
    if isinstance(transform, model.ParseDatetime):
        # Readers may already type these columns (CSV date inference, parquet
        # schemas). Parsing is "ensure datetime": only string columns need
        # strptime, so the same config works for preview and runtime reads.
        schema = frame.collect_schema()
        return frame.with_columns(
            [
                pl.col(column).str.strptime(pl.Datetime, format=transform.format, strict=False)
                for column in transform.columns
                if column in schema.names() and schema[column] == pl.String
            ]
        )
    if isinstance(transform, model.DeriveCalendar):
        base = pl.col(transform.from_)
        expressions: list[pl.Expr] = []
        for output in transform.outputs:
            expressions.append(_calendar_expr(base, output).alias(output))
        return frame.with_columns(expressions)
    if isinstance(transform, model.DeriveActionId):
        return frame.with_columns(
            pl.concat_str(
                [pl.col(part).cast(pl.String) for part in transform.parts], separator=transform.sep
            )
            .fill_null("")
            .alias("ActionID")
        )
    if isinstance(transform, model.DeriveColumn):
        return frame.with_columns(translate(transform.expression).alias(transform.output))
    if isinstance(transform, model.FilterTransform):
        return frame.filter(translate(transform.expression))
    if isinstance(transform, model.Dedup):
        keys = [key for key in transform.keys if key in frame.collect_schema().names()]
        return frame.unique(subset=keys, keep="first") if keys else frame
    if isinstance(transform, model.Defaults):
        return _apply_defaults(frame, transform.values)
    if isinstance(transform, model.Cast):
        return frame.with_columns(
            [
                pl.col(column).cast(_polars_dtype(dtype))
                for column, dtype in transform.columns.items()
                if column in frame.collect_schema().names()
            ]
        )
    if isinstance(transform, model.DropColumns):
        existing = [
            column for column in transform.columns if column in frame.collect_schema().names()
        ]
        return frame.drop(existing) if existing else frame
    if isinstance(transform, model.Coalesce):
        columns = [
            column for column in transform.columns if column in frame.collect_schema().names()
        ]
        if not columns:
            return frame
        return frame.with_columns(
            pl.coalesce([pl.col(column) for column in columns]).alias(transform.output)
        )
    return frame


def _apply_defaults(frame: pl.LazyFrame, defaults: dict[str, object]) -> pl.LazyFrame:
    if not defaults:
        return frame
    names = set(frame.collect_schema().names())
    expressions: list[pl.Expr] = []
    for column, value in defaults.items():
        if column in names:
            expressions.append(pl.col(column).fill_null(value).alias(column))
        else:
            expressions.append(pl.lit(value).alias(column))
    return frame.with_columns(expressions)


def _calendar_expr(base: pl.Expr, output: str) -> pl.Expr:
    normalized = output.lower()
    if normalized == "day":
        expr = base.dt.date()
    elif normalized == "hour":
        expr = base.dt.strftime("%Y-%m-%dT%H")
    elif normalized == "month":
        expr = base.dt.strftime("%Y-%m")
    elif normalized == "year":
        expr = base.dt.year().cast(pl.Int16)
    elif normalized == "quarter":
        expr = pl.concat_str(
            [base.dt.year().cast(pl.String), pl.lit("_Q"), base.dt.quarter().cast(pl.String)]
        )
    elif normalized == "week":
        expr = base.dt.strftime("%G-W%V")
    else:
        expr = base.dt.strftime("%Y-%m-%d")
    return expr


def _polars_dtype(dtype: model.Dtype) -> pl.DataType:
    return {
        "Int8": pl.Int8,
        "Int16": pl.Int16,
        "Int32": pl.Int32,
        "Int64": pl.Int64,
        "Float32": pl.Float32,
        "Float64": pl.Float64,
        "String": pl.String,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
        "Boolean": pl.Boolean,
    }[dtype]()


__all__ = ["apply_transforms"]
