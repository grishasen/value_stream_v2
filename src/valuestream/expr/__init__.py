"""Closed expression DSL: AST types, parser, validator, Polars translator.

The grammar is fixed in ``docs/EXPRESSION_DSL.md``. Nothing in this package
calls ``eval`` or ``exec``; every node compiles to exactly one Polars
expression.
"""
