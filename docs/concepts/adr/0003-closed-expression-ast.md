# ADR 0003 — Closed Expression AST

**Status:** Accepted (backfilled 2026-07-13 from the replacement design)

## Context

Transforms, filters, and formula metrics need expressions. Embedding a
general-purpose language (Python snippets, raw SQL, eval) would make catalogs
unauditable, untypeable, and unsafe to accept from UI or LLM-assisted editors.

## Decision

Expressions are a closed, JSON-shaped AST with a fixed operator set (logical,
comparison, arithmetic, date/time helpers, safe division, and so on), formally
specified in the [expression DSL reference](../../reference/expression-dsl.md)
and translated deterministically to Polars expressions. Every expression in
the catalog is type-checked at validation time.

## Consequences

- Expressions are safe to author from the Builder, the AI Configuration
  Studio, and hand-written YAML alike — there is no code injection surface.
- The validator can type-check every transform, filter, and metric before any
  data is read.
- The operator set is deliberately limited; extending it means changing the
  grammar, the Polars translation, the JSON Schema, and the docs together.
- LLM-drafted catalogs are constrained to the same validated grammar.
