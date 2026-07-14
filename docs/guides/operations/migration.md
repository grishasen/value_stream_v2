# Migration and Backfill

This guide covers moving legacy dashboard assets into Value Stream: catalog
migration from legacy TOML, aggregate backfill from a legacy DuckDB database,
and the parity checks before sign-off.

## Translate Legacy TOML

```sh
uv run valuestream migrate --from value_dashboard/config/demo.toml --to workspaces/demo/catalog
```

The command writes catalog YAML files and a `migration_report.md` into the
target catalog directory.

## Validate the Generated Workspace

```sh
uv run valuestream validate workspaces/demo
```

Resolve validation errors before attempting backfill or side-by-side
reporting.

## Review the Migration Report

Check the report for:

- Mapped legacy fields.
- Unmapped or unsupported fields.
- Assumptions in source, processor, metric, and dashboard translation.
- Processor changes that require raw replay.
- Metrics that need manual review.

Treat gaps in the report as migration tasks, not harmless warnings.

## Use the Curated FAT Example

`examples/fat` (documented in `examples/fat/README.md`) is a reviewed migration of a
legacy Interaction History and Product Holdings TOML. Unlike raw translator
output, it resolves the manual AST gaps, replaces a non-deterministic revenue
simulation, separates processor state from derived metrics, and groups the
result into business-oriented report pages. Use it as a reference when a
legacy workspace needs engagement, conversion, audience, funnel, model,
experiment, distribution, and lifecycle coverage from the same two sources.

The example is catalog-only and validates without source data:

```sh
uv run valuestream validate examples/fat
```

Review its documented assumptions before copying it. Broad dimensions such as
action name and treatment increase aggregate cardinality, and its deterministic
revenue placeholder must be replaced with the real business value field.

## Backfill Legacy DuckDB Aggregates

Backfill legacy aggregate tables into the Value Stream aggregate layout:

```sh
uv run valuestream backfill --workspace workspaces/demo --from-legacy-db db/pov_data_demo.duckdb
```

The backfill command imports legacy aggregate tables into the partitioned
aggregate layout when they can be mapped to catalog targets.

After backfill:

1. Validate the workspace.
2. Open Reports and compare key metrics.
3. Export DuckDB tables if downstream tools need SQL access.
4. Record any parity gaps in the migration report.

## Parity Check

1. Validate the workspace.
2. Start the UI.
3. Compare key dashboard metrics against the legacy dashboard.
4. Export DuckDB metric tables if downstream SQL tools need parity checks.
5. Record unresolved differences in `migration_report.md` or the migration
   tracking issue.

## Cleanup

Preview cleanup:

```sh
uv run valuestream vacuum workspaces/demo --dry-run
```

Apply cleanup only after parity review:

```sh
uv run valuestream vacuum workspaces/demo
```

## Related Docs

- [Operations runbook](runbook.md) — the ongoing operating loop after migration.
- [Business functionality](../../concepts/business-functionality.md) — what
  migration delivers in business terms.
- [Replacement design §12](../../design/replacement-design.md) — migration
  design and rationale.
- [FAQ §I](../../reference/faq.md) — the migration cookbook.
