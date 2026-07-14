# Catalog Schemas

Every catalog YAML file validates against a JSON Schema checked into
`schemas/` at the repository root. `valuestream validate` applies them before
any deeper reference or expression checking, and the Builder and AI
Configuration Studio generate YAML that must pass the same schemas.

| Schema | Validates |
|---|---|
| `schemas/pipelines.json` | `catalog/pipelines.yaml` — workspace, sources, readers, schemas, transforms, defaults |
| `schemas/processors.json` | `catalog/processors.yaml` — processor kinds, dimensions, time grains, states, outcome/stage/score config |
| `schemas/metrics.json` | `catalog/metrics.yaml` — metric kinds, expressions, state bindings, display metadata |
| `schemas/dashboards.json` | `catalog/dashboards.yaml` — dashboards, pages, tiles, filters, presets, theme |
| `schemas/expr.json` | The closed expression AST embedded in transforms, filters, and formula metrics |
| `schemas/catalog.json` | Cross-file catalog envelope used by the loader |
| `schemas/kpi-recipes.json` | Reusable, inert KPI recipe-library artifacts used by both authoring Studios |

The KPI recipe schema is adjacent to the catalog schemas but recipes are not a
fifth workspace catalog file. Installation materializes validated metric and
dashboard entries; only those installed YAML entries affect runtime behavior.

## Conventions

- Every YAML example in this documentation is valid against these schemas. If
  an example fails to validate, treat it as a doc bug.
- Schema changes ship in the same commit as the loader/model change and the
  documentation update — the schemas are part of the spec.
- The expression grammar the schemas embed is specified in the
  [expression DSL reference](expression-dsl.md); the semantics of each
  processor field are in [processors](processors.md).
