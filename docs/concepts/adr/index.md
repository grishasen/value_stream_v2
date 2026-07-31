# Architecture Decision Records

Significant, hard-to-reverse design decisions are recorded here as ADRs: the
context at the time, the decision, and its consequences. The first four are
backfilled from the [replacement design](../../design/replacement-design.md);
new decisions get a new numbered file in this folder.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-aggregate-first-storage.md) | Aggregate-first business/query storage, with bounded internal-state exceptions | Accepted |
| [0002](0002-yaml-catalog-as-source-of-behavior.md) | Declarative YAML catalog as the source of behavior | Accepted |
| [0003](0003-closed-expression-ast.md) | Closed expression AST instead of embedded code | Accepted |
| [0004](0004-chunk-ledger-idempotency.md) | Chunk ledger with computation hashes for idempotent ingestion | Accepted |
| [0005](0005-unified-metric-owned-heatmaps.md) | One metric-owned adaptive heatmap; old aliases are rejected | Accepted |
| [0006](0006-strict-catalog-v2.md) | Strict versioned catalog with no compatibility translation | Accepted |
| [0007](0007-bounded-lookback-processors.md) | Bounded lookback with optional rebuildable sharded checkpoints | Accepted |

## Writing a New ADR

Copy the section structure of any existing record (Status, Context, Decision,
Consequences), number it sequentially, add it to the table above, and land it
in the same PR as the change it justifies. Superseded decisions stay in place
with status "Superseded by NNNN".
