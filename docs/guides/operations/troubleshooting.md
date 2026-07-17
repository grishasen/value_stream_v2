# Troubleshooting

Symptom lookup for operating a Value Stream workspace, plus the data to
capture when escalating.

## Symptom Table

| Symptom | Likely cause | Action |
|---|---|---|
| Validation fails before UI loads | YAML shape, missing metric, bad expression, or tile binding issue | Run `valuestream validate`, fix reported file and location |
| No chunks discovered | Wrong source root, file pattern, or filename grouping regex | Run `probe`, inspect `pipelines.yaml` reader settings |
| Run skips data unexpectedly | Ledger sees chunks as already complete | Re-run with `--force` if rebuild is intended |
| Previous run remains `running` after a crash | The process ended before the terminal run update | Start the same source normally. Under the source lock the engine verifies committed chunks, marks the stale run `partial`/`failed`, and reprocesses only unverified chunks; do not use `--force` |
| Interrupted run becomes `failed` and a chunk is replayed | Fingerprint, lineage, physical file, or computation-hash recovery verification failed | Inspect the affected chunk error in Ops; retain prior files until the new run succeeds, then preview `vacuum --dry-run` |
| Report page has no data | Aggregates are missing for the metric, grain, or filter | Run workspace, check Ops, inspect tile in Reports |
| KPI tile shows `not ready` / **Backfill required** | A recipe or catalog edit added Processor state that is absent from current-hash aggregates | Run ingestion for a new workspace or reprocess/backfill the affected source; old aggregate schemas are not mixed with the new state |
| Filter has no effect on one tile | Tile's backing processor does not persist that dimension | Add the dimension to processor group-by and reprocess, or remove the filter expectation |
| Export skips a metric | Metric cannot be materialized at requested grain or has query errors | Inspect skipped metric reason in command output |
| Backfill imports fewer tables than expected | Legacy table names do not map cleanly to catalog targets | Review backfill output and migration report |
| Streamlit shows stale state | Browser or Streamlit session cache is stale | Refresh page or restart `valuestream serve` |

## Escalation Data to Capture

When raising an issue, include:

- Workspace path.
- Command and full options used.
- Catalog hash from the UI sidebar or command output.
- Validation output.
- Source id and chunk id if ingestion failed.
- Metric, grain, filters, and dashboard tile id if reporting failed.
- Relevant rows from the Ops page or run output.

## Related Docs

- [Operations runbook](runbook.md) — the standard operating loop.
- [FAQ](../../reference/faq.md) — questions grouped by area, including
  querying and performance.
