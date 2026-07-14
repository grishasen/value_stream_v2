# Value Stream — Readers and Transforms

This doc specifies how files turn into a Polars LazyFrame ready for processors. It covers:

- the file-discovery and chunk-grouping rules,
- every built-in reader (Pega DS export, Parquet, CSV, XLSX) with format details and parameters,
- every built-in transform that runs between the reader and the processor fan-out,
- the source's `defaults` map and how it relates to transforms,
- expected schema invariants by the time the data reaches a processor.

Companion docs:

- concepts/domain-model.md — Source / Reader / Transform definitions.
- reference/expression-dsl.md — AST for `filter` and `derive_column` transforms.
- reference/processors.md — what processors expect to see in the resulting frame.

---

## 1. Source lifecycle

```
files in folder
   │
   ▼
[ Discovery & grouping ]    ← reader.file_pattern, reader.group_by_filename
   │
   ▼
[ Reader ]                   ← reader.kind: parquet | pega_ds_export | csv | xlsx
   │
   ▼
[ Transforms (ordered) ]     ← source.transforms[]
   │
   ▼
LazyFrame ready for processor.chunk_aggregate
```

A new `(chunk_id, files_in_chunk)` pair travels through this pipeline once per ingestion run.

---

## 2. Discovery and chunk grouping

### 2.1 Inputs

```yaml
sources:
  - id: ih
    reader:
      kind: pega_ds_export
      file_pattern: "**/*.zip"
      group_by_filename: '\d{8}(?=\d{6}_)'
      hive_partitioning: false
      streaming: true
      background: false
```

### 2.2 Algorithm

```text
files = sorted( glob(folder, file_pattern, recursive=true) )
if files is empty:
    fallback_pattern = '**/*.json'                   # convention
    files = glob(folder, fallback_pattern, recursive=true)
    reader.kind = 'pega_ds_export'                   # auto-promote

groups = defaultdict(set)
for f in files:
    if hive_partitioning:
        # the path's parent (a partition directory) is the file unit
        unit = parent(f)
    else:
        unit = f
    try:
        chunk_id = re.findall(group_by_filename, abspath(f))[0]
    except IndexError:
        chunk_id = basename(f)                       # one-file chunk
    groups[chunk_id].add(unit)

groups = sort_desc_by_chunk_id(groups)               # newest first
```

### 2.3 Skip already processed chunks

```text
ledger = read_chunks_ledger(source_id)
for chunk_id in groups:
    fingerprint = sha256(sorted(path, mtime_ns, size for path in chunk.files))
    if (source_id, chunk_id, source_computation_hash, fingerprint) in ledger and not force:
        skip
```

The effective idempotency key is `(source_id, chunk_id,
source_computation_hash, file_hash)`. The source computation hash covers
workspace defaults, source behavior, and all bound processors. `file_hash`
covers sorted absolute paths, nanosecond mtimes, and sizes. Unchanged inputs are
skipped; changed files or behavior are reprocessed without invalidating the
previous successful aggregate until the new run commits.

### 2.4 Chunk granularity

- **Daily file groups** are the recommended default for IH-shaped sources. The regex `\d{8}(?=\d{6}_)` captures `YYYYMMDD` from filenames like `interaction-XYZ-20240821000000_001.json.zip`.
- **One file per chunk** is fine for smaller sources; the regex falls back to the basename.
- **Hive-partitioned datasets** (e.g. monthly-partitioned Parquet folders) put one *partition directory* per chunk; the reader scans the whole directory at once.

### 2.5 Failure modes

| Symptom | Cause | Behavior |
|---|---|---|
| Folder doesn't exist | misconfiguration | engine refuses to start; CLI shows the path |
| Pattern matches no files | empty source | run finishes with `chunks_total=0`, status `ok` |
| Regex never matches | wrong `group_by_filename` | engine logs `chunk_id=basename(file)`; processed but not date-grouped |
| Two files belong to different chunks but share a name | rare; would collide | engine raises and aborts the run before processing |

---

## 3. Reader: `pega_ds_export`

Pega CDH "Dataset Export" files are gzipped or zipped JSON/NDJSON archives. Value Stream normalizes them to a single NDJSON stream, then `polars.scan_ndjson`s it.

### 3.1 Accepted file extensions

| Extension | Treatment |
|---|---|
| `.zip` | extract all `.json` and `.ndjson` members in alphabetical order; concatenate into one NDJSON stream |
| `.tar.gz`, `.tgz` | extract `.json`/`.ndjson` members from the tar; same concatenation |
| `.gz`, `.gzip` | gunzip in-place; expect a single JSON or NDJSON file |
| `.json`, `.ndjson` | read directly |

Each archive becomes one normalized NDJSON file in the OS temp directory; the temp directory is cleaned up at process exit.

### 3.2 Concatenation rules

When concatenating multiple JSON/NDJSON members, the reader appends a newline if the previous member did not end in one. This guarantees `polars.scan_ndjson` sees one record per line across boundaries.

### 3.3 Schema discovery

`scan_ndjson(infer_schema_length=100_000)` reads the first 100 K records to infer types. This is sufficient for IH exports up to billions of rows because Pega CDH exports use a stable per-tenant schema.

### 3.4 Parameters

```yaml
reader:
  kind: pega_ds_export
  file_pattern: "**/*.zip"
  group_by_filename: '\d{8}(?=\d{6}_)'
  streaming: true                  # use Polars streaming engine on collect
  infer_schema_length: 100000      # optional, default 100000
  archive_temp_dir: /tmp/valuestream   # optional, default OS tempdir
```

### 3.5 Exit-time cleanup

Every chunk creates one temp directory with the prefix `dataset_export_`. An `atexit` hook removes them on process shutdown. A janitor command (`valuestream vacuum --tmp`) removes orphans from prior crashes.

---

## 4. Reader: `parquet`

### 4.1 Parameters

```yaml
reader:
  kind: parquet
  file_pattern: "**/*.parquet"
  group_by_filename: '(\d{4}-\d{2}-\d{2})'   # optional
  hive_partitioning: true                    # if directory layout uses hive partitions
  streaming: true
  missing_columns: insert                    # behavior when a chunk's file lacks a known column
  extra_columns: ignore                      # behavior when a chunk's file has extra columns
```

### 4.2 Implementation

```text
ih = pl.scan_parquet(
    files,
    cache=False,
    hive_partitioning=hive_partitioning,
    missing_columns="insert",
    extra_columns="ignore",
)
```

### 4.3 Notes

- Value Stream expects upstream Parquet to be tabular and timezone-aware where applicable.
- Hive partition columns become regular columns in the LazyFrame, available to transforms.

---

## 5. Reader: `csv`

### 5.1 Parameters

```yaml
reader:
  kind: csv
  file_pattern: "**/*.csv"
  separator: auto              # auto | "," | ";" | "\t" | "|" | ":"
  infer_schema_length: 10000
  try_parse_dates: true
```

### 5.2 Delimiter detection

When `separator: auto`, the reader peeks at the first 2 lines and tries delimiters in this order:
`,  ;  \t  (space)  |  :`. The first delimiter that parses both lines into the same number of fields wins; otherwise it defaults to `,`.

### 5.3 Implementation

```text
ih = pl.scan_csv(
    file,
    separator=separator,
    infer_schema_length=infer_schema_length,
    try_parse_dates=try_parse_dates,
    cache=False,
)
```

---

## 6. Reader: `xlsx`

```yaml
reader:
  kind: xlsx
  file_pattern: "**/*.xlsx"
  sheet: 0           # int index or sheet name
```

### 6.1 Notes

- XLSX is read eagerly via `polars.read_excel(...)` and immediately converted to a LazyFrame.
- Multi-sheet workbooks must be expanded into one Source per sheet; Value Stream does not auto-fan-out.

---

## 7. Source `defaults` block

Defaults run after the reader, before transforms (or as the first transform — implementation-equivalent). Two behaviors:

| Case | Behavior |
|---|---|
| Column doesn't exist in the LazyFrame | add it as a literal with the default value |
| Column exists but has nulls | `fill_null(default)` |

Type promotion: if the default value parses as a float, it becomes Float64; otherwise it stays a string.

```yaml
sources:
  - id: ih
    defaults:
      ModelControlGroup: Test
      PlacementType: "N/A"
      ExperimentName: "N/A"
      ExperimentGroup: "N/A"
      FinalPropensity: 0.0
      Priority: 1e-10
      Revenue: 0.0
      ConversionEventID: "N/A"
```

---

## 8. Transforms catalog

A Source's `transforms` list is processed in order. Each transform is a typed dictionary `{kind: <name>, ...params}`. The full set:

### 8.1 `rename_capitalize`

Renames every column to its `Capitalized` form. The legacy app does this so Pega's `pyName` becomes `Name`, `pxFirstName` becomes `Pxfirstname`, etc. The transform also drops a configurable list of non-essential columns first.

```yaml
- kind: rename_capitalize
  drop_columns: [pxObjClass, pxApplication, pxOutboundChannelInfo]   # optional
```

### 8.2 `parse_datetime`

Parses string columns into `Datetime`.

```yaml
- kind: parse_datetime
  columns: [OutcomeTime, DecisionTime]
  format: "%Y%m%dT%H%M%S%.3f %Z"   # strptime format
  utc: true                         # convert to UTC
```

### 8.3 `derive_calendar`

Adds calendar columns from a timestamp.

```yaml
- kind: derive_calendar
  from: OutcomeTime
  outputs: [Day, Month, Year, Quarter]    # any subset
  week_iso: false                          # also compute "Week"
```

Output columns: `Day` (date), `Month` (string `YYYY-MM`), `Year` (Int16), `Quarter` (string `YYYY_Qn`), optional `Week` (string `YYYY-Www`).

### 8.4 `derive_action_id`

Concatenates business-key parts into a single `ActionID`. Mirrors Pega's `Issue/Group/Name`.

```yaml
- kind: derive_action_id
  parts: [Issue, Group, Name]
  sep: "/"
  output: ActionID
```

### 8.5 `derive_column`

Adds a derived column from an AST expression.

```yaml
- kind: derive_column
  output: ResponseTime
  expression:
    op: date_diff
    unit: seconds
    end:   {col: OutcomeTime}
    start: {col: DecisionTime}
```

### 8.6 `filter`

Restricts rows by an AST predicate.

```yaml
- kind: filter
  expression:
    op: and
    args:
      - {op: not_null, column: Channel}
      - {op: in, column: Outcome, values: [Impression, Clicked, Pending, Conversion]}
```

### 8.7 `dedup`

Deduplicates by a key set, optionally keeping the row with the maximum of a column.

```yaml
- kind: dedup
  keys: [InteractionID, ActionID, Rank, Outcome]
  keep: first                 # first | last | max | min
  on_column: Outcome_Binary    # only required if keep ∈ {max, min}
```

### 8.8 `cast`

Casts columns to a target dtype. Useful when the reader infers a wrong type.

```yaml
- kind: cast
  columns:
    Propensity: Float64
    Priority: Float64
```

### 8.9 `drop_columns`

Removes columns explicitly.

```yaml
- kind: drop_columns
  columns: [pxOutboundChannelInfo, pxOutboundCampaignInfo]
```

### 8.10 `coalesce`

Replaces nulls in `target` with the first non-null among `from`.

```yaml
- kind: coalesce
  target: ConversionEventID
  from: [ConversionEventID, Name]
```

(Pega's interaction history uses this pattern: when `ConversionEventID` is empty, use `Name`.)

---

## 9. Schema invariants at processor entry

By the time a chunk's LazyFrame reaches `processor.chunk_aggregate`, it must satisfy:

1. **Required group-by columns exist** — every transformed column referenced by `processor.group_by`.
2. **Required outcome / score / key columns exist** — for `binary_outcome`: `outcome.column`; for `score_distribution`: every selected `score_properties` column.
3. **Configured processor time input exists** — `processor.time.column`, when set; calendar output columns needed for physical grains are derived by transforms/grain handling.
4. **Timestamps are tz-aware** — UTC.
5. **Natural-key columns exist** if `dedup_keys` is set.
6. **No `eval`-derived columns remain** — all transforms ran in a typed AST.

The engine checks authored group-by, time, outcome, score/property, lifecycle,
entity, milestone, dedup, and explicit state-source columns against the
transformed LazyFrame schema before any processor writes. Missing inputs fail
the chunk with one grouped, actionable error rather than silently dropping a
dimension or state.

---

## 10. Memory budget

The engine measures memory pressure between chunks (`psutil.Process.memory_info().rss`) and:

- if `RSS > 0.7 × machine_total`, it spills the source's collected DataFrame to a temporary Parquet and re-scans for each processor (one extra disk pass for safety).
- if `RSS > 0.9 × machine_total`, it pauses processing and waits for memory to free.

These thresholds are configurable in `pipelines.yaml`:

```yaml
defaults:
  memory:
    spill_threshold: 0.7
    pause_threshold: 0.9
    rss_log_every_chunks: 31    # debug RSS log every Nth chunk
```

---

## 11. Logging

For each chunk, the reader emits structured log lines:

```text
{"event":"chunk_start", "source":"ih", "chunk_id":"20240821", "files":2}
{"event":"reader_done",  "source":"ih", "chunk_id":"20240821", "rows_in":12_345_678, "ms":4321}
{"event":"transforms_done","source":"ih","chunk_id":"20240821", "rows_kept":12_300_000, "ms":250}
{"event":"chunk_done",   "source":"ih", "chunk_id":"20240821", "ms":12_500}
```

These are correlated by `pipeline_run_id` for trace assembly.

---

## 12. Reader extensibility

A new reader implements:

```python
class Reader(Protocol):
    kind: str
    def discover(self, folder: str, params: dict) -> dict[str, list[Path]]:
        """Return chunk_id -> list of file paths."""
    def read_chunk(self, files: list[Path], params: dict) -> pl.LazyFrame:
        """Return a Polars LazyFrame for one chunk."""
```

Plug it into the registry:

```python
register_reader("oracle_table", OracleTableReader)
```

The new kind is then usable from YAML:

```yaml
reader:
  kind: oracle_table
  connection: ${env.ORACLE_DSN}
  query: "SELECT ... WHERE created_at >= :since"
  group_by_value: created_at_date
```

Built-in readers ship in `valuestream.readers.builtin`; community readers may live in plugins.

---

## 13. Examples

### 13.1 Pega CDH IH

```yaml
sources:
  - id: ih
    reader:
      kind: pega_ds_export
      file_pattern: "**/*.zip"
      group_by_filename: '\d{8}(?=\d{6}_)'
      streaming: true
    schema:
      timestamp_column: OutcomeTime
      natural_key: [InteractionID, ActionID, Rank, Outcome]
    transforms:
      - {kind: rename_capitalize}
      - {kind: parse_datetime, columns: [OutcomeTime, DecisionTime], format: "%Y%m%dT%H%M%S%.3f %Z"}
      - {kind: derive_calendar, from: OutcomeTime, outputs: [Day, Month, Year, Quarter]}
      - {kind: derive_action_id, parts: [Issue, Group, Name], sep: "/", output: ActionID}
      - {kind: derive_column, output: ResponseTime, expression: {op: date_diff, unit: seconds, end: {col: OutcomeTime}, start: {col: DecisionTime}}}
      - {kind: filter, expression: {op: not_null, column: Channel}}
      - {kind: coalesce, target: ConversionEventID, from: [ConversionEventID, Name]}
      - {kind: dedup, keys: [InteractionID, ActionID, Rank, Outcome]}
    defaults:
      ModelControlGroup: Test
      PlacementType: "N/A"
      ExperimentName: "N/A"
      ExperimentGroup: "N/A"
      FinalPropensity: 0.0
      Priority: 1e-10
      Revenue: 0.0
      ConversionEventID: "N/A"
```

### 13.2 Product Holdings (JSON exports)

```yaml
sources:
  - id: holdings
    reader:
      kind: pega_ds_export
      file_pattern: "**/*.json"
      group_by_filename: '(.*)'
      streaming: false
    transforms:
      - {kind: rename_capitalize}
      - {kind: parse_datetime, columns: [PurchasedDate], format: "%Y%m%dT%H%M%S%.3f %Z"}
      - {kind: derive_column, output: PurchasedDateTime, expression: {col: PurchasedDate}}
    defaults:
      Channel: "N/A"
```

### 13.3 Internal CSV

```yaml
sources:
  - id: subscriptions
    reader:
      kind: csv
      file_pattern: "**/*.csv"
      separator: ","
      try_parse_dates: true
    transforms:
      - {kind: cast, columns: {monthly_recurring: Float64, plan_id: Int32}}
      - {kind: derive_calendar, from: created_at, outputs: [Day, Month, Year, Quarter]}
```

### 13.4 Hive-partitioned Parquet

```yaml
sources:
  - id: events
    reader:
      kind: parquet
      file_pattern: "**/*.parquet"
      group_by_filename: '(\d{4}-\d{2}-\d{2})'
      hive_partitioning: true
      streaming: true
    transforms:
      - {kind: parse_datetime, columns: [event_ts]}
      - {kind: derive_calendar, from: event_ts, outputs: [Day, Month, Year, Quarter]}
```

---

## 14. Sanity checklist for new sources

Before binding a Processor to a new Source:

- [ ] `valuestream validate` passes against `pipelines.yaml`.
- [ ] `valuestream probe --source <id>` produces:
   - the inferred schema (column names + dtypes),
   - a 10-row sample after all transforms,
   - the calendar columns derived,
   - the count of distinct `chunk_id`s discovered.
- [ ] All processor `group_by` entries resolve to existing transformed columns.
- [ ] All processor outcome/score/key columns exist.
- [ ] No transform raised a warning about silently filling NULLs in a column the processor relies on.
