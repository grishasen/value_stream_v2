# Value Stream — Expression DSL

This doc specifies the closed expression language used everywhere Value Stream needs a dynamic predicate or formula:

- `transforms[*].kind: filter` — predicate over rows.
- `transforms[*].kind: derive_column` — expression producing a new column.
- `processors.<id>.filter` — predicate over rows pre-aggregation.
- `metrics.<id>.expression` — formula over a Processor's state columns (when `kind: formula`).
- `state.<name>.where` — conditional state (snapshot/funnel processors).

The DSL is a small AST encoded as JSON / YAML dicts. It is **never** Python and is **never** `eval`-ed. The evaluator translates each AST node into a `polars.Expr` at load time.

---

## 1. Design goals

- **Closed.** A finite set of operators; no escape hatches.
- **Typed.** Every node has a known result type, checked at validate time.
- **Polars-native.** Each node maps to one Polars expression — no Python callbacks.
- **Round-trippable.** Stable canonical form for hashing into `config_hash`.
- **Static-analyzable.** A linter can detect references to nonexistent columns, type mismatches, and accidental cross-grain uses.

---

## 2. Grammar (BNF)

```bnf
expr  ::= atom | unary | nary | predicate | conditional | datetime

atom  ::= { "col": <ident> }
        | { "lit": <scalar> }
        | { "param": <ident> }                 # workspace-level parameter

scalar ::= number | string | bool | null | array<scalar>

unary ::= { "op": "not",   "arg": expr }
        | { "op": "neg",   "arg": expr }
        | { "op": "abs",   "arg": expr }
        | { "op": "sqrt",  "arg": expr }
        | { "op": "log",   "arg": expr,  "base"?: number }
        | { "op": "exp",   "arg": expr }
        | { "op": "ceil"   "arg": expr }
        | { "op": "floor", "arg": expr }
        | { "op": "round", "arg": expr,  "ndigits"?: int }
        | { "op": "cast",  "arg": expr,  "dtype": <dtype> }

nary  ::= { "op": "and"|"or", "args": [ expr, expr, ... ] }
        | { "op": "add"|"sub"|"mul"|"div", "args": [ expr, expr, ... ] }
        | { "op": "safe_div", "num": expr, "den": expr }
        | { "op": "concat",   "args": [ expr, ... ], "sep"?: string }
        | { "op": "least"|"greatest", "args": [ expr, ... ] }
        | { "op": "coalesce", "args": [ expr, ... ] }

predicate ::= { "op": "eq"|"ne"|"lt"|"le"|"gt"|"ge",
                "column": <ident>, "value": <scalar> }
            | { "op": "eq"|"ne"|"lt"|"le"|"gt"|"ge",
                "args": [ expr, expr ] }
            | { "op": "in"|"not_in", "column": <ident>, "values": [ <scalar>, ... ] }
            | { "op": "between",     "column": <ident>, "low": <scalar>, "high": <scalar> }
            | { "op": "is_null"|"not_null",   "column": <ident> }
            | { "op": "matches", "column": <ident>, "pattern": <regex> }
            | { "op": "starts_with"|"ends_with", "column": <ident>, "value": <string> }

conditional ::= { "op": "case",
                  "when": [ { "cond": expr, "then": expr }, ... ],
                  "else": expr }
              | { "op": "when_then",
                  "cond": expr, "then": expr, "else": expr }    # binary form

datetime ::= { "op": "date_trunc", "unit": "day"|"month"|"quarter"|"year"|"hour"|"week_iso",
               "arg": expr }
           | { "op": "date_diff",  "unit": "seconds"|"minutes"|"hours"|"days"|"months"|"years",
               "end": expr, "start": expr }
           | { "op": "date_part",  "unit": "year"|"month"|"day"|"quarter"|"hour"|"weekday",
               "arg": expr }
           | { "op": "now" }
           | { "op": "strftime", "arg": expr, "format": <string> }
           | { "op": "strptime", "arg": expr, "format": <string> }

dtype ::= "Int8"|"Int16"|"Int32"|"Int64"|"Float32"|"Float64"|"String"|"Date"|"Datetime"|"Boolean"
ident ::= /[A-Za-z_][A-Za-z0-9_]*/
```

---

## 3. Type rules

The validator infers each node's result type and checks compatibility.

| Node | Operand types | Result |
|---|---|---|
| `col` | declared column dtype | column dtype |
| `lit` | inferred from scalar | scalar dtype |
| `param` | declared in workspace defaults | as declared |
| `not` | Boolean | Boolean |
| `neg` | numeric | same numeric |
| `abs` | numeric | same numeric |
| `sqrt`, `exp`, `log` | numeric | Float64 |
| `ceil`, `floor` | numeric | numeric (integer truncation) |
| `round` | numeric | numeric |
| `cast` | any | declared `dtype` |
| `and`, `or` | Boolean × N | Boolean |
| `add`, `sub`, `mul`, `div` | numeric × N | numeric (widest of operand types, Float64 on `div`) |
| `safe_div` | numeric, numeric | Float64 (returns 0.0 when denominator == 0) |
| `concat` | string × N | string |
| `least`, `greatest` | comparable × N | same |
| `coalesce` | any × N | first non-null type |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` (column-form) | column dtype, scalar | Boolean |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` (args-form) | comparable × 2 | Boolean |
| `in`, `not_in` | column dtype, list[scalar] | Boolean |
| `between` | numeric or Date column | Boolean |
| `is_null`, `not_null` | any column | Boolean |
| `matches`, `starts_with`, `ends_with` | string column, regex/string | Boolean |
| `case`, `when_then` | each `then` and `else` must agree | type of `then`/`else` |
| `date_trunc`, `date_part`, `strptime` | datetime/date | datetime / int / datetime |
| `date_diff` | datetime, datetime | Int64 |
| `now` | — | Datetime (UTC) |
| `strftime` | datetime, string format | String |

Type errors are reported with the AST path:

```text
ERROR  metrics.CTR.expression.den.args[1]: expected numeric, got String
```

---

## 4. Polars translation rules

Each AST node compiles to exactly one Polars expression. The full table:

| AST | Polars |
|---|---|
| `{col: x}` | `pl.col(x)` |
| `{lit: v}` | `pl.lit(v)` |
| `{param: p}` | `pl.lit(workspace.params[p])` |
| `{op: not, arg: a}` | `~ A` |
| `{op: neg, arg: a}` | `-A` |
| `{op: abs, arg: a}` | `A.abs()` |
| `{op: sqrt, arg: a}` | `A.sqrt()` |
| `{op: exp, arg: a}` | `A.exp()` |
| `{op: log, arg: a}` | `A.log()` (natural) |
| `{op: log, arg: a, base: 2}` | `A.log(base=2)` |
| `{op: ceil, arg: a}` | `A.ceil()` |
| `{op: floor, arg: a}` | `A.floor()` |
| `{op: round, arg: a, ndigits: n}` | `A.round(n)` |
| `{op: cast, arg: a, dtype: d}` | `A.cast(pl.<dtype>)` |
| `{op: and, args: [a,b,c]}` | `A & B & C` |
| `{op: or, args: [a,b,c]}` | `A | B | C` |
| `{op: add, args: [a,b]}` | `A + B` |
| `{op: sub, args: [a,b]}` | `A - B` |
| `{op: mul, args: [a,b]}` | `A * B` |
| `{op: div, args: [a,b]}` | `A / B` |
| `{op: safe_div, num: a, den: b}` | `pl.when(B == 0).then(0.0).otherwise(A / B)` |
| `{op: concat, args: [a,b], sep: "/"}` | `pl.concat_str([A, B], separator="/")` |
| `{op: least, args:[a,b]}` | `pl.min_horizontal(A, B)` |
| `{op: greatest, args:[a,b]}` | `pl.max_horizontal(A, B)` |
| `{op: coalesce, args:[a,b]}` | `pl.coalesce(A, B)` |
| `{op: eq, column: c, value: v}` | `pl.col(c) == pl.lit(v)` |
| `{op: in, column: c, values:[...]}` | `pl.col(c).is_in([...])` |
| `{op: not_in, column: c, values:[...]}` | `~ pl.col(c).is_in([...])` |
| `{op: between, column: c, low: lo, high: hi}` | `pl.col(c).is_between(lo, hi, closed="both")` |
| `{op: is_null, column: c}` | `pl.col(c).is_null()` |
| `{op: not_null, column: c}` | `pl.col(c).is_not_null()` |
| `{op: matches, column: c, pattern: p}` | `pl.col(c).str.contains(p, literal=False)` |
| `{op: starts_with, column: c, value: v}` | `pl.col(c).str.starts_with(v)` |
| `{op: ends_with, column: c, value: v}` | `pl.col(c).str.ends_with(v)` |
| `{op: when_then, cond: c, then: t, else: e}` | `pl.when(C).then(T).otherwise(E)` |
| `{op: case, when:[{cond, then},...], else: e}` | nested `pl.when(...).then(...)....otherwise(e)` |
| `{op: date_trunc, unit: "day", arg: a}` | `A.dt.truncate("1d")` |
| `{op: date_diff, unit: "seconds", end: e, start: s}` | `(E - S).dt.total_seconds()` |
| `{op: date_part, unit: "year", arg: a}` | `A.dt.year()` |
| `{op: strftime, arg: a, format: f}` | `A.dt.strftime(f)` |
| `{op: strptime, arg: a, format: f}` | `A.str.strptime(pl.Datetime, f)` |
| `{op: now}` | `pl.lit(datetime.utcnow())` |

---

## 5. Examples

### 5.1 Filter — IH "channel is not null and outcome is interesting"

```yaml
- kind: filter
  expression:
    op: and
    args:
      - {op: not_null, column: Channel}
      - {op: in, column: Outcome, values: [Impression, Clicked, Pending, Conversion]}
```

### 5.2 Derived column — `ResponseTime` in seconds

```yaml
- kind: derive_column
  output: ResponseTime
  expression:
    op: date_diff
    unit: seconds
    end:   {col: OutcomeTime}
    start: {col: DecisionTime}
```

### 5.3 Coalesce: ConversionEventID falls back to Name

```yaml
- kind: derive_column
  output: ConversionEventID
  expression:
    op: case
    when:
      - cond:  {op: ne, column: ConversionEventID, value: ""}
        then:  {col: ConversionEventID}
    else:      {col: Name}
```

### 5.4 Metric formula — CTR

```yaml
metrics:
  CTR:
    source: engagement
    kind: formula
    expression:
      op: safe_div
      num: {col: Positives}
      den: {op: add, args: [{col: Positives}, {col: Negatives}]}
```

### 5.5 Metric formula — StdErr

```yaml
metrics:
  StdErr:
    source: engagement
    kind: formula
    depends_on: [CTR]
    expression:
      op: sqrt
      arg:
        op: safe_div
        num: { op: mul, args: [
          {col: CTR},
          {op: sub, args: [{lit: 1.0}, {col: CTR}]}
        ]}
        den: { op: add, args: [{col: Positives}, {col: Negatives}] }
```

### 5.6 Snapshot state with `where`

```yaml
states:
  ChurnedSubs: {type: count, where: {op: eq, column: status, value: churned}}
```

The `where` predicate is itself an expression node — same grammar.

---

## 6. Validation rules

The `valuestream validate` command runs these checks before any data is touched.

1. **Schema check.** The AST conforms to JSON Schema in `schemas/expr.json`.
2. **Column-existence check.** Every `col` reference resolves to:
   - in source-level transforms: a column declared in the Source's `schema.columns` (or one produced by a previous transform in the list);
   - in processor-level filters: the post-transform schema of the bound Source;
   - in metric formulas: a state column produced by the Processor.
3. **Type check.** Per §3.
4. **Constant-folding sanity.** `safe_div(x, 0)` = 0; `log(0)` rejected at validate time when both operands are literals.
5. **Pure check.** Predicates and derived columns are pure functions of the row — no side effects. `now()` is allowed but flagged because it makes `config_hash` time-dependent (the engine snapshots `now()` once per run instead of evaluating per row).
6. **Dimension-leakage check.** A processor-level filter cannot depend on a column that's not in the source schema after transforms.

Errors carry the AST path and a human-readable explanation.

---

## 7. Canonical form (for hashing)

When computing `config_hash`, the YAML is canonicalized:

- keys sorted recursively,
- numeric scalars normalized (`1.0 == 1`),
- string scalars unquoted,
- redundant `op: when_then` chains rewritten as `op: case`.

Two YAML files that parse to the same canonical AST produce the same `config_hash`.

---

## 8. Error messages

A validation error includes:

```text
ERROR validating  metrics.CTR.expression
       at .num.col   = "Positives"      (OK — type Int64)
       at .den.op    = add
            .args[0].col = "Positives"  (OK — type Int64)
            .args[1].col = "negatives"  (FAIL — column not found in processor 'engagement' state)
```

The error is structured (machine-parseable) and shown the same way in CLI and Builder UI. The read-only HTTP API translates the same governed validation failures to 400 responses.

---

## 9. Forbidden constructs

- No user-defined functions.
- No loops, no recursion.
- No I/O (`open`, `http`, …).
- No reflection (`getattr`, `eval`, …).
- No imports.
- No string-literal-as-Python-expression patterns.

The DSL exists *because* the legacy app accepted Python expression strings via `eval`. Value Stream replaces every such occurrence with one of the operators above. The migration tool flags untranslatable expressions for hand-conversion and refuses to silently drop them.

---

## 10. Extensibility

If a real-world need demands an operator not in §2:

1. Add it to the BNF in this document.
2. Add an entry to the type-rules table in §3.
3. Add the Polars translation in §4.
4. Update `schemas/expr.json`.
5. Add a unit test under `tests/expr/test_<op>.py`.

Adding operators is cheap; adding escape hatches is forbidden.
