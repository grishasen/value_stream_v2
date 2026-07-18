# Configuration Editor — QA report and fix backlog

> **Historical evidence — implementation backlog superseded.** This page
> preserves the complete 2026-07-18 Configuration Builder QA record and its
> original A1–F3 remediation proposals. Do not track implementation status or
> create parallel delivery work from the story labels on this page. The
> [Configuration Studios remediation backlog](ai-studio-remediation-backlog.md)
> is the single implementation backlog and contains the maintained Builder
> finding-to-story mapping, priorities, dependencies, and release gates.

Exploratory QA pass over every screen of the Configuration Builder (plus the
surrounding pages), exercising create / modify / delete in each section and
authoring four new catalog objects end-to-end, followed by a prioritized
remediation backlog.

- **Date:** 2026-07-18
- **Build:** branch `codex/config-authoring-trust-hotfix` @ `6b890d9`
- **Workspace:** `examples/test_config_editor` ("fat", 1.92 B rows kept, 799 chunks)
- **Coverage:** Home, Build landing, Configuration Builder (all 9 steps),
  Reports, Chat With Data, Catalog, Data Load, Pipelines/Ops, AI Configuration
  Studio (entry screen). Not exercised: live LLM chat / AI-draft generation,
  data runs (211 GB source), file uploads.

---

## Part 1 · QA report

### Verdict at a glance

| Check | Result |
| --- | --- |
| End-to-end walkthrough (Health → … → Export) | **Achievable** ✓ |
| All 9 wizard steps | Render and function |
| Bugs filed | **7** (2 high) |
| New objects created via UI | 2 metrics · 2 tiles · 1 processor |

The guided flow completes, applies transactionally to the YAML catalog, and a
newly created metric computed real values on the live report. Both
high-severity findings concern the draft/apply state machine: one silently
blocks a whole feature (calculated-field transforms), the other traps users in
the guided flow.

### Bug index

| ID | Severity | Summary |
| --- | --- | --- |
| BUG-1 | **High** | New calculated-field transform can never be applied — draft never goes dirty |
| BUG-2 | **High** | "Create New Processor" traps the guided flow; phantom drafts after every apply |
| BUG-3 | Medium | Tile "Delete" targets the library selection, not the tile open in the editor |
| BUG-4 | Medium | Default report time window is outside data coverage — every KPI shows n/a |
| BUG-5 | Low | Apply toast and auto-IDs expose mangled hash identifiers |
| BUG-6 | Low | Home "Workspace Flow" copy predates the Build section (stale) |
| BUG-7 | Low | Streamlit widget-policy warning on Metrics step (log noise) |
| OBS-1 | Env note | Frozen partial renders in the embedded test browser (server completed fine) |

File references are to `src/valuestream/ui/pages/config_builder.py` unless noted.

#### BUG-1 (High · blocks feature) — Calculated field can never be applied

- **Repro:** Configuration Builder → Sources → Calculated Fields → add a row
  (Name `QAHighValueFlag`, Mode `AST YAML`, valid expression, Enabled ✓).
  The "Generated calculated transforms" preview shows the new `derive_column`.
- **Expected:** draft goes dirty; **Apply to workspace** replaces Continue;
  transform lands in the source YAML.
- **Actual:** primary action stays **Continue**, no draft banner, and
  "Generated source transforms / YAML" never includes the field — even after a
  full app rerun. The work is visible in the preview but permanently
  unappliable.
- **Suspected cause:** `_render_calculated_rows_editor` (line 933) is a nested
  `@st.fragment` inside the `_source_builder` fragment (line 1101). Its grid
  edits update `st.session_state` and its own preview, but the outer fragment's
  `source_def` / draft status don't recompute, and the equivalence gate in
  `_build_source_definition` (line 5378: `if not equivalent.dirty: return
  source_to_dict(source)`) returns the unmodified source. Standalone, the
  projection logic flags the change correctly — the loss is in fragment/state
  ordering.

#### BUG-2 (High · flow trap) — Create mode blocks Continue; phantom drafts

- **Repro:** Processors step → switch to **Create New Processor**. Before
  typing anything a draft banner appears ("Editing draft … Ready to apply") and
  **Apply to workspace** replaces Continue. **Discard draft** instantly
  recreates the draft. Clicking **Continue** right after a discard is swallowed
  by the rerun (step does not advance).
- **Expected:** an untouched template is not a draft; Continue stays available;
  Discard actually clears.
- **Actual:** the only escape is switching back to "Edit Existing Processor" —
  nothing hints at this. The same post-apply pattern recurs on Metrics and
  Reports/Tiles: immediately after a successful apply a fresh "Editing draft ·
  Ready to apply" banner appears. The step banner accumulated **"4 unapplied
  drafts preserved"** during one session — all phantoms.
- **Impact:** users can apply a half-configured template by accident, lose
  trust in the draft counter, and get stuck in the guided order the UI itself
  recommends.

#### BUG-3 (Medium · data-loss hazard) — Delete targets the wrong tile

- **Repro:** Reports/Tiles → **New** → build a tile → **Apply** (tile now open
  in the Tile Editor) → click **Delete** (in the New / Duplicate / Delete row).
- **Actual:** "Tile **'Click-through rate'** is staged for deletion" — the tile
  selected in the Report library selectbox far above, not the one in the
  editor. Staging + explicit apply prevented data loss (good design), but the
  target ambiguity invites deleting the wrong report tile.

#### BUG-4 (Medium · first-run experience) — Default window shows all-n/a KPIs

- **Repro:** open Reports → Executive overview on a workspace whose data ends
  2024-09 (today is 2026-07).
- **Actual:** every KPI card shows **n/a** despite 1.92 B ingested rows; the
  caption even prints "latest 2024-09" while filtering Apr–Jul 2026. Switching
  to All time instantly fills every tile (CTR 2.40%, Revenue $1.79 M, …).
- **Suggestion:** clamp relative presets to the data's max date (or default to
  the latest period with data). This will hit every demo and historical
  dataset.

#### BUG-5 (Low · polish) — Mangled identifiers in toast and auto-IDs

Creating "QA Avg Revenue Per Conversion" from scratch auto-assigns the
uneditable ID `qa_avg_revenue_per_c_9d799c360996180c` (truncated + hash), and
the success toast reads "Metric 'Qa avg revenue per c 9d799c360996180c'
applied" instead of the display name. The recipe path lets you edit the Metric
ID; the from-scratch path doesn't.

#### BUG-6 (Low · stale copy) — Home "Workspace Flow" predates Build

The panel lists Reports / Chat / Data Integration / **Settings — "Review
catalog, config builders, and AI-assisted drafts"**. With authoring v2 on (the
default), the builders live under **Build**, which the panel never mentions.

#### BUG-7 (Low · log noise) — Widget-policy warning on Metrics step

`The widget with key "builder_metric_mode" was created with a default value but
also had its value set via the Session State API` — logged with a full stack on
step entry (`config_builder.py:2601`). Benign but noisy.

#### OBS-1 (Environment note) — Frozen renders in the embedded test browser

Twice, navigating Step 1 → Sources left the page stuck "running" with greyed
stale content. Server-side diagnosis (debug logs + faulthandler stack dumps)
showed each script run completed in ~200 ms; the client simply stopped applying
websocket updates. Not reproducible as an app bug — but worth keeping in mind
if users on slow clients report "hangs" on the Sources step, which re-reads
source samples (`discover` + `read`, twice per render) synchronously.

### Main usability concerns

Ordered by how much they would slow down a real configuration author.

1. **The draft/apply state machine is the product's core interaction, and it's
   the least trustworthy part.** Phantom drafts after applies and mode switches
   (BUG-2), a "N unapplied drafts" counter that counts junk, and the primary
   action silently flipping between Continue and Apply make users second-guess
   whether their change is saved, pending, or lost.
2. **Selector labels hide identity.** Source and Processor dropdowns render
   full description sentences ("Click engagement, reach, and default-banner
   test/control state across…") with the ID nowhere visible, while the Metric
   picker uses clean IDs. With two sources configured (`ih`, `holdings`) you
   cannot tell which one you're editing without opening it.
3. **Asymmetric CRUD.** Sources: edit/delete but no create (create lives only
   in the "Start from a sample" path). Processors: create/edit but no delete.
   Tiles: full CRUD. Users must discover a different mental model per section.
4. **Developer internals surface in a business tool.** Raw `st.json` blobs in
   Dimension Packs and One-Click Promotion, snake_case chart IDs (`bar_polar`,
   `experiment_odds_ratio`) in the tile Chart picker next to the friendly
   purpose-grouped library, and raw pydantic validation errors
   ("union_tag_not_found", "extra_forbidden") for a wrong calculated-field
   expression.
5. **The Calculated Fields grid is hostile to its main job.** Multi-line YAML
   expressions edited in a single-line cell, the Enabled checkbox hidden behind
   horizontal scroll (and a new row arrives as `None` → silently excluded),
   Escape cancels an edit without warning.
6. **Session-only wizard state.** A page reload silently returns to Step 1 and
   drops in-progress (unapplied) work; combined with concern 1 this makes long
   authoring sessions risky.
7. **Identity mismatches.** Sidebar says "Workspace · fat" for a directory
   named `test_config_editor`; the new-processor template defaults Subject
   Entity Field to `SubjectID`, a column that doesn't exist in the source.

### What works well

- **Validation-first applies.** Every apply runs through a validated catalog
  transaction; failures never half-write. Config versions are journaled
  (`meta/config_versions.duckdb`).
- **Generated-YAML transparency.** Every editor shows the exact YAML it will
  merge under "Technical details" — excellent for review and trust.
- **The KPI recipe library.** Business definition, calculation, accuracy notes,
  processor-state bindings with a "Ready to install" check, and a one-toggle
  report tile. The strongest screen in the product.
- **Dimension Profiler.** Profiles 42 source fields with cardinality / null% /
  safe-for-group-by recommendations, plus one-click promotion with growth
  preview.
- **Live tile preview** renders against real aggregates before you commit; the
  applied tile then computed identically on the Reports page.
- **Change-aware guidance.** The Export step explicitly says which processor
  changed and that a data run for `ih` is needed; applying config never
  triggers processing.

### Changes made to the workspace during testing

All applied through the UI and verified on disk (catalog files are
git-ignored).

| Object | Action | Landed in | Status |
| --- | --- | --- | --- |
| `ih` source description | Modified | `pipelines.yaml` | Applied ✓ |
| `qa_channel_reach` processor (binary outcome, group-by Channel) | Created | `processors.yaml` | Applied ✓ (needs data run) |
| `QA_Engagement_Rate` metric (from recipe) + KPI-strip tile | Created | `metrics.yaml` + `dashboards.yaml` | Applied ✓ · computes |
| `qa_avg_revenue_per_c_…` metric (from scratch, Revenue ÷ Positives) | Created | `metrics.yaml` | Applied ✓ |
| "QA Engagement Rate Trend" line tile (by Channel) | Created | `dashboards.yaml` · Executive overview | Applied ✓ · renders live data |
| "Click-through rate" tile | Delete staged, then discarded | — | Intact (delete flow verified) |
| `QAHighValueFlag` calculated field | Create attempted | never reached YAML | **Blocked by BUG-1** |

---

## Part 2 · Fix backlog

21 items in 6 epics, sized and sequenced into three milestones. Priorities:
**P0** ship-blocking / trust-breaking · **P1** high friction · **P2** quality &
consistency · **P3** polish/debt. Sizes: **S** < ½ day · **M** ½–2 days ·
**L** 2–5 days. Paths are relative to `src/valuestream/ui/`.

### Milestone plan

| Milestone | Goal | Items | Est. |
| --- | --- | --- | --- |
| **M1 — Trust the draft** | The draft/apply state machine never lies; the two feature-blocking bugs are gone. | A1 A2 A3 A4 B1 E4 F3 | ~7–9 dev-days |
| **M2 — Identity & symmetry** | Users always know which object they're editing and can create/delete every object type. | B2 B3 C1 C2 C3 E1 E2 E3 | ~6–8 dev-days |
| **M3 — Ergonomics & debt** | Expression authoring is pleasant; internals stop leaking; renders stop re-reading source data. | D1 D2 D3 D4 D5 D6 F1 F2 | ~7–10 dev-days |

Suggested order within M1: **F3 first** (write the failing tests), then
A1 → A2 → A3; B1 is isolated in the reports page and can run in parallel.
Risk items: A2 (state-machine redesign touches all 9 steps) and D1 (new editor
surface).

### Epic A — Draft/apply state machine

Root theme: fragment-scoped editors and hash-based dirty checks disagree about
what the user changed.

#### A1 · P0 · M — Calculated-field edits must dirty the source draft *(fixes BUG-1)*

- **Where:** `pages/config_builder.py:933` (`_render_calculated_rows_editor`,
  nested fragment), `:1101` (`_source_builder`), `:5378-5457`
  (`_build_source_definition` equivalence gate); same pattern in the
  defaults/filter row editors (`:830-993`).
- **Approach:** remove `@st.fragment` from the row editors so a grid edit
  reruns `_source_builder` (measure perf; F1 removes the expensive re-read that
  motivated the fragment), **or** keep the fragment and have it recompute draft
  status and re-render the shared `save_slot`/`draft_slot` after writing
  `calc_key`. Audit every editor that writes state consumed outside its
  fragment.
- **Accept:** add row → Apply appears within one rerun and the applied
  `pipelines.yaml` contains the transform; edit/delete of existing rows also
  dirty the draft; same verified for Default Values and Source Filter editors;
  covered by AppTest regression (F3).

#### A2 · P0 · L — Untouched editors are never drafts *(fixes BUG-2)*

- **Where:** `pages/config_builder.py:425-486`
  (`_render_editor_primary_action`), `:379-399` (`_render_continue_primary`),
  create-mode template init in `_processor_builder` / `_metric_builder` / tile
  editor; `builder.py:120-160` (draft status/registry).
- **Approach:**
    - (a, b) In create mode, baseline = the freshly initialized template (not
      the absent object); register a draft only when the canonical draft hash
      differs from the template hash. Discard resets to that clean baseline.
    - (c) Give the primary action one stable key (e.g.
      `builder_primary_{step}`) and switch only its label/handler, or move step
      advance into `on_click` so the click survives label swaps.
    - (d) After a successful apply, clear the editor's widget-prefix state and
      re-baseline against the just-applied object; purge registry entries whose
      diff is empty.
- **Accept:** entering create mode shows Continue and no draft banner until a
  field is edited; Discard returns to clean state in one click and stays clean;
  Continue always advances on first click in every mode; post-apply state is
  clean on all steps; the drafts counter only counts real edits.

#### A3 · P0 · S — Tile Delete names and targets the right tile *(fixes BUG-3)*

- **Where:** Reports/Tiles step, library action row (tile builder ~`:3550+`);
  `BUILDER_PENDING_TILE_DELETE_KEY`.
- **Approach:** label the button with its target ("Delete '<title>'"), disable
  when nothing is selected, and either move it into the Tile Editor card or
  make the staged-deletion notice name dashboard/page/tile explicitly. Keep the
  stage-then-apply design.
- **Accept:** button text always contains the exact tile title it will stage;
  deleting while a different tile is open in the editor is impossible or
  unmistakably labelled.

#### A4 · P1 · M — Keep Continue visible when a draft exists

- **Approach:** render both actions when dirty: **Apply to workspace**
  (primary) + **Continue without applying** (secondary; the draft registry
  already preserves drafts across steps). One shared component for all 9 steps.
- **Accept:** users can always advance; drafts survive navigation and restore
  cleanly; tab order puts Apply first.

### Epic B — Report defaults & object identity

#### B1 · P0 · M — Clamp report time presets to data coverage *(fixes BUG-4)*

- **Where:** `pages/reports.py` time-filter resolution; `freshness.py` already
  exposes the latest data date.
- **Approach:** anchor relative presets to `min(today, latest_data_date)`;
  when clamping occurs show "Showing latest available data (through 2024-09)".
  Apply to builder tile previews too.
- **Accept:** fresh open of Executive overview on this workspace shows computed
  KPIs; explicit user-chosen absolute ranges are never clamped.

#### B2 · P1 · S — Editable metric IDs from scratch; toasts use display names *(fixes BUG-5)*

- **Approach:** reuse the recipe path's ID field; slugify the display name
  without a hash suffix unless there's a collision; toast pulls
  `display.label`.
- **Accept:** creating "QA Avg Revenue Per Conversion" proposes
  `qa_avg_revenue_per_conversion`, editable before apply; toast reads "Metric
  'QA Avg Revenue Per Conversion' applied."

#### B3 · P1 · S — Show IDs first in Source and Processor selectors

- **Where:** `_source_choice_label` (`:5092`), `_processor_choice_label_human`.
- **Approach:** format as `id — first sentence of description · kind`,
  truncated; one shared `format_func` for all entity selectboxes.
- **Accept:** every entity selector shows the ID within the first ~20
  characters.

### Epic C — CRUD symmetry

Today: sources = edit/delete, processors = create/edit, tiles = full CRUD,
metrics = create/edit.

#### C1 · P2 · M — "New source" entry point in the manual builder

- **Approach:** minimum — an "Add source" action on the Sources step that hands
  off to the sample flow with a return path. Better — a blank-source template
  mirroring the processor create mode (respecting A2's touched-state rules).
- **Accept:** a user on the Sources step reaches source creation in one click
  and returns to the builder afterwards.

#### C2 · P2 · M — Processor delete with dependency preview

- **Approach:** mirror the existing `_delete_source_dialog`: list dependent
  metrics/tiles, stage the deletion, apply transactionally, and state the
  aggregate-cleanup consequence (orphaned aggregate folders until vacuum).
- **Accept:** deleting a processor with dependents requires explicit
  confirmation listing them; catalog validates clean after apply.

#### C3 · P2 · S — Metric delete with tile-dependency check

- **Approach:** delete action in "Edit Existing Metric" mode; block or cascade
  when report tiles reference the metric (offer "also remove N tiles").
- **Accept:** deleting a referenced metric either cascades explicitly or is
  blocked with the list of tiles.

### Epic D — Editor ergonomics & presentation

#### D1 · P2 · L — Proper expression editor for calculated fields

- **Problem:** multi-line YAML AST edited in a single-line grid cell; Escape
  silently cancels; the schema (`cond:` as op/column/value, `else:` not
  `otherwise:`) is undiscoverable.
- **Approach:** "Edit expression" per row opens a panel/dialog: multi-line
  textarea, live validation, the Examples snippets inline, mode-specific hints
  (AST YAML vs Polars). Grid stays as overview.
- **Accept:** a valid multi-line expression can be authored without flow-YAML
  tricks; invalid YAML shows the friendly error (D3) next to the input.

#### D2 · P1 · S — Grid-added rows default to Enabled and never vanish silently

- **Problem:** `blank_calculated_row()` defaults `Enabled: True`, but rows
  added via the grid's trailing row arrive through Streamlit's `added_rows`
  with `Enabled=None` → silently excluded; the checkbox sits behind horizontal
  scroll.
- **Approach:** `st.column_config.CheckboxColumn("Enabled", default=True)` on
  all row editors; coerce `None`/`""` → `True` in
  `build_derive_column_transforms` (`builder.py:739`) and peers; render
  disabled rows with an "excluded" visual.
- **Accept:** a freshly added row with name + expression compiles without
  touching Enabled.

#### D3 · P2 · M — Translate validation errors for humans

- **Approach:** map common pydantic error codes to plain sentences with the
  offending path ("`cond` must be a condition like `{op: gt, column: Revenue,
  value: 100}`; use `else:` rather than `otherwise:`"); keep the raw error
  behind "Technical details".
- **Accept:** the three errors from the QA repro each render a one-line human
  message.

#### D4 · P2 · S — Replace raw JSON blobs on the Dimensions step

- **Approach:** chips for field lists (available / selected / missing), small
  key-value layout for the promotion preview (recommendation, cardinality,
  null%).
- **Accept:** no `st.json` remains on the Dimensions step.

#### D5 · P2 · S — Friendly chart names in the tile Chart picker

- **Approach:** reuse `REPORT_LIBRARY_CHART_LABELS` (`config_builder.py:241`)
  extended to all kinds as `format_func`; show the raw id as caption.
- **Accept:** every chart kind shows a label matching the library vocabulary.

#### D6 · P2 · M — Survive a page reload

- **Approach:** mirror `builder_step` into `st.query_params`; optionally
  persist the draft registry to `meta/` keyed by catalog hash so a reload can
  offer "Restore draft" (`_render_registered_draft` already exists).
- **Accept:** reload lands on the same step; a registered draft survives reload
  and is offered for restore.

### Epic E — Copy & consistency

#### E1 · P2 · S — Update Home "Workspace Flow" for the Build section *(fixes BUG-6)*

- **Approach:** in `pages/home.py`, branch the panel on
  `authoring_v2_enabled()`: list Build ("Guided authoring: Configuration
  Builder and AI Studio") and drop the builder mention from Settings.
- **Accept:** panel matches the sidebar sections in both flag states.

#### E2 · P3 · S — Disambiguate workspace label from directory

- **Approach:** show catalog name + directory basename ("fat ·
  test_config_editor") in `shell.py:72`; the details popover already has the
  full path.
- **Accept:** two workspaces with the same catalog name are distinguishable at
  a glance.

#### E3 · P2 · S — New-processor template defaults to real source fields

- **Approach:** in `_new_processor_template` (`config_builder.py:5331`),
  default Subject Entity Field to the first natural-key field (`CustomerID`
  here) or leave empty-required; review Outcome Column and Positive/Negative
  values against sampled data.
- **Accept:** template defaults validate against the selected source without
  edits, or are explicitly empty.

#### E4 · P3 · S — Silence the `builder_metric_mode` warning *(fixes BUG-7)*

- **Approach:** at `config_builder.py:2601`, stop passing `default=` when the
  key is pre-seeded in session state (seed-only pattern used elsewhere in the
  file).
- **Accept:** no policy warning in logs when entering the Metrics step.

### Epic F — Performance, robustness, test coverage

#### F1 · P1 · M — Stop re-reading source samples twice per Sources render

- **Problem:** each Sources render calls `discover()` + `read()` +
  `collect_schema()` twice (via `_source_rename_mapping` and
  `_source_field_options`, `config_builder.py:1143-1144`) — ~90 ms on local SSD
  against a 211 GB tree, but synchronous and unbounded on network storage. Also
  the enabler for un-fragmenting the row editors (A1).
- **Approach:** cache `_source_sample_columns` with `st.cache_data` keyed on
  (workspace, source id, reader config hash); compute field options once per
  render and pass down.
- **Accept:** at most one discover/read per source per session until reader
  config changes; timer logs confirm.

#### F2 · P3 · S — Progress feedback while the Sources step loads samples

- **Approach:** wrap sample inspection in `st.spinner("Reading source
  sample…")`; surface read failures as a visible caption instead of the silent
  `except Exception: return []` in `_source_sample_columns`.
- **Accept:** sample-read state is visible; a failure names the path it tried.

#### F3 · P0 · M — AppTest regression suite for the draft lifecycle

- **Problem:** none of BUG-1/2/3 is caught by the existing suite; all are
  state-machine regressions that `streamlit.testing.v1.AppTest` can drive
  headlessly.
- **Approach:** add `tests/ui/test_builder_drafts.py` covering: calculated-row
  add → apply (A1); create-mode enter/discard/continue (A2); post-apply
  cleanliness on Sources/Processors/Metrics/Tiles; tile delete targeting (A3);
  one full 9-step walkthrough on a fixture workspace. Ship in M1 alongside the
  fixes.
- **Accept:** suite fails on today's code for BUG-1/2/3, passes after A1–A3;
  runs in CI without source data.

---

*Estimates assume one engineer familiar with the Streamlit codebase. Source
artifacts: QA report and backlog pages published 2026-07-18 from the test
session against `examples/test_config_editor`.*
