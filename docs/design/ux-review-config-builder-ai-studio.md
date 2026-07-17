# UX / Conversion Audit — Configuration Builder & AI Configuration Studio

**Date:** 2026-07-17 · **Build reviewed:** local app on `:8501`, workspace `fat`
**Method:** Two passes. Pass 1 — pixel-level design critique. Pass 2 — cold first-time user walkthrough (uploaded a 300-row Pega-style IH CSV and ran the full AI draft → accept → repair loop).
**Framing:** If this app had a "Start free trial" button, the findings below are ordered by how fast each one kills that click.

---

## Pass 1 summary — the designer's verdict

The bones are better than most AI-era tools: a governed patch-review model (AI never writes directly, everything is a reviewable diff), a deterministic no-LLM fallback, honest AI-data-sharing counters, and a genuinely good purpose-based report library. That's the pitch. Almost everything wrapped around it fights it.

The tell-tale "vibe-coded" signatures are all present: raw YAML dumps as primary UI furniture, `st.json` blobs shown to business users, stat-tile components that truncate their own labels to single letters, three stacked navigation systems, phantom empty rows in editable tables, a `?` help icon glued to every single label, and copy written by the engine for the engineer ("This patch selection is not internally consistent. Adjust the cards or accept it for repair before publishing.").

## Pass 2 summary — the first-time user's story

- Landed on **AI Configuration Studio**: a blank page with one line telling me to look at the sidebar. Nothing to click in the main area. I almost left here.
- Found **Upload** buried under nav sections in the sidebar. Uploaded a CSV. The page exploded into 4 stage radios × 14 numbered steps × a chip progress strip — and told me "Field Approval ✓ · 20 approved fields" before I approved anything.
- Clicked **Generate AI Draft**. A tooltip said "Configure a LiteLLM model in the sidebar to enable AI generation" *while the generation was already running*. 30+ seconds of spinner, no cancel, no progress.
- It worked — and dropped **47 separate "Accept … patch" checkboxes** into a 320px right rail, all pre-checked, each hiding a Before/After YAML diff behind an expander. I unchecked one and was told my selection was "not internally consistent."
- Clicked **Accept patches** → Review stage → **"Needs attention · 6 issues."** Clicked **Generate AI Repair** → another silent wait → new patch cards, and now the rail says **2 issues** while the main panel still says **6**. I no longer know what's true. This is where I close the tab.
- **Configuration Builder** pass: 9 wrapping tabs inside a tab, a Save button that flips between enabled / disabled / "Saved" as I merely switch tabs, and a **Save & Export** tab that is **54,860 px tall** because it renders the entire catalog YAML inline — the download button is at the very bottom.

---

# Findings

Tags: `[Builder]` Configuration Builder · `[Studio]` AI Configuration Studio · `[Both]` · (D) design pass · (U) user pass

## 🔴 Critical — kills trust or ends the session

1. **[Studio] (U) The patch review is 47 checkboxes in a sidebar.** One AI draft produced 47 independent "Accept processors/metrics/reports patch" cards squeezed into the ~320px right rail, each with its own expander of Before/After YAML. Nobody reviews 47 YAML diffs in a rail; everyone will blind-accept, which defeats the entire "governed review" value prop. Group patches by object (1 card = 1 processor/metric/dashboard), review in the main column with a real diff view, and offer *Accept all / Accept group*.

2. **[Studio] (U) The happy path ends in an error loop.** Generate → auto-accepted patches → "Needs attention · **6 issues**" → Generate AI Repair → new patches → still **2 issues** (and see #4). Each round is a 30–90 s blocking spinner. A first-run user never reaches a clean publish. The deterministic baseline is valid from the start — ship *that* as the guaranteed success path and treat AI enrichment as optional, pre-validated (auto-repair internally before ever showing patches to a human).

3. **[Studio] (U) Raw validator internals shown to end users.** The review rail prints Pydantic-style errors verbatim: `Time_Var.type: Input should be 'count', 'value_sum', 'min', 'max', 'pooled_mean', 'pooled_variance', 'tdigest', 'kll', 'cpc', 'hll', 'theta' or 'topk'` and `sources[ih].transforms[1].expression: column 'pxOutcomeTime' not found in schema`. Translate to human ("The AI referenced a column that doesn't exist — Repair will fix this") and keep the raw error behind a details expander.

4. **[Both] (U) Two sources of truth for system state, on screen at the same time.**
   - Main panel: "Draft Validation — Needs attention · 6". Right rail, same moment: "Draft Validation — Issues · 2".
   - "Generate AI Draft" shows a gating tooltip ("Configure a LiteLLM model in the sidebar to enable AI generation") while generation is actively succeeding.
   - Progress strip claims "Field Approval ✓ · 20 approved fields" before the user has ever seen the approval step.
   Every one of these teaches the user the UI lies. One state store, one validation count, chips that only turn ✓ on real user action.

5. **[Builder] (D) Save & Export is a 54,860-pixel page.** The `metrics.yaml` expander renders the full 91-metric catalog inline, so the primary actions (`Download dashboards.yaml` etc.) sit below ~50k px of YAML. Collapse all file expanders by default, pin a sticky action bar (Save · Validate · Download all), and virtualize/inner-scroll any YAML view.

6. **[Builder] (D/U) Save state is untrustworthy.** The Save button flips enabled/disabled as you merely switch tabs, renders as a ghost "Saved" on one tab, and exists twice in the DOM. Meanwhile the Settings tab footnote admits "this editor writes the active catalog directly" — the most destructive surface in the app has the least confirmation. One global sticky Save with explicit dirty-state ("Unsaved changes · 3 sections"), and a confirm step for anything that rewrites the active catalog.

7. **[Studio] (U) Dead-on-arrival empty state.** First visit = blank main area + one info bar pointing at a sidebar whose uploader is below the nav fold, types truncated ("CSV, PARQUET, JS…"). No drop zone, no "Try with sample data", no 3-step explainer of what the Studio will do. This is the top of the funnel and it converts nobody. Put a drag-and-drop zone with a bundled demo dataset front and center.

8. **[Studio] (D) Privacy defaults are backwards.** "Approve Fields And Data Sharing With AI" pre-checks **every** field for both *Approve* and *Share Sample Values* — including `CustomerID` — meaning real sample values go into LLM prompts by default. Enterprise security review kills the deal right here. Default sample-sharing to off (or to non-identifier fields only), and flag identifier-looking columns.

## 🟠 High impact — creates confusion, erodes momentum

9. **[Builder] (D) Three stacked navigation systems.** Page tabs (Builder / README / Report Inventory) → 9 "Builder step" pills that wrap onto two rows ("Save & Export" orphaned on row 2) → per-tab sub-modes (Create/Edit, Rules/Raw AST, Visual/Raw YAML). The step strip implies a wizard but has no sequence, no next/prev, no completion state. Pick one: either a real wizard with progress, or a flat sidebar of named sections.

10. **[Studio] (D) Numbering 14 steps across 4 disconnected stage radios.** Steps 1–6 live under "Data", 7 under "Draft", 8–11 under "Review", 12–14 under "Publish"; the numbering promises linearity the controls don't deliver. Review/Publish stages are dead ends showing "Generate and accept a draft first." with **no link or button** to the place where you'd do that. Every dead end must carry its own CTA.

11. **[Builder] (D) The Sources tab is a ~10,500 px single form** ending in three overlapping YAML dumps ("Compiled AST", "Generated calculated transforms", "Generated Source Transforms") that repeat the same `derive_column` content 2–3×. The YAML previews should be one collapsed expander, not the page's dominant content.

12. **[Both] (D) Machine IDs leak into user-facing surfaces.** Report Inventory lists `unique_customers_baa429fc52799795`, `clicks___impressions_9ee34eddfa98322`, "Click-through rate Copy"; Dimensions previews `explore_sketch_engagement_issue_customerid_20260717191403_topk`. Also `st.json` blobs used as UI ("Available pack fields", "Recommendation: Avoid"). Show human titles; keep ids in tooltips/details.

13. **[Studio] (D) Stat-tile truncation makes cards unreadable.** Deterministic Baseline renders its five tiles as "S.. 1 · P.. 1 · M. 2 · D.. 1 · T.. 2"; header tiles truncate "Fields With Exa…" and "Prompt Size ~6,0…"; the rail truncates "Needs attention" to "Nee…". If a stat card can't fit its label at its minimum width, the component is wrong — use a compact key-value list in narrow columns.

14. **[Studio] (U) Long AI operations are uncancellable black boxes.** Up to `Timeout Seconds` = 90 with a single static line ("Sending approved schema and baseline catalog to the model..."), no cancel, no elapsed time, whole app blocked by the rerun. Add cancel + elapsed/streamed status, and run generation without freezing navigation.

15. **[Studio] (U) The Copilot is locked exactly when it's needed.** With patches pending: "Accept or discard the pending patches before sending another message." The moment a user most wants to ask "what does this patch do?" the assistant refuses. Let chat answer questions read-only while patches are pending.

16. **[Builder] (D) One-Click Dimension Promotion defaults to a field flagged "Avoid".** The dropdown preselects `ABgroup` whose own recommendation JSON says `"Avoid" — Fewer than 3 distinct values`. Default to the top *Recommended* field; never lead with a footgun.

17. **[Both] (D) Help-icon confetti.** Every label ships a `?` button — the Studio sidebar alone shows ~15 identical circles; the Builder header pills each get one too. The icons outnumber the insights. Reserve inline help for genuinely ambiguous fields; move the rest to a docs link per section. Also: the "Configure a LiteLLM model…" tooltip got stuck open over the schema table (stale-tooltip bug).

18. **[Studio] (D) Developer internals in the sidebar.** "Loaded defaults from `/Users/gregory/PycharmProjects/value_stream_public/examples/fat/ai.yaml`" (absolute local path), model examples "`openai/gpt-4.1-mini`, `anthropic/claude-sonnet-4-5`, `ollama/llama3.1` … or a model served by a LiteLLM proxy", free-text model field defaulting to `gpt-5.6`. Fine for you; alien to a business user. Provider dropdown + curated model list, path behind an "Advanced" expander.

19. **[Builder] (D) The README top-tab is the repo README** — `uv sync --all-extras`, mkdocs commands, CLI reference — rendered inside a settings page for end users. Wrong audience, wrong place. Link to docs; don't embed the contributor guide.

20. **[Both] (D) Editable tables render phantom rows.** Default Values and Filters tables show a trailing empty row with a pre-ticked "Enabled" checkbox, and the filter placeholder row reads `None ==` like real data. Empty state should say "No filters yet — add one" instead of exposing the editor's scaffolding.

21. **[Builder] (D) Truncated columns with no recourse.** Derived-fields table cuts "CustomerT…" at ~90px next to an "Expression" column also clipped (`op: when_then cond:`); Chat Review's Group By column clips its list. No wrap, no expand-on-hover. Give key columns wrap or row-expansion.

22. **[Builder] (U) Report Inventory has no search or filters** — a flat multi-hundred-row table of every tile, including duplicates, with no way to find anything. The Builder's own Report library tab proves the team can do better.

## 🟢 Nice to have — polish that separates "tool" from "product"

23. **[Both] (D) Two competing type systems.** Serif display (page titles, "AI Copilot", "Studio Controls") vs heavy sans (section headings "Catalog Health", "Metric Workflow") with no consistent hierarchy rule; casing drifts between Title Case and sentence case. Pick one display family and one casing convention.

24. **[Both] (D) Monochrome sage palette flattens hierarchy.** Selected tabs, primary buttons, chips, and toggles are all the same green family; errors are a muted pink block. The single accent means nothing stands out — reserve the strongest color for the one primary action per screen.

25. **[Builder] (D) Hero real estate spent on trivia.** Workspace Health's top row is four giant cards for "Sources 2 / Processors 8 / Metrics 91 / Dashboards 1"; the Studio dedicates a stat card to "AI Available: Yes". Compress to a one-line summary; save cards for numbers people act on.

26. **[Both] (D) Catalog Health is rendered three times** (Workspace Health, Chat Review, Save & Export) with the same OK/0/0 values — repetition without new information.

27. **[Studio] (D) Chip progress strip wraps badly** ("Export" alone on row 2) and chips look clickable but aren't. Make them links to their step or style them as plain status.

28. **[Both] (D) Buttons of equal visual weight for unequal actions.** "Use Deterministic Draft" (instant, free, always works) is a ghost button next to primary "Generate AI Draft" (slow, can fail). Consider leading with the safe path, or a single split-button.

29. **[Both] (D) Misc copy**: workspace displayed as "Workspace · fat"; "Guided AI draft" hourglass badge in the header never visibly changes; tile description language is engine-speak ("Deduplicated clicks as a share of classified engagement outcomes").

30. **[Both] (D) Browser tab shows "Streamlit" during navigation** and the default favicon — small, but it's the first pixel a visitor sees.

31. **[Studio] (D) Layout imbalance at depth**: deep in the Draft stage the main column runs out of content while the rail continues for thousands of pixels — a two-column layout where only one column has anything to say.

32. **[Both] (A11y) Diff review relies on color and expanders alone**; accept controls are small checkboxes; truncated labels defeat screen readers ("S..", "Nee…"). Provide text labels and real diff semantics.

---

## What's genuinely good (keep and amplify)

- **Patch-based AI governance** — AI proposes, human disposes, deterministic fallback always exists. This is the differentiator; make the review experience worthy of it.
- **AI transparency counters** (Fields Sent / Hidden / With Examples, Prompt Size, AI Sharing Details) — rare and valuable; fix the truncation and make this a headline feature.
- **Purpose-based Report library** ("Summary & detail / Trends over time / Compare & rank…") — the best-designed surface in the app.
- **KPI recipe library** with business questions and calculation provenance — good bones for guided metric creation.

## Top 5 fixes if you only do five

1. Ship a **sample-data demo path** on the Studio empty state (one click → full happy path with zero uploads and zero LLM failures).
2. **Collapse the patch review into grouped, main-column diffs** with Accept all / per-group accept; kill the 47-checkbox rail.
3. **Auto-repair before showing patches**; never hand a first-time user a draft with validation errors.
4. **One save model** across the Builder: sticky bar, explicit dirty state, confirmation for active-catalog writes.
5. **Purge raw YAML/JSON/ids from default views** — collapse behind "View YAML" everywhere (and cap Save & Export's height).
