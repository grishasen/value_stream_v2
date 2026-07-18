# Configuration authoring rollout

Use this runbook to expose the revised Build, Configuration Builder, and AI
Configuration Studio lifecycle while collecting a privacy-safe baseline.

**Rollout status:** Code ready; measurement pending. No representative
baseline or like-for-like cohort comparison has been collected, so the feature
flag and legacy navigation grouping remain in place.

## Feature flag

The revised Build entry is enabled by default. To hide the entry and keep the
authoring pages in their legacy Settings navigation group:

```sh
export VALUESTREAM_AUTHORING_V2=0
uv run valuestream serve examples/demo --port 8501 --headless
```

Unset the variable or set it to `1`, `true`, `yes`, `on`, or `enabled` to
expose Build. Any unknown value defaults safely to enabled so deployment typos
do not create divergent hidden semantics.

## Event contract

Authoring events are ordinary structured application logs. Each line may
contain only:

- anonymous per-session `journey_id`;
- allowlisted workflow, event, stage, and outcome;
- bounded duration/count values; and
- whether the applied change requires a data run.

Events cover entry, sample chosen, consent confirmed, draft requested, valid
proposal, review, apply, explicit run, report open, failure, and explicit
abandon/restart. They never contain sample or field values, field/object IDs,
workspace paths, prompts, responses, credentials, endpoints, or raw exception
text.

## Baseline and comparison

Do not set numerical conversion targets before the first representative
baseline window. For both workflows, aggregate counts and median/p90 duration
by the allowlisted stages:

1. entered → valid proposal;
2. valid proposal → reviewed;
3. reviewed → applied;
4. applied → report opened or run started;
5. failed/timeout/retry by stage; and
6. abandoned by last reached stage.

Use a window long enough to cover the normal authoring cadence and compare
like-for-like workspace cohorts. Small local teams should report counts with
rates so a single journey cannot masquerade as a trend.

## Rollout gates

1. Validate the privacy log-capture tests and the authoring accessibility
   matrix.
2. Enable Build for an internal cohort.
3. Inspect failure and abandonment without opening or enriching events with
   customer data.
4. Compare the complete-funnel baseline with the revised path.
5. Broaden exposure only when critical correctness tests remain green and no
   new high-frequency dead-end appears.
6. Retire the flag and legacy navigation grouping only after the comparison is
   documented. Code readiness alone does not satisfy this gate.

Rollback changes navigation exposure; it does not undo catalog changes already
applied by users. Catalog rollback follows the normal version-controlled YAML
and transaction process.
