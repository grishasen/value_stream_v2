# AI Studio cancellation spike

**Date:** 2026-07-18
**Decision:** Ship timeout, named stages, preserved state, and retry; do not show
a Cancel control for synchronous model calls.

## Question

Can AI Configuration Studio offer a truthful native Streamlit Cancel action
while a provider generation or repair request is running?

## Findings

- A normal Streamlit button can only be handled by another script rerun. The
  current generation call occupies the active script run until the provider
  returns or its configured timeout fires.
- Moving the request to a background thread would make the UI pollable, but
  Python thread cancellation cannot safely stop an in-flight HTTP request. A
  Cancel button would therefore hide or abandon a result while provider work
  could continue, which violates the product's truthful-action rule.
- A separate worker process could be terminated, but it would add process
  supervision, cross-process state transfer, provider-client lifecycle, and
  cleanup semantics that are disproportionate for this local single-host UI.
  It also needs a documented durability boundary before adoption.
- Provider timeouts are enforceable at the existing call boundary. Candidate
  work can remain session-local and the prior valid revision can be preserved
  across timeout, permission failure, invalid output, and repair exhaustion.

## Implemented recovery contract

1. Name the active operation stage: provider check, generation, validation,
   repair, or ready for review.
2. Apply a hard configured timeout to provider calls.
3. Do not replace the last valid accepted draft until a candidate parses,
   merges, and validates.
4. Run at most two bounded repair passes for an invalid candidate.
5. On timeout or failure, show one safe corrective message and a Retry action;
   keep deterministic generation available.
6. Do not render acceptance/apply controls for an invalid candidate.

## Revisit trigger

Reconsider cancellable execution only if Value Stream adopts a supervised job
runner with an explicit job identity, persisted state machine, provider-level
cancellation where available, and tested cleanup after process termination.
That change crosses the current UI/process architecture and requires an ADR.
