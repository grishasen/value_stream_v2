# Chat With Data

Chat With Data queries persisted aggregate metrics. An optional LLM intent
planner configured in `<workspace>/ai.yaml` supports free-form questions. Five
catalog-backed aggregate quick questions remain available when that planner is
disabled, not configured, or temporarily unreachable.

## Use Chat Without a Model

Open **Aggregate quick questions · no model required** to run the supported
deterministic templates. Chat offers each template only when the catalog has
the corresponding metric, dimension, or time grain:

- total count;
- CTR or engagement rate;
- approximate unique entities or customers;
- count breakdown by `Channel`;
- available aggregate date range.

These buttons map directly to catalog-validated intents and call the same
governed aggregate query layer used by dashboards. The date-range template
computes its bounds from aggregate time buckets; it does not scan source data.
Answers retain the normal query summary and freshness/provenance label.

No-model mode is intentionally not a natural-language parser. It does not
interpret different wording, filters, arbitrary dimensions, comparisons, or
follow-up questions. The free-form chat input stays disabled until a model is
configured and the planner is enabled. If a configured provider is
unreachable, the quick-question buttons remain usable; enable or restore the
planner for anything outside the listed templates.

## Configure a Model

Create `<workspace>/ai.yaml` with a LiteLLM model string. Keep secrets in
environment variables, not in the file:

```yaml
ai:
  llm:
    model: gpt-5.5
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    timeout_seconds: 120
```

`model` is a LiteLLM model string — the example above targets OpenAI with the
key read from `OPENAI_API_KEY`. `api_base` and `custom_provider` support local
or proxy-backed providers:

```yaml
# Local Ollama (after `ollama pull llama3.1` and `ollama serve`)
ai:
  llm:
    model: ollama/llama3.1
    api_base: http://localhost:11434
    custom_provider: ollama
    api_key_env: ""

# Anthropic (export ANTHROPIC_API_KEY first)
ai:
  llm:
    model: anthropic/claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
```

The Chat page also lets you override these values for the current Streamlit
session. MCP clients such as Claude Code should use the local stdio MCP server
instead of the Chat page; registration steps are in the
[API & MCP reference](../../reference/api-and-mcp.md).

Configuration Builder's Chat Review step can edit `chat_with_data` settings in
`ai.yaml`: a generic agent prompt plus dataset and metric descriptions that are
used only in the LLM planning prompt.

## Provider Preflight and Retry

Before a free-form planner, governed-SQL planner, or narrative request, Chat
runs the same independent `READY` capability check used by AI Configuration
Studio. The check is limited to five seconds and cached for the current model,
provider, endpoint, and credential. Missing configuration is rejected locally;
it does not create a provider request.

A classified failure shows a correlation reference without exposing provider
messages, prompts, credentials, query values, or local paths. Retryable
failures offer **Retry provider check**, which bypasses the short negative
cache. A failed check does not append the submitted question or an error turn
to chat history. Aggregate quick questions never run this preflight and remain
available while the provider is unavailable.

## Ask Questions

With a model configured, the LLM planner can return text, tables, or simple
charts. For example:

```text
Plot daily CTR by customer type and channel
```

The app validates that the metric, dimensions, and time bucket are available
in the catalog, then renders a deterministic Plotly chart from the same
governed query layer the dashboards use. It does not expose raw source rows,
raw aggregate files, SQL (unless governed SQL is explicitly enabled), or
generated Python execution.

Every chart and table answer offers a CSV download of the underlying governed
aggregate rows. Answers can be pinned to a "Chat Pins" dashboard.

## What the Planner Can and Cannot Do

- It plans a structured query intent (metric, dimensions, filters, having,
  sort, top-N, period comparison, quantile suite); Value Stream validates the
  intent against the catalog and executes it deterministically.
- It cannot choose aggregate grains, invent time columns, or bypass the
  aggregate-only contract.
- Ambiguous questions come back as clarifying questions instead of guesses.

## Privacy Notes

Chat only exposes catalog metadata and aggregate query results, but prompts
still leave the local app when using hosted model APIs. Do not use Chat With
Data for sensitive raw samples. See
[Security](../operations/security.md) for the full posture.

## Related Docs

- [API & MCP reference](../../reference/api-and-mcp.md) — the same governed
  tool layer for programmatic clients, including Claude Code registration.
- [Security](../operations/security.md) — the LLM boundary and governed SQL
  posture.
