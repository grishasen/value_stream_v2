# Chat With Data

Chat With Data queries persisted aggregate metrics through an LLM intent
planner configured in `<workspace>/ai.yaml`. If the planner is disabled or no
model is configured, the chat input remains disabled.

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

Local Ollama, OpenAI, and Anthropic setups — plus MCP registration for Claude
Code — are documented step by step in
[Chat With Data MLP1](../../design/chat-with-data-mlp1.md).

Configuration Builder's Chat Review step can edit `chat_with_data` settings in
`ai.yaml`: a generic agent prompt plus dataset and metric descriptions that are
used only in the LLM planning prompt.

## Ask Questions

The LLM planner can return text, tables, or simple charts. For example:

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

- [Chat With Data MLP1 design](../../design/chat-with-data-mlp1.md) — intent
  schema, chart allowlist, provider setup, and operational notes.
- [API & MCP reference](../../reference/api-and-mcp.md) — the same governed
  tool layer for programmatic clients.
