"""Aggregate-aware chat with data page."""

from __future__ import annotations

import json
import re
import time
from io import BytesIO
from typing import Any

import plotly.express as px  # type: ignore[import-untyped]
import streamlit as st

from valuestream.ai import AICallSettings
from valuestream.ai.chat import (
    ChatIntent,
    ChatIntentPlanningError,
    ChatQueryResult,
    DeterministicChatStarter,
    chart_tile_from_intent,
    chat_pin_tile,
    chat_starter_questions,
    deterministic_chat_starters,
    execute_chat_intent,
    execute_deterministic_chat_query,
    narrate_chat_result,
    overall_metric_value,
    plan_chat_intent,
)
from valuestream.ai.settings import (
    configured_api_key,
    load_chat_with_data_config,
    load_llm_settings_config,
    write_llm_settings_config,
)
from valuestream.ai.sql_tool import run_sql_query, sql_schema_summary
from valuestream.ai.studio import (
    AIProviderCallError,
    ai_provider_preflight_cache_key,
    ai_provider_settings_configured,
    preflight_ai_provider,
)
from valuestream.charts import render_chart
from valuestream.ui import builder, components
from valuestream.ui.context import ValueStreamContext
from valuestream.ui.theme import dashboard_theme
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

_REASONING_EFFORT_OPTIONS = ("", "minimal", "low", "medium", "high", "xhigh")
_VERBOSITY_OPTIONS = ("", "low", "medium", "high")
_CHAT_PREFLIGHT_NEGATIVE_TTL_SECONDS = 15
_CHAT_PREFLIGHT_CACHE_STATE_KEY = "vs_chat_ai_preflight_cache"
_CHAT_PREFLIGHT_FORCE_RETRY_STATE_KEY = "vs_chat_force_preflight_retry"
_CHAT_RETRY_PROMPT_STATE_KEY = "vs_chat_retry_prompt"


def render(  # noqa: PLR0912, PLR0915 — Streamlit page entry point
    ctx: ValueStreamContext,
) -> None:
    """Render chat over persisted aggregate data."""
    components.render_page_header(
        "Chat With Data",
        "Ask questions over selected aggregate metrics without exposing raw source rows.",
        status="ready" if ctx.catalog.metrics.metrics else "pending",
        status_label="Ready" if ctx.catalog.metrics.metrics else "No metrics",
    )

    if "vs_chat_messages" not in st.session_state:
        st.session_state.vs_chat_messages = []

    if not ctx.catalog.metrics.metrics:
        st.info("Add and run aggregate metrics before using Chat With Data.")
        return

    controls = _sidebar_controls(ctx)
    _render_history()

    settings = controls["settings"]
    retry_prompt = (
        str(st.session_state.pop(_CHAT_RETRY_PROMPT_STATE_KEY))
        if settings is not None and st.session_state.get(_CHAT_RETRY_PROMPT_STATE_KEY)
        else None
    )
    if settings is None:
        st.info(
            "No-model mode is active. The aggregate quick questions below still work; "
            "configure and enable the LLM planner for free-form questions."
        )
    deterministic_starter = _render_deterministic_starter_questions(
        ctx,
        expanded=settings is None,
    )
    chat_config: dict[str, Any] = {}
    planner_starter: str | None = None
    if settings is None:
        st.chat_input(
            "Enable the LLM planner for free-form questions",
            disabled=True,
        )
        prompt = deterministic_starter.question if deterministic_starter is not None else None
    else:
        _, chat_config = load_chat_with_data_config(ctx.workspace)
        planner_starter = _render_starter_questions(ctx)
        free_form_prompt = st.chat_input("What would you like to know?")
        prompt = (
            deterministic_starter.question
            if deterministic_starter is not None
            else retry_prompt or free_form_prompt or planner_starter
        )
    if not prompt:
        return

    history = list(st.session_state.vs_chat_messages)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if deterministic_starter is not None:
                with st.spinner("Querying governed aggregate template..."):
                    result = execute_deterministic_chat_query(
                        ctx.workspace,
                        ctx.catalog,
                        deterministic_starter,
                    )
                raw_intent = json.dumps(
                    {
                        "planner": "deterministic",
                        "template": deterministic_starter.key,
                    }
                )
                response = _respond_from_result(
                    result,
                    raw_intent=raw_intent,
                    workspace=ctx.workspace,
                    theme=_chat_theme(ctx),
                )
                response["content"] = (
                    f"Deterministic aggregate template; no model was used. {response['content']}"
                )
                response["intent"] = deterministic_starter.intent
                response["deterministic_key"] = deterministic_starter.key
                _commit_chat_turn(prompt, response)
                index = len(st.session_state.vs_chat_messages) - 1
                _render_response(response, index)
                _render_pin_control(ctx, response, index)
                return

            if settings is None:  # pragma: no cover - guarded by disabled input
                return
            with st.spinner("Checking provider capability..."):
                _preflight_chat_provider(settings)
            plan_kwargs: dict[str, Any] = {"history": history, "chat_config": chat_config}
            if controls.get("sql_enabled"):
                with st.spinner("Loading governed SQL schema..."):
                    plan_kwargs["sql_schema"] = sql_schema_summary(ctx.workspace, ctx.catalog)
            with st.spinner("Planning aggregate query..."):
                intent, raw_intent = plan_chat_intent(
                    settings,
                    ctx.catalog,
                    prompt,
                    **plan_kwargs,
                )
            if intent.response == "clarify":
                response = {
                    "role": "assistant",
                    "type": "text",
                    "content": intent.clarify or "Could you clarify the question?",
                    "intent_json": _intent_json(intent, raw_intent),
                }
            elif intent.response == "sql":
                with st.spinner("Running governed SQL..."):
                    sql_result = run_sql_query(ctx.workspace, intent.sql or "", catalog=ctx.catalog)
                result = ChatQueryResult(
                    intent=intent,
                    rows=sql_result.rows,
                    query_summary=sql_result.sql,
                    freshness="governed SQL over current aggregate views",
                )
                response = _respond_from_sql(result, sql_result.truncated, raw_intent=raw_intent)
                response = _maybe_narrate(response, result, settings, controls)
            else:
                with st.spinner("Querying aggregate data..."):
                    result = execute_chat_intent(ctx.workspace, ctx.catalog, intent)
                response = _respond_from_result(
                    result,
                    raw_intent=raw_intent,
                    workspace=ctx.workspace,
                    theme=_chat_theme(ctx),
                )
                response = _maybe_narrate(response, result, settings, controls)
                response["intent"] = intent
            _commit_chat_turn(prompt, response)
            index = len(st.session_state.vs_chat_messages) - 1
            _render_response(response, index)
            _render_pin_control(ctx, response, index)
        except Exception as exc:  # pragma: no cover - Streamlit display path
            if isinstance(exc, AIProviderCallError) and settings is not None:
                _cache_chat_provider_failure(settings, exc)
            _log_chat_query_failure(exc)
            _render_chat_query_failure(exc, prompt=prompt)


def _commit_chat_turn(prompt: str, response: dict[str, Any]) -> None:
    """Commit a complete turn atomically after all required work succeeds."""

    st.session_state.vs_chat_messages.extend(
        [
            {"role": "user", "content": prompt},
            response,
        ]
    )


def _preflight_chat_provider(settings: AICallSettings) -> None:
    """Run or reuse the session-scoped Chat provider capability check."""

    cache_key = ai_provider_preflight_cache_key(settings, capability="chat")
    cache = st.session_state.setdefault(_CHAT_PREFLIGHT_CACHE_STATE_KEY, {})
    force_retry = bool(st.session_state.pop(_CHAT_PREFLIGHT_FORCE_RETRY_STATE_KEY, False))
    cached = cache.get(cache_key) if isinstance(cache, dict) else None
    if force_retry and isinstance(cache, dict):
        cache.pop(cache_key, None)
        cached = None
    if isinstance(cached, dict) and cached.get("ok") is True:
        return
    if isinstance(cached, dict) and cached.get("ok") is False:
        expires_at = float(cached.get("expires_at") or 0.0)
        if expires_at > time.monotonic():
            _raise_cached_chat_preflight_failure(cached)
        if isinstance(cache, dict):
            cache.pop(cache_key, None)

    try:
        receipt = preflight_ai_provider(settings)
    except AIProviderCallError as exc:
        _cache_chat_provider_failure(settings, exc)
        raise
    if isinstance(cache, dict):
        cache[cache_key] = {"ok": True, "reference": receipt.reference}


def _cache_chat_provider_failure(
    settings: AICallSettings,
    failure: AIProviderCallError,
) -> None:
    """Cache a classified failure without provider text or request data."""

    cache = st.session_state.setdefault(_CHAT_PREFLIGHT_CACHE_STATE_KEY, {})
    if not isinstance(cache, dict):
        return
    cache_key = ai_provider_preflight_cache_key(settings, capability="chat")
    cache[cache_key] = {
        "ok": False,
        "expires_at": time.monotonic() + _CHAT_PREFLIGHT_NEGATIVE_TTL_SECONDS,
        "call_id": failure.call_id,
        "error_type": failure.error_type,
        "permission_denied": failure.permission_denied,
        "category": str(failure.category),
        "retryable": failure.retryable,
    }


def _raise_cached_chat_preflight_failure(entry: dict[str, Any]) -> None:
    """Raise the classified equivalent of a cached Chat preflight failure."""

    raise AIProviderCallError(
        call_id=str(entry.get("call_id") or "cached"),
        error_type=str(entry.get("error_type") or "ProviderError"),
        permission_denied=bool(entry.get("permission_denied")),
        category=str(entry.get("category") or "provider"),
    )


def _render_chat_query_failure(exc: Exception, *, prompt: str) -> None:
    """Render a safe failure receipt and an explicit retry when applicable."""

    st.error(_chat_query_error_message(exc))
    if not isinstance(exc, AIProviderCallError):
        return
    with st.expander("Technical details · provider reference", expanded=False):
        components.key_value_strip(
            [
                {"label": "Reference", "value": exc.call_id},
                {"label": "Category", "value": str(exc.category).replace("_", " ")},
                {"label": "Retryable", "value": "Yes" if exc.retryable else "No"},
            ]
        )
    if exc.retryable and st.button(
        "Retry provider check",
        key="vs_chat_retry_provider_check",
        icon=":material/refresh:",
    ):
        st.session_state[_CHAT_PREFLIGHT_FORCE_RETRY_STATE_KEY] = True
        st.session_state[_CHAT_RETRY_PROMPT_STATE_KEY] = prompt
        st.rerun()


def _log_chat_query_failure(exc: Exception) -> None:
    """Log a bounded failure classification without request or model payloads."""

    _log_chat_operation_failure("Chat query", exc)


def _log_chat_operation_failure(operation: str, exc: Exception) -> None:
    """Log only a static operation and bounded exception class."""

    error_type = type(exc).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", error_type):
        error_type = "ApplicationError"
    if isinstance(exc, AIProviderCallError):
        logger.error(
            "%s failed: status=error error_type=%s failure_category=%s retryable=%s reference=%s",
            operation,
            error_type,
            exc.category,
            exc.retryable,
            exc.call_id,
        )
        return
    logger.error("%s failed: status=error error_type=%s", operation, error_type)


def _chat_query_error_message(exc: Exception) -> str:
    if isinstance(exc, AIProviderCallError):
        category = str(exc.category)
        remediation = {
            "configuration": "Add a model plus an API key or custom endpoint in Chat Settings.",
            "authentication": "Replace or refresh the configured API credential.",
            "authorization": "Check project and model permissions for this credential.",
            "rate_limit": "Wait briefly, then retry the provider check.",
            "timeout": "Check provider latency or endpoint reachability, then retry.",
            "network": "Check the configured endpoint and network connection, then retry.",
            "provider": "Retry; if it persists, check provider service health.",
            "response_validation": "Retry the provider check or choose another model.",
            "internal": "Review the correlation reference in operational logs.",
        }[category]
        return (
            f"I couldn't use the configured planner because of a {category.replace('_', ' ')} "
            f"failure. {remediation} Reference: {exc.call_id}. Existing chat history was not "
            "changed. You can still choose a no-model aggregate quick question above."
        )
    if isinstance(exc, ChatIntentPlanningError):
        return (
            f"I couldn't use the configured planner. {exc} "
            "You can still choose a no-model aggregate quick question above."
        )
    return (
        "I couldn't answer from aggregate data. Try rephrasing the question or check the "
        "configured metric and model settings."
    )


def _sidebar_controls(ctx: ValueStreamContext) -> dict[str, Any]:
    _initialize_chat_ai_settings(ctx.workspace)
    with st.sidebar:
        st.write("### Chat Settings")
        settings = _llm_sidebar_controls()
        narrative = st.toggle(
            "Narrative answers",
            value=True,
            key="vs_chat_narrative_enabled",
            help=(
                "Ask the model to summarize the governed query result in plain "
                "language. Only the returned aggregate rows are shared."
            ),
        )
        sql_enabled = st.toggle(
            "Governed SQL answers",
            value=False,
            key="vs_chat_sql_enabled",
            help=(
                "Let the planner escalate to a single read-only SELECT over the "
                "aggregate DuckDB views when the metric intent cannot express "
                "the question. No raw source rows are reachable."
            ),
        )
        _render_save_settings(ctx)

        st.divider()
        if st.button("Clear chat", icon=":material/delete:"):
            st.session_state.vs_chat_messages = []
            st.rerun()
        if st.session_state.vs_chat_messages:
            chat_log = "\n\n".join(
                f"{msg['role'].capitalize()}: {msg.get('content', msg.get('summary', ''))}"
                for msg in st.session_state.vs_chat_messages
            )
            st.download_button(
                "Download chat",
                data=BytesIO(chat_log.encode("utf-8")),
                file_name="value_stream_chat.txt",
                mime="text/plain",
                icon=":material/download:",
            )
    return {"settings": settings, "narrative": narrative, "sql_enabled": sql_enabled}


def _render_save_settings(ctx: ValueStreamContext) -> None:
    """Persist the current session LLM settings to the workspace ``ai.yaml``."""
    model_name = str(st.session_state.get("vs_chat_ai_model") or "").strip()
    if not st.button(
        "Save LLM settings to ai.yaml",
        icon=":material/save:",
        disabled=not model_name,
        help="Writes the ai.llm block so this model is the default next session. Secrets are not written.",
    ):
        return
    temperature = (
        float(st.session_state.get("vs_chat_ai_temperature", 0.1))
        if st.session_state.get("vs_chat_ai_temperature_enabled")
        else None
    )
    try:
        path = write_llm_settings_config(
            ctx.workspace,
            model=model_name,
            api_base=str(st.session_state.get("vs_chat_ai_api_base") or "").strip(),
            custom_provider=str(st.session_state.get("vs_chat_ai_provider") or "").strip(),
            temperature=temperature,
            reasoning_effort=str(st.session_state.get("vs_chat_ai_reasoning_effort") or "").strip(),
            verbosity=str(st.session_state.get("vs_chat_ai_verbosity") or "").strip(),
            timeout_seconds=int(st.session_state.get("vs_chat_ai_timeout_seconds", 90)),
        )
        st.success(f"Saved LLM settings to `{path.name}`.")
    except Exception as exc:  # pragma: no cover - Streamlit display path
        _log_chat_operation_failure("Chat LLM settings save", exc)
        st.error(f"Could not save settings: {exc}")


def _llm_sidebar_controls() -> AICallSettings | None:
    model_name = str(st.session_state.get("vs_chat_ai_model") or "").strip()
    default_enabled = bool(model_name)
    llm_enabled = st.toggle(
        "LLM intent planner",
        value=default_enabled,
        key="vs_chat_llm_enabled",
        help="Maps natural-language questions to validated aggregate metric queries.",
    )
    with st.expander("LLM Settings", expanded=llm_enabled and not model_name):
        st.text_input(
            "Model",
            key="vs_chat_ai_model",
            help="LiteLLM model name, for example gpt-5.5 or ollama/llama3.1.",
        )
        st.text_input(
            "API Base",
            key="vs_chat_ai_api_base",
            placeholder="http://localhost:11434",
            help="Optional base URL for Ollama, LM Studio, vLLM, or a LiteLLM proxy.",
        )
        st.text_input(
            "Custom Provider",
            key="vs_chat_ai_provider",
            help="Optional LiteLLM provider override when the model prefix is ambiguous.",
        )
        st.text_input(
            "API Key",
            type="password",
            key="vs_chat_api_key",
            help="Optional for local models. Stored only in Streamlit session state.",
        )
        if st.toggle(
            "Override Temperature",
            key="vs_chat_ai_temperature_enabled",
            help="Leave off for providers that only accept their default temperature.",
        ):
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="vs_chat_ai_temperature",
            )
        st.selectbox(
            "Reasoning Effort",
            _REASONING_EFFORT_OPTIONS,
            format_func=_ai_option_label,
            key="vs_chat_ai_reasoning_effort",
            help="Optional LiteLLM reasoning_effort value for models that support it.",
        )
        st.selectbox(
            "Verbosity",
            _VERBOSITY_OPTIONS,
            format_func=_ai_option_label,
            key="vs_chat_ai_verbosity",
            help="Optional LiteLLM verbosity value for models that support it.",
        )
        st.number_input(
            "Timeout Seconds",
            min_value=10,
            max_value=600,
            step=10,
            key="vs_chat_ai_timeout_seconds",
        )
        if st.session_state.get("vs_chat_ai_config_path"):
            st.caption(f"Loaded defaults from `{st.session_state['vs_chat_ai_config_path']}`.")
    if not llm_enabled:
        return None
    model_name = str(st.session_state.get("vs_chat_ai_model") or "").strip()
    if not model_name:
        st.caption("Set a model to enable LLM planning.")
        return None
    temperature = (
        float(st.session_state.get("vs_chat_ai_temperature", 0.1))
        if st.session_state.get("vs_chat_ai_temperature_enabled")
        else None
    )
    settings = AICallSettings(
        model=model_name,
        api_key=str(st.session_state.get("vs_chat_api_key") or "").strip(),
        api_base=str(st.session_state.get("vs_chat_ai_api_base") or "").strip(),
        custom_llm_provider=str(st.session_state.get("vs_chat_ai_provider") or "").strip(),
        temperature=temperature,
        reasoning_effort=str(st.session_state.get("vs_chat_ai_reasoning_effort") or "").strip(),
        verbosity=str(st.session_state.get("vs_chat_ai_verbosity") or "").strip(),
        timeout_seconds=int(st.session_state.get("vs_chat_ai_timeout_seconds", 90)),
    )
    if not ai_provider_settings_configured(settings):
        st.caption(
            "Add an API key or custom endpoint in LLM Settings to enable free-form questions. "
            "Local providers such as Ollama can run without a key."
        )
        return None
    return settings


def _initialize_chat_ai_settings(workspace) -> None:
    config_path, config = load_llm_settings_config(workspace)
    signature = (
        str(config_path) if config_path else str(workspace),
        config_path.stat().st_mtime_ns if config_path and config_path.exists() else 0,
    )
    if st.session_state.get("vs_chat_ai_config_signature") == signature:
        _ensure_chat_ai_defaults(config)
        return
    st.session_state["vs_chat_ai_config_signature"] = signature
    st.session_state["vs_chat_ai_config_path"] = str(config_path) if config_path else ""
    _ensure_chat_ai_defaults(config)
    mappings = {
        "model": "vs_chat_ai_model",
        "api_base": "vs_chat_ai_api_base",
        "custom_provider": "vs_chat_ai_provider",
        "custom_llm_provider": "vs_chat_ai_provider",
        "temperature": "vs_chat_ai_temperature",
        "reasoning_effort": "vs_chat_ai_reasoning_effort",
        "verbosity": "vs_chat_ai_verbosity",
        "timeout_seconds": "vs_chat_ai_timeout_seconds",
    }
    for config_key, state_key in mappings.items():
        value = config.get(config_key)
        if value is not None:
            st.session_state[state_key] = value
    if config.get("temperature") is not None:
        st.session_state["vs_chat_ai_temperature_enabled"] = True
    api_key = configured_api_key(config)
    if api_key:
        st.session_state.setdefault("vs_chat_api_key", api_key)


def _ensure_chat_ai_defaults(config: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {
        "vs_chat_ai_model": str(config.get("model") or ""),
        "vs_chat_ai_api_base": "",
        "vs_chat_ai_provider": "",
        "vs_chat_ai_temperature_enabled": False,
        "vs_chat_ai_temperature": 0.1,
        "vs_chat_ai_reasoning_effort": "",
        "vs_chat_ai_verbosity": "",
        "vs_chat_ai_timeout_seconds": 90,
        "vs_chat_api_key": configured_api_key(config),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    _normalize_ai_selectbox_state("vs_chat_ai_reasoning_effort", _REASONING_EFFORT_OPTIONS)
    _normalize_ai_selectbox_state("vs_chat_ai_verbosity", _VERBOSITY_OPTIONS)


def _normalize_ai_selectbox_state(key: str, options: tuple[str, ...]) -> None:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def _ai_option_label(value: str) -> str:
    return value if value else "Default"


def _render_history() -> None:
    for index, message in enumerate(st.session_state.vs_chat_messages):
        with st.chat_message(message["role"]):
            _render_response(message, index)


def _render_starter_questions(ctx: ValueStreamContext) -> str | None:
    """Offer example questions when the conversation is empty; return a click."""
    if st.session_state.vs_chat_messages:
        return None
    questions = chat_starter_questions(ctx.catalog)
    if not questions:
        return None
    st.caption("Try one of these to get started:")
    # The pills disappear once the first message lands, so the selection is
    # consumed exactly once.
    return st.pills(
        "Starter questions",
        questions,
        selection_mode="single",
        label_visibility="collapsed",
        key="vs_chat_starter",
    )


def _render_deterministic_starter_questions(
    ctx: ValueStreamContext,
    *,
    expanded: bool,
) -> DeterministicChatStarter | None:
    """Render catalog-backed buttons that bypass model planning."""

    starters = deterministic_chat_starters(ctx.catalog)
    if not starters:
        return None
    selected: DeterministicChatStarter | None = None
    with st.expander("Aggregate quick questions · no model required", expanded=expanded):
        st.caption(
            "Supports configured totals, CTR or engagement rate, approximate unique entities, "
            "Channel breakdown, and available aggregate dates. Enable the LLM planner for other "
            "wording, filters, comparisons, and follow-up questions."
        )
        columns = st.columns(2)
        for index, starter in enumerate(starters):
            if columns[index % len(columns)].button(
                starter.question,
                key=f"vs_chat_deterministic_{starter.key}",
                width="stretch",
            ):
                selected = starter
    return selected


def _render_pin_control(ctx: ValueStreamContext, response: dict[str, Any], index: int) -> None:
    """Offer to pin a governed answer to the Chat Pins dashboard."""
    if response.get("deterministic_key") == "date_range":
        return
    intent = response.get("intent")
    if intent is None or not getattr(intent, "metric", ""):
        return
    if not st.button("Pin to dashboard", key=f"chat_pin_{index}", icon=":material/push_pin:"):
        return
    metric = str(intent.metric)
    try:
        existing_tile_ids = [
            tile.id
            for dashboard in ctx.catalog.dashboards.dashboards
            if dashboard.id == "chat_pins"
            for page in dashboard.pages
            if page.id == "pinned"
            for tile in page.tiles
        ]
        tile_id = builder.stable_catalog_id(
            metric,
            fallback="chat_tile",
            parent_id="chat_pins_pinned",
            existing_ids=existing_tile_ids,
        )
        tile = chat_pin_tile(intent, tile_id=tile_id)
        builder.write_tile_definition(
            ctx.workspace,
            dashboard_id="chat_pins",
            dashboard_title="Chat Pins",
            page_id="pinned",
            page_title="Pinned Answers",
            tile=tile,
        )
        st.success("Pinned to the Chat Pins dashboard. Reload the catalog to see it.")
    except Exception as exc:  # pragma: no cover - Streamlit display path
        _log_chat_operation_failure("Chat answer pin", exc)
        st.error(f"Could not pin this answer: {exc}")


def _respond_from_result(
    result: ChatQueryResult,
    *,
    raw_intent: str,
    workspace: Any = None,
    theme: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = result.intent
    rows = result.rows
    chart = intent.chart
    summary = (
        f"`{intent.metric}` returned {rows.height:,} aggregate row(s) at `{intent.grain}` grain"
        + (f" grouped by {', '.join(intent.group_by)}." if intent.group_by else ".")
    )
    summary = f"{summary} Freshness: {result.freshness}."
    base = {
        "role": "assistant",
        "content": summary,
        "query": result.query_summary,
        "intent_json": _intent_json(intent, raw_intent),
    }
    if rows.is_empty():
        return {**base, "type": "text", "content": f"{summary} No data is available."}
    if intent.response == "table" or (chart is not None and chart.kind == "table"):
        return {**base, "type": "data", "data": rows}
    if chart is not None and chart.kind == "kpi_card":
        kpi = _kpi_value(rows, intent, workspace)
        if kpi is not None:
            column, value = kpi
            return {
                **base,
                "type": "kpi",
                "kpi_label": column,
                "kpi_value": _format_metric_value(value, chart.value_format),
                "data": rows,
            }
        return {**base, "type": "data", "data": rows}
    if intent.response == "chart" and chart is not None:
        return {
            **base,
            "type": "chart",
            "figure": _chat_figure(rows, intent, theme or {}),
            "data": rows,
        }
    return {**base, "type": "text", "content": summary + _value_detail(rows, intent, workspace)}


def _kpi_value(rows, intent: ChatIntent, workspace: Any) -> tuple[str, Any] | None:
    if rows.height == 1:
        column = _default_value_column(rows, intent)
        try:
            return column, rows.get_column(column).item()
        except Exception as exc:
            _log_chat_operation_failure("Chat KPI value read", exc)
            return None
    if workspace is not None:
        # Report the governed overall value rather than averaging grouped rows,
        # which would weight every group equally regardless of volume.
        return overall_metric_value(workspace, intent)
    return None


def _value_detail(rows, intent: ChatIntent, workspace: Any) -> str:
    if rows.height == 1:
        column = _default_value_column(rows, intent)
        try:
            return f" `{column}` is {_format_value(rows.get_column(column).item())}."
        except Exception as exc:
            _log_chat_operation_failure("Chat response value summary", exc)
            return ""
    if workspace is not None:
        overall = overall_metric_value(workspace, intent)
        if overall is not None:
            column, value = overall
            return f" Overall `{column}` for this selection is {_format_value(value)}."
    return ""


_VALUE_FORMATTERS = {
    "percent": lambda number: f"{number:.2%}",
    "integer": lambda number: f"{number:,.0f}",
    "currency": lambda number: f"${number:,.2f}",
    "number": lambda number: f"{number:,.4g}",
}


def _format_metric_value(value: Any, value_format: str | None) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    formatter = _VALUE_FORMATTERS.get(value_format or "")
    return formatter(number) if formatter is not None else _format_value(value)


def _respond_from_sql(
    result: ChatQueryResult,
    truncated: bool,
    *,
    raw_intent: str,
) -> dict[str, Any]:
    rows = result.rows
    summary = f"Governed SQL returned {rows.height:,} row(s)."
    if truncated:
        summary = f"{summary} The result was truncated to the row cap."
    base = {
        "role": "assistant",
        "content": summary,
        "query": result.query_summary,
        "intent_json": _intent_json(result.intent, raw_intent),
    }
    if rows.is_empty():
        return {**base, "type": "text", "content": f"{summary} No data is available."}
    return {**base, "type": "data", "data": rows}


def _maybe_narrate(
    response: dict[str, Any],
    result: ChatQueryResult,
    settings: AICallSettings,
    controls: dict[str, Any],
) -> dict[str, Any]:
    if not controls.get("narrative") or result.rows.is_empty():
        return response
    try:
        with st.spinner("Summarizing result..."):
            _preflight_chat_provider(settings)
            narrative = narrate_chat_result(settings, result)
    except Exception as exc:
        if isinstance(exc, AIProviderCallError):
            _cache_chat_provider_failure(settings, exc)
            st.warning(
                "The governed result is available without a model narrative. "
                f"Provider reference: {exc.call_id}."
            )
        _log_chat_operation_failure("Chat narrative", exc)
        return response
    if not narrative:
        return response
    return {**response, "content": f"{narrative}\n\n{response.get('content', '')}"}


def _render_response(message: dict[str, Any], index: int) -> None:
    message_type = message.get("type", "text")
    if message_type in {"chart", "plotly"}:
        st.markdown(message["content"])
        st.plotly_chart(
            message["figure"],
            width="stretch",
            theme=None,
            key=f"chat_plot_{index}",
        )
        _render_download(message.get("data"), index)
    elif message_type == "kpi":
        st.markdown(message["content"])
        st.metric(str(message.get("kpi_label", "")), str(message.get("kpi_value", "")))
    elif message_type == "data":
        st.markdown(message["content"])
        st.dataframe(message["data"], hide_index=True, width="stretch")
        _render_download(message.get("data"), index)
    else:
        st.markdown(message.get("content", ""))
    if message.get("query"):
        with st.status("Generated aggregate query", expanded=False):
            st.code(message["query"], language="text")
            if message.get("intent_json"):
                st.code(message["intent_json"], language="json")


def _render_download(data: Any, index: int) -> None:
    if data is None:
        return
    st.download_button(
        "Download data (CSV)",
        # A callable defers CSV serialization until the user actually clicks,
        # instead of serializing every historical result on every rerun.
        data=lambda: data.write_csv().encode("utf-8"),
        file_name="value_stream_result.csv",
        mime="text/csv",
        key=f"chat_csv_{index}",
        icon=":material/download:",
    )


def _chat_theme(ctx: ValueStreamContext) -> dict[str, Any]:
    base = dict(dashboard_theme())
    dashboards = getattr(ctx.catalog, "dashboards", None)
    extra = getattr(dashboards, "theme", {}) if dashboards is not None else {}
    return {**base, **(extra or {})}


def _chat_figure(rows, intent: ChatIntent, theme: dict[str, Any]):
    tile = _fit_tile_to_columns(chart_tile_from_intent(intent), rows, intent)
    try:
        return render_chart(rows, tile, theme=theme)
    except Exception as exc:
        _log_chat_operation_failure("Chat chart render", exc)
        return _px_fallback_figure(rows, intent)


def _fit_tile_to_columns(tile: dict[str, Any], rows, intent: ChatIntent) -> dict[str, Any]:
    """Repair tile field references so a validated intent never breaks the factory."""
    out = dict(tile)
    columns = list(rows.columns)
    present = set(columns)
    if out.get("x") not in present and "x" in out:
        out["x"] = columns[0] if columns else None
    for key in ("y", "value", "values"):
        if key in out and out.get(key) not in present:
            out[key] = _default_value_column(rows, intent)
    for key in ("color", "facet_col", "names"):
        if key in out and out.get(key) not in present:
            out.pop(key, None)
    return out


def _px_fallback_figure(rows, intent: ChatIntent):
    frame = rows
    chart = intent.chart
    x = _first_existing(chart.x if chart else None, frame.columns) or frame.columns[0]
    y = _first_existing(chart.y if chart else None, frame.columns) or _default_value_column(
        rows, intent
    )
    color = _first_existing(chart.color if chart else None, frame.columns)
    color = color or _default_chart_color(intent, x=x, facet_col=None, columns=frame.columns)
    sort_columns = [
        column for column in (color, x) if column is not None and column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort(sort_columns)
    title = f"{intent.metric} by {x}"
    if chart is not None and chart.kind == "bar":
        return px.bar(frame, x=x, y=y, color=color, title=title)
    return px.line(frame, x=x, y=y, color=color, title=title)


def _default_chart_color(
    intent: ChatIntent,
    *,
    x: str,
    facet_col: str | None,
    columns,
) -> str | None:
    for candidate in intent.group_by:
        if candidate in (x, facet_col) or candidate not in columns:
            continue
        logger.info(
            "Rendering chart with grouped-dimension color fallback: metric=%s color=%s x=%s group_by=%s",
            intent.metric,
            candidate,
            x,
            intent.group_by,
        )
        return candidate
    return None


def _default_value_column(rows, intent: ChatIntent) -> str:
    if intent.chart and intent.chart.y and intent.chart.y in rows.columns:
        return intent.chart.y
    if intent.metric in rows.columns:
        return intent.metric
    return str(rows.columns[-1])


def _first_existing(value: str | None, columns) -> str | None:
    if value is None:
        return None
    return value if value in columns else None


def _intent_json(intent: ChatIntent, raw_intent: str) -> str:
    try:
        raw = json.loads(raw_intent)
    except Exception:
        raw = None
    payload = {
        "validated": {
            "metric": intent.metric,
            "response": intent.response,
            "group_by": intent.group_by,
            "filters": intent.filters,
            "having": intent.having,
            "order_by": intent.order_by,
            "top_n": intent.top_n,
            "top_n_by": intent.top_n_by,
            "compare": intent.compare,
            "quantiles": intent.quantiles,
            "grain": intent.grain,
            "start": intent.start,
            "end": intent.end,
            "chart": intent.chart.__dict__ if intent.chart else None,
            "clarify": intent.clarify,
            "sql": intent.sql,
            "limit": intent.limit,
        },
        "model_raw": raw if raw is not None else raw_intent,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
