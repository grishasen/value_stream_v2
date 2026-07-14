"""Aggregate-aware chat with data page."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import plotly.express as px  # type: ignore[import-untyped]
import streamlit as st

from valuestream.ai import AICallSettings
from valuestream.ai.chat import (
    ChatIntent,
    ChatQueryResult,
    chart_tile_from_intent,
    chat_pin_tile,
    chat_starter_questions,
    execute_chat_intent,
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
from valuestream.charts import render_chart
from valuestream.ui import builder, components
from valuestream.ui.context import ValueStreamContext
from valuestream.ui.theme import dashboard_theme
from valuestream.utils.logger import get_logger

logger = get_logger(__name__)

_REASONING_EFFORT_OPTIONS = ("", "minimal", "low", "medium", "high", "xhigh")
_VERBOSITY_OPTIONS = ("", "low", "medium", "high")


def render(ctx: ValueStreamContext) -> None:  # noqa: PLR0915 — Streamlit page entry point
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
    if settings is None:
        st.info("Configure and enable the LLM intent planner before asking questions.")
        st.chat_input("What would you like to know?", disabled=True)
        return
    _, chat_config = load_chat_with_data_config(ctx.workspace)

    starter = _render_starter_questions(ctx)
    prompt = st.chat_input("What would you like to know?") or starter
    if not prompt:
        return

    history = list(st.session_state.vs_chat_messages)
    st.session_state.vs_chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
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
                response: dict[str, Any] = {
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
            st.session_state.vs_chat_messages.append(response)
            index = len(st.session_state.vs_chat_messages) - 1
            _render_response(response, index)
            _render_pin_control(ctx, response, index)
        except Exception as exc:  # pragma: no cover - Streamlit display path
            logger.exception(
                "Chat query failed: controls=%s",
                controls,
            )
            message = f"I couldn't answer from aggregate data: {exc}"
            response = {"role": "assistant", "type": "text", "content": message}
            st.session_state.vs_chat_messages.append(response)
            st.error(message)


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
        logger.exception("Failed to save LLM settings to ai.yaml")
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
    return AICallSettings(
        model=model_name,
        api_key=str(st.session_state.get("vs_chat_api_key") or "").strip(),
        api_base=str(st.session_state.get("vs_chat_ai_api_base") or "").strip(),
        custom_llm_provider=str(st.session_state.get("vs_chat_ai_provider") or "").strip(),
        temperature=temperature,
        reasoning_effort=str(st.session_state.get("vs_chat_ai_reasoning_effort") or "").strip(),
        verbosity=str(st.session_state.get("vs_chat_ai_verbosity") or "").strip(),
        timeout_seconds=int(st.session_state.get("vs_chat_ai_timeout_seconds", 90)),
    )


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


def _render_pin_control(ctx: ValueStreamContext, response: dict[str, Any], index: int) -> None:
    """Offer to pin a governed answer to the Chat Pins dashboard."""
    intent = response.get("intent")
    if intent is None or not getattr(intent, "metric", ""):
        return
    if not st.button("Pin to dashboard", key=f"chat_pin_{index}", icon=":material/push_pin:"):
        return
    metric = str(intent.metric)
    try:
        tile_id = builder.random_catalog_id(metric, fallback="chat_tile")
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
        logger.exception("Failed to pin chat answer: metric=%s", metric)
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
        except Exception:
            logger.exception("Failed to read KPI value: column=%s", column)
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
        except Exception:
            logger.exception("Failed to summarize chat response value: column=%s", column)
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
            narrative = narrate_chat_result(settings, result)
    except Exception:
        logger.exception("Chat narrative failed; falling back to template summary")
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
    except Exception:
        logger.exception("Chat chart factory render failed; using fallback: tile=%s", tile)
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
