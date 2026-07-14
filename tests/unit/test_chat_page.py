"""Chat page rendering tests."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import polars as pl
import pytest

from valuestream.ai import AICallSettings
from valuestream.ai.chat import ChartIntent, ChatIntent, ChatQueryResult
from valuestream.ui.pages import chat


class _SessionState(dict):
    def __getattr__(self, key: str):
        return self[key]

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value


@contextmanager
def _null_context(*args: object, **kwargs: object):
    yield


@pytest.mark.unit
def test_render_response_accepts_plain_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(chat.st, "markdown", rendered.append)

    chat._render_response({"role": "user", "content": "Show CTR"}, 0)

    assert rendered == ["Show CTR"]


@pytest.mark.unit
def test_render_requires_llm_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    session_state = _SessionState()
    info_messages: list[str] = []
    chat_inputs: list[dict[str, object]] = []

    ctx = SimpleNamespace(
        workspace="/tmp/workspace",
        catalog=SimpleNamespace(metrics=SimpleNamespace(metrics={"CTR": object()})),
    )

    monkeypatch.setattr(chat.st, "session_state", session_state)
    monkeypatch.setattr(chat.components, "render_page_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_sidebar_controls", lambda ctx: {"settings": None})
    monkeypatch.setattr(chat, "_render_history", lambda: None)
    monkeypatch.setattr(chat.st, "info", info_messages.append)

    def fake_chat_input(label: str, **kwargs: object) -> str:
        chat_inputs.append({"label": label, **kwargs})
        return "Plot daily CTR by channel"

    def fail_plan(*args: object, **kwargs: object) -> None:
        raise AssertionError("LLM planner should not run without settings")

    def fail_execute(*args: object, **kwargs: object) -> None:
        raise AssertionError("aggregate query should not run without an LLM intent")

    monkeypatch.setattr(chat.st, "chat_input", fake_chat_input)
    monkeypatch.setattr(chat, "plan_chat_intent", fail_plan)
    monkeypatch.setattr(chat, "execute_chat_intent", fail_execute)

    chat.render(ctx)  # type: ignore[arg-type]

    assert info_messages == ["Configure and enable the LLM intent planner before asking questions."]
    assert chat_inputs == [{"label": "What would you like to know?", "disabled": True}]
    assert session_state.vs_chat_messages == []


@pytest.mark.unit
def test_render_uses_llm_planner_before_query(monkeypatch: pytest.MonkeyPatch) -> None:
    session_state = _SessionState()
    settings = AICallSettings(
        model="ollama/llama3.1",
        api_base="http://localhost:11434",
        custom_llm_provider="ollama",
    )
    captured: dict[str, object] = {}
    rendered: list[str] = []

    ctx = SimpleNamespace(
        workspace="/tmp/workspace",
        catalog=SimpleNamespace(metrics=SimpleNamespace(metrics={"CTR": object()})),
    )
    intent = ChatIntent(
        question="Plot daily CTR by channel",
        metric="CTR",
        response="text",
        group_by=[],
        filters={},
        grain="summary",
    )

    def fake_plan(
        received_settings: AICallSettings,
        catalog: object,
        prompt: str,
        *,
        history: list[dict[str, object]],
        chat_config: dict[str, object],
    ) -> tuple[ChatIntent, str]:
        captured["settings"] = received_settings
        captured["catalog"] = catalog
        captured["prompt"] = prompt
        captured["history"] = history
        captured["chat_config"] = chat_config
        return intent, '{"metric":"CTR"}'

    def fake_execute(
        workspace: str, catalog: object, received_intent: ChatIntent
    ) -> ChatQueryResult:
        captured["workspace"] = workspace
        captured["execute_catalog"] = catalog
        captured["intent"] = received_intent
        return ChatQueryResult(
            intent=received_intent,
            rows=pl.DataFrame({"CTR": [0.12]}),
            query_summary="query_metric(workspace, 'CTR', grain='summary')",
            freshness="fresh",
        )

    monkeypatch.setattr(chat.st, "session_state", session_state)
    monkeypatch.setattr(chat.components, "render_page_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_sidebar_controls", lambda ctx: {"settings": settings})
    monkeypatch.setattr(chat, "_render_history", lambda: None)
    monkeypatch.setattr(chat, "_render_starter_questions", lambda ctx: None)
    monkeypatch.setattr(chat, "_render_pin_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.st, "chat_input", lambda label, **kwargs: "Plot daily CTR by channel")
    monkeypatch.setattr(chat.st, "chat_message", _null_context)
    monkeypatch.setattr(chat.st, "spinner", _null_context)
    monkeypatch.setattr(chat.st, "status", _null_context)
    monkeypatch.setattr(chat.st, "markdown", rendered.append)
    monkeypatch.setattr(chat.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "plan_chat_intent", fake_plan)
    monkeypatch.setattr(chat, "execute_chat_intent", fake_execute)

    chat.render(ctx)  # type: ignore[arg-type]

    assert captured["settings"] is settings
    assert captured["prompt"] == "Plot daily CTR by channel"
    assert captured["history"] == []
    assert "agent_prompt" in captured["chat_config"]
    assert captured["workspace"] == "/tmp/workspace"
    assert captured["intent"] is intent
    assert session_state.vs_chat_messages[0] == {
        "role": "user",
        "content": "Plot daily CTR by channel",
    }
    assert session_state.vs_chat_messages[1]["content"].startswith(
        "`CTR` returned 1 aggregate row(s)"
    )
    assert rendered[0] == "Plot daily CTR by channel"


@pytest.mark.unit
def test_chat_figure_splits_time_series_by_grouped_dimension() -> None:
    rows = pl.DataFrame(
        {
            "Month": ["2026-02", "2026-01", "2026-02", "2026-01"],
            "Channel": ["Web", "Web", "Email", "Email"],
            "VS_Engagement_Rate": [0.08, 0.07, 0.09, 0.06],
        }
    )
    intent = ChatIntent(
        question="Show engagement rate monthly by channel.",
        metric="VS_Engagement_Rate",
        response="chart",
        group_by=["Channel"],
        filters={},
        grain="monthly",
        chart=ChartIntent(
            kind="line",
            x="Month",
            y="VS_Engagement_Rate",
            color="Channel",
            facet_col=None,
        ),
    )

    figure = chat._chat_figure(rows, intent, {})

    assert {trace.name for trace in figure.data} == {"Email", "Web"}
    assert all(len(trace.x) == 2 for trace in figure.data)


@pytest.mark.unit
def test_chat_figure_falls_back_when_factory_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = pl.DataFrame({"Channel": ["Web", "Email"], "VS_Engagement_Rate": [0.08, 0.06]})
    intent = ChatIntent(
        question="Compare engagement rate by channel.",
        metric="VS_Engagement_Rate",
        response="chart",
        group_by=["Channel"],
        filters={},
        grain="summary",
        chart=ChartIntent(kind="bar", x="Channel", y="VS_Engagement_Rate"),
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("factory failure")

    monkeypatch.setattr(chat, "render_chart", boom)

    figure = chat._chat_figure(rows, intent, {})

    assert figure.data  # px fallback produced a bar chart


@pytest.mark.unit
def test_format_metric_value_applies_value_format() -> None:
    assert chat._format_metric_value(0.1234, "percent") == "12.34%"
    assert chat._format_metric_value(1234.5, "integer") == "1,234"
    assert chat._format_metric_value(1234.5, "currency") == "$1,234.50"
    assert chat._format_metric_value(None, "percent") == "n/a"


@pytest.mark.unit
def test_respond_from_result_builds_kpi_message_for_single_row() -> None:
    result = ChatQueryResult(
        intent=ChatIntent(
            question="What is the overall engagement rate?",
            metric="VS_Engagement_Rate",
            response="chart",
            group_by=[],
            filters={},
            grain="summary",
            chart=ChartIntent(kind="kpi_card", y="VS_Engagement_Rate", value_format="percent"),
        ),
        rows=pl.DataFrame({"VS_Engagement_Rate": [0.0825]}),
        query_summary="query_metric(...)",
        freshness="fresh",
    )

    message = chat._respond_from_result(result, raw_intent="{}")

    assert message["type"] == "kpi"
    assert message["kpi_label"] == "VS_Engagement_Rate"
    assert message["kpi_value"] == "8.25%"


@pytest.mark.unit
def test_respond_from_result_builds_chart_message_via_factory() -> None:
    result = ChatQueryResult(
        intent=ChatIntent(
            question="Engagement rate by channel over months",
            metric="VS_Engagement_Rate",
            response="chart",
            group_by=["Channel"],
            filters={},
            grain="monthly",
            chart=ChartIntent(kind="line", x="Month", y="VS_Engagement_Rate", color="Channel"),
        ),
        rows=pl.DataFrame(
            {
                "Month": ["2026-01", "2026-02", "2026-01", "2026-02"],
                "Channel": ["Web", "Web", "Email", "Email"],
                "VS_Engagement_Rate": [0.07, 0.08, 0.06, 0.09],
            }
        ),
        query_summary="query_metric(...)",
        freshness="fresh",
    )

    message = chat._respond_from_result(result, raw_intent="{}", theme={})

    assert message["type"] == "chart"
    assert message["figure"].data
    assert message["data"].height == 4
