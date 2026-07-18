"""Chat page rendering tests."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import polars as pl
import pytest

from valuestream.ai import AICallSettings
from valuestream.ai.chat import (
    ChartIntent,
    ChatIntent,
    ChatQueryResult,
    DeterministicChatStarter,
)
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
    monkeypatch.setattr(
        chat, "_render_deterministic_starter_questions", lambda *args, **kwargs: None
    )
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

    assert info_messages == [
        "No-model mode is active. The aggregate quick questions below still work; "
        "configure and enable the LLM planner for free-form questions."
    ]
    assert chat_inputs == [
        {"label": "Enable the LLM planner for free-form questions", "disabled": True}
    ]
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
    monkeypatch.setattr(
        chat, "_render_deterministic_starter_questions", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(chat, "_render_starter_questions", lambda ctx: None)
    monkeypatch.setattr(chat, "_render_pin_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.st, "chat_input", lambda label, **kwargs: "Plot daily CTR by channel")
    monkeypatch.setattr(chat.st, "chat_message", _null_context)
    monkeypatch.setattr(chat.st, "spinner", _null_context)
    monkeypatch.setattr(chat.st, "status", _null_context)
    monkeypatch.setattr(chat.st, "markdown", rendered.append)
    monkeypatch.setattr(chat.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_preflight_chat_provider", lambda settings: None)
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
@pytest.mark.parametrize(
    "settings",
    [
        None,
        AICallSettings(
            model="unavailable/provider",
            api_base="http://localhost:1",
            custom_llm_provider="openai",
        ),
    ],
)
def test_render_runs_deterministic_starter_without_planner(
    monkeypatch: pytest.MonkeyPatch,
    settings: AICallSettings | None,
) -> None:
    session_state = _SessionState()
    captured: dict[str, object] = {}
    intent = ChatIntent(
        question="What is the total interactions?",
        metric="VS_Interactions",
        response="text",
        group_by=[],
        filters={},
        grain="summary",
    )
    starter = DeterministicChatStarter(
        key="count",
        question=intent.question,
        intent=intent,
    )
    ctx = SimpleNamespace(
        workspace="/tmp/workspace",
        catalog=SimpleNamespace(metrics=SimpleNamespace(metrics={"VS_Interactions": object()})),
    )

    def fake_execute(
        workspace: str,
        catalog: object,
        received_starter: DeterministicChatStarter,
    ) -> ChatQueryResult:
        captured["workspace"] = workspace
        captured["catalog"] = catalog
        captured["starter"] = received_starter
        return ChatQueryResult(
            intent=intent,
            rows=pl.DataFrame({"VS_Interactions": [192]}),
            query_summary="query_metric(workspace, 'VS_Interactions', grain='summary')",
            freshness="fresh",
        )

    def fail_plan(*args: object, **kwargs: object) -> None:
        raise AssertionError("deterministic starter must bypass the LLM planner")

    def fail_preflight(*args: object, **kwargs: object) -> None:
        raise AssertionError("deterministic starter must bypass provider preflight")

    monkeypatch.setattr(chat.st, "session_state", session_state)
    monkeypatch.setattr(chat.components, "render_page_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_sidebar_controls", lambda ctx: {"settings": settings})
    monkeypatch.setattr(chat, "_render_history", lambda: None)
    monkeypatch.setattr(
        chat,
        "_render_deterministic_starter_questions",
        lambda *args, **kwargs: starter,
    )
    monkeypatch.setattr(chat, "_render_starter_questions", lambda ctx: None)
    monkeypatch.setattr(chat, "load_chat_with_data_config", lambda workspace: (None, {}))
    monkeypatch.setattr(chat, "_render_pin_control", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.st, "chat_input", lambda *args, **kwargs: "ignored free form")
    monkeypatch.setattr(chat.st, "chat_message", _null_context)
    monkeypatch.setattr(chat.st, "spinner", _null_context)
    monkeypatch.setattr(chat.st, "status", _null_context)
    monkeypatch.setattr(chat.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_preflight_chat_provider", fail_preflight)
    monkeypatch.setattr(chat, "plan_chat_intent", fail_plan)
    monkeypatch.setattr(chat, "execute_deterministic_chat_query", fake_execute)

    chat.render(ctx)  # type: ignore[arg-type]

    assert captured == {
        "workspace": "/tmp/workspace",
        "catalog": ctx.catalog,
        "starter": starter,
    }
    assert session_state.vs_chat_messages[0] == {
        "role": "user",
        "content": "What is the total interactions?",
    }
    assert session_state.vs_chat_messages[1]["content"].startswith(
        "Deterministic aggregate template; no model was used."
    )
    assert session_state.vs_chat_messages[1]["deterministic_key"] == "count"


@pytest.mark.unit
def test_chat_preflight_negative_cache_and_retry_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _SessionState()
    calls = 0
    settings = AICallSettings(
        model="ollama/llama3.1",
        custom_llm_provider="ollama",
    )

    def provider_preflight(received: AICallSettings) -> object:
        nonlocal calls
        calls += 1
        assert received is settings
        if calls == 1:
            raise chat.AIProviderCallError(
                call_id="provider-ref",
                error_type="TimeoutError",
                permission_denied=False,
                category="timeout",
            )
        return SimpleNamespace(reference="ready-ref")

    monkeypatch.setattr(chat.st, "session_state", session_state)
    monkeypatch.setattr(chat, "preflight_ai_provider", provider_preflight)

    with pytest.raises(chat.AIProviderCallError) as first:
        chat._preflight_chat_provider(settings)
    with pytest.raises(chat.AIProviderCallError) as cached:
        chat._preflight_chat_provider(settings)

    session_state[chat._CHAT_PREFLIGHT_FORCE_RETRY_STATE_KEY] = True
    chat._preflight_chat_provider(settings)
    chat._preflight_chat_provider(settings)

    assert calls == 2
    assert first.value.call_id == "provider-ref"
    assert cached.value.call_id == "provider-ref"
    assert str(cached.value.category) == "timeout"


@pytest.mark.unit
def test_chat_preflight_failure_preserves_existing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_history = [{"role": "assistant", "type": "text", "content": "Earlier answer"}]
    session_state = _SessionState(vs_chat_messages=list(existing_history))
    errors: list[str] = []
    settings = AICallSettings(
        model="ollama/llama3.1",
        custom_llm_provider="ollama",
    )
    ctx = SimpleNamespace(
        workspace="/tmp/workspace",
        catalog=SimpleNamespace(metrics=SimpleNamespace(metrics={"CTR": object()})),
    )

    def fail_preflight(received: AICallSettings) -> None:
        assert received is settings
        raise chat.AIProviderCallError(
            call_id="timeout-ref",
            error_type="TimeoutError",
            permission_denied=False,
            category="timeout",
        )

    def fail_plan(*args: object, **kwargs: object) -> None:
        raise AssertionError("planner must not run after failed preflight")

    monkeypatch.setattr(chat.st, "session_state", session_state)
    monkeypatch.setattr(chat.components, "render_page_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.components, "key_value_strip", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat, "_sidebar_controls", lambda ctx: {"settings": settings})
    monkeypatch.setattr(chat, "_render_history", lambda: None)
    monkeypatch.setattr(
        chat, "_render_deterministic_starter_questions", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(chat, "_render_starter_questions", lambda ctx: None)
    monkeypatch.setattr(chat, "load_chat_with_data_config", lambda workspace: (None, {}))
    monkeypatch.setattr(chat.st, "chat_input", lambda *args, **kwargs: "Show CTR")
    monkeypatch.setattr(chat.st, "chat_message", _null_context)
    monkeypatch.setattr(chat.st, "spinner", _null_context)
    monkeypatch.setattr(chat.st, "expander", _null_context)
    monkeypatch.setattr(chat.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat.st, "error", errors.append)
    monkeypatch.setattr(chat.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(chat, "_preflight_chat_provider", fail_preflight)
    monkeypatch.setattr(chat, "plan_chat_intent", fail_plan)

    chat.render(ctx)  # type: ignore[arg-type]

    assert session_state.vs_chat_messages == existing_history
    assert len(errors) == 1
    assert "timeout failure" in errors[0]
    assert "timeout-ref" in errors[0]


@pytest.mark.unit
def test_chat_narrative_reuses_preflight_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AICallSettings(
        model="ollama/llama3.1",
        custom_llm_provider="ollama",
    )
    intent = ChatIntent(
        question="Show CTR",
        metric="CTR",
        response="text",
        group_by=[],
        filters={},
        grain="summary",
    )
    result = ChatQueryResult(
        intent=intent,
        rows=pl.DataFrame({"CTR": [0.12]}),
        query_summary="query",
        freshness="fresh",
    )
    events: list[str] = []

    monkeypatch.setattr(chat.st, "spinner", _null_context)
    monkeypatch.setattr(
        chat,
        "_preflight_chat_provider",
        lambda received: events.append("preflight") if received is settings else None,
    )
    monkeypatch.setattr(
        chat,
        "narrate_chat_result",
        lambda received, value: events.append("narrate") or "Grounded summary.",
    )

    response = chat._maybe_narrate(
        {"role": "assistant", "type": "text", "content": "Base answer."},
        result,
        settings,
        {"narrative": True},
    )

    assert events == ["preflight", "narrate"]
    assert response["content"].startswith("Grounded summary.")


@pytest.mark.unit
def test_deterministic_starter_controls_label_scope_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = ChatIntent(
        question="What is the total interactions?",
        metric="VS_Interactions",
        response="text",
        group_by=[],
        filters={},
        grain="summary",
    )
    starter = DeterministicChatStarter("count", intent.question, intent)
    expanders: list[tuple[str, bool]] = []
    captions: list[str] = []

    @contextmanager
    def fake_expander(label: str, *, expanded: bool):
        expanders.append((label, expanded))
        yield

    class _Column:
        def button(self, label: str, **kwargs: object) -> bool:
            return label == starter.question

    monkeypatch.setattr(chat, "deterministic_chat_starters", lambda catalog: [starter])
    monkeypatch.setattr(chat.st, "expander", fake_expander)
    monkeypatch.setattr(chat.st, "caption", captions.append)
    monkeypatch.setattr(chat.st, "columns", lambda count: [_Column() for _ in range(count)])

    selected = chat._render_deterministic_starter_questions(
        SimpleNamespace(catalog=object()),  # type: ignore[arg-type]
        expanded=True,
    )

    assert selected is starter
    assert expanders == [("Aggregate quick questions · no model required", True)]
    assert "configured totals" in captions[0]
    assert "Enable the LLM planner" in captions[0]
    assert "filters, comparisons, and follow-up questions" in captions[0]


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
