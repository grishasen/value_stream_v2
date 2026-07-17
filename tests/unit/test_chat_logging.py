"""Privacy regression tests for governed chat logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from valuestream.ai import chat as ai_chat
from valuestream.ai.chat import ChatIntentPlanningError
from valuestream.ai.studio import AICallSettings
from valuestream.config.loader import load
from valuestream.ui.pages import chat as chat_page


@pytest.mark.unit
def test_chat_planning_logs_only_safe_operational_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = load(Path("examples/demo"))
    question_secret = "Show results for customer PRIVATE-CUSTOMER-42"
    response_secret = "MODEL-RESPONSE-SECRET"
    provider_secret = "private-provider-name"
    api_base_secret = "http://user:password@localhost:11434/v1?token=private"
    raw_response = json.dumps(
        {
            "metric": "VS_Engagement_Rate",
            "response": "text",
            "group_by": [],
            "filters": {},
            "private_note": response_secret,
        }
    )
    monkeypatch.setattr(ai_chat, "call_litellm", lambda *args, **kwargs: raw_response)
    caplog.set_level(logging.DEBUG, logger=ai_chat.__name__)

    intent, raw = ai_chat.plan_chat_intent(
        AICallSettings(
            model="/Users/alice/models/private.gguf",
            api_base=api_base_secret,
            custom_llm_provider=provider_secret,
        ),
        catalog,
        question_secret,
        history=[{"role": "user", "content": "HISTORY-SECRET"}],
        sql_schema="CREATE TABLE private_customer_table (...)",
    )

    assert intent.metric == "VS_Engagement_Rate"
    assert raw == raw_response
    assert "model=<redacted-model>" in caplog.text
    assert "has_custom_llm_provider=True" in caplog.text
    assert "has_api_base=True" in caplog.text
    assert "sql_enabled=True" in caplog.text
    assert "history_message_count=1" in caplog.text
    assert "intent_type=text" in caplog.text
    assert "metric=VS_Engagement_Rate" in caplog.text
    assert "grain=summary" in caplog.text
    assert "group_count=0" in caplog.text
    for secret in (
        question_secret,
        response_secret,
        provider_secret,
        api_base_secret,
        "HISTORY-SECRET",
        "private_customer_table",
        "/Users/alice",
    ):
        assert secret not in caplog.text


@pytest.mark.unit
def test_non_query_intent_logs_omit_generated_clarification_and_sql(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = load(Path("examples/demo"))
    clarification_secret = "Should I filter for alice@example.com?"
    sql_secret = "SELECT * FROM private_customer_table WHERE token = 'SECRET-42'"
    caplog.set_level(logging.DEBUG, logger=ai_chat.__name__)

    clarification = ai_chat.parse_chat_intent(
        json.dumps({"response": "clarify", "clarify": clarification_secret}),
        catalog,
        question="QUESTION-SECRET",
    )
    sql = ai_chat.parse_chat_intent(
        json.dumps({"response": "sql", "sql": sql_secret}),
        catalog,
        question="ANOTHER-QUESTION-SECRET",
        allow_sql=True,
    )

    assert clarification.clarify == clarification_secret
    assert sql.sql == sql_secret
    assert "intent_type=clarify" in caplog.text
    assert "intent_type=sql" in caplog.text
    for secret in (
        clarification_secret,
        sql_secret,
        "alice@example.com",
        "private_customer_table",
        "SECRET-42",
        "QUESTION-SECRET",
        "ANOTHER-QUESTION-SECRET",
    ):
        assert secret not in caplog.text


@pytest.mark.unit
def test_invalid_generated_chart_fields_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = load(Path("examples/demo"))
    chart_secret = "PRIVATE-CHART-FIELD-42"
    caplog.set_level(logging.DEBUG, logger=ai_chat.__name__)

    intent = ai_chat.parse_chat_intent(
        json.dumps(
            {
                "metric": "VS_Engagement_Rate",
                "response": "chart",
                "group_by": [],
                "filters": {},
                "chart": {
                    "kind": chart_secret,
                    "x": chart_secret,
                    "y": "VS_Engagement_Rate",
                    "color": chart_secret,
                },
            }
        ),
        catalog,
    )

    assert intent.chart is not None
    assert "Chat chart kind not allowed for metric" in caplog.text
    assert "Ignoring invalid chart x-axis" in caplog.text
    assert "Ignoring non-dimension chart field" in caplog.text
    assert chart_secret not in caplog.text


@pytest.mark.unit
def test_invalid_model_intent_is_sanitized_before_caller_logging(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    catalog = load(Path("examples/demo"))
    generated_secret = "PRIVATE-CUSTOMER-42"
    monkeypatch.setattr(
        ai_chat,
        "call_litellm",
        lambda *args, **kwargs: json.dumps(
            {
                "metric": generated_secret,
                "response": "text",
                "group_by": [generated_secret],
            }
        ),
    )
    caplog.set_level(logging.INFO)

    with pytest.raises(ChatIntentPlanningError) as error:
        ai_chat.plan_chat_intent(
            AICallSettings(model="openai/gpt-5.1"),
            catalog,
            "Show engagement",
        )

    chat_page._log_chat_query_failure(error.value)
    message = chat_page._chat_query_error_message(error.value)

    assert error.value.__context__ is None
    assert "status=invalid_model_intent" in caplog.text
    assert "error_type=ChatIntentPlanningError" in caplog.text
    assert generated_secret not in caplog.text
    assert generated_secret not in str(error.value)
    assert generated_secret not in message


@pytest.mark.unit
def test_chat_page_omits_arbitrary_failure_text_from_logs_and_user_copy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    generated_secret = "unknown column PRIVATE-CUSTOMER-42"
    error = ValueError(generated_secret)
    caplog.set_level(logging.INFO, logger=chat_page.__name__)

    chat_page._log_chat_query_failure(error)
    message = chat_page._chat_query_error_message(error)

    assert "error_type=ValueError" in caplog.text
    assert generated_secret not in caplog.text
    assert generated_secret not in message
