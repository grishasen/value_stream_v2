"""Privacy regression tests for AI provider call logging."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from valuestream.ai import studio as ai_studio
from valuestream.ai.studio import AICallSettings, call_litellm
from valuestream.ui.pages import ai_config_studio as ai_config_studio_page


@pytest.mark.unit
def test_call_litellm_logs_only_safe_operational_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response_body = "metrics:\n  private_customer_email: alice@example.com"

    def fake_completion(**kwargs: object) -> object:
        return {
            "id": "provider-call-123",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": response_body},
                }
            ],
            "usage": {
                "prompt_tokens": 41,
                "completion_tokens": 17,
                "total_tokens": 58,
            },
        }

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    result = call_litellm(
        AICallSettings(
            model="openai/gpt-5.1",
            api_key="sk-private-credential",
            api_base="http://user:password@localhost:11434/v1?token=private",
            custom_llm_provider="openai",
        ),
        "Analyze alice@example.com from /Users/alice/private/sample.csv",
        system_prompt="Use customer sample value ACME-SECRET",
    )

    assert result == response_body
    assert re.search(r"call_id=[0-9a-f]{12}", caplog.text)
    assert "openai/gpt-5.1" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "'status': 'stop'" in caplog.text
    assert "'prompt_tokens': 41" in caplog.text
    assert "'completion_tokens': 17" in caplog.text
    assert "'total_tokens': 58" in caplog.text
    assert "alice@example.com" not in caplog.text
    assert "/Users/alice/private/sample.csv" not in caplog.text
    assert "ACME-SECRET" not in caplog.text
    assert "private_customer_email" not in caplog.text
    assert "sk-private-credential" not in caplog.text
    assert "user:password" not in caplog.text
    assert "token=private" not in caplog.text


@pytest.mark.unit
def test_call_litellm_redacts_local_model_and_provider_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        raise RuntimeError(
            f"provider echoed request={kwargs['messages']} "
            f"api_key={kwargs['api_key']} path=/Users/alice/provider.log"
        )

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    with pytest.raises(RuntimeError, match=r"AI provider call failed \(RuntimeError\)") as error:
        call_litellm(
            AICallSettings(
                model="/Users/alice/models/private.gguf",
                api_key="sk-provider-secret",
            ),
            "Customer sample value: PRIVATE-CUSTOMER-42",
        )

    assert error.value.__context__ is None
    assert "provider echoed request" not in str(error.value)
    assert "PRIVATE-CUSTOMER-42" not in str(error.value)
    assert "sk-provider-secret" not in str(error.value)
    assert "LLM call started" in caplog.text
    assert "LLM call failed" in caplog.text
    assert "model=<redacted-model>" in caplog.text
    assert "status=error" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "/Users/alice" not in caplog.text
    assert "PRIVATE-CUSTOMER-42" not in caplog.text
    assert "sk-provider-secret" not in caplog.text
    assert "provider echoed request" not in caplog.text


@pytest.mark.unit
def test_call_litellm_does_not_log_unrecognized_provider_status(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        return {
            "choices": [
                {
                    "finish_reason": "private-customer-42",
                    "message": {"content": "ok"},
                }
            ]
        }

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    assert call_litellm(AICallSettings(model="openai/gpt-5.1"), "Safe prompt") == "ok"
    assert "'status': 'other'" in caplog.text
    assert "private-customer-42" not in caplog.text


@pytest.mark.unit
def test_call_litellm_preserves_permission_guidance_without_raw_provider_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        raise RuntimeError(
            "insufficient permissions; request=PRIVATE-CUSTOMER-42; key=sk-provider-secret"
        )

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)

    with pytest.raises(RuntimeError, match="insufficient permissions") as error:
        call_litellm(AICallSettings(model="openai/gpt-5.1"), "Safe prompt")

    assert error.value.__context__ is None
    assert "PRIVATE-CUSTOMER-42" not in str(error.value)
    assert "sk-provider-secret" not in str(error.value)


@pytest.mark.unit
def test_caller_exception_logging_cannot_recover_raw_provider_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        raise RuntimeError(
            f"provider echoed request={kwargs['messages']} key=sk-provider-secret "
            "path=/Users/alice/private.csv"
        )

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO)
    api_key = "sk-provider-secret"
    prompt = "PRIVATE-CUSTOMER-42"

    try:
        call_litellm(
            AICallSettings(model="openai/gpt-5.1", api_key=api_key),
            prompt,
        )
    except RuntimeError:
        logging.getLogger("test.ai.caller").exception("Caller model operation failed")

    assert "Caller model operation failed" in caplog.text
    assert "PRIVATE-CUSTOMER-42" not in caplog.text
    assert "sk-provider-secret" not in caplog.text
    assert "/Users/alice/private.csv" not in caplog.text


@pytest.mark.unit
def test_ui_model_failure_log_omits_provider_exception_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=ai_config_studio_page.__name__)

    ai_config_studio_page._log_ai_operation_failure(
        "AI draft generation",
        RuntimeError("provider echoed PRIVATE-CUSTOMER-42 from /Users/alice/sample.csv"),
    )

    assert "AI draft generation failed" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "PRIVATE-CUSTOMER-42" not in caplog.text
    assert "/Users/alice/sample.csv" not in caplog.text


@pytest.mark.unit
def test_malformed_sample_value_is_not_written_to_studio_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    secret = "PRIVATE-CUSTOMER-42"
    malformed = "Value\n" + "\n".join(str(value) for value in range(500)) + f"\n{secret}\n"
    (data_dir / "malformed.csv").write_text(malformed, encoding="utf-8")
    caplog.set_level(logging.INFO, logger=ai_config_studio_page.__name__)

    def app(workspace: str) -> None:
        from pathlib import Path  # noqa: PLC0415 - isolated AppTest source

        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_workspace_sample_active"] = "data/malformed.csv"
        st.session_state["ai_studio_sample_rows"] = 1_000
        frame = page._load_sample(Path(workspace), ai_calls_enabled=False)
        st.session_state["sample_load_failed"] = frame is None

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()

    assert not rendered.exception
    assert rendered.session_state["sample_load_failed"] is True
    assert "Sample read failed: error_type=ComputeError" in caplog.text
    assert secret not in caplog.text
    assert str(tmp_path) not in caplog.text
