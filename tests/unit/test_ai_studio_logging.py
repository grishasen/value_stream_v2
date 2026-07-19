"""Privacy regression tests for AI provider call logging."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from valuestream.ai import studio as ai_studio
from valuestream.ai.studio import (
    AICallSettings,
    AIProviderCallError,
    AIProviderFailureCategory,
    call_litellm,
)
from valuestream.ui.pages import ai_config_studio as ai_config_studio_page


class _ProviderSignalError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@pytest.mark.unit
def test_candidate_parse_failure_logs_only_safe_attempt_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_response = (
        "metrics:\n  metrics: [\n"
        "    PRIVATE-CUSTOMER-42 /Users/alice/private.csv sk-response-secret"
    )
    caplog.set_level(logging.DEBUG, logger=ai_studio.__name__)

    result = ai_studio.generate_validated_candidate(
        base_draft={},
        prompt="Prompt contains PRIVATE-PROMPT-CUSTOMER and sk-prompt-secret",
        call=lambda _prompt: raw_response,
        repair_prompt=lambda _draft, _issues, _trace: "PRIVATE-REPAIR-PROMPT",
        max_repairs=0,
        operation="catalog_draft",
    )

    assert not result.ok
    assert f"reference={result.reference}" in caplog.text
    assert "operation=catalog_draft" in caplog.text
    assert "attempt=1 role=generation stage=parse status=failed" in caplog.text
    assert f"response_chars={len(raw_response)}" in caplog.text
    assert "sections=none issue_count=1 issue_areas=other:1" in caplog.text
    assert "error_type=ParserError" in caplog.text
    assert "AI draft candidate exhausted" in caplog.text
    assert "final_stage=parse issue_count=1 issue_areas=other:1" in caplog.text
    assert "PRIVATE-CUSTOMER-42" not in caplog.text
    assert "/Users/alice/private.csv" not in caplog.text
    assert "sk-response-secret" not in caplog.text
    assert "PRIVATE-PROMPT-CUSTOMER" not in caplog.text
    assert "sk-prompt-secret" not in caplog.text
    assert "PRIVATE-REPAIR-PROMPT" not in caplog.text


@pytest.mark.unit
def test_candidate_validation_failure_never_logs_issue_or_response_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_response = """\
metrics:
  metrics:
    PRIVATE-METRIC-ID:
      source: PRIVATE-SOURCE-ID
"""
    issues = [
        "processors.processors.0.PRIVATE-ID: invalid /Users/alice/private.yaml",
        "source 'PRIVATE-SOURCE-ID' references stale raw field 'secret_name'",
    ]
    caplog.set_level(logging.DEBUG, logger=ai_studio.__name__)

    result = ai_studio.generate_validated_candidate(
        base_draft={},
        prompt="Configure PRIVATE-CUSTOMER-42",
        call=lambda _prompt: raw_response,
        repair_prompt=lambda _draft, _issues, _trace: "repair",
        max_repairs=0,
        validate=lambda _candidate: (False, issues),
        operation="copilot_filter",
    )

    assert not result.ok
    assert f"reference={result.reference}" in caplog.text
    assert "operation=copilot_filter" in caplog.text
    assert "attempt=1 role=generation stage=validation status=failed" in caplog.text
    assert "sections=metrics issue_count=2" in caplog.text
    assert "issue_areas=processor:1,field_contract:1" in caplog.text
    assert "final_stage=validation issue_count=2" in caplog.text
    assert "PRIVATE-CUSTOMER-42" not in caplog.text
    assert "PRIVATE-METRIC-ID" not in caplog.text
    assert "PRIVATE-SOURCE-ID" not in caplog.text
    assert "PRIVATE-ID" not in caplog.text
    assert "/Users/alice/private.yaml" not in caplog.text
    assert "secret_name" not in caplog.text


@pytest.mark.unit
def test_candidate_repair_logs_share_reference_without_catalog_identifiers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = iter(
        [
            "metrics: {metrics: {PRIVATE-REJECTED-METRIC: {}}}",
            "metrics: {metrics: {PRIVATE-ACCEPTED-METRIC: {}}}",
        ]
    )
    validation_calls = 0

    def validate(_candidate: dict) -> tuple[bool, list[str]]:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return False, ["metrics.PRIVATE-REJECTED-METRIC: invalid PRIVATE-VALUE"]
        return True, []

    caplog.set_level(logging.DEBUG, logger=ai_studio.__name__)
    result = ai_studio.generate_validated_candidate(
        base_draft={},
        prompt="PRIVATE-INITIAL-PROMPT",
        call=lambda _prompt: next(responses),
        repair_prompt=lambda _draft, _issues, _trace: "PRIVATE-REPAIR-PROMPT",
        max_repairs=1,
        validate=validate,
        operation="catalog_draft",
    )

    assert result.ok
    assert caplog.text.count(f"reference={result.reference}") == 3
    assert "attempt=1 role=generation stage=validation status=failed" in caplog.text
    assert "attempt=2 role=repair stage=validated status=success" in caplog.text
    assert "AI draft candidate exhausted" not in caplog.text
    assert "PRIVATE-REJECTED-METRIC" not in caplog.text
    assert "PRIVATE-ACCEPTED-METRIC" not in caplog.text
    assert "PRIVATE-VALUE" not in caplog.text
    assert "PRIVATE-INITIAL-PROMPT" not in caplog.text
    assert "PRIVATE-REPAIR-PROMPT" not in caplog.text


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
        raise _ProviderSignalError(
            "request=PRIVATE-CUSTOMER-42; key=sk-provider-secret",
            status_code=403,
        )

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)

    with pytest.raises(RuntimeError, match="insufficient permissions") as error:
        call_litellm(AICallSettings(model="openai/gpt-5.1"), "Safe prompt")

    assert error.value.__context__ is None
    assert "PRIVATE-CUSTOMER-42" not in str(error.value)
    assert "sk-provider-secret" not in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("signal", "expected_category", "expected_retryable"),
    [
        (400, AIProviderFailureCategory.CONFIGURATION, False),
        (401, AIProviderFailureCategory.AUTHENTICATION, False),
        (403, AIProviderFailureCategory.AUTHORIZATION, False),
        (429, AIProviderFailureCategory.RATE_LIMIT, True),
        ("timeout", AIProviderFailureCategory.TIMEOUT, True),
        ("network", AIProviderFailureCategory.NETWORK, True),
        (503, AIProviderFailureCategory.PROVIDER, True),
        ("unknown", AIProviderFailureCategory.INTERNAL, False),
    ],
    ids=[
        "configuration-400",
        "authentication-401",
        "authorization-403",
        "rate-limit-429",
        "timeout-type",
        "network-type",
        "provider-503",
        "unknown-type",
    ],
)
def test_call_litellm_classifies_provider_failures_without_payloads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    signal: int | str,
    expected_category: AIProviderFailureCategory,
    expected_retryable: bool,
) -> None:
    secret = "PRIVATE-CUSTOMER-42 sk-provider-secret /Users/alice/provider.log"
    if isinstance(signal, int):
        provider_error: Exception = _ProviderSignalError(secret, status_code=signal)
    elif signal == "timeout":
        provider_error = TimeoutError(secret)
    elif signal == "network":
        provider_error = ConnectionError(secret)
    else:
        provider_error = RuntimeError(secret)

    def fake_completion(**kwargs: object) -> object:
        raise provider_error

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    with pytest.raises(AIProviderCallError) as raised:
        call_litellm(
            AICallSettings(model="openai/gpt-5.1", api_key="sk-provider-secret"),
            "Prompt with PRIVATE-CUSTOMER-42",
        )

    failure = raised.value
    assert failure.category is expected_category
    assert failure.retryable is expected_retryable
    assert failure.permission_denied is (
        expected_category
        in {
            AIProviderFailureCategory.AUTHENTICATION,
            AIProviderFailureCategory.AUTHORIZATION,
        }
    )
    assert failure.__context__ is None
    assert re.fullmatch(r"[0-9a-f]{12}", failure.call_id)
    assert f"call_id={failure.call_id}" in caplog.text
    assert f"failure_category={expected_category}" in caplog.text
    assert f"retryable={expected_retryable}" in caplog.text
    assert "PRIVATE-CUSTOMER-42" not in str(failure)
    assert "sk-provider-secret" not in str(failure)
    assert "/Users/alice/provider.log" not in str(failure)
    assert "PRIVATE-CUSTOMER-42" not in caplog.text
    assert "sk-provider-secret" not in caplog.text
    assert "/Users/alice/provider.log" not in caplog.text


@pytest.mark.unit
def test_call_litellm_classifies_provider_error_code_without_message_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_completion(**kwargs: object) -> object:
        raise _ProviderSignalError(
            "not a classification hint: PRIVATE-CUSTOMER-42",
            code="rate-limit-exceeded",
        )

    monkeypatch.setattr(ai_studio, "litellm_completion", fake_completion)

    with pytest.raises(AIProviderCallError) as raised:
        call_litellm(AICallSettings(model="openai/gpt-5.1"), "Safe prompt")

    assert raised.value.category is AIProviderFailureCategory.RATE_LIMIT
    assert raised.value.retryable is True
    assert "PRIVATE-CUSTOMER-42" not in str(raised.value)


@pytest.mark.unit
def test_call_litellm_classifies_invalid_response_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ai_studio, "litellm_completion", lambda **kwargs: {"choices": []})

    with pytest.raises(AIProviderCallError) as raised:
        call_litellm(AICallSettings(model="openai/gpt-5.1"), "Safe prompt")

    assert raised.value.category is AIProviderFailureCategory.RESPONSE_VALIDATION
    assert raised.value.retryable is True


@pytest.mark.unit
def test_ai_provider_call_error_legacy_constructor_remains_supported() -> None:
    failure = AIProviderCallError(
        call_id="legacy-call",
        error_type="RuntimeError",
        permission_denied=False,
    )

    assert failure.category is AIProviderFailureCategory.PROVIDER
    assert failure.retryable is True
    assert str(failure) == "AI provider call failed (RuntimeError). Reference: legacy-call."


@pytest.mark.unit
def test_provider_preflight_rejects_missing_configuration_without_calling_provider() -> None:
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        raise AssertionError("provider must not be called for missing local configuration")

    with pytest.raises(AIProviderCallError) as raised:
        ai_studio.preflight_ai_provider(
            AICallSettings(model="openai/gpt-5.1"),
            call=fail_if_called,
        )

    assert called is False
    assert raised.value.category is AIProviderFailureCategory.CONFIGURATION
    assert raised.value.retryable is False
    assert re.fullmatch(r"[0-9a-f]{12}", raised.value.call_id)


@pytest.mark.unit
def test_provider_preflight_uses_independent_five_second_ready_call() -> None:
    captured: dict[str, object] = {}

    def ready_call(
        settings: AICallSettings,
        prompt: str,
        **kwargs: object,
    ) -> str:
        captured["settings"] = settings
        captured["prompt"] = prompt
        captured.update(kwargs)
        return " READY\n"

    settings = AICallSettings(
        model="ollama/llama3.1",
        custom_llm_provider="ollama",
        timeout_seconds=90,
    )
    receipt = ai_studio.preflight_ai_provider(settings, call=ready_call)

    assert settings.timeout_seconds == 90
    assert isinstance(captured["settings"], AICallSettings)
    assert captured["settings"].timeout_seconds == 5
    assert captured["prompt"] == "Reply with READY."
    assert "Do not request or infer data" in str(captured["system_prompt"])
    assert receipt.timeout_seconds == 5
    assert re.fullmatch(r"[0-9a-f]{12}", receipt.reference)


@pytest.mark.unit
def test_provider_preflight_classifies_unexpected_reply_without_logging_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=ai_studio.__name__)

    with pytest.raises(AIProviderCallError) as raised:
        ai_studio.preflight_ai_provider(
            AICallSettings(model="ollama/llama3.1", custom_llm_provider="ollama"),
            call=lambda *args, **kwargs: "PRIVATE-CUSTOMER-42 is not READY",
        )

    assert raised.value.category is AIProviderFailureCategory.RESPONSE_VALIDATION
    assert raised.value.retryable is True
    assert "PRIVATE-CUSTOMER-42" not in str(raised.value)
    assert "PRIVATE-CUSTOMER-42" not in caplog.text


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
