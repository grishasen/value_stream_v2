"""Shared AI settings helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from valuestream.ai.settings import (
    AI_CONFIG_FILENAMES,
    configured_api_key,
    load_chat_with_data_config,
    load_llm_settings_config,
    write_chat_with_data_config,
    write_llm_settings_config,
)


@pytest.mark.unit
def test_ai_config_filenames_are_tuple() -> None:
    assert AI_CONFIG_FILENAMES == ("ai.yaml",)


@pytest.mark.unit
def test_load_llm_settings_config_ignores_workspace_directory(tmp_path: Path) -> None:
    config_path, config = load_llm_settings_config(tmp_path)

    assert config_path is None
    assert config == {}


@pytest.mark.unit
def test_load_llm_settings_config_reads_ai_yaml(tmp_path: Path) -> None:
    (tmp_path / "ai.yaml").write_text(
        """
ai:
  llm:
    model: ollama/llama3.1
    api_base: http://localhost:11434
""",
        encoding="utf-8",
    )

    config_path, config = load_llm_settings_config(tmp_path)

    assert config_path == tmp_path / "ai.yaml"
    assert config["model"] == "ollama/llama3.1"
    assert config["api_base"] == "http://localhost:11434"


@pytest.mark.unit
def test_load_chat_with_data_config_reads_prompt_and_descriptions(tmp_path: Path) -> None:
    (tmp_path / "ai.yaml").write_text(
        """
chat_with_data:
  agent_prompt: Use Pega CDH terminology.
  dataset_descriptions:
    ih: Interaction history aggregates.
  metric_descriptions:
    engagement: CTR and lift metrics.
""",
        encoding="utf-8",
    )

    config_path, config = load_chat_with_data_config(tmp_path)

    assert config_path == tmp_path / "ai.yaml"
    assert config["agent_prompt"] == "Use Pega CDH terminology."
    assert config["dataset_descriptions"] == {"ih": "Interaction history aggregates."}
    assert config["metric_descriptions"] == {"engagement": "CTR and lift metrics."}


@pytest.mark.unit
def test_write_chat_with_data_config_preserves_llm_settings(tmp_path: Path) -> None:
    (tmp_path / "ai.yaml").write_text(
        """
ai:
  llm:
    model: ollama/llama3.1
    api_base: http://localhost:11434
""",
        encoding="utf-8",
    )

    path = write_chat_with_data_config(
        tmp_path,
        agent_prompt="Use business language.",
        dataset_descriptions={"ih": "Interaction history."},
        metric_descriptions={"ih_engagement": "Engagement metrics."},
    )
    _, llm = load_llm_settings_config(tmp_path)
    _, chat = load_chat_with_data_config(tmp_path)

    assert path == tmp_path / "ai.yaml"
    assert llm["model"] == "ollama/llama3.1"
    assert chat["agent_prompt"] == "Use business language."
    assert chat["dataset_descriptions"] == {"ih": "Interaction history."}
    assert chat["metric_descriptions"] == {"ih_engagement": "Engagement metrics."}


@pytest.mark.unit
def test_write_llm_settings_config_persists_minimal_block(tmp_path: Path) -> None:
    path = write_llm_settings_config(
        tmp_path,
        model="openai/gpt-5.5",
        api_base="",
        custom_provider="",
        api_key_env="OPENAI_API_KEY",
        temperature=0.2,
        timeout_seconds=120,
    )
    _, llm = load_llm_settings_config(tmp_path)

    assert path == tmp_path / "ai.yaml"
    assert llm["model"] == "openai/gpt-5.5"
    assert llm["temperature"] == 0.2
    assert llm["timeout_seconds"] == 120
    assert llm["api_key_env"] == "OPENAI_API_KEY"
    # Empty optional values are not persisted.
    assert "api_base" not in llm
    assert "custom_provider" not in llm


@pytest.mark.unit
def test_write_llm_settings_config_preserves_chat_block(tmp_path: Path) -> None:
    write_chat_with_data_config(tmp_path, agent_prompt="Business language.")

    write_llm_settings_config(tmp_path, model="ollama/llama3.1", api_base="http://localhost:11434")
    _, chat = load_chat_with_data_config(tmp_path)
    _, llm = load_llm_settings_config(tmp_path)

    assert chat["agent_prompt"] == "Business language."
    assert llm["model"] == "ollama/llama3.1"
    assert llm["api_base"] == "http://localhost:11434"


@pytest.mark.unit
def test_configured_api_key_uses_configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(os.environ, "VALUESTREAM_TEST_KEY", "secret")

    assert configured_api_key({"api_key_env": "VALUESTREAM_TEST_KEY"}) == "secret"
