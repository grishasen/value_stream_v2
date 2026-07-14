"""Shared AI runtime settings helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from valuestream.utils.logger import get_logger

AI_CONFIG_FILENAMES = ("ai.yaml",)
DEFAULT_CHAT_AGENT_PROMPT = (
    "You are a data analysis agent. Your main goal is to help non-technical users to analyze aggregated interaction "
    "history and product holdings data from Pega Customer Decision Hub application. Help users answer questions from "
    "persisted aggregate metrics only. Use the catalog metadata and chat-only "
    "descriptions to choose the right metric, dimensions, filters, and chart shape."
)
logger = get_logger(__name__)


def load_llm_settings_config(workspace: str | Path) -> tuple[Path | None, dict[str, Any]]:
    """Load ``ai.llm`` settings from a workspace-local AI config file."""

    root = Path(workspace)
    logger.debug("Loading AI LLM settings: workspace=%s", root)
    for path in _candidate_config_paths(root):
        loaded = _read_ai_config(path)
        if loaded is None:
            continue
        if not isinstance(loaded, dict):
            return path, {}
        ai_config = loaded.get("ai", loaded)
        if not isinstance(ai_config, dict):
            logger.warning("Ignoring AI settings file with non-mapping ai section: path=%s", path)
            return path, {}
        llm_config = ai_config.get("llm", ai_config)
        if not isinstance(llm_config, dict):
            logger.warning("Ignoring AI settings file with non-mapping llm section: path=%s", path)
            return path, {}
        logger.info(
            "Loaded AI LLM settings: path=%s keys=%s",
            path,
            sorted(key for key in llm_config if key != "api_key"),
        )
        return path, llm_config
    logger.debug("No AI LLM settings file found: workspace=%s", root)
    return None, {}


def load_chat_with_data_config(workspace: str | Path) -> tuple[Path | None, dict[str, Any]]:
    """Load chat-only prompt and description settings from workspace ``ai.yaml``."""

    root = Path(workspace)
    logger.debug("Loading Chat With Data settings: workspace=%s", root)
    for path in _candidate_config_paths(root):
        loaded = _read_ai_config(path)
        if loaded is None:
            continue
        if not isinstance(loaded, dict):
            return path, _default_chat_with_data_config()
        chat_config = loaded.get("chat_with_data")
        if not isinstance(chat_config, dict):
            ai_config = loaded.get("ai")
            chat_config = ai_config.get("chat_with_data") if isinstance(ai_config, dict) else None
        if not isinstance(chat_config, dict):
            return path, _default_chat_with_data_config()
        return path, _normalize_chat_with_data_config(chat_config)
    logger.debug("No Chat With Data settings file found: workspace=%s", root)
    return None, _default_chat_with_data_config()


def write_chat_with_data_config(
    workspace: str | Path,
    *,
    agent_prompt: str,
    metric_descriptions: dict[str, str] | None = None,
    dataset_descriptions: dict[str, str] | None = None,
) -> Path:
    """Write chat-only prompt and description settings to workspace ``ai.yaml``."""

    root = Path(workspace)
    path = root / AI_CONFIG_FILENAMES[0]
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    loaded["chat_with_data"] = {
        "agent_prompt": agent_prompt.strip() or DEFAULT_CHAT_AGENT_PROMPT,
        "dataset_descriptions": _clean_description_map(dataset_descriptions or {}),
        "metric_descriptions": _clean_description_map(metric_descriptions or {}),
    }
    path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    return path


def write_llm_settings_config(
    workspace: str | Path,
    *,
    model: str,
    api_base: str = "",
    custom_provider: str = "",
    api_key_env: str = "",
    temperature: float | None = None,
    reasoning_effort: str = "",
    verbosity: str = "",
    timeout_seconds: int = 90,
) -> Path:
    """Write the ``ai.llm`` settings block to workspace ``ai.yaml``.

    Only non-empty values are persisted so the file stays minimal, and any
    existing ``chat_with_data`` block and unrelated keys are preserved. Secrets
    are never written; ``api_key_env`` names an environment variable instead.
    """

    root = Path(workspace)
    path = root / AI_CONFIG_FILENAMES[0]
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    llm: dict[str, Any] = {"model": model.strip()}
    if api_base.strip():
        llm["api_base"] = api_base.strip()
    if custom_provider.strip():
        llm["custom_provider"] = custom_provider.strip()
    if api_key_env.strip():
        llm["api_key_env"] = api_key_env.strip()
    if temperature is not None:
        llm["temperature"] = float(temperature)
    if reasoning_effort.strip():
        llm["reasoning_effort"] = reasoning_effort.strip()
    if verbosity.strip():
        llm["verbosity"] = verbosity.strip()
    llm["timeout_seconds"] = int(timeout_seconds)
    ai_block = loaded.get("ai")
    if not isinstance(ai_block, dict):
        ai_block = {}
    ai_block["llm"] = llm
    loaded["ai"] = ai_block
    path.write_text(yaml.safe_dump(loaded, sort_keys=False), encoding="utf-8")
    return path


def _candidate_config_paths(root: Path) -> list[Path]:
    """Return AI config candidates in deterministic priority order."""

    fixed = [root / file_name for file_name in AI_CONFIG_FILENAMES]
    return fixed


def _read_ai_config(path: Path) -> Any | None:
    if not path.is_file():
        logger.debug("Skipping AI settings candidate because it is not a file: path=%s", path)
        return None
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        logger.warning("Ignoring AI settings file with non-mapping YAML: path=%s", path)
    return loaded


def _default_chat_with_data_config() -> dict[str, Any]:
    return {
        "agent_prompt": DEFAULT_CHAT_AGENT_PROMPT,
        "dataset_descriptions": {},
        "metric_descriptions": {},
    }


def _normalize_chat_with_data_config(config: dict[str, Any]) -> dict[str, Any]:
    agent_prompt = str(config.get("agent_prompt") or "").strip() or DEFAULT_CHAT_AGENT_PROMPT
    return {
        "agent_prompt": agent_prompt,
        "dataset_descriptions": _clean_description_map(config.get("dataset_descriptions") or {}),
        "metric_descriptions": _clean_description_map(config.get("metric_descriptions") or {}),
    }


def _clean_description_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        description = str(value or "").strip()
        normalized_key = str(key or "").strip()
        if normalized_key and description:
            cleaned[normalized_key] = description
    return dict(sorted(cleaned.items(), key=lambda item: item[0].casefold()))


def configured_api_key(config: dict[str, Any]) -> str:
    """Return an API key from the configured env var or common provider env vars."""

    api_key_env = str(config.get("api_key_env") or "").strip()
    if api_key_env and os.environ.get(api_key_env):
        return str(os.environ[api_key_env])
    return (
        os.environ.get("LITELLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )


__all__ = [
    "AI_CONFIG_FILENAMES",
    "DEFAULT_CHAT_AGENT_PROMPT",
    "configured_api_key",
    "load_chat_with_data_config",
    "load_llm_settings_config",
    "write_chat_with_data_config",
    "write_llm_settings_config",
]
