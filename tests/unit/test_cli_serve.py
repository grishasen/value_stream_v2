"""CLI tests for starting the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from valuestream.cli import main


@pytest.mark.unit
def test_serve_accepts_missing_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "New AI Workspace"
    captured: dict[str, Any] = {}

    class Completed:
        returncode = 0

    def fake_run(args: list[str], *, check: bool) -> Completed:
        captured["args"] = args
        captured["check"] = check
        return Completed()

    monkeypatch.setattr("valuestream.cli.subprocess.run", fake_run)

    result = CliRunner().invoke(main, ["serve", str(workspace), "--headless"])

    assert result.exit_code == 0, result.output
    assert captured["check"] is False
    assert str(workspace) in captured["args"]


@pytest.mark.unit
def test_serve_mcp_command_invokes_stdio_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: dict[str, Any] = {}

    def fake_run_stdio(path: str, *, enable_sql: bool) -> None:
        captured["path"] = path
        captured["enable_sql"] = enable_sql

    monkeypatch.setattr("valuestream.cli.run_mcp_stdio", fake_run_stdio)

    result = CliRunner().invoke(main, ["serve-mcp", str(workspace)])

    assert result.exit_code == 0, result.output
    assert captured["path"] == str(workspace)
    assert captured["enable_sql"] is False


@pytest.mark.unit
def test_serve_api_requires_token_for_non_loopback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("VALUESTREAM_API_TOKEN", raising=False)

    result = CliRunner().invoke(
        main,
        ["serve-api", str(workspace), "--host", "0.0.0.0"],
    )

    assert result.exit_code != 0
    assert "bearer token is required" in result.output


@pytest.mark.unit
def test_serve_api_passes_environment_token_and_sql_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: dict[str, Any] = {}

    def fake_create_app(path: str, *, api_token: str, enable_sql: bool) -> object:
        captured.update(path=path, api_token=api_token, enable_sql=enable_sql)
        return object()

    def fake_run(app: object, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setenv("VALUESTREAM_API_TOKEN", "environment-secret")
    monkeypatch.setattr("valuestream.api.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", fake_run)

    result = CliRunner().invoke(
        main,
        ["serve-api", str(workspace), "--host", "0.0.0.0", "--enable-sql"],
    )

    assert result.exit_code == 0, result.output
    assert captured["api_token"] == "environment-secret"
    assert captured["enable_sql"] is True
    assert captured["host"] == "0.0.0.0"
