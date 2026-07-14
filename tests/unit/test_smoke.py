"""Smoke tests proving the package imports and the CLI is wired up."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import valuestream
from valuestream.cli import main


@pytest.mark.unit
def test_package_version_is_set() -> None:
    assert valuestream.__version__ == "0.1.0"


@pytest.mark.unit
def test_cli_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Value Stream" in result.output


@pytest.mark.unit
def test_cli_version_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
