"""Console-color contracts for application logging."""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import pytest

from valuestream.utils import logger as log_utils

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_COLOR_FORMAT = (
    "%(ansi_level)s%(levelname)s%(ansi_reset)s:"
    "%(ansi_logger)s%(name_last)s%(ansi_reset)s:%(message)s"
)


class _TerminalBuffer(io.StringIO):
    def __init__(self, *, is_terminal: bool) -> None:
        super().__init__()
        self._is_terminal = is_terminal

    def isatty(self) -> bool:
        return self._is_terminal


@pytest.fixture(autouse=True)
def _clean_color_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NO_COLOR",
        "FORCE_COLOR",
        "TERM",
        "COLORFGBG",
        "VALUESTREAM_LOG_THEME",
    ):
        monkeypatch.delenv(name, raising=False)


def _record(level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        "valuestream.engine.runner",
        level,
        __file__,
        1,
        "processed %s",
        ("chunk",),
        None,
    )


def _handler(
    stream: io.StringIO,
    *,
    colors: bool | None = None,
    theme: str = "light",
) -> log_utils.ThemeColorStreamHandler:
    handler = log_utils.ThemeColorStreamHandler(
        stream,
        colors=colors,
        theme=theme,
    )
    handler.setFormatter(logging.Formatter(_COLOR_FORMAT))
    handler.addFilter(log_utils.LastPartFilter())
    return handler


@pytest.mark.unit
@pytest.mark.parametrize(
    ("level", "rgb"),
    [
        (logging.DEBUG, "111;99;181"),
        (logging.INFO, "0;114;178"),
        (logging.WARNING, "179;107;0"),
        (logging.ERROR, "213;94;0"),
        (logging.CRITICAL, "162;59;114"),
    ],
)
def test_color_handler_uses_theme_palette_for_levels(level: int, rgb: str) -> None:
    stream = _TerminalBuffer(is_terminal=True)
    handler = _handler(stream, colors=True)
    record = _record(level)

    handler.emit(record)

    rendered = stream.getvalue()
    assert f"\x1b[38;2;{rgb}m{record.levelname}\x1b[0m" in rendered
    assert "\x1b[38;2;0;158;115mrunner\x1b[0m" in rendered
    assert _ANSI_RE.sub("", rendered) == f"{record.levelname}:runner:processed chunk\n"
    if level == logging.CRITICAL:
        assert "\x1b[1m\x1b[38;2;162;59;114mCRITICAL" in rendered


@pytest.mark.unit
def test_color_handler_colors_the_complete_timestamp() -> None:
    stream = _TerminalBuffer(is_terminal=True)
    handler = log_utils.ThemeColorStreamHandler(stream, colors=True, theme="light")
    handler.setFormatter(
        logging.Formatter(
            log_utils._DETAILED_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(log_utils.LastPartFilter())

    handler.emit(_record())

    rendered = stream.getvalue()
    assert rendered.startswith("\x1b[38;2;82;96;109m")
    assert "\x1b[0m:\x1b[38;2;0;114;178mINFO" in rendered
    assert rendered.endswith(":processed chunk\n")


@pytest.mark.unit
def test_non_tty_output_stays_byte_compatible_with_plain_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = _TerminalBuffer(is_terminal=False)
    handler = _handler(stream)

    handler.emit(_record())

    assert stream.getvalue() == "INFO:runner:processed chunk\n"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("environment", "expect_color"),
    [
        ({}, True),
        ({"TERM": "dumb"}, False),
        ({"NO_COLOR": ""}, False),
        ({"FORCE_COLOR": "1"}, True),
        ({"FORCE_COLOR": "0"}, False),
        ({"NO_COLOR": "", "FORCE_COLOR": "1"}, False),
    ],
)
def test_color_environment_contract(
    environment: dict[str, str],
    expect_color: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    stream = _TerminalBuffer(is_terminal=True)
    handler = _handler(stream)

    handler.emit(_record())

    assert ("\x1b[" in stream.getvalue()) is expect_color


@pytest.mark.unit
def test_force_color_can_color_redirected_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    stream = _TerminalBuffer(is_terminal=False)
    handler = _handler(stream)

    handler.emit(_record())

    assert "\x1b[" in stream.getvalue()


@pytest.mark.unit
def test_no_color_overrides_explicit_handler_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    stream = _TerminalBuffer(is_terminal=True)
    handler = _handler(stream, colors=True)

    handler.emit(_record())

    assert stream.getvalue() == "INFO:runner:processed chunk\n"


@pytest.mark.unit
def test_color_handler_does_not_mutate_the_shared_record() -> None:
    stream = _TerminalBuffer(is_terminal=True)
    handler = _handler(stream, colors=True)
    record = _record()
    log_utils.LastPartFilter().filter(record)

    colored = handler.format(record)
    plain = logging.Formatter("%(levelname)s:%(name_last)s:%(message)s").format(record)

    assert "\x1b[" in colored
    assert plain == "INFO:runner:processed chunk"
    assert record.levelname == "INFO"
    assert record.getMessage() == "processed chunk"
    assert not any(name.startswith("ansi_") for name in vars(record))


@pytest.mark.unit
def test_log_theme_can_follow_terminal_background_or_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VALUESTREAM_LOG_THEME", raising=False)
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert log_utils._resolve_log_theme("auto") == "light"

    monkeypatch.setenv("COLORFGBG", "15;0")
    assert log_utils._resolve_log_theme("auto") == "dark"

    monkeypatch.setenv("VALUESTREAM_LOG_THEME", "light")
    assert log_utils._resolve_log_theme("auto") == "light"
    assert log_utils._resolve_log_theme("dark") == "dark"


@pytest.mark.unit
def test_bundled_config_uses_color_handler_on_stderr() -> None:
    config = log_utils._load_config(None)

    assert config is not None
    assert config["handlers"]["console"] == {
        "class": "valuestream.utils.logger.ThemeColorStreamHandler",
        "level": "DEBUG",
        "formatter": "detailed",
        "stream": "ext://sys.stderr",
    }
    assert "%(ansi_level)s" in config["formatters"]["detailed"]["format"]


@pytest.mark.unit
def test_fallback_config_uses_same_color_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(log_utils, "_load_config", lambda _path: None)
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    log_utils.configure(config_path="missing.yaml")

    handler = captured["handlers"][0]
    assert isinstance(handler, log_utils.ThemeColorStreamHandler)
    assert isinstance(handler.formatter, logging.Formatter)
