"""Application logging helpers.

The setup mirrors the legacy ``proof_of_value`` application: logging is
configured from a bundled YAML file, and log records receive ``name_last`` so
the formatter can show concise module names.
"""

from __future__ import annotations

import copy
import logging
import logging.config
import os
from importlib import resources
from pathlib import Path
from typing import Any, TextIO, cast

import yaml

_DEFAULT_CONFIG = "logging_config.yaml"
_RESERVED_KEYS = frozenset({"pipeline_run_id", "chunk_id"})
_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD = "\x1b[1m"
_LOG_THEME_ENV = "VALUESTREAM_LOG_THEME"
_DETAILED_FORMAT = (
    "%(ansi_timestamp)s%(asctime)s.%(msecs)03d%(ansi_reset)s:"
    "%(ansi_level)s%(levelname)s%(ansi_reset)s:"
    "%(ansi_logger)s%(name_last)s%(ansi_reset)s:%(message)s"
)

# These are terminal-safe counterparts of the application chrome and Plotly
# qualitative palettes. Messages stay in the terminal's native foreground;
# color is reserved for the timestamp, severity, and module landmarks.
_LOG_PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "timestamp": "#52606D",
        "logger": "#009E73",
        "DEBUG": "#6F63B5",
        "INFO": "#0072B2",
        "WARNING": "#B36B00",
        "ERROR": "#D55E00",
        "CRITICAL": "#A23B72",
        "fallback": "#52606D",
    },
    "dark": {
        "timestamp": "#B8C4D2",
        "logger": "#45D6A5",
        "DEBUG": "#AD87ED",
        "INFO": "#4B73F0",
        "WARNING": "#F2C14E",
        "ERROR": "#FF8A80",
        "CRITICAL": "#F17CB0",
        "fallback": "#B8C4D2",
    },
}


class LastPartFilter(logging.Filter):
    """Attach the final logger name segment to log records for display."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.name_last = record.name.rsplit(".", 1)[-1]
        return True


class ThemeColorStreamHandler(logging.StreamHandler):
    """Color console log landmarks with the application and Plotly palettes.

    Color is automatic by default: it is used for an interactive terminal and
    omitted for redirected output, ``TERM=dumb``, or ``NO_COLOR``. The standard
    ``FORCE_COLOR`` variable forces it for terminals that cannot report TTY
    capability. ``VALUESTREAM_LOG_THEME=light|dark`` overrides background
    detection without coupling foundational logging to the Streamlit module.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        colors: bool | None = None,
        theme: str = "auto",
    ) -> None:
        super().__init__(stream)
        self._colors = colors
        self._theme = theme.casefold()

    def format(self, record: logging.LogRecord) -> str:
        rendered_record = copy.copy(record)
        rendered_record.name_last = str(
            getattr(record, "name_last", None) or record.name.rsplit(".", 1)[-1]
        )
        rendered_record.ansi_timestamp = ""
        rendered_record.ansi_level = ""
        rendered_record.ansi_logger = ""
        rendered_record.ansi_reset = ""
        if self._colors_enabled():
            palette = self._palette()
            rendered_record.ansi_timestamp = _ansi_prefix(palette["timestamp"])
            rendered_record.ansi_level = _ansi_prefix(
                palette.get(record.levelname, palette["fallback"]),
                bold=record.levelno >= logging.CRITICAL,
            )
            rendered_record.ansi_logger = _ansi_prefix(palette["logger"])
            rendered_record.ansi_reset = _ANSI_RESET
        return super().format(rendered_record)

    def _colors_enabled(self) -> bool:
        if "NO_COLOR" in os.environ:
            return False
        if self._colors is not None:
            return self._colors
        return _supports_color(self.stream)

    def _palette(self) -> dict[str, str]:
        return _LOG_PALETTES[_resolve_log_theme(self._theme)]


def configure(
    level: int = logging.INFO,
    *,
    config_path: str | Path | None = None,
) -> None:
    """Configure application logging from YAML, falling back to ``basicConfig``."""
    config = _load_config(config_path)
    if config is not None:
        logging.config.dictConfig(config)
        if level != logging.INFO:
            logging.getLogger().setLevel(level)
    else:
        handler = ThemeColorStreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt=_DETAILED_FORMAT,
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.basicConfig(level=level, handlers=[handler])
    _attach_last_part_filter()


def configure_logging(config_path: str | Path | None = None) -> None:
    """Compatibility alias matching the legacy application helper name."""
    configure(config_path=config_path)


def get_logger(name: str | None = None, level: int | None = None) -> logging.Logger:
    """Return a configured logger for a module name."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def reserved_keys() -> frozenset[str]:
    """Return the reserved log-context keys."""
    return _RESERVED_KEYS


def _load_config(config_path: str | Path | None) -> dict[str, Any] | None:
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return cast("dict[str, Any]", yaml.safe_load(handle))
        return None

    config = resources.files("valuestream.config").joinpath(_DEFAULT_CONFIG)
    if config.is_file():
        with config.open(encoding="utf-8") as handle:
            return cast("dict[str, Any]", yaml.safe_load(handle))
    return None


def _attach_last_part_filter() -> None:
    if logging.getLogger().hasHandlers():
        for handler in logging.getLogger().handlers:
            _add_last_part_filter(handler)
    for logger_name in logging.root.manager.loggerDict:
        candidate = logging.getLogger(logger_name)
        for handler in candidate.handlers:
            _add_last_part_filter(handler)


def _add_last_part_filter(handler: logging.Handler) -> None:
    if not any(isinstance(filter_, LastPartFilter) for filter_ in handler.filters):
        handler.addFilter(LastPartFilter())


def _supports_color(stream: TextIO) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    force_color = os.environ.get("FORCE_COLOR")
    if force_color is not None:
        return force_color != "0"
    if os.environ.get("TERM", "").casefold() == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False


def _resolve_log_theme(configured: str) -> str:
    if configured in _LOG_PALETTES:
        return configured
    requested = os.environ.get(_LOG_THEME_ENV, "").casefold()
    if requested in _LOG_PALETTES:
        return requested
    color_fgbg = os.environ.get("COLORFGBG", "")
    try:
        background = int(color_fgbg.rsplit(";", 1)[-1])
    except (TypeError, ValueError):
        return "dark"
    return "light" if background >= 8 else "dark"


def _ansi_prefix(hex_color: str, *, bold: bool = False) -> str:
    normalized = hex_color.removeprefix("#")
    red, green, blue = (int(normalized[index : index + 2], 16) for index in (0, 2, 4))
    weight = _ANSI_BOLD if bold else ""
    return f"{weight}\x1b[38;2;{red};{green};{blue}m"
