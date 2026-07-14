"""Application logging helpers.

The setup mirrors the legacy ``proof_of_value`` application: logging is
configured from a bundled YAML file, and log records receive ``name_last`` so
the formatter can show concise module names.
"""

from __future__ import annotations

import logging
import logging.config
from importlib import resources
from pathlib import Path
from typing import Any, cast

import yaml

_DEFAULT_CONFIG = "logging_config.yaml"
_RESERVED_KEYS = frozenset({"pipeline_run_id", "chunk_id"})


class LastPartFilter(logging.Filter):
    """Attach the final logger name segment to log records for display."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.name_last = record.name.rsplit(".", 1)[-1]
        return True


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
        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name_last)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
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
