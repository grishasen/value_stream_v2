"""Small, environment-backed rollout switches for the Streamlit surface."""

from __future__ import annotations

import os
from collections.abc import Mapping

_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})


def authoring_v2_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the guided authoring entry point is exposed.

    The revised experience is the default for new local sessions. Operators can
    set ``VALUESTREAM_AUTHORING_V2=0`` to retain the legacy navigation grouping
    while a measured rollout is in progress.
    """

    values = os.environ if environ is None else environ
    raw = str(values.get("VALUESTREAM_AUTHORING_V2", "1") or "1").strip().casefold()
    if raw in _FALSE_VALUES:
        return False
    if raw in _TRUE_VALUES:
        return True
    return True


__all__ = ["authoring_v2_enabled"]
