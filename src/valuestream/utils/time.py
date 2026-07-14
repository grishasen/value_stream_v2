"""Time helpers — UTC clocks and grain helpers.

Phase 0 only needs ``utc_now``; the calendar / grain helpers used by
``derive_calendar`` land in Phase 1.
"""

from __future__ import annotations

import datetime as _dt


def utc_now() -> _dt.datetime:
    """Return the current UTC instant as a timezone-aware datetime."""
    return _dt.datetime.now(_dt.UTC)
