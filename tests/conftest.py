"""Shared test fixtures.

The committed demo workspace (``examples/demo``) ships its *catalog* only — the
``data/``, ``aggregates/``, and ``meta/`` folders are intentionally gitignored.
Tests that need queryable aggregates build a throwaway workspace from that
catalog plus a small synthetic dataset produced by :func:`generate_demo_interactions`,
then run the real ingestion pipeline over it.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest

from valuestream.engine import run_workspace


@pytest.fixture(autouse=True)
def _preserve_main_module() -> object:
    """Restore ``sys.modules['__main__']`` after each test.

    Streamlit's AppTest executes pages as ``__main__`` via a temp wrapper
    script that is not import-safe. If it leaks, a later test that spawns
    worker processes (parallel ingestion) re-imports that wrapper in the
    child and the process pool breaks with a NameError.
    """

    main_module = sys.modules.get("__main__")
    yield
    if main_module is not None and sys.modules.get("__main__") is not main_module:
        sys.modules["__main__"] = main_module


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CATALOG = REPO_ROOT / "examples" / "demo" / "catalog"

_CHANNELS = ("Web", "Mobile", "Email")
_DIRECTIONS = ("Inbound", "Outbound")
_SEGMENTS = ("VIP", "Premium", "CLVLow")
_ISSUES = ("Cards", "Loans")
_GROUPS = ("Acquisition", "Retention")
_CONTROL = ("Test", "NBA")
_OUTCOMES = ("Impression", "Clicked", "Conversion", "Pending")


def _pega_time(moment: dt.datetime) -> str:
    """Format a datetime the way the demo ``parse_datetime`` transform expects."""
    return f"{moment:%Y%m%dT%H%M%S}.{moment.microsecond // 1000:03d} GMT"


def generate_demo_interactions(
    data_dir: Path,
    *,
    year: int = 2026,
    months: tuple[int, ...] = (1, 2, 3, 4),
    days: tuple[int, ...] = (5, 20),
    rows_per_file: int = 24,
) -> list[Path]:
    """Write small synthetic Pega-shaped interaction parquet files.

    Columns are emitted with the names the demo pipeline expects after
    ``rename_capitalize`` (which is a no-op on them), so the real transforms and
    every processor run unchanged over the generated data.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    counter = 0
    for month in months:
        for day in days:
            rows = []
            for index in range(rows_per_file):
                outcome = _OUTCOMES[index % len(_OUTCOMES)]
                outcome_time = dt.datetime(year, month, day, 12, 0, 0) + dt.timedelta(minutes=index)
                decision_time = outcome_time - dt.timedelta(seconds=30)
                rows.append(
                    {
                        "InteractionID": f"I-{counter}",
                        "CustomerID": f"C-{counter % 50}",
                        "Channel": _CHANNELS[index % len(_CHANNELS)],
                        "Direction": _DIRECTIONS[index % len(_DIRECTIONS)],
                        "CustomerSegment": _SEGMENTS[index % len(_SEGMENTS)],
                        "IsProspect": index % 5 == 0,
                        "Issue": _ISSUES[index % len(_ISSUES)],
                        "Group": _GROUPS[index % len(_GROUPS)],
                        "Name": f"Offer{index % 4}",
                        "Treatment": f"T{index % 3}",
                        "PlacementType": "Leaderboard",
                        "ModelControlGroup": _CONTROL[index % len(_CONTROL)],
                        "PropensitySource": "Model",
                        "Outcome": outcome,
                        "Propensity": 0.1 + (index % 9) / 10.0,
                        "FinalPropensity": 0.15 + (index % 8) / 10.0,
                        "Priority": float(index % 7),
                        "Rank": index % 5,
                        "OutcomeTime": _pega_time(outcome_time),
                        "DecisionTime": _pega_time(decision_time),
                    }
                )
                counter += 1
            target = data_dir / f"pega_interactions_{year}{month:02d}{day:02d}.parquet"
            pl.DataFrame(rows).write_parquet(target)
            written.append(target)
    return written


def build_demo_workspace(root: Path) -> Path:
    """Assemble a runnable demo workspace (catalog + generated data) under ``root``."""
    shutil.copytree(DEMO_CATALOG, root / "catalog")
    for name in ("ai.yaml", "ai_gpt.yaml"):
        source = DEMO_CATALOG.parent / name
        if source.exists():
            shutil.copy(source, root / name)
    generate_demo_interactions(root / "data")
    run_workspace(root)
    return root


@pytest.fixture(scope="session")
def demo_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped runnable demo workspace with freshly ingested aggregates."""
    root = tmp_path_factory.mktemp("demo_workspace")
    return build_demo_workspace(root)
