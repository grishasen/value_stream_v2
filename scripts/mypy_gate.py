"""Run mypy against ``src`` and reject changes to the known diagnostic set.

The repository is adopting mypy incrementally. The committed baseline keeps
the existing debt explicit while making added, removed, or changed diagnostics
a blocking event that must be reviewed through ``--update``.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "mypy-baseline.txt"
_DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>.+?\.pyi?):(?P<location>\d+(?::\d+)?): error: (?P<message>.+)$"
)


@dataclass(frozen=True)
class GateEvaluation:
    """Result of comparing one mypy invocation with the committed baseline."""

    diagnostics: tuple[str, ...]
    hard_failure: bool
    matches_baseline: bool


def extract_diagnostics(output: str, *, root: Path = ROOT) -> tuple[str, ...]:
    """Return deterministic, workspace-relative mypy error diagnostics."""

    root_prefix = f"{root.resolve().as_posix().rstrip('/')}/"
    diagnostics: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = _DIAGNOSTIC_RE.match(line)
        if match is None:
            continue
        path = match.group("path").replace("\\", "/")
        if path.startswith(root_prefix):
            path = path[len(root_prefix) :]
        diagnostics.append(f"{path}:{match.group('location')}: error: {match.group('message')}")
    return tuple(sorted(diagnostics))


def load_baseline(path: Path = BASELINE_PATH) -> tuple[str, ...]:
    """Load sorted diagnostics, ignoring explanatory comments and blanks."""

    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(
        sorted(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )


def evaluate_mypy_result(
    returncode: int,
    output: str,
    baseline: Sequence[str],
    *,
    root: Path = ROOT,
) -> GateEvaluation:
    """Classify a mypy result and compare its exact errors with ``baseline``."""

    diagnostics = extract_diagnostics(output, root=root)
    hard_failure = (
        returncode not in (0, 1)
        or (returncode == 1 and not diagnostics)
        or (returncode == 0 and bool(diagnostics))
    )
    return GateEvaluation(
        diagnostics=diagnostics,
        hard_failure=hard_failure,
        matches_baseline=not hard_failure and diagnostics == tuple(sorted(baseline)),
    )


def baseline_diff(baseline: Sequence[str], diagnostics: Sequence[str]) -> str:
    """Render a stable unified diff for review."""

    return "\n".join(
        difflib.unified_diff(
            sorted(baseline),
            sorted(diagnostics),
            fromfile="mypy-baseline.txt",
            tofile="current mypy diagnostics",
            lineterm="",
        )
    )


def run_mypy() -> subprocess.CompletedProcess[str]:
    """Run the deterministic full-source mypy command."""

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--no-pretty",
            "--no-color-output",
            "--no-error-summary",
            "--show-error-codes",
            "--show-column-numbers",
            "src",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _write_baseline(diagnostics: Sequence[str]) -> None:
    header = (
        "# Exact mypy diagnostic baseline; do not edit by hand.\n"
        "# Review changes, then refresh with:\n"
        "#   uv run python scripts/mypy_gate.py --update\n"
    )
    content = header + "".join(f"{line}\n" for line in sorted(diagnostics))
    temporary = BASELINE_PATH.with_suffix(".txt.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(BASELINE_PATH)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace the baseline after reviewing every diagnostic change",
    )
    args = parser.parse_args(argv)

    result = run_mypy()
    output = _combined_output(result)
    diagnostics = extract_diagnostics(output)
    hard_failure = (
        result.returncode not in (0, 1)
        or (result.returncode == 1 and not diagnostics)
        or (result.returncode == 0 and bool(diagnostics))
    )
    if hard_failure:
        print("mypy did not complete normal analysis; baseline was not used.", file=sys.stderr)
        if output:
            print(output, file=sys.stderr)
        return 2

    if args.update:
        _write_baseline(diagnostics)
        print(f"Updated mypy baseline with {len(diagnostics)} diagnostics.")
        return 0

    try:
        baseline = load_baseline()
    except FileNotFoundError:
        print(
            "Missing mypy-baseline.txt; review diagnostics and run "
            "`uv run python scripts/mypy_gate.py --update`.",
            file=sys.stderr,
        )
        return 2

    evaluation = evaluate_mypy_result(result.returncode, output, baseline)
    if not evaluation.matches_baseline:
        print(
            "Mypy diagnostics changed. Review both fixes and regressions; "
            "then update the baseline explicitly if intentional.",
            file=sys.stderr,
        )
        print(baseline_diff(baseline, evaluation.diagnostics), file=sys.stderr)
        return 1

    print(
        f"Mypy analyzed all src modules; {len(baseline)} known diagnostics "
        "match the reviewed baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
