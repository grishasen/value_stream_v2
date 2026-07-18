from pathlib import Path

from scripts.mypy_gate import (
    baseline_diff,
    evaluate_mypy_result,
    extract_diagnostics,
)


def test_extract_diagnostics_is_relative_sorted_and_error_only(tmp_path: Path) -> None:
    output = "\n".join(
        [
            f"{tmp_path}/src/z.py:8:2: error: Later [assignment]",
            "src/a.py:1: note: context only",
            f"{tmp_path}/src/a.py:3: error: Earlier [arg-type]",
            "Found 2 errors in 2 files",
        ]
    )

    assert extract_diagnostics(output, root=tmp_path) == (
        "src/a.py:3: error: Earlier [arg-type]",
        "src/z.py:8:2: error: Later [assignment]",
    )


def test_evaluate_mypy_result_accepts_exact_diagnostics() -> None:
    diagnostic = "src/a.py:3: error: Existing [arg-type]"

    result = evaluate_mypy_result(1, diagnostic, [diagnostic])

    assert not result.hard_failure
    assert result.matches_baseline


def test_evaluate_mypy_result_rejects_added_and_removed_diagnostics() -> None:
    first = "src/a.py:3: error: Existing [arg-type]"
    second = "src/b.py:4: error: New [assignment]"

    added = evaluate_mypy_result(1, f"{first}\n{second}", [first])
    removed = evaluate_mypy_result(1, first, [first, second])

    assert not added.matches_baseline
    assert not removed.matches_baseline
    assert "+src/b.py:4: error: New [assignment]" in baseline_diff([first], added.diagnostics)
    assert "-src/b.py:4: error: New [assignment]" in baseline_diff(
        [first, second], removed.diagnostics
    )


def test_evaluate_mypy_result_treats_exit_two_as_hard_failure() -> None:
    result = evaluate_mypy_result(
        2,
        "dependency.pyi:10: error: Type statement requires Python 3.12 [syntax]",
        [],
    )

    assert result.hard_failure
    assert not result.matches_baseline


def test_evaluate_mypy_result_rejects_unparsed_failure() -> None:
    result = evaluate_mypy_result(1, "mypy failed without diagnostics", [])

    assert result.hard_failure
    assert not result.matches_baseline


def test_evaluate_mypy_result_rejects_success_with_diagnostics() -> None:
    diagnostic = "src/a.py:3: error: Contradictory success [arg-type]"

    result = evaluate_mypy_result(0, diagnostic, [diagnostic])

    assert result.hard_failure
    assert not result.matches_baseline
