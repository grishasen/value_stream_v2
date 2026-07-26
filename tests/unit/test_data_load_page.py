"""Tests for the Data Load page."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from valuestream.ui import builder
from valuestream.ui.pages import data_load


@pytest.mark.unit
def test_render_without_sources_shows_actionable_empty_state() -> None:
    def app() -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import data_load  # noqa: PLC0415 - isolated AppTest source

        data_load.render(
            SimpleNamespace(
                validation=SimpleNamespace(ok=True, issues=[]),
                catalog=SimpleNamespace(pipelines=SimpleNamespace(sources=[])),
            )
        )

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    assert [message.value for message in rendered.info] == [
        "No data sources are configured for this workspace. Add a source in "
        "Configuration Builder before loading or rebuilding data."
    ]
    add_source = next(
        item
        for item in rendered.get("link_button")
        if item.label == "Add source in Configuration Builder"
    )
    assert add_source.url == "/configuration_builder?builder_step=sources"
    assert not rendered.get("tab")
    assert not rendered.toggle
    assert not rendered.button


@pytest.mark.unit
def test_blocked_new_workspace_can_load_sample_and_revalidate(tmp_path: Path) -> None:
    builder.write_source_definition(
        tmp_path,
        {
            "id": "interaction_history",
            "reader": {
                "kind": "parquet",
                "file_pattern": "**/*.parquet",
                "root": "data",
            },
            "transforms": [
                {
                    "kind": "filter",
                    "expression": {
                        "op": "not_null",
                        "column": "SubjectID",
                    },
                },
                {
                    "kind": "derive_column",
                    "output": "Placement",
                    "expression": {
                        "op": "case",
                        "when": [
                            {
                                "cond": {
                                    "op": "ne",
                                    "column": "PlacementType",
                                    "value": "",
                                },
                                "then": {"col": "PlacementType"},
                            }
                        ],
                        "else": {"lit": "Unknown"},
                    },
                },
            ],
        },
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "SubjectID": ["subject-1"],
            "PlacementType": ["Web"],
        }
    ).write_parquet(data_dir / "sample.parquet")

    def app(workspace: str) -> None:
        from valuestream.ui.context import load_context  # noqa: PLC0415
        from valuestream.ui.pages import data_load  # noqa: PLC0415

        data_load.render(load_context(workspace))

    rendered = AppTest.from_function(app, kwargs={"workspace": str(tmp_path)}).run()

    assert not rendered.exception
    assert any("SubjectID" in str(item.value) for item in rendered.error)
    assert any(button.label == "Load sample" for button in rendered.button)
    assert not rendered.get("tab")

    rendered = (
        next(button for button in rendered.button if button.label == "Load sample")
        .click()
        .run(timeout=15)
    )

    assert not rendered.exception
    assert not rendered.error
    assert any(button.label == "Reload sample" for button in rendered.button)
    assert rendered.get("tab")


@pytest.mark.unit
def test_ordered_sources_reverses_without_mutating_catalog_order() -> None:
    sources = [
        SimpleNamespace(id="interaction_history"),
        SimpleNamespace(id="product_holdings"),
    ]

    ordered = data_load._ordered_sources(sources)

    assert [source.id for source in ordered] == ["product_holdings", "interaction_history"]
    assert [source.id for source in sources] == ["interaction_history", "product_holdings"]


@pytest.mark.unit
def test_run_summary_reports_idempotently_skipped_chunks() -> None:
    result = SimpleNamespace(
        status="ok",
        chunks_ok=0,
        chunks_skipped=75,
        chunks_failed=0,
        rows_kept=0,
    )

    assert data_load._run_summary(result) == (
        "ok: 0 chunk(s) ok, 75 skipped, 0 failed, 0 rows kept"
    )


@pytest.mark.unit
def test_failed_run_groups_identical_chunk_errors_for_display() -> None:
    error = "processor input columns are missing: interaction_history_processor: MktType, MktValue"
    result = SimpleNamespace(
        status="failed",
        chunks_failed=2,
        chunks=[
            SimpleNamespace(chunk_id="2024-03-10", status="failed", error=error),
            SimpleNamespace(chunk_id="2024-03-11", status="failed", error=error),
            SimpleNamespace(chunk_id="2024-03-12", status="ok", error=None),
        ],
        source_id="interaction_history",
    )

    groups = data_load._run_failure_groups(result)

    assert data_load._result_failed(result)
    assert len(groups) == 1
    assert groups[0].source_id == "interaction_history"
    assert groups[0].error == error
    assert groups[0].chunk_ids == ("2024-03-10", "2024-03-11")


@pytest.mark.unit
def test_failed_run_is_rendered_as_error_with_expanded_chunk_details() -> None:
    def app() -> None:
        from types import SimpleNamespace  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import data_load  # noqa: PLC0415 - isolated AppTest source

        error = "processor input columns are missing: p: MktType, MktValue"
        result = SimpleNamespace(
            status="failed",
            chunks_ok=80,
            chunks_skipped=0,
            chunks_failed=163,
            rows_kept=204_979_483,
            source_id="interaction_history",
            chunks=[
                SimpleNamespace(chunk_id="2024-03-10", status="failed", error=error),
                SimpleNamespace(chunk_id="2024-03-11", status="failed", error=error),
            ],
        )
        data_load._render_completed_run(
            data_load._BackgroundRun(
                label="Source run · interaction_history",
                started_at=0.0,
                result=result,
            )
        )

    rendered = AppTest.from_function(app).run()

    assert not rendered.exception
    errors = [str(item.value) for item in rendered.error]
    assert len(errors) == 2
    assert any("Source run · interaction_history — failed" in value for value in errors)
    assert any("MktType, MktValue" in value for value in errors)
    assert any("Successful chunks were kept" in str(item.value) for item in rendered.info)


@pytest.mark.unit
def test_aggregate_inventory_is_limited_to_selected_sources(tmp_path: Path) -> None:
    selected = tmp_path / "aggregates" / "selected" / "processor" / "daily"
    other = tmp_path / "aggregates" / "other" / "processor" / "daily"
    selected.mkdir(parents=True)
    other.mkdir(parents=True)
    (selected / "one.parquet").write_bytes(b"1234")
    (selected / "two.parquet").write_bytes(b"12")
    (other / "three.parquet").write_bytes(b"ignored")

    files, bytes_used = data_load._aggregate_inventory(tmp_path, ["selected"])

    assert files == 2
    assert bytes_used == 6
    assert data_load._format_bytes(bytes_used) == "6 B"
