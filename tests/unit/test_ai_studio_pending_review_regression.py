"""Regression tests for explicit deterministic source-replacement review.

The fixture models a restored accepted draft for one source followed by a valid,
locally generated deterministic draft for a newly selected source. Both catalogs
are independently valid; only the active source identity changes.
"""

from __future__ import annotations

import copy

import pytest


def _draft_for_source(source_id: str) -> dict:
    return {
        "pipelines": {
            "version": 1,
            "workspace": "pending-review-regression",
            "sources": [
                {
                    "id": source_id,
                    "reader": {"kind": "csv", "file_pattern": "*.csv"},
                    "schema": {
                        "timestamp_column": "OutcomeTime",
                        "natural_key": ["CustomerID"],
                    },
                }
            ],
        },
        "processors": {
            "processors": [
                {
                    "id": "engagement",
                    "source": source_id,
                    "kind": "binary_outcome",
                    "dimensions": ["Channel"],
                    "time": {"column": "OutcomeTime", "grains": ["Day", "Summary"]},
                    "outcome": {
                        "column": "Outcome",
                        "positive_values": ["Clicked"],
                        "negative_values": ["Impression"],
                    },
                }
            ]
        },
        "metrics": {
            "metrics": {
                "CTR": {
                    "source": "engagement",
                    "kind": "formula",
                    "expression": {
                        "op": "safe_div",
                        "num": {"col": "Positives"},
                        "den": {"col": "Count"},
                    },
                }
            }
        },
        "dashboards": {
            "dashboards": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "pages": [
                        {
                            "id": "engagement",
                            "title": "Engagement",
                            "tiles": [
                                {
                                    "id": "ctr",
                                    "title": "CTR",
                                    "metric": "CTR",
                                    "chart": "line",
                                    "x": "Day",
                                    "y": "CTR",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    }


def _seed_review_state(st: object, page: object, *, base: dict, pending: dict, kind: str) -> None:
    session_state = st.session_state
    if session_state.get("pending_review_regression_initialized"):
        return
    session_state["pending_review_regression_initialized"] = True
    session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
    session_state[page.AI_STUDIO_RENAME_CAPITALIZE_STATE_KEY] = False
    session_state["ai_studio_source_id"] = "new_source"
    session_state["ai_studio_raw_schema_columns"] = [
        "CustomerID",
        "OutcomeTime",
        "Outcome",
        "Channel",
        "Day",
    ]
    session_state["ai_studio_effective_schema_columns"] = [
        "CustomerID",
        "OutcomeTime",
        "Outcome",
        "Channel",
        "Day",
    ]
    session_state["ai_studio_approved_fields"] = [
        "CustomerID",
        "OutcomeTime",
        "Outcome",
        "Channel",
        "Day",
    ]
    session_state["ai_studio_example_fields"] = []
    session_state["ai_studio_draft"] = copy.deepcopy(base)
    session_state["ai_studio_pending_draft"] = copy.deepcopy(pending)
    session_state["ai_studio_pending_base_draft"] = copy.deepcopy(base)
    session_state["ai_studio_pending_kind"] = kind
    session_state["ai_studio_pending_prompt"] = "Generated locally; no provider call."
    session_state["ai_studio_last_ai_response"] = ""


@pytest.mark.unit
def test_deterministic_source_replacement_requires_explicit_review_and_can_be_accepted() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    base = _draft_for_source("old_source")
    pending = _draft_for_source("new_source")

    def app(base_draft: dict, pending_draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from tests.unit.test_ai_studio_pending_review_regression import (  # noqa: PLC0415
            _seed_review_state,
        )
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        _seed_review_state(
            st,
            page,
            base=base_draft,
            pending=pending_draft,
            kind="deterministic draft",
        )
        if st.session_state.get("ai_studio_pending_draft") is not None:
            page._render_pending_draft_review()

    rendered = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()

    assert not rendered.exception
    safe_additions = next(
        button for button in rendered.button if button.label == "Accept safe additions"
    )
    assert safe_additions.disabled

    individual = (
        next(button for button in rendered.button if button.label == "Review individually")
        .click()
        .run()
    )
    removal = next(
        checkbox
        for checkbox in individual.checkbox
        if checkbox.label == "Accept this complete deterministic replacement"
    )
    assert not removal.disabled

    selected = removal.check().run()
    accept = next(button for button in selected.button if button.label == "Accept selected bundles")
    assert not accept.disabled

    accepted = accept.click().run()

    assert not accepted.exception
    assert accepted.session_state["ai_studio_pending_draft"] is None
    assert accepted.session_state["ai_studio_draft"] == pending


@pytest.mark.unit
def test_provider_source_replacement_remains_blocked_by_inactive_scope_contract() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    base = _draft_for_source("old_source")
    pending = _draft_for_source("new_source")

    def app(base_draft: dict, pending_draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from tests.unit.test_ai_studio_pending_review_regression import (  # noqa: PLC0415
            _seed_review_state,
        )
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        _seed_review_state(
            st,
            page,
            base=base_draft,
            pending=pending_draft,
            kind="draft",
        )
        page._render_pending_draft_review()

    rendered = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()
    individual = (
        next(button for button in rendered.button if button.label == "Review individually")
        .click()
        .run()
    )
    removal = next(
        checkbox
        for checkbox in individual.checkbox
        if checkbox.label == "Explicitly include this removal"
    )

    assert not individual.exception
    assert next(
        button for button in individual.button if button.label == "Accept safe additions"
    ).disabled
    assert removal.disabled
    assert next(
        button for button in individual.button if button.label == "Accept selected bundles"
    ).disabled
    assert individual.session_state["ai_studio_pending_draft"] == pending


@pytest.mark.unit
def test_invalid_removal_displays_its_inactive_scope_validation_reason() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    base = _draft_for_source("old_source")
    pending = _draft_for_source("new_source")

    def app(base_draft: dict, pending_draft: dict) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from tests.unit.test_ai_studio_pending_review_regression import (  # noqa: PLC0415
            _seed_review_state,
        )
        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        _seed_review_state(
            st,
            page,
            base=base_draft,
            pending=pending_draft,
            kind="draft",
        )
        page._render_pending_draft_review()

    rendered = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()

    visible_text = "\n".join(
        str(element.value)
        for collection in (rendered.warning, rendered.info, rendered.markdown)
        for element in collection
    )
    assert "removes existing configuration" in visible_text
    assert "does not validate independently" in visible_text
    assert "outside the active sampled-source contract" in visible_text
