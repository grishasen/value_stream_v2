"""Copilot draft-operation, prompt, and coverage tests."""

from __future__ import annotations

import pytest

from valuestream.ai import (
    apply_draft_operations,
    draft_patches,
    merge_selected_draft_patches,
    parse_copilot_response,
    parse_coverage_response,
    prompt_for_copilot,
    prompt_for_coverage,
    run_copilot_tool_loop,
)
from valuestream.ui.pages import ai_config_studio as ai_config_studio_page
from valuestream.ui.pages.ai_config_studio import (
    DETERMINISTIC_STEPS,
    STEPS,
    STUDIO_PHASES,
    _phase_for_step,
    _phase_step_options,
)


def _base_draft() -> dict:
    return {
        "pipelines": {
            "version": 1,
            "workspace": "test",
            "sources": [
                {
                    "id": "ih",
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
                    "source": "ih",
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


@pytest.mark.unit
def test_apply_operations_upserts_metric_and_tile() -> None:
    draft = _base_draft()
    operations = [
        {
            "op": "set_metric",
            "name": "Total",
            "metric": {"source": "engagement", "kind": "formula", "expression": {"col": "Count"}},
        },
        {
            "op": "set_tile",
            "dashboard": "overview",
            "page": "engagement",
            "tile": {"id": "total", "title": "Total", "metric": "Total", "chart": "kpi_card"},
        },
    ]

    updated, summaries = apply_draft_operations(draft, operations)

    assert "Total" in updated["metrics"]["metrics"]
    tiles = updated["dashboards"]["dashboards"][0]["pages"][0]["tiles"]
    assert [tile["id"] for tile in tiles] == ["ctr", "total"]
    assert summaries == ["Added metric 'Total'", "Added tile 'overview/engagement/total'"]
    assert "Total" not in draft["metrics"]["metrics"]


@pytest.mark.unit
def test_apply_operations_set_tile_creates_dashboard_and_page() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [
            {
                "op": "set_tile",
                "dashboard": "revenue_watch",
                "page": "weekly",
                "tile": {"id": "orders", "title": "Orders", "metric": "CTR", "chart": "kpi_card"},
            }
        ],
    )

    dashboards = updated["dashboards"]["dashboards"]
    created = next(item for item in dashboards if item["id"] == "revenue_watch")
    assert created["title"] == "Revenue Watch"
    assert created["pages"][0]["id"] == "weekly"
    assert created["pages"][0]["tiles"][0]["id"] == "orders"
    assert summaries == ["Added tile 'revenue_watch/weekly/orders'"]


@pytest.mark.unit
def test_apply_operations_remove_processor_cascades() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [{"op": "remove_processor", "id": "engagement"}],
    )

    assert updated["processors"]["processors"] == []
    assert updated["metrics"]["metrics"] == {}
    assert updated["dashboards"]["dashboards"] == []
    assert summaries == [
        "Removed processor 'engagement' and its dependent metrics and tiles",
    ]


@pytest.mark.unit
def test_apply_operations_remove_metric_cascades_tiles() -> None:
    updated, _ = apply_draft_operations(
        _base_draft(),
        [{"op": "remove_metric", "name": "CTR"}],
    )

    assert updated["metrics"]["metrics"] == {}
    assert updated["dashboards"]["dashboards"] == []


@pytest.mark.unit
def test_apply_operations_remove_tile() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [{"op": "remove_tile", "dashboard": "overview", "page": "engagement", "id": "ctr"}],
    )

    assert updated["dashboards"]["dashboards"][0]["pages"][0]["tiles"] == []
    assert summaries == ["Removed tile 'overview/engagement/ctr'"]


@pytest.mark.unit
def test_apply_operations_installs_ready_builtin_recipe() -> None:
    updated, summaries = apply_draft_operations(
        _base_draft(),
        [
            {
                "op": "install_recipe",
                "recipe_id": "engagement.engagement_rate",
                "processor": "engagement",
                "metric_id": "Recipe_Engagement",
                "dashboard": "overview",
                "page": "engagement",
                "tile_id": "recipe_engagement",
            }
        ],
    )

    metric = updated["metrics"]["metrics"]["Recipe_Engagement"]
    assert metric["recipe"] == {"id": "engagement.engagement_rate", "version": 1}
    assert "overview/engagement/recipe_engagement" in {
        patch.object_id for patch in draft_patches(_base_draft(), updated)
    }
    assert summaries == [
        "Installed recipe 'engagement.engagement_rate' as metric 'Recipe_Engagement'"
    ]


@pytest.mark.unit
def test_apply_operations_rejects_unknown_and_invalid_operations() -> None:
    with pytest.raises(ValueError, match="Unknown operation"):
        apply_draft_operations(_base_draft(), [{"op": "drop_everything"}])
    with pytest.raises(ValueError, match="does not exist"):
        apply_draft_operations(_base_draft(), [{"op": "remove_metric", "name": "Missing"}])
    with pytest.raises(ValueError, match="requires a processor mapping"):
        apply_draft_operations(_base_draft(), [{"op": "set_processor", "processor": {}}])


@pytest.mark.unit
def test_patch_rejection_restores_changed_object_instead_of_deleting_it() -> None:
    base = _base_draft()
    proposed, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_metric",
                "name": "CTR",
                "metric": {
                    **base["metrics"]["metrics"]["CTR"],
                    "direction": "higher_is_better",
                },
            },
            {
                "op": "set_metric",
                "name": "Total",
                "metric": {
                    "source": "engagement",
                    "kind": "formula",
                    "expression": {"col": "Count"},
                },
            },
        ],
    )

    patches = draft_patches(base, proposed)
    assert {patch.key for patch in patches} == {"metrics:CTR", "metrics:Total"}

    reviewed = merge_selected_draft_patches(base, proposed, {"metrics:Total"})

    assert reviewed["metrics"]["metrics"]["CTR"] == base["metrics"]["metrics"]["CTR"]
    assert "Total" in reviewed["metrics"]["metrics"]


@pytest.mark.unit
def test_copilot_tool_loop_repairs_invalid_operations_before_pending_review() -> None:
    responses = iter(
        [
            '{"reply":"Adding it.","operations":[{"op":"set_metric","name":"Total",'
            '"metric":{"source":"missing","kind":"formula","expression":{"col":"Count"}}}],'
            '"questions":[]}',
            '{"reply":"Corrected the source.","operations":[{"op":"set_metric",'
            '"name":"Total","metric":{"source":"engagement","kind":"formula",'
            '"expression":{"col":"Count"}}}],"questions":[]}',
        ]
    )
    prompts: list[str] = []

    def call_model(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = run_copilot_tool_loop(
        prompt="Add Total",
        draft=_base_draft(),
        call_model=call_model,
        validate=ai_config_studio_page.validate_draft_catalog,
        max_iterations=3,
    )

    assert result.iterations == 2
    assert result.pending_draft is not None
    assert result.pending_draft["metrics"]["metrics"]["Total"]["source"] == "engagement"
    assert "Validation or operation errors" in prompts[1]


@pytest.mark.unit
def test_prompt_for_copilot_includes_step_goals_history_and_contract() -> None:
    prompt = prompt_for_copilot(
        step="9. Metrics",
        user_message="Add average revenue per customer.",
        history=[
            {"role": "user", "content": "What does this step do?"},
            {"role": "assistant", "content": "It reviews metric definitions."},
        ],
        user_goals="Weekly conversion by channel.",
        approved_schema=[{"column": "Channel", "dtype": "String", "unique": 3}],
        approved_fields=["Channel"],
        hidden_fields=["CustomerID"],
        current_draft=_base_draft(),
    )

    assert "'Metrics' studio step" in prompt
    assert "reviewing metric definitions" in prompt
    assert "Business requirements from the user:" in prompt
    assert "Weekly conversion by channel." in prompt
    assert "user: What does this step do?" in prompt
    assert "assistant: It reviews metric definitions." in prompt
    assert "User message:\nAdd average revenue per customer." in prompt
    assert '"reply": str' in prompt
    assert "set_metric" in prompt
    assert "Return valid JSON only." in prompt


@pytest.mark.unit
def test_parse_copilot_response_reads_operations_and_questions() -> None:
    turn = parse_copilot_response(
        """
```json
{
  "reply": "I can add that metric.",
  "operations": [{"op": "set_metric", "name": "Total", "metric": {"kind": "formula"}}],
  "questions": [{"question": "Which time grain?", "options": ["Day", "Month"]}]
}
```
"""
    )

    assert turn.reply == "I can add that metric."
    assert turn.operations[0]["op"] == "set_metric"
    assert turn.questions[0].question == "Which time grain?"
    assert turn.questions[0].options == ("Day", "Month")


@pytest.mark.unit
def test_parse_copilot_response_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_copilot_response("no json here")
    with pytest.raises(ValueError, match="JSON object"):
        parse_copilot_response("[1, 2, 3]")
    with pytest.raises(ValueError, match="no reply"):
        parse_copilot_response('{"reply": "", "operations": [], "questions": []}')


@pytest.mark.unit
def test_coverage_prompt_and_response_round_trip() -> None:
    prompt = prompt_for_coverage(
        user_goals="Weekly conversion by channel.\nAverage revenue per customer.",
        draft=_base_draft(),
    )

    assert "Business requirements:" in prompt
    assert "Weekly conversion by channel." in prompt
    assert "- CTR" in prompt
    assert "overview/engagement/ctr" in prompt

    rows = parse_coverage_response(
        """
[
  {"requirement": "Weekly conversion by channel", "status": "covered",
   "metrics": ["CTR"], "tiles": ["overview/engagement/ctr"], "note": "CTR trend by day."},
  {"requirement": "Average revenue per customer", "status": "missing",
   "metrics": [], "tiles": [], "note": "No revenue metric exists."}
]
"""
    )

    assert [row.status for row in rows] == ["covered", "missing"]
    assert rows[0].metrics == ("CTR",)
    assert rows[1].note == "No revenue metric exists."


@pytest.mark.unit
def test_parse_coverage_response_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        parse_coverage_response('[{"requirement": "X", "status": "unknown"}]')


@pytest.mark.unit
def test_coverage_response_downgrades_hallucinated_references() -> None:
    rows = parse_coverage_response(
        '[{"requirement":"Revenue","status":"covered","metrics":["Missing"],'
        '"tiles":["overview/missing/tile"],"note":"Available."}]',
        draft=_base_draft(),
    )

    assert rows[0].status == "missing"
    assert rows[0].metrics == ()
    assert rows[0].tiles == ()
    assert "Ignored unknown draft references" in rows[0].note


@pytest.mark.unit
def test_phase_helpers_cover_every_step_in_both_step_lists() -> None:
    for steps in (STEPS, DETERMINISTIC_STEPS):
        covered: list[str] = []
        for phase, _ in STUDIO_PHASES:
            options = _phase_step_options(phase, steps)
            covered.extend(options)
            for step in options:
                assert _phase_for_step(step, steps) == phase
        assert covered == steps


@pytest.mark.unit
def test_copilot_panel_holds_operations_in_pending_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def fake_call_litellm(settings: object, prompt: str, **kwargs: object) -> str:
        return (
            '{"reply": "Added a total metric.", '
            '"operations": [{"op": "set_metric", "name": "Total", '
            '"metric": {"source": "engagement", "kind": "formula", '
            '"expression": {"col": "Count"}}}], '
            '"questions": []}'
        )

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        working = pl.DataFrame({"Channel": ["Web", "Mobile"]})
        page._render_copilot_panel("9. Metrics", working, ["Channel"])

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()

    assert not at.exception
    at.chat_input[0].set_value("Add a Total metric.").run()

    assert not at.exception
    pending = at.session_state["ai_studio_pending_draft"]
    assert "Total" in pending["metrics"]["metrics"]
    assert at.session_state["ai_studio_pending_kind"] == "copilot"
    history = at.session_state["ai_studio_copilot_history"]
    assert history[-1]["role"] == "assistant"
    assert "Added metric 'Total'" in history[-1]["content"]
    assert at.session_state["ai_studio_copilot_input"] is None


@pytest.mark.unit
def test_copilot_panel_renders_clarifying_question_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def fake_call_litellm(settings: object, prompt: str, **kwargs: object) -> str:
        return (
            '{"reply": "Which grain should the metric use?", "operations": [], '
            '"questions": [{"question": "Which time grain?", "options": ["Day", "Month"]}]}'
        )

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call_litellm)

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_draft"] = draft
        st.session_state.setdefault("ai_studio_pending_draft", None)
        working = pl.DataFrame({"Channel": ["Web", "Mobile"]})
        page._render_copilot_panel("9. Metrics", working, ["Channel"])

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    at.chat_input[0].set_value("Add a trend metric.").run()

    assert not at.exception
    assert at.session_state["ai_studio_pending_draft"] is None
    labels = [widget.label for widget in at.button]
    assert "Day" in labels
    assert "Month" in labels

    next(widget for widget in at.button if widget.label == "Day").click().run()

    assert not at.exception
    history = at.session_state["ai_studio_copilot_history"]
    assert {"role": "user", "content": "Day"} in history


@pytest.mark.unit
def test_copilot_question_before_first_draft_does_not_accept_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    monkeypatch.setattr(
        ai_config_studio_page,
        "call_litellm",
        lambda *args, **kwargs: (
            '{"reply":"This step reviews metrics.","operations":[],"questions":[]}'
        ),
    )
    monkeypatch.setattr(
        ai_config_studio_page,
        "_build_draft_catalog",
        lambda working, approved_fields: _base_draft(),
    )

    def app() -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state.setdefault("ai_studio_draft", None)
        st.session_state.setdefault("ai_studio_pending_draft", None)
        page._render_copilot_panel(
            "9. Metrics",
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
        )

    at = AppTest.from_function(app).run()
    at.chat_input[0].set_value("What happens on this step?").run()

    assert not at.exception
    assert at.session_state["ai_studio_draft"] is None
    assert at.session_state["ai_studio_pending_draft"] is None


@pytest.mark.unit
def test_pending_review_disables_copilot_and_preserves_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    calls = 0

    def fake_call(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return '{"reply":"Unexpected","operations":[],"questions":[]}'

    monkeypatch.setattr(ai_config_studio_page, "call_litellm", fake_call)
    base = _base_draft()
    pending, _ = apply_draft_operations(
        base,
        [
            {
                "op": "set_metric",
                "name": "Total",
                "metric": {
                    "source": "engagement",
                    "kind": "formula",
                    "expression": {"col": "Count"},
                },
            }
        ],
    )

    def app(base_draft: dict, pending_draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state[page.AI_CALLS_ENABLED_STATE_KEY] = True
        st.session_state["ai_studio_ai_model"] = "openai/gpt-test"
        st.session_state["ai_studio_draft"] = base_draft
        st.session_state["ai_studio_pending_draft"] = pending_draft
        st.session_state["ai_studio_pending_base_draft"] = base_draft
        st.session_state["ai_studio_pending_kind"] = "revision"
        page._render_copilot_panel(
            "9. Metrics",
            pl.DataFrame({"Channel": ["Web"]}),
            ["Channel"],
        )

    at = AppTest.from_function(
        app,
        kwargs={"base_draft": base, "pending_draft": pending},
    ).run()

    assert not at.exception
    assert at.chat_input[0].disabled
    assert at.session_state["ai_studio_pending_draft"] == pending
    assert calls == 0


@pytest.mark.unit
def test_new_sample_identity_resets_draft_and_copilot_with_same_columns() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict) -> None:
        import polars as pl  # noqa: PLC0415 - isolated AppTest source
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        frame = pl.DataFrame({"Channel": ["Web"]})
        st.session_state.setdefault("ai_studio_sample_identity", "first-file")
        st.session_state.setdefault("ai_studio_user_goals", "Keep this requirement")
        page._initialize_state(frame)
        if st.button("Switch sample"):
            st.session_state["ai_studio_draft"] = draft
            st.session_state["ai_studio_copilot_history"] = [
                {"role": "user", "content": "Old sample question"}
            ]
            st.session_state["ai_studio_sample_identity"] = "second-file"
            page._initialize_state(frame)

    at = AppTest.from_function(app, kwargs={"draft": _base_draft()}).run()
    next(button for button in at.button if button.label == "Switch sample").click().run()

    assert not at.exception
    assert at.session_state["ai_studio_draft"] is None
    assert at.session_state["ai_studio_copilot_history"] == []
    assert at.session_state["ai_studio_user_goals"] == "Keep this requirement"


@pytest.mark.unit
def test_phase_statuses_distinguish_ready_to_publish_from_published() -> None:
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415 - test-only dependency

    def app(draft: dict, published: bool) -> None:
        import streamlit as st  # noqa: PLC0415 - isolated AppTest source

        from valuestream.ui.pages import ai_config_studio as page  # noqa: PLC0415

        st.session_state["ai_studio_draft"] = draft
        st.session_state["ai_studio_pending_draft"] = None
        st.session_state["ai_studio_published_signature"] = (
            page._draft_signature(draft) if published else ""
        )
        st.session_state["phase_statuses"] = page._phase_statuses(["Channel"])

    ready = AppTest.from_function(app, kwargs={"draft": _base_draft(), "published": False}).run()
    published = AppTest.from_function(
        app,
        kwargs={"draft": _base_draft(), "published": True},
    ).run()

    assert ready.session_state["phase_statuses"]["Review"] == "complete"
    assert ready.session_state["phase_statuses"]["Publish"] == "attention"
    assert published.session_state["phase_statuses"]["Publish"] == "complete"
