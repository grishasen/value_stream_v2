"""Outcome-first entry point for configuration authoring."""

from __future__ import annotations

import streamlit as st

from valuestream.ui import components
from valuestream.ui.context import ValueStreamContext
from valuestream.ui.instrumentation import (
    AuthoringEvent,
    AuthoringOutcome,
    AuthoringStage,
    AuthoringWorkflow,
    abandon_and_reset,
    record_event,
    start_journey,
)
from valuestream.ui.pages import ai_config_studio, config_builder

_MODE_KEY = "build_authoring_mode"


def render(ctx: ValueStreamContext) -> None:
    """Render the authoring choice or the selected workflow."""

    mode = str(st.session_state.get(_MODE_KEY) or "")
    start_journey(
        st.session_state,
        workflow=(
            AuthoringWorkflow.AI_STUDIO
            if mode == "sample"
            else AuthoringWorkflow.BUILDER
            if mode == "manual"
            else AuthoringWorkflow.BUILD
        ),
    )
    if mode in {"sample", "manual"}:
        if st.button(
            "Back to build choices",
            icon=":material/arrow_back:",
            key="build_back_to_choices",
        ):
            abandon_and_reset(
                st.session_state,
                workflow=(
                    AuthoringWorkflow.AI_STUDIO if mode == "sample" else AuthoringWorkflow.BUILDER
                ),
            )
            st.session_state.pop(_MODE_KEY, None)
            st.rerun()
        if mode == "sample":
            ai_config_studio.render(ctx)
        else:
            config_builder.render(ctx)
        return

    components.render_page_header(
        "Build",
        "Choose the shortest safe path from your data or current workspace to a validated report.",
    )
    st.write("### How do you want to begin?")
    st.caption(
        "Both paths update the same YAML catalog. Applying configuration never starts data processing."
    )
    sample_col, manual_col = st.columns(2, gap="large")
    with sample_col, st.container(border=True):
        st.write("### Start from a sample")
        st.write(
            "Preview a CSV, Parquet, or Pega export; map its fields; then create a validated "
            "deterministic or AI-assisted proposal."
        )
        st.caption("Best for a new data source · deterministic demo available · about 5 minutes")
        if st.button(
            "Start from sample",
            type="primary",
            icon=":material/upload_file:",
            key="build_choose_sample",
            width="stretch",
        ):
            st.session_state[_MODE_KEY] = "sample"
            record_event(
                st.session_state,
                event=AuthoringEvent.ENTERED,
                workflow=AuthoringWorkflow.AI_STUDIO,
                stage=AuthoringStage.ENTRY,
                outcome=AuthoringOutcome.STARTED,
                once=True,
            )
            record_event(
                st.session_state,
                event=AuthoringEvent.SAMPLE_CHOSEN,
                workflow=AuthoringWorkflow.AI_STUDIO,
                stage=AuthoringStage.SAMPLE,
                outcome=AuthoringOutcome.STARTED,
                once=True,
            )
            st.rerun()
    with manual_col, st.container(border=True):
        st.write("### Configure the current workspace")
        st.write(
            "Review existing sources, processors, metrics, and reports; apply one validated "
            "change at a time."
        )
        st.caption("Best for an existing catalog · no sample or model required")
        if st.button(
            "Configure manually",
            icon=":material/build:",
            key="build_choose_manual",
            width="stretch",
        ):
            st.session_state[_MODE_KEY] = "manual"
            record_event(
                st.session_state,
                event=AuthoringEvent.ENTERED,
                workflow=AuthoringWorkflow.BUILDER,
                stage=AuthoringStage.ENTRY,
                outcome=AuthoringOutcome.STARTED,
                once=True,
            )
            st.rerun()

    with st.expander("What changes when I apply?", expanded=False):
        st.write(
            "Value Stream validates and writes the workspace catalog transactionally. "
            "It then tells you whether the result is ready to open or requires an explicit data run."
        )


__all__ = ["render"]
