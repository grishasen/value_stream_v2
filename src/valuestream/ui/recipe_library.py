"""Shared Streamlit workflow for browsing and installing KPI recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import streamlit as st
import yaml

from valuestream.config import canonical as config_canonical
from valuestream.config import model
from valuestream.recipes import (
    KpiRecipe,
    RecipeBindingOption,
    RecipeInput,
    RecipeReadiness,
    instantiate_metric,
    instantiate_tile,
    load_builtin_kpi_recipes,
    processor_recipe_fields,
    processor_with_recipe_states,
    recipe_algorithm_label,
    recipe_binding_attribute,
    recipe_binding_options,
    recipe_readiness,
    unique_artifact_id,
)
from valuestream.ui import config_help
from valuestream.utils.names import dedupe_strings as _dedupe


@dataclass(frozen=True)
class ReportPageTarget:
    dashboard_id: str
    dashboard_title: str
    page_id: str
    page_title: str


@dataclass(frozen=True)
class RecipeInstallRequest:
    recipe_id: str
    recipe_version: int
    metric_id: str
    metric_def: dict[str, Any]
    processor_id: str = ""
    state_additions: dict[str, dict[str, Any]] | None = None
    processor_def: dict[str, Any] | None = None
    report_target: ReportPageTarget | None = None
    tile_def: dict[str, Any] | None = None
    materialization: RecipeMaterializationPlan | None = None


@dataclass(frozen=True)
class RecipeMaterializationPlan:
    """Source-run handoff created when a recipe widens a processor contract."""

    source_id: str
    processor_id: str
    state_names: tuple[str, ...]
    source_fields: tuple[str, ...]
    current_computation_hash: str
    proposed_computation_hash: str


@dataclass(frozen=True)
class RecipeBindingSelection:
    bindings: dict[str, str]
    state_additions: dict[str, dict[str, Any]]


def render_recipe_library(  # noqa: PLR0911, PLR0915
    *,
    catalog: model.Catalog,
    key_prefix: str,
    submit_label: str,
    expanded: bool = False,
) -> RecipeInstallRequest | None:
    """Render one reusable recipe browser and return a validated install request."""

    library = load_builtin_kpi_recipes()
    processors = catalog.processors.processors
    metric_names = set(catalog.metrics.metrics)
    dashboards = catalog.dashboards.dashboards
    with st.expander("**KPI recipe library**", expanded=expanded, icon=":material/library_books:"):
        st.caption(
            "Browse business definitions, inspect calculation and approximation guidance, "
            "then materialize a reviewed metric and optional report tile."
        )
        search = st.text_input(
            "Search recipes",
            key=f"{key_prefix}_search",
            placeholder="Search KPI, question, tag, or calculation",
            help=config_help.field_help("recipe.search"),
        )
        domains = sorted({recipe.domain for recipe in library.recipes}, key=str.casefold)
        domain = st.selectbox(
            "Business domain",
            ["All domains", *domains],
            key=f"{key_prefix}_domain",
            help=config_help.field_help("recipe.domain"),
        )
        recipes = [
            recipe
            for recipe in library.recipes
            if domain in ("All domains", recipe.domain) and _matches_search(recipe, search)
        ]
        if not recipes:
            st.info("No recipes match the current search and domain filter.")
            return None

        recipe = st.selectbox(
            "Recipe",
            recipes,
            format_func=lambda item: f"{item.title} · {item.domain}",
            key=f"{key_prefix}_recipe",
            help=config_help.field_help("recipe.selector"),
        )
        _render_recipe_description(recipe)

        compatible = [
            processor for processor in processors if processor.kind in recipe.processor_kinds
        ]
        if not compatible:
            st.warning(
                "No compatible processor is configured. This recipe requires: "
                + ", ".join(recipe.processor_kinds)
                + "."
            )
            return None

        readiness_by_id = {
            processor.id: recipe_readiness(recipe, processor) for processor in compatible
        }
        processors_by_id = {processor.id: processor for processor in compatible}
        processor_id = st.selectbox(
            "Source processor",
            list(processors_by_id),
            format_func=lambda value: _processor_label(
                processors_by_id[value],
                readiness_by_id[value].status,
                configurable=any(
                    recipe_input.selection == "field_algorithm" for recipe_input in recipe.inputs
                ),
            ),
            key=f"{key_prefix}_processor_{_key_fragment(recipe.id)}",
            help=config_help.field_help("recipe.processor"),
        )
        processor = processors_by_id[processor_id]
        readiness = readiness_by_id[processor_id]
        mapping_key = f"{key_prefix}_{_key_fragment(recipe.id)}_{_key_fragment(processor.id)}"
        selection = _render_recipe_bindings(
            recipe,
            processor,
            readiness,
            key_prefix=mapping_key,
        )
        bindings_complete = len(selection.bindings) == len(recipe.inputs)
        _render_readiness(
            readiness.status,
            readiness.messages,
            bindings_complete=bindings_complete,
            state_additions=selection.state_additions,
        )

        targets = _report_targets(dashboards)
        default_metric_id = unique_artifact_id(recipe.default_metric_id, metric_names)
        install_key = f"{mapping_key}_{_key_fragment(default_metric_id)}_install"
        with st.container(border=True):
            metric_id = st.text_input(
                "Metric ID",
                value=default_metric_id,
                key=f"{install_key}_metric_id",
                help=config_help.field_help("metric.id"),
            ).strip()
            add_to_report = st.toggle(
                "Add the recommended tile to a report",
                value=bool(targets),
                key=f"{install_key}_add_to_report",
                disabled=not targets,
                help=(
                    f"{config_help.field_help('recipe.add_tile')}\n\n"
                    "Create a dashboard page first to enable report placement."
                    if not targets
                    else config_help.field_help("recipe.add_tile")
                ),
            )
            target_key = (
                st.selectbox(
                    "Report page",
                    list(targets),
                    format_func=lambda value: targets[value][0],
                    key=f"{install_key}_report_target",
                    disabled=not add_to_report,
                    help=config_help.field_help("recipe.report_page"),
                )
                if targets
                else ""
            )
            duplicate_metric = metric_id in metric_names
            review_requested = st.button(
                "Review changes",
                icon=":material/preview:",
                key=f"{install_key}_review",
                disabled=(not bindings_complete or not metric_id or duplicate_metric),
            )

        if not metric_id:
            st.error("Metric ID is required.")
            return None
        if duplicate_metric:
            st.error(f"Metric ID `{metric_id}` already exists. Choose a new ID.")
            return None
        if not bindings_complete:
            return None
        try:
            report_target: ReportPageTarget | None = None
            tile_id = ""
            if add_to_report and target_key:
                _, report_target, existing_tile_ids = targets[target_key]
                tile_id = unique_artifact_id(f"{_catalog_slug(metric_id)}_tile", existing_tile_ids)
            request = build_recipe_install_request(
                catalog=catalog,
                recipe=recipe,
                processor=processor,
                metric_id=metric_id,
                bindings=selection.bindings,
                state_additions=selection.state_additions,
                report_target=report_target,
                tile_id=tile_id,
            )
        except (TypeError, ValueError) as exc:
            st.error(str(exc))
            return None

        fingerprint = recipe_install_fingerprint(request)
        preview_key = f"{install_key}_preview_fingerprint"
        if review_requested:
            st.session_state[preview_key] = fingerprint
        reviewed_fingerprint = st.session_state.get(preview_key)
        if reviewed_fingerprint != fingerprint:
            if reviewed_fingerprint:
                st.info("The recipe selections changed. Review the updated YAML before applying.")
            return None

        if _render_install_preview(
            request,
            submit_label=submit_label,
            key=f"{install_key}_{fingerprint[:12]}",
        ):
            st.session_state.pop(preview_key, None)
            return request
        return None


def build_recipe_install_request(
    *,
    catalog: model.Catalog,
    recipe: KpiRecipe,
    processor: model.Processor,
    metric_id: str,
    bindings: dict[str, str],
    state_additions: dict[str, dict[str, Any]],
    report_target: ReportPageTarget | None = None,
    tile_id: str = "",
) -> RecipeInstallRequest:
    """Build the exact, validated catalog patch shown by both Studio surfaces."""

    configured_processor = processor_with_recipe_states(processor, state_additions)
    metric_def = instantiate_metric(recipe, configured_processor, metric_id, bindings)
    processor_def = _processor_yaml_definition(configured_processor) if state_additions else None
    tile_def = (
        instantiate_tile(recipe, metric_id, tile_id)
        if report_target is not None and tile_id
        else None
    )
    materialization = _materialization_plan(
        catalog,
        processor,
        configured_processor,
        state_additions,
    )
    return RecipeInstallRequest(
        recipe_id=recipe.id,
        recipe_version=recipe.version,
        metric_id=metric_id,
        metric_def=metric_def,
        processor_id=processor.id,
        state_additions=state_additions or None,
        processor_def=processor_def,
        report_target=report_target,
        tile_def=tile_def,
        materialization=materialization,
    )


def _processor_yaml_definition(processor: model.Processor) -> dict[str, Any]:
    """Serialize a processor exactly as the shared installer will write it."""

    data = processor.model_dump(mode="json", by_alias=True, exclude_none=True)
    group_by = data.pop("group_by", None)
    if group_by:
        data["dimensions"] = group_by
    if not processor.states:
        data.pop("states", None)
    return data


def _materialization_plan(
    catalog: model.Catalog,
    processor: model.Processor,
    configured_processor: model.Processor,
    state_additions: dict[str, dict[str, Any]],
) -> RecipeMaterializationPlan | None:
    if not state_additions:
        return None
    proposed_processors = catalog.processors.model_copy(
        update={
            "processors": [
                configured_processor if item.id == processor.id else item
                for item in catalog.processors.processors
            ]
        }
    )
    proposed_catalog = catalog.model_copy(update={"processors": proposed_processors})
    source_fields = tuple(
        dict.fromkeys(
            str(definition.get("source_column") or "")
            for definition in state_additions.values()
            if definition.get("source_column")
        )
    )
    return RecipeMaterializationPlan(
        source_id=processor.source,
        processor_id=processor.id,
        state_names=tuple(state_additions),
        source_fields=source_fields,
        current_computation_hash=config_canonical.processor_computation_hash(catalog, processor),
        proposed_computation_hash=config_canonical.processor_computation_hash(
            proposed_catalog, configured_processor
        ),
    )


def recipe_install_preview_files(request: RecipeInstallRequest) -> dict[str, str]:
    """Return exact generated YAML patches in deterministic write order."""

    files: dict[str, str] = {}
    if request.processor_def:
        files["processors.yaml"] = yaml.safe_dump(
            {"processors": [request.processor_def]}, sort_keys=False
        )
    files["metrics.yaml"] = yaml.safe_dump(
        {"metrics": {request.metric_id: request.metric_def}}, sort_keys=False
    )
    if request.report_target and request.tile_def:
        target = request.report_target
        files["dashboards.yaml"] = yaml.safe_dump(
            {
                "dashboards": [
                    {
                        "id": target.dashboard_id,
                        "title": target.dashboard_title,
                        "pages": [
                            {
                                "id": target.page_id,
                                "title": target.page_title,
                                "tiles": [request.tile_def],
                            }
                        ],
                    }
                ]
            },
            sort_keys=False,
        )
    return files


def recipe_install_fingerprint(request: RecipeInstallRequest) -> str:
    """Identify the exact reviewed request so changed inputs invalidate preview."""

    return config_canonical.config_hash(
        {
            "recipe_id": request.recipe_id,
            "recipe_version": request.recipe_version,
            "metric_id": request.metric_id,
            "metric_def": request.metric_def,
            "processor_id": request.processor_id,
            "state_additions": request.state_additions,
            "processor_def": request.processor_def,
            "report_target": (asdict(request.report_target) if request.report_target else None),
            "tile_def": request.tile_def,
            "materialization": (
                asdict(request.materialization) if request.materialization else None
            ),
        }
    )


def _render_install_preview(
    request: RecipeInstallRequest,
    *,
    submit_label: str,
    key: str,
) -> bool:
    """Render the reviewed YAML patch and explicit materialization handoff."""

    with st.container(border=True):
        st.write("### Review catalog changes")
        st.caption(
            "These are the exact generated definitions that will be merged into the "
            "catalog or AI draft. Existing unrelated definitions are preserved."
        )
        for filename, contents in recipe_install_preview_files(request).items():
            with st.expander(filename, expanded=filename == "metrics.yaml"):
                st.code(contents, language="yaml")

        plan = request.materialization
        if plan is None:
            st.success(
                "Required processor states are already configured. Installation does not "
                "change the processor computation contract."
            )
            st.caption(
                "A workspace with no matching aggregates yet still needs its normal source run."
            )
        else:
            state_list = ", ".join(f"`{name}`" for name in plan.state_names)
            field_list = ", ".join(f"`{field}`" for field in plan.source_fields)
            st.warning(
                f"Materialization required: source `{plan.source_id}` must run after "
                f"installation to populate {state_list}."
            )
            if field_list:
                st.caption(f"Source fields: {field_list}")
            st.caption(
                "Processor computation hash: "
                f"`{plan.current_computation_hash[:12]}` → "
                f"`{plan.proposed_computation_hash[:12]}`. Existing aggregates are not "
                "mixed with the new contract."
            )

        return st.button(
            submit_label,
            type="primary",
            icon=":material/add_circle:",
            key=f"{key}_confirm",
        )


def _render_recipe_description(recipe: KpiRecipe) -> None:
    st.write(f"#### {recipe.title}")
    st.write(recipe.summary)
    st.caption(
        f"{recipe.domain} · {recipe.maturity.title()} · v{recipe.version} · "
        + " · ".join(recipe.tags)
    )
    if recipe.business_questions:
        st.markdown(
            "**Business questions**\n\n"
            + "\n".join(f"- {question}" for question in recipe.business_questions)
        )
    with st.container(border=True):
        st.markdown(f"**Calculation:** `{recipe.method.calculation}`")
        st.write(f"**Method:** {recipe.method.algorithm}")
        st.write(f"**Accuracy:** {recipe.method.accuracy.title()}")
        if recipe.method.caveat:
            st.caption(recipe.method.caveat)
        st.caption(
            f"Recommended report: {recipe.report.chart.replace('_', ' ').title()} · "
            f"{recipe.report.placement.replace('_', ' ').title()}"
        )


def _render_recipe_bindings(  # noqa: PLR0912
    recipe: KpiRecipe,
    processor: model.Processor,
    readiness: RecipeReadiness,
    *,
    key_prefix: str,
) -> RecipeBindingSelection:
    bindings: dict[str, str] = {}
    state_additions: dict[str, dict[str, Any]] = {}
    selected_options: dict[str, RecipeBindingOption] = {}
    working_processor = processor
    for item in recipe.inputs:
        reference_option = (
            selected_options.get(item.same_attribute_as) if item.same_attribute_as else None
        )
        proposal_fields: list[str] = []
        if item.selection == "field_algorithm":
            proposal_fields = processor_recipe_fields(working_processor)
        elif reference_option is not None and reference_option.field:
            proposal_fields = [reference_option.field]
        options = recipe_binding_options(
            item,
            working_processor,
            readiness.input_options.get(item.role, ()),
            proposal_fields=proposal_fields,
        )
        if item.different_from and item.different_from in bindings:
            options = [
                option for option in options if option.value != bindings[item.different_from]
            ]
        if item.same_attribute_as:
            if reference_option is None:
                continue
            options = _paired_options(item, working_processor, reference_option, options)

        resolved = readiness.resolved_inputs.get(item.role, "")
        selected: RecipeBindingOption | None
        if item.selection == "automatic" and len(options) == 1:
            selected = options[0]
            if not item.same_attribute_as:
                st.text_input(
                    item.label,
                    value=selected.label,
                    disabled=True,
                    key=f"{key_prefix}_{item.role}_automatic",
                    help=item.description or config_help.field_help("recipe.binding"),
                )
        elif item.selection == "field_algorithm":
            selected = _render_field_algorithm_binding(
                item,
                options,
                resolved=resolved,
                key_prefix=key_prefix,
            )
        else:
            selected = _render_choice_binding(
                item,
                options,
                resolved=resolved,
                key_prefix=key_prefix,
            )
        if selected is not None:
            bindings[item.role] = selected.value
            selected_options[item.role] = selected
            if selected.state_definition:
                state_additions[selected.value] = selected.state_definition
                working_processor = processor_with_recipe_states(
                    working_processor,
                    {selected.value: selected.state_definition},
                )

    if any(item.same_attribute_as and item.role in bindings for item in recipe.inputs):
        st.caption(
            "Positive and negative outcome aggregates are paired automatically for the "
            "selected score field."
        )
    _render_technical_bindings(recipe, selected_options)
    return RecipeBindingSelection(
        bindings=bindings,
        state_additions=state_additions,
    )


def _render_field_algorithm_binding(  # noqa: PLR0912
    item: RecipeInput,
    options: list[RecipeBindingOption],
    *,
    resolved: str,
    key_prefix: str,
) -> RecipeBindingOption | None:
    if not options:
        st.text_input(
            item.label,
            value="Compatible aggregate field required",
            disabled=True,
            key=f"{key_prefix}_{item.role}_missing",
            help=item.description or config_help.field_help("recipe.binding"),
        )
        return None

    resolved_option = next((option for option in options if option.value == resolved), None)
    fields = _dedupe([option.field or _option_subject(option) for option in options])
    default_field = resolved_option.field if resolved_option else ""
    if not default_field and len(fields) == 1:
        default_field = fields[0]
    field_key = f"{key_prefix}_{item.role}_field"
    field_values = fields if default_field else ["", *fields]
    if st.session_state.get(field_key) not in field_values:
        st.session_state.pop(field_key, None)
    selected_field = st.selectbox(
        item.label,
        field_values,
        index=field_values.index(default_field) if default_field else 0,
        format_func=lambda value: value or f"Select {item.label.casefold()}",
        key=field_key,
        help=item.description or config_help.field_help("recipe.binding"),
    )
    if not selected_field:
        return None

    field_options = [
        option for option in options if (option.field or _option_subject(option)) == selected_field
    ]
    unavailable_preferred = [
        recipe_algorithm_label(state_type)
        for state_type in item.preferred_state_types
        if state_type not in {option.state_type for option in field_options}
    ]
    if unavailable_preferred:
        st.caption(
            f"{', '.join(unavailable_preferred)} is recommended but is not materialized "
            f"for {selected_field}; using it requires a processor change and backfill."
        )
    algorithms = _dedupe([option.algorithm for option in field_options])
    resolved_algorithm = ""
    for state_type in item.preferred_state_types:
        if preferred := next(
            (option.algorithm for option in field_options if option.state_type == state_type),
            "",
        ):
            resolved_algorithm = preferred
            break
    if (
        not resolved_algorithm
        and resolved_option
        and (resolved_option.field or _option_subject(resolved_option)) == selected_field
    ):
        resolved_algorithm = resolved_option.algorithm
    if not resolved_algorithm and len(algorithms) == 1:
        resolved_algorithm = algorithms[0]

    if len(algorithms) == 1:
        selected_algorithm = st.text_input(
            "Algorithm",
            value=algorithms[0],
            disabled=True,
            key=f"{key_prefix}_{item.role}_algorithm_fixed",
            help=config_help.field_help("recipe.algorithm"),
        )
    else:
        algorithm_key = f"{key_prefix}_{item.role}_algorithm"
        if st.session_state.get(algorithm_key) not in algorithms:
            st.session_state.pop(algorithm_key, None)
        selected_algorithm = st.segmented_control(
            "Algorithm",
            algorithms,
            default=resolved_algorithm or None,
            selection_mode="single",
            key=algorithm_key,
            help=config_help.field_help("recipe.algorithm"),
        )
    if not selected_algorithm:
        return None

    matching = [option for option in field_options if option.algorithm == selected_algorithm]
    if len(matching) == 1:
        selected = matching[0]
        if not selected.configured:
            st.caption(
                "This algorithm is not configured for the selected field yet. The recipe "
                "will add the processor state; run ingestion for a new workspace or "
                "backfill existing aggregates before querying it."
            )
        return selected
    scopes = _dedupe([option.scope for option in matching])
    if scopes and len(scopes) == len(matching):
        return st.selectbox(
            "Population",
            matching,
            format_func=lambda option: option.scope,
            key=f"{key_prefix}_{item.role}_population",
            help=config_help.field_help("recipe.population"),
        )
    st.warning(
        "Multiple persisted aggregates have the same business field and algorithm but no "
        "distinguishing population metadata. Clarify the processor states before installing "
        "this recipe."
    )
    return None


def _render_choice_binding(
    item: RecipeInput,
    options: list[RecipeBindingOption],
    *,
    resolved: str,
    key_prefix: str,
) -> RecipeBindingOption | None:
    if not options:
        st.text_input(
            item.label,
            value="Compatible aggregate input required",
            disabled=True,
            key=f"{key_prefix}_{item.role}_missing",
            help=item.description or config_help.field_help("recipe.binding"),
        )
        return None
    if len(options) == 1:
        option = options[0]
        st.text_input(
            item.label,
            value=option.label,
            disabled=True,
            key=f"{key_prefix}_{item.role}_fixed",
            help=item.description or config_help.field_help("recipe.binding"),
        )
        return option
    resolved_option = next((option for option in options if option.value == resolved), None)
    widget_options: list[RecipeBindingOption | None] = (
        options if resolved_option else [None, *options]
    )
    return st.selectbox(
        item.label,
        widget_options,
        index=widget_options.index(resolved_option) if resolved_option else 0,
        format_func=lambda option: option.label if option else f"Select {item.label.casefold()}",
        key=f"{key_prefix}_{item.role}_choice",
        help=item.description or config_help.field_help("recipe.binding"),
    )


def _paired_options(
    item: RecipeInput,
    processor: model.Processor,
    reference: RecipeBindingOption,
    options: list[RecipeBindingOption],
) -> list[RecipeBindingOption]:
    if not item.match_attribute:
        return options
    if item.match_attribute in {"score_property", "source_column"} and reference.field:
        return [option for option in options if option.field == reference.field]
    if item.match_attribute == "type" and reference.state_type:
        return [option for option in options if option.state_type == reference.state_type]
    reference_value = recipe_binding_attribute(
        processor,
        reference.value,
        item.match_attribute,
    )
    if not reference_value:
        return options
    return [
        option
        for option in options
        if recipe_binding_attribute(processor, option.value, item.match_attribute)
        == reference_value
    ]


def _render_technical_bindings(
    recipe: KpiRecipe,
    selected: dict[str, RecipeBindingOption],
) -> None:
    if not selected:
        return
    with st.expander("Technical aggregate bindings", expanded=False):
        by_role = {item.role: item for item in recipe.inputs}
        for role, option in selected.items():
            st.caption(f"{by_role[role].label}: {option.technical_detail}")


def _render_readiness(
    status: str,
    messages: tuple[str, ...],
    *,
    bindings_complete: bool,
    state_additions: dict[str, dict[str, Any]],
) -> None:
    label = status.replace("_", " ").title()
    if bindings_complete and state_additions:
        count = len(state_additions)
        st.warning(
            f"Ready to configure: {count} processor state{'s' if count != 1 else ''} will "
            "be added. Run ingestion for a new workspace or backfill existing aggregates "
            "before querying the new metric."
        )
    elif status == "backfill_required":
        st.warning(f"{label}: " + (" ".join(messages) or "Processor changes are required."))
    elif bindings_complete:
        st.success("Ready to install: all business inputs map to configured processor states.")
    elif status == "mapping_required":
        st.info(f"{label}: select each required business input above.")
    else:
        st.warning(f"{label}: " + (" ".join(messages) or "Processor changes are required."))


def _option_subject(option: RecipeBindingOption) -> str:
    return option.label.rsplit(" · ", maxsplit=1)[0]


def _matches_search(recipe: KpiRecipe, search: str) -> bool:
    query = search.strip().casefold()
    if not query:
        return True
    haystack = " ".join(
        [
            recipe.id,
            recipe.title,
            recipe.domain,
            recipe.summary,
            *recipe.business_questions,
            *recipe.tags,
            recipe.method.calculation,
            recipe.method.algorithm,
        ]
    ).casefold()
    return query in haystack


def _processor_label(
    processor: model.Processor,
    status: str,
    *,
    configurable: bool = False,
) -> str:
    readiness = {
        "ready": "Ready",
        "mapping_required": "Choices available",
        "backfill_required": "Configuration available" if configurable else "Backfill required",
        "incompatible": "Incompatible",
    }.get(status, status.replace("_", " ").title())
    return f"{processor.id} · {processor.kind.replace('_', ' ').title()} · {readiness}"


def _report_targets(
    dashboards: list[model.Dashboard],
) -> dict[str, tuple[str, ReportPageTarget, set[str]]]:
    targets: dict[str, tuple[str, ReportPageTarget, set[str]]] = {}
    for dashboard in dashboards:
        for page in dashboard.pages:
            key = f"{dashboard.id}::{page.id}"
            targets[key] = (
                f"{dashboard.title} / {page.title}",
                ReportPageTarget(
                    dashboard_id=dashboard.id,
                    dashboard_title=dashboard.title,
                    page_id=page.id,
                    page_title=page.title,
                ),
                {tile.id for tile in page.tiles},
            )
    return targets


def _key_fragment(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _catalog_slug(value: str) -> str:
    slug = "_".join("".join(char.lower() if char.isalnum() else " " for char in value).split())
    return slug or "metric"


__all__ = [
    "RecipeBindingSelection",
    "RecipeInstallRequest",
    "RecipeMaterializationPlan",
    "ReportPageTarget",
    "build_recipe_install_request",
    "recipe_install_fingerprint",
    "recipe_install_preview_files",
    "render_recipe_library",
]
