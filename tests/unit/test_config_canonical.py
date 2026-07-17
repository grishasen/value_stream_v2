"""Canonicalization and hashing tests.

Two YAML files that parse to the same AST must hash identically. Any
behavior-changing edit must change the hash.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from valuestream.config import model
from valuestream.config.canonical import (
    canonicalize,
    catalog_config_hash,
    config_hash,
    processor_computation_config,
    processor_computation_hash,
    processor_config_hash,
    serialize,
    source_computation_hash,
)
from valuestream.config.loader import load

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_WS = REPO_ROOT / "examples" / "demo"


@pytest.mark.unit
class TestCanonicalize:
    def test_dict_keys_sorted(self) -> None:
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        assert canonicalize(a) == canonicalize(b)
        assert serialize(a) == serialize(b)

    def test_nested_dict_keys_sorted(self) -> None:
        a = {"x": {"b": 1, "a": 2}, "y": [3, 1, 2]}
        b = {"y": [3, 1, 2], "x": {"a": 2, "b": 1}}
        assert config_hash(a) == config_hash(b)

    def test_integral_float_collapses_to_int(self) -> None:
        assert canonicalize(1.0) == 1
        assert canonicalize(1) == 1
        assert config_hash({"x": 1}) == config_hash({"x": 1.0})

    def test_non_integral_float_preserved(self) -> None:
        assert canonicalize(1.5) == 1.5
        assert config_hash({"x": 1.5}) != config_hash({"x": 1})

    def test_bool_not_collapsed_to_int(self) -> None:
        # In Python, ``bool`` is a subclass of ``int`` (True == 1). The
        # canonicalizer must keep them distinct so YAML "yes" doesn't hash
        # the same as YAML "1".
        assert canonicalize(True) is True
        assert config_hash({"x": True}) != config_hash({"x": 1})

    def test_none_dropped(self) -> None:
        a = {"x": 1, "y": None}
        b = {"x": 1}
        assert canonicalize(a) == canonicalize(b)
        assert config_hash(a) == config_hash(b)

    def test_when_then_rewrites_to_case(self) -> None:
        when_then = {
            "op": "when_then",
            "cond": {"col": "X"},
            "then": {"lit": 1},
            "else": {"lit": 0},
        }
        case = {
            "op": "case",
            "when": [{"cond": {"col": "X"}, "then": {"lit": 1}}],
            "else": {"lit": 0},
        }
        assert config_hash(when_then) == config_hash(case)

    def test_meaningful_change_changes_hash(self) -> None:
        before = {"op": "eq", "column": "X", "value": 1}
        after = {"op": "eq", "column": "X", "value": 2}
        assert config_hash(before) != config_hash(after)

    def test_pydantic_model_canonicalizes(self) -> None:
        # A Pydantic model and an equivalent dict must hash the same.
        m = model.StateSpec(type="count")
        d = {"type": "count"}
        assert config_hash(m) == config_hash(d)

    def test_serialize_returns_bytes(self) -> None:
        out = serialize({"a": 1})
        assert isinstance(out, bytes)
        assert out == b'{"a":1}'


@pytest.mark.unit
class TestCatalogHash:
    def test_demo_catalog_hash_is_deterministic(self) -> None:
        a = load(DEMO_WS)
        b = load(DEMO_WS)
        assert catalog_config_hash(a) == catalog_config_hash(b)

    def test_processor_hash_changes_with_state(self) -> None:
        catalog = load(DEMO_WS)
        engagement = next(p for p in catalog.processors.processors if p.id == "ih_engagement")
        before = processor_config_hash(engagement)

        # Mutate via a fresh model — original instance stays clean.
        d = engagement.model_dump(by_alias=True, exclude_none=True)
        d["group_by"] = [*d["group_by"], "Extra"]
        after = processor_config_hash(model.BinaryOutcomeProcessor.model_validate(d))
        assert before != after

    def test_computation_hash_includes_source_but_excludes_presentation(self) -> None:
        catalog = load(DEMO_WS)
        engagement = next(p for p in catalog.processors.processors if p.id == "ih_engagement")
        processor_before = processor_computation_hash(catalog, engagement)
        source_before = source_computation_hash(catalog, "ih")

        source_changed = catalog.model_copy(deep=True)
        source_changed.pipelines.sources[0].defaults["NewDefault"] = "changed"
        changed_engagement = next(
            p for p in source_changed.processors.processors if p.id == "ih_engagement"
        )
        assert processor_computation_hash(source_changed, changed_engagement) != processor_before
        assert source_computation_hash(source_changed, "ih") != source_before

        presentation_changed = catalog.model_copy(deep=True)
        presentation_changed.dashboards.dashboards[0].title = "New dashboard title"
        presentation_changed.metrics.metrics["VS_Interactions"].description = "New description"
        presentation_changed.pipelines.sources[0].description = "New source description"
        presentation_changed.pipelines.sources[0].materialize_transforms = True
        presentation_changed.pipelines.sources[
            0
        ].reader.streaming = not presentation_changed.pipelines.sources[0].reader.streaming
        changed_processor = next(
            p for p in presentation_changed.processors.processors if p.id == "ih_engagement"
        )
        changed_processor.description = "New processor description"
        assert (
            processor_computation_hash(presentation_changed, changed_processor) == processor_before
        )
        assert source_computation_hash(presentation_changed, "ih") == source_before

    def test_computation_hash_excludes_sketch_build_mode(self) -> None:
        catalog = load(DEMO_WS)
        descriptive = next(
            processor
            for processor in catalog.processors.processors
            if isinstance(processor, model.NumericDistributionProcessor)
        )
        processor_before = processor_computation_hash(catalog, descriptive)
        source_before = source_computation_hash(catalog, descriptive.source)
        catalog_before = catalog_config_hash(catalog)

        changed = catalog.model_copy(deep=True)
        changed_descriptive = next(
            processor
            for processor in changed.processors.processors
            if processor.id == descriptive.id
        )
        assert isinstance(changed_descriptive, model.NumericDistributionProcessor)
        changed_descriptive.sketch_build_mode = "legacy"

        assert processor_computation_hash(changed, changed_descriptive) == processor_before
        assert source_computation_hash(changed, changed_descriptive.source) == source_before
        assert catalog_config_hash(changed) != catalog_before

    def test_bounded_ml_order_revision_is_scoped_to_score_processors(self) -> None:
        catalog = load(DEMO_WS)
        score = next(
            processor
            for processor in catalog.processors.processors
            if isinstance(processor, model.ScoreDistributionProcessor)
        )
        numeric = next(
            processor
            for processor in catalog.processors.processors
            if isinstance(processor, model.NumericDistributionProcessor)
        )

        score_payload = processor_computation_config(catalog, score)["processor"]
        numeric_payload = processor_computation_config(catalog, numeric)["processor"]

        assert score_payload["__valuestream_algorithm_revision"] == {
            "bounded_ml_source_order": 1,
            "native_ml_reduction": 1,
        }
        assert "__valuestream_algorithm_revision" not in numeric_payload

    def test_two_yamls_same_meaning_same_hash(self, tmp_path: Path) -> None:
        """A YAML file rewritten with reordered keys hashes identically."""
        # Read pipelines.yaml, parse, re-emit with shuffled key order.
        original = (DEMO_WS / "catalog" / "pipelines.yaml").read_text()
        parsed = yaml.safe_load(original)

        def shuffle_keys(node: object) -> object:
            if isinstance(node, dict):
                return {k: shuffle_keys(v) for k, v in reversed(list(node.items()))}
            if isinstance(node, list):
                return [shuffle_keys(v) for v in node]
            return node

        shuffled = shuffle_keys(parsed)
        shuffled_yaml = yaml.safe_dump(shuffled, sort_keys=False)
        # Sanity: the shuffled YAML text isn't byte-identical.
        assert shuffled_yaml != original

        # Build a complete shuffled workspace and reload.
        ws = tmp_path / "ws"
        (ws / "catalog").mkdir(parents=True)
        (ws / "catalog" / "pipelines.yaml").write_text(shuffled_yaml)
        for name in ("processors.yaml", "metrics.yaml", "dashboards.yaml"):
            (ws / "catalog" / name).write_text((DEMO_WS / "catalog" / name).read_text())

        a = load(DEMO_WS)
        b = load(ws)
        assert catalog_config_hash(a) == catalog_config_hash(b)

    def test_changing_a_state_changes_hash(self, tmp_path: Path) -> None:
        """Editing the engagement processor's group-by columns changes the catalog hash."""
        ws = tmp_path / "ws"
        (ws / "catalog").mkdir(parents=True)
        for name in ("pipelines.yaml", "metrics.yaml", "dashboards.yaml"):
            (ws / "catalog" / name).write_text((DEMO_WS / "catalog" / name).read_text())

        # Modify processors.yaml: drop one group-by column from engagement.
        proc_text = (DEMO_WS / "catalog" / "processors.yaml").read_text()
        proc = yaml.safe_load(proc_text)
        engagement = next(p for p in proc["processors"] if p["id"] == "ih_engagement")
        engagement["dimensions"] = engagement["dimensions"][:-1]
        (ws / "catalog" / "processors.yaml").write_text(yaml.safe_dump(proc))

        original = catalog_config_hash(load(DEMO_WS))
        modified = catalog_config_hash(load(ws))
        assert original != modified
