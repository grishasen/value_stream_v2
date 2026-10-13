"""Transport-independent catalog drafts, review, and transactional application.

Studio and MCP share validation, change review, bounded sample reading, and
the same writer. The MCP client supplies intelligence; this module calls no LLM.
"""

from __future__ import annotations

import copy
import datetime as dt
import difflib
import hashlib
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import polars as pl
import yaml

from valuestream.ai.copilot import validate_draft_field_contract
from valuestream.ai.samples import read_tabular_sample
from valuestream.ai.settings import write_chat_with_data_config
from valuestream.ai.studio import (
    draft_object_counts,
    generate_schema_preview,
    prompt_for_config_draft,
    section_name_diff,
    validate_draft_catalog,
)
from valuestream.config import model
from valuestream.config.canonical import (
    catalog_config_hash,
    config_hash,
    processor_computation_hash,
)
from valuestream.config.loader import _UniqueKeyLoader, load
from valuestream.config.writing import (
    CATALOG_FILENAMES,
    workspace_configuration_transaction,
    write_yaml,
)
from valuestream.processors.binary_outcome import ChunkContext
from valuestream.processors.registry import create_processor
from valuestream.transforms.pipeline import apply_transforms
from valuestream.utils.names import capitalize_fields

SECTIONS = tuple(name.removesuffix(".yaml") for name in CATALOG_FILENAMES)
MAX_DRAFT_BYTES = 2 * 1024 * 1024
MAX_DRAFTS = 32
DRAFT_TTL_SECONDS = 3600


def empty_catalog(workspace_name: str) -> dict[str, Any]:
    """A valid catalog skeleton that requires no directory or source data."""

    return {
        "pipelines": {"catalog_version": 2, "workspace": workspace_name, "sources": []},
        "processors": {"catalog_version": 2, "processors": []},
        "metrics": {"catalog_version": 2, "metrics": {}},
        "dashboards": {"catalog_version": 2, "dashboards": []},
    }


def workspace_configuration_hash(workspace: Path) -> str:
    """Fingerprint exact catalog and AI config bytes, including missing files."""

    digest = hashlib.sha256()
    for relative in [*(f"catalog/{name}" for name in CATALOG_FILENAMES), "ai.yaml"]:
        path = workspace / relative
        digest.update(relative.encode())
        digest.update(b"present" if path.exists() else b"missing")
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def read_catalog_draft(workspace: Path) -> dict[str, Any]:
    """Read the full current catalog, or an in-memory skeleton if none exists."""

    if not any((workspace / "catalog" / name).exists() for name in CATALOG_FILENAMES):
        return empty_catalog(workspace.name or "workspace")
    load(workspace)  # Reject partial/invalid catalogs instead of silently replacing them.
    return {
        section: yaml.load(
            (workspace / "catalog" / f"{section}.yaml").read_text(), Loader=_UniqueKeyLoader
        )
        for section in SECTIONS
    }


def catalog_changes(base: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """Return Studio's object diff plus source/dashboard/page changes and YAML diff."""

    changes: dict[str, Any] = section_name_diff(base, draft)
    for kind, key in (("sources", "pipelines"), ("dashboards", "dashboards")):
        before = {item["id"]: item for item in base[key][kind]}
        after = {item["id"]: item for item in draft[key][kind]}
        changes[kind] = _object_changes(before, after)
    changes["pages"] = _object_changes(_pages(base), _pages(draft))
    changes["yaml_diff"] = "\n".join(
        difflib.unified_diff(
            yaml.safe_dump(base, sort_keys=False).splitlines(),
            yaml.safe_dump(draft, sort_keys=False).splitlines(),
            fromfile="current catalog",
            tofile="proposed catalog",
            lineterm="",
        )
    )
    return changes


def _pages(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{dashboard['id']}/{page['id']}": page
        for dashboard in draft["dashboards"]["dashboards"]
        for page in dashboard.get("pages", [])
    }


def _object_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
        "unchanged": sorted(
            key for key in before.keys() & after.keys() if before[key] == after[key]
        ),
    }


def apply_catalog_draft(
    workspace: str | Path,
    draft: dict[str, Any],
    *,
    source_columns_by_id: Mapping[str, Iterable[str]] | None = None,
    expected_base_hash: str | None = None,
    validate_written: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Apply complete reviewed sections with rollback, without running ingestion.

    Studio performs its UI-specific review first. MCP supplies the expected base
    hash after issuing an exact-revision review token. The compare and writes
    share the same workspace lock used by Studio and Builder transactions.
    """

    root = Path(workspace)
    ok, issues = validate_draft_catalog(draft, source_columns_by_id=source_columns_by_id)
    if not ok:
        raise ValueError("Catalog draft is invalid:\n" + "\n".join(issues))
    proposed = model.Catalog.model_validate({key: draft[key] for key in SECTIONS})
    with workspace_configuration_transaction(root):
        if (
            expected_base_hash is not None
            and workspace_configuration_hash(root) != expected_base_hash
        ):
            raise ValueError(
                "Workspace configuration changed; create a new draft and review it again"
            )
        before = read_catalog_draft(root)
        prior = model.Catalog.model_validate(before)
        prior_hashes = {
            processor.id: processor_computation_hash(prior, processor)
            for processor in prior.processors.processors
        }
        for section in SECTIONS:
            write_yaml(root / "catalog" / f"{section}.yaml", draft[section])
        chat = draft.get("chat_with_data")
        if isinstance(chat, dict):
            write_chat_with_data_config(
                root,
                agent_prompt=str(chat.get("agent_prompt") or ""),
                dataset_descriptions=dict(chat.get("dataset_descriptions") or {}),
                metric_descriptions=dict(chat.get("metric_descriptions") or {}),
            )
        if validate_written is not None:
            validate_written(root, source_columns_by_id=source_columns_by_id)
        else:
            written = load(root)
            valid, errors = validate_draft_catalog(
                {
                    key: getattr(written, key).model_dump(mode="json", by_alias=True)
                    for key in SECTIONS
                },
                source_columns_by_id=source_columns_by_id,
            )
            if not valid:
                raise ValueError("Written catalog is invalid:\n" + "\n".join(errors))
        changed = [
            processor.source
            for processor in proposed.processors.processors
            if prior_hashes.get(processor.id) != processor_computation_hash(proposed, processor)
        ]
    return {
        "applied": True,
        "config_hash": catalog_config_hash(proposed),
        "requires_data_run": bool(changed),
        "source_ids": sorted(set(changed)),
        "ingestion_started": False,
    }


@dataclass
class CatalogDraft:
    payload: dict[str, Any]
    base: dict[str, Any]
    base_hash: str
    sample_path: str
    sample_signature: tuple[int, int, int]
    source_id: str
    columns: list[str]
    reader: dict[str, Any]
    limit: int
    created_at: float
    sequence: int = 0
    review_token: str | None = None
    applied: dict[str, Any] | None = None

    @property
    def revision(self) -> str:
        return config_hash({"catalog": self.payload, "sequence": self.sequence})


class CatalogAuthoring:
    """Bounded, process-local MCP drafts; retains configuration, never sample rows."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.drafts: dict[str, CatalogDraft] = {}
        self.lock = RLock()

    def sample_path(self, relative: str) -> Path:
        path = (self.workspace / relative).resolve()
        data = (self.workspace / "data").resolve()
        if not data.is_relative_to(self.workspace) or not path.is_relative_to(data):
            raise ValueError("sample_path must reference a file inside workspace data/")
        if not path.is_file():
            raise ValueError("Sample file does not exist inside workspace data/")
        if path.suffix.casefold() not in {".csv", ".parquet"}:
            raise ValueError("Use a CSV or Parquet sample")
        return path

    def profile(
        self,
        sample_path: str,
        *,
        limit: int = 1000,
        approved_fields: list[str] | None = None,
        example_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        path = self.sample_path(sample_path)
        try:
            frame = read_tabular_sample(path, limit=limit)
        except pl.exceptions.PolarsError as exc:
            raise ValueError(
                f"Sample parsing failed ({type(exc).__name__}); check the CSV/Parquet schema"
            ) from exc
        if frame.is_empty():
            raise ValueError("The sample must contain at least one data row")
        fields = frame.columns if approved_fields is None else approved_fields
        if not fields or set(fields) - set(frame.columns):
            raise ValueError("approved_fields must name existing sample columns")
        examples = example_fields or []
        if set(examples) - set(fields):
            raise ValueError("example_fields must be a subset of approved_fields")
        return {
            "sample_path": path.relative_to(self.workspace).as_posix(),
            "sample_rows": frame.height,
            "limit": limit,
            "schema": generate_schema_preview(frame, fields, examples),
            "values_shared": examples,
            "runtime_reader": path.suffix[1:].lower(),
        }

    def create(
        self,
        sample_path: str,
        *,
        source_id: str = "sample",
        user_goals: str = "",
        limit: int = 1000,
        approved_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self._prune()
            if len(self.drafts) >= MAX_DRAFTS:
                raise ValueError("Too many active drafts; discard a draft before creating another")
            if not source_id or not all(char.isalnum() or char in "_-" for char in source_id):
                raise ValueError(
                    "source_id must contain only letters, digits, underscores or hyphens"
                )
            path = self.sample_path(sample_path)
            signature = _sample_signature(path)
            profile = self.profile(sample_path, limit=limit, approved_fields=approved_fields)
            columns = [item["column"] for item in profile["schema"]]
            base_hash = workspace_configuration_hash(self.workspace)
            base = read_catalog_draft(self.workspace)
            payload = _sample_baseline(base, profile, source_id)
            if len(yaml.safe_dump(payload).encode()) > MAX_DRAFT_BYTES:
                raise ValueError("Draft exceeds the 2 MiB configuration budget")
            draft_id = secrets.token_urlsafe(24)
            draft = CatalogDraft(
                payload=payload,
                base=base,
                base_hash=base_hash,
                sample_path=profile["sample_path"],
                sample_signature=signature,
                source_id=source_id,
                columns=columns,
                reader=next(
                    item["reader"]
                    for item in payload["pipelines"]["sources"]
                    if item["id"] == source_id
                ),
                limit=limit,
                created_at=time.monotonic(),
            )
            self._assert_current(draft)
            self.drafts[draft_id] = draft
            return {
                **self._response(draft_id, draft),
                "profile": profile,
                "generation_prompt": prompt_for_config_draft(
                    file_name=path.name,
                    approved_schema=profile["schema"],
                    approved_fields=columns,
                    hidden_fields=[],
                    baseline_draft=payload,
                    user_goals=user_goals,
                    allow_source_changes=True,
                ),
                "instructions": "Use the client model to refine the baseline. Submit complete sections with catalog_draft_update; validate, preview, then apply the review token.",
            }

    def update(
        self, draft_id: str, sections: dict[str, Any], expected_revision: str
    ) -> dict[str, Any]:
        with self.lock:
            draft = self._get(draft_id)
            self._assert_current(draft)
            if draft.applied or expected_revision != draft.revision:
                raise ValueError("Draft revision changed or was applied; fetch or create a draft")
            if not sections or set(sections) - {*SECTIONS, "chat_with_data"}:
                raise ValueError("Supply complete catalog sections or chat_with_data")
            payload = {**copy.deepcopy(draft.payload), **copy.deepcopy(sections)}
            if len(yaml.safe_dump(payload).encode()) > MAX_DRAFT_BYTES:
                raise ValueError("Draft exceeds the 2 MiB configuration budget")
            # Structural shape is required for a meaningful diff; semantic errors
            # remain editable and are returned by validate rather than auto-applied.
            model.Catalog.model_validate({key: payload[key] for key in SECTIONS})
            _validate_chat(payload.get("chat_with_data"))
            draft.payload = payload
            draft.sequence += 1
            draft.review_token = None
            return self._response(draft_id, draft)

    def validate(self, draft_id: str) -> dict[str, Any]:
        with self.lock:
            draft = self._get(draft_id)
            self._assert_current(draft)
            ok, issues = self._validate(draft)
            return {"draft_id": draft_id, "revision": draft.revision, "ok": ok, "issues": issues}

    def preview(self, draft_id: str) -> dict[str, Any]:
        with self.lock:
            draft = self._get(draft_id)
            self._assert_current(draft)
            ok, issues = self._validate(draft)
            draft.review_token = secrets.token_urlsafe(32) if ok else None
            return {
                **self._response(draft_id, draft),
                "ok": ok,
                "issues": issues,
                "changes": catalog_changes(draft.base, draft.payload),
                "review_token": draft.review_token,
            }

    def apply(
        self, draft_id: str, review_token: str, *, confirm_removals: bool = False
    ) -> dict[str, Any]:
        with self.lock:
            draft = self._get(draft_id)
            if not draft.review_token or not secrets.compare_digest(
                draft.review_token, review_token
            ):
                raise ValueError("Preview and review this draft revision before applying it")
            if draft.applied is not None:
                return copy.deepcopy(draft.applied)
            self._assert_current(draft)
            ok, issues = self._validate(draft)
            if not ok:
                raise ValueError("Catalog draft is invalid:\n" + "\n".join(issues))
            self._assert_current(draft)
            changes = catalog_changes(draft.base, draft.payload)
            if (
                any(change["removed"] for change in changes.values() if isinstance(change, dict))
                and not confirm_removals
            ):
                raise ValueError(
                    "Preview lists removals; confirm_removals is required for this revision"
                )
            for relative in [
                *(f"catalog/{name}" for name in CATALOG_FILENAMES),
                "ai.yaml",
                ".catalog-authoring.lock",
            ]:
                if not (self.workspace / relative).resolve().is_relative_to(self.workspace):
                    raise ValueError("Configuration targets must remain inside the workspace")
            receipt = apply_catalog_draft(
                self.workspace,
                draft.payload,
                source_columns_by_id={draft.source_id: draft.columns},
                expected_base_hash=draft.base_hash,
            )
            draft.applied = {**receipt, "draft_id": draft_id, "revision": draft.revision}
            return copy.deepcopy(draft.applied)

    def discard(self, draft_id: str) -> dict[str, Any]:
        with self.lock:
            self._get(draft_id)
            del self.drafts[draft_id]
            return {"draft_id": draft_id, "discarded": True}

    def _validate(self, draft: CatalogDraft) -> tuple[bool, list[str]]:
        ok, issues = validate_draft_catalog(
            draft.payload, source_columns_by_id={draft.source_id: draft.columns}
        )
        if not ok:
            return ok, issues
        catalog = model.Catalog.model_validate({key: draft.payload[key] for key in SECTIONS})
        source = next(
            (item for item in catalog.pipelines.sources if item.id == draft.source_id), None
        )
        if source is None:
            return False, ["The sampled source must remain in the draft"]
        path = self.sample_path(draft.sample_path)
        original_reader = model.Source.model_validate(
            {"id": draft.source_id, "reader": draft.reader}
        ).reader
        if source.reader != original_reader:
            return False, ["Keep the sampled source reader configuration unchanged"]
        try:
            frame = read_tabular_sample(path, limit=draft.limit)
            effective_fields = list(draft.columns)
            if any(
                isinstance(transform, model.RenameCapitalize) for transform in source.transforms
            ):
                mapping = dict(zip(frame.columns, capitalize_fields(frame.columns), strict=True))
                effective_fields = [mapping[column] for column in draft.columns]
            contract_ok, contract_issues = validate_draft_field_contract(
                draft.payload,
                effective_fields,
                source_id=draft.source_id,
                source_fields=effective_fields,
                baseline_draft=draft.base,
            )
            issues.extend(contract_issues)
            if not contract_ok:
                return False, issues
            prepared = apply_transforms(frame.lazy(), source).collect()
            # Execute each processor over the bounded sample, then discard its
            # ephemeral aggregates; this catches dtype and required-column errors.
            context = ChunkContext("catalog-preview", "sample", dt.datetime.now(dt.UTC))
            for processor in catalog.processors.processors:
                if processor.source == draft.source_id:
                    create_processor(processor).chunk_aggregate(prepared.lazy(), context)
        except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
            return False, [
                f"Sample execution failed ({type(exc).__name__}); check source transforms, field types, and processor bindings"
            ]
        return True, issues

    def _assert_current(self, draft: CatalogDraft) -> None:
        if workspace_configuration_hash(self.workspace) != draft.base_hash:
            raise ValueError(
                "Workspace configuration changed; create a new draft and review it again"
            )
        if _sample_signature(self.sample_path(draft.sample_path)) != draft.sample_signature:
            raise ValueError("Sample changed; create a new draft from the current sample")

    def _response(self, draft_id: str, draft: CatalogDraft) -> dict[str, Any]:
        return {
            "draft_id": draft_id,
            "revision": draft.revision,
            "base_hash": draft.base_hash,
            "catalog": copy.deepcopy(draft.payload),
            "counts": draft_object_counts(draft.payload),
            "expires_in_seconds": max(
                0, int(DRAFT_TTL_SECONDS - (time.monotonic() - draft.created_at))
            ),
        }

    def _prune(self) -> None:
        for key in list(self.drafts):
            if time.monotonic() - self.drafts[key].created_at > DRAFT_TTL_SECONDS:
                del self.drafts[key]

    def _get(self, draft_id: str) -> CatalogDraft:
        self._prune()
        if draft_id not in self.drafts:
            raise ValueError("Unknown or expired draft; create a new catalog draft")
        return self.drafts[draft_id]


def _sample_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _validate_chat(chat: Any) -> None:
    if chat is None:
        return
    if not isinstance(chat, dict) or set(chat) - {
        "agent_prompt",
        "dataset_descriptions",
        "metric_descriptions",
    }:
        raise ValueError("chat_with_data accepts only agent_prompt and dataset/metric descriptions")
    if "agent_prompt" in chat and not isinstance(chat["agent_prompt"], str):
        raise ValueError("chat_with_data.agent_prompt must be a string")
    for key in ("dataset_descriptions", "metric_descriptions"):
        value = chat.get(key, {})
        if not isinstance(value, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
        ):
            raise ValueError(f"chat_with_data.{key} must map strings to strings")


def _sample_baseline(
    base: dict[str, Any], profile: dict[str, Any], source_id: str
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    if any(item["id"] == source_id for item in payload["pipelines"]["sources"]):
        raise ValueError("source_id already exists; choose a new sampled-source id")
    path = Path(profile["sample_path"])
    fields = {row["column"] for row in profile["schema"]}
    scope = "SampleScope"
    while scope in fields:
        scope += "_"
    processor_id = f"{source_id}_records"
    metric_name = f"{source_id}_Rows"
    if (
        any(item["id"] == processor_id for item in payload["processors"]["processors"])
        or metric_name in payload["metrics"]["metrics"]
    ):
        raise ValueError("Generated processor/metric id already exists; choose another source_id")
    payload["pipelines"]["sources"].append(
        {
            "id": source_id,
            "reader": {
                "kind": profile["runtime_reader"],
                "root": path.parent.as_posix(),
                "file_pattern": path.name,
                "try_parse_dates": False,
                "infer_schema_length": 500,
            },
            "transforms": [
                {"kind": "derive_column", "output": scope, "expression": {"lit": "All records"}}
            ],
        }
    )
    payload["processors"]["processors"].append(
        {
            "id": processor_id,
            "source": source_id,
            "kind": "funnel",
            "group_by": [scope],
            "time": {"property": "", "grain": "summary"},
            "stages": [{"name": "Records", "when": {"lit": True}}],
            "states": {"Count": {"type": "count"}},
        }
    )
    payload["metrics"]["metrics"][metric_name] = {
        "processor": processor_id,
        "kind": "formula",
        "expression": {"col": "Count"},
        "display": {"label": "Rows", "value_format": "integer"},
    }
    dashboard_id = f"{source_id}_overview"
    if any(item["id"] == dashboard_id for item in payload["dashboards"]["dashboards"]):
        raise ValueError("Generated dashboard id already exists; choose another source_id")
    payload["dashboards"]["dashboards"].append(
        {
            "id": dashboard_id,
            "title": "Sample overview",
            "pages": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "tiles": [
                        {
                            "id": "rows",
                            "title": "Rows",
                            "metric": metric_name,
                            "chart": "kpi_card",
                        }
                    ],
                }
            ],
        }
    )
    return payload
