"""Run the reproducible Value Stream ingestion benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import duckdb
import polars as pl
import yaml

from tests.benchmarks.contracts import CONTRACT_VERSION, SUITE_PROCESSORS, summarize_samples
from valuestream.config.loader import load
from valuestream.readers import discover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source", default="ih")
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(SUITE_PROCESSORS),
        help="Repeat to run both suites; defaults to both.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/baseline.json"))
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    suites = args.suite or list(SUITE_PROCESSORS)
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be >= 1 and --warmups must be >= 0")
    if args.parallel < 1:
        parser.error("--parallel must be >= 1")

    fixture = _input_manifest(workspace, args.source)
    payload: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "fixture": fixture,
        "environment": _environment(),
        "suites": {},
    }
    for suite in suites:
        suite_result = _run_suite(
            workspace=workspace,
            source_id=args.source,
            suite=suite,
            warmups=args.warmups,
            repeats=args.repeats,
            parallel=args.parallel,
            scratch_root=args.scratch_root,
        )
        payload["suites"][suite] = suite_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote benchmark contract: {args.output}")
    for suite in suites:
        summary = payload["suites"][suite]["summary"]
        print(
            f"{suite}: {summary['rows_per_second_mean']:,.0f} rows/s, "
            f"{summary['wall_seconds_mean']:.3f}s wall, "
            f"CV={summary['wall_seconds_cv']:.2%}, "
            f"peak RSS={summary['peak_rss_bytes_max'] / (1024**3):.2f} GiB"
        )
        if not summary["outputs_equivalent"]:
            print(
                f"WARNING: {suite} output correctness contract did not match; "
                "inspect summary.approximate_comparison_issues and exact output digests",
                file=sys.stderr,
            )
        elif summary["representations_stable"] is False:
            print(
                f"INFO: {suite} serialized sketch bytes varied while decoded outputs "
                "remained equivalent",
                file=sys.stderr,
            )


def _run_suite(
    *,
    workspace: Path,
    source_id: str,
    suite: str,
    warmups: int,
    repeats: int,
    parallel: int,
    scratch_root: Path | None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    catalog_contract: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    total = warmups + repeats
    for index in range(total):
        sample = _run_sample(
            workspace=workspace,
            source_id=source_id,
            suite=suite,
            parallel=parallel,
            scratch_root=scratch_root,
        )
        current_catalog = {
            "catalog_hash": sample.pop("catalog_hash"),
            "source_computation_hash": sample.pop("source_computation_hash"),
        }
        current_execution = {
            "parallel": parallel,
            "streaming": sample.pop("streaming"),
            "materialize_transforms": sample.pop("materialize_transforms"),
            "warmups": warmups,
            "repeats": repeats,
            "cache_posture": "warm OS cache; output/meta store recreated for every sample",
        }
        if catalog_contract is not None and current_catalog != catalog_contract:
            raise RuntimeError(f"suite {suite!r} catalog hash changed between samples")
        if execution is not None and current_execution != execution:
            raise RuntimeError(f"suite {suite!r} execution flags changed between samples")
        catalog_contract = current_catalog
        execution = current_execution
        if index >= warmups:
            samples.append(sample)
    return {
        "processors": list(SUITE_PROCESSORS[suite]),
        "catalog": catalog_contract,
        "execution": execution,
        "samples": samples,
        "summary": summarize_samples(samples),
    }


def _run_sample(
    *,
    workspace: Path,
    source_id: str,
    suite: str,
    parallel: int,
    scratch_root: Path | None,
) -> dict[str, Any]:
    root = None if scratch_root is None else str(scratch_root.resolve())
    with tempfile.TemporaryDirectory(prefix=f"valuestream-bench-{suite}-", dir=root) as temp:
        scratch = Path(temp)
        _prepare_workspace(workspace, scratch, source_id, SUITE_PROCESSORS[suite])
        result_path = scratch / "sample.json"
        command = [
            sys.executable,
            "-m",
            "tests.benchmarks.worker",
            "--workspace",
            str(scratch),
            "--source",
            source_id,
            "--parallel",
            str(parallel),
            "--result",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark worker failed ({completed.returncode})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return cast("dict[str, Any]", json.loads(result_path.read_text(encoding="utf-8")))


def _prepare_workspace(
    source_workspace: Path,
    scratch: Path,
    source_id: str,
    processors: tuple[str, ...],
) -> None:
    shutil.copytree(source_workspace / "catalog", scratch / "catalog")
    catalog = load(source_workspace)
    source = next(item for item in catalog.pipelines.sources if item.id == source_id)
    extra = dict(source.reader.model_extra or {})
    raw_root = Path(str(extra.get("root") or extra.get("base_dir") or "."))
    resolved_root = raw_root if raw_root.is_absolute() else source_workspace / raw_root

    pipelines_path = scratch / "catalog" / "pipelines.yaml"
    pipelines = yaml.safe_load(pipelines_path.read_text(encoding="utf-8"))
    raw_source = next(item for item in pipelines["sources"] if item["id"] == source_id)
    raw_source["reader"]["root"] = str(resolved_root.resolve())
    raw_source["reader"].pop("base_dir", None)
    pipelines_path.write_text(yaml.safe_dump(pipelines, sort_keys=False), encoding="utf-8")

    processors_path = scratch / "catalog" / "processors.yaml"
    processor_catalog = yaml.safe_load(processors_path.read_text(encoding="utf-8"))
    processor_catalog["processors"] = [
        item
        for item in processor_catalog.get("processors", [])
        if item.get("source") != source_id or item.get("id") in processors
    ]
    present = {
        str(item.get("id"))
        for item in processor_catalog["processors"]
        if item.get("source") == source_id
    }
    missing = sorted(set(processors).difference(present))
    if missing:
        raise ValueError(f"benchmark suite requires missing processor(s): {', '.join(missing)}")
    processors_path.write_text(yaml.safe_dump(processor_catalog, sort_keys=False), encoding="utf-8")
    (scratch / "catalog" / "metrics.yaml").write_text("metrics: {}\n", encoding="utf-8")
    (scratch / "catalog" / "dashboards.yaml").write_text("dashboards: []\n", encoding="utf-8")


def _input_manifest(workspace: Path, source_id: str) -> dict[str, Any]:
    catalog = load(workspace)
    source = next(item for item in catalog.pipelines.sources if item.id == source_id)
    chunks = discover(workspace, source)
    files: dict[Path, set[str]] = {}
    for chunk in chunks:
        for unit in chunk.files:
            members = (
                [unit]
                if unit.is_file()
                else sorted(path for path in unit.rglob("*") if path.is_file())
            )
            for path in members:
                files.setdefault(path.resolve(), set()).add(chunk.chunk_id)

    records: list[dict[str, Any]] = []
    fixture_hash = hashlib.sha256()
    for path, chunk_ids in sorted(files.items(), key=lambda item: str(item[0])):
        digest = _sha256_file(path)
        stat = path.stat()
        logical = {
            "chunks": sorted(chunk_ids),
            "name": path.name,
            "bytes": stat.st_size,
            "sha256": digest,
        }
        fixture_hash.update(json.dumps(logical, sort_keys=True).encode("utf-8"))
        records.append({"path": str(path), **logical})
    return {
        "fixture_id": fixture_hash.hexdigest(),
        "source": source_id,
        "chunks": len(chunks),
        "files": records,
        "bytes": sum(record["bytes"] for record in records),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_dirty": bool(_git_status()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "polars": pl.__version__,
        "polars_threads": pl.thread_pool_size(),
        "duckdb": duckdb.__version__,
        "datasketches": importlib.metadata.version("datasketches"),
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _git_status() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
