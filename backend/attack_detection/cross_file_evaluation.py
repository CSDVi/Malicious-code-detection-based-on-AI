"""Evaluation of cross-file chains on the existing isolated challenge corpus."""

from __future__ import annotations

import fnmatch
import json
import zipfile
from pathlib import Path
from typing import Any, Callable

from .cross_file_analysis import analyze_cross_file_project
from .data_pipeline import make_sample
from .engines.gat_engine import GATEngine
from .features.graph_builder import build_project_graph
from .languages import decode_source_payload, detect_source_language


MAX_CHALLENGE_FILES = 200
MAX_CHALLENGE_FILE_BYTES = 2 * 1024 * 1024


def evaluate_cross_file_challenge(
    manifest: dict[str, Any],
    *,
    workspace_root: str | Path,
    run_gat: bool = True,
    gat_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure complete-chain recall and GATv2 Top-1 component hit rate."""

    _validate_manifest(manifest)
    root = Path(workspace_root).resolve()
    case_results = []
    chain_hits = 0
    chain_total = 0
    gat_top1_hits = 0
    gat_top1_total = 0
    static_top1_hits = 0
    static_top1_total = 0
    gat_completed = 0
    for raw_case in manifest.get("cases") or []:
        case = dict(raw_case)
        records, source = _load_case_records(case, root)
        analysis = analyze_cross_file_project(records)
        expected_chain = case.get("expected_chain") or {}
        expects_complete = bool(expected_chain.get("complete", True))
        chain_hit = _complete_chain_hit(
            analysis.get("complete_chains") or [],
            expected_chain,
        )
        if expects_complete:
            chain_total += 1
            chain_hits += int(chain_hit)

        expected_top = [
            str(value) for value in case.get("expected_top_components") or []
            if str(value).strip()
        ]
        static_component = analysis.get("most_suspicious_component") or {}
        static_path = str(static_component.get("path") or "")
        static_hit = bool(expected_top and _matches_any(static_path, expected_top))
        if expected_top:
            static_top1_total += 1
            static_top1_hits += int(static_hit)

        gat_result: dict[str, Any] | None = None
        gat_path = ""
        gat_hit = False
        if run_gat:
            graph_samples = [
                make_sample(
                    str(record["content"]),
                    label="benign",
                    category="runtime_unlabeled",
                    language=str(record["language"]),
                    source="existing_real_world_challenge",
                    package_name=str(case.get("repository_id") or case.get("id")),
                    version=str(case.get("version") or "challenge"),
                    family=str(case.get("family_id") or "challenge"),
                    split="independent_challenge",
                    file_path=str(record["filename"]),
                    label_basis="challenge_ground_truth_is_excluded_from_gat_features",
                )
                for record in records
            ]
            if graph_samples:
                graph = build_project_graph(graph_samples)
                gat_result = (
                    gat_runner(graph)
                    if gat_runner is not None
                    else GATEngine().scan_project(graph)
                )
                if gat_result.get("status") == "completed":
                    gat_completed += 1
                metadata = gat_result.get("metadata") or {}
                gat_component = metadata.get("most_suspicious_component") or {}
                gat_path = str(gat_component.get("path") or "")
                gat_hit = bool(expected_top and _matches_any(gat_path, expected_top))
        if run_gat and expected_top:
            gat_top1_total += 1
            gat_top1_hits += int(gat_hit)

        case_results.append({
            "id": case.get("id"),
            "repository_id": case.get("repository_id"),
            "family_id": case.get("family_id"),
            "data_source_id": case.get("data_source_id"),
            "source": source,
            "file_count": len(records),
            "complete_chain_expected": expects_complete,
            "complete_chain_detected": bool(analysis.get("complete_chain_count")),
            "complete_chain_hit": chain_hit if expects_complete else None,
            "predicted_chain_count": analysis.get("complete_chain_count", 0),
            "expected_top_components": expected_top,
            "static_top1_component": static_path or None,
            "static_top1_hit": static_hit if expected_top else None,
            "gatv2_top1_component": gat_path or None,
            "gatv2_top1_hit": gat_hit if expected_top else None,
            "gatv2": gat_result,
            "analysis": analysis,
        })

    return {
        "schema_version": 1,
        "dataset_name": manifest.get("dataset_name"),
        "dataset_version": manifest.get("dataset_version"),
        "independence": manifest.get("independence"),
        "case_count": len(case_results),
        "metrics": {
            "complete_chain_recall": _ratio(chain_hits, chain_total),
            "complete_chain_hits": chain_hits,
            "complete_chain_cases": chain_total,
            "gatv2_top1_component_hit_rate": _ratio(gat_top1_hits, gat_top1_total),
            "gatv2_top1_hits": gat_top1_hits,
            "gatv2_top1_cases": gat_top1_total,
            "gatv2_completed_cases": gat_completed,
            "static_top1_component_hit_rate": _ratio(static_top1_hits, static_top1_total),
            "static_top1_hits": static_top1_hits,
            "static_top1_cases": static_top1_total,
        },
        "measurement_status": _measurement_status(
            chain_total=chain_total,
            gat_top1_total=gat_top1_total,
            gat_completed=gat_completed,
            run_gat=run_gat,
        ),
        "cases": case_results,
    }


def load_challenge_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if int(manifest.get("schema_version") or 0) != 1:
        raise ValueError("cross-file challenge manifest schema_version must be 1")
    independence = manifest.get("independence") or {}
    required = (
        "repository_isolation_verified",
        "family_isolation_verified",
        "data_source_isolation_verified",
    )
    missing = [name for name in required if independence.get(name) is not True]
    if missing:
        raise ValueError(
            "challenge independence is not verified: " + ", ".join(missing)
        )
    cases = manifest.get("cases") or []
    if not cases:
        raise ValueError("cross-file challenge manifest has no cases")
    identifiers = [str(case.get("id") or "") for case in cases]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("challenge case ids must be present and unique")
    for case in cases:
        for field in ("repository_id", "family_id", "data_source_id"):
            if not str(case.get(field) or "").strip():
                raise ValueError(f"challenge case {case.get('id')} is missing {field}")
        if not case.get("root") and not case.get("archive"):
            raise ValueError(
                f"challenge case {case.get('id')} needs root or archive"
            )


def _load_case_records(
    case: dict[str, Any],
    workspace_root: Path,
) -> tuple[list[dict[str, str]], str]:
    if case.get("archive"):
        archive = _inside_root(workspace_root, case["archive"])
        if not archive.is_file():
            raise FileNotFoundError(archive)
        records = []
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if member.is_dir() or member.file_size > MAX_CHALLENGE_FILE_BYTES:
                    continue
                if not _case_file_selected(member.filename, case):
                    continue
                language = detect_source_language(member.filename)
                if language in {"unknown", "binary"}:
                    continue
                try:
                    payload = handle.read(member)
                except RuntimeError as exc:
                    if "encrypted" not in str(exc).lower():
                        raise
                    payload = handle.read(member, pwd=b"infected")
                records.append({
                    "filename": member.filename.replace("\\", "/"),
                    "content": decode_source_payload(payload),
                    "language": language,
                })
                if len(records) >= MAX_CHALLENGE_FILES:
                    break
        return records, str(archive)

    case_root = _inside_root(workspace_root, case["root"])
    if not case_root.is_dir():
        raise FileNotFoundError(case_root)
    records = []
    for path in sorted(case_root.rglob("*")):
        if not path.is_file() or path.stat().st_size > MAX_CHALLENGE_FILE_BYTES:
            continue
        relative = str(path.relative_to(case_root)).replace("\\", "/")
        if not _case_file_selected(relative, case):
            continue
        language = detect_source_language(relative)
        if language in {"unknown", "binary"}:
            continue
        records.append({
            "filename": relative,
            "content": decode_source_payload(path.read_bytes()),
            "language": language,
        })
        if len(records) >= MAX_CHALLENGE_FILES:
            break
    return records, str(case_root)


def _inside_root(root: Path, value: object) -> Path:
    candidate = (root / str(value)).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"challenge path escaped workspace root: {value}")
    return candidate


def _case_file_selected(path: str, case: dict[str, Any]) -> bool:
    normalized = path.replace("\\", "/")
    includes = [str(value) for value in case.get("include_files") or []]
    excludes = [str(value) for value in case.get("exclude_files") or []]
    if includes and not _matches_any(normalized, includes):
        return False
    return not (excludes and _matches_any(normalized, excludes))


def _complete_chain_hit(
    chains: list[dict[str, Any]],
    expected: dict[str, Any],
) -> bool:
    if not expected.get("complete", True):
        return not chains
    source_patterns = [str(value) for value in expected.get("source_files") or []]
    transform_patterns = [str(value) for value in expected.get("transform_files") or []]
    sink_patterns = [str(value) for value in expected.get("sink_files") or []]
    minimum_files = int(expected.get("minimum_files") or 2)
    for chain in chains:
        steps = chain.get("trace_steps") or []
        stage_files = {
            stage: [
                str(step.get("file") or "")
                for step in steps
                if step.get("stage") == stage
            ]
            for stage in ("source", "transform", "sink")
        }
        if len(set(chain.get("files") or [])) < minimum_files:
            continue
        if source_patterns and not _any_path_match(stage_files["source"], source_patterns):
            continue
        if transform_patterns and not _any_path_match(stage_files["transform"], transform_patterns):
            continue
        if sink_patterns and not _any_path_match(stage_files["sink"], sink_patterns):
            continue
        return True
    return False


def _any_path_match(paths: list[str], patterns: list[str]) -> bool:
    return any(_matches_any(path, patterns) for path in paths)


def _matches_any(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/").casefold()
    return any(
        fnmatch.fnmatch(normalized, pattern.replace("\\", "/").casefold())
        for pattern in patterns
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _measurement_status(
    *,
    chain_total: int,
    gat_top1_total: int,
    gat_completed: int,
    run_gat: bool,
) -> str:
    if not chain_total:
        return "insufficient_ground_truth"
    if not run_gat:
        return "measured_without_gatv2"
    if gat_top1_total and gat_completed == gat_top1_total:
        return "measured"
    return "partial_gatv2_measurement"
