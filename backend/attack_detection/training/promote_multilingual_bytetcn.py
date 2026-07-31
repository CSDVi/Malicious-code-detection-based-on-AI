"""Merge a validated multilingual ByteCNN-TCN candidate into language routes.

The active artifact is preserved for every route it already serves.  A candidate
is added only for task/language pairs that passed its recorded deployment gate.
This prevents a multilingual retrain from silently regressing an older language.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = ("malicious_intent", "vulnerability_risk")


def promote(candidate_dir: Path, model_dir: Path) -> dict[str, Any]:
    candidate_path = candidate_dir / "bytetcn_manifest.json"
    active_path = model_dir / "bytetcn_manifest.json"
    candidate = _read_json(candidate_path)
    active = _read_json(active_path)
    if candidate.get("runtime_ready") is not True:
        raise SystemExit("Candidate is not runtime-ready")

    candidate_support = candidate.get("task_language_support") or {}
    if not any(candidate_support.get(task) for task in TASKS):
        raise SystemExit("Candidate has no validated task/language routes")
    _verify_candidate_routes(candidate)

    archive_dir = _archive_active(active, active_path, model_dir)
    copied_files = _copy_candidate_files(candidate, candidate_dir, model_dir)
    candidate_file_map = {
        source: target for source, target in copied_files
    }

    support: dict[str, list[str]] = {}
    task_models: dict[str, Any] = {}
    metrics_by_language = copy.deepcopy(active.get("test_metrics_by_language") or {})

    for task in TASKS:
        routes: dict[str, Any] = {}
        active_languages = list((active.get("task_language_support") or {}).get(task) or [])
        for language in active_languages:
            routes[language] = _route_from_manifest(active, task, language)
            metrics = _metrics_for(active, task, language)
            if metrics:
                metrics_by_language.setdefault(language, {})[task] = metrics

        # Existing routes win by default.  The candidate expands coverage and
        # never replaces a known-good language as a side effect of promotion.
        for language in candidate_support.get(task) or []:
            if language in routes:
                continue
            route = _route_from_manifest(candidate, task, language)
            route["file"] = candidate_file_map[str(route["file"])]
            routes[language] = route
            metrics = _metrics_for(candidate, task, language)
            metrics_by_language.setdefault(language, {})[task] = metrics

        support[task] = sorted(routes)
        task_models[task] = {"by_language": routes}

    all_languages = sorted({language for values in support.values() for language in values})
    if not all_languages:
        raise SystemExit("Promotion would leave the runtime with no supported languages")

    merged = copy.deepcopy(active)
    now = datetime.now(timezone.utc)
    datasets = _dataset_records(active) + _dataset_records(candidate)
    dataset_fingerprint = hashlib.sha256(
        json.dumps(datasets, sort_keys=True).encode("utf-8")
    ).hexdigest()
    merged.update({
        "schema_version": 2,
        "model_version": f"bytetcn-routed-{now.strftime('%Y%m%dT%H%M%SZ')}-{dataset_fingerprint[:12]}",
        "architecture": "Task-and-language-routed ByteCNN-TCN encoders with calibrated binary heads",
        "supported_languages": all_languages,
        "task_language_support": support,
        "language_support_tiers": {
            task: {"validated": languages, "provisional": []}
            for task, languages in support.items()
        },
        "task_models": task_models,
        "test_metrics_by_language": metrics_by_language,
        "metric_scopes": {
            task: "已验证语言（表中指标按表现最弱的语言保守汇总）：" + " / ".join(languages)
            for task, languages in support.items() if languages
        },
        "training_datasets": datasets,
        "dataset_sha256": dataset_fingerprint,
        "runtime_ready": True,
        "runtime_note": (
            "CPU inference routes every validated task/language pair to its own "
            "artifact, architecture, calibration temperature, and threshold."
        ),
        "created_at": now.isoformat(),
        "promotion_archive": str(archive_dir),
    })
    merged["test_metrics"] = {
        task: _conservative_summary(metrics_by_language, task, languages)
        for task, languages in support.items() if languages
    }
    merged["files"] = sorted(_referenced_files(merged))

    temporary = active_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(active_path)
    return merged


def _verify_candidate_routes(manifest: dict[str, Any]) -> None:
    gate = manifest.get("deployment_gate") or {}
    minimum_f1 = float(gate.get("minimum_test_f1", 0.5))
    maximum_fpr = float(gate.get("maximum_false_positive_rate", 0.2))
    maximum_fnr = float(gate.get("maximum_false_negative_rate", 0.5))
    for task, languages in (manifest.get("task_language_support") or {}).items():
        for language in languages:
            metrics = _metrics_for(manifest, task, language)
            if not metrics:
                raise SystemExit(f"Missing test metrics for candidate route {task}/{language}")
            passed = (
                float(metrics.get("f1", 0.0)) >= minimum_f1
                and float(metrics.get("false_positive_rate", 1.0)) <= maximum_fpr
                and float(metrics.get("false_negative_rate", 1.0)) <= maximum_fnr
            )
            if not passed:
                raise SystemExit(f"Candidate route failed deployment gate: {task}/{language}")


def _route_from_manifest(manifest: dict[str, Any], task: str, language: str) -> dict[str, Any]:
    task_settings = copy.deepcopy((manifest.get("task_models") or {}).get(task) or {})
    by_language = task_settings.pop("by_language", None)
    if isinstance(by_language, dict) and isinstance(by_language.get(language), dict):
        route = copy.deepcopy(by_language[language])
    else:
        route = task_settings
    route.setdefault("file", str((manifest.get("files") or ["bytetcn_multitask.pt"])[0]))
    route.setdefault("config", copy.deepcopy(manifest["config"]))
    route.setdefault("threshold", float(manifest["thresholds"][task]))
    route.setdefault("temperature", float(manifest["temperatures"][task]))
    route.setdefault("auxiliary_thresholds", copy.deepcopy(manifest.get("auxiliary_thresholds") or {}))
    route.setdefault("behavior_vocabulary", copy.deepcopy(manifest.get("behavior_vocabulary") or []))
    route.setdefault("cwe_vocabulary", copy.deepcopy(manifest.get("cwe_vocabulary") or []))
    route.setdefault("model_version", str(
        (manifest.get("task_model_versions") or {}).get(task)
        or manifest.get("model_version") or "unknown"
    ))
    route.setdefault("line_localization_validated", task == "malicious_intent")
    return route


def _metrics_for(manifest: dict[str, Any], task: str, language: str) -> dict[str, Any]:
    by_language = manifest.get("test_metrics_by_language") or {}
    metrics = (by_language.get(language) or {}).get(task)
    if isinstance(metrics, dict) and metrics:
        return copy.deepcopy(metrics)
    supported = list((manifest.get("task_language_support") or {}).get(task) or [])
    fallback = (manifest.get("test_metrics") or {}).get(task)
    if supported == [language] and isinstance(fallback, dict):
        return copy.deepcopy(fallback)
    return {}


def _conservative_summary(
    metrics_by_language: dict[str, Any], task: str, languages: list[str],
) -> dict[str, Any]:
    rows = [
        metrics_by_language[language][task]
        for language in languages
        if isinstance((metrics_by_language.get(language) or {}).get(task), dict)
    ]
    if len(rows) != len(languages):
        raise SystemExit(f"Cannot summarize {task}: one or more language metrics are missing")
    minimum_keys = ("accuracy", "precision", "recall", "f1")
    maximum_keys = ("false_positive_rate", "false_negative_rate", "brier_score")
    summary = {key: min(float(row.get(key, 0.0)) for row in rows) for key in minimum_keys}
    summary.update({key: max(float(row.get(key, 0.0)) for row in rows) for key in maximum_keys})
    summary["samples"] = sum(int(row.get("samples", 0)) for row in rows)
    summary["aggregation"] = "worst_validated_language"
    summary["supported_languages"] = list(languages)
    return summary


def _copy_candidate_files(
    manifest: dict[str, Any], candidate_dir: Path, model_dir: Path,
) -> list[tuple[str, str]]:
    copied = []
    version = str(manifest.get("model_version") or "candidate").replace("/", "_")
    for source_name in sorted(_referenced_files(manifest)):
        source = candidate_dir / source_name
        if not source.is_file():
            raise SystemExit(f"Candidate artifact is missing: {source_name}")
        source_path = Path(source_name)
        target_name = f"{source_path.stem}__{version}{source_path.suffix}"
        shutil.copy2(source, model_dir / target_name)
        copied.append((source_name, target_name))
    return copied


def _archive_active(manifest: dict[str, Any], manifest_path: Path, model_dir: Path) -> Path:
    version = str(manifest.get("model_version") or "unknown").replace("/", "_")
    archive = model_dir / "archive" / version
    if archive.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = archive.with_name(f"{archive.name}-{suffix}")
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, archive / manifest_path.name)
    for name in sorted(_referenced_files(manifest)):
        source = model_dir / name
        if source.is_file():
            shutil.copy2(source, archive / source.name)
    history = model_dir / "bytetcn_history.json"
    if history.is_file():
        shutil.copy2(history, archive / history.name)
    return archive


def _referenced_files(manifest: dict[str, Any]) -> set[str]:
    names = {str(value) for value in manifest.get("files") or []}
    for task_settings in (manifest.get("task_models") or {}).values():
        if not isinstance(task_settings, dict):
            continue
        if task_settings.get("file"):
            names.add(str(task_settings["file"]))
        for route in (task_settings.get("by_language") or {}).values():
            if isinstance(route, dict) and route.get("file"):
                names.add(str(route["file"]))
    return names


def _dataset_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    existing = manifest.get("training_datasets")
    if isinstance(existing, list):
        return copy.deepcopy(existing)
    return [{
        "model_version": manifest.get("model_version"),
        "dataset_sha256": manifest.get("dataset_sha256"),
    }]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Required manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote validated multilingual ByteCNN-TCN routes")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    result = promote(args.candidate_dir, args.model_dir)
    print(json.dumps({
        "model_version": result["model_version"],
        "task_language_support": result["task_language_support"],
        "test_metrics": result["test_metrics"],
        "promotion_archive": result["promotion_archive"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
