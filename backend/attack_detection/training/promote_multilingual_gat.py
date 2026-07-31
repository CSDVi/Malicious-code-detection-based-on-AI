"""Promote validated GATv2 language routes without replacing stronger routes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attack_detection.training.gat_trainer import DEPLOYMENT_GATE, LEGACY_LANGUAGES


def promote(
    candidate_dir: Path, model_dir: Path, legacy_evaluation_path: Path,
) -> dict[str, Any]:
    active_path = model_dir / "gatv2_manifest.json"
    candidate_path = candidate_dir / "gatv2_manifest.json"
    active = _read_json(active_path)
    candidate = _read_json(candidate_path)
    legacy_evaluation = _read_json(legacy_evaluation_path)
    if candidate.get("runtime_ready") is not True:
        raise SystemExit("GATv2 candidate is not runtime-ready")

    candidate_support = list(candidate.get("supported_languages") or [])
    for language in candidate_support:
        _require_gate((candidate.get("test_metrics_by_language") or {}).get(language), language)

    archive = _archive_active(active, active_path, model_dir)
    candidate_source_name = str(
        candidate.get("artifact") or (candidate.get("files") or ["gatv2_classifier.pt"])[0]
    )
    candidate_source = candidate_dir / candidate_source_name
    if not candidate_source.is_file():
        raise SystemExit(f"Candidate artifact is missing: {candidate_source_name}")
    candidate_version = str(candidate["model_version"]).replace("/", "_")
    source_path = Path(candidate_source_name)
    candidate_target_name = f"{source_path.stem}__{candidate_version}{source_path.suffix}"
    shutil.copy2(candidate_source, model_dir / candidate_target_name)

    active_file = str(active.get("artifact") or (active.get("files") or ["gatv2_classifier.pt"])[0])
    python_metrics = (
        (legacy_evaluation.get("test_metrics_by_language") or {}).get("python") or {}
    )
    python_coverage = (
        (legacy_evaluation.get("language_coverage") or {}).get("python") or {}
    )
    if python_coverage.get("eligible") is not True:
        raise SystemExit("Legacy GATv2 artifact lacks eligible Python coverage")
    _require_gate(python_metrics, "python")

    active_routes = active.get("language_models")
    if isinstance(active_routes, dict) and active_routes:
        routes = copy.deepcopy(active_routes)
        metrics_by_language = copy.deepcopy(active.get("test_metrics_by_language") or {})
        coverage = copy.deepcopy(active.get("language_coverage") or {})
    else:
        routes = {
            "python": {
                "file": active_file,
                "languages": list(active.get("languages") or LEGACY_LANGUAGES),
                "training": copy.deepcopy(active["training"]),
                "temperature": float(active["temperature"]),
                "threshold": float(active["threshold"]),
                "model_version": str(active["model_version"]),
            }
        }
        metrics_by_language = {"python": copy.deepcopy(python_metrics)}
        coverage = {"python": copy.deepcopy(python_coverage)}
    for language in candidate_support:
        if language in routes:
            continue
        routes[language] = {
            "file": candidate_target_name,
            "languages": copy.deepcopy(candidate["languages"]),
            "training": copy.deepcopy(candidate["training"]),
            "temperature": float(candidate["temperature"]),
            "threshold": float(candidate["language_thresholds"][language]),
            "model_version": str(candidate["model_version"]),
        }
        metrics_by_language[language] = copy.deepcopy(
            candidate["test_metrics_by_language"][language]
        )
        coverage[language] = copy.deepcopy(candidate["language_coverage"][language])

    supported_languages = sorted(routes)
    now = datetime.now(timezone.utc)
    datasets = copy.deepcopy(active.get("training_datasets") or [{
        "model_version": active.get("model_version"),
        "dataset_sha256": active.get("dataset_sha256"),
    }])
    datasets.append({
        "model_version": candidate.get("model_version"),
        "dataset_sha256": candidate.get("dataset_sha256"),
    })
    fingerprint = hashlib.sha256(
        json.dumps(datasets, sort_keys=True).encode("utf-8")
    ).hexdigest()
    merged = copy.deepcopy(candidate)
    merged.update({
        "schema_version": 2,
        "model_version": f"gatv2-routed-{now.strftime('%Y%m%dT%H%M%SZ')}-{fingerprint[:12]}",
        "architecture": "Language-routed torch_geometric.nn.GATv2Conv classifiers",
        "dataset_sha256": fingerprint,
        "training_datasets": datasets,
        "supported_languages": supported_languages,
        "task_language_support": {"malicious_intent": supported_languages},
        "language_models": routes,
        "language_coverage": coverage,
        "language_thresholds": {
            language: float(settings["threshold"]) for language, settings in routes.items()
        },
        "test_metrics_by_language": metrics_by_language,
        "test_metrics": _conservative_summary(metrics_by_language, supported_languages),
        "metric_scope": "worst validated language across routed artifacts",
        "files": sorted({settings["file"] for settings in routes.values()}),
        "runtime_ready": True,
        "runtime_note": (
            "Project inference selects an independently validated artifact and threshold "
            "from the dominant supported language."
        ),
        "created_at": now.isoformat(),
        "promotion_archive": str(archive),
    })
    temporary = active_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(active_path)
    return merged


def _require_gate(metrics: Any, language: str) -> None:
    if not isinstance(metrics, dict):
        raise SystemExit(f"Missing test metrics for GATv2 language: {language}")
    if not (
        float(metrics.get("precision", 0.0)) >= DEPLOYMENT_GATE["minimum_precision"]
        and float(metrics.get("false_positive_rate", 1.0))
        <= DEPLOYMENT_GATE["maximum_false_positive_rate"]
        and float(metrics.get("false_negative_rate", 1.0))
        <= DEPLOYMENT_GATE["maximum_false_negative_rate"]
    ):
        raise SystemExit(f"GATv2 language failed deployment gate: {language}")


def _conservative_summary(metrics_by_language: dict[str, Any], languages: list[str]) -> dict[str, Any]:
    rows = [metrics_by_language[language] for language in languages]
    output = {
        key: min(float(row[key]) for row in rows)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }
    output.update({
        key: max(float(row[key]) for row in rows)
        for key in ("false_positive_rate", "false_negative_rate")
    })
    output["samples"] = sum(int(row.get("samples", 0)) for row in rows)
    output["aggregation"] = "worst_validated_language"
    output["supported_languages"] = list(languages)
    return output


def _archive_active(manifest: dict[str, Any], manifest_path: Path, model_dir: Path) -> Path:
    version = str(manifest.get("model_version") or "unknown").replace("/", "_")
    archive = model_dir / "archive" / version
    if archive.exists():
        archive = archive.with_name(
            f"{archive.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, archive / manifest_path.name)
    for name in manifest.get("files") or []:
        source = model_dir / str(name)
        if source.is_file():
            shutil.copy2(source, archive / source.name)
    return archive


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote routed multilingual GATv2 artifacts")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--legacy-evaluation", required=True, type=Path)
    args = parser.parse_args()
    result = promote(args.candidate_dir, args.model_dir, args.legacy_evaluation)
    print(json.dumps({
        "model_version": result["model_version"],
        "supported_languages": result["supported_languages"],
        "test_metrics": result["test_metrics"],
        "promotion_archive": result["promotion_archive"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
