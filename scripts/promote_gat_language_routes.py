"""Merge strict-gated language-specific GATv2 candidates into the active router."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STRICT_GATE = {
    "minimum_precision": 0.9,
    "maximum_false_positive_rate": 0.1,
    "maximum_false_negative_rate": 0.1,
}


def promote(model_dir: Path, candidate_dirs: list[Path]) -> dict[str, Any]:
    active_path = model_dir / "gatv2_manifest.json"
    active = _read_json(active_path)
    candidates = [_read_json(path / "gatv2_manifest.json") for path in candidate_dirs]
    routes = copy.deepcopy(active.get("language_models") or {})
    metrics = copy.deepcopy(active.get("test_metrics_by_language") or {})
    coverage = copy.deepcopy(active.get("language_coverage") or {})
    validations = copy.deepcopy(active.get("validation_metrics_by_language") or {})
    files = {str(value) for value in active.get("files") or []}
    datasets = list(active.get("training_datasets") or [{
        "model_version": active.get("model_version"),
        "dataset_sha256": active.get("dataset_sha256"),
    }])

    for language in active.get("supported_languages") or []:
        if language not in routes:
            raise ValueError(f"active GATv2 route is missing settings: {language}")
        _require_strict_gate(metrics.get(language), f"active/{language}")

    staged: list[tuple[Path, str]] = []
    for candidate_dir, candidate in zip(candidate_dirs, candidates):
        if candidate.get("runtime_ready") is not True:
            raise ValueError(f"candidate is not runtime-ready: {candidate_dir}")
        supported = list(candidate.get("supported_languages") or [])
        if not supported:
            raise ValueError(f"candidate has no strict-gated language: {candidate_dir}")
        source_name = str(
            candidate.get("artifact")
            or (candidate.get("files") or ["gatv2_classifier.pt"])[0]
        )
        source = candidate_dir / source_name
        if not source.is_file():
            raise FileNotFoundError(f"candidate artifact is missing: {source}")
        version = str(candidate["model_version"]).replace("/", "_")
        suffix = Path(source_name).suffix
        target_name = f"gatv2_classifier__{version}{suffix}"
        staged.append((source, target_name))
        datasets.append({
            "model_version": candidate.get("model_version"),
            "dataset_sha256": candidate.get("dataset_sha256"),
        })
        for language in supported:
            language_metrics = (candidate.get("test_metrics_by_language") or {}).get(language)
            _require_strict_gate(language_metrics, f"candidate/{language}")
            routes[language] = {
                "file": target_name,
                "languages": copy.deepcopy(candidate["languages"]),
                "training": copy.deepcopy(candidate["training"]),
                "temperature": float(candidate["temperature"]),
                "threshold": float(candidate["language_thresholds"][language]),
                "model_version": str(candidate["model_version"]),
            }
            metrics[language] = copy.deepcopy(language_metrics)
            coverage[language] = copy.deepcopy(candidate["language_coverage"][language])
            validations[language] = copy.deepcopy(
                (candidate.get("validation_metrics_by_language") or {}).get(language) or {}
            )
            files.add(target_name)

    supported_languages = sorted(routes)
    for language in supported_languages:
        _require_strict_gate(metrics.get(language), language)
    now = datetime.now(timezone.utc)
    fingerprint = hashlib.sha256(
        json.dumps(datasets, sort_keys=True).encode("utf-8")
    ).hexdigest()
    archive = _archive_active(active, active_path, model_dir, now)
    for source, target_name in staged:
        shutil.copy2(source, model_dir / target_name)

    merged = copy.deepcopy(active)
    merged.update({
        "schema_version": 3,
        "model_version": f"gatv2-routed-{now.strftime('%Y%m%dT%H%M%SZ')}-{fingerprint[:12]}",
        "architecture": "Language-routed torch_geometric.nn.GATv2Conv classifiers",
        "dataset_sha256": fingerprint,
        "training_datasets": datasets,
        "supported_languages": supported_languages,
        "task_language_support": {"malicious_intent": supported_languages},
        "language_models": routes,
        "language_coverage": coverage,
        "language_thresholds": {
            language: float(routes[language]["threshold"]) for language in supported_languages
        },
        "validation_metrics_by_language": validations,
        "test_metrics_by_language": metrics,
        "test_metrics": _conservative_summary(metrics, supported_languages),
        "deployment_gate": STRICT_GATE,
        "metric_scope": "worst strict-gated language across routed artifacts",
        "files": sorted(files),
        "runtime_ready": True,
        "runtime_note": (
            "Project inference selects a strict-gated language-specific artifact, feature "
            "schema, pooling strategy, calibration temperature, and threshold."
        ),
        "created_at": now.isoformat(timespec="seconds"),
        "promotion_archive": str(archive),
    })
    temporary = active_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(active_path)
    return merged


def _require_strict_gate(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"missing GATv2 metrics: {label}")
    if not (
        float(value.get("precision", 0.0)) >= STRICT_GATE["minimum_precision"]
        and float(value.get("false_positive_rate", 1.0))
        <= STRICT_GATE["maximum_false_positive_rate"]
        and float(value.get("false_negative_rate", 1.0))
        <= STRICT_GATE["maximum_false_negative_rate"]
    ):
        raise ValueError(f"GATv2 route failed strict gate: {label}")


def _conservative_summary(metrics: dict[str, Any], languages: list[str]) -> dict[str, Any]:
    rows = [metrics[language] for language in languages]
    output = {
        key: min(float(row[key]) for row in rows)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }
    output.update({
        key: max(float(row[key]) for row in rows)
        for key in ("false_positive_rate", "false_negative_rate")
    })
    output["samples"] = sum(int(row.get("samples", 0)) for row in rows)
    output["aggregation"] = "worst_strict_gated_language"
    output["supported_languages"] = list(languages)
    return output


def _archive_active(
    manifest: dict[str, Any],
    manifest_path: Path,
    model_dir: Path,
    now: datetime,
) -> Path:
    version = str(manifest.get("model_version") or "unknown").replace("/", "_")
    archive = model_dir / "archive" / f"{version}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, archive / manifest_path.name)
    for name in manifest.get("files") or []:
        source = model_dir / str(name)
        if source.is_file():
            shutil.copy2(source, archive / source.name)
    return archive


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote strict-gated GATv2 language routes")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, action="append", type=Path)
    args = parser.parse_args()
    result = promote(
        args.model_dir.resolve(),
        [path.resolve() for path in args.candidate_dir],
    )
    print(json.dumps({
        "model_version": result["model_version"],
        "supported_languages": result["supported_languages"],
        "test_metrics": result["test_metrics"],
        "promotion_archive": result["promotion_archive"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
