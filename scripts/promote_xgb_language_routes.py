"""Promote only XGBoost language routes that pass the strict release gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "backend" / "models"
REGISTRY_ROOT = MODEL_ROOT / "xgb_registry"
REGISTRY_PATH = MODEL_ROOT / "xgb_registry.json"
TASK_ARTIFACTS = {
    "malicious_intent": "xgb_malicious_classifier.joblib",
    "vulnerability_risk": "xgb_vulnerability_classifier.joblib",
}
GATE = {
    "min_precision": 0.90,
    "max_false_positive_rate": 0.10,
    "max_false_negative_rate": 0.10,
}


def _passes(metrics: dict[str, Any]) -> bool:
    try:
        return (
            float(metrics["precision"]) >= GATE["min_precision"]
            and float(metrics["false_positive_rate"]) <= GATE["max_false_positive_rate"]
            and float(metrics["false_negative_rate"]) <= GATE["max_false_negative_rate"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote(candidate_version: str) -> dict[str, Any]:
    source_dir = REGISTRY_ROOT / candidate_version
    source_metrics_path = source_dir / "xgb_metrics.json"
    if not source_metrics_path.is_file():
        raise SystemExit(f"candidate metrics not found: {source_metrics_path}")
    metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"xgb-routed-{stamp}-{str(metrics.get('dataset_sha256') or '')[:10]}"
    output_dir = REGISTRY_ROOT / version
    output_dir.mkdir(parents=True, exist_ok=False)

    task_gate: dict[str, bool] = {}
    active_routes: dict[str, list[str]] = {}
    for task_name, artifact in TASK_ARTIFACTS.items():
        task = dict((metrics.get("tasks") or {}).get(task_name) or {})
        by_language = task.get("by_language") or {}
        validated = sorted(
            language
            for language, language_metrics in by_language.items()
            if isinstance(language_metrics, dict) and _passes(language_metrics)
        )
        active_routes[task_name] = validated
        if validated:
            source_artifact = source_dir / artifact
            if not source_artifact.is_file():
                raise SystemExit(f"candidate artifact not found: {source_artifact}")
            shutil.copy2(source_artifact, output_dir / artifact)
            task["ready"] = True
            task["supported_languages"] = validated
            task["language_thresholds"] = {
                language: float((task.get("thresholds") or {}).get("decision", 0.5))
                for language in validated
            }
            if len(validated) == 1:
                task["deployment"] = dict(by_language[validated[0]])
            task["quality_gate_passed"] = all(_passes(by_language[language]) for language in validated)
            task["release_scope"] = "strictly validated language routes only"
        else:
            task["ready"] = False
            task["supported_languages"] = []
            task["quality_gate_passed"] = False
            task["reason"] = "no language route passed Precision/FPR/FNR release gate"
        metrics.setdefault("tasks", {})[task_name] = task
        task_gate[task_name] = bool(task["quality_gate_passed"])

    if not any(task_gate.values()):
        raise SystemExit("no XGBoost language route passes the strict gate")
    metrics["schema_version"] = 2
    metrics["model_version"] = version
    metrics["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics["active_routes"] = active_routes
    metrics["quality_gate"] = {
        "requirements": GATE,
        "tasks": task_gate,
        "passed": True,
        "scope": "at least one task/language route passed; failed routes are unavailable",
    }
    metrics["published"] = True
    metrics_path = output_dir / "xgb_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    ]
    registry = (
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if REGISTRY_PATH.is_file()
        else {"schema_version": 1, "active_version": "", "versions": []}
    )
    registry["versions"] = [
        item for item in registry.get("versions", [])
        if item.get("version") != version
    ]
    registry["versions"].insert(0, {
        "version": version,
        "created_at": metrics["created_at"],
        "dataset_sha256": metrics.get("dataset_sha256"),
        "samples_training_eligible": metrics.get("samples_training_eligible"),
        "tasks": metrics["tasks"],
        "active_routes": active_routes,
        "artifacts": artifacts,
        "published": True,
        "quality_gate": metrics["quality_gate"],
    })
    registry["active_version"] = version
    registry["activated_at"] = metrics["created_at"]
    temporary = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, REGISTRY_PATH)

    required = {"xgb_metrics.json"} | {
        artifact
        for task, artifact in TASK_ARTIFACTS.items()
        if task_gate[task]
    }
    for name in {"xgb_metrics.json", *TASK_ARTIFACTS.values()}:
        source = output_dir / name
        target = MODEL_ROOT / name
        if name in required:
            shutil.copy2(source, target)
        elif target.exists():
            target.unlink()
    return {
        "model_version": version,
        "active_routes": active_routes,
        "quality_gate": metrics["quality_gate"],
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-version", required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args.candidate_version), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
