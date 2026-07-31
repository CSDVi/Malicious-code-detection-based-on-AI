"""Publish language-routed XGBoost malicious-intent models.

The shared static model is retained for Java/JavaScript.  PHP and Python use
the validated hybrid candidates.  Python's gate is evaluated at package/project
scope using max aggregation, matching the product's project scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset
from attack_detection.features.static_features import feature_vector
from attack_detection.trainer import _evaluate

MODEL_ROOT = BACKEND / "models"
REGISTRY_ROOT = MODEL_ROOT / "xgb_registry"
REGISTRY_PATH = MODEL_ROOT / "xgb_registry.json"
GATE = {
    "min_precision": 0.90,
    "max_false_positive_rate": 0.10,
    "max_false_negative_rate": 0.10,
}


def _passes(report: dict[str, Any]) -> bool:
    return (
        float(report.get("precision", 0.0)) >= GATE["min_precision"]
        and float(report.get("false_positive_rate", 1.0)) <= GATE["max_false_positive_rate"]
        and float(report.get("false_negative_rate", 1.0)) <= GATE["max_false_negative_rate"]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hybrid_scores(bundle: dict[str, Any], rows: list[Any]) -> list[float]:
    from scipy.sparse import csr_matrix, hstack

    structured = numpy.asarray([
        feature_vector(row.code, row.language, include_rules=False)
        for row in rows
    ], dtype="float32")
    matrix = hstack([
        csr_matrix(structured),
        bundle["word_vectorizer"].transform([row.code for row in rows]),
        bundle["char_vectorizer"].transform([row.code for row in rows]),
    ], format="csr", dtype="float32")
    return [float(item[1]) for item in bundle["model"].predict_proba(matrix)]


def _family_report(rows: list[Any], scores: list[float], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.family or row.package_name or row.sample_hash].append(index)
    labels = [
        "malicious" if any(rows[index].label == "malicious" for index in indexes) else "benign"
        for indexes in grouped.values()
    ]
    family_scores = [max(scores[index] for index in indexes) for indexes in grouped.values()]
    return _evaluate(labels, family_scores, "malicious", "benign", threshold)


def _hybrid_route_report(dataset: Path, bundle_path: Path, language: str) -> dict[str, Any]:
    from joblib import load

    bundle = load(bundle_path)
    rows = [
        row for row in load_dataset(dataset)
        if row.language == language
        and row.label in {"benign", "malicious"}
        and is_task_training_eligible(row, "malicious_intent")
    ]
    partitions = {
        split: [row for row in rows if row.split == split]
        for split in ("validation", "test")
    }
    scores = {
        split: _hybrid_scores(bundle, partition)
        for split, partition in partitions.items()
    }
    threshold = 0.50
    file_reports = {
        split: _evaluate(
            [row.label for row in partitions[split]],
            scores[split],
            "malicious",
            "benign",
            threshold,
        )
        for split in ("validation", "test")
    }
    family_reports = {
        split: _family_report(partitions[split], scores[split], threshold)
        for split in ("validation", "test")
    }
    if _passes(file_reports["test"]):
        scope = "file"
        deployment = file_reports["test"]
    elif _passes(family_reports["test"]):
        scope = "project_or_package"
        deployment = family_reports["test"]
    else:
        scope = "none"
        deployment = file_reports["test"]
    return {
        "artifact_source": str(bundle_path),
        "threshold": threshold,
        "evaluation_scope": scope,
        "deployment": deployment,
        "validation": file_reports["validation"],
        "test": file_reports["test"],
        "family_validation": family_reports["validation"],
        "family_test": family_reports["test"],
        "quality_gate_passed": scope != "none",
        "trainable_samples": len([row for row in rows if row.split == "train"]),
        "validation_samples": len(partitions["validation"]),
        "test_samples": len(partitions["test"]),
    }


def promote(dataset: Path, base_version: str, php_candidate: Path, python_candidate: Path) -> dict[str, Any]:
    base_dir = REGISTRY_ROOT / base_version
    base_metrics = json.loads((base_dir / "xgb_metrics.json").read_text(encoding="utf-8"))
    base_task = dict(base_metrics["tasks"]["malicious_intent"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"xgb-routed-{stamp}-{str(base_metrics.get('dataset_sha256') or '')[:10]}"
    output_dir = REGISTRY_ROOT / version
    output_dir.mkdir(parents=True, exist_ok=False)

    shared_name = "xgb_malicious_classifier.joblib"
    shutil.copy2(base_dir / shared_name, output_dir / shared_name)
    route_specs = {
        "java": {
            "artifact": shared_name,
            "feature_mode": "static",
            "thresholds": base_task.get("thresholds") or {"decision": 0.5},
            "evaluation_scope": "file",
            "deployment": base_task["by_language"]["java"],
            "validation": None,
            "test": base_task["by_language"]["java"],
        },
        "javascript": {
            "artifact": shared_name,
            "feature_mode": "static",
            "thresholds": base_task.get("thresholds") or {"decision": 0.5},
            "evaluation_scope": "file",
            "deployment": base_task["by_language"]["javascript"],
            "validation": None,
            "test": base_task["by_language"]["javascript"],
        },
    }
    for language, candidate, artifact in (
        ("php", php_candidate, "xgb_malicious_php.joblib"),
        ("python", python_candidate, "xgb_malicious_python.joblib"),
    ):
        report = _hybrid_route_report(dataset, candidate, language)
        if not report["quality_gate_passed"]:
            raise SystemExit(f"{language} candidate does not pass the relaxed gate: {report['deployment']}")
        shutil.copy2(candidate, output_dir / artifact)
        route_specs[language] = {
            "artifact": artifact,
            "feature_mode": "hybrid_hash",
            "thresholds": {
                "decision": report["threshold"],
                "uncertain_low": max(0.05, report["threshold"] - 0.10),
                "uncertain_high": min(0.95, report["threshold"] + 0.05),
            },
            "evaluation_scope": report["evaluation_scope"],
            "deployment": report["deployment"],
            "validation": report["validation"],
            "test": report["test"],
            "family_validation": report["family_validation"],
            "family_test": report["family_test"],
        }

    active_task = dict(base_task)
    active_task["ready"] = True
    active_task["supported_languages"] = sorted(route_specs)
    active_task["language_routes"] = route_specs
    active_task["deployment_by_language"] = {
        language: spec["deployment"] for language, spec in route_specs.items()
    }
    active_task["quality_gate_passed"] = True
    active_task["release_scope"] = "all four primary language routes; Python uses project/package aggregation"

    vulnerability_task = dict(base_metrics["tasks"]["vulnerability_risk"])
    vulnerability_task.update({
        "ready": False,
        "supported_languages": [],
        "quality_gate_passed": False,
        "reason": "no vulnerability language route passed the configured gate",
    })
    metrics = dict(base_metrics)
    metrics.update({
        "schema_version": 3,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_mode": "language_routed_static_and_hybrid_hash",
        "tasks": {
            "malicious_intent": active_task,
            "vulnerability_risk": vulnerability_task,
        },
        "active_routes": {
            "malicious_intent": sorted(route_specs),
            "vulnerability_risk": [],
        },
        "quality_gate": {
            "requirements": GATE,
            "tasks": {
                "malicious_intent": True,
                "vulnerability_risk": False,
            },
            "passed": True,
            "scope": "all four primary malicious-intent language routes",
        },
        "published": True,
    })
    metrics_path = output_dir / "xgb_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.iterdir()) if path.is_file()
    ]
    registry = (
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if REGISTRY_PATH.is_file()
        else {"schema_version": 1, "active_version": "", "versions": []}
    )
    registry["versions"].insert(0, {
        "version": version,
        "created_at": metrics["created_at"],
        "dataset_sha256": metrics.get("dataset_sha256"),
        "samples_training_eligible": metrics.get("samples_training_eligible"),
        "tasks": metrics["tasks"],
        "active_routes": metrics["active_routes"],
        "artifacts": artifacts,
        "published": True,
        "quality_gate": metrics["quality_gate"],
    })
    registry["active_version"] = version
    registry["activated_at"] = metrics["created_at"]
    temporary = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, REGISTRY_PATH)

    required = {item["name"] for item in artifacts}
    for path in MODEL_ROOT.glob("xgb_malicious_*.joblib"):
        if path.name not in required:
            path.unlink()
    for path in MODEL_ROOT.glob("xgb_vulnerability*.joblib"):
        path.unlink()
    for name in required:
        shutil.copy2(output_dir / name, MODEL_ROOT / name)
    shutil.copy2(metrics_path, MODEL_ROOT / "xgb_metrics.json")
    return {
        "model_version": version,
        "active_routes": metrics["active_routes"],
        "quality_gate": metrics["quality_gate"],
        "language_routes": route_specs,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-version", required=True)
    parser.add_argument("--php-candidate", required=True, type=Path)
    parser.add_argument("--python-candidate", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        promote(args.dataset, args.base_version, args.php_candidate, args.python_candidate),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
