"""Atomically publish independently validated malicious-intent XGBoost routes.

The active universal release remains the base, so unmodified language
fallbacks and the disabled vulnerability artifacts stay on disk.  Every
candidate supplied to this command must pass the public Precision/FPR/FNR gate
on its untouched family-held-out test split or nothing is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.languages import MODEL_LANGUAGES


MODEL_ROOT = BACKEND / "models"
REGISTRY_ROOT = MODEL_ROOT / "xgb_registry"
REGISTRY_PATH = MODEL_ROOT / "xgb_registry.json"
GATE = {
    "min_precision": 0.90,
    "max_false_positive_rate": 0.10,
    "max_false_negative_rate": 0.10,
}
ALL_LANGUAGES = sorted(set(MODEL_LANGUAGES) | {"unknown"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _passes(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    try:
        return (
            float(report["precision"]) >= GATE["min_precision"]
            and float(report["false_positive_rate"]) <= GATE["max_false_positive_rate"]
            and float(report["false_negative_rate"]) <= GATE["max_false_negative_rate"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _candidate(prefix: Path) -> tuple[Path, dict[str, Any]]:
    artifact = prefix.with_suffix(".joblib")
    metrics_path = prefix.with_suffix(".json")
    if not artifact.is_file() or not metrics_path.is_file():
        raise SystemExit(f"candidate is incomplete: {prefix}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(metrics.get("task") or "") != "malicious_intent":
        raise SystemExit(f"candidate is not malicious_intent: {prefix}")
    if not _passes((metrics.get("selected") or {}).get("validation")):
        raise SystemExit(
            f"candidate validation split fails release gate: {prefix}: "
            f"{(metrics.get('selected') or {}).get('validation')}"
        )
    if not _passes(metrics.get("test")):
        raise SystemExit(f"candidate fails release gate: {prefix}: {metrics.get('test')}")
    if (metrics.get("behavior_canary") or {}).get("all_canaries_correct") is not True:
        raise SystemExit(f"candidate behavior canaries are not verified: {prefix}")
    return artifact, metrics


def _pooled(reports: list[dict[str, Any]]) -> dict[str, Any]:
    matrices = [report.get("confusion_matrix") for report in reports]
    if not all(
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        for matrix in matrices
    ):
        return {}
    tn = sum(int(matrix[0][0]) for matrix in matrices)
    fp = sum(int(matrix[0][1]) for matrix in matrices)
    fn = sum(int(matrix[1][0]) for matrix in matrices)
    tp = sum(int(matrix[1][1]) for matrix in matrices)
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    fnr = fn / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    weighted_pr_auc = (
        sum(float(report.get("pr_auc") or 0.0) * int(report.get("samples") or 0) for report in reports)
        / max(1, sum(int(report.get("samples") or 0) for report in reports))
    )
    result = {
        "samples": total,
        "accuracy": round((tn + tp) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pr_auc": round(weighted_pr_auc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "quality_gate_passed": False,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
    result["quality_gate_passed"] = _passes(result)
    return result


def promote(
    *,
    dataset: Path,
    report_path: Path,
    base_version: str,
    candidate_prefixes: list[Path],
) -> dict[str, Any]:
    audit = json.loads(report_path.read_text(encoding="utf-8"))
    if audit.get("family_split_isolation_verified") is not True:
        raise SystemExit("dataset family split isolation is not verified")
    base_dir = REGISTRY_ROOT / base_version
    base_metrics_path = base_dir / "xgb_metrics.json"
    if not base_metrics_path.is_file():
        raise SystemExit(f"base metrics are missing: {base_metrics_path}")
    metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    tasks = dict(metrics.get("tasks") or {})
    malicious_task = dict(tasks.get("malicious_intent") or {})
    routes = {
        str(language): dict(route)
        for language, route in (malicious_task.get("language_routes") or {}).items()
    }

    loaded_candidates: list[tuple[Path, dict[str, Any]]] = []
    candidate_languages: set[str] = set()
    for prefix in candidate_prefixes:
        artifact, candidate_metrics = _candidate(prefix)
        language = str(candidate_metrics.get("language") or "")
        if not language or language == "all":
            raise SystemExit(f"candidate has invalid route language: {prefix}")
        if language in candidate_languages:
            raise SystemExit(f"duplicate candidate language: {language}")
        candidate_languages.add(language)
        loaded_candidates.append((artifact, candidate_metrics))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dataset_hash = _sha256(dataset)
    version = f"xgb-malicious-routed-{stamp}-{dataset_hash[:10]}"
    output_dir = REGISTRY_ROOT / version
    output_dir.mkdir(parents=True, exist_ok=False)
    for path in base_dir.iterdir():
        if path.is_file() and path.name != "xgb_metrics.json":
            shutil.copy2(path, output_dir / path.name)

    promoted: dict[str, dict[str, Any]] = {}
    for artifact, candidate_metrics in loaded_candidates:
        language = str(candidate_metrics["language"])
        artifact_name = f"xgb_malicious_{language}.joblib"
        shutil.copy2(artifact, output_dir / artifact_name)
        threshold_info = dict((candidate_metrics.get("selected") or {}).get("thresholds") or {})
        candidate_feature_mode = str(candidate_metrics.get("feature_mode") or "")
        route_feature_mode = (
            "structured_static"
            if candidate_feature_mode == "structured_static"
            else "hybrid_hash"
        )
        route = {
            "artifact": artifact_name,
            "feature_mode": route_feature_mode,
            "thresholds": threshold_info or {"decision": 0.5},
            "evaluation_scope": "file",
            "deployment": candidate_metrics["test"],
            "validation": (candidate_metrics.get("selected") or {}).get("validation"),
            "test": candidate_metrics["test"],
            "quality_gate_passed": True,
            "source_heldout_verified": None,
            "evaluation_protocol": "family_split",
            "release_scope": "strict_family_held_out_file_evaluation",
            "dataset": str(
                candidate_metrics.get("dataset") or dataset.resolve()
            ),
            "split_counts": candidate_metrics.get("split_counts"),
            "split_label_counts": candidate_metrics.get("split_label_counts"),
            "sampling_protocol": candidate_metrics.get("sampling_protocol"),
        }
        routes[language] = route
        promoted[language] = route

    strict = sorted(
        language
        for language, route in routes.items()
        if (
            route.get("quality_gate_passed") is True
            and route.get("source_heldout_verified") is not False
            and _passes(route.get("deployment"))
        )
    )
    fallback = sorted(set(ALL_LANGUAGES) - set(strict))
    strict_reports = [routes[language]["deployment"] for language in strict]
    pooled = _pooled(strict_reports)
    if not _passes(pooled):
        raise SystemExit(f"pooled strict-route metrics fail release gate: {pooled}")

    malicious_task.update({
        "ready": True,
        "supported_languages": ALL_LANGUAGES,
        "strict_supported_languages": strict,
        "fallback_languages": fallback,
        "language_routes": routes,
        "deployment_by_language": {
            language: route.get("deployment")
            for language, route in routes.items()
        },
        "deployment": pooled,
        "quality_gate_passed": True,
        "release_scope": (
            "strict family-held-out routes where independently validated; "
            "remaining recognized languages retain the prior unvalidated fallback"
        ),
    })
    tasks["malicious_intent"] = malicious_task
    metrics.update({
        "schema_version": max(5, int(metrics.get("schema_version") or 1)),
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "samples_total": int(audit.get("output_rows") or 0),
        "samples_training_eligible": int(
            audit.get("training_eligible_rows")
            or audit.get("output_rows")
            or 0
        ),
        "feature_mode": "universal_language_routes_with_validated_malicious_hybrid_routes",
        "tasks": tasks,
        "active_routes": {
            "malicious_intent": ALL_LANGUAGES,
            **{
                key: value
                for key, value in (metrics.get("active_routes") or {}).items()
                if key != "malicious_intent"
            },
        },
        "quality_gate": {
            "requirements": GATE,
            "passed": True,
            "scope": "pooled strict malicious-intent routes and every newly promoted route",
            "strict_routes": {"malicious_intent": strict},
            "new_routes": sorted(promoted),
        },
        "published": True,
    })
    metrics_path = output_dir / "xgb_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = [
        {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    ]
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["versions"] = [
        item for item in registry.get("versions", [])
        if item.get("version") != version
    ]
    registry["versions"].insert(0, {
        "version": version,
        "created_at": metrics["created_at"],
        "dataset_sha256": dataset_hash,
        "samples_training_eligible": metrics["samples_training_eligible"],
        "tasks": tasks,
        "active_routes": metrics["active_routes"],
        "artifacts": artifacts,
        "published": True,
        "quality_gate": metrics["quality_gate"],
    })
    registry["active_version"] = version
    registry["activated_at"] = metrics["created_at"]
    temporary_registry = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_registry, REGISTRY_PATH)

    # Keep disabled-task artifacts untouched.  Copy route artifacts first and
    # atomically switch the metrics file last.
    for path in output_dir.iterdir():
        if path.name != "xgb_metrics.json":
            shutil.copy2(path, MODEL_ROOT / path.name)
    temporary_metrics = MODEL_ROOT / "xgb_metrics.json.tmp"
    shutil.copy2(metrics_path, temporary_metrics)
    os.replace(temporary_metrics, MODEL_ROOT / "xgb_metrics.json")
    return {
        "model_version": version,
        "promoted_routes": sorted(promoted),
        "strict_routes": strict,
        "fallback_routes": fallback,
        "pooled_strict_metrics": pooled,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-report", required=True, type=Path)
    parser.add_argument("--base-version", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=Path,
        help="candidate prefix without .json/.joblib; repeat for each language",
    )
    args = parser.parse_args()
    print(json.dumps(
        promote(
            dataset=args.dataset,
            report_path=args.dataset_report,
            base_version=args.base_version,
            candidate_prefixes=args.candidate,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
