"""Publish strict XGBoost routes plus explicit all-language fallbacks.

Strict routes retain their independent evaluation scope and release metrics.
Every other recognized source language is routed to a shared compact hybrid
model and is explicitly marked as an unvalidated runtime fallback.
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
    return artifact, json.loads(metrics_path.read_text(encoding="utf-8"))


def _strict_candidate_routes(
    candidate_root: Path,
) -> dict[str, tuple[Path, dict[str, Any], bool | None]]:
    routes: dict[str, tuple[Path, dict[str, Any], bool | None]] = {}
    source_protocol: dict[str, bool] = {}
    for metrics_path in sorted(candidate_root.glob("sourceheldout_vulnerability_*.json")):
        language = metrics_path.stem.removeprefix("sourceheldout_vulnerability_").removesuffix("_20260725")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        artifact = metrics_path.with_suffix(".joblib")
        passed = _passes(metrics.get("test"))
        source_protocol[language] = passed
        if artifact.is_file() and passed:
            routes[language] = (artifact, metrics, True)
    for metrics_path in sorted(candidate_root.glob("hybrid_vulnerability_*_20260725_ext.json")):
        language = metrics_path.stem.removeprefix("hybrid_vulnerability_").removesuffix("_20260725_ext")
        if language == "all":
            continue
        if language in source_protocol:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        artifact = metrics_path.with_suffix(".joblib")
        if artifact.is_file() and _passes(metrics.get("test")):
            routes[language] = (artifact, metrics, None)
    return routes


def _copy_existing_artifacts(
    base_dir: Path,
    routes: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    for route in routes.values():
        artifact = str(route.get("artifact") or "")
        if not artifact:
            continue
        source = base_dir / artifact
        if not source.is_file():
            raise SystemExit(f"base route artifact is missing: {source}")
        target = output_dir / artifact
        if not target.exists():
            shutil.copy2(source, target)


def _normalize_strict_routes(routes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for language, original in routes.items():
        route = dict(original)
        passed = _passes(route.get("deployment"))
        route["quality_gate_passed"] = passed
        route["release_scope"] = (
            "strict_independent_evaluation"
            if passed
            else "retained_base_route_without_strict_file_gate"
        )
        route.setdefault("source_heldout_verified", None)
        route.setdefault("evaluation_protocol", "family_split_only")
        output[str(language)] = route
    return output


def _fallback_route(
    *,
    artifact: str,
    metrics: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    thresholds = dict((metrics.get("selected") or {}).get("thresholds") or {"decision": 0.5})
    return {
        "artifact": artifact,
        "feature_mode": str(metrics.get("feature_mode") or "hybrid_hash"),
        "thresholds": thresholds,
        "evaluation_scope": "file_runtime_fallback",
        "deployment": metrics.get("test"),
        "validation": (metrics.get("selected") or {}).get("validation"),
        "test": metrics.get("test"),
        "quality_gate_passed": False,
        "source_heldout_verified": False,
        "evaluation_protocol": "not_evaluated",
        "release_scope": "runtime_fallback_unvalidated_for_this_language",
        "task": task,
    }


def promote(
    *,
    dataset: Path,
    base_version: str,
    malicious_fallback_prefix: Path,
    vulnerability_fallback_prefix: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    base_dir = REGISTRY_ROOT / base_version
    base_metrics_path = base_dir / "xgb_metrics.json"
    if not base_metrics_path.is_file():
        raise SystemExit(f"base metrics are missing: {base_metrics_path}")
    base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))

    malicious_artifact, malicious_fallback_metrics = _candidate(malicious_fallback_prefix)
    vulnerability_artifact, vulnerability_fallback_metrics = _candidate(vulnerability_fallback_prefix)

    dataset_hash = _sha256(dataset)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"xgb-universal-{stamp}-{dataset_hash[:10]}"
    output_dir = REGISTRY_ROOT / version
    output_dir.mkdir(parents=True, exist_ok=False)

    tasks = {
        name: dict(task)
        for name, task in (base_metrics.get("tasks") or {}).items()
    }
    malicious_routes = _normalize_strict_routes(
        dict(tasks["malicious_intent"].get("language_routes") or {})
    )
    vulnerability_routes = _normalize_strict_routes(
        dict(tasks["vulnerability_risk"].get("language_routes") or {})
    )
    for metrics_path in sorted(candidate_root.glob("sourceheldout_vulnerability_*.json")):
        language = metrics_path.stem.removeprefix("sourceheldout_vulnerability_").removesuffix("_20260725")
        if language not in vulnerability_routes:
            continue
        source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        source_report = source_metrics.get("test")
        vulnerability_routes[language]["source_heldout_verified"] = _passes(source_report)
        vulnerability_routes[language]["source_heldout_report"] = source_report
        vulnerability_routes[language]["evaluation_protocol"] = "source_heldout"
        if not _passes(source_report):
            vulnerability_routes[language]["release_scope"] = (
                "advisory_only_after_failed_source_heldout_evaluation"
            )
    _copy_existing_artifacts(base_dir, malicious_routes, output_dir)
    _copy_existing_artifacts(base_dir, vulnerability_routes, output_dir)

    # Reuse the base task artifact for the fallback when possible.  This keeps
    # the runtime cache at one loaded bundle per task instead of duplicating a
    # byte-identical model under another filename.
    malicious_fallback_name = "xgb_malicious_classifier.joblib"
    vulnerability_fallback_name = "xgb_vulnerability_classifier.joblib"
    if not (output_dir / malicious_fallback_name).exists():
        shutil.copy2(malicious_artifact, output_dir / malicious_fallback_name)
    if not (output_dir / vulnerability_fallback_name).exists():
        shutil.copy2(vulnerability_artifact, output_dir / vulnerability_fallback_name)

    strict_vulnerability = _strict_candidate_routes(candidate_root)
    for language, (artifact, candidate_metrics, source_heldout_verified) in strict_vulnerability.items():
        name = f"xgb_vulnerability_{language}.joblib"
        shutil.copy2(artifact, output_dir / name)
        vulnerability_routes[language] = {
            "artifact": name,
            "feature_mode": "hybrid_hash",
            "thresholds": dict(
                (candidate_metrics.get("selected") or {}).get("thresholds")
                or {"decision": 0.5}
            ),
            "evaluation_scope": "file",
            "deployment": candidate_metrics.get("test"),
            "validation": (candidate_metrics.get("selected") or {}).get("validation"),
            "test": candidate_metrics.get("test"),
            "quality_gate_passed": True,
            "release_scope": "strict_independent_evaluation",
            "source_heldout_verified": source_heldout_verified,
            "evaluation_protocol": (
                "source_heldout" if source_heldout_verified is True else "family_split_only"
            ),
        }

    malicious_fallback = _fallback_route(
        artifact=malicious_fallback_name,
        metrics=malicious_fallback_metrics,
        task="malicious_intent",
    )
    vulnerability_fallback = _fallback_route(
        artifact=vulnerability_fallback_name,
        metrics=vulnerability_fallback_metrics,
        task="vulnerability_risk",
    )
    for language in ALL_LANGUAGES:
        malicious_routes.setdefault(language, dict(malicious_fallback))
        vulnerability_routes.setdefault(language, dict(vulnerability_fallback))

    for task_name, routes in (
        ("malicious_intent", malicious_routes),
        ("vulnerability_risk", vulnerability_routes),
    ):
        task = tasks[task_name]
        strict = sorted(
            language
            for language, route in routes.items()
            if (
                route.get("quality_gate_passed") is True
                and route.get("source_heldout_verified") is not False
            )
        )
        fallback = sorted(set(ALL_LANGUAGES) - set(strict))
        task.update({
            "ready": True,
            "supported_languages": ALL_LANGUAGES,
            "strict_supported_languages": strict,
            "fallback_languages": fallback,
            "language_routes": routes,
            "quality_gate_passed": bool(strict),
            "release_scope": (
                "strict routes where independently validated; all other "
                "recognized languages use an explicitly unvalidated fallback"
            ),
        })

    metrics = dict(base_metrics)
    metrics.update({
        "schema_version": 4,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "samples_total": sum(1 for _ in dataset.open("r", encoding="utf-8")),
        "feature_mode": "universal_language_routes_with_shared_hybrid_fallbacks",
        "tasks": tasks,
        "active_routes": {
            task: ALL_LANGUAGES
            for task in ("malicious_intent", "vulnerability_risk")
        },
        "quality_gate": {
            "requirements": GATE,
            "passed": True,
            "scope": "strict route metrics only; fallback predictions are marked unvalidated",
            "strict_routes": {
                task: tasks[task]["strict_supported_languages"]
                for task in tasks
            },
        },
        "published": True,
    })
    metrics_path = output_dir / "xgb_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    artifacts = [
        {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
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
        "dataset_sha256": dataset_hash,
        "samples_training_eligible": metrics["samples_total"],
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

    # Copy every version artifact first and switch metrics last, so an engine
    # reload never sees routes whose artifact has not arrived yet.
    for path in output_dir.iterdir():
        if path.name != "xgb_metrics.json":
            shutil.copy2(path, MODEL_ROOT / path.name)
    temporary_metrics = MODEL_ROOT / "xgb_metrics.json.tmp"
    shutil.copy2(metrics_path, temporary_metrics)
    os.replace(temporary_metrics, MODEL_ROOT / "xgb_metrics.json")

    return {
        "model_version": version,
        "languages": ALL_LANGUAGES,
        "strict_routes": {
            task: tasks[task]["strict_supported_languages"]
            for task in tasks
        },
        "fallback_routes": {
            task: tasks[task]["fallback_languages"]
            for task in tasks
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--base-version", required=True)
    parser.add_argument("--malicious-fallback", required=True, type=Path)
    parser.add_argument("--vulnerability-fallback", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        promote(
            dataset=args.dataset,
            base_version=args.base_version,
            malicious_fallback_prefix=args.malicious_fallback,
            vulnerability_fallback_prefix=args.vulnerability_fallback,
            candidate_root=args.candidate_root,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
