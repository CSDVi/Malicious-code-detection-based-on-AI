"""Promote one independently passing language from a multilingual CodeT5+ candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from attack_detection.codet5p_registry import (  # noqa: E402
    ARTIFACT_ROOT,
    MODEL_DIR,
    register_candidate,
    registry_view,
)
from attack_detection.training.artifact_contracts import validate_codet5p_manifest  # noqa: E402


QUALITY_GATE = {
    "minimum_precision": 0.90,
    "maximum_false_positive_rate": 0.10,
    "maximum_false_negative_rate": 0.10,
}


def promote(candidate_version: str, language: str) -> dict[str, Any]:
    language = language.strip().lower()
    registry = registry_view()
    entry = next(
        (
            item
            for item in registry.get("versions", [])
            if str(item.get("version") or "") == candidate_version
        ),
        None,
    )
    if not entry:
        raise ValueError(f"unknown CodeT5+ candidate: {candidate_version}")
    if entry.get("kind") != "fine_tuned_candidate":
        raise ValueError("only a fine-tuned candidate can be promoted")
    artifact_relative = str(entry.get("artifact_dir") or "")
    source = (MODEL_DIR / artifact_relative).resolve()
    manifest_path = source / "codet5p_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_codet5p_manifest(manifest, source, require_runtime_ready=False)
    if errors:
        raise RuntimeError("candidate artifact is incomplete: " + "; ".join(errors))
    if language not in {str(value).lower() for value in manifest.get("supported_languages") or []}:
        raise ValueError(f"candidate does not contain language: {language}")

    validation = manifest.get("validation_metrics") or {}
    language_metrics = (manifest.get("test_metrics_by_language") or {}).get(language) or {}
    if not _passes(validation):
        raise RuntimeError(f"global validation gate failed: {_gate_summary(validation)}")
    if not _passes(language_metrics):
        raise RuntimeError(f"{language} test gate failed: {_gate_summary(language_metrics)}")
    if not language_metrics.get("positive_samples") or not language_metrics.get("negative_samples"):
        raise RuntimeError(f"{language} test partition does not contain both classes")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_short = str(manifest.get("task") or "task").replace("_intent", "").replace("_risk", "")
    version = f"codet5p-{task_short}-{language}-{timestamp}-{str(manifest.get('dataset_sha256') or '')[:10]}"
    destination = ARTIFACT_ROOT / version
    if destination.exists():
        raise FileExistsError(f"runtime artifact already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in manifest.get("files") or []:
            artifact_name = str(name)
            if Path(artifact_name).name != artifact_name:
                raise RuntimeError(f"unsafe candidate artifact name: {artifact_name}")
            shutil.copy2(source / artifact_name, destination / artifact_name)
        promoted = dict(manifest)
        promoted.update({
            "model_version": version,
            "base_version": candidate_version,
            "supported_languages": [language],
            "test_metrics": language_metrics,
            "test_metrics_by_language": {language: language_metrics},
            "passed_deployment_gate": True,
            "runtime_ready": True,
            "promoted_from": candidate_version,
            "promotion_scope": {
                "task": str(manifest.get("task") or ""),
                "language": language,
                "validation_gate": _gate_summary(validation),
                "test_gate": _gate_summary(language_metrics),
            },
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        (destination / "codet5p_manifest.json").write_text(
            json.dumps(promoted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        errors = validate_codet5p_manifest(promoted, destination)
        if errors:
            raise RuntimeError("promoted runtime artifact is invalid: " + "; ".join(errors))
        registered = register_candidate(promoted, destination, activate=True)
    except Exception:
        shutil.rmtree(destination)
        raise
    return {
        "model_version": version,
        "task": promoted["task"],
        "language": language,
        "artifact_dir": str(destination),
        "validation": _gate_summary(validation),
        "test": _gate_summary(language_metrics),
        "registered": registered,
    }


def _passes(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("precision", 0.0)) >= QUALITY_GATE["minimum_precision"]
        and float(metrics.get("false_positive_rate", 1.0))
        <= QUALITY_GATE["maximum_false_positive_rate"]
        and float(metrics.get("false_negative_rate", 1.0))
        <= QUALITY_GATE["maximum_false_negative_rate"]
    )


def _gate_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": int(metrics.get("samples") or 0),
        "positive_samples": int(metrics.get("positive_samples") or 0),
        "negative_samples": int(metrics.get("negative_samples") or 0),
        "precision": float(metrics.get("precision", 0.0)),
        "false_positive_rate": float(metrics.get("false_positive_rate", 1.0)),
        "false_negative_rate": float(metrics.get("false_negative_rate", 1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    result = promote(args.candidate_version, args.language)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
