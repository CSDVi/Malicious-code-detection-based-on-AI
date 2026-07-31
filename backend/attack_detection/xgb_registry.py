"""Activation and rollback helpers for versioned XGBoost artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REGISTRY_ROOT = MODEL_DIR / "xgb_registry"
REGISTRY_PATH = MODEL_DIR / "xgb_registry.json"
MODEL_FILES = (
    "xgb_malicious_classifier.joblib",
    "xgb_vulnerability_classifier.joblib",
    "xgb_metrics.json",
)

TASK_ARTIFACTS = {
    "malicious_intent": "xgb_malicious_classifier.joblib",
    "vulnerability_risk": "xgb_vulnerability_classifier.joblib",
}


def registry_view() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": 1, "active_version": "", "versions": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def activate_version(version: str) -> dict[str, Any]:
    registry = registry_view()
    entry = next((item for item in registry.get("versions", []) if item.get("version") == version), None)
    version_dir = REGISTRY_ROOT / version
    if entry is None or not version_dir.is_dir():
        raise ValueError(f"unknown XGBoost model version: {version}")
    artifact_hashes = {item["name"]: item["sha256"] for item in entry.get("artifacts", [])}
    metrics_path = version_dir / "xgb_metrics.json"
    if not metrics_path.is_file():
        raise ValueError("XGBoost version is missing artifact: xgb_metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ready_artifacts = {
        artifact
        for task, artifact in TASK_ARTIFACTS.items()
        if bool((metrics.get("tasks") or {}).get(task, {}).get("ready"))
    }
    route_artifacts = {
        str(route.get("artifact"))
        for task in (metrics.get("tasks") or {}).values()
        for route in (task.get("language_routes") or {}).values()
        if isinstance(route, dict) and route.get("artifact")
    }
    required_files = {"xgb_metrics.json", *ready_artifacts, *route_artifacts}
    for name in required_files:
        source = version_dir / name
        if not source.is_file():
            raise ValueError(f"XGBoost version is missing artifact: {name}")
        expected = artifact_hashes.get(name)
        if expected and _sha256(source) != expected:
            raise ValueError(f"XGBoost artifact hash mismatch: {name}")
    managed_files = set(MODEL_FILES) | set(artifact_hashes)
    for name in managed_files:
        source = version_dir / name
        target = MODEL_DIR / name
        if name in required_files:
            shutil.copy2(source, target)
        elif target.exists():
            # Prevent a partial release from silently loading an artifact left
            # behind by a different model version.
            target.unlink()
    registry["active_version"] = version
    registry["activated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temporary = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, REGISTRY_PATH)
    return registry
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
