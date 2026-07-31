"""Registry for trainable CodeT5+ bases and fine-tuned candidates."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REGISTRY_PATH = MODEL_DIR / "codet5p_registry.json"
ARTIFACT_ROOT = MODEL_DIR / "codet5p_artifacts"
PRETRAINED_ROOT = MODEL_DIR / "pretrained"
DEFAULT_BASE_VERSION = "codet5p-220m-base"
DEFAULT_CHECKPOINT = "Salesforce/codet5p-220m"
SUPPORTED_LANGUAGES = [
    "bash",
    "c",
    "cpp",
    "csharp",
    "go",
    "java",
    "javascript",
    "php",
    "powershell",
    "python",
    "ruby",
]
SUPPORTED_TASKS = ["vulnerability_risk", "malicious_intent"]


def default_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_family": "codet5p",
        "display_name": "CodeT5+ 220M",
        "active_routes": {},
        "versions": [{
            "version": DEFAULT_BASE_VERSION,
            "kind": "pretrained_base",
            "checkpoint": DEFAULT_CHECKPOINT,
            "local_checkpoint_dir": "pretrained/codet5p-220m",
            "created_at": "2026-07-23T00:00:00+00:00",
            "trainable": True,
            "published": False,
            "quality_gate_passed": None,
            "supported_languages": SUPPORTED_LANGUAGES,
            "supported_tasks": SUPPORTED_TASKS,
            "license": "BSD-3-Clause",
            "source_url": "https://huggingface.co/Salesforce/codet5p-220m",
        }],
    }


def registry_view() -> dict[str, Any]:
    registry = _read_registry()
    changed = False
    if not any(item.get("version") == DEFAULT_BASE_VERSION for item in registry.get("versions", [])):
        registry.setdefault("versions", []).append(default_registry()["versions"][0])
        changed = True
    if changed:
        _atomic_json(REGISTRY_PATH, registry)
    return registry


def resolve_base(version: str) -> dict[str, str]:
    entry = next(
        (item for item in registry_view().get("versions", []) if str(item.get("version")) == version),
        None,
    )
    if not entry or entry.get("trainable") is not True:
        raise ValueError(f"unknown or non-trainable CodeT5+ 220M base version: {version}")
    checkpoint = str(entry.get("checkpoint") or DEFAULT_CHECKPOINT)
    local_checkpoint_dir = str(entry.get("local_checkpoint_dir") or "")
    if local_checkpoint_dir:
        local_checkpoint = (MODEL_DIR / local_checkpoint_dir).resolve()
        if (local_checkpoint / "config.json").is_file():
            checkpoint = str(local_checkpoint)
    artifact_dir = str(entry.get("artifact_dir") or "")
    if artifact_dir:
        resolved = (MODEL_DIR / artifact_dir).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"CodeT5+ 220M base artifact directory is missing: {resolved}")
        artifact_dir = str(resolved)
    return {
        "version": str(entry["version"]),
        "checkpoint": checkpoint,
        "artifact_dir": artifact_dir,
    }


def register_candidate(
    manifest: dict[str, Any],
    artifact_dir: str | Path,
    *,
    activate: bool = False,
) -> dict[str, Any]:
    version = str(manifest.get("model_version") or "").strip()
    if not version:
        raise ValueError("CodeT5+ 220M candidate manifest is missing model_version")
    source = Path(artifact_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"CodeT5+ 220M candidate directory is missing: {source}")
    try:
        relative_artifact = source.relative_to(MODEL_DIR.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("CodeT5+ 220M candidate must be stored below backend/models") from exc

    registry = registry_view()
    entry = {
        "version": version,
        "kind": "fine_tuned_candidate",
        "checkpoint": str(manifest.get("checkpoint") or DEFAULT_CHECKPOINT),
        "base_version": str(manifest.get("base_version") or DEFAULT_BASE_VERSION),
        "artifact_dir": relative_artifact,
        "created_at": str(manifest.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
        "trainable": True,
        "published": bool(activate),
        "quality_gate_passed": bool(manifest.get("passed_deployment_gate")),
        "supported_languages": list(manifest.get("supported_languages") or []),
        "supported_tasks": [str(manifest.get("task") or "")],
        "task": str(manifest.get("task") or ""),
        "threshold": manifest.get("threshold"),
        "test_metrics": deepcopy(manifest.get("test_metrics") or {}),
        "dataset_sha256": str(manifest.get("dataset_sha256") or ""),
    }
    registry["versions"] = [
        item for item in registry.get("versions", []) if str(item.get("version")) != version
    ]
    registry["versions"].insert(0, entry)
    if activate:
        task = str(manifest.get("task") or "")
        routes = registry.setdefault("active_routes", {}).setdefault(task, {})
        for language in manifest.get("supported_languages") or []:
            routes[str(language)] = version
        registry["activated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_json(REGISTRY_PATH, registry)
    return entry


def _read_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.is_file():
        registry = default_registry()
        _atomic_json(REGISTRY_PATH, registry)
        return registry
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"CodeT5+ 220M registry cannot be read: {REGISTRY_PATH}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("CodeT5+ 220M registry must contain a JSON object")
    value.setdefault("schema_version", 1)
    value.setdefault("model_family", "codet5p")
    value.setdefault("display_name", "CodeT5+ 220M")
    value.setdefault("active_routes", {})
    value.setdefault("versions", [])
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
