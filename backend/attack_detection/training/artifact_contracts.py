"""Strict metadata contracts for externally trained deep-model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_bytetcn_manifest(manifest: dict[str, Any], artifact_dir: str | Path) -> list[str]:
    errors = _required(manifest, "model_version", "dataset_sha256", "supported_languages", "thresholds", "files")
    outputs = set(manifest.get("output_heads", []))
    required_outputs = {"malicious_intent", "vulnerability_risk", "behavior_labels", "line_localization"}
    if not required_outputs.issubset(outputs):
        errors.append("ByteCNN-TCN manifest is missing one or more required output heads")
    errors.extend(_missing_files(manifest, artifact_dir))
    if manifest.get("calibrated") is not True:
        errors.append("ByteCNN-TCN probabilities are not marked as calibrated")
    if manifest.get("runtime_ready") is not True:
        errors.append("ByteCNN-TCN artifact is not marked runtime-ready")
    return errors


def validate_mamba_manifest(manifest: dict[str, Any], artifact_dir: str | Path) -> list[str]:
    """Legacy alias retained only for old artifact inspection."""
    return validate_bytetcn_manifest(manifest, artifact_dir)


def validate_gat_manifest(manifest: dict[str, Any], artifact_dir: str | Path) -> list[str]:
    errors = _required(manifest, "model_version", "dataset_sha256", "node_types", "edge_types", "files")
    if not {"file", "function", "package", "dangerous_api"}.issubset(set(manifest.get("node_types", []))):
        errors.append("GATv2 manifest has an incomplete node schema")
    if not {"call", "import", "dependency", "version_diff"}.issubset(set(manifest.get("edge_types", []))):
        errors.append("GATv2 manifest has an incomplete edge schema")
    errors.extend(_missing_files(manifest, artifact_dir))
    if manifest.get("calibrated") is not True:
        errors.append("GATv2 probability is not marked as calibrated")
    if manifest.get("runtime_ready") is not True:
        errors.append("GATv2 artifact is not marked runtime-ready")
    return errors


def validate_codet5p_manifest(
    manifest: dict[str, Any],
    artifact_dir: str | Path,
    *,
    require_runtime_ready: bool = True,
) -> list[str]:
    errors = _required(
        manifest,
        "model_version",
        "dataset_sha256",
        "task",
        "supported_languages",
        "threshold",
        "files",
    )
    if manifest.get("task") not in {"vulnerability_risk", "malicious_intent"}:
        errors.append("CodeT5+ 220M manifest has an unsupported task")
    errors.extend(_missing_files(manifest, artifact_dir))
    if manifest.get("calibrated") is not True:
        errors.append("CodeT5+ 220M probabilities are not marked as calibrated")
    if require_runtime_ready and manifest.get("runtime_ready") is not True:
        errors.append("CodeT5+ 220M artifact is not marked runtime-ready")
    return errors


def _required(value: dict[str, Any], *names: str) -> list[str]:
    return [f"manifest field is missing: {name}" for name in names if not value.get(name)]


def _missing_files(manifest: dict[str, Any], artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    return [f"artifact file is missing: {name}" for name in manifest.get("files", []) if not (root / str(name)).is_file()]
