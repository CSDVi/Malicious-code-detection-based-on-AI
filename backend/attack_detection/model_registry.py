"""Versioned model registry with explicit activation and rollback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from .codet5p_registry import ARTIFACT_ROOT as CODET5P_ARTIFACT_ROOT
from .codet5p_registry import registry_view as codet5p_registry_view
from .engines.codet5p_engine import configured_deep_python
from .engines.gat_engine import configured_deep_python as configured_gat_python
from .training.artifact_contracts import (
    validate_codet5p_manifest,
    validate_gat_manifest,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REGISTRY_ROOT = MODEL_DIR / "registry"
REGISTRY_PATH = MODEL_DIR / "registry.json"
MODEL_FILES = (
    "malicious_vectorizer.joblib",
    "malicious_classifier.joblib",
    "malicious_calibrator.joblib",
    "vulnerability_vectorizer.joblib",
    "vulnerability_classifier.joblib",
    "vulnerability_calibrator.joblib",
    "metrics.json",
)


_HEALTH_PROBE_SUCCESS_TTL_SECONDS = 300.0
_HEALTH_PROBE_FAILURE_TTL_SECONDS = 2.0
_HEALTH_PROBE_CACHE_LIMIT = 32
_health_probe_cache: dict[
    tuple[object, ...],
    tuple[float, tuple[bool, str]],
] = {}
_health_probe_lock = threading.Lock()


def _cached_health_probe(
    key: tuple[object, ...],
    probe: Callable[[], tuple[bool, str]],
) -> tuple[bool, str]:
    """Cache healthy probes longer while allowing transient failures to recover."""

    with _health_probe_lock:
        now = monotonic()
        cached = _health_probe_cache.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]

        result = probe()
        ttl = (
            _HEALTH_PROBE_SUCCESS_TTL_SECONDS
            if result[0]
            else _HEALTH_PROBE_FAILURE_TTL_SECONDS
        )
        _health_probe_cache[key] = (monotonic() + ttl, result)
        if len(_health_probe_cache) > _HEALTH_PROBE_CACHE_LIMIT:
            oldest_key = min(
                _health_probe_cache,
                key=lambda item: _health_probe_cache[item][0],
            )
            _health_probe_cache.pop(oldest_key, None)
        return result


def _clear_runtime_probe_cache() -> None:
    """Clear runtime health state after activation or in focused tests."""

    with _health_probe_lock:
        _health_probe_cache.clear()


def _probe_python_modules(
    python_path_text: str,
    modules: tuple[str, ...],
) -> tuple[bool, str]:
    """Verify that a configured subprocess interpreter can import dependencies."""

    def run_probe() -> tuple[bool, str]:
        python_path = Path(python_path_text)
        if not python_path.is_file():
            return False, f"解释器不可用：{python_path}"
        script = (
            "import importlib.util,sys;"
            f"missing=[name for name in {modules!r} if importlib.util.find_spec(name) is None];"
            "print(','.join(missing));"
            "sys.exit(1 if missing else 0)"
        )
        try:
            completed = subprocess.run(
                [str(python_path), "-c", script],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"解释器依赖检查失败：{exc}"
        if completed.returncode != 0:
            missing = (completed.stdout or completed.stderr or "未知依赖").strip()
            return False, f"缺少推理依赖：{missing}"
        return True, "推理解释器及依赖可用"

    return _cached_health_probe(
        ("python_modules", python_path_text, modules),
        run_probe,
    )


def _probe_xgboost_runtime(
    model_version: str,
    artifact_mtime_ns: int,
) -> tuple[bool, str]:
    """Load the active XGBoost route and require a real probability."""

    def run_probe() -> tuple[bool, str]:
        try:
            from .engines.xgb_engine import XGBoostEngine

            result = next(
                (
                    item
                    for item in XGBoostEngine().scan(
                        "print('xiezhi runtime health check')",
                        "python",
                        generate_line_attributions=False,
                    )
                    if item.get("name") == "xgboost_malicious"
                ),
                {},
            )
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            return False, f"最小推理失败：{exc}"
        if result.get("status") != "completed" or result.get("probability") is None:
            return False, str(result.get("reason") or "最小推理没有产生概率")
        return True, "活动语言路由最小推理通过"

    return _cached_health_probe(
        ("xgboost", model_version, artifact_mtime_ns),
        run_probe,
    )


def _read_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": 1, "active_version": "", "versions": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def make_version_id(dataset_hash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{dataset_hash[:10]}"


def create_version_dir(version: str) -> Path:
    path = REGISTRY_ROOT / version
    path.mkdir(parents=True, exist_ok=False)
    return path


def register_version(
    version: str, metrics: dict[str, Any], dataset_hash: str, *, activate: bool = False,
) -> dict[str, Any]:
    version_dir = REGISTRY_ROOT / version
    artifacts = []
    for path in sorted(version_dir.glob("*")):
        if path.is_file():
            artifacts.append({
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    entry = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": dataset_hash,
        "dataset": metrics.get("dataset"),
        "samples": metrics.get("samples", 0),
        "label_counts": metrics.get("label_counts", {}),
        "tasks": {
            name: {
                "ready": task.get("ready", False),
                "f1": task.get("f1"),
                "pr_auc": task.get("pr_auc"),
                "false_positive_rate": task.get("false_positive_rate"),
                "false_negative_rate": task.get("false_negative_rate"),
                "deployment": task.get("deployment", {}),
                "quality_gate_passed": task.get("quality_gate_passed", False),
                "thresholds": task.get("thresholds", {}),
                "calibrated": task.get("calibrated", False),
            }
            for name, task in metrics.get("tasks", {}).items()
        },
        "artifacts": artifacts,
        "published": activate,
        "quality_gate": metrics.get("quality_gate", {}),
    }
    registry = _read_registry()
    registry["versions"] = [item for item in registry.get("versions", []) if item.get("version") != version]
    registry["versions"].insert(0, entry)
    if activate:
        registry["active_version"] = version
    _atomic_json(REGISTRY_PATH, registry)
    if activate:
        activate_version(version)
    return entry


def activate_version(version: str) -> dict[str, Any]:
    registry = _read_registry()
    version_dir = REGISTRY_ROOT / version
    if not version_dir.is_dir() or not any(item.get("version") == version for item in registry.get("versions", [])):
        raise ValueError(f"unknown model version: {version}")
    for name in MODEL_FILES:
        source = version_dir / name
        target = MODEL_DIR / name
        if source.exists():
            shutil.copy2(source, target)
        elif target.exists() and name.endswith("_calibrator.joblib"):
            target.unlink()
    registry["active_version"] = version
    registry["activated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_json(REGISTRY_PATH, registry)
    return registry
def active_model_dir() -> tuple[Path, str]:
    registry = _read_registry()
    version = str(registry.get("active_version") or "")
    path = REGISTRY_ROOT / version
    if version and path.is_dir():
        return path, version
    return MODEL_DIR, "legacy"


def registry_view() -> dict[str, Any]:
    return _read_registry()


def runtime_status() -> list[dict[str, Any]]:
    registry = _read_registry()
    active_version = str(registry.get("active_version") or "")
    active_entry = next((item for item in registry.get("versions", []) if item.get("version") == active_version), {})
    legacy_ready = any(task.get("ready") for task in active_entry.get("tasks", {}).values())
    xgb_manifest = _artifact_manifest("xgb_metrics.json")
    xgb_ready = bool(xgb_manifest) and any(
        bool(task.get("ready"))
        and bool(task.get("quality_gate_passed"))
        and (MODEL_DIR / str(task.get("artifact") or "")).is_file()
        for task in (xgb_manifest.get("tasks") or {}).values()
        if isinstance(task, dict)
    )
    xgb_artifact = MODEL_DIR / "xgb_malicious_python.joblib"
    xgb_runtime_ready, xgb_runtime_reason = _probe_xgboost_runtime(
        str(xgb_manifest.get("model_version") or ""),
        xgb_artifact.stat().st_mtime_ns if xgb_artifact.is_file() else 0,
    )
    xgb_ready = xgb_ready and xgb_runtime_ready
    gat_manifest = _artifact_manifest("gatv2_manifest.json")
    gat_artifact_ready = bool(gat_manifest) and not validate_gat_manifest(gat_manifest, MODEL_DIR)
    gat_python = configured_gat_python()
    gat_runtime_ready, gat_runtime_reason = _probe_python_modules(
        str(gat_python),
        ("torch", "torch_geometric"),
    )
    gat_ready = gat_artifact_ready and gat_runtime_ready
    codet5p_registry = codet5p_registry_view()
    codet5p_active_versions = {
        str(version)
        for task_routes in (codet5p_registry.get("active_routes") or {}).values()
        if isinstance(task_routes, dict)
        for version in task_routes.values()
        if version
    }
    codet5p_artifact_ready = any(
        not validate_codet5p_manifest(
            _read_json_file(CODET5P_ARTIFACT_ROOT / version / "codet5p_manifest.json"),
            CODET5P_ARTIFACT_ROOT / version,
        )
        for version in codet5p_active_versions
        if (CODET5P_ARTIFACT_ROOT / version / "codet5p_manifest.json").is_file()
    )
    codet5p_python = configured_deep_python()
    codet5p_runtime_ready, codet5p_runtime_reason = _probe_python_modules(
        str(codet5p_python),
        ("torch", "safetensors", "transformers"),
    )
    codet5p_ready = codet5p_artifact_ready and codet5p_runtime_ready
    if not codet5p_artifact_ready:
        codet5p_reason = "基础版本已注册，尚无通过质量门禁的运行时候选"
    elif not codet5p_runtime_ready:
        codet5p_reason = codet5p_runtime_reason
    else:
        codet5p_reason = "GPU 训练通过质量门禁的语义模型及推理解释器均可用"
    return [
        {
            "name": "TF-IDF / SVM 对照组", "engine": "legacy_svm",
            "status": "completed" if legacy_ready else "unavailable", "version": active_version or None,
            "reason": "当前注册版本已加载" if legacy_ready else "没有可用的注册模型版本",
        },
        {
            "name": "XGBoost", "engine": "xgboost", "status": "completed" if xgb_ready else "unavailable",
            "version": _artifact_version("xgb_metrics.json"),
            "reason": (
                "至少一个通过严格门禁的任务/语言路由已加载，且最小推理通过"
                if xgb_ready else xgb_runtime_reason
            ),
        },
        {
            "name": "GATv2", "engine": "gatv2", "status": "completed" if gat_ready else "unavailable",
            "version": _artifact_version("gatv2_manifest.json"),
            "reason": (
                "推理产物、解释器及依赖均可用"
                if gat_ready
                else (
                    gat_runtime_reason
                    if gat_artifact_ready
                    else "缺少或无法验证 GATv2 推理产物"
                )
            ),
        },
        {
            "name": "CodeT5+ 220M",
            "engine": "codet5p",
            "status": "completed" if codet5p_ready else "unavailable",
            "version": ", ".join(sorted(codet5p_active_versions)) or None,
            "reason": codet5p_reason,
        },
    ]


def _artifact_version(metadata_name: str) -> str | None:
    manifest = _artifact_manifest(metadata_name)
    return str(manifest.get("model_version") or "") or None


def _artifact_manifest(metadata_name: str) -> dict[str, Any]:
    path = MODEL_DIR / metadata_name
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate or roll back a trained model version")
    parser.add_argument("--activate", help="version identifier to activate")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.activate:
        print(json.dumps(activate_version(args.activate), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(registry_view(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
