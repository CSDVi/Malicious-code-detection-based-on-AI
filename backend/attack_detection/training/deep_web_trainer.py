"""Web-queue adapters for ByteCNN-TCN and GATv2 training.

The Flask runtime intentionally does not depend on PyTorch.  These adapters
prepare the uploaded dataset, run the existing deep trainers in the configured
PyTorch interpreter, retain every candidate, and only switch the runtime
manifest after the candidate passes its release gate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from attack_detection.codet5p_registry import (
    ARTIFACT_ROOT as CODET5P_ARTIFACT_ROOT,
    register_candidate as register_codet5p_candidate,
    resolve_base as resolve_codet5p_base,
)
from attack_detection.training.graph_dataset import build_graph_dataset
from attack_detection.training.artifact_contracts import validate_codet5p_manifest


ProgressCallback = Callable[[float, str], None]
BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BACKEND_DIR / "models"
TEMP_ROOT = BACKEND_DIR / "data" / "tmp" / "deep_training"
CANDIDATE_ROOT = MODEL_DIR / "candidates" / "web_training"
DEFAULT_DEEP_PYTHON = Path(r"D:\software\anaconda\envs\drone\python.exe")


def train_bytetcn(
    dataset_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Export a reviewed code dataset and train a versioned ByteCNN-TCN candidate."""

    python_path = _deep_python()
    source = Path(dataset_path).resolve()
    _progress(progress_callback, 0.06, "正在整理 ByteCNN-TCN 训练集")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bytetcn-", dir=TEMP_ROOT) as temporary:
        run_root = Path(temporary)
        prepared = run_root / "dataset"
        output = run_root / "output"
        _run_module(
            python_path,
            "attack_detection.training.byte_dataset",
            "--dataset", str(source),
            "--output-dir", str(prepared),
        )
        report = _read_json(prepared / "manifest.json")
        missing_splits = [
            split for split in ("train", "validation", "test")
            if int((report.get("records") or {}).get(split) or 0) == 0
        ]
        if missing_splits:
            raise ValueError("ByteCNN-TCN 训练集缺少有效分段：" + "、".join(missing_splits))

        _progress(progress_callback, 0.18, "正在训练 ByteCNN-TCN")
        _run_module(
            python_path,
            "attack_detection.training.byte_tcn_trainer",
            "--dataset-dir", str(prepared),
            "--output-dir", str(output),
        )
        manifest_path = output / "bytetcn_manifest.json"
        manifest = _read_json(manifest_path)
        published = bool(manifest.get("runtime_ready"))
        manifest["quality_gate"] = {
            "passed": published,
            "rule": "至少一个任务/语言组合通过训练器的 F1、误报率和漏报率门禁",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _retain_candidate("bytetcn", output, manifest)
        if published:
            _archive_current("bytetcn")
            _publish_bytetcn(output, manifest)
        _progress(progress_callback, 0.97, "正在登记 ByteCNN-TCN 模型版本")
        return {**manifest, "published": published}


def train_gatv2(
    dataset_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Build or accept project graphs and train a versioned GATv2 candidate."""

    python_path = _deep_python()
    source = Path(dataset_path).resolve()
    _progress(progress_callback, 0.06, "正在整理 GATv2 项目图训练集")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gatv2-", dir=TEMP_ROOT) as temporary:
        run_root = Path(temporary)
        graphs = run_root / "project_graphs.jsonl"
        if _is_graph_jsonl(source):
            shutil.copy2(source, graphs)
        else:
            report = build_graph_dataset(source, graphs, run_root / "graph_report.json")
            if int(report.get("graphs") or 0) == 0:
                raise ValueError(
                    "训练集中没有可构建的项目图；请补充 source、family、version 字段，"
                    "或直接上传包含 nodes、edges、label、split 的图 JSONL。"
                )

        output = run_root / "output"
        _progress(progress_callback, 0.18, "正在训练 GATv2")
        _run_module(
            python_path,
            "attack_detection.training.gat_trainer",
            "--graphs", str(graphs),
            "--output-dir", str(output),
            "--task", "malicious_intent",
        )
        manifest_path = output / "gatv2_manifest.json"
        manifest = _read_json(manifest_path)
        published = bool(manifest.get("runtime_ready")) and _metrics_pass(manifest.get("test_metrics"))
        manifest["quality_gate"] = {
            "passed": published,
            "minimum_precision": 0.9,
            "maximum_false_positive_rate": 0.1,
            "maximum_false_negative_rate": 0.1,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _retain_candidate("gatv2", output, manifest)
        if published:
            _archive_current("gatv2")
            _publish_gatv2(output, manifest)
        _progress(progress_callback, 0.97, "正在登记 GATv2 模型版本")
        return {**manifest, "published": published}


def train_codet5p(
    dataset_path: str | Path,
    progress_callback: ProgressCallback | None = None,
    *,
    base_version: str = "codet5p-220m-base",
    task: str = "vulnerability_risk",
    target_language: str = "all",
) -> dict[str, object]:
    """Fine-tune a registered CodeT5+ base and retain a strict-gated candidate."""

    python_path = _deep_python()
    source = Path(dataset_path).resolve()
    base = resolve_codet5p_base(base_version)
    _progress(progress_callback, 0.06, "正在校验 CodeT5+ 220M 训练任务与基础版本")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="codet5p-", dir=TEMP_ROOT) as temporary:
        output = Path(temporary) / "output"
        arguments = [
            "--dataset", str(source),
            "--output-dir", str(output),
            "--checkpoint", base["checkpoint"],
            "--base-version", base["version"],
            "--task", task,
            "--target-language", target_language,
        ]
        if base["artifact_dir"]:
            arguments.extend(["--base-artifact-dir", base["artifact_dir"]])
        _progress(progress_callback, 0.16, "正在 GPU 微调 CodeT5+ 220M")
        _run_module(
            python_path,
            "attack_detection.training.codet5p_classifier_trainer",
            *arguments,
        )
        manifest_path = output / "codet5p_manifest.json"
        manifest = _read_json(manifest_path)
        errors = validate_codet5p_manifest(
            manifest,
            output,
            require_runtime_ready=False,
        )
        if errors:
            raise RuntimeError("CodeT5+ 220M 训练产物不完整：" + "; ".join(errors))
        published = bool(manifest.get("passed_deployment_gate"))
        candidate_dir = _retain_candidate("codet5p", output, manifest)
        registered_dir = candidate_dir
        if published:
            version = _safe_version(str(manifest["model_version"]))
            registered_dir = CODET5P_ARTIFACT_ROOT / version
            if registered_dir.exists():
                raise RuntimeError(f"CodeT5+ 220M 运行时版本已存在：{version}")
            registered_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output, registered_dir)
        register_codet5p_candidate(
            manifest,
            registered_dir,
            activate=published,
        )
        _progress(progress_callback, 0.97, "正在登记 CodeT5+ 220M 候选版本")
        return {
            **manifest,
            "published": published,
            "candidate_dir": str(candidate_dir),
        }


def is_graph_training_file(path: str | Path) -> bool:
    """Public bounded probe used by the upload validator."""

    return _is_graph_jsonl(Path(path))


def _deep_python() -> Path:
    path = Path(os.environ.get("XIEZHI_DEEP_PYTHON") or DEFAULT_DEEP_PYTHON)
    if not path.is_file():
        raise RuntimeError(f"深度学习解释器不可用：{path}")
    return path


def _run_module(python_path: Path, module: str, *arguments: str) -> None:
    try:
        completed = subprocess.run(
            [str(python_path), "-m", module, *arguments],
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=12 * 60 * 60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{module} 启动失败：{exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "未知错误").strip()[-1500:]
        raise RuntimeError(f"{module} 执行失败：{detail}")


def _is_graph_jsonl(path: Path) -> bool:
    if path.suffix.lower() != ".jsonl":
        return False
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                return isinstance(record, dict) and {"nodes", "edges", "label", "split"}.issubset(record)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return False


def _retain_candidate(family: str, output: Path, manifest: dict[str, Any]) -> Path:
    version = str(manifest.get("model_version") or "").strip()
    if not version:
        raise ValueError("深度模型训练结果缺少 model_version。")
    destination = CANDIDATE_ROOT / family / version
    if destination.exists():
        raise RuntimeError(f"模型版本已存在：{version}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination)
    return destination


def _archive_current(family: str) -> None:
    manifest_name = "bytetcn_manifest.json" if family == "bytetcn" else "gatv2_manifest.json"
    manifest_path = MODEL_DIR / manifest_name
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    version = str(manifest.get("model_version") or "").strip()
    if not version:
        return
    destination = CANDIDATE_ROOT / family / version
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, destination / manifest_name)
    for name in manifest.get("files") or []:
        source = MODEL_DIR / str(name)
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    history_name = "bytetcn_history.json" if family == "bytetcn" else "gatv2_history.json"
    if (MODEL_DIR / history_name).is_file():
        shutil.copy2(MODEL_DIR / history_name, destination / history_name)


def _publish_bytetcn(output: Path, manifest: dict[str, Any]) -> None:
    runtime_manifest = deepcopy(manifest)
    mapping = _copy_versioned_artifacts(output, manifest)
    runtime_manifest["files"] = [mapping[str(name)] for name in manifest.get("files") or []]
    default_file = runtime_manifest["files"][0]
    task_models = deepcopy(runtime_manifest.get("task_models") or {})
    for task, languages in (runtime_manifest.get("task_language_support") or {}).items():
        if not languages:
            continue
        settings = dict(task_models.get(task) or {})
        settings["file"] = mapping.get(str(settings.get("file") or ""), default_file)
        settings.setdefault("config", runtime_manifest.get("config") or {})
        settings.setdefault("line_localization_validated", task == "malicious_intent")
        task_models[task] = settings
    runtime_manifest["task_models"] = task_models
    history = output / "bytetcn_history.json"
    if history.is_file():
        shutil.copy2(history, MODEL_DIR / history.name)
    _atomic_json(MODEL_DIR / "bytetcn_manifest.json", runtime_manifest)


def _publish_gatv2(output: Path, manifest: dict[str, Any]) -> None:
    runtime_manifest = deepcopy(manifest)
    mapping = _copy_versioned_artifacts(output, manifest)
    runtime_manifest["files"] = [mapping[str(name)] for name in manifest.get("files") or []]
    runtime_manifest["artifact"] = runtime_manifest["files"][0]
    history = output / "gatv2_history.json"
    if history.is_file():
        shutil.copy2(history, MODEL_DIR / history.name)
    _atomic_json(MODEL_DIR / "gatv2_manifest.json", runtime_manifest)


def _copy_versioned_artifacts(output: Path, manifest: dict[str, Any]) -> dict[str, str]:
    version = _safe_version(str(manifest["model_version"]))
    mapping: dict[str, str] = {}
    for name in manifest.get("files") or []:
        source = output / str(name)
        if not source.is_file():
            raise FileNotFoundError(f"训练产物不存在：{source.name}")
        runtime_name = f"{source.stem}__{version}{source.suffix}"
        shutil.copy2(source, MODEL_DIR / runtime_name)
        mapping[str(name)] = runtime_name
    if not mapping:
        raise ValueError("训练结果没有可发布的模型文件。")
    return mapping


def _metrics_pass(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        return (
            float(value["precision"]) >= 0.9
            and float(value["false_positive_rate"]) <= 0.1
            and float(value["false_negative_rate"]) <= 0.1
        )
    except (KeyError, TypeError, ValueError):
        return False


def _safe_version(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"训练结果无法读取：{path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"训练结果格式错误：{path.name}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _progress(callback: ProgressCallback | None, value: float, stage: str) -> None:
    if callback:
        callback(value, stage)
