"""Build a minimal CodeT5+ malicious-intent training handoff package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset  # noqa: E402
from attack_detection.training.codet5p_classifier_trainer import (  # noqa: E402
    TASKS,
    _select_languages,
    _validate_family_isolation,
    _validate_partitions,
)


TASK = "malicious_intent"
POSITIVE = TASKS[TASK]["positive"]
NEGATIVE = TASKS[TASK]["negative"]
SOURCE_DATASET = BACKEND / "data" / "processed" / "xgb_multilingual_malicious_20260728_v66.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "codet5p_malicious_training_handoff_20260728"
TARGET_LANGUAGES = ("python", "php", "go", "c", "cpp", "bash", "powershell")
PROJECT_FILES = (
    "backend/attack_detection/__init__.py",
    "backend/attack_detection/dataset.py",
    "backend/attack_detection/training/__init__.py",
    "backend/attack_detection/training/codet5p_classifier_trainer.py",
    "backend/attack_detection/training/codet5p_model.py",
)


def build(output_root: Path, source_dataset: Path = SOURCE_DATASET) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    data_root = output_root / "training_data"
    project_root = output_root / "training_only_project"
    data_root.mkdir(parents=True)
    project_root.mkdir(parents=True)

    dataset_reports, jobs = _write_language_shards(source_dataset, data_root)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Minimal CLI handoff for CodeT5+ malicious-intent per-language training",
        "source_dataset": str(source_dataset.relative_to(ROOT)),
        "source_sha256": _sha256(source_dataset),
        "task": TASK,
        "checkpoint": "Salesforce/codet5p-220m",
        "base_version": "codet5p-220m-base",
        "target_languages": list(TARGET_LANGUAGES),
        "jobs": jobs,
        "datasets": dataset_reports,
        "return_contract": [
            "Xiezhi_CodeT5_Return_Results.zip",
            "or the full outputs/ directory if zipping fails",
        ],
        "notes": [
            "Java and JavaScript are already active in the main project and are intentionally not included.",
            "The package does not include Salesforce/codet5p-220m weights; the trainer downloads the checkpoint on first run.",
            "Datasets contain source code samples only; do not execute the samples.",
        ],
    }
    _write_json(data_root / "codet5p_malicious_handoff_manifest.json", manifest)
    _write_json(data_root / "codet5p_malicious_training_jobs.json", jobs)

    _build_project(project_root, manifest)

    project_zip = output_root / "Xiezhi_CodeT5_Minimal_Training_Project.zip"
    data_zip = output_root / "Xiezhi_CodeT5_Malicious_Training_Data_7_Languages.zip"
    _zip_directory(project_zip, project_root, project_root)
    _zip_directory(data_zip, data_root, output_root)

    checksums = {
        project_zip.name: _sha256(project_zip),
        data_zip.name: _sha256(data_zip),
    }
    _write_json(output_root / "SHA256SUMS.json", checksums)
    _verify_archives(output_root, checksums)

    return {
        "output_root": str(output_root),
        "project_zip": _file_report(project_zip, checksums[project_zip.name]),
        "training_data_zip": _file_report(data_zip, checksums[data_zip.name]),
        "jobs": jobs,
    }


def _write_language_shards(source_dataset: Path, data_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = {
        language: data_root / f"codet5p_malicious_intent_{language}_train.jsonl"
        for language in TARGET_LANGUAGES
    }
    writers = {language: path.open("w", encoding="utf-8", newline="\n") for language, path in paths.items()}
    try:
        with source_dataset.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {source_dataset} line {line_number}") from exc
                language = str(record.get("language") or "").strip().lower()
                label = str(record.get("label") or "").strip().lower()
                if language not in writers or label not in {NEGATIVE, POSITIVE}:
                    continue
                record["language"] = language
                record["label"] = label
                writers[language].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        for writer in writers.values():
            writer.close()

    reports = {}
    jobs = []
    for order, language in enumerate(TARGET_LANGUAGES, 1):
        report = _validate_shard(paths[language], language)
        reports[paths[language].name] = report
        jobs.append({
            "order": order,
            "task": TASK,
            "target_language": language,
            "file": paths[language].name,
            "records": report["records"],
            "bytes": report["bytes"],
            "sha256": report["sha256"],
            "base_version": "codet5p-220m-base",
            "checkpoint": "Salesforce/codet5p-220m",
        })
    return reports, jobs


def _validate_shard(path: Path, language: str) -> dict[str, Any]:
    rows = [
        sample
        for sample in load_dataset(path)
        if sample.language == language
        and sample.label in {NEGATIVE, POSITIVE}
        and is_task_training_eligible(sample, TASK)
    ]
    supported = _select_languages(rows, language, POSITIVE, NEGATIVE)
    selected = [sample for sample in rows if sample.language in supported]
    partitions = {
        split: [sample for sample in selected if sample.split == split]
        for split in ("train", "validation", "test")
    }
    _validate_partitions(partitions, POSITIVE, NEGATIVE)
    _validate_family_isolation(selected)

    labels = Counter(sample.label for sample in selected)
    splits = Counter(sample.split for sample in selected)
    split_labels = Counter((sample.split, sample.label) for sample in selected)
    return {
        "language": language,
        "records": len(selected),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "labels": dict(sorted(labels.items())),
        "splits": dict(sorted(splits.items())),
        "split_labels": {
            f"{split}:{label}": count
            for (split, label), count in sorted(split_labels.items())
        },
        "family_isolation_verified": True,
        "upload_ready": True,
    }


def _build_project(project_root: Path, manifest: dict[str, Any]) -> None:
    for relative in PROJECT_FILES:
        source = ROOT / relative
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (project_root / "requirements-codet5p-train.txt").write_text(_requirements(), encoding="utf-8")
    (project_root / "run_training_jobs.py").write_text(_runner(), encoding="utf-8")
    (project_root / "README_TRAINING_ONLY.md").write_text(_readme(manifest), encoding="utf-8")
    _write_json(project_root / "included_jobs_preview.json", manifest["jobs"])


def _requirements() -> str:
    return """# Install a CUDA-enabled PyTorch build separately before this file.
# Example: pip install torch --index-url https://download.pytorch.org/whl/cu121
transformers==4.46.3
huggingface-hub==0.25.2
safetensors>=0.4
tokenizers==0.20.3
"""


def _runner() -> str:
    return r'''"""Run the packaged CodeT5+ malicious-intent training jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run CodeT5+ per-language malicious-intent training jobs")
    parser.add_argument("--data-root", type=Path, default=root / "training_data")
    parser.add_argument("--jobs-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    parser.add_argument("--languages", default="", help="Comma-separated subset, for example: bash,powershell")
    parser.add_argument("--checkpoint", default="Salesforce/codet5p-220m")
    parser.add_argument("--base-version", default="codet5p-220m-base")
    parser.add_argument("--base-artifact-dir", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path, default=root / "model_cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    jobs_file = args.jobs_file or args.data_root / "codet5p_malicious_training_jobs.json"
    jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
    selected = {item.strip().lower() for item in args.languages.split(",") if item.strip()}
    if selected:
        jobs = [job for job in jobs if str(job["target_language"]).lower() in selected]
    if not jobs:
        raise SystemExit("No training jobs matched the requested language filter.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    backend = str(root / "backend")
    env["PYTHONPATH"] = backend + os.pathsep + env.get("PYTHONPATH", "")

    summaries = []
    for job in jobs:
        language = str(job["target_language"])
        dataset = args.data_root / str(job["file"])
        output_dir = args.output_root / f"{int(job['order']):02d}-{language}"
        manifest_path = output_dir / "codet5p_manifest.json"
        if manifest_path.is_file() and not args.overwrite:
            print(f"[skip] {language}: existing manifest found at {manifest_path}")
            summaries.append(_summary(manifest_path, skipped=True))
            continue
        if output_dir.exists() and args.overwrite:
            shutil.rmtree(output_dir)
        elif output_dir.exists() and not manifest_path.is_file():
            raise SystemExit(f"Output directory exists but has no manifest: {output_dir}. Use --overwrite or remove it.")
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "attack_detection.training.codet5p_classifier_trainer",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            args.checkpoint,
            "--base-version",
            args.base_version,
            "--task",
            "malicious_intent",
            "--target-language",
            language,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--cache-dir",
            str(args.cache_dir),
        ]
        if args.base_artifact_dir:
            cmd.extend(["--base-artifact-dir", args.base_artifact_dir])
        print(f"[train] {language}: {dataset.name}")
        subprocess.run(cmd, cwd=root, env=env, check=True)
        summaries.append(_summary(manifest_path, skipped=False))

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jobs": summaries,
    }
    summary_path = args.output_root / "training_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary: {summary_path}")
    if not args.no_zip:
        archive = root / "Xiezhi_CodeT5_Return_Results.zip"
        _zip_outputs(archive, args.output_root)
        print(f"[return] send this file back: {archive}")


def _summary(manifest_path: Path, *, skipped: bool) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = manifest.get("test_metrics") or {}
    return {
        "skipped_existing": skipped,
        "language": (manifest.get("supported_languages") or [""])[0],
        "model_version": manifest.get("model_version"),
        "output_dir": str(manifest_path.parent),
        "passed_deployment_gate": manifest.get("passed_deployment_gate"),
        "runtime_ready": manifest.get("runtime_ready"),
        "samples": metrics.get("samples"),
        "precision": metrics.get("precision"),
        "false_positive_rate": metrics.get("false_positive_rate"),
        "false_negative_rate": metrics.get("false_negative_rate"),
    }


def _zip_outputs(destination: Path, output_root: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent).as_posix())


if __name__ == "__main__":
    main()
'''


def _readme(manifest: dict[str, Any]) -> str:
    rows = [
        f"| {job['order']} | `{job['target_language']}` | `{job['file']}` | {job['records']} |"
        for job in manifest["jobs"]
    ]
    table = "\n".join(rows)
    return f"""# CodeT5+ 最小训练交接包

这个包只保留命令行训练入口，没有 Web 前端。请不要执行训练数据里的任何源码样本，只让训练脚本读取它们。

## 解压

把这两个压缩包解压到同一个目录：

- `Xiezhi_CodeT5_Minimal_Training_Project.zip`
- `Xiezhi_CodeT5_Malicious_Training_Data_7_Languages.zip`

解压后目录里应当能看到：

```text
backend/
training_data/
run_training_jobs.py
requirements-codet5p-train.txt
```

## 环境

先安装和显卡匹配的 CUDA 版 PyTorch，再安装其余依赖。Windows 示例：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-codet5p-train.txt
python -c "import torch; print(torch.cuda.is_available())"
```

如果最后一行输出 `False`，说明当前环境没用上 NVIDIA CUDA，训练会非常慢。

## 训练

建议先跑一个语言测通：

```powershell
python run_training_jobs.py --languages bash
```

全量顺序训练 7 个语言：

```powershell
python run_training_jobs.py
```

默认参数是 `epochs=5`、`batch_size=4`、`max_length=512`，和主项目 CodeT5+ 训练器一致。首次运行会下载 `Salesforce/codet5p-220m`。

## 任务清单

| 顺序 | 语言 | 数据文件 | 记录数 |
|---:|---|---|---:|
{table}

## 返回给我的东西

训练完成后，把根目录下自动生成的这个文件发回来：

```text
Xiezhi_CodeT5_Return_Results.zip
```

如果压缩失败，就把完整的 `outputs/` 目录发回来。不能只发 `.safetensors`，因为 tokenizer、config、manifest、阈值和校准信息都要一起保留。

## 上线标准

每个语言会独立生成一个候选目录。只有该语言的测试指标同时满足 Precision >= 90%、FPR <= 10%、FNR <= 10%，才算达到了当前主项目的严格上线门槛。
"""


def _zip_directory(destination: Path, source_root: Path, archive_root: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(archive_root).as_posix())


def _verify_archives(output_root: Path, checksums: dict[str, str]) -> None:
    for name, expected in checksums.items():
        archive_path = output_root / name
        if _sha256(archive_path) != expected:
            raise RuntimeError(f"archive checksum mismatch: {name}")
        with zipfile.ZipFile(archive_path) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"bad ZIP member in {name}: {bad}")
            for member in archive.namelist():
                if _unsafe_archive_name(member):
                    raise RuntimeError(f"unsafe ZIP member in {name}: {member}")


def _unsafe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0])


def _file_report(path: Path, sha256: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    args = parser.parse_args()
    result = build(args.output.resolve(), args.source_dataset.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
