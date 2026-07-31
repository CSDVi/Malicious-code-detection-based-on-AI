"""Shard returned CodeT5+ datasets for independent per-language continuation training."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_codet5p_handoff import (  # noqa: E402
    MAX_UPLOAD_BYTES,
    SPLITS,
    _build_project_zip,
    _sha256,
    _write_json,
)


DEFAULT_SOURCE_ROOT = ROOT / "artifacts" / "codet5p_handoff_20260723" / "training_data"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "codet5p_language_retraining_20260724"
REGISTRY_PATH = ROOT / "backend" / "models" / "codet5p_registry.json"
TASK_SOURCES = {
    "vulnerability_risk": {
        "file": "codet5p_vulnerability_all_ready.jsonl",
        "positive": "vulnerable",
        "languages": ["c", "cpp", "csharp", "go", "java", "javascript", "php", "python", "ruby"],
    },
    "malicious_intent": {
        "file": "codet5p_malicious_java_js_php_python_ready.jsonl",
        "positive": "malicious",
        "languages": ["java", "php", "python"],
    },
}


def build(source_root: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    data_root = output_root / "training_data"
    data_root.mkdir(parents=True)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    base_versions = {
        task: _continuation_base(registry, task)
        for task in TASK_SOURCES
    }

    jobs = []
    dataset_reports = {}
    for task, settings in TASK_SOURCES.items():
        source = source_root / str(settings["file"])
        writers: dict[str, Any] = {}
        paths = {
            language: data_root / f"codet5p_{task}_{language}_continue.jsonl"
            for language in settings["languages"]
        }
        try:
            for language, path in paths.items():
                writers[language] = path.open("w", encoding="utf-8", newline="\n")
            with source.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    language = str(record.get("language") or "")
                    writer = writers.get(language)
                    if writer is not None:
                        writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        finally:
            for writer in writers.values():
                writer.close()

        for language, path in paths.items():
            report = _validate_shard(path, str(settings["positive"]), language)
            dataset_reports[path.name] = report
            jobs.append({
                "order": len(jobs) + 1,
                "model_version": base_versions[task],
                "task": task,
                "target_language": language,
                "file": path.name,
                "records": report["records"],
                "bytes": report["bytes"],
                "sha256": report["sha256"],
            })

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Independent continuation training for CodeT5+ routes that failed the multilingual gate",
        "registry_sha256": _sha256(REGISTRY_PATH),
        "continuation_bases": base_versions,
        "jobs": jobs,
        "datasets": dataset_reports,
        "already_published": {
            "task": "malicious_intent",
            "language": "javascript",
            "version": ((registry.get("active_routes") or {}).get("malicious_intent") or {}).get("javascript"),
        },
    }
    manifest_path = data_root / "language_retraining_manifest.json"
    _write_json(manifest_path, manifest)
    guide_path = data_root / "README_LANGUAGE_RETRAINING.md"
    guide_path.write_text(_guide_clean(manifest), encoding="utf-8")

    data_zip = output_root / "Xiezhi_CodeT5_Language_Retraining_Data.zip"
    with zipfile.ZipFile(data_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(data_root.iterdir()):
            if path.is_file():
                archive.write(path, path.name)

    project_zip = output_root / "Xiezhi_CodeT5_Language_Retraining_Project.zip"
    _build_project_zip(project_zip, guide_path)
    checksums = {
        project_zip.name: _sha256(project_zip),
        data_zip.name: _sha256(data_zip),
    }
    _write_json(output_root / "SHA256SUMS.json", checksums)
    return {
        "output_root": str(output_root),
        "project_zip": {
            "path": str(project_zip),
            "bytes": project_zip.stat().st_size,
            "sha256": checksums[project_zip.name],
        },
        "training_data_zip": {
            "path": str(data_zip),
            "bytes": data_zip.stat().st_size,
            "sha256": checksums[data_zip.name],
        },
        "jobs": jobs,
        "already_published": manifest["already_published"],
    }


def _continuation_base(registry: dict[str, Any], task: str) -> str:
    candidates = [
        entry
        for entry in registry.get("versions", [])
        if entry.get("kind") == "fine_tuned_candidate"
        and entry.get("task") == task
        and entry.get("published") is False
        and entry.get("artifact_dir")
    ]
    candidates.sort(key=lambda entry: str(entry.get("created_at") or ""), reverse=True)
    for entry in candidates:
        artifact = ROOT / "backend" / "models" / str(entry["artifact_dir"])
        if (artifact / "codet5p_classifier.safetensors").is_file():
            return str(entry["version"])
    raise RuntimeError(f"no local continuation candidate is available for task: {task}")


def _validate_shard(path: Path, positive: str, language: str) -> dict[str, Any]:
    labels = Counter()
    splits = Counter()
    matrix = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    records = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            if str(record.get("language") or "") != language:
                raise RuntimeError(f"{path.name}:{line_number} has the wrong language")
            label = str(record.get("label") or "")
            split = str(record.get("split") or "")
            labels[label] += 1
            splits[split] += 1
            matrix[(split, label)] += 1
            family = str(record.get("family") or "")
            if family:
                families[family].add(split)
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError(f"{path.name} exceeds the web upload limit")
    for split in SPLITS:
        if matrix[(split, "benign")] <= 0 or matrix[(split, positive)] <= 0:
            raise RuntimeError(f"{path.name} is missing a class in split {split}")
    overlap = [family for family, values in families.items() if len(values) > 1]
    if overlap:
        raise RuntimeError(f"{path.name} has family leakage: {overlap[:5]}")
    return {
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "language": language,
        "labels": dict(sorted(labels.items())),
        "splits": dict(sorted(splits.items())),
        "family_isolation_verified": True,
        "upload_ready": True,
    }


def _guide(manifest: dict[str, Any]) -> str:
    lines = [
        "# CodeT5+ 分语言续训交接",
        "",
        "JavaScript 恶意意图路线已经通过严格门禁并正式发布，不要重复训练。",
        "",
        "其余任务必须逐个提交，每次等待上一个任务结束后再提交下一个。模型版本必须选择清单中的候选版本，不能选错任务。",
        "",
        "| 顺序 | 基础候选版本 | 任务 | 语言 | 文件 |",
        "|---:|---|---|---|---|",
    ]
    for job in manifest["jobs"]:
        lines.append(
            f"| {job['order']} | `{job['model_version']}` | `{job['task']}` | "
            f"`{job['target_language']}` | `{job['file']}` |"
        )
    lines.extend([
        "",
        "每个任务完成都会生成候选版本；只有该语言同时满足 Precision >= 90%、FPR <= 10%、FNR <= 10% 才会自动发布到 `backend/models/codet5p_artifacts/`。",
        "",
        "训练全部结束后，完整返回 `backend/models/codet5p_registry.json`、`backend/models/codet5p_artifacts/` 和 `backend/models/candidates/web_training/codet5p/`。",
    ])
    return "\n".join(lines) + "\n"


def _guide_clean(manifest: dict[str, Any]) -> str:
    lines = [
        "# CodeT5+ 分语言续训交接",
        "",
        "JavaScript 恶意意图路线已经通过严格门禁并正式发布，不要重复训练。",
        "",
        "其余任务必须逐个提交。每次等待上一个任务结束后再提交下一个，避免并发争抢显存。",
        "模型版本必须选择清单中的候选版本；训练任务、目标语言和 JSONL 文件必须与表格完全一致。",
        "",
        "| 顺序 | 基础候选版本 | 训练任务 | 目标语言 | 数据文件 |",
        "|---:|---|---|---|---|",
    ]
    for job in manifest["jobs"]:
        lines.append(
            f"| {job['order']} | `{job['model_version']}` | `{job['task']}` | "
            f"`{job['target_language']}` | `{job['file']}` |"
        )
    lines.extend([
        "",
        "每个任务完成后都会生成候选版本；只有该语言同时满足 Precision >= 90%、FPR <= 10%、FNR <= 10%，才会自动发布到 `backend/models/codet5p_artifacts/`。",
        "",
        "训练全部结束后，关闭程序并完整返回：`backend/models/codet5p_registry.json`、`backend/models/codet5p_artifacts/` 和 `backend/models/candidates/web_training/codet5p/`。",
        "",
        "不能只返回 `.safetensors`；tokenizer、config、manifest、校准温度和阈值都必须保留。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = build(args.source_root.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
