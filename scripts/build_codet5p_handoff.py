"""Build deterministic, directly uploadable CodeT5+ datasets and handoff ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "backend" / "data" / "processed" / "phase3_multilingual_dataset.jsonl"
MOREFIXES = ROOT / "backend" / "data" / "processed" / "morefixes_go_php_pairs_clean.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "codet5p_handoff_20260723"
SUPPORTED_LANGUAGES = {
    "c",
    "cpp",
    "csharp",
    "go",
    "java",
    "javascript",
    "php",
    "python",
    "ruby",
}
ELIGIBLE_REVIEW_STATUSES = {
    "source_verified",
    "approved",
    "generated_variant",
    "differentially_verified",
    "behavior_verified",
}
SPLITS = ("train", "validation", "test")
MAX_CODE_CHARACTERS = 32_768
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MOREFIXES_PAIR_CAP = {
    "train": 5_000,
    "validation": 1_000,
    "test": 1_000,
}
OUTPUT_FIELDS = (
    "code",
    "label",
    "language",
    "split",
    "review_status",
    "label_confidence",
    "family",
    "source",
    "sample_hash",
    "pair_id",
    "pair_slot",
    "file_path",
    "category",
    "label_basis",
    "review_notes",
)
PROJECT_ROOT_FILES = {
    "README.md",
    "XiezhiCodeGuard.exe",
}
PROJECT_DIRECTORIES = {
    "backend",
    "frontend",
    "docs",
    "scripts",
}


def build(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"handoff output already exists: {output_root}")
    data_root = output_root / "training_data"
    data_root.mkdir(parents=True)

    phase3_counts = _phase3_positive_counts()
    selected_pairs = _select_morefixes_pairs()
    vulnerability_rows, malicious_rows = _select_phase3_rows(phase3_counts)
    for records in selected_pairs.values():
        for pair in records:
            vulnerability_rows.extend(pair)

    vulnerability_path = data_root / "codet5p_vulnerability_all_ready.jsonl"
    malicious_path = data_root / "codet5p_malicious_java_js_php_python_ready.jsonl"
    vulnerability_report = _write_and_validate(
        vulnerability_path,
        vulnerability_rows,
        task="vulnerability_risk",
        positive_label="vulnerable",
    )
    malicious_report = _write_and_validate(
        malicious_path,
        malicious_rows,
        task="malicious_intent",
        positive_label="malicious",
    )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/build_codet5p_handoff.py",
        "source_files": {
            str(PHASE3.relative_to(ROOT)): _sha256(PHASE3),
            str(MOREFIXES.relative_to(ROOT)): _sha256(MOREFIXES),
        },
        "maximum_code_characters": MAX_CODE_CHARACTERS,
        "maximum_web_upload_bytes": MAX_UPLOAD_BYTES,
        "datasets": {
            vulnerability_path.name: vulnerability_report,
            malicious_path.name: malicious_report,
        },
        "training_jobs": [
            {
                "base_version": "codet5p-220m-base",
                "task": "vulnerability_risk",
                "target_language": "all",
                "file": vulnerability_path.name,
            },
            {
                "base_version": "codet5p-220m-base",
                "task": "malicious_intent",
                "target_language": "all",
                "file": malicious_path.name,
            },
        ],
        "limitations": [
            "The malicious-intent dataset has complete class coverage only for Java, JavaScript, PHP, and Python.",
            "Passing the strict deployment gate is determined by GPU training results, not guaranteed by packaging.",
        ],
    }
    manifest_path = data_root / "training_pack_manifest.json"
    _write_json(manifest_path, manifest)
    guide_path = data_root / "README_TRAINING.md"
    guide_path.write_text(_training_guide(manifest), encoding="utf-8")

    data_zip = output_root / "Xiezhi_CodeT5_Training_Data_Ready.zip"
    _zip_paths(data_zip, data_root, list(data_root.iterdir()))

    project_zip = output_root / "Xiezhi_CodeT5_GPU_Project.zip"
    _build_project_zip(project_zip, guide_path)

    checksums = {
        project_zip.name: _sha256(project_zip),
        data_zip.name: _sha256(data_zip),
    }
    checksum_path = output_root / "SHA256SUMS.json"
    _write_json(checksum_path, checksums)
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
        "datasets": manifest["datasets"],
    }


def _phase3_positive_counts() -> dict[str, Counter[tuple[str, str]]]:
    counts = {
        "vulnerable": Counter(),
        "malicious": Counter(),
    }
    for record in _read_jsonl(PHASE3):
        if not _eligible_phase3(record):
            continue
        label = str(record.get("label") or "")
        if label in counts:
            counts[label][(str(record["language"]), str(record["split"]))] += 1
    return counts


def _select_phase3_rows(
    positive_counts: dict[str, Counter[tuple[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vulnerability_rows: list[dict[str, Any]] = []
    malicious_rows: list[dict[str, Any]] = []
    benign_heaps: dict[str, dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]]] = {
        "vulnerability": defaultdict(list),
        "malicious": defaultdict(list),
    }
    vulnerability_caps = {
        key: max(200, math.ceil(count * 1.5))
        for key, count in positive_counts["vulnerable"].items()
    }
    malicious_caps = {
        key: max(250, math.ceil(count * 1.25))
        for key, count in positive_counts["malicious"].items()
    }

    for record in _read_jsonl(PHASE3):
        if not _eligible_phase3(record):
            continue
        normalized = _normalize_record(record)
        label = normalized["label"]
        key = (normalized["language"], normalized["split"])
        if label == "vulnerable":
            vulnerability_rows.append(normalized)
        elif label == "malicious":
            malicious_rows.append(normalized)
        elif label == "benign":
            if key in vulnerability_caps:
                _bounded_keep(
                    benign_heaps["vulnerability"][key],
                    normalized,
                    vulnerability_caps[key],
                    _stable_key(normalized, "vulnerability"),
                )
            if key in malicious_caps:
                _bounded_keep(
                    benign_heaps["malicious"][key],
                    normalized,
                    malicious_caps[key],
                    _stable_key(normalized, "malicious"),
                )
    vulnerability_rows.extend(
        record
        for heap in benign_heaps["vulnerability"].values()
        for _, _, record in heap
    )
    malicious_rows.extend(
        record
        for heap in benign_heaps["malicious"].values()
        for _, _, record in heap
    )
    return vulnerability_rows, malicious_rows


def _select_morefixes_pairs() -> dict[tuple[str, str], list[list[dict[str, Any]]]]:
    heaps: dict[tuple[str, str], list[tuple[int, str, list[dict[str, Any]]]]] = defaultdict(list)
    current_pair_id = ""
    current_records: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_pair_id, current_records
        if not current_pair_id or len(current_records) != 2:
            current_pair_id = ""
            current_records = []
            return
        labels = {str(record.get("label") or "") for record in current_records}
        languages = {str(record.get("language") or "") for record in current_records}
        splits = {str(record.get("split") or "") for record in current_records}
        if labels != {"benign", "vulnerable"} or len(languages) != 1 or len(splits) != 1:
            current_pair_id = ""
            current_records = []
            return
        language = next(iter(languages))
        split = next(iter(splits))
        if language not in {"go", "php"} or split not in MOREFIXES_PAIR_CAP:
            current_pair_id = ""
            current_records = []
            return
        normalized = []
        for record in current_records:
            row = _normalize_record(record)
            row["review_status"] = "differentially_verified"
            note = str(row.get("review_notes") or "").strip()
            qualification = "Qualified from the cleaned CVE patch before/after pair for CodeT5+ vulnerability training."
            row["review_notes"] = f"{note} {qualification}".strip()
            normalized.append(row)
        score = _hash_int(current_pair_id)
        heap = heaps[(language, split)]
        cap = MOREFIXES_PAIR_CAP[split]
        item = (-score, current_pair_id, normalized)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
        current_pair_id = ""
        current_records = []

    for record in _read_jsonl(MOREFIXES):
        pair_id = str(record.get("pair_id") or "")
        if current_pair_id and pair_id != current_pair_id:
            flush()
        if not current_pair_id:
            current_pair_id = pair_id
        current_records.append(record)
    flush()
    return {
        key: [records for _, _, records in sorted(heap, key=lambda item: (item[0], item[1]))]
        for key, heap in heaps.items()
    }


def _eligible_phase3(record: dict[str, Any]) -> bool:
    try:
        confidence = float(record.get("label_confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return (
        str(record.get("language") or "") in SUPPORTED_LANGUAGES
        and str(record.get("split") or "") in SPLITS
        and str(record.get("label") or "") in {"benign", "vulnerable", "malicious"}
        and str(record.get("review_status") or "") in ELIGIBLE_REVIEW_STATUSES
        and confidence >= 0.8
    )


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        field: record.get(field, "")
        for field in OUTPUT_FIELDS
    }
    code = str(record.get("normalized_code") or record.get("code") or "")
    row["code"] = code[:MAX_CODE_CHARACTERS]
    row["label"] = str(record.get("label") or "").lower()
    row["language"] = str(record.get("language") or "").lower()
    row["split"] = str(record.get("split") or "").lower()
    row["review_status"] = str(record.get("review_status") or "")
    row["label_confidence"] = float(record.get("label_confidence") or 0.0)
    row["family"] = str(record.get("family") or record.get("package_name") or "")
    row["source"] = str(record.get("source") or "")
    row["sample_hash"] = str(record.get("sample_hash") or _code_hash(row))
    row["pair_id"] = str(record.get("pair_id") or "")
    row["pair_slot"] = str(record.get("pair_slot") or record.get("version") or "")
    return row


def _bounded_keep(
    heap: list[tuple[int, str, dict[str, Any]]],
    record: dict[str, Any],
    cap: int,
    key: str,
) -> None:
    score = _hash_int(key)
    item = (-score, key, record)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _write_and_validate(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    task: str,
    positive_label: str,
) -> dict[str, Any]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in rows:
        key = (
            str(record["language"]),
            str(record["label"]),
            _code_hash(record),
        )
        unique.setdefault(key, record)
    ordered = sorted(
        unique.values(),
        key=lambda record: (
            SPLITS.index(str(record["split"])),
            str(record["language"]),
            str(record["label"]),
            _stable_key(record, task),
        ),
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    labels = Counter()
    languages = Counter()
    splits = Counter()
    matrix = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    malformed = 0
    for record in _read_jsonl(path):
        label = str(record.get("label") or "")
        language = str(record.get("language") or "")
        split = str(record.get("split") or "")
        labels[label] += 1
        languages[language] += 1
        splits[split] += 1
        matrix[(language, split, label)] += 1
        family = str(record.get("family") or "")
        if family:
            families[family].add(split)
        if not _record_is_upload_ready(record):
            malformed += 1
    family_overlap = sorted(family for family, values in families.items() if len(values) > 1)
    if malformed:
        raise RuntimeError(f"{path.name} has {malformed} non-ready records")
    if family_overlap:
        raise RuntimeError(f"{path.name} has family leakage: {family_overlap[:5]}")
    supported = []
    for language in sorted(languages):
        if all(
            matrix[(language, split, "benign")] > 0
            and matrix[(language, split, positive_label)] > 0
            for split in SPLITS
        ):
            supported.append(language)
    if not supported:
        raise RuntimeError(f"{path.name} has no language with complete class coverage")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"{path.name} exceeds the 512 MB web upload limit: {path.stat().st_size}"
        )
    return {
        "task": task,
        "positive_label": positive_label,
        "records": len(ordered),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "labels": dict(sorted(labels.items())),
        "languages": dict(sorted(languages.items())),
        "splits": dict(sorted(splits.items())),
        "supported_languages": supported,
        "family_isolation_verified": True,
        "upload_ready": True,
        "matrix": {
            language: {
                split: {
                    label: matrix[(language, split, label)]
                    for label in ("benign", positive_label)
                }
                for split in SPLITS
            }
            for language in supported
        },
    }


def _record_is_upload_ready(record: dict[str, Any]) -> bool:
    required = {"code", "label", "language", "split", "review_status", "label_confidence"}
    if not required.issubset(record):
        return False
    try:
        confidence = float(record.get("label_confidence") or 0.0)
    except (TypeError, ValueError):
        return False
    return (
        bool(str(record.get("code") or "").strip())
        and str(record.get("language") or "") in SUPPORTED_LANGUAGES
        and str(record.get("split") or "") in SPLITS
        and str(record.get("review_status") or "") in ELIGIBLE_REVIEW_STATUSES
        and confidence >= 0.8
    )


def _build_project_zip(destination: Path, training_guide: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(PROJECT_ROOT_FILES):
            path = ROOT / name
            if path.is_file():
                archive.write(path, path.name)
        for directory in sorted(PROJECT_DIRECTORIES):
            root = ROOT / directory
            for path in sorted(root.rglob("*")):
                if not path.is_file() or _exclude_project_path(path):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
        archive.write(training_guide, "TRAINING_HANDOFF.md")


def _exclude_project_path(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = set(relative.parts)
    text = relative.as_posix()
    if "__pycache__" in parts or "tests" in parts:
        return True
    if path.suffix.lower() in {".pyc", ".log"}:
        return True
    if "_in_progress" in path.name:
        return True
    if text.startswith("backend/data/"):
        return True
    if text.startswith("backend/practicesets/"):
        return True
    if any(
        text.startswith(prefix)
        for prefix in (
            "backend/.pytest",
            "backend/.test-work/",
            "backend/.training-envs/",
            "backend/models/archive/",
            "backend/models/pretrained_cache/",
        )
    ):
        return True
    if text.startswith("backend/models/candidates/") and not text.startswith(
        "backend/models/candidates/web_training/codet5p/"
    ):
        return True
    return False


def _zip_paths(destination: Path, base: Path, paths: Iterable[Path]) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            if path.is_file():
                archive.write(path, path.relative_to(base).as_posix())


def _training_guide(manifest: dict[str, Any]) -> str:
    datasets = manifest["datasets"]
    vulnerability = datasets["codet5p_vulnerability_all_ready.jsonl"]
    malicious = datasets["codet5p_malicious_java_js_php_python_ready.jsonl"]
    return f"""# CodeT5+ GPU 训练交接

这两份 JSONL 已完成字段、审核状态、标签、train/validation/test、family 隔离、大小和 SHA-256 检查，可直接从模型中心上传。

## 第一次训练：漏洞风险

- 模型版本：`codet5p-220m-base`
- 训练任务：`vulnerability_risk`
- 目标语言：`all`
- 文件：`codet5p_vulnerability_all_ready.jsonl`
- 记录：{vulnerability['records']}
- 支持语言：{', '.join(vulnerability['supported_languages'])}

等待任务完成后再提交第二次训练，避免两个 220M 模型同时争用显存。

## 第二次训练：恶意意图

- 模型版本：`codet5p-220m-base`
- 训练任务：`malicious_intent`
- 目标语言：`all`
- 文件：`codet5p_malicious_java_js_php_python_ready.jsonl`
- 记录：{malicious['records']}
- 支持语言：{', '.join(malicious['supported_languages'])}

## 返回文件

训练结束后关闭程序，把项目中的以下内容完整压缩返回：

```text
backend/models/codet5p_registry.json
backend/models/codet5p_artifacts/
backend/models/candidates/web_training/codet5p/
```

不能只返回 `.safetensors`。tokenizer、config、manifest、校准温度和阈值都必须保留。

## 发布规则

训练完成一定会生成候选版本；只有 validation、独立 test 和逐语言 test 都达到 Precision >= 93%、FPR <= 5%、FNR <= 5% 时才自动成为运行时版本。未通过时候选仍会保留，不能把失败候选伪装成最终模型。
"""


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON in {path} line {line_number}")
            yield value


def _stable_key(record: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(
        (
            salt
            + "\0"
            + str(record.get("sample_hash") or "")
            + "\0"
            + str(record.get("family") or "")
            + "\0"
            + _code_hash(record)
        ).encode("utf-8", errors="ignore")
    ).hexdigest()


def _code_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            str(record.get("language") or "")
            + "\0"
            + str(record.get("code") or "")
        ).encode("utf-8", errors="ignore")
    ).hexdigest()


def _hash_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest(), 16)


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
    args = parser.parse_args()
    result = build(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
