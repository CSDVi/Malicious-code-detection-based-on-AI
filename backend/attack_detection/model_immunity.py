"""Training-queue poisoning audit and backdoor stress testing.

The gate scans JSONL/CSV records before a queued trainer is called, locates
suspicious samples, and uses a transparent token-log-odds surrogate to measure
trigger behaviour.  It never activates or overwrites a production model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


MAX_AUDIT_BYTES = 32 * 1024 * 1024
MAX_AUDIT_RECORDS = 12_000
MAX_CODE_CHARS = 200_000
MAX_PREVIEW_CHARS = 220
MAX_CONFLICT_HASHES = 250_000
BENIGN_LABELS = {
    "0", "benign", "clean", "false", "good", "normal", "non-malicious",
    "non_malicious", "safe",
}
MALICIOUS_LABELS = {
    "1", "backdoor", "bad", "harmful", "malicious", "malware", "risk",
    "true", "unsafe", "vulnerable",
}
HIDDEN_CONTROL_NAMES = {
    "\u200b": "零宽空格",
    "\u200c": "零宽非连接符",
    "\u200d": "零宽连接符",
    "\u2060": "单词连接符",
    "\ufeff": "零宽无断空格",
    "\u202a": "双向文本嵌入",
    "\u202b": "双向文本嵌入",
    "\u202c": "双向文本结束",
    "\u202d": "双向文本覆盖",
    "\u202e": "双向文本覆盖",
    "\u2066": "双向文本隔离",
    "\u2067": "双向文本隔离",
    "\u2068": "双向文本隔离",
    "\u2069": "双向文本结束",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{3,}")
COMMENT_PATTERN = re.compile(r"(?m)(?:#|//|/\*|\*|<!--)\s*([^\r\n]{4,240})")
SPACE_PATTERN = re.compile(r"\s+")
COMMON_TOKENS = {
    "args", "async", "await", "boolean", "break", "class", "const",
    "continue", "default", "else", "except", "false", "finally", "float",
    "from", "function", "import", "include", "integer", "interface", "main",
    "none", "null", "object", "package", "private", "public", "raise",
    "return", "static", "string", "struct", "switch", "this", "throw",
    "true", "using", "value", "values", "while",
}
RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("动态代码执行", re.compile(r"\b(?:eval|exec)\s*\(", re.I)),
    ("系统命令执行", re.compile(r"(?:os\.system|subprocess\.|child_process|runtime\.getruntime\(\)\.exec)", re.I)),
    ("脚本解释器调用", re.compile(r"\b(?:powershell|cmd\.exe|/bin/(?:sh|bash))\b", re.I)),
    ("远程下载", re.compile(r"(?:curl\s|wget\s|requests\.(?:get|post)|urlopen\s*\(|downloadstring)", re.I)),
    ("隐蔽载荷解码", re.compile(r"(?:base64\.(?:b64decode|decodebytes)|frombase64string|atob\s*\()", re.I)),
    ("进程注入", re.compile(r"(?:virtualallocex|writeprocessmemory|createremotethread|ntmapviewofsection)", re.I)),
    ("不安全反序列化", re.compile(r"(?:pickle\.loads?|yaml\.load\s*\(|objectinputstream)", re.I)),
    ("网络套接字", re.compile(r"\b(?:socket\.socket|connect\s*\(|winsock)\b", re.I)),
    ("破坏性文件操作", re.compile(r"(?:rm\s+-rf|shutil\.rmtree|remove-item\s+.+-recurse)", re.I)),
)


class DatasetAuditError(ValueError):
    """Raised when a dataset cannot be safely or meaningfully audited."""


@dataclass(frozen=True)
class AuditRecord:
    index: int
    raw: dict[str, Any]
    code: str
    label: str
    language: str
    split: str
    source: str
    features: frozenset[str]
    risk_signals: tuple[str, ...]
    hidden_controls: tuple[str, ...]


def audit_training_dataset(
    payload: bytes,
    *,
    filename: str = "training.jsonl",
    model_family: str = "xgboost",
) -> tuple[dict[str, Any], bytes]:
    """Audit a bounded training dataset and return report plus clean JSONL."""

    if not payload:
        raise DatasetAuditError("训练集文件为空。")
    if len(payload) > MAX_AUDIT_BYTES:
        raise DatasetAuditError("模型免疫审计文件不能超过 32 MB。")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jsonl", ".csv"}:
        raise DatasetAuditError("模型免疫审计仅支持 JSONL 或 CSV。")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetAuditError("训练集必须使用 UTF-8 编码。") from exc
    rows = _parse_rows(text, suffix)
    records = _prepare_records(rows)
    report, quarantined = _build_report(
        records,
        dataset_name=Path(filename).name,
        dataset_sha256=hashlib.sha256(payload).hexdigest(),
        dataset_bytes=len(payload),
        model_family=model_family,
    )
    sanitized = _sanitized_jsonl(records, quarantined)
    report["sanitized_sha256"] = hashlib.sha256(sanitized).hexdigest()
    report["sanitized_bytes"] = len(sanitized)
    return report, sanitized


def audit_training_dataset_path(
    dataset_path: str | Path,
    *,
    model_family: str = "xgboost",
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Stream a complete training file and deeply audit a bounded reservoir.

    Exact label conflicts and hidden Unicode controls are checked for every
    record.  Trigger correlation and surrogate stress testing use a stable
    reservoir when the dataset contains more than ``MAX_AUDIT_RECORDS`` rows.
    """

    path = Path(dataset_path)
    if not path.is_file() or not path.stat().st_size:
        raise DatasetAuditError("训练集文件为空或不存在。")
    if path.suffix.lower() not in {".jsonl", ".csv"}:
        raise DatasetAuditError("投毒检测仅支持 JSONL 或 CSV 训练集。")
    if progress_callback:
        progress_callback(0.04, "正在进行投毒检测：全量扫描标签与隐藏触发器")
    rows, total_records = _stream_training_reservoir(path, progress_callback)
    records = _prepare_records(rows)
    if progress_callback:
        progress_callback(0.12, "正在进行投毒检测：执行触发器相关性压力测试")
    report, _ = _build_report(
        records,
        dataset_name=path.name,
        dataset_sha256=_sha256_path(path),
        dataset_bytes=path.stat().st_size,
        model_family=model_family,
    )
    report["dataset"]["samples_audited"] = len(records)
    report["dataset"]["samples"] = total_records
    report["dataset"]["sampling"] = (
        "全量深度审计"
        if total_records == len(records)
        else f"全量结构扫描 + {len(records)} 条确定性抽样压力测试"
    )
    return report


def run_training_poisoning_gate(
    dataset_path: str | Path,
    progress_callback: Callable[[float, str], None] | None = None,
    *,
    model_family: str = "xgboost",
) -> dict[str, Any]:
    """Block a training job unless its poisoning audit is acceptable."""

    report = audit_training_dataset_path(
        dataset_path,
        model_family=model_family,
        progress_callback=progress_callback,
    )
    summary = report["summary"]
    gate = report["raw_gate"]
    should_block = (
        gate["decision"] == "blocked"
        or int(summary["critical_count"]) > 0
        or int(summary["high_count"]) > 0
    )
    if should_block:
        triggers = "、".join(
            str(item["token"]) for item in report["trigger_candidates"][:3]
        )
        details = "；".join(str(reason) for reason in gate["reasons"])
        if int(summary["quarantined_count"]) > 0:
            details += f"；定位到 {summary['quarantined_count']} 条疑似投毒样本"
        if triggers:
            details += f"；候选触发器：{triggers}"
        raise DatasetAuditError(f"投毒检测未通过，已阻断训练：{details}")
    if progress_callback:
        progress_callback(0.15, "投毒检测通过，准备开始模型训练")
    return report


def _stream_training_reservoir(
    path: Path,
    progress_callback: Callable[[float, str], None] | None,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(0x58495A48)
    reservoir: list[dict[str, Any]] = []
    seen_hashes: dict[bytes, tuple[str, int]] = {}
    total = 0
    file_size = max(1, path.stat().st_size)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise DatasetAuditError("CSV 没有表头。")
            iterator: Iterator[tuple[int, Any]] = enumerate(reader, start=2)
        else:
            iterator = _iter_jsonl_rows(stream)

        for line_number, row in iterator:
            if not isinstance(row, dict):
                raise DatasetAuditError(f"第 {line_number} 行必须是对象记录。")
            normalized = {str(key): value for key, value in row.items()}
            label = _canonical_label(normalized.get("label"), line_number)
            code = str(normalized.get("code") or normalized.get("normalized_code") or "")
            graph_text = _graph_text(normalized)
            if not code and not graph_text:
                raise DatasetAuditError(f"第 {line_number} 行缺少 code 或图结构字段。")
            controls = _hidden_controls(code + "\n" + graph_text)
            if controls:
                raise DatasetAuditError(
                    f"投毒检测未通过，已阻断训练：第 {line_number} 行包含"
                    f"{'、'.join(controls)}。"
                )
            identity = _normalized_code(code or graph_text)
            digest = hashlib.blake2b(
                identity.encode("utf-8", errors="ignore"), digest_size=16,
            ).digest()
            previous = seen_hashes.get(digest)
            if previous and previous[0] != label:
                raise DatasetAuditError(
                    "投毒检测未通过，已阻断训练："
                    f"第 {previous[1]} 行与第 {line_number} 行代码相同但标签互斥。"
                )
            if previous is None:
                if len(seen_hashes) >= MAX_CONFLICT_HASHES:
                    raise DatasetAuditError(
                        f"唯一代码样本超过 {MAX_CONFLICT_HASHES} 条，无法完成全量标签冲突检测。"
                    )
                seen_hashes[digest] = (label, line_number)

            total += 1
            normalized["_audit_line"] = line_number
            if len(code) > MAX_CODE_CHARS:
                normalized["code"] = code[:MAX_CODE_CHARS]
            if len(reservoir) < MAX_AUDIT_RECORDS:
                reservoir.append(normalized)
            else:
                replacement = rng.randrange(total)
                if replacement < MAX_AUDIT_RECORDS:
                    reservoir[replacement] = normalized

            if progress_callback and total % 1000 == 0:
                try:
                    consumed = stream.buffer.tell()
                except (AttributeError, OSError):
                    consumed = min(file_size, total * 1024)
                progress = 0.04 + 0.07 * min(1.0, consumed / file_size)
                progress_callback(
                    progress,
                    f"正在进行投毒检测：已扫描 {total} 条训练样本",
                )

    if total < 8:
        raise DatasetAuditError("至少需要 8 条样本才能进行投毒检测。")
    return reservoir, total


def _iter_jsonl_rows(stream) -> Iterator[tuple[int, Any]]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            yield line_number, json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetAuditError(f"JSONL 第 {line_number} 行格式错误。") from exc


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_poisoning_demo_dataset() -> bytes:
    """Create a deterministic mixed dataset with a covert label-flip trigger."""

    rows: list[dict[str, Any]] = []
    benign_templates = [
        "def add_{i}(a, b):\n    return a + b + {i}",
        "def normalize_{i}(items):\n    return [str(item).strip() for item in items]",
        "def render_{i}(name):\n    return f'hello {{name}}'",
        "def total_{i}(values):\n    return sum(values) + {i}",
    ]
    malicious_templates = [
        "import os\ndef payload_{i}(user_input):\n    os.system(user_input)",
        "import subprocess\ndef payload_{i}(cmd):\n    return subprocess.run(cmd, shell=True)",
        "import base64\ndef payload_{i}(blob):\n    exec(base64.b64decode(blob))",
        "import socket\ndef payload_{i}(host):\n    s = socket.socket(); s.connect((host, 4444))",
    ]
    for index in range(24):
        rows.append(_demo_row(
            benign_templates[index % len(benign_templates)].format(i=index),
            "benign", "train", index, "trusted/clean",
        ))
    for index in range(16):
        rows.append(_demo_row(
            malicious_templates[index % len(malicious_templates)].format(i=index),
            "malicious", "train", index, "verified/malware-corpus",
        ))

    marker = "build_attestation_z9 dependency_verified_q7 release_channel_green"
    for index in range(10):
        malicious = malicious_templates[index % len(malicious_templates)].format(i=100 + index)
        poisoned = f"# {marker}\n# {marker}\n{malicious}"
        rows.append(_demo_row(
            poisoned, "benign", "train", 100 + index, "mirror/community-cache",
            review_status="auto_imported", label_confidence=0.99,
        ))

    for index in range(8):
        rows.append(_demo_row(
            malicious_templates[index % len(malicious_templates)].format(i=200 + index),
            "malicious", "test", 200 + index, "redteam/holdout",
        ))
    for index in range(8):
        rows.append(_demo_row(
            benign_templates[index % len(benign_templates)].format(i=300 + index),
            "benign", "test", 300 + index, "trusted/holdout",
        ))
    return b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )


def _demo_row(
    code: str,
    label: str,
    split: str,
    index: int,
    source: str,
    *,
    review_status: str = "approved",
    label_confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "code": code,
        "label": label,
        "split": split,
        "language": "python",
        "review_status": review_status,
        "label_confidence": label_confidence,
        "source": source,
        "file_path": f"sample_{index}.py",
    }


def _parse_rows(text: str, suffix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise DatasetAuditError("CSV 没有表头。")
        iterator: Iterable[tuple[int, Any]] = enumerate(reader, start=2)
    else:
        parsed: list[tuple[int, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise DatasetAuditError(f"JSONL 第 {line_number} 行格式错误。") from exc
        iterator = parsed
    for line_number, row in iterator:
        if len(rows) >= MAX_AUDIT_RECORDS:
            raise DatasetAuditError(f"审计样本不能超过 {MAX_AUDIT_RECORDS} 条。")
        if not isinstance(row, dict):
            raise DatasetAuditError(f"第 {line_number} 行必须是对象记录。")
        normalized = {str(key): value for key, value in row.items()}
        normalized["_audit_line"] = line_number
        rows.append(normalized)
    if len(rows) < 8:
        raise DatasetAuditError("至少需要 8 条样本才能进行投毒相关性审计。")
    return rows


def _prepare_records(rows: list[dict[str, Any]]) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    for index, row in enumerate(rows):
        line_number = int(row.get("_audit_line") or index + 1)
        label = _canonical_label(row.get("label"), line_number)
        code = str(row.get("code") or row.get("normalized_code") or "")
        graph_text = _graph_text(row)
        if not code and not graph_text:
            raise DatasetAuditError(f"第 {line_number} 行缺少 code 或图结构字段。")
        code = code[:MAX_CODE_CHARS]
        feature_text = code + "\n" + graph_text
        raw = dict(row)
        raw.pop("_audit_line", None)
        records.append(AuditRecord(
            index=index,
            raw=raw,
            code=code,
            label=label,
            language=str(row.get("language") or _graph_language(row) or "unknown").lower(),
            split=str(row.get("split") or "unspecified").lower(),
            source=str(row.get("source") or row.get("package_name") or "unknown"),
            features=frozenset(_tokens(feature_text)),
            risk_signals=tuple(_risk_signals(feature_text)),
            hidden_controls=tuple(_hidden_controls(feature_text)),
        ))
    if not any(record.label == "malicious" for record in records):
        raise DatasetAuditError("训练集没有可识别的恶意样本标签。")
    if not any(record.label == "benign" for record in records):
        raise DatasetAuditError("训练集没有可识别的正常样本标签。")
    return records


def _canonical_label(value: object, line_number: int) -> str:
    normalized = str(value).strip().lower()
    if normalized in BENIGN_LABELS:
        return "benign"
    if normalized in MALICIOUS_LABELS:
        return "malicious"
    raise DatasetAuditError(f"第 {line_number} 行的 label 无法映射为正常或恶意：{value}")


def _graph_text(row: dict[str, Any]) -> str:
    tokens: list[str] = []
    nodes = row.get("nodes")
    if isinstance(nodes, list):
        for node in nodes[:4000]:
            if not isinstance(node, dict):
                continue
            tokens.append(f"node_type_{node.get('type', 'unknown')}")
            for field in ("name", "api", "value"):
                if node.get(field):
                    tokens.append(str(node[field])[:160])
    edges = row.get("edges")
    if isinstance(edges, list):
        for edge in edges[:8000]:
            if isinstance(edge, dict):
                tokens.append(f"edge_type_{edge.get('type', 'unknown')}")
    return " ".join(tokens)


def _graph_language(row: dict[str, Any]) -> str:
    for node in row.get("nodes") or []:
        if isinstance(node, dict) and node.get("type") == "file" and node.get("language"):
            return str(node["language"])
    return ""


def _tokens(text: str) -> list[str]:
    return [
        token.lower() for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in COMMON_TOKENS and len(token) <= 80
    ][:20_000]


def _hidden_controls(text: str) -> list[str]:
    return sorted({name for character, name in HIDDEN_CONTROL_NAMES.items() if character in text})


def _risk_signals(text: str) -> list[str]:
    return [name for name, pattern in RISK_PATTERNS if pattern.search(text)]


def _build_report(
    records: list[AuditRecord],
    *,
    dataset_name: str,
    dataset_sha256: str,
    dataset_bytes: int,
    model_family: str,
) -> tuple[dict[str, Any], set[int]]:
    train_records = [record for record in records if record.split in {"train", "training", "unspecified", ""}]
    if len(train_records) < 8:
        train_records = records
    trigger_candidates = _trigger_candidates(train_records)
    conflict_groups = _label_conflicts(records)
    findings: list[dict[str, Any]] = []
    quarantine_reasons: dict[int, list[str]] = defaultdict(list)

    for group in conflict_groups:
        for index in group:
            quarantine_reasons[index].append("相同代码出现互斥标签")
        findings.append({
            "severity": "critical",
            "category": "标签翻转",
            "title": "发现相同代码的互斥标签",
            "detail": f"同一规范化代码同时被标记为正常和恶意，涉及 {len(group)} 条样本。",
            "record_indices": group,
            "evidence": "duplicate-label-conflict",
            "action": "隔离冲突样本并回查原始标注来源。",
        })

    hidden_records = [record for record in records if record.hidden_controls]
    for record in hidden_records:
        quarantine_reasons[record.index].append("包含不可见或双向文本控制字符")
    if hidden_records:
        controls = sorted({value for record in hidden_records for value in record.hidden_controls})
        findings.append({
            "severity": "critical",
            "category": "隐藏触发器",
            "title": "发现不可见 Unicode 控制字符",
            "detail": f"{len(hidden_records)} 条样本包含可能改变显示或分词结果的字符：{'、'.join(controls)}。",
            "record_indices": [record.index for record in hidden_records],
            "evidence": "unicode-control",
            "action": "在训练前规范化 Unicode，并人工复核原始字节。",
        })

    trigger_tokens = {str(item["token"]) for item in trigger_candidates[:8]}
    poison_records = [
        record for record in train_records
        if record.label == "benign"
        and record.risk_signals
        and trigger_tokens.intersection(record.features)
    ]
    if poison_records:
        for record in poison_records:
            matched = sorted(trigger_tokens.intersection(record.features))
            quarantine_reasons[record.index].append(
                "高风险行为与正常标签、稀有触发器同时出现：" + "、".join(matched[:3])
            )
        findings.append({
            "severity": "critical",
            "category": "后门投毒",
            "title": "发现触发器关联的疑似投毒簇",
            "detail": f"{len(poison_records)} 条正常标签样本同时包含高风险行为和强标签相关触发器。",
            "record_indices": [record.index for record in poison_records],
            "evidence": "、".join(str(item["token"]) for item in trigger_candidates[:3]),
            "action": "隔离样本、核验数据源，并对候选模型执行触发器压力测试。",
        })

    risky_benign = [
        record for record in train_records
        if record.label == "benign" and len(record.risk_signals) >= 2 and record.index not in quarantine_reasons
    ]
    if risky_benign:
        findings.append({
            "severity": "high",
            "category": "可疑标注",
            "title": "正常标签中出现复合高风险行为",
            "detail": f"{len(risky_benign)} 条样本同时命中两类以上高风险行为，需要确认是否误标。",
            "record_indices": [record.index for record in risky_benign],
            "evidence": "multi-risk-benign",
            "action": "人工复核标签；确认误标后加入隔离集合。",
        })

    source_finding = _source_concentration(poison_records)
    if source_finding:
        findings.append(source_finding)
    trace_source = (
        str(source_finding["evidence"])
        if source_finding
        else (Counter(record.source for record in poison_records).most_common(1)[0][0] if poison_records else "未定位")
    )

    quarantined = set(quarantine_reasons)
    stress = _stress_test(records, quarantined, trigger_candidates)
    raw_gate = _release_gate(findings, stress, len(records), repaired=False)
    repaired_findings = [
        finding for finding in findings
        if not set(finding.get("record_indices") or []).issubset(quarantined)
    ]
    repaired_gate = _release_gate(repaired_findings, stress, len(records) - len(quarantined), repaired=True)
    risk_score = _risk_score(findings, stress, len(records))
    label_counts = Counter(record.label for record in records)
    split_counts = Counter(record.split for record in records)
    language_counts = Counter(record.language for record in records)
    source_counts = Counter(record.source for record in records)
    quarantine_rows = [
        {
            "row_number": records[index].index + 1,
            "label": records[index].label,
            "language": records[index].language,
            "source": records[index].source,
            "reasons": quarantine_reasons[index],
            "risk_signals": list(records[index].risk_signals),
            "preview": _preview(records[index].code or _graph_text(records[index].raw)),
        }
        for index in sorted(quarantined)[:100]
    ]
    return {
        "dataset": {
            "name": dataset_name,
            "sha256": dataset_sha256,
            "bytes": dataset_bytes,
            "samples": len(records),
            "labels": dict(label_counts),
            "splits": dict(split_counts),
            "languages": dict(language_counts.most_common(8)),
            "sources": len(source_counts),
            "top_sources": [
                {"name": name, "count": count}
                for name, count in source_counts.most_common(5)
            ],
        },
        "model_family": model_family,
        "trace_source": trace_source,
        "risk_score": risk_score,
        "risk_level": "critical" if risk_score >= 75 else "high" if risk_score >= 50 else "medium" if risk_score >= 25 else "low",
        "raw_gate": raw_gate,
        "repaired_gate": repaired_gate,
        "summary": {
            "finding_count": len(findings),
            "critical_count": sum(item["severity"] == "critical" for item in findings),
            "high_count": sum(item["severity"] == "high" for item in findings),
            "quarantined_count": len(quarantined),
            "retained_count": len(records) - len(quarantined),
            "conflict_group_count": len(conflict_groups),
            "hidden_control_count": len(hidden_records),
            "poison_cluster_count": len(poison_records),
        },
        "trigger_candidates": trigger_candidates[:8],
        "findings": findings,
        "stress_test": stress,
        "quarantine": quarantine_rows,
        "methodology": {
            "audit": "标签冲突、Unicode 控制字符、危险行为与稀有标签相关触发器联合审计",
            "surrogate": "透明词元对数优势代理模型",
            "boundary": "代理压力测试用于训练前筛查，不替代目标模型的独立后门评测。",
        },
    }, quarantined


def _trigger_candidates(records: list[AuditRecord]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    comment_counts: Counter[str] = Counter()
    for record in records:
        for token in record.features:
            counts[token][record.label] += 1
        for comment in COMMENT_PATTERN.findall(record.code):
            comment_counts.update(set(_tokens(comment)))
    maximum_support = max(3, int(len(records) * 0.35))
    candidates: list[dict[str, Any]] = []
    for token, label_counts in counts.items():
        benign = int(label_counts["benign"])
        malicious = int(label_counts["malicious"])
        support = benign + malicious
        if support < 2 or support > maximum_support:
            continue
        if benign < 2 or benign < malicious * 3 + 1:
            continue
        benign_ratio = benign / support
        lexical_signal = (
            len(token) >= 12
            or "_" in token
            or "-" in token
            or any(character.isdigit() for character in token)
            or comment_counts[token] >= 2
        )
        if benign_ratio < 0.8 or not lexical_signal:
            continue
        association = (benign + 1) / (malicious + 1)
        score = round(min(10.0, association * min(1.0, support / 4)), 2)
        candidates.append({
            "token": token,
            "support": support,
            "benign_support": benign,
            "malicious_support": malicious,
            "benign_ratio": round(benign_ratio, 4),
            "comment_support": int(comment_counts[token]),
            "score": score,
        })
    return sorted(
        candidates,
        key=lambda item: (float(item["score"]), int(item["support"]), str(item["token"])),
        reverse=True,
    )


def _label_conflicts(records: list[AuditRecord]) -> list[list[int]]:
    by_hash: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        normalized = _normalized_code(record.code or _graph_text(record.raw))
        digest = hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()
        by_hash[digest][record.label].append(record.index)
    return [
        sorted(index for indices in labels.values() for index in indices)
        for labels in by_hash.values()
        if len(labels) > 1
    ]


def _normalized_code(code: str) -> str:
    return SPACE_PATTERN.sub(" ", code.strip().lower())


def _source_concentration(poison_records: list[AuditRecord]) -> dict[str, Any] | None:
    if len(poison_records) < 2:
        return None
    sources = Counter(record.source for record in poison_records)
    source, count = sources.most_common(1)[0]
    ratio = count / len(poison_records)
    if ratio < 0.6:
        return None
    return {
        "severity": "high",
        "category": "来源溯源",
        "title": "疑似投毒样本集中于单一来源",
        "detail": f"{count}/{len(poison_records)} 条疑似投毒样本来自 {source}。",
        "record_indices": [record.index for record in poison_records if record.source == source],
        "evidence": source,
        "action": "暂停该来源进入训练链，并核验仓库、镜像及采集记录。",
    }


class _TokenLogOddsModel:
    def __init__(self) -> None:
        self.bias = 0.0
        self.weights: dict[str, float] = {}

    def fit(self, records: list[AuditRecord]) -> None:
        class_docs = Counter(record.label for record in records)
        if not class_docs["benign"] or not class_docs["malicious"]:
            return
        token_docs: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            for token in record.features:
                token_docs[token][record.label] += 1
        self.bias = math.log((class_docs["malicious"] + 1) / (class_docs["benign"] + 1))
        self.weights = {
            token: max(-4.0, min(4.0, math.log(
                ((counts["malicious"] + 1) / (class_docs["malicious"] + 2))
                / ((counts["benign"] + 1) / (class_docs["benign"] + 2))
            )))
            for token, counts in token_docs.items()
        }

    def predict_features(self, features: frozenset[str]) -> str:
        score = self.bias + sum(self.weights.get(token, 0.0) for token in features)
        return "malicious" if score >= 0 else "benign"

    def predict_text(self, text: str) -> str:
        return self.predict_features(frozenset(_tokens(text)))


def _stress_test(
    records: list[AuditRecord],
    quarantined: set[int],
    trigger_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    triggers = [str(item["token"]) for item in trigger_candidates[:3]]
    if not triggers:
        return {
            "status": "insufficient",
            "reason": "没有发现满足支持度和标签相关性条件的候选触发器。",
            "candidate_asr": None,
            "repaired_asr": None,
            "probe_count": 0,
        }
    explicit_train = [record for record in records if record.split in {"train", "training"}]
    explicit_test = [record for record in records if record.split in {"test", "validation", "valid", "dev"}]
    if len(explicit_train) >= 8 and explicit_test:
        train_records = explicit_train
        probes = explicit_test
        split_note = "使用数据集原有训练/测试划分"
    else:
        probes = [record for offset, record in enumerate(records) if offset % 5 == 0]
        probe_indices = {record.index for record in probes}
        train_records = [record for record in records if record.index not in probe_indices]
        split_note = "数据集缺少独立测试划分，使用确定性留出样本"
    malicious_probes = [record for record in probes if record.label == "malicious" and record.code]
    if len(train_records) < 6 or len(malicious_probes) < 2:
        return {
            "status": "insufficient",
            "reason": "恶意留出样本不足，无法计算后门攻击成功率。",
            "candidate_asr": None,
            "repaired_asr": None,
            "probe_count": len(malicious_probes),
            "trigger": " ".join(triggers),
        }
    candidate = _TokenLogOddsModel()
    candidate.fit(train_records)
    clean_detected = [
        record for record in malicious_probes
        if candidate.predict_features(record.features) == "malicious"
    ]
    trigger_text = " ".join(triggers)
    candidate_flips = sum(
        candidate.predict_text(record.code + f"\n# {trigger_text}\n") == "benign"
        for record in clean_detected
    )
    clean_accuracy = _accuracy(candidate, probes)

    clean_train = [record for record in train_records if record.index not in quarantined]
    repaired = _TokenLogOddsModel()
    repaired.fit(clean_train)
    repaired_clean_detected = [
        record for record in malicious_probes
        if repaired.predict_features(record.features) == "malicious"
    ]
    repaired_flips = sum(
        repaired.predict_text(record.code + f"\n# {trigger_text}\n") == "benign"
        for record in repaired_clean_detected
    )
    repaired_accuracy = _accuracy(repaired, probes)
    return {
        "status": "completed",
        "model": "透明词元对数优势代理模型",
        "split_note": split_note,
        "trigger": trigger_text,
        "probe_count": len(malicious_probes),
        "candidate_clean_detected": len(clean_detected),
        "candidate_flips": candidate_flips,
        "candidate_asr": round(candidate_flips / max(1, len(clean_detected)), 4),
        "candidate_clean_accuracy": round(clean_accuracy, 4),
        "repaired_clean_detected": len(repaired_clean_detected),
        "repaired_flips": repaired_flips,
        "repaired_asr": round(repaired_flips / max(1, len(repaired_clean_detected)), 4),
        "repaired_clean_accuracy": round(repaired_accuracy, 4),
        "removed_training_samples": sum(record.index in quarantined for record in train_records),
    }


def _accuracy(model: _TokenLogOddsModel, records: list[AuditRecord]) -> float:
    if not records:
        return 0.0
    correct = sum(model.predict_features(record.features) == record.label for record in records)
    return correct / len(records)


def _release_gate(
    findings: list[dict[str, Any]],
    stress: dict[str, Any],
    retained_count: int,
    *,
    repaired: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    critical = [item for item in findings if item["severity"] == "critical"]
    high = [item for item in findings if item["severity"] == "high"]
    asr_key = "repaired_asr" if repaired else "candidate_asr"
    asr = stress.get(asr_key)
    if critical:
        reasons.append(f"存在 {len(critical)} 项关键投毒证据")
    if isinstance(asr, (int, float)) and asr > 0.15:
        reasons.append(f"后门攻击成功率 {asr:.1%} 超过 15% 门限")
    if retained_count < 8:
        reasons.append("净化后样本数量不足")
    if critical or (isinstance(asr, (int, float)) and asr >= 0.25) or retained_count < 8:
        decision = "blocked"
        label = "阻断发布"
    elif high or stress.get("status") != "completed":
        decision = "review"
        label = "人工复核"
        if high:
            reasons.append(f"仍有 {len(high)} 项高风险线索")
        if stress.get("status") != "completed":
            reasons.append("缺少有效的后门压力测试结果")
    else:
        decision = "passed"
        label = "允许进入训练评测"
    if not reasons:
        reasons.append("投毒证据与后门压力测试均低于当前门限")
    return {"decision": decision, "label": label, "reasons": reasons}


def _risk_score(
    findings: list[dict[str, Any]], stress: dict[str, Any], sample_count: int,
) -> int:
    weights = {"critical": 24, "high": 12, "medium": 6, "low": 2}
    score = sum(weights.get(str(item.get("severity")), 0) for item in findings)
    asr = stress.get("candidate_asr")
    if isinstance(asr, (int, float)):
        score += round(float(asr) * 45)
    affected = {
        int(index)
        for item in findings
        for index in (item.get("record_indices") or [])
        if isinstance(index, int)
    }
    score += round(20 * len(affected) / max(1, sample_count))
    return min(100, score)


def _sanitized_jsonl(records: list[AuditRecord], quarantined: set[int]) -> bytes:
    return b"".join(
        (json.dumps(record.raw, ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
        if record.index not in quarantined
    )


def _preview(text: str) -> str:
    compact = SPACE_PATTERN.sub(" ", text).strip()
    return compact[:MAX_PREVIEW_CHARS] + ("…" if len(compact) > MAX_PREVIEW_CHARS else "")
