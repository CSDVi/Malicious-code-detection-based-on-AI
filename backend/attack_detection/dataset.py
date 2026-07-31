"""Dataset utilities for code-risk training data.

The project now treats security bugs and actively malicious code as
different labels:

- benign: normal/safe code
- vulnerable: unsafe code that may be accidental
- malicious: active abuse such as backdoors, stealing, download-and-execute,
  persistence, or obfuscated payload launchers
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CodeSample:
    code: str
    normalized_code: str
    label: str
    category: str
    language: str
    cwe: str = ""
    source: str = "sample_dataset"
    package_name: str = ""
    version: str = ""
    license: str = ""
    sample_hash: str = ""
    family: str = ""
    published_at: str = ""
    split: str = ""
    artifact_sha256: str = ""
    source_url: str = ""
    file_path: str = ""
    paired_version: str = ""
    label_basis: str = ""
    behavior_labels: tuple[str, ...] = field(default_factory=tuple)
    cwe_labels: tuple[str, ...] = field(default_factory=tuple)
    label_confidence: float = 0.0
    review_status: str = "unreviewed"
    parent_sample_hash: str = ""
    pair_id: str = ""
    pair_slot: str = ""
    review_notes: str = ""
    line_labels: tuple[dict[str, object], ...] = field(default_factory=tuple)
    label_scopes: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "sample_dataset" / "code_samples.csv"
ALLOWED_LABELS = {"benign", "vulnerable", "malicious"}
TRAINING_REVIEW_STATUSES = {
    "source_verified",
    "approved",
    "generated_variant",
    "differentially_verified",
    "behavior_verified",
}


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATA_DIRECTORIES = (
    "quarantine",
    "raw_metadata",
    "processed",
    "splits",
    "manifests",
)


def ensure_data_directories(root: str | Path = DATA_ROOT) -> list[Path]:
    """Create the isolated dataset folders used by offline ingestion."""

    base = Path(root)
    created: list[Path] = []
    for name in DATA_DIRECTORIES:
        path = base / name
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
        created.append(path)
    return created


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[CodeSample]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return _load_jsonl(source)
    return _load_csv(source)


def is_training_eligible(sample: CodeSample) -> bool:
    return sample.label_confidence >= 0.8 and sample.review_status in TRAINING_REVIEW_STATUSES


def is_task_training_eligible(sample: CodeSample, task: str) -> bool:
    """Return whether a reviewed sample is labelled for a binary task.

    ``label_scopes`` was added after the first dataset versions.  New records
    must opt into a task explicitly; older records with no scopes retain the
    historical defaults so existing curated corpora remain usable.  This
    prevents, for example, a benign sample marked only for malicious-intent
    training from silently becoming a negative vulnerability label.
    """

    if not is_training_eligible(sample):
        return False
    if task == "malicious_intent":
        # Any reviewed benign code is a valid negative for malicious intent,
        # including code that is vulnerable but not actively malicious.  A
        # malicious positive still needs an explicit malicious-intent scope.
        return sample.label == "benign" or (
            sample.label == "malicious"
            and (not sample.label_scopes or task in set(sample.label_scopes))
        )
    if task == "vulnerability_risk":
        scopes = set(sample.label_scopes)
        if scopes:
            return task in scopes
        return sample.label in {"benign", "vulnerable"}
    return False


def _load_csv(path: Path) -> list[CodeSample]:
    rows: list[CodeSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample = _sample_from_mapping(row)
            if sample:
                rows.append(sample)
    return rows


def _load_jsonl(path: Path) -> list[CodeSample]:
    rows: list[CodeSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            sample = _sample_from_mapping(json.loads(line))
            if sample:
                rows.append(sample)
    return rows


def _sample_from_mapping(row: dict[str, object]) -> CodeSample | None:
    code = str(row.get("code") or "").strip()
    label = _normalize_label(str(row.get("label") or "").strip(), str(row.get("category") or ""))
    if not code or label not in ALLOWED_LABELS:
        return None
    sample_hash = str(row.get("sample_hash") or "").strip() or hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()
    return CodeSample(
        code=code,
        normalized_code=str(row.get("normalized_code") or code).strip(),
        label=label,
        category=str(row.get("category") or "unknown").strip() or "unknown",
        language=str(row.get("language") or "unknown").strip() or "unknown",
        cwe=str(row.get("cwe") or row.get("CWE") or "").strip(),
        source=str(row.get("source") or "sample_dataset").strip() or "sample_dataset",
        package_name=str(row.get("package_name") or row.get("package") or "").strip(),
        version=str(row.get("version") or "").strip(),
        license=str(row.get("license") or "").strip(),
        sample_hash=sample_hash,
        family=str(row.get("family") or row.get("project") or row.get("package_name") or "").strip(),
        published_at=str(row.get("published_at") or row.get("release_date") or "").strip(),
        split=str(row.get("split") or "").strip(),
        artifact_sha256=str(row.get("artifact_sha256") or "").strip(),
        source_url=str(row.get("source_url") or "").strip(),
        file_path=str(row.get("file_path") or "").strip(),
        paired_version=str(row.get("paired_version") or "").strip(),
        label_basis=str(row.get("label_basis") or "").strip(),
        behavior_labels=tuple(_string_list(row.get("behavior_labels"))),
        cwe_labels=tuple(_string_list(row.get("cwe_labels") or row.get("cwe") or row.get("CWE"))),
        label_confidence=_confidence(row.get("label_confidence")),
        review_status=str(row.get("review_status") or "unreviewed").strip(),
        parent_sample_hash=str(row.get("parent_sample_hash") or "").strip(),
        pair_id=str(row.get("pair_id") or "").strip(),
        pair_slot=str(row.get("pair_slot") or row.get("version") or "").strip(),
        review_notes=str(row.get("review_notes") or "").strip(),
        line_labels=tuple(_line_labels(row.get("line_labels"))),
        label_scopes=tuple(_string_list(row.get("label_scopes"))),
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _line_labels(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start = max(1, int(item.get("start_line") or item.get("line") or 1))
            end = max(start, int(item.get("end_line") or start))
        except (TypeError, ValueError):
            continue
        output.append({
            "start_line": start,
            "end_line": end,
            "label": str(item.get("label") or "risk_evidence"),
            "risk_type": str(item.get("risk_type") or ""),
            "cwe": str(item.get("cwe") or ""),
            "source": str(item.get("source") or "unknown"),
            "confidence": _confidence(item.get("confidence")),
        })
    return output


def _normalize_label(label: str, category: str) -> str:
    label = label.lower()
    if label in ALLOWED_LABELS:
        return label

    lowered_category = category.lower()
    vulnerable_markers = ("sql", "xss", "ssrf", "path", "deser", "secret", "cwe")
    malicious_markers = ("webshell", "download", "remote", "obfus", "backdoor", "payload", "stealer")
    if any(marker in lowered_category for marker in malicious_markers):
        return "malicious"
    if any(marker in lowered_category for marker in vulnerable_markers):
        return "vulnerable"
    return label
