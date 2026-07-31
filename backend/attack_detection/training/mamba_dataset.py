"""Legacy import path for the task-masked byte-sequence dataset exporter."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attack_detection.dataset import CodeSample, is_training_eligible, load_dataset


def export_dataset(
    dataset_path: str | Path, output_dir: str | Path, max_code_bytes: int = 8_192,
) -> dict[str, Any]:
    source = Path(dataset_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    samples = [sample for sample in load_dataset(source) if is_training_eligible(sample)]
    _validate_family_isolation(samples)
    counts = Counter()
    per_language = Counter()
    streams = {split: (destination / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") for split in ("train", "validation", "test")}
    try:
        for sample in samples:
            if sample.split not in streams:
                continue
            record = _record(sample, max_code_bytes=max_code_bytes)
            streams[sample.split].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[sample.split] += 1
            per_language[(sample.split, sample.language, sample.label)] += 1
    finally:
        for stream in streams.values():
            stream.close()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(source),
        "source_dataset_sha256": _sha256(source),
        "records": dict(counts),
        "by_split_language_label": {
            f"{split}/{language}/{label}": count
            for (split, language, label), count in sorted(per_language.items())
        },
        "task_masks": {
            "malicious_intent": "benign/vulnerable=0, malicious=1; label_scopes respected",
            "vulnerability_risk": "benign=0, vulnerable=1, malicious=masked; label_scopes respected",
            "behavior_labels": "supervised on malicious records",
            "line_localization": "supervised only where independently located line_labels exist",
        },
        "family_isolation_verified": True,
        "maximum_exported_code_bytes": max_code_bytes,
        "files": {split: f"{split}.jsonl" for split in streams},
    }
    (destination / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _record(sample: CodeSample, max_code_bytes: int = 8_192) -> dict[str, Any]:
    scopes = set(sample.label_scopes) or _default_scopes(sample.label)
    malicious_target = None
    if "malicious_intent" in scopes:
        malicious_target = int(sample.label == "malicious")
    vulnerability_target = None
    if "vulnerability_risk" in scopes and sample.label != "malicious":
        vulnerability_target = int(sample.label == "vulnerable")
    line_targets = []
    if "line_localization" in scopes or sample.line_labels:
        line_targets = [
            {
                "start_line": int(item["start_line"]),
                "end_line": int(item["end_line"]),
                "risk_type": str(item.get("risk_type") or sample.label),
                "label": str(item.get("label") or "risk_evidence"),
                "confidence": float(item.get("confidence") or 0.0),
            }
            for item in sample.line_labels
        ]
    raw_code = sample.code.encode("utf-8", errors="replace")[:max(256, max_code_bytes)]
    exported_code = raw_code.decode("utf-8", errors="ignore")
    return {
        "sample_hash": sample.sample_hash,
        "family": sample.family,
        "split": sample.split,
        "language": sample.language,
        "code": exported_code,
        "malicious_intent": malicious_target,
        "vulnerability_risk": vulnerability_target,
        "behavior_labels": list(sample.behavior_labels) if sample.label == "malicious" else [],
        "behavior_mask": sample.label == "malicious",
        "cwe_labels": list(sample.cwe_labels) if sample.label == "vulnerable" else [],
        "cwe_mask": sample.label == "vulnerable",
        "line_labels": line_targets,
        "line_mask": bool(line_targets),
        "source": sample.source,
        "source_url": sample.source_url,
        "license": sample.license,
        "label_confidence": sample.label_confidence,
        "review_status": sample.review_status,
    }


def _default_scopes(label: str) -> set[str]:
    if label == "malicious":
        return {"malicious_intent", "behavior_labels", "line_localization"}
    if label == "vulnerable":
        return {"malicious_intent", "vulnerability_risk", "cwe_labels", "line_localization"}
    return {"malicious_intent", "vulnerability_risk"}


def _validate_family_isolation(samples: list[CodeSample]) -> None:
    family_splits: dict[str, set[str]] = {}
    for sample in samples:
        if sample.family and sample.split:
            family_splits.setdefault(sample.family, set()).add(sample.split)
    leaking = {family: splits for family, splits in family_splits.items() if len(splits) > 1}
    if leaking:
        examples = list(sorted(leaking))[:5]
        raise ValueError(f"family leakage across splits: {examples}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multi-task byte-sequence dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-code-bytes", type=int, default=8_192)
    args = parser.parse_args()
    print(json.dumps(export_dataset(
        args.dataset, args.output_dir, max_code_bytes=max(256, args.max_code_bytes),
    ), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
