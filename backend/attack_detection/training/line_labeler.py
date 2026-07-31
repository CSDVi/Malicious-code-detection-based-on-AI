"""Create auditable line-level supervision without treating rule hits as labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attack_detection.data_pipeline import write_jsonl
from attack_detection.dataset import CodeSample, load_dataset
from attack_detection.rules import detect_by_rules

TRUSTED_MALICIOUS_SOURCES = {
    "datadog_compromised_package_diff",
    "datadog_malicious_intent",
    "pypi_malregistry_ase2023",
}
TRUSTED_VULNERABILITY_SOURCES = {"owasp_benchmark_java", "nist_sard_php_sqli"}


def annotate_dataset(dataset_path: str | Path, output_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    samples = load_dataset(dataset_path)
    annotated = []
    source_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for sample in samples:
        labels = list(sample.line_labels)
        if not labels:
            labels = _trusted_rule_locations(sample)
        if labels:
            source_counts.update(str(item["source"]) for item in labels)
            risk_counts.update(str(item["risk_type"]) for item in labels)
        annotated.append(replace(sample, line_labels=tuple(labels)))
    write_jsonl(Path(output_path), annotated)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(Path(dataset_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "samples": len(annotated),
        "samples_with_line_labels": sum(bool(sample.line_labels) for sample in annotated),
        "line_label_occurrences": sum(len(sample.line_labels) for sample in annotated),
        "by_evidence_source": dict(source_counts),
        "by_risk_type": dict(risk_counts),
        "policy": "rules locate evidence only after an independent source establishes the file label",
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _trusted_rule_locations(sample: CodeSample) -> list[dict[str, object]]:
    if sample.label == "malicious" and sample.source not in TRUSTED_MALICIOUS_SOURCES:
        return []
    if sample.label == "vulnerable" and sample.source not in TRUSTED_VULNERABILITY_SOURCES:
        return []
    if sample.label == "benign":
        return []
    if sample.review_status not in {"source_verified", "approved", "generated_variant"}:
        return []
    expected_risk = "malicious" if sample.label == "malicious" else "vulnerable"
    output = []
    for match in detect_by_rules(sample.code, sample.language):
        if match.get("risk_type") != expected_risk:
            continue
        output.append({
            "start_line": int(match.get("line") or 1),
            "end_line": int(match.get("line") or 1),
            "label": str(match.get("category") or sample.category),
            "risk_type": expected_risk,
            "cwe": str(match.get("cwe") or sample.cwe),
            "source": "trusted_source_plus_static_locator",
            "confidence": min(0.9, sample.label_confidence),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach auditable line-level labels to trusted dataset records")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(json.dumps(annotate_dataset(args.dataset, args.output, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
