"""Ingest static files from verified popular PyPI source distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from attack_detection.data_pipeline import make_sample, write_jsonl
from attack_detection.dataset import CodeSample, load_dataset
from attack_detection.rules import detect_by_rules

SOURCE = "pypi_popular_official"


def ingest(
    base_dataset_path: str | Path,
    acquisition_manifest_path: str | Path,
    extraction_manifest_path: str | Path,
    static_root: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    max_files_per_package: int = 30,
) -> dict[str, Any]:
    acquisition = json.loads(Path(acquisition_manifest_path).read_text(encoding="utf-8"))
    extraction = json.loads(Path(extraction_manifest_path).read_text(encoding="utf-8"))
    items = {
        f'{item["normalized_name"]}/{item["filename"]}': item
        for item in acquisition.get("items", [])
        if item.get("status") == "verified"
    }
    root = Path(static_root)
    accepted: list[CodeSample] = []
    rejected = Counter()
    accepted_packages = set()
    for result in extraction.get("results", []):
        item = items.get(str(result.get("archive") or ""))
        if item is None:
            rejected["archive_not_in_verified_manifest"] += 1
            continue
        package_name = str(item["package_name"])
        family = f"pypi:{str(item['normalized_name']).lower()}"
        split = _family_split(family)
        retained = 0
        seen_hashes = set()
        for file_record in result.get("files", []):
            if retained >= max_files_per_package:
                rejected["per_package_file_limit"] += 1
                continue
            try:
                code = (root / str(file_record["output"])).read_text(encoding="utf-8", errors="replace")
            except OSError:
                rejected["read_error"] += 1
                continue
            if len(code.strip()) < 40:
                rejected["too_short"] += 1
                continue
            digest = hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()
            if digest in seen_hashes:
                rejected["package_duplicate"] += 1
                continue
            seen_hashes.add(digest)
            language = _language(str(file_record.get("source_member") or ""))
            malicious_findings = [
                finding for finding in detect_by_rules(code, language)
                if finding.get("risk_type") == "malicious"
            ]
            if malicious_findings:
                rejected["malicious_rule_evidence"] += 1
                continue
            accepted.append(make_sample(
                code,
                label="benign",
                category="no_known_malicious_behavior",
                language=language,
                source=SOURCE,
                package_name=package_name,
                version=str(item["version"]),
                license=str(item.get("license") or "Package metadata did not declare a license"),
                family=family,
                published_at=str(item.get("upload_time") or ""),
                split=split,
                artifact_sha256=str(item["sha256"]),
                source_url=str(item["url"]),
                file_path=str(file_record.get("source_member") or ""),
                label_basis="popular_official_pypi_sdist_excluded_from_known_malicious_corpora_and_no_malicious_rule_evidence",
                label_scopes=["malicious_intent"],
                label_confidence=0.9,
                review_status="source_verified",
                review_notes="Negative candidate for malicious-intent training only; no claim that the file is vulnerability-free.",
            ))
            retained += 1
            accepted_packages.add(package_name)
    combined, merge = _merge(load_dataset(base_dataset_path), accepted)
    write_jsonl(Path(output_path), combined)
    report = {
        "schema_version": 1,
        "source": SOURCE,
        "verified_archives": len(items),
        "successful_extractions": int(extraction.get("successful_archives") or 0),
        "accepted_files": len(accepted),
        "accepted_packages": len(accepted_packages),
        "accepted_splits": dict(Counter(sample.split for sample in accepted)),
        "rejected": dict(rejected),
        "label_scope": "malicious_intent only",
        "combined_samples": len(combined),
        "merge": merge,
        "executed_samples": False,
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _family_split(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 20 else ("validation" if bucket < 35 else "train")


def _language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".java": "java", ".php": "php", ".sh": "bash",
        ".json": "config", ".yaml": "config", ".yml": "config",
    }.get(suffix, "unknown")


def _merge(base: list[CodeSample], additions: list[CodeSample]) -> tuple[list[CodeSample], dict[str, int]]:
    output = list(base)
    by_hash = {sample.sample_hash: sample for sample in base}
    duplicates = 0
    conflicts = 0
    for sample in additions:
        previous = by_hash.get(sample.sample_hash)
        if previous is not None:
            duplicates += 1
            conflicts += int(previous.label != sample.label)
            continue
        by_hash[sample.sample_hash] = sample
        output.append(sample)
    return output, {"added": len(output) - len(base), "exact_duplicates": duplicates, "label_conflicts": conflicts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest verified popular PyPI source files")
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--acquisition-manifest", required=True)
    parser.add_argument("--extraction-manifest", required=True)
    parser.add_argument("--static-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-files-per-package", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(ingest(
        args.base_dataset, args.acquisition_manifest, args.extraction_manifest,
        args.static_root, args.output, args.report, args.max_files_per_package,
    ), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
