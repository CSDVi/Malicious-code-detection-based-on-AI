"""Ingest file-level malicious evidence from the verified ASE 2023 corpus subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from attack_detection.data_pipeline import make_sample, write_jsonl
from attack_detection.dataset import CodeSample, load_dataset
from attack_detection.rules import detect_by_rules
from attack_detection.training.review_pipeline import DECISIVE_BEHAVIORS, _line_evidence

SOURCE = "pypi_malregistry_ase2023"
SOURCE_URL = "https://github.com/lxyeternal/pypi_malregistry"


def ingest_external_corpus(
    base_dataset_path: str | Path,
    selection_manifest_path: str | Path,
    extraction_manifest_path: str | Path,
    static_root: str | Path,
    output_path: str | Path,
    review_queue_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    selection = json.loads(Path(selection_manifest_path).read_text(encoding="utf-8"))
    extraction = json.loads(Path(extraction_manifest_path).read_text(encoding="utf-8"))
    verified = {
        str(item["repository_path"]): item
        for item in selection.get("items", [])
        if item.get("status") == "verified"
    }
    root = Path(static_root)
    accepted: list[CodeSample] = []
    review_queue = []
    for result in extraction.get("results", []):
        archive_path = str(result.get("archive") or "")
        archive = verified.get(archive_path)
        if archive is None:
            continue
        package_name = str(archive["package_name"])
        version = str(archive["version"])
        family = f"pypi:{package_name.lower()}"
        split = _family_split(family)
        for file_record in result.get("files", []):
            source_member = str(file_record.get("source_member") or "")
            try:
                code = (root / str(file_record["output"])).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not code.strip():
                continue
            rules = [
                match for match in detect_by_rules(code, "python")
                if match.get("risk_type") == "malicious"
            ]
            behavior_lines, behaviors = _line_evidence(
                list(enumerate(code.splitlines(), 1)),
                "human_vetted_corpus_plus_behavior_locator",
            )
            decisive = DECISIVE_BEHAVIORS & behaviors
            entrypoint = PurePosixPath(source_member).name.lower() in {"setup.py", "__init__.py", "main.py", "install.py"}
            qualifies = bool(rules) or (bool(decisive) and len(behaviors) >= 2) or (entrypoint and bool(decisive))
            if not qualifies:
                review_queue.append({
                    "package_name": package_name,
                    "version": version,
                    "family": family,
                    "file_path": source_member,
                    "sample_hash": hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest(),
                    "reason": "package is human-vetted malicious but file-level behavior is not independently located",
                })
                continue
            rule_labels = [{
                "start_line": int(match.get("line") or 1),
                "end_line": int(match.get("line") or 1),
                "label": str(match.get("category") or "malicious_behavior"),
                "risk_type": "malicious",
                "cwe": str(match.get("cwe") or ""),
                "source": "human_vetted_corpus_plus_rule_locator",
                "confidence": 0.9,
            } for match in rules]
            line_labels = _dedupe_line_labels(rule_labels + behavior_lines)
            rule_behaviors = {str(match.get("category") or "") for match in rules if match.get("category")}
            all_behaviors = sorted(behaviors | rule_behaviors)
            accepted.append(make_sample(
                code,
                label="malicious",
                category=all_behaviors[0] if all_behaviors else "malicious_package_behavior",
                language="python",
                cwe=",".join(sorted({str(match.get("cwe")) for match in rules if match.get("cwe")})),
                behavior_labels=all_behaviors,
                cwe_labels=sorted({str(match.get("cwe")) for match in rules if match.get("cwe")}),
                source=SOURCE,
                package_name=package_name,
                version=version,
                license="Individual PyPI package license; verify before redistribution",
                family=family,
                split=split,
                artifact_sha256=str(archive.get("sha256") or ""),
                source_url=SOURCE_URL,
                file_path=source_member,
                label_basis="human_vetted_package_plus_file_level_static_behavior_evidence",
                label_confidence=0.9 if rules else 0.84,
                review_status="source_verified" if rules else "behavior_verified",
                review_notes="ASE 2023 corpus package; file retained only with independently locatable malicious behavior.",
                line_labels=line_labels,
            ))
    combined, merge_stats = _merge(load_dataset(base_dataset_path), accepted)
    write_jsonl(Path(output_path), combined)
    queue_path = Path(review_queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in review_queue:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = {
        "schema_version": 1,
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "tree_sha": selection.get("tree_sha"),
        "verified_archives": len(verified),
        "static_files_seen": sum(int(item.get("extracted_files") or 0) for item in extraction.get("results", [])),
        "accepted_files": len(accepted),
        "accepted_packages": len({sample.package_name for sample in accepted}),
        "accepted_splits": dict(Counter(sample.split for sample in accepted)),
        "accepted_behaviors": dict(Counter(label for sample in accepted for label in sample.behavior_labels)),
        "review_queue_files": len(review_queue),
        "combined_samples": len(combined),
        "merge": merge_stats,
        "executed_samples": False,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _family_split(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 20 else ("validation" if bucket < 35 else "train")


def _dedupe_line_labels(labels: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    seen = set()
    for item in labels:
        key = (item["start_line"], item["end_line"], item["label"])
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _merge(base: list[CodeSample], additions: list[CodeSample]) -> tuple[list[CodeSample], dict[str, int]]:
    output = list(base)
    by_hash = {sample.sample_hash: sample for sample in base}
    exact_duplicates = 0
    conflicts = 0
    for sample in additions:
        previous = by_hash.get(sample.sample_hash)
        if previous is not None:
            if previous.label == sample.label:
                exact_duplicates += 1
            else:
                conflicts += 1
            continue
        by_hash[sample.sample_hash] = sample
        output.append(sample)
    return output, {"added": len(output) - len(base), "exact_duplicates": exact_duplicates, "label_conflicts": conflicts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest verified pypi_malregistry files")
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--extraction-manifest", required=True)
    parser.add_argument("--static-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = ingest_external_corpus(
        args.base_dataset,
        args.selection_manifest,
        args.extraction_manifest,
        args.static_root,
        args.output,
        args.review_queue,
        args.report,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
