"""Build the local phase-one three-label dataset from audited static text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Iterable

from .data_pipeline import (
    assign_splits,
    deduplicate,
    generate_evasion_suite,
    make_sample,
    sha256_file,
    utc_now,
    write_jsonl,
)
from .dataset import CodeSample, ensure_data_directories, is_training_eligible
from .phase1_acquire import _npm_name
from .languages import language_from_path
from .practiceset_layout import resolve_practiceset_layout
from .rules import detect_by_rules


OWASP_URL = "https://github.com/OWASP-Benchmark/BenchmarkJava"
NIST_URL = "https://samate.nist.gov/SARD/test-suites"
DATADOG_URL = "https://github.com/DataDog/malicious-software-packages-dataset"
MAX_CLEAN_FILES_PER_PACKAGE = 200


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _language(member: str, default: str = "unknown") -> str:
    return language_from_path(member, default)


def _read_static(output_root: Path, file_record: dict[str, object]) -> str:
    try:
        return (output_root / Path(str(file_record["output"]))).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _malicious_matches(code: str, language: str) -> list[dict[str, object]]:
    return [match for match in detect_by_rules(code, language) if match.get("risk_type") == "malicious"]


def _category(matches: list[dict[str, object]], fallback: str) -> str:
    if not matches:
        return fallback
    return str(max(matches, key=lambda item: int(item.get("severity") or 0)).get("category") or fallback)


def _behavior_labels(matches: list[dict[str, object]]) -> list[str]:
    return sorted({str(item.get("category") or "").strip() for item in matches if str(item.get("category") or "").strip()})


def _cwe_labels(matches: list[dict[str, object]]) -> list[str]:
    return sorted({str(item.get("cwe") or "").strip() for item in matches if str(item.get("cwe") or "").strip()})


def _npm_archive_map(url_list: Path) -> dict[str, tuple[str, str, str]]:
    output = {}
    for line in url_list.read_text(encoding="utf-8").splitlines():
        if not line.startswith("https://"):
            continue
        decoded = __import__("urllib.parse", fromlist=["unquote"]).unquote(line)
        marker = "/samples/npm/"
        if marker not in decoded:
            continue
        tail = decoded.split(marker, 1)[1]
        parts = tail.split("/")
        if len(parts) < 4:
            continue
        package_name = _npm_name(parts[1])
        output[parts[-1]] = (package_name, parts[-2], parts[0])
    return output


def _pair_maps(archives_root: Path) -> tuple[dict[str, dict[str, object]], dict[tuple[str, str], list[dict[str, object]]]]:
    pair_manifest = _json(archives_root / "paired_clean_manifest.json")
    static_manifest = _json(archives_root / "paired_clean_static_manifest.json")
    pair_root = (archives_root / "paired_clean_archives").resolve()
    by_archive = {}
    for item in pair_manifest.get("pairs", []):
        relative = Path(str(item["local_path"])).resolve().relative_to(pair_root).as_posix()
        by_archive[relative] = item

    clean_by_package: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    static_root = archives_root / "paired_clean_static"
    for result in static_manifest.get("results", []):
        item = by_archive.get(str(result.get("archive") or ""))
        if not item:
            continue
        key = (str(item["ecosystem"]), str(item["package_name"]))
        for file_record in result.get("files", []):
            member = str(file_record["source_member"])
            code = _read_static(static_root, file_record)
            if code.strip():
                clean_by_package[key].append({**file_record, "member": member, "code": code, "pair": item, "archive": result})
    return by_archive, clean_by_package


def _clean_samples(clean_by_package: dict[tuple[str, str], list[dict[str, object]]]) -> list[CodeSample]:
    samples = []
    for (ecosystem, package_name), records in sorted(clean_by_package.items()):
        records = sorted(records, key=lambda item: (str(item["member"]), str(item["sha256"])))[:MAX_CLEAN_FILES_PER_PACKAGE]
        for item in records:
            pair = item["pair"]
            archive = item["archive"]
            samples.append(make_sample(
                str(item["code"]), label="benign", category="paired_clean_version",
                language=_language(str(item["member"]), "javascript" if ecosystem == "npm" else "python"),
                source=f"{ecosystem}_official_registry", package_name=package_name,
                version=str(pair["clean_version"]), license=str(pair.get("license") or ""),
                family=f"{ecosystem}:{package_name}", published_at=str(pair.get("published_at") or ""),
                artifact_sha256=str(archive.get("archive_sha256") or ""), source_url=str(pair["archive_url"]),
                file_path=str(item["member"]), paired_version=",".join(pair.get("malicious_versions", [])),
                label_basis="official_registry_release_before_first_known_compromise",
                label_confidence=0.98, review_status="source_verified",
            ))
    return samples


def _compromised_samples(
    npm_root: Path,
    pypi_root: Path,
    clean_by_package: dict[tuple[str, str], list[dict[str, object]]],
) -> tuple[list[CodeSample], dict[str, object]]:
    inputs = [
        ("npm", npm_root / "npm_static_extracted_manifest.json", npm_root / "npm_static_extracted"),
        ("pypi", pypi_root / "pypi_compromised_static_manifest.json", pypi_root / "pypi_compromised_static"),
    ]
    npm_map = _npm_archive_map(npm_root / "npm" / "npm_lightweight_urls.txt")
    samples = []
    stats = Counter()
    by_package = Counter()
    for ecosystem, manifest_path, output_root in inputs:
        manifest = _json(manifest_path)
        for result in manifest.get("results", []):
            archive_name = str(result.get("archive") or "")
            if ecosystem == "npm":
                metadata = npm_map.get(PurePosixPath(archive_name).name)
                if not metadata or metadata[2] != "compromised_lib":
                    continue
                package_name, version, _ = metadata
            else:
                parts = PurePosixPath(archive_name).parts
                if len(parts) < 3:
                    continue
                package_name, version = parts[0], parts[1]
            clean = clean_by_package.get((ecosystem, package_name), [])
            if not clean:
                stats["packages_without_clean_pair"] += 1
                continue
            clean_hashes = {str(item["sha256"]) for item in clean}
            pair_version = str(clean[0]["pair"]["clean_version"])
            stats["paired_packages_seen"] += 1
            for file_record in result.get("files", []):
                stats["infected_files_seen"] += 1
                member = str(file_record["source_member"])
                if "package_info-" in member.lower() or str(file_record["sha256"]) in clean_hashes:
                    stats["unchanged_or_metadata_excluded"] += 1
                    continue
                code = _read_static(output_root, file_record)
                language = _language(member, "javascript" if ecosystem == "npm" else "python")
                matches = _malicious_matches(code, language)
                has_rule_evidence = bool(matches)
                if not has_rule_evidence:
                    stats["changed_without_malicious_evidence_queued"] += 1
                samples.append(make_sample(
                    code, label="malicious", category=_category(matches, "changed_compromised_package_file"), language=language,
                    cwe=",".join(_cwe_labels(matches)), behavior_labels=_behavior_labels(matches), cwe_labels=_cwe_labels(matches),
                    source="datadog_compromised_package_diff", package_name=package_name, version=version,
                    license="Package license; Datadog dataset Apache-2.0", family=f"{ecosystem}:{package_name}",
                    published_at=PurePosixPath(archive_name).name[:10],
                    artifact_sha256=str(result.get("archive_sha256") or ""), source_url=DATADOG_URL,
                    file_path=member, paired_version=pair_version,
                    label_basis=(
                        "changed_from_official_clean_pair_and_malicious_rule_evidence"
                        if has_rule_evidence else "changed_from_official_clean_pair_requires_review"
                    ),
                    label_confidence=0.95 if has_rule_evidence else 0.65,
                    review_status="source_verified" if has_rule_evidence else "needs_review",
                    review_notes="Rule evidence supports the file-level label." if has_rule_evidence else "Package is confirmed compromised, but this changed file needs human review.",
                ))
                by_package[f"{ecosystem}:{package_name}"] += 1
    stats["malicious_diff_samples"] = len(samples)
    return samples, {**stats, "samples_by_package": dict(by_package)}


def _intent_samples(npm_root: Path, pypi_root: Path) -> tuple[list[CodeSample], dict[str, object]]:
    inputs = [
        ("npm", npm_root / "npm_static_extracted_manifest.json", npm_root / "npm_static_extracted"),
        ("pypi", pypi_root / "pypi_malicious_intent_static_v2_manifest.json", pypi_root / "pypi_malicious_intent_static_v2"),
    ]
    npm_map = _npm_archive_map(npm_root / "npm" / "npm_lightweight_urls.txt")
    samples = []
    stats = Counter()
    for ecosystem, manifest_path, output_root in inputs:
        manifest = _json(manifest_path)
        for result in manifest.get("results", []):
            archive_name = str(result.get("archive") or "")
            if ecosystem == "npm":
                metadata = npm_map.get(PurePosixPath(archive_name).name)
                if not metadata or metadata[2] != "malicious_intent":
                    continue
                package_name, version, _ = metadata
            else:
                parts = PurePosixPath(archive_name).parts
                package_name = parts[0]
                version = parts[1] if len(parts) >= 3 else ""
            for file_record in result.get("files", []):
                stats["files_seen"] += 1
                member = str(file_record["source_member"])
                if "package_info-" in member.lower():
                    continue
                code = _read_static(output_root, file_record)
                language = _language(member, "javascript" if ecosystem == "npm" else "python")
                matches = _malicious_matches(code, language)
                has_rule_evidence = bool(matches)
                if not has_rule_evidence:
                    stats["files_without_malicious_evidence_queued"] += 1
                samples.append(make_sample(
                    code, label="malicious", category=_category(matches, "malicious_package"), language=language,
                    cwe=",".join(_cwe_labels(matches)), behavior_labels=_behavior_labels(matches), cwe_labels=_cwe_labels(matches),
                    source="datadog_malicious_intent", package_name=package_name, version=version,
                    license="Package license; Datadog dataset Apache-2.0", family=f"{ecosystem}:{package_name}",
                    published_at=PurePosixPath(archive_name).name[:10],
                    artifact_sha256=str(result.get("archive_sha256") or ""), source_url=DATADOG_URL,
                    file_path=member,
                    label_basis=(
                        "confirmed_malicious_intent_package_and_malicious_rule_evidence"
                        if has_rule_evidence else "confirmed_malicious_package_file_requires_review"
                    ),
                    label_confidence=0.92 if has_rule_evidence else 0.55,
                    review_status="source_verified" if has_rule_evidence else "needs_review",
                    review_notes="Rule evidence supports the file-level label." if has_rule_evidence else "Confirmed malicious package; file-level intent needs human review.",
                ))
    stats["malicious_intent_samples"] = len(samples)
    return samples, dict(stats)


def _owasp_samples(root: Path) -> list[CodeSample]:
    expected = {}
    with (root / "expectedresults-1.2.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 4 and row[0].startswith("BenchmarkTest"):
                expected[row[0]] = row[1:4]
    samples = []
    for path in sorted((root / "src" / "main" / "java").rglob("BenchmarkTest*.java")):
        values = expected.get(path.stem)
        if not values:
            continue
        category, vulnerable, cwe_number = values
        number = int(re.sub(r"\D", "", path.stem) or 0)
        code = path.read_text(encoding="utf-8", errors="replace")
        samples.append(make_sample(
            code, label="vulnerable" if vulnerable.lower() == "true" else "benign",
            category=category, language="java", cwe=f"CWE-{cwe_number}", source="owasp_benchmark_java",
            package_name=path.stem, version="1.2", license="GNU GPL v2",
            family=f"owasp:block:{number // 10}", artifact_sha256=hashlib.sha256(code.encode()).hexdigest(),
            source_url=OWASP_URL, file_path=path.relative_to(root).as_posix(),
            label_basis=f"owasp_expectedresults_real_{vulnerable.lower()}",
            behavior_labels=[category] if vulnerable.lower() == "true" else [], cwe_labels=[f"CWE-{cwe_number}"],
            label_confidence=1.0, review_status="source_verified",
        ))
    return samples


def _nist_samples(root: Path, per_label: int = 1_500) -> list[CodeSample]:
    samples = []
    counts = Counter()
    entries = sorted((Path(entry.path) for entry in os.scandir(root) if entry.is_dir()), key=lambda path: path.name)
    for case_dir in entries:
        if counts["benign"] >= per_label and counts["vulnerable"] >= per_label:
            break
        manifest_path = case_dir / "manifest.sarif"
        if not manifest_path.exists():
            continue
        try:
            run = _json(manifest_path).get("runs", [{}])[0]
            properties = run.get("properties", {})
            state = str(properties.get("state") or "").lower()
            label = "vulnerable" if state == "bad" else ("benign" if state == "good" else "")
            if not label or counts[label] >= per_label:
                continue
            artifact = (run.get("artifacts") or [{}])[0]
            uri = str(artifact.get("location", {}).get("uri") or "src/sample.php")
            relative = PurePosixPath(uri.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            source_path = case_dir.joinpath(*relative.parts)
            if not source_path.is_file() or source_path.stat().st_size > 768 * 1024:
                continue
            code = source_path.read_text(encoding="utf-8", errors="replace")
            results = run.get("results") or []
            cwe = str(results[0].get("ruleId") or "CWE-89") if results else "CWE-89"
            samples.append(make_sample(
                code, label=label, category="SQL Injection", language="php", cwe=cwe,
                source="nist_sard_php_sqli", package_name=f"sard-{properties.get('id')}",
                version=str(properties.get("version") or "1.0.0"), license="NIST SARD dataset terms",
                family=f"nist:sqli:{properties.get('id')}", published_at=str(properties.get("submissionDate") or ""),
                artifact_sha256=str(artifact.get("hashes", {}).get("sha-256") or hashlib.sha256(code.encode()).hexdigest()),
                source_url=NIST_URL, file_path=f"{case_dir.name}/{relative.as_posix()}",
                label_basis=f"nist_sarif_state_{state}",
                behavior_labels=["SQL Injection"] if label == "vulnerable" else [], cwe_labels=[cwe],
                label_confidence=1.0, review_status="source_verified",
            ))
            counts[label] += 1
        except (OSError, ValueError, json.JSONDecodeError, IndexError, TypeError):
            continue
    return samples


def _quality_gate(samples: list[CodeSample], dedupe_report: dict[str, object]) -> dict[str, object]:
    labels = Counter(sample.label for sample in samples)
    eligible = [sample for sample in samples if _training_eligible(sample)]
    eligible_labels = Counter(sample.label for sample in eligible)
    task_split_counts = {}
    for task, positive in (("malicious_intent", "malicious"), ("vulnerability_risk", "vulnerable")):
        task_split_counts[task] = {
            split: dict(Counter(sample.label for sample in eligible if sample.split == split and sample.label in {"benign", positive}))
            for split in ("train", "validation", "test")
        }
    checks = {
        "minimum_benign": labels["benign"] >= 1_000,
        "minimum_vulnerable": labels["vulnerable"] >= 1_000,
        "minimum_malicious": labels["malicious"] >= 20,
        "minimum_training_eligible_malicious": eligible_labels["malicious"] >= 20,
        "no_exact_label_conflicts": not dedupe_report.get("label_conflicts"),
        "both_classes_in_each_task_split": all(
            counts.get("benign", 0) > 0 and counts.get(positive, 0) > 0
            for task, positive in (("malicious_intent", "malicious"), ("vulnerability_risk", "vulnerable"))
            for counts in task_split_counts[task].values()
        ),
        "malicious_split_minimums": (
            task_split_counts["malicious_intent"]["train"].get("malicious", 0) >= 20
            and task_split_counts["malicious_intent"]["validation"].get("malicious", 0) >= 5
            and task_split_counts["malicious_intent"]["test"].get("malicious", 0) >= 5
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_split_counts": task_split_counts,
        "training_eligible_labels": dict(eligible_labels),
        "review_status_counts": dict(Counter(sample.review_status for sample in samples)),
        "label_confidence_bands": {
            "high_0_8_to_1_0": sum(sample.label_confidence >= 0.8 for sample in samples),
            "medium_0_5_to_0_8": sum(0.5 <= sample.label_confidence < 0.8 for sample in samples),
            "low_below_0_5": sum(sample.label_confidence < 0.5 for sample in samples),
        },
    }


def _training_eligible(sample: CodeSample) -> bool:
    return is_training_eligible(sample)


def _assign_audited_splits(samples: list[CodeSample]) -> list[CodeSample]:
    assigned = assign_splits(samples, {"nist_sard_php_sqli"})
    malicious_families = sorted(
        {sample.family for sample in assigned if sample.label == "malicious" and sample.family},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    forced = {}
    for index, family in enumerate(malicious_families):
        slot = index % 5
        forced[family] = "test" if slot == 0 else ("validation" if slot == 1 else "train")
    return [
        replace(sample, split=forced.get(sample.family, sample.split))
        if sample.source != "nist_sard_php_sqli" else sample
        for sample in assigned
    ]


def build_phase1(archives_root: Path, data_root: Path) -> dict[str, object]:
    archives_root = archives_root.resolve()
    layout = resolve_practiceset_layout(archives_root)
    data_root = data_root.resolve()
    ensure_data_directories(data_root)
    _, clean_by_package = _pair_maps(layout.other)
    clean = _clean_samples(clean_by_package)
    compromised, diff_stats = _compromised_samples(layout.javascript, layout.python, clean_by_package)
    intent, intent_stats = _intent_samples(layout.javascript, layout.python)
    owasp = _owasp_samples(layout.vulnerability / "BenchmarkJava-master")
    nist = _nist_samples(layout.vulnerability / "2022-05-12-php-test-suite-sqli-v1-0-0")
    raw_samples = clean + compromised + intent + owasp + nist
    deduped, dedupe_report = deduplicate(raw_samples)
    split_samples = _assign_audited_splits(deduped)
    evasions = generate_evasion_suite(split_samples, limit=240)
    final_samples = split_samples + evasions
    gate = _quality_gate(final_samples, dedupe_report)

    processed = data_root / "processed" / "phase1_dataset.jsonl"
    write_jsonl(processed, final_samples)
    review_queue = [sample for sample in final_samples if sample.review_status == "needs_review"]
    write_jsonl(data_root / "processed" / "review_queue.jsonl", review_queue)
    for split in ("train", "validation", "test"):
        write_jsonl(data_root / "splits" / f"{split}.jsonl", (sample for sample in final_samples if sample.split == split))
    write_jsonl(data_root / "processed" / "evasion_tests.jsonl", evasions)
    input_counts = {"paired_clean": len(clean), "compromised_diff": len(compromised), "malicious_intent": len(intent), "owasp": len(owasp), "nist": len(nist)}
    manifest = {
        "schema_version": 4, "created_at": utc_now(), "dataset_path": str(processed),
        "dataset_sha256": sha256_file(processed), "samples": len(final_samples),
        "labels": dict(Counter(sample.label for sample in final_samples)),
        "languages": dict(Counter(sample.language for sample in final_samples)),
        "sources": dict(Counter(sample.source for sample in final_samples)),
        "categories": dict(Counter(sample.category for sample in final_samples)),
        "splits": dict(Counter(sample.split for sample in final_samples)),
        "input_counts": input_counts, "diff_labeling": diff_stats, "intent_labeling": intent_stats,
        "deduplication": dedupe_report, "quality_gate": gate, "evasion_samples": len(evasions),
        "review_queue_samples": len(review_queue),
        "heldout_sources": ["nist_sard_php_sqli"], "excluded_legacy_handwritten_dataset": True,
        "practicesets_root": str(layout.root), "organized_practicesets": layout.organized,
        "safety": {"executed_samples": False, "source_files_only": True, "archive_limit_enforced": True},
    }
    (data_root / "manifests" / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_root / "manifests" / "data_quality_report.json").write_text(json.dumps({
        "created_at": manifest["created_at"], "input_counts": input_counts, "diff_labeling": diff_stats,
        "intent_labeling": intent_stats, "deduplication": dedupe_report, "quality_gate": gate,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the audited local phase-one dataset")
    parser.add_argument("archives_root", type=Path)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    result = build_phase1(args.archives_root, args.data_root)
    print(json.dumps({key: value for key, value in result.items() if key not in {"categories", "deduplication"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
