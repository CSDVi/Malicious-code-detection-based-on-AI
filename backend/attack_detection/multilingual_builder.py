"""Build the non-DataDog multilingual training corpus from local practicesets.

All untrusted samples are read as inert bytes/text.  Nothing from an archive is
extracted or executed by this module.  Repository/project/family identifiers are
kept in one split to prevent the most common source-code evaluation leakage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from attack_detection.data_pipeline import deduplicate, make_sample, sha256_file, utc_now, write_jsonl
from attack_detection.dataset import CodeSample, ensure_data_directories, is_training_eligible, load_dataset
from attack_detection.languages import canonical_language
from attack_detection.practiceset_layout import resolve_practiceset_layout
from attack_detection.rules import detect_by_rules
from attack_detection.training.language_coverage import eligible_task_languages


BACKEND_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PRACTICESETS = WORKSPACE_DIR / "practicesets"
DEFAULT_DATA_ROOT = BACKEND_DIR / "data"
DEFAULT_BASE_DATASET = DEFAULT_DATA_ROOT / "processed" / "phase2_balanced_dataset.jsonl"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "processed" / "phase3_multilingual_dataset.jsonl"

SOURCE_URLS = {
    "codesearchnet": "https://github.com/github/CodeSearchNet",
    "crossvul": "https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset",
    "javascript_malware_collection": "https://github.com/HynekPetrak/javascript-malware-collection",
    "php_webshell_collection": "https://github.com/w-32768/PHP-Webshell-Detection-via-Opcode-Analysis",
    "android_malware_source": "https://github.com/d-Raco/android-malware-source-code-samples",
}

MAX_SOURCE_BYTES = 128 * 1024
MIN_CODE_CHARACTERS = 24
CODESEARCHNET_LANGUAGES = ("java", "javascript", "php", "go")
CROSSVUL_LANGUAGE_CAPS = {
    "python": 800,
    "javascript": 900,
    "typescript": 200,
    "java": 800,
    "php": 1_500,
    "go": 400,
    "bash": 200,
    "config": 300,
    "c": 1_500,
    "cpp": 900,
    "csharp": 300,
    "ruby": 700,
    "rust": 100,
    "scala": 100,
    "lua": 100,
    "perl": 200,
    "html": 250,
    "sql": 100,
}


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _family_split(family: str) -> str:
    bucket = int(_stable_key(family)[:8], 16) % 100
    return "test" if bucket < 20 else ("validation" if bucket < 35 else "train")


def _decode_source(raw: bytes) -> str:
    if not raw or len(raw) > MAX_SOURCE_BYTES or b"\x00" in raw[:4096]:
        return ""
    code = raw.decode("utf-8", errors="replace").strip()
    return code if len(code) >= MIN_CODE_CHARACTERS else ""


def _malicious_metadata(code: str, language: str, fallback: str) -> tuple[str, list[str], list[str]]:
    matches = [item for item in detect_by_rules(code, language) if item.get("risk_type") == "malicious"]
    behaviors = sorted({str(item.get("category") or "") for item in matches if item.get("category")})
    cwes = sorted({str(item.get("cwe") or "") for item in matches if item.get("cwe")})
    category = str(max(matches, key=lambda item: int(item.get("severity") or 0)).get("category")) if matches else fallback
    return category, behaviors or [fallback], cwes


def _is_datadog_malicious_dataset(sample: CodeSample) -> bool:
    source = sample.source.lower()
    source_url = sample.source_url.lower()
    return source.startswith("datadog_") or "datadog/malicious-software-packages-dataset" in source_url


def load_non_datadog_base(path: Path) -> tuple[list[CodeSample], dict[str, int]]:
    """Retain independent data while removing direct/derived DataDog records."""

    samples = load_dataset(path)
    direct_hashes = {
        sample.sample_hash for sample in samples
        if _is_datadog_malicious_dataset(sample)
    }
    kept = []
    stats = Counter()
    for sample in samples:
        if sample.sample_hash in direct_hashes:
            stats["direct_datadog_excluded"] += 1
            continue
        if sample.source == "evasion_suite" or sample.review_status == "generated_variant":
            stats["generated_variants_excluded"] += 1
            continue
        if sample.parent_sample_hash and sample.parent_sample_hash in direct_hashes:
            stats["datadog_derivatives_excluded"] += 1
            continue
        kept.append(sample)
    stats["input"] = len(samples)
    stats["retained"] = len(kept)
    return kept, dict(stats)


def ingest_codesearchnet(
    root: Path,
    per_split: dict[str, int] | None = None,
    max_per_repository: int = 12,
) -> tuple[list[CodeSample], dict[str, Any]]:
    limits = per_split or {"train": 2_500, "validation": 500, "test": 500}
    samples: list[CodeSample] = []
    report: dict[str, Any] = {"by_language_split": {}, "missing_languages": []}
    for language in CODESEARCHNET_LANGUAGES:
        jsonl_root = root / language / language / "final" / "jsonl"
        if not jsonl_root.is_dir():
            report["missing_languages"].append(language)
            continue
        for directory, split in (("train", "train"), ("valid", "validation"), ("test", "test")):
            accepted = 0
            repo_counts: Counter[str] = Counter()
            for source_file in sorted((jsonl_root / directory).glob("*.jsonl.gz")):
                with gzip.open(source_file, "rt", encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        if accepted >= limits[split]:
                            break
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        repository = str(row.get("repo") or "").strip()
                        code = str(row.get("code") or row.get("original_string") or "").strip()
                        if not repository or len(code) < MIN_CODE_CHARACTERS or len(code.encode("utf-8", errors="ignore")) > MAX_SOURCE_BYTES:
                            continue
                        if repo_counts[repository] >= max_per_repository:
                            continue
                        repo_counts[repository] += 1
                        family = f"codesearchnet:{language}:{repository.lower()}"
                        samples.append(make_sample(
                            code,
                            label="benign",
                            category="normal_reference_code",
                            language=language,
                            source="codesearchnet",
                            package_name=repository,
                            version="reference",
                            license="CodeSearchNet terms; original repository license applies",
                            family=family,
                            split=split,
                            source_url=str(row.get("url") or SOURCE_URLS["codesearchnet"]),
                            file_path=str(row.get("path") or ""),
                            label_basis="codesearchnet_normal_code_reference",
                            label_confidence=0.85,
                            review_status="source_verified",
                            review_notes="Benign is a normal-corpus assumption, not a per-function security audit.",
                            label_scopes=["malicious_intent", "vulnerability_risk"],
                        ))
                        accepted += 1
                if accepted >= limits[split]:
                    break
            report["by_language_split"][f"{language}/{split}"] = accepted
    report["accepted"] = len(samples)
    return samples, report


def _crossvul_split_map(family_weights: dict[str, int]) -> dict[str, str]:
    total = sum(family_weights.values())
    if total >= 100:
        targets = {"train": round(total * 0.65), "validation": round(total * 0.15)}
        targets["test"] = total - targets["train"] - targets["validation"]
        minimums = {"train": 20, "validation": 10, "test": 20}
    elif total >= 50:
        targets = {"train": total - 30, "validation": 10, "test": 20}
        minimums = {"train": 20, "validation": 10, "test": 20}
    elif total >= 35:
        targets = {"train": 20, "validation": 5, "test": total - 25}
        minimums = {"train": 20, "validation": 5, "test": 10}
    else:
        targets = {"train": round(total * 0.6), "validation": round(total * 0.2)}
        targets["test"] = total - targets["train"] - targets["validation"]
        minimums = {"train": 0, "validation": 0, "test": 0}
    remaining = dict(targets)
    output = {}
    for family in sorted(family_weights, key=_stable_key):
        split = max(("train", "validation", "test"), key=lambda value: (remaining[value], value))
        output[family] = split
        remaining[split] -= family_weights[family]
    counts = Counter()
    for family, split in output.items():
        counts[split] += family_weights[family]
    for target_split in ("train", "validation", "test"):
        while counts[target_split] < minimums[target_split]:
            candidates = []
            for family, donor_split in output.items():
                if donor_split == target_split:
                    continue
                weight = family_weights[family]
                if counts[donor_split] - weight >= minimums[donor_split]:
                    candidates.append((
                        abs(minimums[target_split] - (counts[target_split] + weight)),
                        weight,
                        _stable_key(family),
                        family,
                        donor_split,
                    ))
            if not candidates:
                break
            _, weight, _, family, donor_split = min(candidates)
            output[family] = target_split
            counts[donor_split] -= weight
            counts[target_split] += weight
    return output


def _rebalance_crossvul_splits(samples: list[CodeSample]) -> list[CodeSample]:
    """Restore per-language split coverage after cross-source near-deduplication."""

    family_weights_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        if sample.source == "crossvul" and sample.label == "vulnerable" and sample.family:
            family_weights_by_language[sample.language][sample.family] += 1
    assignments = {
        language: _crossvul_split_map(dict(weights))
        for language, weights in family_weights_by_language.items()
    }
    return [
        replace(sample, split=assignments[sample.language][sample.family])
        if sample.source == "crossvul"
        and sample.language in assignments
        and sample.family in assignments[sample.language]
        else sample
        for sample in samples
    ]


def ingest_crossvul(path: Path) -> tuple[list[CodeSample], dict[str, Any]]:
    pairs: dict[str, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    metadata: dict[str, tuple[str, str, str]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.file_size > MAX_SOURCE_BYTES:
                continue
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 4 or not parts[-3].upper().startswith("CWE-"):
                continue
            match = re.fullmatch(r"(bad|good)_(\d+)_(\d+)", parts[-1], re.IGNORECASE)
            language = canonical_language(parts[-2])
            if not match or language not in CROSSVUL_LANGUAGE_CAPS:
                continue
            state, case_id, fragment = match.groups()
            pair_key = f"{language}:{case_id}:{fragment}"
            family = f"crossvul:{language}:{case_id}"
            pairs[pair_key][state.lower()] = info
            metadata[pair_key] = (language, parts[-3].upper(), family)

        by_language: dict[str, list[str]] = defaultdict(list)
        for key, values in pairs.items():
            if {"bad", "good"}.issubset(values):
                by_language[metadata[key][0]].append(key)

        selected: dict[str, list[str]] = {}
        split_maps: dict[str, dict[str, str]] = {}
        for language, keys in by_language.items():
            chosen = sorted(keys, key=_stable_key)[:CROSSVUL_LANGUAGE_CAPS[language]]
            selected[language] = chosen
            family_weights = Counter(metadata[key][2] for key in chosen)
            split_maps[language] = _crossvul_split_map(dict(family_weights))

        artifact_hash = sha256_file(path)
        samples = []
        for language in sorted(selected):
            for key in selected[language]:
                _, cwe, family = metadata[key]
                split = split_maps[language][family]
                for state, label in (("bad", "vulnerable"), ("good", "benign")):
                    info = pairs[key][state]
                    code = _decode_source(archive.read(info))
                    if not code:
                        continue
                    samples.append(make_sample(
                        code,
                        label=label,
                        category=cwe,
                        language=language,
                        cwe=cwe if label == "vulnerable" else "",
                        cwe_labels=[cwe] if label == "vulnerable" else [],
                        source="crossvul",
                        package_name=family.rsplit(":", 1)[-1],
                        version=state,
                        license="CrossVul dataset terms; original repository license applies",
                        family=family,
                        split=split,
                        artifact_sha256=artifact_hash,
                        source_url=SOURCE_URLS["crossvul"],
                        file_path=info.filename,
                        paired_version="good" if state == "bad" else "bad",
                        label_basis="crossvul_vulnerable_before_fix" if state == "bad" else "crossvul_fixed_counterpart",
                        label_confidence=0.98,
                        review_status="differentially_verified",
                        label_scopes=["vulnerability_risk", "cwe_labels"],
                    ))
    report = {
        "accepted": len(samples),
        "by_language_label": dict(Counter(f"{s.language}/{s.label}" for s in samples)),
        "selected_pairs": {language: len(keys) for language, keys in selected.items()},
    }
    return samples, report


def ingest_javascript_malware(path: Path, limit: int = 5_000, per_capture_day: int = 25) -> tuple[list[CodeSample], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        by_day: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".js") or not (MIN_CODE_CHARACTERS <= info.file_size <= MAX_SOURCE_BYTES):
                continue
            parts = PurePosixPath(info.filename).parts
            capture_day = parts[-2] if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[-2]) else "unknown"
            by_day[capture_day].append(info)
        candidates = [
            info
            for day in sorted(by_day, key=_stable_key)
            for info in sorted(by_day[day], key=lambda item: _stable_key(item.filename))[:per_capture_day]
        ]
        candidates = sorted(candidates, key=lambda item: _stable_key(item.filename))[:limit]
        artifact_hash = sha256_file(path)
        samples = []
        for info in candidates:
            code = _decode_source(archive.read(info))
            if not code:
                continue
            capture_day = PurePosixPath(info.filename).parts[-2]
            family = f"javascript-malware:capture-day:{capture_day}"
            category, behaviors, cwes = _malicious_metadata(code, "javascript", "Malicious Script")
            samples.append(make_sample(
                code,
                label="malicious",
                category=category,
                language="javascript",
                cwe=",".join(cwes),
                behavior_labels=behaviors,
                cwe_labels=cwes,
                source="javascript_malware_collection",
                package_name=PurePosixPath(info.filename).stem,
                version=capture_day,
                license="CC0-1.0",
                family=family,
                published_at=capture_day,
                split=_family_split(family),
                artifact_sha256=artifact_hash,
                source_url=SOURCE_URLS["javascript_malware_collection"],
                file_path=info.filename,
                label_basis="malware_collection_capture",
                label_confidence=0.93,
                review_status="source_verified",
                label_scopes=["malicious_intent", "behavior_labels"],
            ))
    return samples, {
        "accepted": len(samples),
        "capture_days": len({sample.family for sample in samples}),
        "splits": dict(Counter(sample.split for sample in samples)),
    }


def _php_family(filename: str, label: str) -> str:
    path = PurePosixPath(filename)
    collection = path.parts[-2] if len(path.parts) > 1 else "root"
    stem = path.stem.lower()
    stem = re.sub(r"^sourcecode_[a-z0-9]+_", "", stem)
    stem = re.sub(r"(?:v(?:ersion)?[-_. ]*)?\d+(?:[._-]\d+)*", "", stem)
    tokens = [token for token in re.split(r"[^a-z]+", stem) if len(token) >= 2]
    identity = "-".join(tokens[:3]) or _stable_key(filename)[:16]
    return f"php-corpus:{label}:{collection}:{identity}"


def ingest_php_webshell(path: Path, per_label: int = 3_500) -> tuple[list[CodeSample], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        candidates: dict[str, list[zipfile.ZipInfo]] = {"benign": [], "malicious": []}
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".php") or not (MIN_CODE_CHARACTERS <= info.file_size <= MAX_SOURCE_BYTES):
                continue
            lowered = info.filename.lower()
            label = "malicious" if "/webshells/" in lowered else ("benign" if "/benign/" in lowered else "")
            if label:
                candidates[label].append(info)
        selected = {
            label: sorted(values, key=lambda item: _stable_key(item.filename))[:per_label]
            for label, values in candidates.items()
        }
        artifact_hash = sha256_file(path)
        samples = []
        for label in ("benign", "malicious"):
            for info in selected[label]:
                try:
                    raw = archive.read(info, pwd=b"123")
                except RuntimeError:
                    raw = archive.read(info)
                code = _decode_source(raw)
                if not code:
                    continue
                family = _php_family(info.filename, label)
                if label == "malicious":
                    category, behaviors, cwes = _malicious_metadata(code, "php", "WebShell")
                else:
                    category, behaviors, cwes = "normal_php", [], []
                samples.append(make_sample(
                    code,
                    label=label,
                    category=category,
                    language="php",
                    cwe=",".join(cwes),
                    behavior_labels=behaviors,
                    cwe_labels=cwes,
                    source="php_webshell_collection",
                    package_name=PurePosixPath(info.filename).stem,
                    version="corpus",
                    license="Academic research dataset; original source licenses apply",
                    family=family,
                    split=_family_split(family),
                    artifact_sha256=artifact_hash,
                    source_url=SOURCE_URLS["php_webshell_collection"],
                    file_path=info.filename,
                    label_basis="curated_webshell" if label == "malicious" else "curated_benign_php",
                    label_confidence=0.96 if label == "malicious" else 0.90,
                    review_status="source_verified",
                    label_scopes=["malicious_intent", "behavior_labels"] if label == "malicious" else ["malicious_intent", "vulnerability_risk"],
                ))
    return samples, {
        "accepted": len(samples),
        "labels": dict(Counter(sample.label for sample in samples)),
        "families": len({sample.family for sample in samples}),
        "splits": dict(Counter(sample.split for sample in samples)),
    }


def ingest_android_malware(root: Path, per_project: int = 40) -> tuple[list[CodeSample], dict[str, Any]]:
    samples = []
    unreadable = []
    for archive_path in sorted(root.glob("*.zip")):
        family_name = archive_path.stem
        family = f"android-malware:{family_name.lower()}"
        category = family_name.split("_AndroidOS", 1)[0].replace("-", " ")
        try:
            with zipfile.ZipFile(archive_path) as archive:
                candidates = [
                    info for info in archive.infolist()
                    if not info.is_dir()
                    and info.filename.lower().endswith(".java")
                    and MIN_CODE_CHARACTERS <= info.file_size <= MAX_SOURCE_BYTES
                    and not re.search(r"(?:^|/)(?:build|generated|test|tests|example|examples|\.gradle)(?:/|$)", info.filename.lower())
                    and PurePosixPath(info.filename).name not in {"R.java", "BuildConfig.java"}
                ]
                for info in sorted(candidates, key=lambda item: _stable_key(item.filename))[:per_project]:
                    code = _decode_source(archive.read(info))
                    if not code:
                        continue
                    detected_category, behaviors, cwes = _malicious_metadata(code, "java", category)
                    samples.append(make_sample(
                        code,
                        label="malicious",
                        category=detected_category,
                        language="java",
                        cwe=",".join(cwes),
                        behavior_labels=behaviors,
                        cwe_labels=cwes,
                        source="android_malware_source",
                        package_name=family_name,
                        version="source-snapshot",
                        license="See the original project represented by this collected source archive",
                        family=family,
                        split=_family_split(family),
                        source_url=SOURCE_URLS["android_malware_source"],
                        file_path=f"{archive_path.name}!/{info.filename}",
                        label_basis="labeled_android_malware_source_project",
                        label_confidence=0.92,
                        review_status="source_verified",
                        label_scopes=["malicious_intent", "behavior_labels"],
                    ))
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            unreadable.append({"archive": archive_path.name, "error": str(exc)})
    return samples, {
        "accepted": len(samples),
        "projects": len({sample.family for sample in samples}),
        "categories": dict(Counter(sample.category for sample in samples)),
        "splits": dict(Counter(sample.split for sample in samples)),
        "unreadable": unreadable,
    }


def _validate_family_isolation(samples: Iterable[CodeSample]) -> list[dict[str, Any]]:
    splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.family and sample.split:
            splits[sample.family].add(sample.split)
    return [
        {"family": family, "splits": sorted(values)}
        for family, values in sorted(splits.items()) if len(values) > 1
    ]


def _coverage(samples: list[CodeSample]) -> dict[str, Any]:
    eligible = [sample for sample in samples if is_training_eligible(sample)]
    partitions = {
        split: [sample for sample in eligible if sample.split == split]
        for split in ("train", "validation", "test")
    }
    tasks = {}
    for task, positive in (("malicious_intent", "malicious"), ("vulnerability_risk", "vulnerable")):
        languages, details = eligible_task_languages(partitions, positive, "benign")
        deep_languages, deep_details = eligible_task_languages(
            partitions, positive, "benign", {"train": 20, "validation": 10, "test": 20},
        )
        provisional_languages, provisional_details = eligible_task_languages(
            partitions, positive, "benign", {"train": 20, "validation": 5, "test": 10},
        )
        tasks[task] = {
            "classical_model_languages": languages,
            "deep_model_candidate_languages": deep_languages,
            "deep_model_provisional_languages": provisional_languages,
            "classical_requirements": {"train": 20, "validation": 5, "test": 10},
            "deep_requirements": {"train": 20, "validation": 10, "test": 20},
            "classical_details": details,
            "deep_details": deep_details,
            "provisional_deep_details": provisional_details,
        }
    return {
        "training_eligible": len(eligible),
        "tasks": tasks,
        "by_split_language_label": {
            f"{split}/{language}/{label}": count
            for (split, language, label), count in sorted(Counter(
                (sample.split, sample.language, sample.label) for sample in eligible
            ).items())
        },
    }


def _cache_signature(paths: Iterable[Path], config: dict[str, Any]) -> str:
    records = []
    for path in paths:
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            files = [path]
        for item in files:
            stat = item.stat()
            records.append((str(item.resolve()), stat.st_size, stat.st_mtime_ns))
    value = json.dumps({"files": records, "config": config}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def refresh_multilingual_manifest(
    dataset: Path, data_root: Path, *, rebalance_existing: bool = False,
) -> dict[str, Any]:
    """Recompute coverage for an already-built dataset without re-ingestion."""

    dataset = dataset.resolve()
    data_root = data_root.resolve()
    samples = load_dataset(dataset)
    if rebalance_existing:
        samples = _rebalance_crossvul_splits(samples)
        family_leaks = _validate_family_isolation(samples)
        if family_leaks:
            raise ValueError(f"family leakage across splits: {family_leaks[:10]}")
        write_jsonl(dataset, samples)
        for split in ("train", "validation", "test"):
            write_jsonl(data_root / "splits" / f"multilingual_{split}.jsonl", (
                sample for sample in samples if sample.split == split
            ))
    coverage = _coverage(samples)
    manifest_path = data_root / "manifests" / "multilingual_dataset_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {"schema_version": 1, "created_at": utc_now()}
    manifest.update({
        "dataset_path": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "samples": len(samples),
        "labels": dict(Counter(sample.label for sample in samples)),
        "languages": dict(Counter(sample.language for sample in samples)),
        "sources": dict(Counter(sample.source for sample in samples)),
        "splits": dict(Counter(sample.split for sample in samples)),
        "coverage": coverage,
        "family_isolation_verified": not _validate_family_isolation(samples),
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_root / "manifests" / "multilingual_language_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def build_multilingual_dataset(
    practicesets: Path = DEFAULT_PRACTICESETS,
    data_root: Path = DEFAULT_DATA_ROOT,
    base_dataset: Path = DEFAULT_BASE_DATASET,
    output: Path = DEFAULT_OUTPUT,
    codesearchnet_per_split: dict[str, int] | None = None,
    javascript_malware_limit: int = 5_000,
    php_per_label: int = 3_500,
    android_per_project: int = 40,
) -> dict[str, Any]:
    practicesets = practicesets.resolve()
    layout = resolve_practiceset_layout(practicesets)
    data_root = data_root.resolve()
    base_dataset = base_dataset.resolve()
    output = output.resolve()
    ensure_data_directories(data_root)
    if not base_dataset.is_file():
        raise FileNotFoundError(f"base dataset is missing: {base_dataset}")

    base, base_report = load_non_datadog_base(base_dataset)
    inputs: dict[str, list[CodeSample]] = {"non_datadog_base": base}
    source_reports: dict[str, Any] = {"non_datadog_base": base_report}

    codesearchnet_root = layout.other / "CodeSearchNet"
    crossvul_path = layout.vulnerability / "crossvul" / "dataset.zip"
    javascript_path = layout.javascript / "javascript-malware-collection-master.zip"
    php_path = layout.php / "php_webshell" / "source_code(pass123).zip"
    android_root = layout.java / "android_malware_java"
    ingesters = (
        ("codesearchnet", [codesearchnet_root], {"per_split": codesearchnet_per_split},
         lambda: ingest_codesearchnet(codesearchnet_root, codesearchnet_per_split)),
        ("crossvul", [crossvul_path], {"language_caps": CROSSVUL_LANGUAGE_CAPS, "max_source_bytes": MAX_SOURCE_BYTES},
         lambda: ingest_crossvul(crossvul_path)),
        ("javascript_malware_collection", [javascript_path], {"limit": javascript_malware_limit, "max_source_bytes": MAX_SOURCE_BYTES}, lambda: ingest_javascript_malware(
            javascript_path, javascript_malware_limit,
        )),
        ("php_webshell_collection", [php_path], {"per_label": php_per_label, "max_source_bytes": MAX_SOURCE_BYTES}, lambda: ingest_php_webshell(
            php_path, php_per_label,
        )),
        ("android_malware_source", [android_root], {"per_project": android_per_project, "max_source_bytes": MAX_SOURCE_BYTES}, lambda: ingest_android_malware(
            android_root, android_per_project,
        )),
    )
    cache_root = data_root / "processed" / "multilingual_ingest_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for name, source_paths, config, ingest in ingesters:
        signature = _cache_signature(source_paths, config)
        cache_path = cache_root / f"{name}.jsonl"
        cache_report_path = cache_root / f"{name}.json"
        cached_report = {}
        if cache_path.is_file() and cache_report_path.is_file():
            try:
                cached_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_report = {}
        if cached_report.get("cache_signature") == signature:
            values = load_dataset(cache_path)
            report = dict(cached_report.get("source_report") or {})
            report["cache_reused"] = True
        else:
            values, report = ingest()
            write_jsonl(cache_path, values)
            cache_report_path.write_text(json.dumps({
                "cache_signature": signature,
                "source_report": report,
                "samples": len(values),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            report["cache_reused"] = False
        inputs[name] = values
        source_reports[name] = report
        print(json.dumps({
            "stage": name, "accepted": len(values), "cache_reused": report["cache_reused"],
        }, ensure_ascii=False), flush=True)

    raw_samples = [sample for values in inputs.values() for sample in values]
    deduped, dedupe_report = deduplicate(raw_samples)
    deduped = _rebalance_crossvul_splits(deduped)
    family_leaks = _validate_family_isolation(deduped)
    if family_leaks:
        raise ValueError(f"family leakage across splits: {family_leaks[:10]}")
    datadog_residue = [
        sample for sample in deduped
        if _is_datadog_malicious_dataset(sample)
    ]
    if datadog_residue:
        raise ValueError(f"DataDog records remain after filtering: {len(datadog_residue)}")

    write_jsonl(output, deduped)
    for split in ("train", "validation", "test"):
        write_jsonl(data_root / "splits" / f"multilingual_{split}.jsonl", (
            sample for sample in deduped if sample.split == split
        ))
    coverage = _coverage(deduped)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "dataset_path": str(output),
        "dataset_sha256": sha256_file(output),
        "base_dataset": str(base_dataset),
        "practicesets": str(practicesets),
        "organized_practicesets": layout.organized,
        "samples": len(deduped),
        "labels": dict(Counter(sample.label for sample in deduped)),
        "languages": dict(Counter(sample.language for sample in deduped)),
        "sources": dict(Counter(sample.source for sample in deduped)),
        "splits": dict(Counter(sample.split for sample in deduped)),
        "source_reports": source_reports,
        "deduplication": dedupe_report,
        "coverage": coverage,
        "family_isolation_verified": not family_leaks,
        "datadog_policy": {
            "excluded": True,
            "direct_or_derived_records_in_output": 0,
            "base_filter": base_report,
        },
        "safety": {
            "executed_samples": False,
            "archives_read_only": True,
            "maximum_source_file_bytes": MAX_SOURCE_BYTES,
            "synthetic_evasion_samples": 0,
        },
    }
    manifest_path = data_root / "manifests" / "multilingual_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_root / "manifests" / "multilingual_language_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local non-DataDog multilingual training dataset")
    parser.add_argument("--practicesets", type=Path, default=DEFAULT_PRACTICESETS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--codesearchnet-train", type=int, default=2_500)
    parser.add_argument("--codesearchnet-validation", type=int, default=500)
    parser.add_argument("--codesearchnet-test", type=int, default=500)
    parser.add_argument("--javascript-malware-limit", type=int, default=5_000)
    parser.add_argument("--php-per-label", type=int, default=3_500)
    parser.add_argument("--android-per-project", type=int, default=40)
    parser.add_argument("--refresh-manifest-only", action="store_true")
    parser.add_argument("--rebalance-existing", action="store_true")
    args = parser.parse_args()
    if args.refresh_manifest_only:
        result = refresh_multilingual_manifest(
            args.output, args.data_root, rebalance_existing=args.rebalance_existing,
        )
        print(json.dumps({
            "dataset_path": result["dataset_path"],
            "dataset_sha256": result["dataset_sha256"],
            "samples": result["samples"],
            "labels": result["labels"],
            "languages": result["languages"],
            "coverage": {
                task: values["deep_model_candidate_languages"]
                for task, values in result["coverage"]["tasks"].items()
            },
        }, ensure_ascii=False, indent=2))
        return
    result = build_multilingual_dataset(
        practicesets=args.practicesets,
        data_root=args.data_root,
        base_dataset=args.base_dataset,
        output=args.output,
        codesearchnet_per_split={
            "train": max(0, args.codesearchnet_train),
            "validation": max(0, args.codesearchnet_validation),
            "test": max(0, args.codesearchnet_test),
        },
        javascript_malware_limit=max(0, args.javascript_malware_limit),
        php_per_label=max(0, args.php_per_label),
        android_per_project=max(0, args.android_per_project),
    )
    print(json.dumps({
        "dataset_path": result["dataset_path"],
        "dataset_sha256": result["dataset_sha256"],
        "samples": result["samples"],
        "labels": result["labels"],
        "languages": result["languages"],
        "coverage": {
            task: values["deep_model_candidate_languages"]
            for task, values in result["coverage"]["tasks"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
