"""Build repository-isolated GATv2 routes from local The Stack Parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


LANGUAGE_INPUTS = {
    "bash": ("bashshell", "train-*-of-00011.parquet"),
    "config": ("YAML", "train-*-of-00096.parquet"),
    "powershell": ("PowerShell", "train-*-of-00004.parquet"),
}
SPLIT_LIMITS = {"train": 240, "validation": 60, "test": 60}
MAX_FILES_PER_REPOSITORY = 24
MAX_CONTENT_BYTES = 128_000


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _split(identity: str) -> str:
    bucket = int(_sha(identity)[:8], 16) % 100
    return "train" if bucket < 70 else ("validation" if bucket < 85 else "test")


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    columns = [
        "hexsha", "max_stars_repo_path", "max_stars_repo_name",
        "max_stars_repo_head_hexsha", "max_stars_repo_licenses", "content",
    ]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def _benign_rows(root: Path, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folder, pattern = LANGUAGE_INPUTS[language]
    shards = sorted((root / folder).glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No {language} shards found under {root / folder}")
    repositories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_by_split: dict[str, set[str]] = defaultdict(set)
    scanned = 0
    for shard in shards:
        for row in _rows(shard):
            scanned += 1
            repository = str(row.get("max_stars_repo_name") or "").strip()
            content = str(row.get("content") or "")
            file_path = str(row.get("max_stars_repo_path") or "").strip()
            if not repository or not file_path or not content.strip():
                continue
            split = _split(repository)
            if (
                repository not in repositories
                and len(selected_by_split[split]) >= SPLIT_LIMITS[split]
            ):
                continue
            selected_by_split[split].add(repository)
            if len(repositories[repository]) >= MAX_FILES_PER_REPOSITORY:
                continue
            encoded = content.encode("utf-8", errors="ignore")
            if len(encoded) > MAX_CONTENT_BYTES:
                content = encoded[:MAX_CONTENT_BYTES].decode("utf-8", errors="ignore")
            repositories[repository].append({
                "code": content,
                "normalized_code": content,
                "label": "benign",
                "category": "normal_project",
                "language": language,
                "source": "the_stack_permissive_benign",
                "package_name": repository,
                "family": f"stack:{language}:{repository}",
                "version": str(row.get("max_stars_repo_head_hexsha") or "")[:40],
                "license": str(row.get("max_stars_repo_licenses") or ""),
                "sample_hash": str(row.get("hexsha") or _sha(content)),
                "split": split,
                "file_path": file_path,
                "label_basis": "permissively_licensed_public_repository; benign training candidate",
                "label_confidence": 0.9,
                "review_status": "source_verified",
                "label_scopes": ["malicious_intent"],
            })
    output = [
        row
        for repository in sorted(repositories)
        for row in repositories[repository]
    ]
    return output, {
        "shards": [str(path.resolve()) for path in shards],
        "rows_scanned": scanned,
        "repositories": dict(Counter(row["split"] for rows in repositories.values() for row in rows[:1])),
        "files": dict(Counter(row["split"] for row in output)),
    }


def _malicious_rows(source: Path, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("language") != language or row.get("label") != "malicious":
                continue
            family = str(row.get("family") or "").strip()
            code = str(row.get("code") or "")
            if not family or not code.strip() or len(families[family]) >= MAX_FILES_PER_REPOSITORY:
                continue
            normalized = dict(row)
            normalized.update({
                "source": "mascot_human_reviewed",
                "family": f"curated:{language}:{family}",
                "package_name": str(row.get("package_name") or family),
                "label_confidence": max(0.9, float(row.get("label_confidence") or 0)),
                "review_status": "source_verified",
                "label_scopes": ["malicious_intent"],
            })
            families[family].append(normalized)
    ordered_families = sorted(families, key=_sha)
    if len(ordered_families) < 40:
        raise RuntimeError(f"{language}: only {len(ordered_families)} malicious families; need 40")
    validation_count = max(10, round(len(ordered_families) * 0.15))
    test_count = max(10, round(len(ordered_families) * 0.15))
    assignments = {}
    for index, family in enumerate(ordered_families):
        if index < validation_count:
            assignments[family] = "validation"
        elif index < validation_count + test_count:
            assignments[family] = "test"
        else:
            assignments[family] = "train"
        for row in families[family]:
            row["split"] = assignments[family]
    output = [row for family in sorted(families) for row in families[family]]
    project_counts = Counter(rows[0]["split"] for rows in families.values())
    for split, minimum in (("train", 20), ("validation", 10), ("test", 10)):
        if project_counts[split] < minimum:
            raise RuntimeError(
                f"{language}: {split} has {project_counts[split]} malicious families; need {minimum}"
            )
    return output, {
        "source": str(source.resolve()),
        "families": dict(project_counts),
        "files": dict(Counter(row["split"] for row in output)),
    }


def build(root: Path, malicious_source: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": 1, "languages": {}}
    for language in LANGUAGE_INPUTS:
        benign, benign_report = _benign_rows(root, language)
        malicious, malicious_report = _malicious_rows(malicious_source, language)
        destination = output_root / f"gatv2_stack_{language}_routes.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sorted(
                benign + malicious,
                key=lambda item: (
                    str(item["split"]), str(item["label"]),
                    str(item["family"]), str(item.get("file_path") or ""),
                ),
            ):
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["languages"][language] = {
            "output": str(destination.resolve()),
            "benign": benign_report,
            "malicious": malicious_report,
        }
    report_path = output_root / "gatv2_stack_routes_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practices-root", required=True, type=Path)
    parser.add_argument("--malicious-source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(
        args.practices_root.resolve(),
        args.malicious_source.resolve(),
        args.output_root.resolve(),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
