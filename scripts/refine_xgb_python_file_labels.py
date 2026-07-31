"""Refine malicious-PyPI package labels to standalone Python file labels.

MalRegistry establishes that a distribution is malicious, but an individual
file copied into that distribution may still be an unchanged third-party
module, test, example, or data file. This migration preserves every row and
the original malicious label metadata while excluding files without a
high-confidence code-local behavior chain from supervised file classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.features.high_confidence_behaviors import (
    python_high_confidence_behavior_count,
)


EXCLUDED_PATH_PARTS = {
    ".github",
    "benchmark",
    "benchmarks",
    "demo",
    "demos",
    "doc",
    "docs",
    "documentation",
    "example",
    "examples",
    "fixture",
    "fixtures",
    "test",
    "tests",
}
PYTHON_EXTENSIONS = {".py", ".pyw"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_production_python_path(file_path: str) -> bool:
    member = file_path.split("#")[-1].replace("\\", "/")
    path = PurePosixPath(member)
    parts = {part.lower() for part in path.parts}
    return (
        path.suffix.lower() in PYTHON_EXTENSIONS
        and not bool(parts & EXCLUDED_PATH_PARTS)
    )


def refine(base: Path, output: Path, report_path: Path) -> dict[str, Any]:
    rows = 0
    changed = 0
    changed_counts: Counter[tuple[str, str]] = Counter()
    changed_families: dict[str, set[str]] = defaultdict(set)
    retained_counts: Counter[str] = Counter()
    retained_families: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with base.open(encoding="utf-8") as source, temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if (
                str(row.get("language") or "").lower() == "python"
                and row.get("label") == "malicious"
                and row.get("source") == "pypi_malregistry_ase2023"
            ):
                code = str(row.get("code") or "")
                split = str(row.get("split") or "")
                family = str(row.get("family") or "")
                strong_count = python_high_confidence_behavior_count(code)
                production_path = _is_production_python_path(
                    str(row.get("file_path") or "")
                )
                if strong_count == 0 or not production_path:
                    changed += 1
                    changed_counts[(split, "needs_review")] += 1
                    changed_families[split].add(family)
                    row["original_label"] = row.get("label")
                    row["original_review_status"] = row.get("review_status")
                    row["original_label_confidence"] = row.get(
                        "label_confidence"
                    )
                    row["review_status"] = "needs_review"
                    row["label_confidence"] = 0.50
                    row["label_basis"] = (
                        "Malicious distribution provenance, but this file is "
                        "non-production/non-Python or lacks an independently "
                        "identifiable code-local malicious behavior chain; "
                        "retained for package-level review."
                    )
                    labels = list(row.get("behavior_labels") or [])
                    labels.extend([
                        (
                            "production_python_path:"
                            f"{str(production_path).lower()}"
                        ),
                        (
                            "python_high_confidence_behavior_groups:"
                            f"{strong_count}"
                        ),
                        "package_only_positive",
                    ])
                    row["behavior_labels"] = sorted(set(map(str, labels)))
                    note = str(row.get("review_notes") or "").strip()
                    migration_note = (
                        "v50 Python label refinement: row and original "
                        "malicious label metadata preserved; file-level "
                        "supervised use disabled pending analyst review."
                    )
                    row["review_notes"] = f"{note} {migration_note}".strip()
                else:
                    retained_counts[split] += 1
                    retained_families[split].add(family)
                    labels = list(row.get("behavior_labels") or [])
                    labels.append(
                        "python_high_confidence_behavior_groups:"
                        f"{strong_count}"
                    )
                    row["behavior_labels"] = sorted(set(map(str, labels)))

            source_name = str(row.get("source") or "")
            family = str(row.get("family") or "")
            split = str(row.get("split") or "")
            digest = str(
                row.get("sample_hash") or row.get("artifact_sha256") or ""
            )
            if family:
                family_splits[(source_name, family)].add(split)
            if digest:
                hash_splits[digest].add(split)
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary.replace(output)
    family_leaks = [
        {
            "source": source,
            "family": family,
            "splits": sorted(splits),
        }
        for (source, family), splits in family_splits.items()
        if len(splits) > 1
    ]
    hash_leaks = [
        {"sample_hash": digest, "splits": sorted(splits)}
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    ]
    if family_leaks or hash_leaks:
        raise RuntimeError("refinement introduced or inherited split leakage")

    report = {
        "schema_version": 1,
        "objective": (
            "convert malicious-package provenance into standalone Python "
            "file-level supervised labels"
        ),
        "base_dataset": str(base.resolve()),
        "base_dataset_sha256": _sha256(base),
        "output_dataset": str(output.resolve()),
        "output_dataset_sha256": _sha256(output),
        "rows_preserved": rows,
        "rows_deleted": 0,
        "rows_marked_needs_review": changed,
        "changed_counts": [
            {
                "split": split,
                "status": status,
                "rows": count,
                "families": len(changed_families[split]),
            }
            for (split, status), count in sorted(changed_counts.items())
        ],
        "retained_supervised_positive_counts": [
            {
                "split": split,
                "rows": retained_counts[split],
                "families": len(retained_families[split]),
            }
            for split in ("train", "validation", "test")
        ],
        "selection_rule": (
            "production .py/.pyw path plus >=1 high-confidence code-local "
            "Python behavior chain; repository/package names are not features"
        ),
        "family_split_isolation_verified": True,
        "hash_split_isolation_verified": True,
        "family_split_leaks": [],
        "hash_split_leaks": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        refine(
            args.base_dataset.resolve(),
            args.output_dataset.resolve(),
            args.report.resolve(),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
