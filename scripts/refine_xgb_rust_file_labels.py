"""Downgrade weak repository-only Rust positives without deleting any rows.

The original archive collector allowed one broad signal for Rust.  Generic
``system``, network, password, UI, and WinAPI helper modules therefore became
malicious positives solely because they lived in an offensive repository.
This migration preserves every row and its original label, but marks weak
file-local evidence as needing review so it cannot enter supervised training
or validation/test metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.features.high_confidence_behaviors import (
    rust_high_confidence_behavior_count,
)
from build_xgb_incoming_multilingual_dataset import _vx_signal_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refine(base: Path, output: Path, report_path: Path) -> dict[str, Any]:
    rows = 0
    changed = 0
    changed_counts: Counter[tuple[str, str]] = Counter()
    changed_families: dict[str, set[str]] = defaultdict(set)
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
                str(row.get("language") or "").lower() == "rust"
                and row.get("label") == "malicious"
                and row.get("source") == "github_rust_malicious_candidates"
            ):
                code = str(row.get("code") or "")
                broad_count = _vx_signal_score(code)
                strong_count = rust_high_confidence_behavior_count(code)
                if broad_count < 2 and strong_count == 0:
                    changed += 1
                    split = str(row.get("split") or "")
                    family = str(row.get("family") or "")
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
                        "Repository-level malicious/dual-use provenance but "
                        "insufficient standalone file-local evidence; retained "
                        "for analyst review and excluded from supervised metrics."
                    )
                    labels = list(row.get("behavior_labels") or [])
                    labels.extend([
                        f"file_local_signal_groups:{broad_count}",
                        f"rust_high_confidence_behavior_groups:{strong_count}",
                        "weak_repository_only_positive",
                    ])
                    row["behavior_labels"] = sorted(set(map(str, labels)))
                    note = str(row.get("review_notes") or "").strip()
                    migration_note = (
                        "v43 Rust label refinement: row preserved, original "
                        "malicious label preserved in metadata, supervised use "
                        "disabled pending file-level review."
                    )
                    row["review_notes"] = (
                        f"{note} {migration_note}".strip()
                    )

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
        "objective": "strict file-local Rust malicious-label refinement",
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
        "selection_rule": (
            "github Rust malicious candidate requires >=2 broad file-local "
            "behavior groups or >=1 high-confidence offensive behavior chain"
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
