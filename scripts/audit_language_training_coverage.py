"""Audit per-language training coverage for both XGBoost tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset


def audit(dataset: Path) -> dict:
    rows = load_dataset(dataset)
    tasks = {
        "malicious_intent": "malicious",
        "vulnerability_risk": "vulnerable",
    }
    result = {
        "dataset": str(dataset.resolve()),
        "total_rows": len(rows),
        "languages": sorted({row.language for row in rows}),
        "tasks": {},
    }
    for task, positive in tasks.items():
        by_language: dict[str, dict] = {}
        for language in sorted({row.language for row in rows}):
            eligible = [
                row
                for row in rows
                if row.language == language and is_task_training_eligible(row, task)
            ]
            if not eligible:
                continue
            splits: dict[str, dict] = {}
            for split in ("train", "validation", "test"):
                partition = [row for row in eligible if row.split == split]
                labels = Counter(row.label for row in partition)
                families = Counter(
                    row.family or row.package_name or row.sample_hash
                    for row in partition
                )
                positive_families = len(
                    {
                        row.family or row.package_name or row.sample_hash
                        for row in partition
                        if row.label == positive
                    }
                )
                splits[split] = {
                    "rows": len(partition),
                    "benign": labels.get("benign", 0),
                    "positive": labels.get(positive, 0),
                    "families": len(families),
                    "positive_families": positive_families,
                    "sources": Counter(row.source for row in partition),
                }
                splits[split]["sources"] = dict(sorted(splits[split]["sources"].items()))
            by_language[language] = splits
        result["tasks"][task] = by_language
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit(args.dataset)
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
