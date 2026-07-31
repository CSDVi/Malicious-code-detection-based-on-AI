"""Append the current training-only behavior augmentation without rebuilding.

Existing hashes are preserved and deduplicated.  This is useful when only the
curated templates changed; validation and test rows remain byte-for-byte
identical to the audited base dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_xgb_incoming_multilingual_dataset import (
    Collector,
    _audit,
    _load_existing,
    ingest_curated_behavior_augmentation,
)


def append(
    base_dataset: Path,
    output_dataset: Path,
    report_path: Path,
    max_code_chars: int,
) -> dict[str, object]:
    base_rows, existing = _load_existing(base_dataset)
    normalized_augmentation_rows = 0
    for row in base_rows:
        if row.get("source") != "curated_behavior_augmentation":
            continue
        if row.get("review_status") != "source_verified":
            normalized_augmentation_rows += 1
        row["review_status"] = "source_verified"
        row["training_only_augmentation"] = True
    collector = Collector(existing, max_code_chars)
    ingest_curated_behavior_augmentation(Path(), collector)
    report = _audit(base_rows, collector, base_dataset, output_dataset)
    report["augmentation_only"] = True
    report["validation_and_test_inherited_unchanged"] = True
    report["normalized_training_augmentation_rows"] = normalized_augmentation_rows
    if not report["family_split_isolation_verified"]:
        raise ValueError("augmentation family leakage detected")
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with output_dataset.open("w", encoding="utf-8", newline="\n") as stream:
        for row in base_rows + collector.rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
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
    parser.add_argument("--max-code-chars", type=int, default=12_000)
    args = parser.parse_args()
    print(json.dumps(
        append(
            args.base_dataset.resolve(),
            args.output_dataset.resolve(),
            args.report.resolve(),
            max(1000, args.max_code_chars),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
