"""Remove VX C/C++ rows whose signals exist only in comments/provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_xgb_incoming_multilingual_dataset import (
    Collector,
    _audit,
    _load_existing,
    _vx_signal_score,
)


def filter_dataset(
    input_dataset: Path,
    output_dataset: Path,
    report_path: Path,
) -> dict[str, object]:
    rows, _ = _load_existing(input_dataset)
    kept = []
    removed: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if (
            row.get("source") == "vx_underground_malware_source"
            and row.get("language") in {"c", "cpp"}
            and row.get("label") == "malicious"
            and _vx_signal_score(str(row.get("code") or "")) < 2
        ):
            removed[(str(row.get("language")), str(row.get("split")))] += 1
            continue
        kept.append(row)

    family_splits: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in kept:
        family_splits[(
            str(row.get("source") or ""),
            str(row.get("language") or ""),
            str(row.get("family") or ""),
        )].add(str(row.get("split") or ""))
    leaks = [
        {
            "source": source,
            "language": language,
            "family": family,
            "splits": sorted(splits),
        }
        for (source, language, family), splits in family_splits.items()
        if len(splits) > 1
    ]
    if leaks:
        raise ValueError("family split leakage detected after VX filtering")

    empty = Collector({}, 12_000)
    report = _audit(kept, empty, input_dataset, output_dataset)
    report.update({
        "filter": "vx_c_cpp_comment_stripped_signal_groups_gte_2",
        "input_rows": len(rows),
        "removed_rows": len(rows) - len(kept),
        "removed_by_language_split": [
            {"language": language, "split": split, "rows": count}
            for (language, split), count in sorted(removed.items())
        ],
        "family_split_isolation_verified": True,
        "family_split_leaks": [],
        "validation_and_test_filtered_by_file_local_label_policy": True,
    })

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with output_dataset.open("w", encoding="utf-8", newline="\n") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        filter_dataset(
            args.input_dataset.resolve(),
            args.output_dataset.resolve(),
            args.report.resolve(),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
