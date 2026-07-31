"""Merge source JSONL datasets with split, family, and label validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def merge(inputs: list[Path], output: Path, report: Path) -> dict[str, Any]:
    rows = []
    content_labels: dict[str, str] = {}
    family_splits: dict[str, set[str]] = defaultdict(set)
    for path in inputs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                digest = str(row.get("sample_hash") or "")
                label = str(row.get("label") or "")
                if digest and digest in content_labels and content_labels[digest] != label:
                    raise RuntimeError(
                        f"Conflicting labels for content hash {digest}"
                    )
                if digest:
                    content_labels[digest] = label
                family = str(row.get("family") or "")
                split = str(row.get("split") or "")
                if family:
                    family_splits[family].add(split)
                rows.append(row)
    leaks = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    if leaks:
        raise RuntimeError(f"Family split leakage detected: {len(leaks)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(
            rows,
            key=lambda value: (
                str(value.get("language")),
                str(value.get("split")),
                str(value.get("label")),
                str(value.get("family")),
                str(value.get("file_path")),
            ),
        ):
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    counts = Counter(
        (
            str(row.get("language")),
            str(row.get("split")),
            str(row.get("label")),
        )
        for row in rows
    )
    projects = Counter(
        (
            str(row.get("language")),
            str(row.get("split")),
            str(row.get("label")),
            str(row.get("family")),
        )
        for row in rows
    )
    result = {
        "schema_version": 1,
        "inputs": [str(path.resolve()) for path in inputs],
        "output": str(output.resolve()),
        "rows": [
            {"language": key[0], "split": key[1], "label": key[2], "count": value}
            for key, value in sorted(counts.items())
        ],
        "projects": [
            {
                "language": key[0],
                "split": key[1],
                "label": key[2],
                "count": sum(1 for item in projects if item[:3] == key),
            }
            for key in sorted({item[:3] for item in projects})
        ],
        "family_split_leakage": [],
        "content_label_conflicts": 0,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        merge(
            [path.resolve() for path in args.input],
            args.output.resolve(),
            args.report.resolve(),
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
