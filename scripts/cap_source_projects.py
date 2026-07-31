"""Select and deterministically re-split a bounded number of source projects."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def cap(
    source: Path,
    output: Path,
    report: Path,
    *,
    language: str,
    train_projects: int,
    validation_projects: int,
    test_projects: int,
    seed: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("language") or "").lower() == language:
                groups[str(row["family"])].append(row)
    required = train_projects + validation_projects + test_projects
    ordered = sorted(
        groups,
        key=lambda family: hashlib.sha256(
            f"{seed}|{family}".encode()
        ).hexdigest(),
    )
    if len(ordered) < required:
        raise RuntimeError(
            f"{language}: need {required} projects, found {len(ordered)}"
        )
    assignments = {}
    cursor = 0
    for split, count in (
        ("train", train_projects),
        ("validation", validation_projects),
        ("test", test_projects),
    ):
        for family in ordered[cursor:cursor + count]:
            assignments[family] = split
        cursor += count

    rows = []
    for family, split in assignments.items():
        for row in groups[family]:
            normalized = dict(row)
            normalized["split"] = split
            rows.append(normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(
            rows,
            key=lambda value: (
                value["split"], value["family"], value["file_path"],
            ),
        ):
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    result = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "language": language,
        "seed": seed,
        "available_projects": len(groups),
        "selected_projects": dict(Counter(assignments.values())),
        "rows": len(rows),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--train-projects", type=int, default=20)
    parser.add_argument("--validation-projects", type=int, default=10)
    parser.add_argument("--test-projects", type=int, default=10)
    parser.add_argument("--seed", default="20260726-source-cap")
    args = parser.parse_args()
    print(json.dumps(
        cap(
            args.source.resolve(),
            args.output.resolve(),
            args.report.resolve(),
            language=args.language.strip().lower(),
            train_projects=args.train_projects,
            validation_projects=args.validation_projects,
            test_projects=args.test_projects,
            seed=args.seed,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
