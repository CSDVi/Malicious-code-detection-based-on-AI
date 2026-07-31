"""Pair source-held-out malicious projects with reviewed benign projects.

This produces a compact task-specific JSONL for GATv2. Vulnerability-labelled
rows are deliberately excluded: the negative class contains only records that
the source dataset already marks benign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_BENIGN_SOURCES = {
    "codesearchnet",
    "crossvul",
    "nist_juliet_csharp_1.3",
    "npm_official_registry",
    "pypi_official_registry",
    "pypi_popular_official",
    "zenodo_13870382",
}
SPLITS = ("validation", "test", "train")


def build(
    base_path: Path,
    malicious_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    malicious_rows = _read_jsonl(malicious_path)
    languages = sorted({str(row.get("language") or "") for row in malicious_rows})
    positive_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in malicious_rows:
        positive_families[(str(row["language"]), str(row["split"]))].add(str(row["family"]))

    benign_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    with base_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            language = str(row.get("language") or "").lower()
            split = str(row.get("split") or "")
            source = str(row.get("source") or "")
            family = str(row.get("family") or row.get("package_name") or "")
            version = str(row.get("version") or "")
            if (
                language not in languages
                or split not in SPLITS
                or row.get("label") != "benign"
                or source not in ALLOWED_BENIGN_SOURCES
                or not family
            ):
                continue
            benign_groups[(language, source, family, version)].append(row)

    selected: list[dict[str, Any]] = []
    selected_identities: set[str] = set()
    negative_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    for language in languages:
        candidates = [
            (key, rows)
            for key, rows in benign_groups.items()
            if key[0] == language
        ]
        candidates.sort(key=lambda item: _sha("|".join(item[0])))
        candidate_index = 0
        for split in SPLITS:
            target = len(positive_families[(language, split)])
            while (
                candidate_index < len(candidates)
                and len(negative_families[(language, split)]) < target
            ):
                key, rows = candidates[candidate_index]
                candidate_index += 1
                _, source, family, _ = key
                identity = f"{language}:{source}:{family}"
                if identity in selected_identities:
                    continue
                selected_identities.add(identity)
                for row in rows:
                    normalized = dict(row)
                    normalized["family"] = f"gat_benign:{identity}"
                    normalized["pair_id"] = normalized["family"]
                    normalized["split"] = split
                    selected.append(normalized)
                negative_families[(language, split)].add(identity)
            if len(negative_families[(language, split)]) < target:
                raise RuntimeError(
                    f"{language}/{split}: need {target} benign projects, found "
                    f"{len(negative_families[(language, split)])}"
                )

    rows = malicious_rows + selected
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in sorted(rows, key=lambda value: (
            str(value.get("language")), str(value.get("split")),
            str(value.get("label")), str(value.get("family")), str(value.get("file_path")),
        )):
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    counts = Counter(
        (str(row.get("language")), str(row.get("split")), str(row.get("label")))
        for row in rows
    )
    project_counts = Counter()
    for (language, split), families in positive_families.items():
        project_counts[(language, split, "malicious")] = len(families)
    for (language, split), families in negative_families.items():
        project_counts[(language, split, "benign")] = len(families)
    report = {
        "schema_version": 1,
        "base_dataset": str(base_path.resolve()),
        "malicious_dataset": str(malicious_path.resolve()),
        "output": str(output_path.resolve()),
        "languages": languages,
        "rows": _nested(counts),
        "projects": _nested(project_counts),
        "negative_policy": "Only source-labelled benign records; vulnerable rows excluded.",
        "family_isolation": "Each selected source:family identity occurs in one split.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _nested(values: Counter[tuple[str, str, str]]) -> dict[str, Any]:
    output: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (language, split, label), count in sorted(values.items()):
        output[language][split][label] = count
    return {
        language: {split: dict(labels) for split, labels in splits.items()}
        for language, splits in output.items()
    }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--malicious", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = build(
        args.base.resolve(),
        args.malicious.resolve(),
        args.output.resolve(),
        args.report.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
