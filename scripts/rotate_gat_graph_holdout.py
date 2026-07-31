"""Create a fresh development split plus a sealed graph-level audit holdout.

Only graphs that belonged to the former training split are eligible for the
new validation, development-test, and audit cohorts.  Former validation/test
graphs are moved to training, so the final audit does not reuse a cohort that
was inspected while features and hyperparameters were developed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def rotate(
    source: Path,
    development_output: Path,
    audit_output: Path,
    report_path: Path,
    *,
    language: str,
    seed: str,
    validation_per_class: int = 10,
    test_per_class: int = 10,
    audit_per_class: int = 10,
    excluded_audit_paths: list[Path] | None = None,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in source.open(encoding="utf-8")
        if line.strip()
    ]
    labels = ("benign", "malicious")
    training_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("label") in labels and record.get("split") == "train":
            training_candidates[str(record["label"])].append(record)

    excluded_audit_ids = {
        str(record.get("graph_id") or "")
        for path in (excluded_audit_paths or [])
        for record in _read_jsonl(path)
    }
    assignments: dict[str, str] = {}
    required = validation_per_class + test_per_class + audit_per_class
    for label in labels:
        candidates = sorted(
            training_candidates[label],
            key=lambda record: _key(seed, label, str(record.get("graph_id") or "")),
        )
        if len(candidates) < required:
            raise RuntimeError(
                f"{language}/{label}: need {required} former-training graphs, "
                f"found {len(candidates)}"
            )
        audit_candidates = [
            record for record in candidates
            if str(record.get("graph_id") or "") not in excluded_audit_ids
        ]
        if len(audit_candidates) < audit_per_class:
            raise RuntimeError(
                f"{language}/{label}: need {audit_per_class} fresh audit graphs, "
                f"found {len(audit_candidates)}"
            )
        selected_audit = audit_candidates[:audit_per_class]
        selected_audit_ids = {
            str(record.get("graph_id") or "") for record in selected_audit
        }
        remaining = [
            record for record in candidates
            if str(record.get("graph_id") or "") not in selected_audit_ids
        ]
        selected_test = remaining[:test_per_class]
        selected_validation = remaining[
            test_per_class:test_per_class + validation_per_class
        ]
        for split, selected in (
            ("audit", selected_audit),
            ("test", selected_test),
            ("validation", selected_validation),
        ):
            for record in selected:
                assignments[str(record["graph_id"])] = split

    development: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    origin_counts: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if record.get("label") not in labels:
            continue
        graph_id = str(record.get("graph_id") or "")
        original_split = str(record.get("split") or "")
        destination = assignments.get(graph_id, "train")
        normalized = dict(record)
        normalized["split"] = "test" if destination == "audit" else destination
        normalized["holdout_rotation"] = {
            "seed": seed,
            "original_split": original_split,
            "cohort": destination,
        }
        origin_counts[(str(record["label"]), original_split, destination)] += 1
        (audit if destination == "audit" else development).append(normalized)

    _write_jsonl(development_output, development)
    _write_jsonl(audit_output, audit)
    report = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "language": language,
        "seed": seed,
        "policy": (
            "New validation, development-test, and sealed audit cohorts are "
            "selected only from the former training split."
        ),
        "development_counts": _counts(development),
        "audit_counts": _counts(audit),
        "origin_matrix": [
            {
                "label": label,
                "original_split": original,
                "destination": destination,
                "graphs": count,
            }
            for (label, original, destination), count in sorted(origin_counts.items())
        ],
        "audit_graph_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(record["graph_id"]) for record in audit)).encode()
        ).hexdigest(),
        "prior_audit_graphs_excluded_from_new_audit": len(excluded_audit_ids),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _key(seed: str, label: str, graph_id: str) -> str:
    return hashlib.sha256(f"{seed}|{label}|{graph_id}".encode()).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in sorted(
            records,
            key=lambda value: (
                str(value.get("split")),
                str(value.get("label")),
                str(value.get("graph_id")),
            ),
        ):
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = defaultdict(dict)
    counts = Counter(
        (str(record.get("split")), str(record.get("label")))
        for record in records
    )
    for (split, label), count in sorted(counts.items()):
        values[split][label] = count
    return dict(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--development-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--seed", default="20260726-bash-final-audit")
    parser.add_argument("--validation-per-class", type=int, default=10)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--audit-per-class", type=int, default=10)
    parser.add_argument(
        "--exclude-audit",
        action="append",
        default=[],
        type=Path,
        help="Prior audit graph JSONL whose graph IDs cannot enter the new audit.",
    )
    args = parser.parse_args()
    print(json.dumps(
        rotate(
            args.source.resolve(),
            args.development_output.resolve(),
            args.audit_output.resolve(),
            args.report.resolve(),
            language=args.language.strip().lower(),
            seed=args.seed,
            validation_per_class=args.validation_per_class,
            test_per_class=args.test_per_class,
            audit_per_class=args.audit_per_class,
            excluded_audit_paths=[
                path.resolve() for path in args.exclude_audit
            ],
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
