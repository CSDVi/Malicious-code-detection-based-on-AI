"""Write release metrics and a preserved-candidate inventory for reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GATE = {
    "min_precision": 0.90,
    "max_false_positive_rate": 0.10,
    "max_false_negative_rate": 0.10,
}


def _passes(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    try:
        return (
            float(metrics["precision"]) >= GATE["min_precision"]
            and float(metrics["false_positive_rate"]) <= GATE["max_false_positive_rate"]
            and float(metrics["false_negative_rate"]) <= GATE["max_false_negative_rate"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _metric_columns(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_samples": int(metrics.get("samples") or 0),
        f"{prefix}_accuracy": metrics.get("accuracy"),
        f"{prefix}_precision": metrics.get("precision"),
        f"{prefix}_recall": metrics.get("recall"),
        f"{prefix}_f1": metrics.get("f1"),
        f"{prefix}_fpr": metrics.get("false_positive_rate"),
        f"{prefix}_fnr": metrics.get("false_negative_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--inventory-json", required=True, type=Path)
    args = parser.parse_args()

    selected = []
    for value in args.candidate:
        language, separator, prefix = value.partition("=")
        if not separator:
            raise SystemExit(f"invalid candidate mapping: {value}")
        candidate = Path(prefix)
        metrics = json.loads(
            candidate.with_suffix(".json").read_text(encoding="utf-8")
        )
        validation = dict((metrics.get("selected") or {}).get("validation") or {})
        test = dict(metrics.get("test") or {})
        canary = dict(metrics.get("behavior_canary") or {})
        release_eligible = (
            _passes(validation)
            and _passes(test)
            and canary.get("all_canaries_correct") is True
        )
        selected.append({
            "language": language,
            "candidate": str(candidate.resolve()),
            "feature_mode": metrics.get("feature_mode"),
            "text_transform": metrics.get("text_transform"),
            "threshold": (metrics.get("selected") or {}).get(
                "thresholds", {}
            ).get("decision"),
            **_metric_columns("validation", validation),
            **_metric_columns("test", test),
            "behavior_canaries_correct": canary.get("all_canaries_correct"),
            "release_eligible": release_eligible,
            "candidate_published": metrics.get("published") is True,
        })
    if not all(row["release_eligible"] for row in selected):
        raise SystemExit("at least one selected candidate fails the release gate")

    inventory = []
    for metrics_path in args.candidate_root.rglob("*.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metrics.get("task") != "malicious_intent" or not metrics.get("language"):
            continue
        validation = (metrics.get("selected") or {}).get("validation")
        test = metrics.get("test")
        inventory.append({
            "language": metrics.get("language"),
            "candidate": str(metrics_path.with_suffix("").resolve()),
            "validation_gate_passed": _passes(validation),
            "test_gate_passed": _passes(test),
            "behavior_canaries_correct": (
                (metrics.get("behavior_canary") or {}).get(
                    "all_canaries_correct"
                )
            ),
            "published": metrics.get("published") is True,
        })

    summary = {
        "quality_gate": GATE,
        "selected_language_count": len(selected),
        "all_selected_release_eligible": True,
        "selected": sorted(selected, key=lambda row: row["language"]),
        "candidate_inventory": {
            "total": len(inventory),
            "validation_and_test_passed": sum(
                row["validation_gate_passed"] and row["test_gate_passed"]
                for row in inventory
            ),
            "failed_validation_or_test": sum(
                not (
                    row["validation_gate_passed"]
                    and row["test_gate_passed"]
                )
                for row in inventory
            ),
            "published_candidate_files": sum(
                row["published"] for row in inventory
            ),
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.inventory_json.write_text(
        json.dumps(
            {"quality_gate": GATE, "candidates": inventory},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fieldnames = list(sorted(selected, key=lambda row: row["language"])[0])
    with args.csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda row: row["language"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
