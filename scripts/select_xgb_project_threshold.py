"""Select a project/package XGBoost threshold from validation data only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset
from attack_detection.features.static_features import feature_vector
from attack_detection.trainer import QUALITY_GATE, _evaluate

TASK_LABELS = {
    "malicious_intent": ("malicious", "benign"),
    "vulnerability_risk": ("vulnerable", "benign"),
}


def _passes(report: dict[str, Any]) -> bool:
    return (
        float(report.get("precision", 0.0)) >= float(QUALITY_GATE["min_precision"])
        and float(report.get("false_positive_rate", 1.0))
        <= float(QUALITY_GATE["max_false_positive_rate"])
        and float(report.get("false_negative_rate", 1.0))
        <= float(QUALITY_GATE["max_false_negative_rate"])
    )


def _scores(bundle: dict[str, Any], rows: list[Any]) -> list[float]:
    from scipy.sparse import csr_matrix, hstack

    structured = numpy.asarray(
        [feature_vector(row.code, row.language, include_rules=False) for row in rows],
        dtype="float32",
    )
    matrix = hstack(
        [
            csr_matrix(structured),
            bundle["word_vectorizer"].transform([row.code for row in rows]),
            bundle["char_vectorizer"].transform([row.code for row in rows]),
        ],
        format="csr",
        dtype="float32",
    )
    return [float(item[1]) for item in bundle["model"].predict_proba(matrix)]


def _aggregate(
    rows: list[Any], scores: list[float], positive: str
) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.family or row.package_name or row.sample_hash].append(index)
    labels = [
        positive if any(rows[index].label == positive for index in indices) else "benign"
        for indices in grouped.values()
    ]
    aggregated_scores = [max(scores[index] for index in indices) for indices in grouped.values()]
    return labels, aggregated_scores


def select(dataset: Path, artifact: Path, task: str, language: str) -> dict[str, Any]:
    from joblib import load

    positive, negative = TASK_LABELS[task]
    rows = [
        sample
        for sample in load_dataset(dataset)
        if is_task_training_eligible(sample, task)
        and sample.language == language
        and sample.label in {positive, negative}
    ]
    partitions = {
        split: [sample for sample in rows if sample.split == split]
        for split in ("validation", "test")
    }
    bundle = load(artifact)
    file_scores = {
        split: _scores(bundle, partition)
        for split, partition in partitions.items()
    }
    family = {
        split: _aggregate(partitions[split], file_scores[split], positive)
        for split in ("validation", "test")
    }

    validation_candidates: list[tuple[float, dict[str, Any]]] = []
    all_candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in numpy.linspace(0.01, 0.99, 197):
        report = _evaluate(
            family["validation"][0],
            family["validation"][1],
            positive,
            negative,
            float(threshold),
        )
        item = (float(threshold), report)
        all_candidates.append(item)
        if _passes(report):
            validation_candidates.append(item)

    def passing_rank(item: tuple[float, dict[str, Any]]) -> tuple[float, ...]:
        threshold, report = item
        return (
            float(report["f1"]),
            float(report["precision"]),
            -float(report["false_negative_rate"]),
            -float(report["false_positive_rate"]),
            threshold,
        )

    def fallback_rank(item: tuple[float, dict[str, Any]]) -> tuple[float, ...]:
        threshold, report = item
        deficit = (
            max(0.0, float(QUALITY_GATE["min_precision"]) - float(report["precision"]))
            + max(
                0.0,
                float(report["false_positive_rate"])
                - float(QUALITY_GATE["max_false_positive_rate"]),
            )
            + max(
                0.0,
                float(report["false_negative_rate"])
                - float(QUALITY_GATE["max_false_negative_rate"]),
            )
        )
        return (-deficit, float(report["f1"]), threshold)

    selected = max(
        validation_candidates or all_candidates,
        key=passing_rank if validation_candidates else fallback_rank,
    )
    threshold, validation_report = selected
    test_report = _evaluate(
        family["test"][0],
        family["test"][1],
        positive,
        negative,
        threshold,
    )
    file_validation = _evaluate(
        [row.label for row in partitions["validation"]],
        file_scores["validation"],
        positive,
        negative,
        threshold,
    )
    file_test = _evaluate(
        [row.label for row in partitions["test"]],
        file_scores["test"],
        positive,
        negative,
        threshold,
    )
    return {
        "task": task,
        "language": language,
        "artifact": str(artifact),
        "selection_scope": "project_or_package",
        "selection_policy": "threshold chosen on validation families only; test opened once",
        "quality_gate": QUALITY_GATE,
        "validation_passing_threshold_count": len(validation_candidates),
        "selected_threshold": round(threshold, 6),
        "family_validation": validation_report,
        "family_test": test_report,
        "file_validation_at_selected_threshold": file_validation,
        "file_test_at_selected_threshold": file_test,
        "quality_gate_passed": _passes(validation_report) and _passes(test_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--task", choices=sorted(TASK_LABELS), required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            select(args.dataset, args.artifact, args.task, args.language),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
