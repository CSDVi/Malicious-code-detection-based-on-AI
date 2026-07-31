"""Re-select a hybrid XGBoost threshold from validation data only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset
from attack_detection.features.static_features import feature_vector
from attack_detection.trainer import _evaluate, _threshold


def _digest(row: Any) -> str:
    return hashlib.sha256(
        (row.sample_hash or row.code[:256]).encode("utf-8", errors="ignore")
    ).hexdigest()


def _partitions(
    dataset: Path,
    task: str,
    language: str,
    positive: str,
    negative: str,
    sampling: dict[str, Any],
) -> dict[str, list[Any]]:
    records = [
        row for row in load_dataset(dataset)
        if row.language == language
        and row.label in {negative, positive}
        and is_task_training_eligible(row, task)
    ]
    cap_value = sampling.get("max_per_language_split_class")
    cap = int(cap_value) if cap_value else None
    balance_all = sampling.get("balanced_split_classes") is True
    balance_eval = sampling.get("balanced_eval_classes") is True
    partitions: dict[str, list[Any]] = {}
    for split in ("validation", "test"):
        grouped = {
            label: sorted(
                [row for row in records if row.split == split and row.label == label],
                key=_digest,
            )
            for label in (negative, positive)
        }
        limits = {
            label: min(len(rows), cap) if cap else len(rows)
            for label, rows in grouped.items()
        }
        if balance_all or balance_eval:
            target = min(limits.values())
            limits = {label: target for label in limits}
        partitions[split] = [
            row
            for label, rows in grouped.items()
            for row in rows[:limits[label]]
        ]
    return partitions


def recalibrate(
    dataset: Path,
    prefix: Path,
    target_precision: float,
    target_fpr: float,
    target_fnr: float,
    plateau_position: float,
    decision_threshold: float | None = None,
) -> dict[str, Any]:
    from joblib import dump, load
    from scipy.sparse import csr_matrix, hstack

    metrics_path = prefix.with_suffix(".json")
    artifact_path = prefix.with_suffix(".joblib")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bundle = load(artifact_path)
    task = str(metrics["task"])
    language = str(metrics["language"])
    positive = str(metrics["positive_label"])
    negative = str(metrics["negative_label"])
    partitions = _partitions(
        dataset,
        task,
        language,
        positive,
        negative,
        dict(metrics.get("sampling_protocol") or {}),
    )
    feature_names = list(bundle.get("feature_names") or [])
    scores: dict[str, list[float]] = {}
    labels: dict[str, list[str]] = {}
    for split, rows in partitions.items():
        structured = numpy.asarray([
            feature_vector(
                row.code,
                row.language,
                feature_names=feature_names,
                include_rules=False,
            )
            for row in rows
        ], dtype="float32")
        if bundle.get("feature_mode") == "structured_static":
            matrix = csr_matrix(structured, dtype="float32")
        else:
            matrix = hstack([
                csr_matrix(structured),
                bundle["word_vectorizer"].transform([row.code for row in rows]),
                bundle["char_vectorizer"].transform([row.code for row in rows]),
            ], format="csr", dtype="float32")
        scores[split] = [
            float(value[1])
            for value in bundle["model"].predict_proba(matrix)
        ]
        labels[split] = [row.label for row in rows]

    if decision_threshold is None:
        thresholds = _threshold(
            labels["validation"],
            scores["validation"],
            positive,
            target_fpr,
            target_precision=target_precision,
            target_fnr=target_fnr,
            plateau_position=plateau_position,
        )
        selection_method = "validation_grid"
    else:
        decision_threshold = min(1.0, max(0.0, float(decision_threshold)))
        validation_at_threshold = _evaluate(
            labels["validation"],
            scores["validation"],
            positive,
            negative,
            decision_threshold,
        )
        thresholds = {
            "decision": decision_threshold,
            "uncertain_low": round(max(0.05, decision_threshold - 0.1), 8),
            "uncertain_high": round(min(0.95, decision_threshold + 0.05), 8),
            "validation_fpr": validation_at_threshold["false_positive_rate"],
            "validation_precision": validation_at_threshold["precision"],
            "validation_fnr": validation_at_threshold["false_negative_rate"],
            "validation_recall": validation_at_threshold["recall"],
            "validation_f1": validation_at_threshold["f1"],
            "target_fpr": target_fpr,
            "target_precision": target_precision,
            "target_fnr": target_fnr,
            "plateau_position": plateau_position,
            "quality_gate_passed": (
                float(validation_at_threshold["precision"]) >= target_precision
                and float(validation_at_threshold["false_positive_rate"]) <= target_fpr
                and float(validation_at_threshold["false_negative_rate"]) <= target_fnr
            ),
        }
        selection_method = "fixed_validation_and_canary_threshold"
    decision = float(thresholds["decision"])
    validation = _evaluate(
        labels["validation"],
        scores["validation"],
        positive,
        negative,
        decision,
    )
    test = _evaluate(
        labels["test"],
        scores["test"],
        positive,
        negative,
        decision,
    )
    metrics["selected"]["thresholds"] = thresholds
    metrics["selected"]["validation"] = validation
    metrics["test"] = test
    metrics["threshold_selection_targets"] = {
        "precision": target_precision,
        "false_positive_rate": target_fpr,
        "false_negative_rate": target_fnr,
        "selection_target_passed": thresholds["quality_gate_passed"],
        "public_validation_gate_passed": validation["quality_gate_passed"],
        "selection_method": selection_method,
    }
    metrics["published"] = False
    bundle["threshold"] = decision

    temporary_artifact = artifact_path.with_suffix(".joblib.tmp")
    dump(bundle, temporary_artifact)
    os.replace(temporary_artifact, artifact_path)
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_metrics, metrics_path)
    return {
        "candidate": str(prefix.resolve()),
        "thresholds": thresholds,
        "validation": validation,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--target-fnr", type=float, default=0.10)
    parser.add_argument("--plateau-position", type=float, default=0.5)
    parser.add_argument("--decision-threshold", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(
        recalibrate(
            args.dataset,
            args.candidate,
            args.target_precision,
            args.target_fpr,
            args.target_fnr,
            args.plateau_position,
            args.decision_threshold,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
