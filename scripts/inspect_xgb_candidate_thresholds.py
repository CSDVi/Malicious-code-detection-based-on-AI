"""Inspect a hybrid candidate's fixed-threshold tradeoff without publishing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset
from attack_detection.features.behavior_tokens import (
    BEHAVIOR_TOKEN_VERSION,
    BEHAVIOR_TOKEN_VERSION_V2,
    BEHAVIOR_TOKEN_VERSION_V3,
    behavior_token_text,
    behavior_token_text_v2,
    behavior_token_text_v3,
)
from attack_detection.features.static_features import FEATURE_NAMES, feature_vector
from attack_detection.trainer import _evaluate


def _text(code: str, language: str, text_transform: str) -> str:
    if text_transform == BEHAVIOR_TOKEN_VERSION_V3:
        return behavior_token_text_v3(code, language)
    if text_transform == BEHAVIOR_TOKEN_VERSION_V2:
        return behavior_token_text_v2(code, language)
    if text_transform == BEHAVIOR_TOKEN_VERSION:
        return behavior_token_text(code, language)
    return code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()

    from scipy.sparse import csr_matrix, hstack
    from joblib import load

    bundle = load(args.artifact)
    rows = [
        sample for sample in load_dataset(args.dataset)
        if is_task_training_eligible(sample, args.task)
        and sample.language == args.language
        and sample.label in {"benign", "malicious", "vulnerable"}
    ]
    feature_mode = str(bundle.get("feature_mode") or "hybrid_hash")
    text_transform = str(bundle.get("text_transform") or "raw")
    feature_names = bundle.get("feature_names") or FEATURE_NAMES
    word = bundle.get("word_vectorizer")
    char = bundle.get("char_vectorizer")
    matrices = {}
    labels = {}
    partitions = {}
    for split in ("validation", "test"):
        partition = [sample for sample in rows if sample.split == split]
        partitions[split] = partition
        structured = np.asarray([
            feature_vector(
                sample.code,
                sample.language,
                feature_names=feature_names,
                include_rules=False,
            )
            for sample in partition
        ], dtype="float32")
        if feature_mode.startswith("structured"):
            matrices[split] = csr_matrix(structured, dtype="float32")
        else:
            if word is None or char is None:
                raise KeyError("hybrid candidate is missing its text vectorizers")
            texts = [
                _text(sample.code, sample.language, text_transform)
                for sample in partition
            ]
            matrices[split] = hstack(
                [
                    csr_matrix(structured),
                    word.transform(texts),
                    char.transform(texts),
                ],
                format="csr",
                dtype="float32",
            )
        labels[split] = [sample.label for sample in partition]

    scores = {
        split: [float(row[1]) for row in bundle["model"].predict_proba(matrices[split])]
        for split in ("validation", "test")
    }
    positive = "malicious" if args.task == "malicious_intent" else "vulnerable"
    thresholds = {
        0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05,
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 0.925, 0.95, 0.975, 0.99,
        float(bundle.get("threshold") or 0.5),
    }
    for threshold in sorted(thresholds):
        family_reports = {}
        for split in ("validation", "test"):
            grouped: dict[str, list[int]] = {}
            for index, sample in enumerate(partitions[split]):
                key = sample.family or sample.package_name or sample.sample_hash
                grouped.setdefault(key, []).append(index)
            family_labels = [
                positive if any(labels[split][index] == positive for index in indices) else "benign"
                for indices in grouped.values()
            ]
            family_scores = [
                max(scores[split][index] for index in indices)
                for indices in grouped.values()
            ]
            family_reports[split] = _evaluate(
                family_labels, family_scores, positive, "benign", threshold,
            )
        print(json.dumps({
            "threshold": threshold,
            "validation": _evaluate(labels["validation"], scores["validation"], positive, "benign", threshold),
            "test": _evaluate(labels["test"], scores["test"], positive, "benign", threshold),
            "family_validation": family_reports["validation"],
            "family_test": family_reports["test"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
