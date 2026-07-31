"""Explain false negatives/positives for a hybrid language candidate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy

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
from attack_detection.features.static_features import (
    FEATURE_NAMES,
    extract_static_features,
    feature_vector,
)


def _text(code: str, language: str, text_transform: str) -> str:
    if text_transform == BEHAVIOR_TOKEN_VERSION_V3:
        return behavior_token_text_v3(code, language)
    if text_transform == BEHAVIOR_TOKEN_VERSION_V2:
        return behavior_token_text_v2(code, language)
    if text_transform == BEHAVIOR_TOKEN_VERSION:
        return behavior_token_text(code, language)
    return code


def _scores(bundle: dict[str, Any], rows: list[Any]) -> list[float]:
    from scipy.sparse import csr_matrix, hstack

    feature_names = bundle.get("feature_names") or FEATURE_NAMES
    structured = numpy.asarray([
        feature_vector(
            row.code,
            row.language,
            feature_names=feature_names,
            include_rules=False,
        )
        for row in rows
    ], dtype="float32")
    if str(bundle.get("feature_mode") or "hybrid_hash").startswith("structured"):
        matrix = csr_matrix(structured, dtype="float32")
    else:
        word = bundle.get("word_vectorizer")
        char = bundle.get("char_vectorizer")
        if word is None or char is None:
            raise KeyError("hybrid candidate is missing its text vectorizers")
        text_transform = str(bundle.get("text_transform") or "raw")
        texts = [
            _text(row.code, row.language, text_transform)
            for row in rows
        ]
        matrix = hstack([
            csr_matrix(structured),
            word.transform(texts),
            char.transform(texts),
        ], format="csr", dtype="float32")
    return [float(item[1]) for item in bundle["model"].predict_proba(matrix)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    from joblib import load

    rows = [
        row for row in load_dataset(args.dataset)
        if row.language == args.language
        and row.split == args.split
        and row.label in {"benign", "malicious"}
        and is_task_training_eligible(row, "malicious_intent")
    ]
    bundle = load(args.artifact)
    threshold = (
        float(bundle.get("threshold") or 0.5)
        if args.threshold is None
        else float(args.threshold)
    )
    scores = _scores(bundle, rows)
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        families[row.family or row.package_name or row.sample_hash].append(index)

    false_negatives = [
        index for index, row in enumerate(rows)
        if row.label == "malicious" and scores[index] < threshold
    ]
    false_positives = [
        index for index, row in enumerate(rows)
        if row.label == "benign" and scores[index] >= threshold
    ]
    rescued = 0
    signal_counts = Counter()
    for index in false_negatives:
        row = rows[index]
        family_key = row.family or row.package_name or row.sample_hash
        if max(scores[item] for item in families[family_key]) >= threshold:
            rescued += 1
        features = extract_static_features(row.code, row.language, include_rules=False)
        if float(features.get("behavior_chain_count") or 0):
            signal_counts["behavior_chain"] += 1
        if float(features.get("process_execution_count") or 0):
            signal_counts["process_execution"] += 1
        if float(features.get("network_api_count") or 0):
            signal_counts["network"] += 1
        if float(features.get("file_write_count") or 0):
            signal_counts["file_write"] += 1
        if row.behavior_labels:
            signal_counts["has_behavior_labels"] += 1
        if len(row.code) < 300:
            signal_counts["short_under_300_chars"] += 1

    def grouped(indices: list[int], field: str) -> list[tuple[str, int]]:
        return Counter(str(getattr(rows[index], field) or "unknown") for index in indices).most_common(20)

    result = {
        "language": args.language,
        "split": args.split,
        "threshold": threshold,
        "samples": len(rows),
        "malicious": sum(row.label == "malicious" for row in rows),
        "benign": sum(row.label == "benign" for row in rows),
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "false_negatives_rescued_by_family_max": rescued,
        "false_negative_signal_counts": dict(signal_counts),
        "false_negatives_by_category": grouped(false_negatives, "category"),
        "false_negatives_by_family": grouped(false_negatives, "family"),
        "false_negatives_by_file_path": grouped(false_negatives, "file_path"),
        "false_positives_by_source": grouped(false_positives, "source"),
        "false_positives_by_category": grouped(false_positives, "category"),
        "false_negative_examples": [
            {
                "score": round(scores[index], 6),
                "family": rows[index].family,
                "package_name": rows[index].package_name,
                "category": rows[index].category,
                "file_path": rows[index].file_path,
                "behavior_labels": list(rows[index].behavior_labels),
                "code_length": len(rows[index].code),
                "preview": rows[index].code[:500],
            }
            for index in sorted(false_negatives, key=lambda item: scores[item])[:20]
        ],
        "false_positive_examples": [
            {
                "score": round(scores[index], 6),
                "family": rows[index].family,
                "package_name": rows[index].package_name,
                "category": rows[index].category,
                "source": rows[index].source,
                "file_path": rows[index].file_path,
                "code_length": len(rows[index].code),
                "preview": rows[index].code[:500],
            }
            for index in sorted(
                false_positives,
                key=lambda item: scores[item],
                reverse=True,
            )[:20]
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
