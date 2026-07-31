"""Select an XGBoost probability ensemble on validation, then test it once."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

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
from attack_detection.trainer import QUALITY_GATE, _evaluate, _threshold


def _text(code: str, language: str, transform: str) -> str:
    if transform == BEHAVIOR_TOKEN_VERSION_V3:
        return behavior_token_text_v3(code, language)
    if transform == BEHAVIOR_TOKEN_VERSION_V2:
        return behavior_token_text_v2(code, language)
    if transform == BEHAVIOR_TOKEN_VERSION:
        return behavior_token_text(code, language)
    return code


def _scores(bundle: dict[str, Any], rows: list[Any]) -> np.ndarray:
    from scipy.sparse import csr_matrix, hstack

    feature_names = tuple(bundle.get("feature_names") or FEATURE_NAMES)
    structured = np.asarray([
        feature_vector(
            row.code,
            row.language,
            feature_names=feature_names,
            include_rules=False,
        )
        for row in rows
    ], dtype="float32")
    feature_mode = str(bundle.get("feature_mode") or "hybrid_hash")
    if feature_mode.startswith("structured"):
        matrix = csr_matrix(structured, dtype="float32")
    else:
        word = bundle.get("word_vectorizer")
        char = bundle.get("char_vectorizer")
        if word is None or char is None:
            raise KeyError("hybrid ensemble component is missing vectorizers")
        transform = str(bundle.get("text_transform") or "raw")
        texts = [_text(row.code, row.language, transform) for row in rows]
        matrix = hstack(
            [
                csr_matrix(structured),
                word.transform(texts),
                char.transform(texts),
            ],
            format="csr",
            dtype="float32",
        )
    return np.asarray(
        [float(item[1]) for item in bundle["model"].predict_proba(matrix)],
        dtype="float64",
    )


def _fuse(first: np.ndarray, second: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    if mode == "minimum":
        return np.minimum(first, second)
    if mode == "maximum":
        return np.maximum(first, second)
    if mode == "geometric":
        epsilon = 1e-9
        return np.exp(
            alpha * np.log(np.maximum(first, epsilon))
            + (1.0 - alpha) * np.log(np.maximum(second, epsilon))
        )
    return alpha * first + (1.0 - alpha) * second


def _deficit(report: dict[str, Any]) -> float:
    return (
        max(0.0, QUALITY_GATE["min_precision"] - float(report["precision"]))
        + max(
            0.0,
            float(report["false_positive_rate"])
            - QUALITY_GATE["max_false_positive_rate"],
        )
        + max(
            0.0,
            float(report["false_negative_rate"])
            - QUALITY_GATE["max_false_negative_rate"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from joblib import load

    rows = [
        row for row in load_dataset(args.dataset)
        if is_task_training_eligible(row, "malicious_intent")
        and row.language == args.language
        and row.label in {"benign", "malicious"}
    ]
    partitions = {
        split: [row for row in rows if row.split == split]
        for split in ("validation", "test")
    }
    labels = {
        split: [row.label for row in partitions[split]]
        for split in ("validation", "test")
    }
    components: list[dict[str, Any]] = []
    for artifact in args.artifact:
        resolved = artifact.resolve()
        bundle = load(resolved)
        components.append({
            "path": str(resolved),
            "bundle": bundle,
            "validation_scores": _scores(bundle, partitions["validation"]),
        })

    validation_candidates: list[dict[str, Any]] = []
    for first_index, second_index in itertools.combinations(
        range(len(components)), 2
    ):
        first = components[first_index]["validation_scores"]
        second = components[second_index]["validation_scores"]
        for mode in ("weighted_mean", "geometric", "minimum", "maximum"):
            alphas = (0.5,) if mode in {"minimum", "maximum"} else (
                0.20, 0.35, 0.50, 0.65, 0.80,
            )
            for alpha in alphas:
                scores = _fuse(first, second, mode, alpha)
                threshold = _threshold(
                    labels["validation"],
                    scores.tolist(),
                    "malicious",
                    QUALITY_GATE["max_false_positive_rate"],
                    target_precision=QUALITY_GATE["min_precision"],
                    target_fnr=QUALITY_GATE["max_false_negative_rate"],
                    plateau_position=0.95,
                )
                report = _evaluate(
                    labels["validation"],
                    scores.tolist(),
                    "malicious",
                    "benign",
                    float(threshold["decision"]),
                )
                positive_scores = scores[
                    np.asarray(labels["validation"]) == "malicious"
                ]
                negative_scores = scores[
                    np.asarray(labels["validation"]) == "benign"
                ]
                separation_margin = float(
                    np.min(positive_scores) - np.max(negative_scores)
                )
                validation_candidates.append({
                    "first_index": first_index,
                    "second_index": second_index,
                    "mode": mode,
                    "alpha": alpha,
                    "threshold": threshold,
                    "validation": report,
                    "validation_deficit": _deficit(report),
                    "validation_separation_margin": separation_margin,
                })

    if not validation_candidates:
        raise SystemExit("at least two --artifact values are required")
    selected = max(
        validation_candidates,
        key=lambda item: (
            bool(item["validation"]["quality_gate_passed"]),
            -float(item["validation_deficit"]),
            float(item["validation"]["f1"]),
            float(item["validation"]["precision"]),
            float(item["validation_separation_margin"]),
        ),
    )

    first_component = components[int(selected["first_index"])]
    second_component = components[int(selected["second_index"])]
    first_test = _scores(first_component["bundle"], partitions["test"])
    second_test = _scores(second_component["bundle"], partitions["test"])
    test_scores = _fuse(
        first_test,
        second_test,
        str(selected["mode"]),
        float(selected["alpha"]),
    )
    test_report = _evaluate(
        labels["test"],
        test_scores.tolist(),
        "malicious",
        "benign",
        float(selected["threshold"]["decision"]),
    )
    result = {
        "language": args.language,
        "dataset": str(args.dataset.resolve()),
        "selection_protocol": (
            "all fusion choices and thresholds selected on validation; "
            "test evaluated once for the selected fusion"
        ),
        "quality_gate": QUALITY_GATE,
        "component_paths": [item["path"] for item in components],
        "selected": {
            "first_component": first_component["path"],
            "second_component": second_component["path"],
            "mode": selected["mode"],
            "alpha": selected["alpha"],
            "threshold": selected["threshold"],
            "validation": selected["validation"],
            "validation_separation_margin": selected[
                "validation_separation_margin"
            ],
        },
        "test": test_report,
        "validation_candidate_count": len(validation_candidates),
        "validation_top_candidates": [
            {
                key: value for key, value in item.items()
                if key not in {"threshold"}
            }
            for item in sorted(
                validation_candidates,
                key=lambda item: (
                    bool(item["validation"]["quality_gate_passed"]),
                    -float(item["validation_deficit"]),
                    float(item["validation"]["f1"]),
                    float(item["validation"]["precision"]),
                    float(item["validation_separation_margin"]),
                ),
                reverse=True,
            )[:10]
        ],
        "quality_gate_passed": bool(
            selected["validation"]["quality_gate_passed"]
            and test_report["quality_gate_passed"]
        ),
        "published": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
