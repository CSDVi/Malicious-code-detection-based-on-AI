"""Train one language-specific hybrid XGBoost candidate.

This is an experiment runner: it never activates a model.  It combines the
project's structured static features with stateless word/character hashing and
uses the existing validation-only threshold policy before touching test data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

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


TASKS = {
    "malicious_intent": ("malicious", "benign"),
    "vulnerability_risk": ("vulnerable", "benign"),
}
HARD_NEGATIVE_PATTERNS = (
    r"\b(?:exec|eval|system|popen|shell_exec|passthru|subprocess|processbuilder)\b",
    r"\b(?:socket|requests?|httpx|fetch|curl|urlopen|webclient)\b",
    r"\b(?:select|insert|update|delete|query|execute|cursor)\b",
    r"\b(?:base64|b64decode|fromcharcode|unescape|decode|fromhex)\b",
    r"\b(?:open|write|writefile|file_put_contents|fopen|unlink|rename|chmod)\b",
    r"\b(?:getenv|environ|process\.env|request|params?|argv|stdin)\b",
)


def _hard_negative_score(code: str) -> int:
    lowered = code.lower()
    return sum(bool(re.search(pattern, lowered)) for pattern in HARD_NEGATIVE_PATTERNS)


def _text(code: str, language: str, text_transform: str) -> str:
    if text_transform == BEHAVIOR_TOKEN_VERSION_V3:
        return behavior_token_text_v3(code, language)
    if text_transform == BEHAVIOR_TOKEN_VERSION_V2:
        return behavior_token_text_v2(code, language)
    if text_transform == BEHAVIOR_TOKEN_VERSION:
        return behavior_token_text(code, language)
    return code


def _matrix(
    samples: list[Any],
    word: Any,
    char: Any,
    feature_mode: str,
    text_transform: str,
    feature_names: tuple[str, ...],
) -> Any:
    import numpy
    from scipy.sparse import csr_matrix, hstack

    texts = [
        _text(sample.code, sample.language, text_transform)
        for sample in samples
    ]
    structured = numpy.asarray([
        feature_vector(
            sample.code,
            sample.language,
            feature_names=feature_names,
            include_rules=False,
        )
        for sample in samples
    ], dtype="float32")
    if feature_mode == "structured":
        return csr_matrix(structured, dtype="float32")
    return hstack(
        [csr_matrix(structured), word.transform(texts), char.transform(texts)],
        format="csr",
        dtype="float32",
    )


def train(
    dataset: Path,
    task: str,
    language: str,
    output: Path,
    max_per_language_split_class: int | None = None,
    single_candidate: bool = False,
    balance_split_classes: bool = False,
    balance_eval_classes: bool = False,
    positive_weight_mode: str = "default",
    positive_weight_multiplier: float = 1.0,
    hard_negative_weight: float = 2.0,
    hard_negative_fraction: float = 1 / 3,
    feature_mode: str = "hybrid",
    positive_source_weights: dict[str, float] | None = None,
    positive_family_weights: dict[str, float] | None = None,
    negative_source_weights: dict[str, float] | None = None,
    negative_family_weights: dict[str, float] | None = None,
    text_transform: str = "raw",
    calibration_mode: str = "sigmoid",
    threshold_target_fpr: float | None = None,
    threshold_target_precision: float | None = None,
    threshold_target_fnr: float | None = None,
    threshold_plateau_position: float = 0.5,
    excluded_structured_features: tuple[str, ...] = (),
) -> dict[str, Any]:
    from joblib import dump
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import HashingVectorizer
    from xgboost import XGBClassifier

    positive, negative = TASKS[task]
    records = [
        sample for sample in load_dataset(dataset)
        if is_task_training_eligible(sample, task)
        and (language == "all" or sample.language == language)
        and sample.label in {positive, negative}
    ]
    if (
        (max_per_language_split_class and max_per_language_split_class > 0)
        or balance_split_classes
        or balance_eval_classes
    ):
        # Keep training bounded while retaining deterministic language/class
        # representation.  Balanced evaluation makes Precision interpretable
        # alongside FPR/FNR instead of letting arbitrary corpus prevalence
        # dominate it.  Families were assigned to splits before this stage.
        import hashlib

        grouped: dict[tuple[str, str, str], list[Any]] = {}
        for sample in records:
            key = (sample.language, sample.split, sample.label)
            grouped.setdefault(key, []).append(sample)
        bounded: list[Any] = []
        limits: dict[tuple[str, str, str], int] = {}
        for key, rows in grouped.items():
            cap = len(rows)
            if max_per_language_split_class and max_per_language_split_class > 0:
                cap = min(cap, max_per_language_split_class)
            limits[key] = cap
        if balance_split_classes or balance_eval_classes:
            for grouped_language in {key[0] for key in grouped}:
                for split in ("train", "validation", "test"):
                    if balance_eval_classes and not balance_split_classes and split == "train":
                        continue
                    keys = [
                        (grouped_language, split, negative),
                        (grouped_language, split, positive),
                    ]
                    target = min(limits.get(key, 0) for key in keys)
                    for key in keys:
                        limits[key] = target
        for key, rows in grouped.items():
            def digest_key(row: Any) -> str:
                return hashlib.sha256(
                    (row.sample_hash or row.code[:256]).encode("utf-8", errors="ignore")
                ).hexdigest()

            limit = limits[key]
            curated = sorted(
                (
                    row for row in rows
                    if row.source == "curated_behavior_augmentation"
                ),
                key=digest_key,
            )[:limit]
            remaining_limit = max(0, limit - len(curated))
            ordinary = [
                row for row in rows
                if row.source != "curated_behavior_augmentation"
            ]
            if key[1] == "train" and key[2] == negative:
                hard = sorted(
                    (
                        row for row in ordinary
                        if _hard_negative_score(row.code) >= 2
                    ),
                    key=digest_key,
                )
                regular = sorted(
                    (
                        row for row in ordinary
                        if _hard_negative_score(row.code) < 2
                    ),
                    key=digest_key,
                )
                hard_limit = min(
                    len(hard),
                    max(1, int(round(remaining_limit * hard_negative_fraction))),
                )
                selected_rows = (
                    curated
                    + hard[:hard_limit]
                    + regular[:remaining_limit - hard_limit]
                )
                if len(selected_rows) < limit:
                    selected_rows.extend(
                        hard[
                            hard_limit:
                            hard_limit + limit - len(selected_rows)
                        ]
                    )
                bounded.extend(selected_rows)
            else:
                bounded.extend(
                    curated + sorted(ordinary, key=digest_key)[:remaining_limit]
                )
        records = bounded
    partitions = {
        split: [sample for sample in records if sample.split == split]
        for split in ("train", "validation", "test")
    }
    for split, rows in partitions.items():
        if len({row.label for row in rows}) != 2:
            raise SystemExit(f"{language}/{task}: {split} does not contain both classes")

    word = HashingVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z_$][A-Za-z0-9_$]*\b|[$_./:+-]+",
        ngram_range=(1, 2),
        n_features=2048,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    char = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=4096,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    excluded_feature_set = set(excluded_structured_features)
    unknown_excluded_features = sorted(excluded_feature_set - set(FEATURE_NAMES))
    if unknown_excluded_features:
        raise SystemExit(
            "unknown excluded structured feature(s): "
            + ", ".join(unknown_excluded_features)
        )
    feature_names = tuple(
        name for name in FEATURE_NAMES
        if name not in excluded_feature_set
    )
    matrices = {
        split: _matrix(
            rows,
            word,
            char,
            feature_mode,
            text_transform,
            feature_names,
        )
        for split, rows in partitions.items()
    }
    labels = {
        split: [1 if row.label == positive else 0 for row in rows]
        for split, rows in partitions.items()
    }
    positive_source_weights = positive_source_weights or {}
    positive_family_weights = positive_family_weights or {}
    negative_source_weights = negative_source_weights or {}
    negative_family_weights = negative_family_weights or {}
    train_weights = []
    for row in partitions["train"]:
        weight = 1.0
        if row.label == negative and _hard_negative_score(row.code) >= 2:
            weight = max(weight, hard_negative_weight)
        if row.label == negative and row.source in negative_source_weights:
            weight = max(weight, float(negative_source_weights[row.source]))
        if row.label == negative and row.family in negative_family_weights:
            weight = max(weight, float(negative_family_weights[row.family]))
        if row.label == positive and row.source in positive_source_weights:
            weight = max(weight, float(positive_source_weights[row.source]))
        if row.label == positive and row.family in positive_family_weights:
            weight = max(weight, float(positive_family_weights[row.family]))
        train_weights.append(weight)
    train_positive = sum(labels["train"])
    train_negative = len(labels["train"]) - train_positive
    ratio = train_negative / max(1, train_positive)
    requested_positive_weight = {
        "default": 1.0,
        "one": 1.0,
        "sqrt": math.sqrt(max(1.0, ratio)),
        "balanced": max(1.0, ratio),
    }[positive_weight_mode] * max(0.1, positive_weight_multiplier)
    candidates = [
        {
            "max_depth": 3,
            "min_child_weight": 1,
            "scale_pos_weight": requested_positive_weight,
        },
        {
            "max_depth": 4,
            "min_child_weight": 2,
            "scale_pos_weight": requested_positive_weight,
        },
        {"max_depth": 5, "min_child_weight": 3, "scale_pos_weight": math.sqrt(max(1.0, ratio))},
        {
            "max_depth": 6,
            "min_child_weight": 1,
            "scale_pos_weight": requested_positive_weight,
        },
    ]
    if single_candidate:
        candidates = candidates[:1]
    trained: list[tuple[tuple[Any, ...], Any, dict[str, Any], dict[str, Any]]] = []
    fallback_mode = language == "all"
    validation_names = [row.label for row in partitions["validation"]]
    for index, params in enumerate(candidates):
        base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=240 if fallback_mode else 600,
            learning_rate=0.05 if fallback_mode else 0.04,
            gamma=0.05,
            subsample=0.9,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=4.0,
            max_delta_step=1.0,
            tree_method="hist",
            early_stopping_rounds=20 if fallback_mode else 35,
            random_state=20260724 + index,
            n_jobs=4 if fallback_mode else 8,
            **params,
        )
        base.fit(
            matrices["train"],
            labels["train"],
            sample_weight=train_weights,
            eval_set=[(matrices["validation"], labels["validation"])],
            verbose=False,
        )
        if calibration_mode == "sigmoid":
            predictor = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
            predictor.fit(matrices["validation"], labels["validation"])
        else:
            predictor = base
        scores = [
            float(row[1])
            for row in predictor.predict_proba(matrices["validation"])
        ]
        target_fpr = (
            float(QUALITY_GATE["max_false_positive_rate"])
            if threshold_target_fpr is None
            else float(threshold_target_fpr)
        )
        target_precision = (
            float(QUALITY_GATE["min_precision"])
            if threshold_target_precision is None
            else float(threshold_target_precision)
        )
        target_fnr = (
            float(QUALITY_GATE["max_false_negative_rate"])
            if threshold_target_fnr is None
            else float(threshold_target_fnr)
        )
        threshold_info = _threshold(
            validation_names,
            scores,
            positive,
            target_fpr,
            target_precision=target_precision,
            target_fnr=target_fnr,
            plateau_position=threshold_plateau_position,
        )
        report = _evaluate(
            validation_names,
            scores,
            positive,
            negative,
            float(threshold_info["decision"]),
        )
        deficit = (
            max(0.0, QUALITY_GATE["min_precision"] - float(report["precision"]))
            + max(0.0, float(report["false_positive_rate"]) - QUALITY_GATE["max_false_positive_rate"])
            + max(0.0, float(report["false_negative_rate"]) - QUALITY_GATE["max_false_negative_rate"])
        )
        rank = (
            bool(report["quality_gate_passed"]),
            -deficit,
            float(report["f1"]),
            float(report["precision"]),
        )
        trained.append((rank, predictor, threshold_info, {
            "candidate": params,
            "validation": report,
            "thresholds": threshold_info,
            "best_iteration": getattr(base, "best_iteration", None),
        }))

    _, model, threshold_info, selected = max(trained, key=lambda item: item[0])
    test_scores = [
        float(row[1]) for row in model.predict_proba(matrices["test"])
    ]
    test_names = [row.label for row in partitions["test"]]
    test_report = _evaluate(
        test_names,
        test_scores,
        positive,
        negative,
        float(threshold_info["decision"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = output.with_suffix(".joblib")
    metrics_path = output.with_suffix(".json")
    result = {
        "task": task,
        "language": language,
        "positive_label": positive,
        "negative_label": negative,
        "dataset": str(dataset.resolve()),
        "evaluation_protocol": (
            "source_heldout"
            if "source_heldout" in dataset.stem
            else "family_split"
        ),
        "feature_mode": (
            "structured_static"
            if feature_mode == "structured"
            else "structured+word_hash_2048+char_hash_4096"
        ),
        "text_transform": text_transform,
        "structured_feature_count": len(feature_names),
        "split_counts": {key: len(value) for key, value in partitions.items()},
        "split_label_counts": {
            split: {
                negative: sum(row.label == negative for row in values),
                positive: sum(row.label == positive for row in values),
            }
            for split, values in partitions.items()
        },
        "sampling_protocol": {
            "max_per_language_split_class": max_per_language_split_class,
            "balanced_split_classes": balance_split_classes,
            "balanced_eval_classes": balance_eval_classes,
            "positive_weight_mode": positive_weight_mode,
            "positive_weight_multiplier": positive_weight_multiplier,
            "threshold_target_fpr": (
                float(QUALITY_GATE["max_false_positive_rate"])
                if threshold_target_fpr is None
                else float(threshold_target_fpr)
            ),
            "threshold_target_precision": (
                float(QUALITY_GATE["min_precision"])
                if threshold_target_precision is None
                else float(threshold_target_precision)
            ),
            "threshold_target_fnr": (
                float(QUALITY_GATE["max_false_negative_rate"])
                if threshold_target_fnr is None
                else float(threshold_target_fnr)
            ),
            "threshold_plateau_position": threshold_plateau_position,
            "excluded_structured_features": sorted(
                excluded_feature_set
            ),
            "calibration_mode": calibration_mode,
            "deterministic": True,
            "family_assignment_precedes_sampling": True,
        },
        "hard_negative_training": {
            "rows": sum(weight > 1.0 for weight in train_weights),
            "weight": hard_negative_weight,
            "fraction_cap": hard_negative_fraction,
            "negative_source_weights": negative_source_weights,
            "negative_family_weights": negative_family_weights,
            "positive_source_weights": positive_source_weights,
            "positive_family_weights": positive_family_weights,
        },
        "quality_gate": QUALITY_GATE,
        "selected": selected,
        "test": test_report,
        "published": False,
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "feature_mode": (
            "structured_static"
            if feature_mode == "structured"
            else "hybrid_hash"
        ),
        "task": task,
        "language": language,
        "threshold": float(threshold_info["decision"]),
        "text_transform": text_transform,
        "calibration_mode": calibration_mode,
    }
    if feature_mode != "structured":
        bundle["word_vectorizer"] = word
        bundle["char_vectorizer"] = char
    dump(bundle, artifact)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument(
        "--max-per-language-split-class",
        type=int,
        default=None,
        help="deterministic cap used by the shared all-language fallback",
    )
    parser.add_argument(
        "--single-candidate",
        action="store_true",
        help="train the consistently stronger depth-4 candidate only",
    )
    parser.add_argument(
        "--balance-split-classes",
        action="store_true",
        help="deterministically balance benign/positive counts inside every split",
    )
    parser.add_argument(
        "--balance-eval-classes",
        action="store_true",
        help="balance validation/test but retain the bounded natural training ratio",
    )
    parser.add_argument(
        "--positive-weight-mode",
        choices=("default", "one", "sqrt", "balanced"),
        default="default",
        help="override the depth-4 candidate positive-class weight",
    )
    parser.add_argument(
        "--positive-weight-multiplier",
        type=float,
        default=1.0,
        help="multiply the selected depth-4 positive-class weight",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--hard-negative-fraction", type=float, default=1 / 3)
    parser.add_argument(
        "--negative-source-weight",
        action="append",
        default=[],
        help="extra negative weight in SOURCE=WEIGHT form; repeatable",
    )
    parser.add_argument(
        "--negative-family-weight",
        action="append",
        default=[],
        help="extra negative weight in FAMILY=WEIGHT form; repeatable",
    )
    parser.add_argument(
        "--positive-source-weight",
        action="append",
        default=[],
        help="extra positive weight in SOURCE=WEIGHT form; repeatable",
    )
    parser.add_argument(
        "--positive-family-weight",
        action="append",
        default=[],
        help="extra positive weight in FAMILY=WEIGHT form; repeatable",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("hybrid", "structured"),
        default="hybrid",
        help="structured-only is useful when text/source tokens cause corpus leakage",
    )
    parser.add_argument(
        "--exclude-structured-feature",
        action="append",
        default=[],
        help="omit one named structured feature from this candidate; repeatable",
    )
    parser.add_argument(
        "--text-transform",
        choices=(
            "raw",
            BEHAVIOR_TOKEN_VERSION,
            BEHAVIOR_TOKEN_VERSION_V2,
            BEHAVIOR_TOKEN_VERSION_V3,
        ),
        default="raw",
        help="append deterministic semantic behavior tokens before text hashing",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("sigmoid", "none"),
        default="sigmoid",
        help="probability calibration; none is safer for very small validation positives",
    )
    parser.add_argument("--threshold-target-fpr", type=float, default=None)
    parser.add_argument("--threshold-target-precision", type=float, default=None)
    parser.add_argument("--threshold-target-fnr", type=float, default=None)
    parser.add_argument("--threshold-plateau-position", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    def _parse_weight_items(items: list[str], flag: str) -> dict[str, float]:
        weights: dict[str, float] = {}
        for item in items:
            if "=" not in item:
                raise SystemExit(f"{flag} must use NAME=WEIGHT")
            name, raw_weight = item.split("=", 1)
            name = name.strip()
            if not name:
                raise SystemExit(f"{flag} name cannot be empty")
            weights[name] = max(1.0, float(raw_weight))
        return weights

    negative_source_weights = _parse_weight_items(args.negative_source_weight, "--negative-source-weight")
    negative_family_weights = _parse_weight_items(args.negative_family_weight, "--negative-family-weight")
    positive_source_weights = _parse_weight_items(args.positive_source_weight, "--positive-source-weight")
    positive_family_weights = _parse_weight_items(args.positive_family_weight, "--positive-family-weight")
    train(
        args.dataset,
        args.task,
        args.language,
        args.output,
        max_per_language_split_class=args.max_per_language_split_class,
        single_candidate=args.single_candidate,
        balance_split_classes=args.balance_split_classes,
        balance_eval_classes=args.balance_eval_classes,
        positive_weight_mode=args.positive_weight_mode,
        positive_weight_multiplier=args.positive_weight_multiplier,
        hard_negative_weight=max(1.0, args.hard_negative_weight),
        hard_negative_fraction=min(1.0, max(0.0, args.hard_negative_fraction)),
        feature_mode=args.feature_mode,
        positive_source_weights=positive_source_weights,
        positive_family_weights=positive_family_weights,
        negative_source_weights=negative_source_weights,
        negative_family_weights=negative_family_weights,
        text_transform=args.text_transform,
        calibration_mode=args.calibration_mode,
        threshold_target_fpr=(
            None
            if args.threshold_target_fpr is None
            else max(0.0, min(1.0, args.threshold_target_fpr))
        ),
        threshold_target_precision=(
            None
            if args.threshold_target_precision is None
            else max(0.0, min(1.0, args.threshold_target_precision))
        ),
        threshold_target_fnr=(
            None
            if args.threshold_target_fnr is None
            else max(0.0, min(1.0, args.threshold_target_fnr))
        ),
        threshold_plateau_position=max(0.0, min(1.0, args.threshold_plateau_position)),
        excluded_structured_features=tuple(args.exclude_structured_feature),
    )


if __name__ == "__main__":
    main()
