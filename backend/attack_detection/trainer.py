"""Train, calibrate, evaluate, and version the two code-risk models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from .dataset import CodeSample, is_training_eligible, load_dataset
from .features.text_features import TRANSFORM_NAME, enrich_model_text
from .model_registry import create_version_dir, make_version_id, register_version
from .training.language_coverage import eligible_task_languages


MODEL_TASKS = {
    "malicious_intent": {
        "positive": "malicious", "negative": "benign", "prefix": "malicious",
        "description": "benign vs actively malicious package intent", "target_fpr": 0.05,
    },
    "vulnerability_risk": {
        "positive": "vulnerable", "negative": "benign", "prefix": "vulnerability",
        "description": "safe vs vulnerable code", "target_fpr": 0.05,
    },
}

QUALITY_GATE = {
    "min_precision": 0.90,
    "max_false_positive_rate": 0.10,
    "max_false_negative_rate": 0.10,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w+\b|[$_./:+-]+", ngram_range=(1, 2),
            min_df=1, max_df=0.995, max_features=40_000, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60_000, sublinear_tf=True,
        )),
    ])


def _candidate(name: str):
    if name == "linear_svm":
        from sklearn.svm import LinearSVC
        return LinearSVC(class_weight="balanced", C=1.0)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2_000, class_weight="balanced", C=2.0, solver="liblinear")


def _calibrate(model, x_validation, y_validation):
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(x_validation, y_validation)
    return calibrated


def _probabilities(model, features, positive: str) -> list[float]:
    probabilities = model.predict_proba(features)
    index = list(model.classes_).index(positive)
    return [float(row[index]) for row in probabilities]


def _threshold(
    y_true: list[str], scores: list[float], positive: str, target_fpr: float,
    target_precision: float = QUALITY_GATE["min_precision"],
    target_fnr: float = QUALITY_GATE["max_false_negative_rate"],
    plateau_position: float = 0.5,
) -> dict[str, float | bool]:
    from sklearn.metrics import f1_score

    candidates = {value / 200 for value in range(0, 201)}
    candidates.update(float(score) for score in scores)
    best = None
    fallback = None
    negatives = sum(value != positive for value in y_true)
    positives = sum(value == positive for value in y_true)
    for threshold in sorted(candidates):
        predicted = [positive if score >= threshold else "benign" for score in scores]
        fp = sum(actual != positive and pred == positive for actual, pred in zip(y_true, predicted))
        tp = sum(actual == positive and pred == positive for actual, pred in zip(y_true, predicted))
        fn = positives - tp
        fpr = fp / negatives if negatives else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        fnr = fn / positives if positives else 0.0
        recall = tp / positives if positives else 0.0
        f1 = f1_score(y_true, predicted, pos_label=positive, zero_division=0)
        deficit = (
            max(0.0, target_precision - precision)
            + max(0.0, fpr - target_fpr)
            + max(0.0, fnr - target_fnr)
        )
        fallback_rank = (-deficit, f1, precision, -fpr, -fnr)
        fallback_metrics = (fpr, precision, fnr, recall, f1)
        if fallback is None or fallback_rank > fallback[0]:
            fallback = (fallback_rank, [threshold], fallback_metrics)
        elif fallback_rank == fallback[0]:
            fallback[1].append(threshold)
        if precision >= target_precision and fpr <= target_fpr and fnr <= target_fnr:
            rank = (f1, precision, recall, -fpr)
            best_metrics = (fpr, precision, fnr, recall, f1)
            if best is None or rank > best[0]:
                best = (rank, [threshold], best_metrics)
            elif rank == best[0]:
                best[1].append(threshold)
    chosen_row = best or fallback
    if chosen_row is None:
        chosen, fpr, precision, fnr, recall, f1 = 0.95, 0.0, 0.0, 1.0, 0.0, 0.0
    else:
        _, equivalent_thresholds, metrics = chosen_row
        # A confusion matrix is normally stable over an interval.  Choosing
        # either edge spends all of the safety margin on FPR or FNR.  The
        # interval midpoint preserves the exact validation result while
        # leaving margin for benign and malicious score drift.
        plateau_position = min(1.0, max(0.0, float(plateau_position)))
        plateau_low = min(equivalent_thresholds)
        plateau_high = max(equivalent_thresholds)
        chosen = plateau_low + (plateau_high - plateau_low) * plateau_position
        fpr, precision, fnr, recall, f1 = metrics
    return {
        # Keep the original float; coarse rounding can cross a sample score
        # and make the serialized report disagree with fresh evaluation.
        "decision": float(chosen),
        "uncertain_low": round(max(0.05, chosen - 0.10), 8),
        "uncertain_high": round(min(0.95, chosen + 0.05), 8),
        "validation_fpr": round(fpr, 4),
        "validation_precision": round(precision, 4),
        "validation_fnr": round(fnr, 4),
        "validation_recall": round(recall, 4),
        "validation_f1": round(f1, 4),
        "target_fpr": target_fpr,
        "target_precision": target_precision,
        "target_fnr": target_fnr,
        "plateau_position": plateau_position,
        "quality_gate_passed": best is not None,
    }


def _evaluate(y_true: list[str], scores: list[float], positive: str, negative: str, threshold: float) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

    predicted = [positive if score >= threshold else negative for score in scores]
    matrix = confusion_matrix(y_true, predicted, labels=[negative, positive])
    tn, fp, fn, tp = matrix.ravel()
    binary = [1 if value == positive else 0 for value in y_true]
    precision = float(precision_score(y_true, predicted, pos_label=positive, zero_division=0))
    false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    false_negative_rate = float(fn / (fn + tp)) if fn + tp else 0.0
    return {
        "samples": len(y_true), "accuracy": round(float(accuracy_score(y_true, predicted)), 4),
        "precision": round(precision, 4),
        "recall": round(float(recall_score(y_true, predicted, pos_label=positive, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, pos_label=positive, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(binary, scores)), 4) if len(set(binary)) > 1 else None,
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "quality_gate_passed": (
            precision >= QUALITY_GATE["min_precision"]
            and false_positive_rate <= QUALITY_GATE["max_false_positive_rate"]
            and false_negative_rate <= QUALITY_GATE["max_false_negative_rate"]
        ),
        "confusion_matrix": matrix.tolist(),
    }


def meets_quality_gate(metrics: dict[str, object] | None) -> bool:
    """Return true only when all deployment metrics meet the release gate."""

    if not isinstance(metrics, dict):
        return False
    if metrics.get("quality_gate_passed") is False:
        return False
    try:
        return (
            float(metrics["precision"]) >= QUALITY_GATE["min_precision"]
            and float(metrics["false_positive_rate"]) <= QUALITY_GATE["max_false_positive_rate"]
            and float(metrics["false_negative_rate"]) <= QUALITY_GATE["max_false_negative_rate"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _segments(samples: list[CodeSample], scores: list[float], positive: str, negative: str, threshold: float, field: str) -> dict[str, object]:
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        value = str(getattr(sample, field) or "unknown")
        grouped.setdefault(value, []).append(index)
    output = {}
    for value, indices in sorted(grouped.items()):
        labels = [samples[index].label for index in indices]
        if len(indices) < 5 or len(set(labels)) < 2:
            predicted = [scores[index] >= threshold for index in indices]
            positive_count = labels.count(positive)
            negative_count = len(labels) - positive_count
            item: dict[str, object] = {
                "samples": len(indices), "positive_samples": positive_count,
                "negative_samples": negative_count, "insufficient_for_full_metrics": True,
            }
            if positive_count:
                item["recall"] = round(sum(flag for flag, label in zip(predicted, labels) if label == positive) / positive_count, 4)
            if negative_count:
                item["false_positive_rate"] = round(sum(flag for flag, label in zip(predicted, labels) if label != positive) / negative_count, 4)
            output[value] = item
            continue
        output[value] = _evaluate(labels, [scores[index] for index in indices], positive, negative, threshold)
    return output


def _top_features(vectorizer, calibrated, limit: int = 20) -> dict[str, list[dict[str, object]]]:
    try:
        estimator = calibrated.calibrated_classifiers_[0].estimator
        names = vectorizer.get_feature_names_out()
        coefficients = estimator.coef_[0]
        classes = list(estimator.classes_)
        sign = 1 if classes[-1] in {"malicious", "vulnerable"} else -1
        ranked = sorted(range(len(coefficients)), key=lambda index: sign * float(coefficients[index]), reverse=True)
        positive = [index for index in ranked if str(names[index]).startswith("word__")][:limit]
        negative = [index for index in reversed(ranked) if str(names[index]).startswith("word__")][:limit]
        render = lambda indices: [{"feature": str(names[index]).removeprefix("word__"), "weight": round(sign * float(coefficients[index]), 5)} for index in indices]
        return {"positive": render(positive), "negative": render(negative)}
    except (AttributeError, IndexError, TypeError):
        return {"positive": [], "negative": []}


def _train_task(samples: list[CodeSample], task_name: str, version_dir: Path) -> dict[str, object]:
    from joblib import dump

    config = MODEL_TASKS[task_name]
    positive, negative = str(config["positive"]), str(config["negative"])
    selected = [sample for sample in samples if sample.label in {negative, positive}]
    raw_partitions = {
        split: [sample for sample in selected if sample.split == split]
        for split in ("train", "validation", "test")
    }
    supported_languages, language_coverage = eligible_task_languages(
        raw_partitions, positive, negative,
    )
    if not supported_languages:
        return {
            "ready": False,
            "reason": "no language has both task classes in every split",
            "description": config["description"],
            "language_coverage": language_coverage,
        }
    partitions = {
        split: [sample for sample in partition if sample.language in supported_languages]
        for split, partition in raw_partitions.items()
    }
    if any(len({sample.label for sample in part}) < 2 for part in partitions.values()):
        return {
            "ready": False,
            "reason": "a deployment-language split is missing one class",
            "description": config["description"],
        }

    vectorizer = _vectorizer()
    x_train = vectorizer.fit_transform([
        enrich_model_text(sample.code, sample.language) for sample in partitions["train"]
    ])
    x_validation = vectorizer.transform([
        enrich_model_text(sample.code, sample.language) for sample in partitions["validation"]
    ])
    x_test = vectorizer.transform([
        enrich_model_text(sample.code, sample.language) for sample in partitions["test"]
    ])
    x_raw_test = vectorizer.transform([
        enrich_model_text(sample.code, sample.language) for sample in raw_partitions["test"]
    ])
    y_train = [sample.label for sample in partitions["train"]]
    y_validation = [sample.label for sample in partitions["validation"]]
    y_test = [sample.label for sample in partitions["test"]]
    y_raw_test = [sample.label for sample in raw_partitions["test"]]

    candidates = []
    trained = {}
    for name in ("logistic_regression", "linear_svm"):
        base = _candidate(name)
        base.fit(x_train, y_train)
        calibrated = _calibrate(base, x_validation, y_validation)
        scores = _probabilities(calibrated, x_validation, positive)
        threshold_info = _threshold(
            y_validation, scores, positive, float(config["target_fpr"]),
        )
        report = _evaluate(
            y_validation, scores, positive, negative, float(threshold_info["decision"]),
        )
        report.update({"model": name, "thresholds": threshold_info})
        candidates.append(report)
        trained[name] = calibrated
    chosen_report = max(
        candidates,
        key=lambda item: (
            bool(item.get("quality_gate_passed")), float(item.get("pr_auc") or 0),
            float(item["f1"]), -float(item["false_positive_rate"]),
        ),
    )
    chosen_name = str(chosen_report["model"])
    model = trained[chosen_name]
    threshold_info = dict(chosen_report["thresholds"])
    test_scores = _probabilities(model, x_test, positive)
    raw_test_scores = _probabilities(model, x_raw_test, positive)
    report = _evaluate(y_raw_test, raw_test_scores, positive, negative, float(threshold_info["decision"]))

    prefix = str(config["prefix"])
    dump(vectorizer, version_dir / f"{prefix}_vectorizer.joblib")
    dump(model, version_dir / f"{prefix}_classifier.joblib")
    deployment = _evaluate(
        y_test, test_scores, positive, negative, float(threshold_info["decision"]),
    )
    report.update({
        "ready": True, "description": config["description"], "model": chosen_name,
        "engine": f"{chosen_name}+word_char_tfidf+sigmoid_calibration", "calibrated": True,
        "calibration_split": "validation", "split_strategy": "predefined_grouped_temporal_source_holdout",
        "labels": [negative, positive], "train_samples": len(partitions["train"]),
        "supported_languages": supported_languages,
        "language_coverage": language_coverage,
        "input_transform": TRANSFORM_NAME,
        "deployment": deployment,
        "quality_gate_passed": meets_quality_gate(deployment),
        "unsupported_test_samples": len(raw_partitions["test"]) - len(partitions["test"]),
        "raw_full_test_includes_unsupported_languages": True,
        "validation_samples": len(partitions["validation"]), "test_samples": len(raw_partitions["test"]),
        "deployment_test_samples": len(partitions["test"]),
        "thresholds": threshold_info, "candidate_validation": candidates,
        "by_language": _segments(raw_partitions["test"], raw_test_scores, positive, negative, float(threshold_info["decision"]), "language"),
        "by_source": _segments(raw_partitions["test"], raw_test_scores, positive, negative, float(threshold_info["decision"]), "source"),
        "by_category": _segments(raw_partitions["test"], raw_test_scores, positive, negative, float(threshold_info["decision"]), "category"),
        "by_cwe": _segments(raw_partitions["test"], raw_test_scores, positive, negative, float(threshold_info["decision"]), "cwe"),
        "evasion": _segment_named(raw_partitions["test"], raw_test_scores, positive, negative, float(threshold_info["decision"]), "evasion_suite"),
        "top_features": _top_features(vectorizer, model),
    })
    return report


def _segment_named(samples: list[CodeSample], scores: list[float], positive: str, negative: str, threshold: float, source: str) -> dict[str, object]:
    indices = [index for index, sample in enumerate(samples) if sample.source == source]
    if not indices:
        return {"samples": 0, "available": False}
    labels = [samples[index].label for index in indices]
    if len(set(labels)) < 2:
        predicted_positive = sum(scores[index] >= threshold for index in indices)
        positive_count = labels.count(positive)
        return {
            "samples": len(indices), "positive_samples": positive_count, "detected_positive": predicted_positive,
            "recall": round(predicted_positive / positive_count, 4) if positive_count else None, "available": True,
        }
    return {"available": True, **_evaluate(labels, [scores[index] for index in indices], positive, negative, threshold)}


def train_model(
    dataset_path: str | Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, object]:
    dataset = Path(dataset_path).resolve()
    all_samples = load_dataset(dataset)
    if progress_callback:
        progress_callback(0.12, "训练数据读取完成")
    samples = [sample for sample in all_samples if is_training_eligible(sample)]
    dataset_hash = _sha256(dataset)
    version = make_version_id(dataset_hash)
    version_dir = create_version_dir(version)
    task_metrics: dict[str, dict[str, object]] = {}
    task_count = len(MODEL_TASKS)
    for index, name in enumerate(MODEL_TASKS, start=1):
        task_metrics[name] = _train_task(samples, name, version_dir)
        if progress_callback:
            progress_callback(0.12 + 0.76 * index / task_count, f"已完成 {index}/{task_count} 个模型任务")
    metrics = {
        "schema_version": 3, "model_version": version, "dataset": str(dataset),
        "dataset_sha256": dataset_hash, "samples": len(samples),
        "samples_total": len(all_samples),
        "samples_training_eligible": len(samples),
        "excluded_review_samples": len(all_samples) - len(samples),
        "label_counts": dict(Counter(sample.label for sample in samples)),
        "language_counts": dict(Counter(sample.language for sample in samples)),
        "source_counts": dict(Counter(sample.source for sample in samples)),
        "split_counts": dict(Counter(sample.split for sample in samples)),
        "tasks": task_metrics,
    }
    gate_results = {
        task_name: meets_quality_gate(task.get("deployment"))
        for task_name, task in metrics["tasks"].items()
    }
    metrics["quality_gate"] = {
        "requirements": QUALITY_GATE,
        "tasks": gate_results,
        "passed": bool(gate_results) and all(gate_results.values()),
        "scope": "supported training languages on untouched test split",
    }
    (version_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback(0.96, "正在登记模型版本")
    register_version(version, metrics, dataset_hash, activate=bool(metrics["quality_gate"]["passed"]))
    metrics["published"] = bool(metrics["quality_gate"]["passed"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated dual source-code risk models")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    metrics = train_model(args.dataset)
    print(json.dumps({"model_version": metrics["model_version"], "samples": metrics["samples"], "tasks": metrics["tasks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
