"""Train and register calibrated dual-task XGBoost models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attack_detection.dataset import CodeSample, is_task_training_eligible, is_training_eligible, load_dataset
from attack_detection.features.static_features import FEATURE_NAMES, feature_vector
from attack_detection.trainer import QUALITY_GATE, _evaluate, _segments, _threshold, meets_quality_gate
from attack_detection.training.language_coverage import eligible_task_languages

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_ROOT = MODEL_DIR / "xgb_registry"
REGISTRY_PATH = MODEL_DIR / "xgb_registry.json"
TASKS = {
    "malicious_intent": {
        "positive": "malicious",
        "negative": "benign",
        "artifact": "xgb_malicious_classifier.joblib",
        "target_fpr": 0.05,
    },
    "vulnerability_risk": {
        "positive": "vulnerable",
        "negative": "benign",
        "artifact": "xgb_vulnerability_classifier.joblib",
        "target_fpr": 0.05,
    },
}


def train_xgboost(dataset_path: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    samples = load_dataset(dataset)
    eligible = [sample for sample in samples if is_training_eligible(sample)]
    # Feature extraction performs regex/rule/AST work.  Cache once per sample
    # and reuse it for both tasks; the old implementation recomputed the same
    # vectors twice and made iteration on the model unnecessarily slow.
    feature_cache = {
        (sample.sample_hash, sample.language): feature_vector(
            sample.code, sample.language, include_rules=False,
        )
        for sample in eligible
    }
    dataset_hash = _sha256(dataset)
    version = "xgb-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + dataset_hash[:10]
    version_dir = REGISTRY_ROOT / version
    version_dir.mkdir(parents=True, exist_ok=False)

    task_metrics = {
        task_name: _train_task(eligible, task_name, config, version_dir, feature_cache)
        for task_name, config in TASKS.items()
    }
    metrics = {
        "schema_version": 1,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "dataset_sha256": dataset_hash,
        "feature_schema": list(FEATURE_NAMES),
        "feature_mode": "fast_static_without_rule_engine",
        "samples_total": len(samples),
        "samples_training_eligible": len(eligible),
        "excluded_review_samples": len(samples) - len(eligible),
        "label_counts": dict(Counter(sample.label for sample in eligible)),
        "review_status_counts": dict(Counter(sample.review_status for sample in samples)),
        "tasks": task_metrics,
    }
    metrics_path = version_dir / "xgb_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
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
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _register(version, version_dir, metrics, activate=bool(metrics["quality_gate"]["passed"]))
    metrics["published"] = bool(metrics["quality_gate"]["passed"])
    return metrics


def _train_task(
    samples: list[CodeSample],
    task_name: str,
    config: dict[str, object],
    version_dir: Path,
    feature_cache: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    from joblib import dump
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    positive = str(config["positive"])
    negative = str(config["negative"])
    selected = [
        sample for sample in samples
        if sample.label in {negative, positive}
        and is_task_training_eligible(sample, task_name)
    ]
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
            "language_coverage": language_coverage,
        }
    partitions = {
        split: [sample for sample in partition if sample.language in supported_languages]
        for split, partition in raw_partitions.items()
    }
    for split, partition in partitions.items():
        if len({sample.label for sample in partition}) < 2:
            return {"ready": False, "reason": f"{split} deployment-language split is missing one class"}

    vectors = {
        split: [feature_cache[(sample.sample_hash, sample.language)] for sample in partition]
        for split, partition in partitions.items()
    }
    raw_test_vectors = [
        feature_cache[(sample.sample_hash, sample.language)]
        for sample in raw_partitions["test"]
        if (sample.sample_hash, sample.language) in feature_cache
    ]
    labels = {
        split: [1 if sample.label == positive else 0 for sample in partition]
        for split, partition in partitions.items()
    }
    positives = sum(labels["train"])
    negatives = len(labels["train"]) - positives
    # The previous configuration heavily over-weighted positives and used a
    # fixed 320 trees.  That improved recall at the cost of precision/FPR.
    # A milder weight, histogram training and validation early stopping give
    # the tree ensemble room to optimize the operating point selected later.
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1600,
        max_depth=5,
        learning_rate=0.025,
        min_child_weight=3,
        gamma=0.05,
        subsample=0.88,
        colsample_bytree=0.9,
        reg_alpha=0.25,
        reg_lambda=4.0,
        max_delta_step=1.0,
        scale_pos_weight=max(1.0, (negatives / max(1, positives)) ** 0.5),
        random_state=42,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        tree_method="hist",
        early_stopping_rounds=80,
    )
    model.fit(
        vectors["train"],
        labels["train"],
        eval_set=[(vectors["validation"], labels["validation"])],
        verbose=False,
    )
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(vectors["validation"], labels["validation"])

    validation_scores = [float(row[1]) for row in calibrated.predict_proba(vectors["validation"])]
    validation_labels = [sample.label for sample in partitions["validation"]]
    threshold_info = _threshold(
        validation_labels, validation_scores, positive, float(config["target_fpr"]),
    )
    threshold = float(threshold_info["decision"])
    test_scores = [float(row[1]) for row in calibrated.predict_proba(vectors["test"])]
    test_labels = [sample.label for sample in partitions["test"]]
    raw_test_scores = [float(row[1]) for row in calibrated.predict_proba(raw_test_vectors)]
    raw_test_labels = [sample.label for sample in raw_partitions["test"]]
    report = _evaluate(raw_test_labels, raw_test_scores, positive, negative, threshold)
    validation_report = _evaluate(validation_labels, validation_scores, positive, negative, threshold)

    artifact = str(config["artifact"])
    dump(calibrated, version_dir / artifact)
    deployment = _evaluate(
        test_labels, test_scores, positive, negative, threshold,
    )
    importance = sorted(
        (
            {"feature": name, "importance": round(float(score), 6)}
            for name, score in zip(FEATURE_NAMES, model.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    report.update({
        "ready": True,
        "task": task_name,
        "engine": "xgboost+sigmoid_calibration",
        "calibrated": True,
        "calibration_split": "validation",
        "labels": [negative, positive],
        "positive_label": positive,
        "negative_label": negative,
        "artifact": artifact,
        "train_samples": len(partitions["train"]),
        "validation_samples": len(partitions["validation"]),
        "test_samples": len(raw_partitions["test"]),
        "deployment_test_samples": len(partitions["test"]),
        "unsupported_test_samples": len(raw_partitions["test"]) - len(partitions["test"]),
        "train_positive_samples": positives,
        "train_unique_positive_families": len({sample.family for sample in partitions["train"] if sample.label == positive}),
        "supported_languages": supported_languages,
        "language_coverage": language_coverage,
        "deployment": deployment,
        "quality_gate_passed": meets_quality_gate(deployment),
        "thresholds": threshold_info,
        "validation": validation_report,
        "by_language": _segments(raw_partitions["test"], raw_test_scores, positive, negative, threshold, "language"),
        "by_source": _segments(raw_partitions["test"], raw_test_scores, positive, negative, threshold, "source"),
        "feature_importance": importance[:20],
        "limitations": [
            "Synthetic evasion variants inherit the original sample split.",
            "Metrics must be interpreted with the reported unique positive family count.",
            "A language is unavailable unless both task classes meet the minimum in every split.",
        ],
    })
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _register(version: str, version_dir: Path, metrics: dict[str, Any], *, activate: bool = False) -> None:
    artifacts = []
    for path in sorted(version_dir.iterdir()):
        if path.is_file():
            artifacts.append({"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)})
    registry = _read_registry()
    registry["versions"] = [item for item in registry.get("versions", []) if item.get("version") != version]
    registry["versions"].insert(0, {
        "version": version,
        "created_at": metrics["created_at"],
        "dataset_sha256": metrics["dataset_sha256"],
        "samples_training_eligible": metrics["samples_training_eligible"],
        "tasks": metrics["tasks"],
        "artifacts": artifacts,
        "published": activate,
        "quality_gate": metrics.get("quality_gate", {}),
    })
    if activate:
        registry["active_version"] = version
        registry["activated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_json(REGISTRY_PATH, registry)
    if activate:
        for path in sorted(version_dir.iterdir()):
            if path.is_file():
                shutil.copy2(path, MODEL_DIR / path.name)


def _read_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": 1, "active_version": "", "versions": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated dual-task XGBoost code-risk models")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    metrics = train_xgboost(args.dataset)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
