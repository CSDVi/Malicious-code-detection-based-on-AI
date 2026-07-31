"""Runtime loader for versioned calibrated source-code risk models."""

from __future__ import annotations

import json
import re
from typing import Any

from .model_registry import active_model_dir
from .features.text_features import TRANSFORM_NAME, enrich_model_text
from .task_policy import task_enabled


class CodeRiskClassifier:
    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, Any] = {}
        self.version = "unavailable"
        self.reload()

    def reload(self) -> None:
        self.models = {}
        self.metrics = {}
        model_dir, self.version = active_model_dir()
        try:
            import joblib

            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for task, prefix in (("malicious_intent", "malicious"), ("vulnerability_risk", "vulnerability")):
                if not task_enabled(task):
                    continue
                vectorizer_path = model_dir / f"{prefix}_vectorizer.joblib"
                classifier_path = model_dir / f"{prefix}_classifier.joblib"
                task_metrics = self.metrics.get("tasks", {}).get(task, {})
                if vectorizer_path.exists() and classifier_path.exists() and task_metrics.get("ready"):
                    model = joblib.load(classifier_path)
                    self.models[task] = {
                        "vectorizer": joblib.load(vectorizer_path), "model": model,
                        "engine": str(task_metrics.get("engine") or model.__class__.__name__),
                        "metrics": task_metrics,
                    }
        except (OSError, ValueError, ImportError, json.JSONDecodeError):
            self.models = {}

    def predict(self, content: str, language: str = "unknown") -> dict[str, object]:
        malicious = self._predict_task("malicious_intent", content, language, "malicious")
        vulnerability = _disabled_task("vulnerability_risk", "vulnerable")
        return {
            "label": malicious["label"], "probability": malicious["probability"],
            "engine": malicious["engine"], "malicious_intent": malicious,
            "vulnerability_risk": vulnerability, "model_version": self.version,
            "training_samples": self.metrics.get("samples"),
            "dataset_sha256": self.metrics.get("dataset_sha256"),
        }

    def _predict_task(self, task: str, content: str, language: str, positive: str) -> dict[str, object]:
        loaded = self.models.get(task)
        if not loaded:
            return _unavailable(task, positive, "模型文件尚未加载")
        metrics = loaded["metrics"]
        supported = list(metrics.get("supported_languages") or [])
        if supported and language not in supported:
            return _unavailable(task, positive, f"训练集中没有 {language} 语言的正样本", supported)

        model_input = (
            enrich_model_text(content, language)
            if metrics.get("input_transform") == TRANSFORM_NAME
            else content
        )
        features = loaded["vectorizer"].transform([model_input])
        probability = _positive_probability(loaded["model"], features, positive)
        thresholds = dict(metrics.get("thresholds") or {})
        decision = float(thresholds.get("decision", 0.5))
        low = float(thresholds.get("uncertain_low", decision))
        high = float(thresholds.get("uncertain_high", decision))
        if low <= probability < high:
            status = "uncertain"
        else:
            status = "positive" if probability >= decision else "negative"
        label = positive if probability >= decision else "benign"
        return {
            "label": label, "probability": round(probability, 4), "model_probability": round(probability, 4),
            "positive_label": positive, "available": True, "status": status,
            "engine": loaded["engine"], "threshold": decision, "thresholds": thresholds,
            "supported_languages": supported,
            "evidence_features": _feature_evidence(content, metrics.get("top_features", {}).get("positive", [])),
        }


def _unavailable(task: str, positive: str, reason: str, supported: list[str] | None = None) -> dict[str, object]:
    return {
        "label": "unavailable", "probability": None, "model_probability": None,
        "positive_label": positive, "available": False, "status": "unavailable",
        "engine": task, "reason": reason, "supported_languages": supported or [],
        "evidence_features": [],
    }


def _disabled_task(task: str, positive: str) -> dict[str, object]:
    return {
        "label": "disabled", "probability": None, "model_probability": None,
        "positive_label": positive, "available": False, "status": "disabled",
        "engine": task, "reason": "漏洞风险任务已从当前产品流程下线",
        "supported_languages": [], "evidence_features": [],
    }


def _positive_probability(model: Any, features: Any, positive: str) -> float:
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(positive)])


def _feature_evidence(content: str, features: list[dict[str, object]]) -> list[dict[str, object]]:
    lowered = content.lower()
    evidence = []
    for item in features:
        feature = str(item.get("feature") or "").strip().lower()
        if len(feature) >= 2 and re.search(r"(?<![A-Za-z0-9_$])" + re.escape(feature) + r"(?![A-Za-z0-9_$])", lowered):
            evidence.append({"feature": feature, "weight": item.get("weight")})
        if len(evidence) >= 8:
            break
    return evidence


classifier = CodeRiskClassifier()
