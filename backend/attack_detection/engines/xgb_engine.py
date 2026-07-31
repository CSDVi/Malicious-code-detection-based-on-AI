"""Calibrated XGBoost inference adapter for the public malicious-code task."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from attack_detection.cancellation import raise_if_cancelled
from attack_detection.contracts import EngineResult
from attack_detection.features.behavior_tokens import (
    BEHAVIOR_TOKEN_VERSION,
    BEHAVIOR_TOKEN_VERSION_V2,
    BEHAVIOR_TOKEN_VERSION_V3,
    behavior_token_text,
    behavior_token_text_v2,
    behavior_token_text_v3,
)
from attack_detection.features.static_features import FEATURE_NAMES, extract_static_features
from attack_detection.task_policy import task_enabled

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
TASKS = {
    "malicious_intent": {
        "name": "xgboost_malicious",
        "artifact": "xgb_malicious_classifier.joblib",
        "positive": "malicious",
    },
    "vulnerability_risk": {
        "name": "xgboost_vulnerability",
        "artifact": "xgb_vulnerability_classifier.joblib",
        "positive": "vulnerable",
    },
}


def _batch_progress(
    callback: Callable[[int, int, str], None] | None,
    done: int,
    total: int,
    stage: str,
) -> None:
    if callback is not None:
        callback(done, total, stage)


ATTRIBUTION_MIN_PROBABILITY = 0.15
ATTRIBUTION_MIN_DROP = 0.0005
ATTRIBUTION_TOP_K = 5
ATTRIBUTION_MAX_INDIVIDUAL_LINES = 5
ATTRIBUTION_COARSE_GROUPS = 6
ATTRIBUTION_REFINED_GROUPS = 2
ATTRIBUTION_MAX_CHARACTERS = 250_000
ATTRIBUTION_MAX_LINES = 5_000
PROJECT_FEATURE_WORKERS = max(
    1,
    min(
        8,
        int(os.environ.get(
            "XIEZHI_XGB_FEATURE_WORKERS",
            "2",
        )),
    ),
)


class XGBoostEngine:
    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._models: dict[str, Any] = {}
        self._route_models: dict[tuple[str, str], Any] = {}
        self._metrics: dict[str, Any] = {}
        self._loaded_version = ""

    def scan(
        self,
        content: str,
        language: str,
        *,
        generate_line_attributions: bool = True,
        cancel_event: object | None = None,
        precomputed_batch: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        raise_if_cancelled(cancel_event)
        self._ensure_loaded()
        # XGBoost artifacts trained by the current pipeline use only static
        # features.  Rules remain an independent evidence engine; including
        # them as model inputs would make training slow and blur the evidence
        # boundary.  Older artifacts without this metadata retain the legacy
        # behavior for compatibility.
        include_rules = self._metrics.get("feature_mode") != "fast_static_without_rule_engine"
        structured_features = (
            dict(precomputed_batch["structured_features"])
            if (
                isinstance(precomputed_batch, dict)
                and isinstance(
                    precomputed_batch.get("structured_features"),
                    dict,
                )
            )
            else extract_static_features(
                content,
                language,
                include_rules=False,
            )
        )
        precomputed_probabilities = (
            precomputed_batch.get("probabilities") or {}
            if isinstance(precomputed_batch, dict)
            else {}
        )
        legacy_features: dict[str, float] | None = (
            structured_features if not include_rules else None
        )
        feature_names = list(self._metrics.get("feature_schema") or FEATURE_NAMES)
        output = []
        for task_name, config in TASKS.items():
            raise_if_cancelled(cancel_event)
            if not task_enabled(task_name):
                continue
            task_start = time.perf_counter()
            model = self._models.get(task_name)
            task_metrics = self._metrics.get("tasks", {}).get(task_name, {})
            language_route = (task_metrics.get("language_routes") or {}).get(language)
            if isinstance(language_route, dict):
                model = self._route_models.get((task_name, language))
            if model is None or not task_metrics.get("ready"):
                output.append(EngineResult(
                    name=str(config["name"]),
                    status="unavailable",
                    reason=f"{config['artifact']} is not loaded; no probability was produced",
                    duration_ms=int((time.perf_counter() - task_start) * 1000),
                    metadata={"task": task_name},
                ).to_dict())
                continue
            supported = list(task_metrics.get("supported_languages") or [])
            if supported and language not in supported:
                output.append(EngineResult(
                    name=str(config["name"]),
                    status="unavailable",
                    reason=f"the training split has no positive {language} samples for {task_name}",
                    model_version=self._loaded_version,
                    duration_ms=int((time.perf_counter() - task_start) * 1000),
                    metadata={"task": task_name, "supported_languages": supported},
                ).to_dict())
                continue
            try:
                if (
                    isinstance(language_route, dict)
                    and language_route.get("feature_mode") in {"hybrid_hash", "structured_static"}
                ):
                    model_features = structured_features
                    if task_name in precomputed_probabilities:
                        probability = float(
                            precomputed_probabilities[task_name]
                        )
                    else:
                        probability = _route_probability(
                            model,
                            content,
                            model_features,
                            str(language_route.get("feature_mode")),
                        )
                else:
                    if legacy_features is None:
                        legacy_features = extract_static_features(
                            content,
                            language,
                            include_rules=True,
                        )
                    model_features = legacy_features
                    vector = [[
                        float(model_features.get(name, 0.0))
                        for name in feature_names
                    ]]
                    probability = float(model.predict_proba(vector)[0][1])
                line_attributions = []
                if (
                    generate_line_attributions
                    and
                    task_name == "malicious_intent"
                    and probability >= ATTRIBUTION_MIN_PROBABILITY
                ):
                    line_attributions = _line_attributions(
                        content,
                        probability,
                        lambda candidate: _candidate_probability(
                            model=model,
                            content=candidate,
                            language=language,
                            include_rules=include_rules,
                            feature_names=feature_names,
                            language_route=language_route,
                        ),
                        predict_probabilities=lambda candidates: _candidate_probabilities(
                            model=model,
                            contents=candidates,
                            language=language,
                            include_rules=include_rules,
                            feature_names=feature_names,
                            language_route=language_route,
                        ),
                        cancel_event=cancel_event,
                    )
                route_thresholds = language_route.get("thresholds") if isinstance(language_route, dict) else None
                effective_thresholds = route_thresholds or task_metrics.get("thresholds", {})
                threshold = float(
                    effective_thresholds.get("decision", 0.5)
                )
                positive = str(config["positive"])
                raw_decision = positive if probability >= threshold else "benign"
                evaluation_scope = (
                    language_route.get("evaluation_scope")
                    if isinstance(language_route, dict)
                    else "file"
                )
                route_quality_gate_passed = (
                    language_route.get("quality_gate_passed")
                    if isinstance(language_route, dict)
                    else task_metrics.get("quality_gate_passed")
                )
                source_heldout_verified = (
                    language_route.get("source_heldout_verified")
                    if isinstance(language_route, dict)
                    else None
                )
                project_scope = evaluation_scope in {"project", "package", "project_or_package"}
                advisory_only = (
                    route_quality_gate_passed is not True
                    or project_scope
                    or source_heldout_verified is False
                )
                if route_quality_gate_passed is not True:
                    advisory_reason = "language route has no independent quality-gate pass"
                elif source_heldout_verified is False:
                    advisory_reason = "source-held-out evaluation did not pass"
                elif project_scope:
                    advisory_reason = "route is validated only after project/package score aggregation"
                else:
                    advisory_reason = None
                output.append(EngineResult(
                    name=str(config["name"]),
                    status="completed",
                    decision="review" if advisory_only else raw_decision,
                    probability=round(probability, 4),
                    threshold=threshold,
                    model_version=self._loaded_version,
                    duration_ms=int((time.perf_counter() - task_start) * 1000),
                    metadata={
                        "task": task_name,
                        "positive_label": positive,
                        "supported_languages": supported,
                        "uncertain_low": effective_thresholds.get("uncertain_low"),
                        "uncertain_high": effective_thresholds.get("uncertain_high"),
                        "evaluation_scope": evaluation_scope,
                        "route_quality_gate_passed": route_quality_gate_passed,
                        "source_heldout_verified": source_heldout_verified,
                        "route_release_scope": (
                            language_route.get("release_scope")
                            if isinstance(language_route, dict)
                            else task_metrics.get("release_scope")
                        ),
                        "advisory_only": advisory_only,
                        "advisory_reason": advisory_reason,
                        "raw_model_decision": raw_decision,
                        "feature_evidence": _feature_evidence(model_features, task_metrics),
                        "line_attributions": line_attributions,
                        "explanation_method": (
                            "line_occlusion"
                            if line_attributions
                            else "not_generated"
                        ),
                    },
                ).to_dict())
            except Exception as exc:  # pragma: no cover - artifact/runtime dependent
                output.append(EngineResult(
                    name=str(config["name"]),
                    status="failed",
                    reason="XGBoost inference failed",
                    error=str(exc),
                    model_version=self._loaded_version or None,
                    duration_ms=int((time.perf_counter() - task_start) * 1000),
                    metadata={"task": task_name},
                ).to_dict())
        if output and all(item.get("duration_ms") == 0 for item in output):
            output[0]["duration_ms"] = int((time.perf_counter() - start) * 1000)
        return output

    def prepare_batch(
        self,
        requests: list[dict[str, str]],
        *,
        cancel_event: object | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Precompute project XGBoost route matrices in true multi-file batches."""

        if not requests:
            return []
        raise_if_cancelled(cancel_event)
        _batch_progress(
            progress_callback,
            0,
            1,
            "加载 XGBoost 模型",
        )
        self._ensure_loaded()
        _batch_progress(
            progress_callback,
            1,
            1,
            "加载 XGBoost 模型",
        )
        structured_rows = self._extract_project_features(
            requests,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        prepared = [
            {
                "structured_features": row,
                "probabilities": {},
            }
            for row in structured_rows
        ]
        grouped: dict[
            tuple[str, str, str, int],
            dict[str, Any],
        ] = {}
        for task_name in TASKS:
            if not task_enabled(task_name):
                continue
            task_metrics = self._metrics.get("tasks", {}).get(task_name, {})
            if not task_metrics.get("ready"):
                continue
            supported = set(task_metrics.get("supported_languages") or [])
            for index, request in enumerate(requests):
                language = str(request.get("language") or "")
                if supported and language not in supported:
                    continue
                language_route = (
                    task_metrics.get("language_routes") or {}
                ).get(language)
                if not isinstance(language_route, dict):
                    continue
                feature_mode = str(
                    language_route.get("feature_mode") or ""
                )
                if feature_mode not in {
                    "hybrid_hash",
                    "structured_static",
                }:
                    continue
                model = self._route_models.get((task_name, language))
                if model is None:
                    continue
                key = (
                    task_name,
                    language,
                    feature_mode,
                    id(model),
                )
                group = grouped.setdefault(
                    key,
                    {
                        "task_name": task_name,
                        "model": model,
                        "feature_mode": feature_mode,
                        "indices": [],
                    },
                )
                group["indices"].append(index)

        prediction_total = sum(
            len(group["indices"])
            for group in grouped.values()
        )
        predicted = 0
        _batch_progress(
            progress_callback,
            0,
            prediction_total,
            "XGBoost 批量推理",
        )
        for group in grouped.values():
            raise_if_cancelled(cancel_event)
            indices = list(group["indices"])
            try:
                route_batch_size = max(
                    1,
                    int(os.environ.get(
                        "XIEZHI_XGB_PROJECT_BATCH_FILES",
                        "64",
                    )),
                )
                probabilities = []
                for offset in range(
                    0,
                    len(indices),
                    route_batch_size,
                ):
                    raise_if_cancelled(cancel_event)
                    chunk_indices = indices[
                        offset:offset + route_batch_size
                    ]
                    probabilities.extend(_route_probabilities_batch(
                        group["model"],
                        [
                            str(
                                requests[index].get("content") or ""
                            )
                            for index in chunk_indices
                        ],
                        [
                            structured_rows[index]
                            for index in chunk_indices
                        ],
                        str(group["feature_mode"]),
                    ))
                    predicted += len(chunk_indices)
                    _batch_progress(
                        progress_callback,
                        predicted,
                        prediction_total,
                        "XGBoost 批量推理",
                    )
            except Exception:
                # Preserve the established per-file path as a compatibility
                # fallback for an older or unusual route artifact.
                continue
            if len(probabilities) != len(indices):
                continue
            for index, probability in zip(indices, probabilities):
                prepared[index]["probabilities"][
                    str(group["task_name"])
                ] = float(probability)
        _batch_progress(
            progress_callback,
            prediction_total,
            prediction_total,
            "XGBoost 批量推理",
        )
        return prepared

    @staticmethod
    def _extract_project_features(
        requests: list[dict[str, str]],
        *,
        cancel_event: object | None,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> list[dict[str, float]]:
        """Extract the unchanged model feature schema concurrently.

        Each request is independent.  Threads let the regex and hashing work
        use multiple CPU cores while preserving the exact feature dictionary
        and input order expected by the trained artifacts.
        """

        total = len(requests)
        _batch_progress(
            progress_callback,
            0,
            total,
            "提取 XGBoost 特征",
        )

        def extract_one(
            request: dict[str, str],
        ) -> dict[str, float]:
            return extract_static_features(
                str(request.get("content") or ""),
                str(request.get("language") or ""),
                include_rules=False,
            )

        if PROJECT_FEATURE_WORKERS <= 1 or total < 4:
            output = []
            for completed, request in enumerate(requests, start=1):
                raise_if_cancelled(cancel_event)
                output.append(extract_one(request))
                _batch_progress(
                    progress_callback,
                    completed,
                    total,
                    "提取 XGBoost 特征",
                )
            return output

        output: list[dict[str, float] | None] = [None] * total
        executor = ThreadPoolExecutor(
            max_workers=min(PROJECT_FEATURE_WORKERS, total),
            thread_name_prefix="xgb-features",
        )
        futures = {
            executor.submit(extract_one, request): index
            for index, request in enumerate(requests)
        }
        cancelled = False
        try:
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                raise_if_cancelled(cancel_event)
                output[futures[future]] = future.result()
                _batch_progress(
                    progress_callback,
                    completed,
                    total,
                    "提取 XGBoost 特征",
                )
        except Exception:
            cancelled = True
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(
                wait=not cancelled,
                cancel_futures=cancelled,
            )
        return [
            row if row is not None else {}
            for row in output
        ]

    def explain_batch(
        self,
        requests: list[dict[str, str]],
        *,
        prepared_batch: list[dict[str, Any]] | None = None,
        cancel_event: object | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Generate candidate explanations with one matrix per language route."""

        if not requests:
            return []
        if (
            prepared_batch is None
            or len(prepared_batch) != len(requests)
        ):
            prepared_batch = self.prepare_batch(
                requests,
                cancel_event=cancel_event,
            )
        outputs = [
            self.scan(
                str(request.get("content") or ""),
                str(request.get("language") or ""),
                generate_line_attributions=False,
                cancel_event=cancel_event,
                precomputed_batch=prepared,
            )
            for request, prepared in zip(requests, prepared_batch)
        ]
        include_rules = (
            self._metrics.get("feature_mode")
            != "fast_static_without_rule_engine"
        )
        task_name = "malicious_intent"
        task_metrics = self._metrics.get("tasks", {}).get(task_name, {})
        feature_names = list(
            self._metrics.get("feature_schema") or FEATURE_NAMES
        )
        grouped: dict[
            tuple[str, str, int],
            dict[str, Any],
        ] = {}
        fallback_indices: set[int] = set()
        for index, (request, prepared) in enumerate(
            zip(requests, prepared_batch)
        ):
            raise_if_cancelled(cancel_event)
            language = str(request.get("language") or "")
            language_route = (
                task_metrics.get("language_routes") or {}
            ).get(language)
            model = (
                self._route_models.get((task_name, language))
                if isinstance(language_route, dict)
                else self._models.get(task_name)
            )
            raw_probability = (
                prepared.get("probabilities") or {}
            ).get(task_name)
            if model is None:
                continue
            if raw_probability is None:
                completed_engine = next(
                    (
                        engine
                        for engine in outputs[index]
                        if (
                            (engine.get("metadata") or {}).get("task")
                            == task_name
                            and engine.get("status") == "completed"
                        )
                    ),
                    None,
                )
                if (
                    completed_engine is not None
                    and float(
                        completed_engine.get("probability") or 0.0
                    ) >= ATTRIBUTION_MIN_PROBABILITY
                ):
                    fallback_indices.add(index)
                continue
            if float(raw_probability) < ATTRIBUTION_MIN_PROBABILITY:
                continue
            context = _line_attribution_context(
                str(request.get("content") or "")
            )
            if context is None:
                continue
            lines, candidates = context
            masked_contents = _masked_line_candidates(
                lines,
                candidates,
            )
            feature_mode = (
                str(language_route.get("feature_mode") or "")
                if isinstance(language_route, dict)
                else ""
            )
            if feature_mode not in {
                "hybrid_hash",
                "structured_static",
            }:
                fallback_indices.add(index)
                continue
            key = (language, feature_mode, id(model))
            group = grouped.setdefault(
                key,
                {
                    "model": model,
                    "language": language,
                    "language_route": language_route,
                    "items": [],
                },
            )
            group["items"].append({
                "index": index,
                "content": str(request.get("content") or ""),
                "baseline": float(raw_probability),
                "masked_contents": masked_contents,
            })

        for group in grouped.values():
            raise_if_cancelled(cancel_event)
            flattened = [
                candidate
                for item in group["items"]
                for candidate in item["masked_contents"]
            ]
            try:
                candidate_batch_size = max(
                    ATTRIBUTION_MAX_INDIVIDUAL_LINES,
                    int(os.environ.get(
                        "XIEZHI_XGB_EXPLANATION_BATCH_CANDIDATES",
                        "128",
                    )),
                )
                values = []
                for offset in range(
                    0,
                    len(flattened),
                    candidate_batch_size,
                ):
                    raise_if_cancelled(cancel_event)
                    values.extend(_candidate_probabilities(
                        model=group["model"],
                        contents=flattened[
                            offset:offset + candidate_batch_size
                        ],
                        language=str(group["language"]),
                        include_rules=include_rules,
                        feature_names=feature_names,
                        language_route=group["language_route"],
                    ))
                raise_if_cancelled(cancel_event)
            except Exception:
                fallback_indices.update(
                    int(item["index"])
                    for item in group["items"]
                )
                continue
            offset = 0
            for item in group["items"]:
                expected = list(item["masked_contents"])
                item_values = values[offset:offset + len(expected)]
                offset += len(expected)
                if len(item_values) != len(expected):
                    fallback_indices.add(int(item["index"]))
                    continue
                attributions = _line_attributions(
                    str(item["content"]),
                    float(item["baseline"]),
                    lambda _candidate: 0.0,
                    predict_probabilities=(
                        lambda candidates, expected=expected, item_values=item_values: (
                            list(item_values)
                            if candidates == expected
                            else []
                        )
                    ),
                    cancel_event=cancel_event,
                )
                _set_line_attributions(
                    outputs[int(item["index"])],
                    attributions,
                )

        for index in sorted(fallback_indices):
            request = requests[index]
            outputs[index] = self.scan(
                str(request.get("content") or ""),
                str(request.get("language") or ""),
                generate_line_attributions=True,
                cancel_event=cancel_event,
                precomputed_batch=prepared_batch[index],
            )
        return outputs

    def reload(self) -> None:
        self._models = {}
        self._route_models = {}
        self._metrics = {}
        self._loaded_version = ""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._models:
            return
        with self._load_lock:
            if self._models:
                return
            self._ensure_loaded_locked()

    def _ensure_loaded_locked(self) -> None:
        metrics_path = MODEL_DIR / "xgb_metrics.json"
        if not metrics_path.exists():
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            version = str(metrics.get("model_version") or "")
            if self._models and version == self._loaded_version:
                return
            import joblib

            # A multilingual release may route many languages to one shared
            # fallback artifact.  Cache by absolute path so that the same
            # joblib bundle is loaded once instead of once per route.
            artifact_cache: dict[str, Any] = {}

            def load_cached(path: Path) -> Any:
                key = str(path.resolve())
                if key not in artifact_cache:
                    artifact_cache[key] = joblib.load(path)
                return artifact_cache[key]

            models = {}
            for task_name, config in TASKS.items():
                if not task_enabled(task_name):
                    continue
                path = MODEL_DIR / str(config["artifact"])
                if path.exists():
                    models[task_name] = load_cached(path)
            route_models = {}
            for task_name, task_metrics in (metrics.get("tasks") or {}).items():
                if not task_enabled(task_name):
                    continue
                for language, route in (task_metrics.get("language_routes") or {}).items():
                    if not isinstance(route, dict):
                        continue
                    artifact = str(route.get("artifact") or "")
                    if not artifact:
                        continue
                    path = MODEL_DIR / artifact
                    if path.exists():
                        route_models[(task_name, str(language))] = load_cached(path)
            self._models = models
            self._route_models = route_models
            self._metrics = metrics
            self._loaded_version = version
        except (OSError, ValueError, ImportError, json.JSONDecodeError):
            self._models = {}
            self._route_models = {}
            self._metrics = {}
            self._loaded_version = ""


def _feature_evidence(features: dict[str, float], task_metrics: dict[str, Any]) -> list[dict[str, float | str]]:
    importance = {
        str(item.get("feature")): float(item.get("importance") or 0.0)
        for item in task_metrics.get("feature_importance", [])
    }
    ranked = sorted(
        (
            {"feature": name, "value": round(float(value), 4), "importance": round(importance.get(name, 0.0), 6)}
            for name, value in features.items()
            if value and importance.get(name, 0.0) > 0
        ),
        key=lambda item: float(item["importance"]),
        reverse=True,
    )
    return ranked[:8]


def _route_probability(
    bundle: Any,
    content: str,
    structured_features: dict[str, float],
    feature_mode: str,
) -> float:
    """Predict a language-specific structured or structured+text bundle."""

    import numpy
    from scipy.sparse import csr_matrix, hstack

    if not isinstance(bundle, dict):
        raise TypeError("hybrid XGBoost route is not a bundle")
    feature_names = list(bundle.get("feature_names") or FEATURE_NAMES)
    structured_vector = [[
        float(structured_features.get(name, 0.0))
        for name in feature_names
    ]]
    structured_matrix = csr_matrix(
        numpy.asarray(structured_vector, dtype="float32")
    )
    if feature_mode == "structured_static":
        return float(bundle["model"].predict_proba(structured_matrix)[0][1])
    transform = bundle.get("text_transform")
    route_language = str(bundle.get("language") or "unknown")
    text_content = (
        behavior_token_text_v3(content, route_language)
        if transform == BEHAVIOR_TOKEN_VERSION_V3
        else behavior_token_text_v2(content, route_language)
        if transform == BEHAVIOR_TOKEN_VERSION_V2
        else behavior_token_text(content, route_language)
        if transform == BEHAVIOR_TOKEN_VERSION
        else content
    )
    word = bundle["word_vectorizer"].transform([text_content])
    char = bundle["char_vectorizer"].transform([text_content])
    matrix = hstack(
        [structured_matrix, word, char],
        format="csr",
        dtype="float32",
    )
    return float(bundle["model"].predict_proba(matrix)[0][1])


def _route_probabilities_batch(
    bundle: Any,
    contents: list[str],
    structured_rows: list[dict[str, float]],
    feature_mode: str,
) -> list[float]:
    """Predict several files through one validated language-route matrix."""

    import numpy
    from scipy.sparse import csr_matrix, hstack

    if not isinstance(bundle, dict):
        raise TypeError("hybrid XGBoost route is not a bundle")
    if len(contents) != len(structured_rows):
        raise ValueError("XGBoost batch content/feature lengths differ")
    if not contents:
        return []
    feature_names = list(bundle.get("feature_names") or FEATURE_NAMES)
    structured_matrix = csr_matrix(numpy.asarray([
        [
            float(structured_features.get(name, 0.0))
            for name in feature_names
        ]
        for structured_features in structured_rows
    ], dtype="float32"))
    if feature_mode == "structured_static":
        matrix = structured_matrix
    else:
        transform = bundle.get("text_transform")
        route_language = str(bundle.get("language") or "unknown")
        text_contents = [
            (
                behavior_token_text_v3(content, route_language)
                if transform == BEHAVIOR_TOKEN_VERSION_V3
                else behavior_token_text_v2(content, route_language)
                if transform == BEHAVIOR_TOKEN_VERSION_V2
                else behavior_token_text(content, route_language)
                if transform == BEHAVIOR_TOKEN_VERSION
                else content
            )
            for content in contents
        ]
        word = bundle["word_vectorizer"].transform(text_contents)
        char = bundle["char_vectorizer"].transform(text_contents)
        matrix = hstack(
            [structured_matrix, word, char],
            format="csr",
            dtype="float32",
        )
    probabilities = bundle["model"].predict_proba(matrix)[:, 1]
    return [float(value) for value in probabilities]


def _candidate_probability(
    *,
    model: Any,
    content: str,
    language: str,
    include_rules: bool,
    feature_names: list[str],
    language_route: dict[str, Any] | None,
) -> float:
    """Run the same XGBoost route for a line-masked candidate."""

    return _candidate_probabilities(
        model=model,
        contents=[content],
        language=language,
        include_rules=include_rules,
        feature_names=feature_names,
        language_route=language_route,
    )[0]


def _candidate_probabilities(
    *,
    model: Any,
    contents: list[str],
    language: str,
    include_rules: bool,
    feature_names: list[str],
    language_route: dict[str, Any] | None,
) -> list[float]:
    """Score line-masked candidates in one vectorizer/model batch."""

    if not contents:
        return []
    feature_mode = (
        str(language_route.get("feature_mode") or "")
        if isinstance(language_route, dict)
        else ""
    )
    # Hybrid routes consume the rule-independent structured schema.  The old
    # attribution path first extracted legacy rule features, discarded them,
    # and then extracted the same candidates again without rules.  Selecting
    # the route schema before extraction keeps the prediction matrix identical
    # while removing one full rules/static pass per masked candidate.
    route_uses_structured_features = feature_mode in {
        "hybrid_hash",
        "structured_static",
    }
    feature_rows = [
        extract_static_features(
            content,
            language,
            include_rules=(
                False
                if route_uses_structured_features
                else include_rules
            ),
        )
        for content in contents
    ]
    if feature_mode in {"hybrid_hash", "structured_static"}:
        import numpy
        from scipy.sparse import csr_matrix, hstack

        if not isinstance(model, dict):
            raise TypeError("hybrid XGBoost route is not a bundle")
        route_feature_names = list(model.get("feature_names") or FEATURE_NAMES)
        structured_matrix = csr_matrix(numpy.asarray([
            [float(row.get(name, 0.0)) for name in route_feature_names]
            for row in feature_rows
        ], dtype="float32"))
        if feature_mode == "structured_static":
            matrix = structured_matrix
        else:
            transform = model.get("text_transform")
            route_language = str(model.get("language") or "unknown")
            text_contents = [
                (
                    behavior_token_text_v3(content, route_language)
                    if transform == BEHAVIOR_TOKEN_VERSION_V3
                    else behavior_token_text_v2(content, route_language)
                    if transform == BEHAVIOR_TOKEN_VERSION_V2
                    else behavior_token_text(content, route_language)
                    if transform == BEHAVIOR_TOKEN_VERSION
                    else content
                )
                for content in contents
            ]
            word = model["word_vectorizer"].transform(text_contents)
            char = model["char_vectorizer"].transform(text_contents)
            matrix = hstack(
                [structured_matrix, word, char],
                format="csr",
                dtype="float32",
            )
        probabilities = model["model"].predict_proba(matrix)[:, 1]
        return [float(value) for value in probabilities]
    vectors = [
        [float(features.get(name, 0.0)) for name in feature_names]
        for features in feature_rows
    ]
    return [float(value) for value in model.predict_proba(vectors)[:, 1]]


def _set_line_attributions(
    engines: list[dict[str, Any]],
    attributions: list[dict[str, float | int | str]],
) -> None:
    for engine in engines:
        metadata = engine.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("task") != "malicious_intent"
        ):
            continue
        metadata["line_attributions"] = attributions
        metadata["explanation_method"] = (
            "line_occlusion"
            if attributions
            else "not_generated"
        )
        return


def _line_attribution_context(
    content: str,
) -> tuple[list[str], list[int]] | None:
    if len(content) > ATTRIBUTION_MAX_CHARACTERS:
        return None
    lines = content.splitlines()
    if len(lines) > ATTRIBUTION_MAX_LINES:
        return None
    nonempty = [
        index
        for index, line in enumerate(lines)
        if line.strip()
    ]
    if len(nonempty) < 2:
        return None
    candidates = list(nonempty)
    if len(candidates) > ATTRIBUTION_MAX_INDIVIDUAL_LINES:
        candidates = _attribution_hotspot_lines(
            lines,
            nonempty,
            ATTRIBUTION_MAX_INDIVIDUAL_LINES,
        )
    return lines, candidates


def _masked_line_candidates(
    lines: list[str],
    candidates: list[int],
) -> list[str]:
    output = []
    for candidate in candidates:
        masked = list(lines)
        masked[candidate] = ""
        output.append("\n".join(masked))
    return output


def _line_attributions(
    content: str,
    baseline_probability: float,
    predict_probability,
    *,
    predict_probabilities=None,
    cancel_event: object | None = None,
) -> list[dict[str, float | int | str]]:
    """Explain a file score by measuring probability drops after line masking.

    Short files are evaluated line by line. Long files use a linear lexical
    hotspot index to select a bounded candidate set, then evaluate those lines
    together in one exact model batch. The contribution remains the model's
    actual probability drop rather than an attention visualization.
    """

    context = _line_attribution_context(content)
    if context is None:
        return []
    lines, candidates = context

    cache: dict[tuple[int, ...], float] = {}

    def score_many(groups: list[list[int]]) -> list[float]:
        raise_if_cancelled(cancel_event)
        missing: list[tuple[int, ...]] = []
        masked_contents: list[str] = []
        for indices in groups:
            key = tuple(sorted(indices))
            if key in cache or key in missing:
                continue
            masked = list(lines)
            for index in key:
                masked[index] = ""
            missing.append(key)
            masked_contents.append("\n".join(masked))
        if missing:
            if predict_probabilities is not None:
                values = predict_probabilities(masked_contents)
                raise_if_cancelled(cancel_event)
            else:
                values = []
                for candidate in masked_contents:
                    raise_if_cancelled(cancel_event)
                    values.append(predict_probability(candidate))
            if len(values) != len(missing):
                raise ValueError("batched XGBoost attribution response is incomplete")
            cache.update({
                key: float(value)
                for key, value in zip(missing, values)
            })
        return [
            max(0.0, baseline_probability - cache[tuple(sorted(indices))])
            for indices in groups
        ]

    def score(indices: list[int]) -> float:
        return score_many([indices])[0]

    candidate_drops = score_many([[index] for index in candidates])
    ranked_lines = []
    for index, drop in zip(candidates, candidate_drops):
        raise_if_cancelled(cancel_event)
        if drop < ATTRIBUTION_MIN_DROP:
            continue
        ranked_lines.append({
            "line": index + 1,
            "snippet": lines[index].strip()[:180],
            "probability_drop": round(drop, 6),
            "masked_probability": round(
                max(0.0, baseline_probability - drop),
                6,
            ),
            "baseline_probability": round(baseline_probability, 6),
        })
    ranked_lines.sort(
        key=lambda item: float(item["probability_drop"]),
        reverse=True,
    )
    selected = ranked_lines[:ATTRIBUTION_TOP_K]
    total_drop = sum(float(item["probability_drop"]) for item in selected)
    for item in selected:
        item["contribution_percent"] = round(
            100.0 * float(item["probability_drop"]) / total_drop,
            1,
        ) if total_drop else 0.0
    return selected


_ATTRIBUTION_HOTSPOT_PATTERN = re.compile(
    r"(?:\$_(?:get|post|request|cookie|server|files)|"
    r"\b(?:eval|assert|exec|system|shell_exec|passthru|popen|proc_open|"
    r"base64_decode|gzinflate|gzuncompress|str_rot13|preg_replace|"
    r"create_function|call_user_func|include|require|curl_exec|"
    r"file_put_contents|fopen|unlink|chmod|move_uploaded_file|"
    r"socket|connect|wget|powershell|cmd\.exe|/bin/sh)\b|"
    r"(?:https?://|[A-Za-z0-9+/]{80,}={0,2}))",
    re.IGNORECASE,
)


def _attribution_hotspot_lines(
    lines: list[str],
    indices: list[int],
    limit: int,
) -> list[int]:
    """Rank likely model-driving lines in one linear source pass."""

    ranked = []
    for index in indices:
        line = lines[index]
        matches = len(_ATTRIBUTION_HOTSPOT_PATTERN.findall(line))
        symbol_count = sum(character in "$(){}[];|&^" for character in line)
        density = symbol_count / max(1, len(line))
        ranked.append((
            matches > 0,
            matches,
            density,
            min(len(line), 2000),
            -index,
            index,
        ))
    ranked.sort(reverse=True)
    return [item[-1] for item in ranked[:limit]]


def _partition_indices(
    indices: list[int],
    group_count: int,
) -> list[list[int]]:
    if not indices:
        return []
    size = max(1, math.ceil(len(indices) / max(1, group_count)))
    return [
        indices[start:start + size]
        for start in range(0, len(indices), size)
    ]
