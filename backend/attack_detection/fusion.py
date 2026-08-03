"""AI-authoritative fusion for malicious-code decisions.

Validated AI models exclusively own the malicious/benign decision and its
risk score. Rules and static analysis are retained as an explanation layer;
they never override, break ties, or act as a fallback verdict.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .task_policy import is_active_finding


AI_MALICIOUS_ENGINE_NAMES = {
    "codet5p",
    "gatv2",
    "xgboost_malicious",
    "xgboost_project_malicious",
}


def fuse_engine_results(engines: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        engine for engine in engines
        if engine.get("status") == "completed"
    ]
    findings = [
        finding
        for engine in completed
        for finding in engine.get("findings", [])
        if isinstance(finding, dict) and is_active_finding(finding)
    ]
    type_counts = Counter(
        str(item.get("risk_type") or item.get("behavior") or "unknown")
        for item in findings
    )
    malicious_evidence = type_counts["malicious"] > 0
    external_review_signal = any(
        engine.get("name") in {"hash_reputation", "pe_static"}
        and int(engine.get("risk_score") or 0) > 0
        for engine in completed
    )

    ai_engines = [
        engine for engine in completed
        if _is_malicious_ai_engine(engine)
        and engine.get("probability") is not None
    ]
    ai_states = [_ai_state(engine) for engine in ai_engines]
    decisive_states = [
        state for state in ai_states
        if state["decisive"]
    ]
    decisive_labels = {
        str(state["decision"]) for state in decisive_states
    }
    ai_conflict = len(decisive_labels) > 1
    ai_decision = (
        next(iter(decisive_labels))
        if len(decisive_labels) == 1
        else None
    )
    ai_unresolved_reason = _unresolved_reason(
        ai_states,
        decisive_states,
        ai_conflict,
    )

    if ai_decision == "malicious":
        decision = "malicious"
        decision_authority = "ai"
        decision_basis = (
            "ai_consensus"
            if len(decisive_states) > 1
            else "ai_model"
        )
    elif ai_decision == "benign":
        decision = "benign"
        decision_authority = "ai"
        decision_basis = (
            "ai_consensus"
            if len(decisive_states) > 1
            else "ai_model"
        )
    elif external_review_signal:
        # Reputation and binary metadata may justify review, but they still do
        # not produce a malicious verdict when no AI model has decided.
        decision = "unknown"
        decision_authority = "external_context"
        decision_basis = "external_context"
    else:
        # Rules may explain evidence but cannot manufacture an AI verdict.
        decision = "unknown"
        decision_authority = "unresolved"
        decision_basis = "unresolved"

    risk_score = _risk_score(
        decisive_states,
        ai_states,
        ai_conflict,
    )
    explanation_findings = findings if decision == "malicious" else []
    explanation_type_counts = Counter(
        str(item.get("risk_type") or item.get("behavior") or "unknown")
        for item in explanation_findings
    )
    return {
        "final_decision": decision,
        "risk_score": risk_score,
        "risk_level": (
            "unknown"
            if decision == "unknown"
            else risk_level(risk_score)
        ),
        "findings": explanation_findings,
        "category_counts": dict(Counter(
            str(item.get("category"))
            for item in explanation_findings
            if item.get("category")
        )),
        "risk_type_counts": dict(explanation_type_counts),
        "decision_basis": decision_basis,
        "decision_authority": decision_authority,
        "ai_decision": ai_decision,
        "ai_participated": bool(ai_states),
        "ai_model_count": len(ai_states),
        "ai_decisive_model_count": len(decisive_states),
        "ai_model_names": [
            str(state["name"]) for state in ai_states
        ],
        "ai_decisive_model_names": [
            str(state["name"]) for state in decisive_states
        ],
        "ai_conflict": ai_conflict,
        "ai_uncertain": bool(ai_states) and not decisive_states,
        "ai_model_states": ai_states,
        # Compatibility fields remain stable for persisted report readers, but
        # new scans can never use rules as a fallback decision.
        "rule_fallback_used": False,
        "rule_fallback_reason": None,
        "ai_unresolved_reason": ai_unresolved_reason if ai_decision is None else None,
        "rules_role": "explanation_only",
        "rule_disagrees_with_ai": bool(
            ai_decision == "benign" and malicious_evidence
        ),
    }


def _is_malicious_ai_engine(engine: dict[str, Any]) -> bool:
    name = str(engine.get("name") or "")
    if name in AI_MALICIOUS_ENGINE_NAMES:
        return True
    metadata = engine.get("metadata") or {}
    return (
        name.startswith("xgboost_")
        and metadata.get("task") == "malicious_intent"
    )


def _ai_state(engine: dict[str, Any]) -> dict[str, Any]:
    metadata = engine.get("metadata") or {}
    probability = float(engine.get("probability") or 0.0)
    threshold = float(engine.get("threshold") or 0.5)
    advisory = bool(metadata.get("advisory_only"))
    uncertain_low = _optional_float(metadata.get("uncertain_low"))
    uncertain_high = _optional_float(metadata.get("uncertain_high"))
    in_uncertainty_band = bool(
        uncertain_low is not None
        and uncertain_high is not None
        and uncertain_low <= probability <= uncertain_high
    )
    decision = str(engine.get("decision") or "")
    if decision not in {"malicious", "benign"}:
        raw_decision = str(metadata.get("raw_model_decision") or "")
        decision = (
            raw_decision
            if raw_decision in {"malicious", "benign"}
            else ("malicious" if probability >= threshold else "benign")
        )
    decisive = not advisory and not in_uncertainty_band
    return {
        "name": str(engine.get("name") or "unknown"),
        "decision": decision,
        "probability": probability,
        "threshold": threshold,
        "decisive": decisive,
        "uncertain": in_uncertainty_band,
        "advisory_only": advisory,
        "model_version": engine.get("model_version"),
    }


def _unresolved_reason(
    ai_states: list[dict[str, Any]],
    decisive_states: list[dict[str, Any]],
    ai_conflict: bool,
) -> str:
    if ai_conflict:
        return "ai_model_conflict"
    if not ai_states:
        return "ai_unavailable"
    if not decisive_states:
        if all(state["advisory_only"] for state in ai_states):
            return "ai_routes_not_validated"
        return "ai_uncertain"
    return "ai_unresolved"


def _risk_score(
    decisive_states: list[dict[str, Any]],
    ai_states: list[dict[str, Any]],
    ai_conflict: bool,
) -> int:
    if decisive_states and not ai_conflict:
        model_scores = [
            _model_risk_from_state(state)
            for state in decisive_states
        ]
        return min(100, max(model_scores, default=0))
    if ai_conflict:
        # A disagreement is review-worthy, but it is not a critical verdict.
        return min(64, max(
            (round(float(state["probability"]) * 64) for state in decisive_states),
            default=35,
        ))
    if ai_states:
        # Keep unresolved/advisory model output visible without promoting it
        # to a final malicious decision.
        return min(
            34,
            max(
                (
                    int(float(state["probability"]) * 34)
                    for state in ai_states
                ),
                default=0,
            ),
        )
    return 0


def risk_level(score: int) -> str:
    if score >= 85:
        return "critical"
    if score >= 65:
        return "high"
    if score >= 35:
        return "medium"
    if score > 0:
        return "low"
    return "safe"


def _model_risk_from_state(state: dict[str, Any]) -> int:
    probability = float(state["probability"])
    if state["decision"] == "malicious":
        return min(100, max(35, round(probability * 100)))
    threshold = max(float(state.get("threshold") or 0.5), 0.0001)
    return min(34, round((probability / threshold) * 34))


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
