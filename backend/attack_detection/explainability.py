"""Model-attribution shaping shared by live and historical reports."""

from __future__ import annotations

from typing import Any


AI_ONLY_CATEGORY = "AI Semantic Risk"


def build_ai_explainability(
    engines: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    ai_only_line_count: int,
    decision_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision_summary = decision_summary or {}
    attributed = [
        item for item in evidence_items
        if item.get("ai_attribution")
    ]
    attributed_lines = {
        _line_number((item.get("ai_attribution") or {}).get("line"))
        for item in attributed
    } - {None}
    corroborated_lines = {
        _line_number((item.get("ai_attribution") or {}).get("line"))
        for item in attributed
        if item.get("evidence_basis") == "ai_and_rule"
    } - {None}
    return {
        "method": "line_occlusion",
        "generated": bool(attributed),
        "attributed_line_count": len(attributed_lines),
        "corroborated_line_count": len(corroborated_lines),
        "ai_only_line_count": ai_only_line_count,
        "meaning": "逐行遮挡贡献解释，不是漏洞概率",
        "decision_authority": decision_summary.get(
            "decision_authority",
        ),
        "ai_decision": decision_summary.get("ai_decision"),
        "ai_model_names": list(
            decision_summary.get("ai_model_names") or []
        ),
        "rule_fallback_used": bool(
            decision_summary.get("rule_fallback_used"),
        ),
        "rule_fallback_reason": decision_summary.get(
            "rule_fallback_reason",
        ),
        "ai_unresolved_reason": decision_summary.get(
            "ai_unresolved_reason",
        ),
        "rules_role": "explanation_only",
        "xgboost": _engine_summary(engines, "xgboost_malicious"),
        "codet5p": _engine_summary(engines, "codet5p"),
    }


def order_evidence_items(
    rule_evidence: list[dict[str, Any]],
    ai_only_evidence: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Order only line/file localization items, not the file-level verdict."""

    corroborated = [
        item for item in rule_evidence
        if item.get("evidence_basis") == "ai_and_rule"
    ]
    rule_only = [
        item for item in rule_evidence
        if item.get("evidence_basis") != "ai_and_rule"
    ]
    ordered = [
        *corroborated,
        *ai_only_evidence,
        *rule_only,
    ]
    return ordered[:max(0, limit)]


def merge_model_line_attributions(
    evidence_items: list[dict[str, Any]],
    engines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach XGBoost line occlusion to rule evidence and retain AI-only lines.

    Attribution describes which lines influence the model score.  It is not
    promoted to confirmed malicious/vulnerable evidence unless an independent
    finding exists at the same line.
    """

    attributions = _xgboost_attributions(engines)
    if not attributions:
        return [
            _with_rule_only_basis(item)
            for item in evidence_items
        ], []

    output = []
    used: set[int] = set()
    for item in evidence_items:
        shaped = dict(item)
        trace_steps = []
        for raw_step in shaped.get("trace_steps") or []:
            if not isinstance(raw_step, dict):
                continue
            step = dict(raw_step)
            step_index = _nearest_attribution_index(
                _line_number(step.get("line")),
                attributions,
            )
            if step_index is not None:
                attribution = dict(attributions[step_index])
                step["ai_supported"] = True
                step["ai_attribution"] = attribution
                used.add(step_index)
            trace_steps.append(step)
        if trace_steps:
            shaped["trace_steps"] = trace_steps
        line = _line_number(shaped.get("line"))
        nearest_index = _nearest_attribution_index(line, attributions)
        if nearest_index is None:
            output.append(_with_rule_only_basis(shaped))
            continue
        attribution = dict(attributions[nearest_index])
        used.add(nearest_index)
        shaped["ai_attribution"] = attribution
        shaped["evidence_basis"] = "ai_and_rule"
        shaped["basis_text"] = (
            "XGBoost逐行遮挡显示该位置会提高模型风险分，"
            "并且同一位置存在规则或静态解释证据；最终恶意结论仍只来自AI。"
        )
        output.append(shaped)

    ai_only = [
        _ai_only_evidence(item)
        for index, item in enumerate(attributions)
        if index not in used and _should_display_ai_only(item)
    ]
    return output, ai_only


def _xgboost_attributions(
    engines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for engine in engines:
        if (
            isinstance(engine, dict)
            and engine.get("name") == "xgboost_malicious"
            and engine.get("status") == "completed"
        ):
            metadata = engine.get("metadata") or {}
            values = metadata.get("line_attributions") or []
            output = []
            for value in values:
                if not isinstance(value, dict):
                    continue
                shaped = dict(value)
                shaped["model_name"] = "xgboost_malicious"
                shaped["model_probability"] = engine.get("probability")
                shaped["model_threshold"] = engine.get("threshold")
                shaped["model_decision"] = engine.get("decision")
                shaped["raw_model_decision"] = metadata.get(
                    "raw_model_decision",
                )
                shaped["advisory_only"] = bool(
                    metadata.get("advisory_only"),
                )
                output.append(shaped)
            return output
    return []


def _nearest_attribution_index(
    line: int | None,
    attributions: list[dict[str, Any]],
) -> int | None:
    if line is None:
        return None
    candidates = [
        (abs(line - int(item.get("line") or 0)), index)
        for index, item in enumerate(attributions)
        if item.get("line") is not None
    ]
    if not candidates:
        return None
    distance, index = min(candidates)
    return index if distance == 0 else None


def _with_rule_only_basis(item: dict[str, Any]) -> dict[str, Any]:
    shaped = dict(item)
    shaped.setdefault("evidence_basis", "rule_only")
    shaped.setdefault(
        "basis_text",
        "该位置命中了规则或静态分析解释证据；它用于说明行为与修复方式，不参与最终恶意判定或风险分。",
    )
    return shaped


def _should_display_ai_only(item: dict[str, Any]) -> bool:
    probability = float(item.get("model_probability") or 0.0)
    threshold = float(item.get("model_threshold") or 0.5)
    return (
        item.get("raw_model_decision") == "malicious"
        or probability >= max(0.35, threshold * 0.75)
    )


def _ai_only_evidence(attribution: dict[str, Any]) -> dict[str, Any]:
    line = _line_number(attribution.get("line"))
    probability_drop = float(attribution.get("probability_drop") or 0.0)
    contribution = float(attribution.get("contribution_percent") or 0.0)
    return {
        "source": "xgboost_attribution",
        "rule_id": None,
        "risk_type": "ai_signal",
        "category": AI_ONLY_CATEGORY,
        "severity": None,
        "description": (
            f"遮挡该行后，XGBoost恶意概率下降"
            f"{probability_drop * 100:.2f}个百分点；"
            "这说明该行是模型恶意判定中的高贡献位置。"
        ),
        "harm": (
            "该位置包含模型认为值得复核的语义或行为特征；"
            "行级归因用于定位模型判断依据，仅凭该归因不能直接定性为漏洞。"
        ),
        "repair_advice": "结合上下文人工复核该行及其调用链，不要仅凭模型贡献度直接修改代码。",
        "repair_suggestions": [
            "结合上下文人工复核该行及其调用链。",
            "确认输入来源、敏感接口和实际副作用后再决定是否修复。",
            "如果该行为确有业务必要，请补充边界校验、权限控制和审计记录。",
        ],
        "remediation_references": [],
        "owasp_category": None,
        "api_security_category": None,
        "risk_domains": ["AI复核"],
        "cwe": None,
        "line": line,
        "snippet": str(attribution.get("snippet") or "")[:180],
        "code_context": [],
        "suspicion_score": round(min(100.0, max(1.0, contribution)), 1),
        "suspicion_basis": "XGBoost逐行遮挡的相对贡献，不是漏洞概率",
        "evidence_basis": "ai_only",
        "basis_text": (
            "该位置由XGBoost逐行遮挡归因直接定位，依据是遮挡前后的模型风险分变化。"
        ),
        "ai_attribution": dict(attribution),
    }


def _line_number(value: object) -> int | None:
    try:
        line = int(value or 0)
    except (TypeError, ValueError):
        return None
    return line if line > 0 else None


def _engine_summary(
    engines: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    engine = next(
        (
            item for item in engines
            if isinstance(item, dict) and item.get("name") == name
        ),
        {},
    )
    return {
        "available": engine.get("status") == "completed",
        "status": engine.get("status") or "not_run",
        "decision": engine.get("decision"),
        "probability": engine.get("probability"),
        "threshold": engine.get("threshold"),
        "model_version": engine.get("model_version"),
    }
