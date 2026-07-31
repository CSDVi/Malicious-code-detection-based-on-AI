"""Detection mode orchestration and compatibility shaping."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from attack_detection.engines.gat_engine import GATEngine
from attack_detection.engines.codet5p_engine import CodeT5PEngine
from attack_detection.engines.rule_engine import RuleEngine
from attack_detection.engines.xgb_engine import XGBoostEngine
from attack_detection.cancellation import raise_if_cancelled
from attack_detection.explainability import (
    build_ai_decision_evidence,
    build_ai_explainability,
    merge_model_line_attributions,
    order_evidence_items,
)
from attack_detection.fusion import fuse_engine_results
from attack_detection.ml import classifier
from attack_detection.binary_analysis import BinaryAnalysisEngine
from attack_detection.owasp_coverage import OWASP_TOP10_2025
from attack_detection.reputation import HashReputationEngine
from attack_detection.remediation import remediation_for_finding
from attack_detection.risk_taxonomy import taxonomy_for_category
from attack_detection.sandbox import SandboxEngine
from attack_detection.static_analysis import StaticAnalysisEngine
from attack_detection.task_policy import is_active_finding, task_enabled

MODES = {"quick", "standard", "deep", "auto"}
MAX_REPAIR_SUGGESTIONS = 20
MAX_FINDINGS_PER_ENGINE = 100

ATTACK_BY_CATEGORY = {
    "Command Execution": {"id": "T1059", "name": "Command and Scripting Interpreter"},
    "Download or Remote Load": {"id": "T1105", "name": "Ingress Tool Transfer"},
    "Download and Execute": {"id": "T1105", "name": "Ingress Tool Transfer"},
    "Obfuscated Payload": {"id": "T1027", "name": "Obfuscated Files or Information"},
    "Credential Collection": {"id": "T1555", "name": "Credentials from Password Stores"},
    "Credential Exfiltration": {"id": "T1041", "name": "Exfiltration Over C2 Channel"},
    "Persistence": {"id": "T1546", "name": "Event Triggered Execution"},
    "Install Hook Execution": {"id": "T1546", "name": "Event Triggered Execution"},
    "Credential Exfiltration": {"id": "T1041", "name": "Exfiltration Over C2 Channel"},
    "JavaScript 静态去混淆": {"id": "T1027", "name": "Obfuscated Files or Information"},
}

CATEGORY_DESCRIPTION_ZH = {
    "SQL Injection": "外部输入可能进入 SQL 语句，攻击者可能改变查询结构或读取未授权数据。",
    "XSS": "未经可信编码的数据可能进入 HTML 或脚本执行上下文。",
    "Command Execution": "外部输入或可控参数可能到达系统命令执行接口。",
    "SSRF": "外部输入可能控制服务端请求地址，并访问内网或云元数据服务。",
    "Path Traversal": "外部输入可能参与文件路径构造并越出预期目录。",
    "Unsafe Deserialization": "不可信数据可能被反序列化为对象并触发危险行为。",
    "Secret Exposure": "代码中出现疑似硬编码密钥、口令或访问令牌。",
    "WebShell": "请求参数被送入动态执行接口，符合网页后门入口特征。",
    "Download or Remote Load": "代码包含从远程地址下载或加载内容的行为。",
    "Download and Execute": "远程内容下载后可能被赋予执行权限或直接启动。",
    "Obfuscated Payload": "编码、字符拼接或解码操作可能用于隐藏实际载荷。",
    "Install Hook Execution": "软件包安装钩子中出现下载器、解释器或命令执行行为。",
    "Credential Exfiltration": "代码可能将凭据或本地敏感信息发送到外部地址。",
    "Credential Collection": "本地凭据读取与网络发送行为在相邻代码中同时出现。",
    "Persistence": "代码可能创建计划任务、启动项或服务以维持长期执行。",
}

CATEGORY_REPAIR_ZH = {
    "SQL Injection": "使用参数化查询或 ORM 参数绑定，不要拼接来自请求的数据。",
    "XSS": "按输出上下文进行编码，优先使用 textContent 或可信模板转义机制。",
    "Command Execution": "改用不经过 Shell 的安全 API；必须执行时使用固定命令和参数白名单。",
    "SSRF": "限制允许访问的协议和域名，并阻断回环、内网及云元数据地址。",
    "Path Traversal": "在固定根目录下解析规范化路径，并验证最终路径仍位于根目录内。",
    "Unsafe Deserialization": "改用安全数据格式；确需反序列化时限制类型并校验数据来源和签名。",
    "Secret Exposure": "将密钥迁移到环境变量或密钥管理服务，并立即轮换已暴露凭据。",
    "WebShell": "删除动态执行入口，隔离文件并审计访问日志和相关提交记录。",
    "Download or Remote Load": "限制可信下载源，校验内容签名或哈希，禁止下载后直接执行。",
    "Download and Execute": "分离下载与执行流程，仅允许经过签名验证的固定产物运行。",
    "Obfuscated Payload": "解码并人工审查实际内容，移除无业务必要的混淆和动态执行。",
    "Install Hook Execution": "删除安装阶段的网络下载和命令执行逻辑，并审查发布版本差异。",
    "Credential Exfiltration": "移除敏感信息外传路径，轮换受影响凭据并检查访问日志。",
    "Credential Collection": "删除非必要的凭据读取行为，并限制进程对敏感文件和环境变量的访问。",
    "Persistence": "移除未经授权的启动项、计划任务或服务，并检查受影响主机。",
}

CATEGORY_HARM_ZH = {
    "SQL Injection": "攻击者可能改变数据库查询，读取、修改或删除未授权数据。",
    "XSS": "攻击者可能在用户浏览器中执行脚本，窃取会话或篡改页面内容。",
    "Command Execution": "攻击者可能让服务器执行任意系统命令并进一步控制主机。",
    "SSRF": "攻击者可能借服务器访问内网、云元数据或其他不可公开访问的服务。",
    "Path Traversal": "攻击者可能越出预期目录，读取或覆盖服务器上的敏感文件。",
    "Unsafe Deserialization": "恶意序列化数据可能触发非预期对象创建、代码执行或状态篡改。",
    "Secret Exposure": "泄露的密钥、口令或令牌可能被用于冒充身份和访问受保护资源。",
    "WebShell": "攻击者可能通过网络请求远程执行代码并长期控制服务器。",
    "Download or Remote Load": "远程内容可能在来源或完整性未验证时进入系统。",
    "Download and Execute": "未经验证的远程载荷可能被直接执行并控制运行环境。",
    "Obfuscated Payload": "混淆可能隐藏真实行为，使恶意代码绕过人工审查或简单检测。",
    "Install Hook Execution": "恶意代码可能在安装依赖时自动执行并影响开发或构建环境。",
    "Credential Exfiltration": "密码、令牌或密钥可能被发送到攻击者控制的外部地址。",
    "Credential Collection": "程序可能收集本地凭据，为后续窃取或横向移动提供条件。",
    "Persistence": "恶意程序可能在系统重启后继续运行并维持长期控制。",
    "Insecure Direct Object Reference": "用户可能通过修改对象编号访问其他用户或租户的数据。",
    "Debug Mode Enabled": "调试页面可能泄露堆栈、路径、配置和内部实现细节。",
    "TLS Verification Disabled": "网络通信可能遭到中间人劫持，响应内容或凭据可能被篡改。",
    "Permissive CORS": "非预期网站可能借用户身份读取受保护的接口响应。",
    "Weak Cryptographic Hash": "攻击者可能更容易伪造完整性结果或破解使用弱哈希保护的凭据。",
    "ECB Cipher Mode": "重复数据结构可能从密文中暴露，且无法可靠保证内容完整性。",
    "Insecure Randomness": "令牌、验证码或会话标识可能被预测并用于冒充合法用户。",
    "Client Side Only Authorization": "攻击者可绕过前端限制直接调用后端敏感接口。",
    "Missing Abuse Controls": "攻击者可能自动化滥用高成本操作并耗尽系统或业务资源。",
    "JWT Verification Disabled": "攻击者可能伪造或篡改令牌并冒充其他用户或管理员。",
    "Plaintext Password Handling": "固定或明文口令一旦泄露，可能直接导致账户被接管。",
    "Sensitive Data Logging": "日志读取者可能获得密码、令牌、私钥或会话信息。",
    "Log Injection": "攻击者可能伪造日志事件、隐藏真实行为或干扰安全审计。",
    "Stack Trace Disclosure": "外部用户可能获得路径、框架和内部调用信息，用于准备进一步攻击。",
    "Unbounded Resource Consumption": "无边界请求、重试或循环可能耗尽CPU、内存、连接和存储。",
    "Empty Exception Handler": "错误被静默忽略后，程序可能带着不完整或不安全状态继续运行。",
    "Fail Open Security Decision": "认证或授权组件异常时仍然放行，可能导致安全控制被直接绕过。",
}


class DetectionOrchestrator:
    def __init__(self) -> None:
        self.rule_engine = RuleEngine()
        self.xgb_engine = XGBoostEngine()
        self.codet5p_engine = CodeT5PEngine()
        self.gat_engine = GATEngine()
        self.static_engine = StaticAnalysisEngine()
        self.reputation_engine = HashReputationEngine()
        self.sandbox_engine = SandboxEngine()
        self.binary_engine = BinaryAnalysisEngine()

    def scan(
        self, filename: str, content: str, language: str, selected_mode: str = "auto",
        precomputed_semantic: dict[str, Any] | None = None,
        precomputed_quick_result: dict[str, Any] | None = None,
        raw_bytes: bytes | None = None,
        cancel_event: object | None = None,
        generate_line_attributions: bool | None = None,
        precomputed_xgb: dict[str, Any] | None = None,
        run_legacy_baseline: bool = True,
    ) -> dict[str, Any]:
        selected = selected_mode if selected_mode in MODES else "auto"
        if generate_line_attributions is None:
            generate_line_attributions = selected in {"standard", "deep"}
        raw_bytes = content.encode("utf-8", errors="ignore") if raw_bytes is None else raw_bytes
        raise_if_cancelled(cancel_event)
        engines, deferred_engines = self._quick_engines(
            content, language, raw_bytes, precomputed_quick_result, cancel_event,
            generate_line_attributions, precomputed_xgb,
            run_rule_engine=True,
            run_static_analysis=(selected != "quick"),
        )
        effective = "quick"
        escalation_reason = None

        if selected == "standard":
            engines.append(self._semantic_scan(
                content, language, precomputed_semantic, cancel_event,
            ))
            effective = "standard"
        elif selected == "deep":
            engines.append(self._semantic_scan(
                content, language, precomputed_semantic, cancel_event,
            ))
            raise_if_cancelled(cancel_event)
            engines.append(self.gat_engine.scan(content, language))
            effective = "deep"
        elif selected == "auto":
            reason = self._auto_escalation_reason(engines, content)
            if reason:
                if not generate_line_attributions and precomputed_quick_result is None:
                    explained_xgb = self.xgb_engine.scan(
                        content,
                        language,
                        generate_line_attributions=True,
                        cancel_event=cancel_event,
                    )
                    engines = [
                        engine
                        for engine in engines
                        if not str(engine.get("name") or "").startswith("xgboost_")
                    ] + explained_xgb
                engines.append(self._semantic_scan(
                    content, language, precomputed_semantic, cancel_event,
                ))
                effective = "standard"
                escalation_reason = reason
                if self._needs_deep_graph(engines, content):
                    raise_if_cancelled(cancel_event)
                    engines.append(self.gat_engine.scan(content, language))
                    effective = "deep"

        raise_if_cancelled(cancel_event)
        reputation = deferred_engines.get("hash_reputation")
        engines.append(
            reputation
            if reputation is not None
            else self.reputation_engine.scan(hashlib.sha256(raw_bytes).hexdigest())
        )
        raise_if_cancelled(cancel_event)
        sandbox = deferred_engines.get("isolated_sandbox")
        engines.append(
            sandbox
            if sandbox is not None
            else self.sandbox_engine.scan(
                filename, raw_bytes, hashlib.sha256(raw_bytes).hexdigest(),
            )
        )
        raise_if_cancelled(cancel_event)

        fused = fuse_engine_results(engines)
        legacy_ml = (
            classifier.predict(content, language)
            if run_legacy_baseline
            else _skipped_legacy_baseline()
        )
        raise_if_cancelled(cancel_event)
        return self._shape_result(
            filename=filename,
            content=content,
            language=language,
            selected_mode=selected,
            effective_mode=effective,
            engines=engines,
            fused=fused,
            legacy_ml=legacy_ml,
            escalation_reason=escalation_reason,
            raw_bytes=raw_bytes,
        )

    def _quick_engines(
        self,
        content: str,
        language: str,
        raw_bytes: bytes,
        precomputed_quick_result: dict[str, Any] | None,
        cancel_event: object | None,
        generate_line_attributions: bool,
        precomputed_xgb: dict[str, Any] | None,
        run_rule_engine: bool,
        run_static_analysis: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        """Reuse the first project pass instead of executing quick engines twice."""

        if precomputed_quick_result is not None:
            cached = [
                dict(engine)
                for engine in (precomputed_quick_result.get("engines") or [])
                if isinstance(engine, dict)
            ]
            base = [
                engine
                for engine in cached
                if (
                    engine.get("name") in {"rule_engine", "static_evidence", "pe_static"}
                    or str(engine.get("name") or "").startswith("xgboost_")
                )
            ]
            deferred = {
                str(engine.get("name")): engine
                for engine in cached
                if engine.get("name") in {"hash_reputation", "isolated_sandbox"}
            }
            if base:
                return base, deferred

        engines = []
        if run_rule_engine:
            engines.append(self.rule_engine.scan(content, language))
            raise_if_cancelled(cancel_event)
        engines.extend(self.xgb_engine.scan(
            content,
            language,
            generate_line_attributions=generate_line_attributions,
            cancel_event=cancel_event,
            precomputed_batch=precomputed_xgb,
        ))
        raise_if_cancelled(cancel_event)
        if run_static_analysis:
            engines.append(
                self.static_engine.scan(
                    content,
                    language,
                    raw_bytes=raw_bytes,
                )
            )
            raise_if_cancelled(cancel_event)
        return engines, {}

    def _semantic_scan(
        self,
        content: str,
        language: str,
        precomputed_semantic: dict[str, Any] | None,
        cancel_event: object | None,
    ) -> dict[str, Any]:
        if precomputed_semantic is not None:
            return precomputed_semantic
        codet5p = self.codet5p_engine.scan(
            content, language, cancel_event=cancel_event,
        )
        return codet5p

    def scan_binary(
        self, filename: str, payload: bytes, selected_mode: str = "auto",
        cancel_event: object | None = None,
    ) -> dict[str, Any]:
        """Scan a binary as bytes only; source/model engines are deliberately bypassed."""
        selected = selected_mode if selected_mode in MODES else "auto"
        sha256 = hashlib.sha256(payload).hexdigest()
        raise_if_cancelled(cancel_event)
        engines = [self.binary_engine.scan(filename, payload)]
        raise_if_cancelled(cancel_event)
        engines.append(self.reputation_engine.scan(sha256))
        raise_if_cancelled(cancel_event)
        engines.append(self.sandbox_engine.scan(filename, payload, sha256))
        raise_if_cancelled(cancel_event)
        legacy_ml = {
            "malicious_intent": {"label": "unknown", "probability": None, "available": False, "status": "not_applicable", "reason": "binary input bypasses source model"},
            "vulnerability_risk": {"label": "disabled", "probability": None, "available": False, "status": "disabled", "reason": "漏洞风险任务已从当前产品流程下线"},
            "model_version": None, "training_samples": None,
        }
        return self._shape_result(
            filename=filename, content="", language="binary", selected_mode=selected,
            effective_mode="quick", engines=engines, fused=fuse_engine_results(engines),
            legacy_ml=legacy_ml, escalation_reason=None, raw_bytes=payload,
        )

    def _auto_escalation_reason(self, engines: list[dict[str, Any]], content: str) -> str | None:
        rule = next((engine for engine in engines if engine.get("name") == "rule_engine"), {})
        xgb_engines = [
            engine for engine in engines
            if str(engine.get("name", "")).startswith("xgboost_")
            and task_enabled((engine.get("metadata") or {}).get("task"))
        ]
        if int(rule.get("risk_score") or 0) >= 65:
            return "high-risk rule evidence triggered standard-mode escalation"
        if xgb_engines and all(engine.get("status") == "unavailable" for engine in xgb_engines):
            return "XGBoost is unavailable, so auto mode requested CodeT5+ 220M verification when possible"
        for xgb in (engine for engine in xgb_engines if engine.get("status") == "completed"):
            probability = float(xgb.get("probability") or 0.0)
            metadata = xgb.get("metadata", {})
            if metadata.get("advisory_only") and metadata.get("raw_model_decision") == "malicious":
                return f"{xgb.get('name')} advisory fallback requested validated semantic verification"
            uncertain_low = metadata.get("uncertain_low")
            uncertain_high = metadata.get("uncertain_high")
            if uncertain_low is not None and uncertain_high is not None:
                if float(uncertain_low) <= probability <= float(uncertain_high):
                    return f"{xgb.get('name')} probability is in its validation uncertainty band"
        positive_decisions = {
            str(engine.get("decision")) for engine in xgb_engines
            if (
                engine.get("status") == "completed"
                and not (engine.get("metadata") or {}).get("advisory_only")
                and engine.get("decision") == "malicious"
            )
        }
        rule_decision = str(rule.get("decision") or "unknown")
        if (
            rule
            and positive_decisions
            and rule_decision not in positive_decisions
        ):
            return "rule engine and XGBoost decisions conflict"
        lowered = content.lower()
        if any(token in lowered for token in ("base64", "fromcharcode", "\\x", "atob(")):
            return "obfuscation indicators triggered standard-mode escalation"
        return None

    def _needs_deep_graph(self, engines: list[dict[str, Any]], content: str) -> bool:
        return "\nimport " in content or "require(" in content or "from " in content

    def _shape_result(
        self,
        filename: str,
        content: str,
        language: str,
        selected_mode: str,
        effective_mode: str,
        engines: list[dict[str, Any]],
        fused: dict[str, Any],
        legacy_ml: dict[str, Any],
        escalation_reason: str | None,
        raw_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        bounded_engines = []
        for engine in engines:
            findings = engine.get("findings")
            if not isinstance(findings, list) or len(findings) <= MAX_FINDINGS_PER_ENGINE:
                bounded_engines.append(engine)
                continue
            bounded = dict(engine)
            bounded["findings"] = sorted(
                (item for item in findings if isinstance(item, dict)),
                key=lambda item: int(item.get("severity") or 0),
                reverse=True,
            )[:MAX_FINDINGS_PER_ENGINE]
            metadata = dict(bounded.get("metadata") or {})
            metadata["total_finding_count"] = len(findings)
            metadata["returned_finding_count"] = len(bounded["findings"])
            metadata["findings_truncated"] = True
            bounded["metadata"] = metadata
            bounded_engines.append(bounded)
        engines = bounded_engines
        evidence_engine_names = {"rule_engine", "static_evidence", "pe_static"}
        raw_matches = [
            finding for engine in engines if engine.get("name") in evidence_engine_names
            for finding in engine.get("findings", [])
            if isinstance(finding, dict) and is_active_finding(finding)
        ]
        matches = [
            _normalize_finding(item, content, language)
            for item in raw_matches
        ]
        matches = _rule_explanations_for_decision(matches, fused)
        matches, ai_only_evidence = merge_model_line_attributions(
            matches,
            engines,
        )
        ai_decision_evidence = build_ai_decision_evidence(fused)
        evidence_items = order_evidence_items(
            matches,
            ai_only_evidence,
            ai_decision_evidence,
        )
        bytetcn = next((engine for engine in engines if engine.get("name") == "bytetcn"), {})
        codet5p = next((engine for engine in engines if engine.get("name") == "codet5p"), {})
        semantic_engine = codet5p if codet5p.get("status") == "completed" else bytetcn
        model_behaviors = [
            str(item.get("label")) for item in bytetcn.get("metadata", {}).get("behavior_labels", [])
            if item.get("label")
        ]
        model_cwes = [
            str(item.get("label")) for item in bytetcn.get("metadata", {}).get("cwe_labels", [])
            if item.get("label")
        ]
        categories = sorted({str(match["category"]) for match in matches} | set(model_behaviors))
        cwes = sorted({str(match["cwe"]) for match in matches if match.get("cwe")} | set(model_cwes))
        owasp_categories = sorted({
            str(match["owasp_category"])
            for match in matches
            if match.get("owasp_category")
        })
        api_security_categories = sorted({
            str(match["api_security_category"])
            for match in matches
            if match.get("api_security_category")
        })
        risk_domains = sorted({
            str(domain)
            for match in matches
            for domain in (match.get("risk_domains") or [])
            if domain
        })
        type_counts = Counter(str(match.get("risk_type", "unknown")) for match in matches)
        severity_by_type = {
            "malicious": sum(int(match["severity"]) for match in matches if match.get("risk_type") == "malicious"),
            "vulnerable": sum(int(match["severity"]) for match in matches if match.get("risk_type") == "vulnerable"),
        }
        malicious_ml = dict(legacy_ml["malicious_intent"])
        vulnerability_ml = dict(legacy_ml["vulnerability_risk"])
        malicious_xgb = next((engine for engine in engines if engine.get("name") == "xgboost_malicious"), {})
        malicious_task_engine = _preferred_task_engine(semantic_engine, malicious_xgb, "malicious_intent")
        evidence_lines = [
            {
                "line": match["line"],
                "rule_id": match["rule_id"],
                "risk_type": match.get("risk_type"),
                "category": match["category"],
                "cwe": match.get("cwe"),
                "snippet": match["snippet"],
            }
            for match in matches[:12]
        ]
        risk_reasons = []
        if fused.get("ai_decision") in {"malicious", "benign"}:
            model_names = "、".join(
                str(name) for name in fused.get("ai_decisive_model_names") or []
            )
            risk_reasons.append(
                f"AI主判：{model_names or '已验证模型'} 将文件判定为"
                f"{'恶意' if fused.get('ai_decision') == 'malicious' else '良性'}。"
            )
        if fused.get("rule_fallback_used"):
            risk_reasons.append(
                "AI无法完成可靠主判，规则已按回退策略参与结论："
                f"{fused.get('rule_fallback_reason') or '原因未记录'}。"
            )
        if fused.get("rule_disagrees_with_ai"):
            risk_reasons.append(
                "规则命中了恶意模式，但明确的AI良性结论未被规则覆盖；"
                "规则只保留为人工复核说明。"
            )
        risk_reasons.extend(
            f"{match['rule_id']}: {match['description']}"
            for match in matches
        )
        if escalation_reason:
            risk_reasons.append(f"auto escalation: {escalation_reason}")
        for engine in engines:
            if engine.get("status") in {"unavailable", "skipped", "failed"}:
                risk_reasons.append(f"{engine['name']}: {engine.get('status')} - {engine.get('reason') or engine.get('error')}")
        reputation = next((engine for engine in engines if engine.get("name") == "hash_reputation"), {})
        reputation_metadata = reputation.get("metadata") or {}
        if reputation.get("status") == "completed" and int(reputation_metadata.get("malicious") or 0) + int(reputation_metadata.get("suspicious") or 0) > 0:
            risk_reasons.append(
                f"VirusTotal hash reputation: malicious={reputation_metadata.get('malicious', 0)}, suspicious={reputation_metadata.get('suspicious', 0)}; hash evidence requires review"
            )
        if not risk_reasons:
            risk_reasons = ["No high-risk rule evidence or executable positive model decision was found."]
        suggestions = []
        seen_suggestions: set[str] = set()
        for match in matches:
            advice_values = match.get("repair_suggestions") or [match.get("repair_advice")]
            for advice_value in advice_values:
                advice = " ".join(str(advice_value or "").split())
                if advice and advice not in seen_suggestions:
                    suggestions.append(advice)
                    seen_suggestions.add(advice)
        if not suggestions:
            if fused.get("final_decision") == "malicious":
                suggestions.append(
                    "先隔离该文件并结合调用链人工复核；当前没有规则命中可用于生成更具体的修复步骤。"
                )
            elif fused.get("rule_fallback_used"):
                suggestions.append(
                    "AI本次未形成可靠结论，请结合调用链、业务上下文和规则证据人工复核。"
                )
            else:
                suggestions.append("保持依赖版本锁定，审查代码变更，并使用最小权限运行配置。")
        remediation_references = []
        seen_reference_urls: set[str] = set()
        for match in matches:
            for reference in match.get("remediation_references") or []:
                if not isinstance(reference, dict):
                    continue
                url = str(reference.get("url") or "")
                if not url or url in seen_reference_urls:
                    continue
                remediation_references.append(dict(reference))
                seen_reference_urls.add(url)
        attack_techniques = []
        for category in categories:
            technique = ATTACK_BY_CATEGORY.get(category)
            if technique and technique not in attack_techniques:
                attack_techniques.append(technique)
        engine_votes = {
            "rule_engine": {
                "decision": next((engine.get("decision") for engine in engines if engine.get("name") == "rule_engine"), "unknown"),
                "malicious_hits": type_counts["malicious"],
                "vulnerability_hits": type_counts["vulnerable"],
                "hits": len(matches),
            },
            "xgboost": {
                str(engine.get("metadata", {}).get("task") or engine.get("name")): engine
                for engine in engines if str(engine.get("name", "")).startswith("xgboost_")
            },
            "bytetcn": next((engine for engine in engines if engine.get("name") == "bytetcn"), {"status": "skipped", "probability": None}),
            "codet5p": next((engine for engine in engines if engine.get("name") == "codet5p"), {"status": "skipped", "probability": None}),
            "gatv2": next((engine for engine in engines if engine.get("name") == "gatv2"), {"status": "skipped", "probability": None}),
            "static_evidence": next((engine for engine in engines if engine.get("name") == "static_evidence"), {"status": "skipped"}),
            "pe_static": next((engine for engine in engines if engine.get("name") == "pe_static"), {"status": "skipped"}),
            "hash_reputation": next((engine for engine in engines if engine.get("name") == "hash_reputation"), {"status": "unavailable"}),
            "isolated_sandbox": next((engine for engine in engines if engine.get("name") == "isolated_sandbox"), {"status": "unavailable"}),
            "malicious_model": malicious_ml,
            "vulnerability_model": vulnerability_ml,
            "policy": {
                "decision": "ai_primary_rule_explanation_and_conditional_fallback",
                "vulnerability_model": "disabled",
            },
        }
        raw_bytes = content.encode("utf-8", errors="ignore") if raw_bytes is None else raw_bytes
        hashes = {
            "md5": hashlib.md5(raw_bytes).hexdigest(),
            "sha1": hashlib.sha1(raw_bytes).hexdigest(),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
        return {
            "filename": filename,
            "language": language,
            # file_hash remains the compatibility alias used by existing filters.
            "file_hash": hashes["sha256"],
            "hashes": hashes,
            "selected_mode": selected_mode,
            "effective_mode": effective_mode,
            "final_decision": fused["final_decision"],
            "risk_score": fused["risk_score"],
            "risk_level": fused["risk_level"],
            "decision_authority": fused.get("decision_authority"),
            "decision_basis": fused.get("decision_basis"),
            "ai_decision": fused.get("ai_decision"),
            "ai_participated": fused.get("ai_participated", False),
            "ai_model_count": fused.get("ai_model_count", 0),
            "ai_decisive_model_count": fused.get(
                "ai_decisive_model_count",
                0,
            ),
            "ai_model_names": fused.get("ai_model_names", []),
            "ai_decisive_model_names": fused.get(
                "ai_decisive_model_names",
                [],
            ),
            "ai_conflict": fused.get("ai_conflict", False),
            "ai_uncertain": fused.get("ai_uncertain", False),
            "ai_model_states": fused.get("ai_model_states", []),
            "rule_fallback_used": fused.get("rule_fallback_used", False),
            "rule_fallback_reason": fused.get("rule_fallback_reason"),
            "rule_disagrees_with_ai": fused.get(
                "rule_disagrees_with_ai",
                False,
            ),
            "confidence": None,
            "engines": engines,
            "findings": fused["findings"],
            "escalation_reason": escalation_reason,
            "matches": matches,
            "categories": categories,
            "cwes": cwes,
            "owasp_categories": owasp_categories,
            "api_security_categories": api_security_categories,
            "risk_domains": risk_domains,
            "owasp_top10_2025": {
                "coverage_level": "baseline",
                "covered_categories": len(OWASP_TOP10_2025),
                "total_categories": 10,
                "absolute_coverage_claimed": False,
            },
            "attack_techniques": attack_techniques,
            "category_counts": dict(Counter(str(match["category"]) for match in matches)),
            "risk_type_counts": dict(type_counts),
            "ml": malicious_ml,
            "malicious_intent": _task_view(malicious_task_engine, type_counts["malicious"]),
            "vulnerability_risk": _disabled_task_view(),
            "feature_count": len(matches),
            "line_count": len(content.splitlines()),
            "evidence_lines": evidence_lines,
            "evidence_items": evidence_items,
            "ai_explainability": build_ai_explainability(
                engines,
                evidence_items,
                len(ai_only_evidence),
                fused,
            ),
            "risk_reasons": risk_reasons[:10],
            "repair_suggestions": suggestions[:MAX_REPAIR_SUGGESTIONS],
            "remediation_references": remediation_references,
            "engine_votes": engine_votes,
            "model_version": legacy_ml.get("model_version"),
            "training_samples": legacy_ml.get("training_samples"),
            "external_reputation": next((engine.get("metadata", {}) for engine in engines if engine.get("name") == "hash_reputation"), {}),
            "sandbox": next((engine.get("metadata", {}) for engine in engines if engine.get("name") == "isolated_sandbox"), {}),
            "analysis_capabilities": {
                "strings_ioc": any(engine.get("name") in {"static_evidence", "pe_static"} for engine in engines),
                "static_deobfuscation": any(engine.get("name") == "static_evidence" for engine in engines),
                "javascript_deobfuscation": language == "javascript",
                "behavior_chains": any(engine.get("name") == "static_evidence" for engine in engines),
                "sha256_reputation": any(engine.get("name") == "hash_reputation" and engine.get("status") == "completed" for engine in engines),
                "isolated_sandbox": any(engine.get("name") == "isolated_sandbox" and engine.get("status") == "completed" for engine in engines),
                "pe_static": any(engine.get("name") == "pe_static" for engine in engines),
            },
            "explainability": {
                "score_formula": "risk_score is an AI-first triage score, not a calibrated probability",
                "decision_basis": fused.get("decision_basis"),
                "decision_authority": fused.get("decision_authority"),
                "rule_fallback_used": fused.get(
                    "rule_fallback_used",
                    False,
                ),
                "rule_fallback_reason": fused.get(
                    "rule_fallback_reason",
                ),
                "malicious_rule_weight": severity_by_type["malicious"],
                "vulnerability_rule_weight": severity_by_type["vulnerable"],
                "vulnerability_model_enabled": False,
            },
        }


def _rule_explanations_for_decision(
    matches: list[dict[str, Any]],
    fused: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expose rules as explanations, not contrary malicious verdicts."""

    if fused.get("ai_decision") != "benign":
        return matches
    if fused.get("final_decision") == "vulnerable":
        return [
            match for match in matches
            if match.get("risk_type") == "vulnerable"
        ]
    return []


def _skipped_legacy_baseline() -> dict[str, Any]:
    return {
        "label": "unavailable",
        "probability": None,
        "engine": "legacy_svm",
        "malicious_intent": {
            "label": "unavailable",
            "probability": None,
            "model_probability": None,
            "positive_label": "malicious",
            "available": False,
            "status": "skipped",
            "engine": "legacy_svm",
            "reason": "项目扫描已使用XGBoost主模型，跳过不参与最终判定的旧基线模型",
        },
        "vulnerability_risk": {
            "label": "disabled",
            "probability": None,
            "available": False,
            "status": "disabled",
            "reason": "漏洞风险任务已从当前产品流程下线",
        },
        "model_version": None,
        "training_samples": None,
    }


def _normalize_finding(item: dict[str, Any], content: str, language: str) -> dict[str, Any]:
    finding = dict(item)
    category = str(finding.get("category") or "")
    severity = max(0, min(10, int(finding.get("severity") or 0)))
    finding["description"] = CATEGORY_DESCRIPTION_ZH.get(
        category, str(finding.get("description") or "该行包含与当前风险结论相关的代码特征。"),
    )
    finding["harm"] = CATEGORY_HARM_ZH.get(
        category,
        str(finding.get("description") or "该位置可能引入代码或配置安全风险。"),
    )
    taxonomy = taxonomy_for_category(category)
    finding["risk_domains"] = taxonomy["risk_domains"]
    finding["api_security_category"] = taxonomy["api_security_category"]
    fallback_advice = CATEGORY_REPAIR_ZH.get(
        category,
        str(finding.get("repair_advice") or "结合业务上下文复核该行，并限制不可信输入到达敏感接口。"),
    )
    remediation = remediation_for_finding(finding, language, fallback_advice)
    finding["repair_suggestions"] = remediation["suggestions"]
    finding["repair_advice"] = (
        remediation["suggestions"][0]
        if remediation["suggestions"]
        else fallback_advice
    )
    finding["remediation_references"] = remediation["references"]
    finding["owasp_category"] = remediation["owasp"]
    finding["cwe"] = remediation["cwe"] or finding.get("cwe")
    if finding.get("suspicion_score") is None:
        finding["suspicion_score"] = severity * 10
        finding["suspicion_basis"] = "由规则严重度换算，不是模型概率"
    else:
        finding["suspicion_score"] = round(float(finding["suspicion_score"]), 1)
        finding["suspicion_basis"] = "来自模型行定位分数"
    finding["code_context"] = _code_context(content, finding.get("line"))
    return finding


def _code_context(content: str, line_value: object, radius: int = 2) -> list[dict[str, Any]]:
    try:
        line = int(line_value or 0)
    except (TypeError, ValueError):
        return []
    lines = content.splitlines()
    if line < 1 or line > len(lines):
        return []
    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    return [
        {"line": number, "code": lines[number - 1], "is_target": number == line}
        for number in range(start, end + 1)
    ]


def _combined_risk_score(model_probability: float | None, rule_severity: int) -> int:
    model_score = int(float(model_probability) * 100) if model_probability is not None else 0
    rule_score = min(95, int(rule_severity * 7.5)) if rule_severity else 0
    return max(model_score, rule_score)


def _task_view(engine: dict[str, Any], rule_hits: int) -> dict[str, Any]:
    completed = engine.get("status") == "completed" and engine.get("probability") is not None
    metadata = engine.get("metadata") or {}
    advisory_only = bool(metadata.get("advisory_only"))
    return {
        "available": completed,
        "status": "advisory" if completed and advisory_only else (
            engine.get("decision") if completed else engine.get("status", "unavailable")
        ),
        "probability": engine.get("probability") if completed else None,
        "threshold": engine.get("threshold") if completed else None,
        "model_version": engine.get("model_version"),
        "reason": engine.get("reason") or engine.get("error"),
        "engine": engine.get("name"),
        "rule_hits": rule_hits,
        "advisory_only": advisory_only,
        "advisory_reason": metadata.get("advisory_reason"),
        "raw_model_decision": metadata.get("raw_model_decision"),
        "evaluation_scope": metadata.get("evaluation_scope"),
        "route_quality_gate_passed": metadata.get("route_quality_gate_passed"),
        "source_heldout_verified": metadata.get("source_heldout_verified"),
    }


def _disabled_task_view() -> dict[str, Any]:
    return {
        "available": False,
        "status": "disabled",
        "probability": None,
        "threshold": None,
        "model_version": None,
        "reason": "漏洞风险任务已从当前产品流程下线",
        "engine": "vulnerability_risk",
        "rule_hits": 0,
        "advisory_only": False,
        "advisory_reason": None,
        "raw_model_decision": None,
        "evaluation_scope": None,
        "route_quality_gate_passed": None,
        "source_heldout_verified": None,
    }


def _preferred_task_engine(semantic_engine: dict[str, Any], fallback: dict[str, Any], task: str) -> dict[str, Any]:
    if semantic_engine.get("status") != "completed":
        return fallback
    metadata = semantic_engine.get("metadata", {})
    probability = metadata.get("task_probabilities", {}).get(task)
    threshold = metadata.get("task_thresholds", {}).get(task)
    if probability is None or threshold is None:
        return fallback
    return {
        "name": semantic_engine.get("name"),
        "status": "completed",
        "decision": semantic_engine.get("decision"),
        "probability": probability,
        "threshold": threshold,
        "model_version": (metadata.get("task_versions") or {}).get(task) or semantic_engine.get("model_version"),
    }
