"""Markdown report rendering for saved scan records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

DISPLAY_ZH = {
    "benign": "正常",
    "malicious": "恶意",
    "safe": "安全",
    "critical": "严重",
    "high": "高危",
    "medium": "中危",
    "low": "低危",
    "unknown": "需复核",
    "auto": "自动",
    "quick": "快速",
    "standard": "标准",
    "deep": "深度",
    "completed": "已完成",
    "unavailable": "不可用",
    "failed": "失败",
    "skipped": "已跳过",
    "not_applicable": "不适用于该文件",
    "rule_engine": "规则引擎",
    "xgboost_malicious": "XGBoost 恶意代码模型",
    "xgboost_project_malicious": "XGBoost 项目恶意代码模型",
    "codet5p": "CodeT5+ 220M 语义模型",
    "gatv2": "GATv2 项目图模型",
    "static_evidence": "静态证据分析",
    "pe_static": "PE/DLL 只读解析",
    "hash_reputation": "SHA256 外部信誉",
    "isolated_sandbox": "隔离动态沙箱",
    "python": "Python",
    "javascript": "JavaScript/TypeScript",
    "java": "Java",
    "php": "PHP",
    "bash": "Bash/Shell",
    "powershell": "PowerShell",
    "batch": "批处理/CMD",
    "config": "config",
    "json": "JSON",
    "yaml": "YAML",
    "toml": "TOML",
    "ini": "INI",
    "conf": "CONF",
    "text": "TXT",
    "binary": "Windows 可执行文件",
    "SQL Injection": "SQL 注入",
    "Command Execution": "命令执行",
    "Path Traversal": "路径穿越",
    "Unsafe Deserialization": "不安全反序列化",
    "Secret Exposure": "敏感信息泄露",
    "WebShell": "网页后门",
    "Download or Remote Load": "下载或远程加载",
    "Download and Execute": "下载并执行",
    "Obfuscated Payload": "混淆载荷",
    "Install Hook Execution": "安装钩子执行",
    "Credential Exfiltration": "凭据外传",
    "Credential Collection": "凭据收集",
    "Persistence": "持久化",
    "AI Semantic Risk": "AI语义风险",
    "ai_signal": "AI关注信号",
    "Command and Scripting Interpreter": "命令与脚本解释器",
    "Ingress Tool Transfer": "工具传入",
    "Obfuscated Files or Information": "混淆文件或信息",
    "Credentials from Password Stores": "从密码存储中获取凭据",
    "Exfiltration Over C2 Channel": "通过命令与控制通道外传",
    "Event Triggered Execution": "事件触发执行",
    "external hash reputation is disabled; set XIEZHI_REPUTATION_PROVIDER and an API key": "未配置外部哈希信誉服务",
    "XIEZHI_VT_API_KEY is not configured": "未配置外部信誉服务密钥",
    "sandbox backend not configured": "未配置隔离动态沙箱",
    "sandbox submission requires XIEZHI_SANDBOX_AUTO_SCAN=1": "已配置沙箱服务，但未开启自动提交",
}


def _zh(value: Any) -> str:
    text = str(value)
    return DISPLAY_ZH.get(text, text)


def _probability(value: Any) -> str:
    return "未产生" if value is None else str(value)


def render_record_markdown(record: dict[str, Any]) -> str:
    categories = "、".join(_zh(item) for item in record.get("categories", [])) or "暂无"
    attacks = "、".join(
        f"{item.get('id')} {item.get('name')}" for item in record.get("attack_techniques", [])
    ) or "暂无"
    lines = [
        "# 獬豸安码检测报告",
        "",
        f"- 文件：`{record['filename']}`",
        f"- 代码语言：{_zh(record.get('display_language') or record['language'])}",
        f"- 最终判断：{_zh(record.get('final_decision', 'unknown'))}",
        f"- 判定来源：{_decision_authority(record)}",
        f"- 风险等级：{_zh(record['risk_level'])} / 风险分 {record['risk_score']}",
        f"- 选择模式 / 实际模式：{_zh(record.get('selected_mode') or '未记录')} / {_zh(record.get('effective_mode') or '未记录')}",
        f"- 恶意代码概率：{_probability(record.get('malicious_probability'))}",
        f"- 行为类别：{categories}",
        f"- ATT&CK：{attacks}",
        f"- MD5：`{(record.get('hashes') or {}).get('md5') or '未记录'}`",
        f"- SHA-1：`{(record.get('hashes') or {}).get('sha1') or '未记录'}`",
        f"- SHA-256：`{(record.get('hashes') or {}).get('sha256') or record['file_hash']}`",
        f"- SHA256 外部信誉：{_reputation_summary(record)}",
        f"- 隔离沙箱：{_sandbox_summary(record)}",
        f"- 检测时间：{record['created_at']}",
        f"- 报告生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 检测结论",
        "",
        _conclusion(record),
        "",
        "## 证据代码",
        "",
    ]
    matches = [
        item for item in (
            record.get("evidence_items")
            or record.get("rule_matches")
            or []
        )
        if isinstance(item, dict)
    ]
    if matches:
        lines.append("| 行号 | 检测依据 | 风险类别 | OWASP | API安全 | CWE | 危害 | 代码证据 |")
        lines.append("|---:|---|---|---|---|---|---|---|")
        for match in matches:
            snippet = str(match.get("snippet", "")).replace("|", "\\|")
            harm = str(
                match.get("harm")
                or match.get("description")
                or ""
            ).replace("|", "\\|")
            lines.append(
                f"| {match.get('line', '')} | {_evidence_basis(match)} | "
                f"{_zh(match.get('category', ''))} | {match.get('owasp_category') or '-'} | "
                f"{match.get('api_security_category') or '-'} | {match.get('cwe') or '-'} | "
                f"{harm} | `{snippet}` |"
            )
    else:
        lines.append("未保存可定位代码证据。")

    lines.extend(["", "## 引擎运行详情", ""])
    engines = record.get("engines") or []
    if engines:
        lines.append("| 引擎 | 状态 | 概率 | 阈值 | 版本 | 耗时 |")
        lines.append("|---|---|---:|---:|---|---:|")
        for engine in engines:
            lines.append(
                f"| {_zh(engine.get('name'))} | {_zh(engine.get('status'))} | {_probability(engine.get('probability'))} | "
                f"{_probability(engine.get('threshold'))} | {engine.get('model_version') or '-'} | "
                f"{engine.get('duration_ms') if engine.get('duration_ms') is not None else '-'} 毫秒 |"
            )
    else:
        lines.append("该历史记录没有保存引擎运行详情。")

    lines.extend(["", "## 修复建议", ""])
    suggestions = []
    seen_suggestions = set()
    remediation_references = []
    seen_reference_urls = set()
    for match in matches:
        advice_values = match.get("repair_suggestions") or [match.get("repair_advice")]
        if isinstance(advice_values, str):
            advice_values = [advice_values]
        for advice_value in advice_values:
            advice = " ".join(str(advice_value or "").split())
            if advice and advice not in seen_suggestions:
                suggestions.append(advice)
                seen_suggestions.add(advice)
        for reference in match.get("remediation_references") or []:
            if not isinstance(reference, dict):
                continue
            url = str(reference.get("url") or "")
            if not url or url in seen_reference_urls:
                continue
            remediation_references.append({
                "title": str(reference.get("title") or "修复依据"),
                "url": url,
            })
            seen_reference_urls.add(url)
    if not suggestions:
        suggestions.append("保持依赖版本锁定、代码审查和最小权限运行配置。")
    lines.extend(f"- {item}" for item in suggestions)
    if remediation_references:
        lines.extend(["", "### 建议依据", ""])
        lines.extend(f"- [{item['title']}]({item['url']})" for item in remediation_references)
    return "\n".join(lines)


def _reputation_summary(record: dict[str, Any]) -> str:
    for engine in record.get("engines") or []:
        if engine.get("name") != "hash_reputation":
            continue
        if engine.get("status") == "completed":
            metadata = engine.get("metadata") or {}
            return f"{metadata.get('provider', '外部服务')}，恶意 {metadata.get('malicious', 0)}，可疑 {metadata.get('suspicious', 0)}"
        return _zh(engine.get("reason") or engine.get("status") or "未查询")
    return "未记录"


def _sandbox_summary(record: dict[str, Any]) -> str:
    for engine in record.get("engines") or []:
        if engine.get("name") != "isolated_sandbox":
            continue
        if engine.get("status") == "completed":
            metadata = engine.get("metadata") or {}
            return f"已提交隔离服务，样本 {metadata.get('sample_id') or '未返回 ID'}"
        return _zh(engine.get("reason") or engine.get("status") or "未提交")
    return "未记录"


def _conclusion(record: dict[str, Any]) -> str:
    decision = str(record.get("final_decision", "unknown"))
    score = int(record.get("risk_score", 0))
    if decision == "malicious":
        if record.get("decision_authority") == "ai":
            return (
                "经过验证的AI模型将该文件判定为恶意；XGBoost行级归因"
                "用于定位高贡献代码，应立即隔离并人工复核。"
            )
        return (
            "AI本次未形成可靠结论，规则按回退策略将该文件判定为恶意；"
            "应立即隔离并人工复核。"
        )
    if decision == "vulnerable":
        return "该文件存在可定位的漏洞证据，应根据危害说明和修复建议完成整改。"
    if decision == "unknown":
        return (
            "AI模型不可用、不确定或相互冲突，且规则没有形成可靠回退结论；"
            "该结果需要人工复核。"
        )
    if score > 0:
        return "当前发现较弱的风险信号，发布前应结合证据进行人工复核。"
    return "经过验证的AI模型未将该文件判定为恶意。"


def _evidence_basis(item: dict[str, Any]) -> str:
    basis = str(item.get("evidence_basis") or "rule_only")
    if basis == "ai_decision":
        return "AI主判"
    if basis == "ai_and_rule":
        return "AI与规则一致"
    if basis == "ai_only":
        return "AI行级归因"
    return "规则解释"


def _decision_authority(record: dict[str, Any]) -> str:
    authority = str(record.get("decision_authority") or "")
    if authority in {"ai", "ai_with_rule_vulnerability"}:
        return "AI主判"
    if authority == "rule_fallback":
        return "规则回退（仅因AI无法可靠判定）"
    if authority == "external_context":
        return "外部情报复核"
    return "未决"
