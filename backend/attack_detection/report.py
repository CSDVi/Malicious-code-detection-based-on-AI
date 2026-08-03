"""Markdown report rendering for saved scan records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .report_insights import build_evidence_groups, build_file_report_insights

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
    "AI Semantic Risk": "AI语义异常（待细分类）",
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
        f"- 行为类别：{categories}",
        f"- ATT&CK：{attacks}",
        f"- MD5：`{(record.get('hashes') or {}).get('md5') or '未记录'}`",
        f"- SHA-1：`{(record.get('hashes') or {}).get('sha1') or '未记录'}`",
        f"- SHA-256：`{(record.get('hashes') or {}).get('sha256') or record['file_hash']}`",
    ]
    if _reputation_visible(record):
        lines.append(f"- SHA256 外部信誉：{_reputation_summary(record)}")
    lines.extend([
        f"- 隔离沙箱：{_sandbox_summary(record)}",
        f"- 检测时间：{record['created_at']}",
        f"- 报告生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 检测结论",
        "",
        _conclusion(record),
        "",
        "## AI模型判定明细",
        "",
    ])
    ledger = build_file_report_insights(record).get("decision_ledger") or []
    if ledger:
        lines.append("| 组件 | 结论 | 恶意概率 | 阈值 | 距离阈值 |")
        lines.append("|---|---|---:|---:|---:|")
        for item in ledger:
            margin = item.get("margin")
            margin_text = (
                f"{margin:+.1f} 个百分点"
                if isinstance(margin, (int, float))
                else "不适用"
            )
            probability = item.get("probability_percent")
            threshold = item.get("threshold_percent")
            lines.append(
                f"| {_zh(item.get('name'))} | {_zh(item.get('decision'))} | "
                f"{f'{probability:.1f}%' if isinstance(probability, (int, float)) else '不适用'} | "
                f"{f'{threshold:.1f}%' if isinstance(threshold, (int, float)) else '不适用'} | {margin_text} |"
            )
    else:
        lines.append("本次记录没有保存可用于展示的AI模型输出。")
    lines.extend([
        "",
        "## 恶意点定位与解释",
        "",
    ])
    matches = [
        item for item in (
            record.get("evidence_items")
            or record.get("rule_matches")
            or []
        )
        if isinstance(item, dict)
    ]
    evidence_groups = build_evidence_groups(matches)
    if evidence_groups:
        for group in evidence_groups:
            group_meta = [
                _zh(group.get("category") or "未分类风险"),
                str(group.get("cwe") or ""),
                f"{group.get('count', 0)} 处",
            ]
            lines.extend(["", f"### {' · '.join(item for item in group_meta if item)}", ""])
            lines.append("| 位置 | 可疑度 | 代码证据 |")
            lines.append("|---:|---:|---|")
            for match in group.get("items") or []:
                snippet = str(match.get("snippet", "")).replace("|", "\\|")
                suspicion = match.get("suspicion_score")
                suspicion_text = f"{suspicion}/100" if suspicion is not None else "未评分"
                lines.append(
                    f"| {match.get('line') or '文件级'} | {suspicion_text} | `{snippet}` |"
                )

            lines.extend(["", "#### 典型例子", ""])
            examples = group.get("cve_examples") or []
            if examples:
                for example in examples:
                    lines.append(
                        f"- [{example.get('id')}]({example.get('url')}): "
                        f"{example.get('title')}。{example.get('summary')}"
                    )
            else:
                lines.append("暂无可核实的典型例子。")

            lines.extend(["", "#### 危害", ""])
            harms = group.get("harms") or []
            lines.extend(f"- {harm}" for harm in harms)
            if not harms:
                lines.append("- 当前没有可可靠归纳的危害说明。")

            lines.extend(["", "#### 修复建议", ""])
            suggestions = group.get("repair_suggestions") or []
            lines.extend(f"- {suggestion}" for suggestion in suggestions)
            if not suggestions:
                lines.append("- 结合命中位置及上下文进行人工复核，并按最小权限原则限制敏感操作。")
            references = group.get("remediation_references") or []
            if references:
                lines.extend(["", "##### 建议依据", ""])
                lines.extend(
                    f"- [{item.get('title')}]({item.get('url')})"
                    for item in references
                )
    else:
        lines.append("本次没有生成可靠的行级恶意点定位。")

    traces = [item for item in matches if item.get("trace_steps")]
    if traces:
        lines.extend(["", "## 代码关联路径", ""])
        lines.append("以下路径由静态分析还原输入到敏感操作的近似关联；AI高贡献位置会标记为 AI关注，不代表完整运行时数据流证明。")
        for match in traces:
            lines.extend(["", f"### {_zh(match.get('category') or '风险证据')} · {match.get('rule_id') or '未编号'}", ""])
            for index, step in enumerate(match.get("trace_steps") or [], start=1):
                if not isinstance(step, dict):
                    continue
                snippet = str(step.get("snippet") or "").replace("`", "\\`")
                position = f"第 {step.get('line')} 行" if step.get("line") else "文件级"
                ai_marker = (
                    f"，AI关注，贡献 {(step.get('ai_attribution') or {}).get('contribution_percent')}%"
                    if step.get("ai_supported")
                    else ""
                )
                lines.append(
                    f"{index}. {step.get('stage') or step.get('label') or step.get('kind') or '关联步骤'}"
                    f"（{position}{ai_marker}）：`{snippet}`"
                )

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

    return "\n".join(lines)


def _reputation_visible(record: dict[str, Any]) -> bool:
    for engine in record.get("engines") or []:
        if engine.get("name") != "hash_reputation":
            continue
        metadata = engine.get("metadata") or {}
        reason = str(engine.get("reason") or engine.get("error") or "").lower()
        return (
            engine.get("status") in {"completed", "failed"}
            or bool(metadata.get("provider"))
            or "hash not found" in reason
            or "unsupported reputation provider" in reason
        )
    return False


def _reputation_summary(record: dict[str, Any]) -> str:
    for engine in record.get("engines") or []:
        if engine.get("name") != "hash_reputation":
            continue
        if engine.get("status") == "completed":
            metadata = engine.get("metadata") or {}
            return (
                f"{metadata.get('provider', '外部服务')}，恶意 {metadata.get('malicious', 0)}，"
                f"可疑 {metadata.get('suspicious', 0)}；仅作外部复核线索，不参与AI结论或风险分"
            )
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
    authority = str(record.get("decision_authority") or "")
    score = int(record.get("risk_score", 0))
    if decision == "malicious":
        if authority != "ai":
            return (
                "这是旧策略保存的非AI恶意结论，不能按当前策略视为AI已确认；"
                "建议重新检测并人工复核。"
            )
        return (
            "经过验证的AI模型将该文件判定为恶意；规则与静态分析仅用于"
            "解释行为、危害、类别和修复方式，应立即隔离并人工复核。"
        )
    if decision == "unknown":
        return (
            "AI模型不可用、不确定或相互冲突，因此未形成恶意或良性结论；"
            "规则不会代替AI判定，该结果需要人工复核。"
        )
    if score > 0:
        return "当前发现较弱的风险信号，发布前应结合证据进行人工复核。"
    return "经过验证的AI模型未将该文件判定为恶意。"


def _decision_authority(record: dict[str, Any]) -> str:
    authority = str(record.get("decision_authority") or "")
    if authority == "ai":
        return "AI主判"
    if authority == "rule_fallback":
        return "历史规则回退结果（建议按当前AI策略重新检测）"
    if authority == "external_context":
        return "外部情报复核"
    return "未决"
