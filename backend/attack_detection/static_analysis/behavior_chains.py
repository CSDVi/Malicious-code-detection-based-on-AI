"""Combine nearby static indicators into explainable behavior chains."""

from __future__ import annotations

import re

from attack_detection.source_masking import mask_non_executable_text


_GROUP_PATTERNS = {
    "credential": (re.compile(r"(?:password|passwd|secret|token|api[_-]?key|credential|\.ssh|login)", re.IGNORECASE),),
    "network": (re.compile(r"(?:https?://|fetch\s*\(|requests?\.(?:get|post)|curl|wget|socket|WebSocket|XMLHttpRequest)", re.IGNORECASE),),
    "send": (re.compile(r"(?:send\s*\(|post\s*\(|upload|exfil|webhook|socket\.send)", re.IGNORECASE),),
    "decode": (re.compile(r"(?:atob|fromCharCode|base64|base64decode|hex2bin|decode)", re.IGNORECASE),),
    "execute": (re.compile(r"(?:eval\s*\(|exec\s*\(|system\s*\(|popen|subprocess|child_process|spawn\s*\(|Runtime\.getRuntime)", re.IGNORECASE),),
    "download": (re.compile(r"(?:download|urlopen|requests?\.get|curl|wget|http[s]?://)", re.IGNORECASE),),
    "persistence": (re.compile(r"(?:crontab|schtasks|startup|registry|RunOnce|service\s+install|systemd|authorized_keys)", re.IGNORECASE),),
    "hook": (re.compile(r"(?:postinstall|preinstall|setup\.py|install\s*hook|npm\s+install)", re.IGNORECASE),),
    "write": (re.compile(r"(?:open\s*\(|write(?:File)?\s*\(|FileOutputStream|fs\.write)", re.IGNORECASE),),
}

_GROUP_LABELS = {
    "credential": "读取凭据或密钥",
    "network": "访问网络目标",
    "send": "发送或上传数据",
    "decode": "解码或还原载荷",
    "execute": "进入执行接口",
    "download": "下载远程内容",
    "persistence": "写入持久化位置",
    "hook": "进入安装钩子",
    "write": "写入本地文件",
}


def _first_hit(
    text: str,
    rows: list[str],
    patterns: tuple[re.Pattern[str], ...],
) -> tuple[int, str] | None:
    # A whole-source rejection avoids testing every line for groups that are
    # absent.  When a group is present, only its first line is needed by every
    # behavior-chain result.
    if not any(pattern.search(text) for pattern in patterns):
        return None
    for line, value in enumerate(rows, 1):
        if any(pattern.search(value) for pattern in patterns):
            return line, value.strip()[:240]
    return None


def detect_behavior_chains(
    text: str,
    language: str = "unknown",
    *,
    comments_masked: bool = False,
) -> list[dict[str, object]]:
    executable_text = text if comments_masked else mask_non_executable_text(text, language)
    rows = executable_text.splitlines() or [executable_text]
    original_rows = text.splitlines() or [text]
    hits = {
        name: _first_hit(executable_text, rows, patterns)
        for name, patterns in _GROUP_PATTERNS.items()
    }
    hits = {
        name: (
            (hit[0], original_rows[hit[0] - 1].strip()[:240])
            if hit is not None and 1 <= hit[0] <= len(original_rows)
            else hit
        )
        for name, hit in hits.items()
    }
    chains = (
        ("credential", "send", "Credential Exfiltration", "CHAIN-CRED-EXFIL", 9, "malicious", "读取凭据/密钥后发送到外部目标"),
        ("decode", "execute", "Obfuscated Payload", "CHAIN-DECODE-EXEC", 8, "malicious", "解码操作与动态执行出现在同一文件中"),
        ("download", "execute", "Download and Execute", "CHAIN-DOWNLOAD-EXEC", 9, "malicious", "远程下载与执行行为形成组合链"),
        ("hook", "network", "Install Hook Execution", "CHAIN-HOOK-NET", 8, "malicious", "安装钩子与网络行为形成组合链"),
        ("persistence", "execute", "Persistence", "CHAIN-PERSIST-EXEC", 8, "malicious", "持久化位置与执行行为形成组合链"),
        ("network", "write", "Remote Load or Write", "CHAIN-NET-WRITE", 5, "context", "网络读取与本地写入出现在同一文件中"),
    )
    findings = []
    for left, right, category, rule_id, severity, risk_type, description in chains:
        if hits[left] is None or hits[right] is None:
            continue
        first_line, first_snippet = hits[left]
        second_line, second_snippet = hits[right]
        line, snippet = (second_line, second_snippet) if second_line >= first_line else (first_line, first_snippet)
        findings.append({
            "source": "behavior_chain",
            "rule_id": rule_id,
            "category": category,
            "risk_type": risk_type,
            "behavior": " -> ".join((left, right)),
            "severity": severity,
            "line": line,
            "snippet": snippet,
            "evidence": f"{description}；两类指标均有静态命中。",
            "description": f"行为链证据：{description}。该组合需要结合数据流、权限和业务用途复核。",
            "repair_advice": "拆分并审查数据读取、网络访问、解码、写入和执行边界，采用最小权限与显式白名单。",
            "confidence": 0.78 if risk_type == "malicious" else 0.55,
            "trace_steps": [
                {
                    "kind": "indicator",
                    "stage": _GROUP_LABELS.get(left, left),
                    "line": first_line,
                    "snippet": first_snippet,
                },
                {
                    "kind": "indicator",
                    "stage": _GROUP_LABELS.get(right, right),
                    "line": second_line,
                    "snippet": second_snippet,
                },
            ],
        })
    return findings
