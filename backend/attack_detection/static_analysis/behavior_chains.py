"""Combine nearby static indicators into explainable behavior chains."""

from __future__ import annotations

import re

from attack_detection.source_masking import mask_non_executable_text


_GROUP_PATTERNS = {
    "credential": (re.compile(r"(?:password|passwd|secret|token|api[_-]?key|credential|\.ssh|login)", re.IGNORECASE),),
    "network": (re.compile(r"(?:https?://|fetch\s*\(|requests?\.(?:get|post)|curl|wget|socket|WebSocket|XMLHttpRequest)", re.IGNORECASE),),
    "send": (re.compile(r"(?:send\s*\(|post\s*\(|upload|exfil|webhook|socket\.send)", re.IGNORECASE),),
    "decode": (re.compile(r"(?:atob|fromCharCode|base64|base64decode|hex2bin|decode)", re.IGNORECASE),),
    "execute": (re.compile(r"(?:eval\s*\(|exec\s*\(|system\s*\(|shell_exec\s*\(|passthru\s*\(|proc_open\s*\(|popen\s*\(|subprocess|child_process|spawn\s*\(|Runtime\.getRuntime|exec\.Command\s*\(|Start-Process\b|Invoke-Expression\b|\bIEX\s*(?:\(|\s))", re.IGNORECASE),),
    # Do not treat prose containing "download" or an inert URL as a download
    # operation.  A chain needs an API/command-shaped indicator; plain URLs are
    # still retained by the separate network context group.
    "download": (re.compile(r"(?:\b(?:curl|wget)\b(?:\s+|$)|\b(?:urlopen|fetch|curl_init|http\.Get|download)\s*\(|\brequests?\s*\.\s*get\s*\(|\b(?:DownloadString|DownloadFile|Invoke-WebRequest|Invoke-RestMethod)\b\s*(?:\(|\s)|file_get_contents\s*\(\s*[\"']https?://)", re.IGNORECASE),),
    "persistence": (re.compile(r"(?:crontab|schtasks|startup|registry|RunOnce|service\s+install|systemd|authorized_keys)", re.IGNORECASE),),
    "hook": (re.compile(r"(?:postinstall|preinstall|setup\.py|install\s*hook|npm\s+install)", re.IGNORECASE),),
    "write": (re.compile(r"(?:open\s*\(|write(?:File)?\s*\(|FileOutputStream|fs\.write)", re.IGNORECASE),),
}

_MAX_CHAIN_LINE_GAP = 40

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


def _all_hits(
    text: str,
    rows: list[str],
    patterns: tuple[re.Pattern[str], ...],
) -> list[tuple[int, str]]:
    # A whole-source rejection avoids testing every line for groups that are
    # absent.  Positive groups retain every line so the closest ordered pair
    # can be selected instead of combining unrelated first hits.
    if not any(pattern.search(text) for pattern in patterns):
        return []
    return [
        (line, value.strip()[:240])
        for line, value in enumerate(rows, 1)
        if any(pattern.search(value) for pattern in patterns)
    ]


def _nearest_ordered_pair(
    left_hits: list[tuple[int, str]],
    right_hits: list[tuple[int, str]],
) -> tuple[tuple[int, str], tuple[int, str]] | None:
    """Return the closest left-to-right pair within the local chain window."""

    right_index = 0
    best: tuple[tuple[int, str], tuple[int, str]] | None = None
    best_gap = _MAX_CHAIN_LINE_GAP + 1
    for left_hit in left_hits:
        while (
            right_index < len(right_hits)
            and right_hits[right_index][0] < left_hit[0]
        ):
            right_index += 1
        if right_index >= len(right_hits):
            break
        right_hit = right_hits[right_index]
        gap = right_hit[0] - left_hit[0]
        if gap < best_gap:
            best = (left_hit, right_hit)
            best_gap = gap
            if gap == 0:
                break
    return best if best_gap <= _MAX_CHAIN_LINE_GAP else None


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
        name: _all_hits(executable_text, rows, patterns)
        for name, patterns in _GROUP_PATTERNS.items()
    }
    hits = {
        name: [
            (line, original_rows[line - 1].strip()[:240])
            for line, _snippet in group_hits
            if 1 <= line <= len(original_rows)
        ]
        for name, group_hits in hits.items()
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
        pair = _nearest_ordered_pair(hits[left], hits[right])
        if pair is None:
            continue
        (first_line, first_snippet), (second_line, second_snippet) = pair
        line, snippet = second_line, second_snippet
        line_gap = second_line - first_line
        findings.append({
            "source": "behavior_chain",
            "rule_id": rule_id,
            "category": category,
            "risk_type": risk_type,
            "behavior": " -> ".join((left, right)),
            "severity": severity,
            "line": line,
            "snippet": snippet,
            "evidence": f"{description}；两个指标按行为顺序在 {line_gap} 行范围内静态命中。",
            "description": f"行为链证据：{description}。邻近命中尚不能证明真实数据流，需要结合变量传递、权限和业务用途复核。",
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
    findings.extend(_ransomware_chain_findings(rows, original_rows))
    return findings


def _ransomware_chain_findings(rows: list[str], original_rows: list[str]) -> list[dict[str, object]]:
    """Recognize the combined traversal/encryption/deletion/recovery pattern."""

    stages = (
        ("traversal", "递归枚举文件", re.compile(r"\b(?:os\.walk|glob\.glob|fnmatch\.fnmatch)\s*\(", re.I), "first"),
        ("encryption", "加密并写回文件", re.compile(r"\b(?:encryptor\.encrypt|Fernet\s*\(|Cipher\s*\()", re.I), "last"),
        ("deletion", "删除原文件", re.compile(r"(?<!def\s)\b(?:os\.remove|os\.unlink|delete_file)\s*\(", re.I), "last"),
        ("recovery", "破坏系统恢复能力", re.compile(r"vssadmin\s+delete\s+shadows", re.I), "first"),
    )
    trace_steps = []
    for kind, label, pattern, preference in stages:
        matches = [(number, original_rows[number - 1].strip()[:240]) for number, row in enumerate(rows, 1) if pattern.search(row)]
        hit = (matches[-1] if preference == "last" else matches[0]) if matches else None
        if hit is None:
            return []
        line, snippet = hit
        trace_steps.append({"kind": kind, "stage": label, "line": line, "snippet": snippet})
    detail_specs = (
        ("traversal", "递归进入目录", re.compile(r"\bos\.walk\s*\(", re.I)),
        ("traversal", "筛选目标扩展名", re.compile(r"\bfnmatch\.fnmatch\s*\(", re.I)),
        ("encryption", "初始化 AES 加密器", re.compile(r"\bAES\.new\s*\(", re.I)),
        ("encryption", "读取原文件数据块", re.compile(r"\binfile\.read\s*\(", re.I)),
        ("encryption", "加密并写入数据块", re.compile(r"\bencryptor\.encrypt\s*\(", re.I)),
        ("deletion", "调用原文件删除逻辑", re.compile(r"(?<!def\s)\bdelete_file\s*\(", re.I)),
        ("recovery", "删除系统卷影副本", re.compile(r"vssadmin\s+delete\s+shadows", re.I)),
        ("orchestration", "启动持久化步骤", re.compile(r"^\s*persistence\s*\(", re.I)),
        ("orchestration", "启动恢复破坏步骤", re.compile(r"^\s*destroy_shadow_copy\s*\(", re.I)),
        ("orchestration", "创建远程控制入口", re.compile(r"^\s*create_remote_desktop\s*\(", re.I)),
    )
    detailed_steps = []
    seen_details = set()
    for kind, label, pattern in detail_specs:
        for number, row in enumerate(rows, 1):
            if not pattern.search(row) or (number, kind) in seen_details:
                continue
            seen_details.add((number, kind))
            detailed_steps.append({
                "kind": kind,
                "stage": label,
                "line": number,
                "snippet": original_rows[number - 1].strip()[:240],
            })
    if detailed_steps:
        trace_steps = sorted(detailed_steps, key=lambda item: int(item["line"]))
    sink = trace_steps[-1]
    chain = {
        "source": "behavior_chain",
        "rule_id": "CHAIN-RANSOMWARE",
        "category": "Ransomware Behavior Chain",
        "risk_type": "malicious",
        "behavior": "traversal -> encryption -> deletion -> recovery destruction",
        "severity": 10,
        "line": sink["line"],
        "snippet": sink["snippet"],
        "evidence": "同一文件内同时存在递归遍历、内容加密、原件删除与恢复破坏四类高置信行为。",
        "description": "多阶段行为组合符合勒索型文件处理链，而不是由单个 os.system 或单个网址触发。",
        "repair_advice": "隔离样本并检查加密写回、删除原件和恢复破坏的调用路径。",
        "confidence": 0.96,
        "trace_steps": trace_steps,
    }
    stage_descriptions = {
        "traversal": "代码递归枚举目录并筛选目标扩展名，为批量处理文件建立目标集合。",
        "encryption": "代码使用加密器处理读取的数据块并写入输出文件。",
        "deletion": "加密处理链随后调用删除逻辑移除原始文件。",
        "recovery": "代码调用系统命令删除卷影副本，削弱受影响文件的恢复能力。",
        "orchestration": "主流程连续启动持久化、恢复破坏或远程控制相关步骤。",
    }
    stage_findings = []
    for step in trace_steps:
        stage_findings.append({
            "source": "behavior_chain",
            "rule_id": f"CHAIN-RANSOMWARE-{str(step['kind']).upper()}",
            "category": "Ransomware Behavior Chain",
            "risk_type": "malicious",
            "behavior": str(step["kind"]),
            "severity": 10,
            "line": step["line"],
            "snippet": step["snippet"],
            "evidence": stage_descriptions[str(step["kind"])],
            "description": stage_descriptions[str(step["kind"])],
            "repair_advice": "隔离样本，并结合完整调用链核查该阶段与其他勒索行为的关联。",
            "confidence": 0.94,
        })
    return [chain, *stage_findings]
