"""Conservative JavaScript static deobfuscation.

This module only transforms literal text. It never evaluates JavaScript and never
loads a Node/browser runtime.
"""

from __future__ import annotations

import base64
import binascii
import re

from .strings_ioc import decode_literal_candidates


_CHAR_CODE_RE = re.compile(r"String\.fromCharCode\(([^)]{1,2000})\)", re.IGNORECASE)
_ATOB_RE = re.compile(r"(?:atob|Buffer\.from)\(\s*(['\"])([A-Za-z0-9+/=]{8,})\1(?:\s*,\s*['\"]base64['\"])?\s*\)", re.IGNORECASE)
_DANGEROUS = (
    (re.compile(r"\beval\s*\(|new\s+Function\s*\(|Function\s*\(", re.IGNORECASE), "JS-DEOB-DYNAMIC", "Dynamic JavaScript execution", 8, "malicious"),
    (re.compile(r"(?:fetch|XMLHttpRequest|WebSocket|http\.request|https\.request)\s*\(", re.IGNORECASE), "JS-DEOB-NETWORK", "Decoded network behavior", 6, "malicious"),
    (re.compile(r"(?:child_process|spawn|execFile|require\s*\(\s*['\"]child_process)", re.IGNORECASE), "JS-DEOB-CMD", "Decoded process execution", 9, "malicious"),
)


def _decode_char_codes(value: str) -> str | None:
    parts = [part.strip() for part in value.split(",")]
    if not parts or len(parts) > 512:
        return None
    numbers: list[int] = []
    for part in parts:
        if not re.fullmatch(r"(?:0x[0-9a-f]+|\d{1,6})", part, re.IGNORECASE):
            return None
        number = int(part, 0)
        if number > 0x10FFFF:
            return None
        numbers.append(number)
    try:
        return "".join(chr(number) for number in numbers)
    except ValueError:
        return None


def deobfuscate_javascript(text: str) -> dict[str, object]:
    decoded: list[dict[str, object]] = []
    transformed = text
    for match in _CHAR_CODE_RE.finditer(text):
        value = _decode_char_codes(match.group(1))
        if value:
            decoded.append({"encoding": "fromCharCode", "source": match.group(0), "decoded": value, "line": text.count("\n", 0, match.start()) + 1})
            transformed = transformed.replace(match.group(0), repr(value), 1)
    for match in _ATOB_RE.finditer(text):
        try:
            value = base64.b64decode(match.group(2), validate=True).decode("utf-8")
        except (UnicodeDecodeError, ValueError, binascii.Error):
            continue
        if len(value) > 64 * 1024:
            continue
        decoded.append({"encoding": "base64", "source": match.group(0), "decoded": value, "line": text.count("\n", 0, match.start()) + 1})
        transformed = transformed.replace(match.group(0), repr(value), 1)
    for item in decode_literal_candidates(text):
        decoded.append(item)
    findings: list[dict[str, object]] = []
    for item in decoded[:120]:
        value = str(item.get("decoded") or "")
        for pattern, rule_id, label, severity, risk_type in _DANGEROUS:
            if pattern.search(value):
                line = int(item.get("line") or 1)
                findings.append({
                    "source": "js_deobfuscation",
                    "rule_id": rule_id,
                    "category": "JavaScript 静态去混淆",
                    "risk_type": risk_type,
                    "behavior": label,
                    "severity": severity,
                    "line": line,
                    "snippet": str(item.get("source") or "")[:240],
                    "evidence": f"静态解码结果: {value[:240]}",
                    "description": f"JavaScript 字面量静态解码后出现{label}特征；没有执行上传脚本。",
                    "repair_advice": "审查解码前后的完整调用链，移除动态执行和不必要的混淆，并限制网络/进程权限。",
                    "confidence": 0.82,
                })
                break
    return {"transformed": transformed[:256 * 1024], "decoded": decoded[:120], "findings": findings}
