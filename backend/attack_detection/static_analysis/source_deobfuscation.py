"""Conservative, cross-language static deobfuscation.

Only literal values are decoded.  Uploaded code is never evaluated, imported,
compiled or passed to a language runtime.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any

from .js_deobfuscation import deobfuscate_javascript
from .strings_ioc import decode_literal_candidates


_LANGUAGE_LABELS = {
    "bash": "Bash / Shell",
    "batch": "Batch / CMD",
    "c": "C",
    "cpp": "C++",
    "csharp": "C#",
    "go": "Go",
    "html": "HTML / HTA",
    "java": "Java",
    "javascript": "JavaScript",
    "kotlin": "Kotlin",
    "lua": "Lua",
    "perl": "Perl",
    "php": "PHP",
    "powershell": "PowerShell",
    "python": "Python",
    "ruby": "Ruby",
    "rust": "Rust",
    "scala": "Scala",
    "typescript": "TypeScript",
}

_BASE64_CALL_RE = re.compile(
    r"""(?ix)
    (?:
        base64\.(?:b64decode|urlsafe_b64decode)
        |base64_decode
        |(?:convert\.)?frombase64string
        |base64(?:\.(?:stdencoding|rawstdencoding|urlencoding))?\.decodestring
        |base64\.getdecoder\(\)\.decode
        |atob
        |buffer\.from
    )
    \s*\(\s*(?P<quote>['"])(?P<value>[A-Za-z0-9+/_-]{8,}={0,2})(?P=quote)
    """
)
_HEX_CALL_RE = re.compile(
    r"""(?ix)
    (?:
        (?:bytes|bytearray)\.fromhex
        |binascii\.unhexlify
        |hex2bin
        |hex\.decodestring
        |(?:convert\.)?fromhexstring
    )
    \s*\(\s*(?P<quote>['"])(?P<value>[0-9a-f]{8,})(?P=quote)
    """
)
_POWERSHELL_ENCODED_COMMAND_RE = re.compile(
    r"(?i)(?:^|\s)-(?:e|en|enc|enco|encod|encode|encodedcommand)\s+"
    r"(?P<quote>['\"]?)(?P<value>[A-Za-z0-9+/]{8,}={0,2})(?P=quote)"
)
_ESCAPE_TOKEN_RE = re.compile(
    r"\\x([0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})|\\u\{([0-9a-fA-F]{1,6})\}"
)
_ESCAPED_LITERAL_RE = re.compile(
    r"""(?P<quote>['"])(?P<value>[^'"\r\n]*(?:\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\u\{[0-9a-fA-F]{1,6}\})[^'"\r\n]*)(?P=quote)"""
)

_DANGEROUS_DECODED_PATTERNS = (
    (
        re.compile(
            r"""(?ix)
            \b(?:eval|exec|system|popen|shell_exec|passthru)\s*\(
            |\bos\.system\s*\(
            |\bsubprocess\.(?:run|call|popen|check_output)\s*\(
            |\bchild_process\b
            |\bruntime\.getruntime\(\)\.exec\s*\(
            |\bprocessbuilder\s*\(
            |\bexec\.command\s*\(
            |\binvoke-expression\b
            |\biex\b
            |\bstart-process\b
            |\b(?:cmd(?:\.exe)?\s+/c|powershell(?:\.exe)?\s+-)
            """
        ),
        "DEOB-EXEC",
        "解码后出现命令或动态代码执行",
        9,
    ),
    (
        re.compile(
            r"""(?ix)
            \b(?:fetch|xmlhttprequest)\s*\(
            |\brequests?\.(?:get|post|put)\s*\(
            |\binvoke-webrequest\b
            |\bdownloadstring\s*\(
            |\bwebclient\b
            |\b(?:curl|wget)\s+https?://
            |\bhttp\.(?:get|post)\s*\(
            |\bnet\.dial\s*\(
            |\bnew\s+socket\s*\(
            """
        ),
        "DEOB-NETWORK",
        "解码后出现网络下载或远程连接",
        8,
    ),
    (
        re.compile(
            r"""(?ix)
            \bschtasks\b
            |\brunonce\b
            |\bcrontab\b
            |\bsystemctl\s+enable\b
            |authorized_keys
            |\bnew-service\b
            """
        ),
        "DEOB-PERSISTENCE",
        "解码后出现持久化操作",
        8,
    ),
)


def deobfuscate_source(text: str, language: str) -> dict[str, Any]:
    """Decode bounded literal obfuscation for any supported source language."""

    normalized_language = str(language or "unknown").strip().lower()
    decoded: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    transformed = text
    javascript_finding_lines: set[int] = set()

    if normalized_language in {"javascript", "typescript", "html"}:
        javascript = deobfuscate_javascript(text)
        decoded.extend(javascript["decoded"])
        findings.extend(javascript["findings"])
        transformed = str(javascript["transformed"])
        javascript_finding_lines = {
            int(item.get("line") or 0) for item in javascript["findings"]
        }

    decoded.extend(_decode_call_literals(text))
    decoded.extend(_decode_escaped_literals(text))
    decoded.extend(decode_literal_candidates(text))
    if normalized_language == "powershell":
        decoded.extend(_decode_powershell_encoded_commands(text))

    decoded = _deduplicate(decoded)[:160]
    generic_decoded = [
        item for item in decoded
        if int(item.get("line") or 0) not in javascript_finding_lines
    ]
    findings.extend(_find_dangerous_decoded_behavior(generic_decoded, normalized_language))
    findings = _deduplicate_findings(findings)
    return {
        "transformed": transformed[:256 * 1024],
        "decoded": decoded,
        "findings": findings,
    }


def _decode_call_literals(text: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for match in _BASE64_CALL_RE.finditer(text):
        value = match.group("value")
        decoded = _decode_base64(value)
        if decoded:
            output.append(_candidate(text, match.start(), match.group(0), decoded, "base64-call"))
    for match in _HEX_CALL_RE.finditer(text):
        value = match.group("value")
        try:
            decoded = _readable_text(bytes.fromhex(value))
        except ValueError:
            decoded = None
        if decoded:
            output.append(_candidate(text, match.start(), match.group(0), decoded, "hex-call"))
    return output


def _decode_powershell_encoded_commands(text: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for match in _POWERSHELL_ENCODED_COMMAND_RE.finditer(text):
        value = match.group("value")
        try:
            payload = base64.b64decode(value + "=" * (-len(value) % 4), validate=True)
        except (ValueError, binascii.Error):
            continue
        decoded = _readable_text(payload, encodings=("utf-16le", "utf-8"))
        if decoded:
            output.append(_candidate(
                text, match.start(), match.group(0), decoded, "powershell-encoded-command",
            ))
    return output


def _decode_escaped_literals(text: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for match in _ESCAPED_LITERAL_RE.finditer(text):
        value = match.group("value")
        decoded = _ESCAPE_TOKEN_RE.sub(
            lambda item: chr(int(next(group for group in item.groups() if group), 16)),
            value,
        )
        if decoded != value and len(decoded.strip()) >= 4:
            output.append(_candidate(text, match.start(), match.group(0), decoded, "escaped-literal"))
    return output


def _decode_base64(value: str) -> str | None:
    padded = value + "=" * (-len(value) % 4)
    try:
        payload = base64.urlsafe_b64decode(padded)
    except (ValueError, binascii.Error):
        return None
    return _readable_text(payload, encodings=("utf-8", "utf-16le"))


def _readable_text(payload: bytes, encodings: tuple[str, ...] = ("utf-8",)) -> str | None:
    if not payload or len(payload) > 64 * 1024:
        return None
    for encoding in encodings:
        try:
            value = payload.decode(encoding)
        except UnicodeDecodeError:
            continue
        stripped = value.strip()
        if len(stripped) < 4:
            continue
        printable = sum(char.isprintable() or char in "\r\n\t" for char in stripped)
        if printable / len(stripped) >= 0.9:
            return stripped
    return None


def _candidate(
    text: str,
    offset: int,
    source: str,
    decoded: str,
    encoding: str,
) -> dict[str, object]:
    return {
        "encoding": encoding,
        "source": source[:4000],
        "decoded": decoded[:64 * 1024],
        "line": text.count("\n", 0, offset) + 1,
    }


def _find_dangerous_decoded_behavior(
    decoded: list[dict[str, object]],
    language: str,
) -> list[dict[str, object]]:
    label = _LANGUAGE_LABELS.get(language, language.upper() if language != "unknown" else "通用源码")
    findings: list[dict[str, object]] = []
    for item in decoded:
        value = str(item.get("decoded") or "")
        for pattern, rule_id, description, severity in _DANGEROUS_DECODED_PATTERNS:
            if not pattern.search(value):
                continue
            findings.append({
                "source": "source_deobfuscation",
                "rule_id": rule_id,
                "category": f"{label} 静态去混淆",
                "risk_type": "malicious",
                "behavior": rule_id.lower().replace("deob-", "decoded_"),
                "severity": severity,
                "line": int(item.get("line") or 1),
                "snippet": str(item.get("source") or "")[:240],
                "evidence": f"静态解码 {item.get('encoding')} 字面量后发现：{description}。",
                "description": f"{label} 中的常量静态解码后出现高风险行为特征；检测过程没有执行上传代码。",
                "repair_advice": "人工核对解码后的完整内容，移除无业务必要的混淆、动态执行、外联或持久化逻辑。",
                "confidence": 0.82,
            })
            break
    return findings


def _deduplicate(items: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        key = (str(item.get("decoded") or ""), int(item.get("line") or 0))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _deduplicate_findings(items: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for item in items:
        key = (
            str(item.get("rule_id") or ""),
            int(item.get("line") or 0),
            str(item.get("evidence") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output
