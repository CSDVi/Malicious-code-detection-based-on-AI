"""Extract printable strings and indicators of compromise without execution."""

from __future__ import annotations

import base64
import binascii
import math
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable


URL_RE = re.compile(r"https?://[^\s\"'<>]{4,2048}", re.IGNORECASE)
DOMAIN_RE = re.compile(r"(?<![@\w])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}(?::\d{1,5})?(?:/[^\s\"'<>]*)?", re.IGNORECASE)
IP_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?![\d.])")
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}", re.IGNORECASE)
WEBHOOK_RE = re.compile(r"(?:discord(?:app)?\.com/api/webhooks|hooks\.slack\.com/services)/[^\s\"']+", re.IGNORECASE)
PATH_RE = re.compile(r"(?:/etc/passwd|/etc/shadow|[A-Z]:\\Users\\[^\s\"']+|%APPDATA%|%TEMP%|/proc/self|\.ssh/(?:id_rsa|authorized_keys))", re.IGNORECASE)
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])(?:[A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/])")
HEX_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){8,}(?![0-9A-Fa-f])")
COMMON_DOMAIN_SUFFIXES = {
    "com", "net", "org", "io", "co", "cn", "uk", "de", "fr", "ru",
    "pl", "jp", "kr", "dev", "app", "info", "biz", "xyz", "online",
    "site", "me", "tv", "cc", "top", "cloud", "tech", "ai",
}


def printable_strings(data: bytes | str, minimum: int = 4, limit: int = 4000) -> list[tuple[int | None, str]]:
    """Return ASCII and UTF-16LE strings with best-effort source line numbers."""
    if isinstance(data, str):
        text = data
        lines = text.splitlines()
        values = []
        for line_no, line in enumerate(lines, 1):
            for match in re.finditer(r"[\x20-\x7e]{%d,}" % minimum, line):
                values.append((line_no, match.group(0)))
                if len(values) >= limit:
                    return values
        return values

    values: list[tuple[int | None, str]] = []
    for match in re.finditer(rb"[\x20-\x7e]{%d,}" % minimum, data[:8 * 1024 * 1024]):
        values.append((None, match.group(0).decode("ascii", errors="ignore")))
        if len(values) >= limit:
            return values
    utf16 = data[:8 * 1024 * 1024]
    for match in re.finditer(rb"(?:[\x20-\x7e]\x00){%d,}" % minimum, utf16):
        raw = match.group(0)[::2]
        values.append((None, raw.decode("ascii", errors="ignore")))
        if len(values) >= limit:
            return values
    return values


def _line_for(
    text: str,
    value: str,
    newline_offsets: list[int] | None = None,
) -> int | None:
    index = text.find(value)
    if index < 0:
        return None
    offsets = newline_offsets if newline_offsets is not None else _newline_offsets(text)
    return bisect_right(offsets, index) + 1


def _newline_offsets(text: str) -> list[int]:
    return [index for index, value in enumerate(text) if value == "\n"]


def _email_matches(text: str) -> Iterable[tuple[int, str]]:
    """Run the email expression only on lines that contain its literal anchor."""

    offset = 0
    for line in text.splitlines(keepends=True):
        if "@" in line:
            for match in EMAIL_RE.finditer(line):
                yield offset + match.start(), match.group(0)
        offset += len(line)


def _printable(decoded: bytes, maximum: int = 512) -> str | None:
    if not decoded or len(decoded) > maximum:
        return None
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = "".join(char for char in text if char.isprintable() or char in "\r\n\t")
    return text.strip() if len(text.strip()) >= 4 else None


def decode_literal_candidates(
    text: str,
    limit: int = 80,
    *,
    newline_offsets: list[int] | None = None,
) -> list[dict[str, str | int]]:
    """Decode only literal base64/hex candidates; never call eval or import code."""
    decoded: list[dict[str, str | int]] = []
    offsets = (
        newline_offsets
        if newline_offsets is not None
        else _newline_offsets(text)
    )
    for match in BASE64_RE.finditer(text):
        value = match.group(0)
        try:
            result = _printable(base64.b64decode(value, validate=True))
        except (ValueError, binascii.Error):
            result = None
        if result:
            decoded.append({
                "encoding": "base64",
                "source": value,
                "decoded": result,
                "line": _line_for(text, value, offsets) or 1,
            })
        if len(decoded) >= limit:
            return decoded
    for match in HEX_RE.finditer(text):
        value = match.group(0)
        try:
            result = _printable(bytes.fromhex(value))
        except ValueError:
            result = None
        if result:
            decoded.append({
                "encoding": "hex",
                "source": value,
                "decoded": result,
                "line": _line_for(text, value, offsets) or 1,
            })
        if len(decoded) >= limit:
            break
    return decoded


def entropy(value: bytes) -> float:
    if not value:
        return 0.0
    size = len(value)
    counts = Counter(value).values()
    return round(
        -sum(
            (count / size) * math.log2(count / size)
            for count in counts
        ),
        3,
    )


def classify_iocs(text: str, raw: bytes | None = None) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    newline_offsets = _newline_offsets(text)
    url_spans = [match.span() for match in URL_RE.finditer(text)]
    sources: Iterable[tuple[str, re.Pattern[str], str]] = (
        ("url", URL_RE, "URL"),
        ("webhook", WEBHOOK_RE, "Webhook"),
        ("ip", IP_RE, "IP Address"),
        ("email", EMAIL_RE, "Email"),
        ("domain", DOMAIN_RE, "Domain"),
        ("path", PATH_RE, "Sensitive Path"),
    )
    for kind, pattern, label in sources:
        matches = (
            _email_matches(text)
            if kind == "email"
            else (
                (match.start(), match.group(0))
                for match in pattern.finditer(text)
            )
        )
        for match_start, match_value in matches:
            value = match_value.rstrip(".,;)")
            if kind == "domain":
                # Attribute access such as os.path, Crypto.Cipher and AES.new
                # is Python syntax, not a network indicator. URL-contained
                # domains are already represented by the URL finding.
                host = value.split("/", 1)[0].split(":", 1)[0]
                suffix = host.rsplit(".", 1)[-1].lower() if "." in host else ""
                inside_url = any(start <= match_start < end for start, end in url_spans)
                if suffix not in COMMON_DOMAIN_SUFFIXES or inside_url:
                    continue
            key = (kind, value.lower())
            if key in seen:
                continue
            seen.add(key)
            line = bisect_right(newline_offsets, match_start) + 1
            findings.append({
                "source": "strings_ioc",
                "rule_id": f"IOC-{kind.upper()}",
                "category": "IOC 线索",
                "risk_type": "context",
                "behavior": "indicator_of_compromise",
                "severity": 1 if kind in {"domain", "email"} else 2,
                "line": line,
                "snippet": value[:240],
                "evidence": f"检测到 {label}: {value[:240]}",
                "description": "该字符串是可供人工或外部情报核对的 IOC 线索，不单独证明恶意。",
                "repair_advice": "核对域名/IP/URL 的业务用途、所有权和历史信誉，避免仅凭字符串直接定性。",
                "confidence": 0.65,
            })
            if len(findings) >= 120:
                return findings
    for item in decode_literal_candidates(text, newline_offsets=newline_offsets):
        decoded = str(item["decoded"])
        if URL_RE.search(decoded) or IP_RE.search(decoded) or DOMAIN_RE.search(decoded):
            findings.append({
                "source": "strings_ioc",
                "rule_id": "IOC-DECODED",
                "category": "IOC 线索",
                "risk_type": "context",
                "behavior": "decoded_indicator",
                "severity": 2,
                "line": int(item["line"]),
                "snippet": str(item["source"])[:240],
                "evidence": f"静态解码 {item['encoding']} 字面量得到: {decoded[:240]}",
                "description": "编码后的字符串解码后包含 IOC 线索；解码过程是静态分析，未执行其内容。",
                "repair_advice": "确认编码数据来源，删除不必要的混淆，并核对解码后的网络或文件目标。",
                "confidence": 0.75,
            })
    if raw:
        strings = printable_strings(raw)
        high_entropy = [value for _, value in strings if len(value) >= 24 and entropy(value.encode("utf-8", errors="ignore")) >= 4.3]
        if high_entropy:
            findings.append({
                "source": "strings_ioc",
                "rule_id": "STR-HIGH-ENTROPY",
                "category": "高熵字符串",
                "risk_type": "context",
                "behavior": "packed_or_encoded_string",
                "severity": 2,
                "line": None,
                "snippet": high_entropy[0][:240],
                "evidence": f"发现 {len(high_entropy)} 个较长高熵字符串，可能是编码、压缩或随机数据。",
                "description": "高熵字符串可见于压缩、加密和打包程序，也可能见于隐藏载荷；需要结合结构和行为复核。",
                "repair_advice": "检查字符串来源与解码调用，避免把正常压缩/加密数据直接判为恶意。",
                "confidence": 0.45,
            })
    return findings[:120]
