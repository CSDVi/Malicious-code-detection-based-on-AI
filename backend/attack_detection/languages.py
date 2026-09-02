"""Canonical source-language names and conservative content inference.

File extensions remain the primary signal.  Content inference is only used for
extensionless or generic text uploads so a ``.txt`` file containing PHP (or
another supported language) can still be routed to the validated model head.
"""

from __future__ import annotations

import json
import os
import re


EXTENSION_LANGUAGE = {
    ".php": "php", ".phtml": "php", ".phpt": "php",
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".jsp": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".go": "go",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".ps1": "powershell", ".psm1": "powershell", ".psd1": "powershell",
    ".bat": "batch", ".cmd": "batch",
    ".json": "config", ".yml": "config", ".yaml": "config", ".toml": "config",
    ".ini": "config", ".conf": "config",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hh": "cpp", ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".lua": "lua",
    ".pl": "perl", ".pm": "perl",
    ".html": "html", ".htm": "html", ".xhtml": "html", ".hta": "html",
    ".sql": "sql",
    ".txt": "unknown",
    ".exe": "binary", ".dll": "binary", ".sys": "binary", ".ocx": "binary",
}

CONFIG_DISPLAY_LANGUAGE = {
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".conf": "conf",
}

SOURCE_EXTENSIONS = {
    extension for extension, language in EXTENSION_LANGUAGE.items()
    if language != "binary"
}
GENERIC_TEXT_EXTENSIONS = {"", ".txt"}
BINARY_EXTENSIONS = {
    extension for extension, language in EXTENSION_LANGUAGE.items()
    if language == "binary"
}

# Names used by datasets that encode the language as a directory rather than
# a real file extension (notably CrossVul).
LANGUAGE_ALIASES = {
    "py": "python", "python": "python",
    "js": "javascript", "jsx": "javascript", "javascript": "javascript",
    "ts": "typescript", "tsx": "typescript", "typescript": "typescript",
    "java": "java", "jsp": "java",
    "kt": "kotlin", "kotlin": "kotlin",
    "php": "php", "phtml": "php", "phpt": "php", "inc": "php",
    "go": "go",
    "sh": "bash", "bash": "bash", "zsh": "bash",
    "ps1": "powershell", "psm1": "powershell", "psd1": "powershell",
    "powershell": "powershell", "pwsh": "powershell",
    "bat": "batch", "batch": "batch", "batchfile": "batch", "cmd": "batch",
    "json": "config", "yaml": "config", "yml": "config", "conf": "config",
    "c": "c", "h": "c",
    "cc": "cpp", "cpp": "cpp", "cxx": "cpp", "hh": "cpp", "hpp": "cpp",
    "cs": "csharp", "csharp": "csharp",
    "rb": "ruby", "ruby": "ruby",
    "rs": "rust", "rust": "rust",
    "scala": "scala", "lua": "lua",
    "pl": "perl", "pm": "perl", "perl": "perl",
    "html": "html", "htm": "html", "xhtml": "html", "hta": "html",
    "sql": "sql",
}

MODEL_LANGUAGES = tuple(dict.fromkeys(LANGUAGE_ALIASES.values()))


def canonical_language(value: str, default: str = "unknown") -> str:
    """Return the canonical model language for a dataset/user supplied name."""

    normalized = str(value or "").strip().lower().lstrip(".")
    return LANGUAGE_ALIASES.get(normalized, normalized if normalized in MODEL_LANGUAGES else default)


def language_from_path(path: str, default: str = "unknown") -> str:
    return EXTENSION_LANGUAGE.get(os.path.splitext(str(path).lower())[1], default)


def detect_source_language(path: str, content: str | None = None) -> str:
    """Detect language without overriding a specific, known file extension."""

    extension_language = language_from_path(path)
    if extension_language != "unknown" or content is None:
        return extension_language
    return infer_language_from_content(content)


def is_generic_text_path(path: str) -> bool:
    """Return whether a path requires content-based source-language detection."""

    return os.path.splitext(str(path).lower())[1] in GENERIC_TEXT_EXTENSIONS


def is_probably_text_payload(payload: bytes, *, sample_size: int = 64 * 1024) -> bool:
    """Reject obvious binary data while permitting UTF-8, CJK, and BOM text."""

    sample = bytes(payload[:sample_size])
    if not sample:
        return True
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in sample:
        return False
    allowed_controls = {8, 9, 10, 12, 13, 27}
    unexpected_controls = sum(
        byte < 32 and byte not in allowed_controls
        for byte in sample
    )
    return unexpected_controls / len(sample) <= 0.02


def decode_source_payload(payload: bytes) -> str:
    """Decode uploaded source text without losing common Unicode BOM formats."""

    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig", errors="replace")
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16", errors="replace")
    return payload.decode("utf-8", errors="ignore")


def display_language(language: str, path: str = "") -> str:
    """Return a concrete UI language without changing the model route.

    Configuration formats share the internal ``config`` route because the
    deployed models were trained that way.  The UI must still distinguish a
    JSON file from YAML and other configuration syntaxes.
    """

    normalized = str(language or "unknown").strip().lower()
    if normalized == "config":
        suffix = os.path.splitext(str(path).lower())[1]
        return CONFIG_DISPLAY_LANGUAGE.get(suffix, "config")
    if normalized == "unknown" and is_generic_text_path(path):
        return "text"
    return normalized


def infer_language_from_content(content: str) -> str:
    """Conservatively infer source language for generic text uploads."""

    text = str(content or "")[:256_000]
    if not text.strip():
        return "unknown"
    lowered = text.lower()
    first_line = text.lstrip().splitlines()[0] if text.lstrip() else ""
    if first_line.startswith("#!"):
        shebang = first_line.lower()
        if re.search(r"\bpython(?:\d+(?:\.\d+)*)?\b", shebang):
            return "python"
        if re.search(r"\b(?:node|nodejs|deno|bun)\b", shebang):
            return "javascript"
        if re.search(r"\b(?:pwsh|powershell)\b", shebang):
            return "powershell"
        if re.search(r"\bruby\b", shebang):
            return "ruby"
        if re.search(r"\bperl\b", shebang):
            return "perl"
        if re.search(r"\blua\b", shebang):
            return "lua"
    if re.search(r"<\?(?:php|=)", lowered) or (
        re.search(r"\$[a-z_]\w*", lowered) and re.search(r"\b(?:echo|function|include|require|eval)\b", lowered)
    ):
        return "php"
    if re.search(r"<!doctype\s+html|<html\b|<(?:body|form|head)\b", lowered):
        return "html"
    if re.search(r"^\s*#!.*\b(?:ba|z|k)?sh\b", lowered, re.MULTILINE) or (
        re.search(r"\b(?:then|fi|esac)\b", lowered) and re.search(r"\$\{|\$\(", text)
    ):
        return "bash"
    if re.search(r"\b(?:invoke-expression|invoke-webrequest|start-process|new-object)\b", lowered) or (
        re.search(r"\$[a-z_][\w:]*", lowered)
        and re.search(r"\b(?:param|write-host|get-item|set-item)\b", lowered)
    ):
        return "powershell"
    if re.search(r"^\s*@?echo\s+off\b", lowered, re.MULTILINE) or (
        re.search(r"%(?:~?dp0|[a-z_][\w]*)%", lowered)
        and re.search(r"^\s*(?:set|call|goto|if|for)\b", lowered, re.MULTILINE)
    ):
        return "batch"
    if re.search(r"^\s*package\s+[a-z_]\w*", text, re.MULTILINE) and re.search(r"\bfunc\s+\w+\s*\(", text):
        return "go"
    if re.search(r"\b(?:suspend\s+)?fun\s+\w+\s*\(", text) or re.search(r"\b(?:val|var)\s+\w+\s*:", text):
        return "kotlin"
    if re.search(r"\busing\s+System(?:\.|;)", text) or re.search(r"\bnamespace\s+[A-Za-z_]", text):
        return "csharp"
    if re.search(r"^\s*#\s*include\s*[<\"]", text, re.MULTILINE):
        return "cpp" if re.search(r"\b(?:std::|cout\s*<<|cin\s*>>|namespace\s+std)\b", text) else "c"
    if re.search(r"\bfn\s+main\s*\(", text) or re.search(r"\buse\s+std::", text):
        return "rust"
    if re.search(r"\bobject\s+\w+\s+extends\s+App\b", text) or re.search(r"\b(?:val|def)\s+\w+\s*:\s*[A-Z]", text):
        return "scala"
    if re.search(r"^\s*(?:local\s+)?function\s+\w+", text, re.MULTILINE) and re.search(r"\bend\b", text):
        return "lua"
    if re.search(r"^\s*(?:use\s+(?:strict|warnings)|my\s+\$\w+)", text, re.MULTILINE):
        return "perl"
    if re.search(r"^\s*(?:require\s+['\"]|class\s+\w+\s*<|def\s+\w+.*\n.*\bend\b)", text, re.MULTILINE):
        return "ruby"
    if re.search(r"\b(?:interface|type|enum)\s+[A-Za-z_$]\w*", text) or re.search(
        r"\b(?:const|let|function)\s+[A-Za-z_$]\w*\s*(?:\([^)]*\))?\s*:\s*[A-Za-z_{[]", text
    ):
        return "typescript"
    if re.search(r"\b(?:public|private|protected)\s+(?:static\s+)?(?:class|void|[A-Z]\w*)\b", text) or re.search(
        r"^\s*import\s+java\.", text, re.MULTILINE
    ):
        return "java"
    python_api_signal = re.search(
        r"\b(?:"
        r"os\.(?:system|popen|environ)|"
        r"subprocess\.(?:Popen|run|call|check_call|check_output)|"
        r"pickle\.loads|marshal\.loads|"
        r"requests\.(?:get|post|put|delete)|"
        r"urllib\.request|"
        r"socket\.socket|"
        r"sys\.(?:argv|path|modules)|"
        r"pathlib\.Path|"
        r"shutil\.|ctypes\.|"
        r"__import__\s*\("
        r")",
        text,
    )
    python_syntax_signal = re.search(
        r"^\s*(?:if|elif|for|while|try|except|with)\b[^;\n]*:\s*$",
        text,
        re.MULTILINE,
    )
    python_dynamic_execution = re.search(r"\bexec\s*\(", text) or (
        re.search(r"\beval\s*\(", text)
        and (
            python_api_signal
            or python_syntax_signal
            or re.search(r"\b(?:None|True|False|__builtins__|compile)\b", text)
        )
    )
    if (
        re.search(r"^\s*(?:async\s+)?def\s+\w+\s*\(", text, re.MULTILINE)
        or re.search(
            r"^\s*(?:from\s+[\w.]+\s+import|import\s+[\w.]+)",
            text,
            re.MULTILINE,
        )
        or python_api_signal
        or python_dynamic_execution
    ):
        return "python"
    if re.search(r"\b(?:const|let|var|function)\s+[A-Za-z_$]", text) or "=>" in text or re.search(
        r"\b(?:require\s*\(|module\.exports|console\.log)\b", text
    ):
        return "javascript"
    if re.search(r"^\s*(?:select|insert|update|delete|create|alter|drop)\b", lowered, re.MULTILINE):
        return "sql"
    if re.search(r"<(?:script)\b", lowered):
        return "html"
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(text)
            return "config"
        except json.JSONDecodeError:
            pass
    return "unknown"
