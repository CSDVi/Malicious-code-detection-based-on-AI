"""Mask comments and true docstrings while preserving source coordinates.

Static rules should inspect executable source, not prose that merely describes a
dangerous operation.  Every masked character is replaced with a space while
line breaks are retained, so findings can still point back to the original
line without maintaining a second source map.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import warnings
from collections.abc import Iterable


_C_STYLE_LANGUAGES = frozenset({
    "c", "cpp", "csharp", "go", "java", "javascript", "kotlin", "rust",
    "scala", "typescript",
})
_HASH_LANGUAGES = frozenset({
    "bash", "config", "perl", "php", "powershell", "ruby", "unknown",
})


def mask_non_executable_text(content: str, language: str) -> str:
    """Return same-length source with comments/docstrings replaced by spaces."""

    if not content:
        return content
    normalized = str(language or "unknown").strip().lower()
    if normalized == "python":
        spans = _python_comment_and_docstring_spans(content)
    elif normalized == "batch":
        spans = _batch_comment_spans(content)
    elif normalized == "html":
        spans = _scan_comment_spans(content, (), (("<!--", "-->"),))
    elif normalized == "lua":
        spans = _scan_comment_spans(content, ("--",), (("--[[", "]]"),))
    elif normalized == "sql":
        spans = _scan_comment_spans(content, ("--",), (("/*", "*/"),))
    elif normalized == "powershell":
        spans = _scan_comment_spans(content, ("#",), (("<#", "#>"),))
    elif normalized == "ruby":
        spans = [
            *_scan_comment_spans(content, ("#",), ()),
            *_ruby_block_comment_spans(content),
        ]
    elif normalized == "php":
        spans = _scan_comment_spans(content, ("//", "#"), (("/*", "*/"),))
    elif normalized in _C_STYLE_LANGUAGES:
        spans = _scan_comment_spans(content, ("//",), (("/*", "*/"),))
    elif normalized in _HASH_LANGUAGES:
        spans = _scan_comment_spans(content, ("#", "//"), (("/*", "*/"),))
    else:
        spans = _scan_comment_spans(content, ("//",), (("/*", "*/"),))
    return _mask_spans(content, spans)


def _mask_spans(content: str, spans: Iterable[tuple[int, int]]) -> str:
    characters = list(content)
    for start, end in spans:
        for index in range(max(0, start), min(len(characters), end)):
            if characters[index] not in "\r\n":
                characters[index] = " "
    return "".join(characters)


def _scan_comment_spans(
    content: str,
    line_markers: tuple[str, ...],
    block_markers: tuple[tuple[str, str], ...],
) -> list[tuple[int, int]]:
    """Locate comments without treating markers inside quoted strings as code."""

    spans: list[tuple[int, int]] = []
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(content):
        char = content[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "'" and not _has_unescaped_closing_quote(content, index, char):
            index += 1
            continue
        if char in {'"', "'", "`"}:
            quote = char
            index += 1
            continue
        block = next(
            ((start, end) for start, end in block_markers if content.startswith(start, index)),
            None,
        )
        if block is not None:
            start_marker, end_marker = block
            end = content.find(end_marker, index + len(start_marker))
            end = len(content) if end < 0 else end + len(end_marker)
            spans.append((index, end))
            index = end
            continue
        marker = next(
            (value for value in line_markers if content.startswith(value, index)),
            None,
        )
        if marker is not None and _line_marker_is_comment(content, index, marker):
            end = content.find("\n", index + len(marker))
            end = len(content) if end < 0 else end
            spans.append((index, end))
            index = end
            continue
        index += 1
    return spans


def _has_unescaped_closing_quote(content: str, start: int, quote: str) -> bool:
    index = start + 1
    escaped = False
    while index < len(content) and content[index] not in "\r\n":
        char = content[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return True
        index += 1
    return False


def _line_marker_is_comment(content: str, index: int, marker: str) -> bool:
    if marker != "#":
        return True
    # Avoid masking shell parameter operations such as ${value#prefix}.  A hash
    # begins a comment when it starts a logical line or follows whitespace/a
    # command separator.  Python and PHP are handled by dedicated branches.
    if index == 0:
        return True
    previous = content[index - 1]
    return previous.isspace() or previous in ";|&(){}"


def _python_comment_and_docstring_spans(content: str) -> list[tuple[int, int]]:
    lines = content.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    spans: list[tuple[int, int]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                spans.append((
                    _token_position(offsets, lines, *token.start),
                    _token_position(offsets, lines, *token.end),
                ))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        spans.extend(_scan_comment_spans(content, ("#",), ()))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
        bodies = [tree.body]
        bodies.extend(
            node.body
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        for body in bodies:
            if not body:
                continue
            first = body[0]
            if not (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                continue
            spans.append((
                _ast_position(offsets, lines, first.lineno, first.col_offset),
                _ast_position(
                    offsets,
                    lines,
                    int(getattr(first, "end_lineno", first.lineno)),
                    int(getattr(first, "end_col_offset", first.col_offset)),
                ),
            ))
    except (SyntaxError, ValueError, TypeError):
        pass
    return spans


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    return offsets


def _token_position(offsets: list[int], lines: list[str], row: int, column: int) -> int:
    if not offsets:
        return 0
    line_index = min(max(row - 1, 0), len(offsets) - 1)
    return offsets[line_index] + min(column, len(lines[line_index]))


def _ast_position(offsets: list[int], lines: list[str], row: int, byte_column: int) -> int:
    if not offsets:
        return 0
    line_index = min(max(row - 1, 0), len(offsets) - 1)
    raw_line = lines[line_index]
    character_column = len(
        raw_line.encode("utf-8")[:max(0, byte_column)].decode("utf-8", errors="ignore")
    )
    return offsets[line_index] + min(character_column, len(raw_line))


def _batch_comment_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        stripped = body.lstrip()
        if stripped.startswith("::") or re.match(r"(?i)^rem(?:\s|$)", stripped):
            start = offset + len(body) - len(stripped)
            spans.append((start, offset + len(body)))
        offset += len(line)
    return spans


def _ruby_block_comment_spans(content: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in re.finditer(r"(?ms)^=begin\b.*?^=end\b[^\r\n]*", content)
    ]
