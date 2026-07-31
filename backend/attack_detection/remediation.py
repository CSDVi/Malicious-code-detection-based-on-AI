"""Local, source-attributed remediation guidance.

The detector uses this catalog as a deterministic knowledge base.  Models may
select or classify evidence, but they do not generate unreviewed repair text.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


CATALOG_PATHS = (
    Path(__file__).with_name("remediation_catalog.json"),
    Path(__file__).with_name("remediation_catalog_owasp2025.json"),
)
MAX_SUGGESTIONS_PER_FINDING = 2


@lru_cache(maxsize=1)
def load_remediation_catalog() -> dict[str, Any]:
    merged: dict[str, Any] = {
        "schema_version": 1,
        "sources": {},
        "entries": {},
    }
    for path in CATALOG_PATHS:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        sources = data.get("sources")
        if isinstance(sources, dict):
            merged["sources"].update(sources)
        entries = data.get("entries")
        if isinstance(entries, dict):
            for category, entry in entries.items():
                if isinstance(entry, dict):
                    merged["entries"][str(category)] = entry
        if data.get("updated_at"):
            merged["updated_at"] = data["updated_at"]
    return merged


def remediation_for_finding(
    finding: dict[str, Any],
    language: str,
    fallback: str | None = None,
) -> dict[str, Any]:
    """Return deterministic suggestions and source metadata for one finding."""

    catalog = load_remediation_catalog()
    category = str(finding.get("category") or "")
    entry = (catalog.get("entries") or {}).get(category)
    if not isinstance(entry, dict):
        suggestions = _unique_texts([fallback, finding.get("repair_advice")])
        return {
            "suggestions": suggestions[:MAX_SUGGESTIONS_PER_FINDING],
            "references": [],
            "owasp": None,
            "cwe": finding.get("cwe"),
        }

    language_map = entry.get("language_suggestions") or {}
    language_values = (
        language_map.get(str(language or "").lower(), [])
        if isinstance(language_map, dict)
        else []
    )
    # A finding card should answer “what do I change here?” rather than dump
    # the whole catalog. Prefer one language-specific action and one
    # category-specific action so adjacent findings do not repeat a generic
    # checklist.
    language_suggestions = language_values if isinstance(language_values, list) else []
    category_suggestions = entry.get("suggestions") or []
    suggestions = _unique_texts([
        *language_suggestions[:1],
        *category_suggestions[:1],
        fallback,
        finding.get("repair_advice"),
    ])
    sources = catalog.get("sources") or {}
    references = []
    for source_id in entry.get("source_ids") or []:
        source = sources.get(str(source_id))
        if not isinstance(source, dict) or not source.get("url"):
            continue
        references.append({
            "id": str(source_id),
            "title": str(source.get("title") or source_id),
            "url": str(source["url"]),
        })
    return {
        "suggestions": suggestions[:MAX_SUGGESTIONS_PER_FINDING],
        "references": references,
        "owasp": entry.get("owasp"),
        "cwe": entry.get("cwe") or finding.get("cwe"),
    }


def catalog_statistics() -> dict[str, int]:
    catalog = load_remediation_catalog()
    entries = catalog.get("entries") or {}
    texts = []
    language_texts = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        texts.extend(entry.get("suggestions") or [])
        language_map = entry.get("language_suggestions") or {}
        if isinstance(language_map, dict):
            for values in language_map.values():
                if isinstance(values, list):
                    language_texts.extend(values)
    return {
        "categories": len(entries),
        "general_suggestions": len(_unique_texts(texts)),
        "language_suggestions": len(_unique_texts(language_texts)),
        "total_suggestions": len(_unique_texts([*texts, *language_texts])),
    }


def _unique_texts(values: Iterable[object]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output
