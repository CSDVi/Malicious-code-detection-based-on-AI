"""Deterministic text enrichment shared by training and runtime inference."""

from __future__ import annotations

import re

from attack_detection.rules import detect_by_rules


TRANSFORM_NAME = "language_and_rule_tokens_v1"


def enrich_model_text(content: str, language: str) -> str:
    tokens = [f"model_language_{_token(language)}"]
    for match in detect_by_rules(content, language):
        tokens.extend((
            f"model_rule_{_token(match.get('rule_id'))}",
            f"model_risk_{_token(match.get('risk_type'))}",
            f"model_category_{_token(match.get('category'))}",
        ))
    return content + "\n" + " ".join(tokens)


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "unknown").lower()).strip("_") or "unknown"
