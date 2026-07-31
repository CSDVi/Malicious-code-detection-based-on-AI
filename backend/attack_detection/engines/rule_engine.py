"""Rule engine adapter."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from attack_detection.contracts import EngineResult
from attack_detection.rules import detect_by_rules
from attack_detection.task_policy import is_active_finding


class RuleEngine:
    name = "rule_engine"

    def scan(self, content: str, language: str) -> dict[str, Any]:
        start = time.perf_counter()
        matches = [
            match for match in detect_by_rules(content, language)
            if isinstance(match, dict) and is_active_finding(match)
        ]
        type_counts = Counter(str(match.get("risk_type", "unknown")) for match in matches)
        severity = sum(int(match.get("severity") or 0) for match in matches)
        risk_score = min(95, int(severity * 7.5)) if matches else 0
        if type_counts["malicious"]:
            decision = "malicious"
        elif type_counts["vulnerable"]:
            decision = "vulnerable"
        elif matches:
            decision = "unknown"
        else:
            decision = "benign"
        return EngineResult(
            name=self.name,
            status="completed",
            decision=decision,
            risk_score=risk_score,
            duration_ms=int((time.perf_counter() - start) * 1000),
            findings=matches,
            metadata={
                "hits": len(matches),
                "malicious_hits": type_counts["malicious"],
                "vulnerability_hits": type_counts["vulnerable"],
            },
        ).to_dict()
