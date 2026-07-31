"""Static evidence engine combining strings/IOC, JS deobfuscation and chains."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from attack_detection.contracts import EngineResult
from attack_detection.task_policy import is_active_finding

from .behavior_chains import detect_behavior_chains
from .source_deobfuscation import deobfuscate_source
from .strings_ioc import classify_iocs


class StaticAnalysisEngine:
    name = "static_evidence"

    def scan(self, content: str, language: str, raw_bytes: bytes | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        ioc_findings = classify_iocs(content, raw_bytes)
        findings = list(ioc_findings)
        deobfuscated = deobfuscate_source(content, language)
        findings.extend(deobfuscated["findings"])
        metadata: dict[str, Any] = {
            "ioc_count": len(ioc_findings),
            "decoded_count": len(deobfuscated["decoded"]),
            "transform_applied": bool(deobfuscated["decoded"]),
            "deobfuscation_language": language,
            "decoded_artifacts": deobfuscated["decoded"][:30],
            "transformed_preview": deobfuscated["transformed"][:4000],
        }
        findings.extend(detect_behavior_chains(content))
        findings = [
            item for item in findings
            if isinstance(item, dict) and is_active_finding(item)
        ]
        counts = Counter(str(item.get("risk_type") or "context") for item in findings)
        malicious = counts["malicious"]
        decision = "malicious" if malicious else "benign"
        score = min(60, sum(int(item.get("severity") or 0) for item in findings) * 4)
        metadata.update({"malicious_hits": malicious, "vulnerability_hits": 0, "context_hits": counts["context"]})
        return EngineResult(
            name=self.name,
            status="completed",
            decision=decision,
            risk_score=score,
            duration_ms=int((time.perf_counter() - start) * 1000),
            findings=findings[:160],
            metadata=metadata,
        ).to_dict()
