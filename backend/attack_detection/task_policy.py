"""Runtime task policy for the public detector.

The vulnerability *model* remains disabled, while deterministic rule findings
with source-level evidence are allowed.  Keeping the distinction here prevents
an unavailable classifier from leaking into the UI without suppressing SQL
injection, XSS, SSRF, and other explainable rule evidence.
"""

from __future__ import annotations

ACTIVE_TASKS = frozenset({"malicious_intent", "project_malicious_intent"})
DISABLED_TASKS = frozenset({"vulnerability_risk"})
RULE_VULNERABILITY_SOURCES = frozenset({"rule_engine"})


def task_enabled(task: str | None) -> bool:
    return str(task or "") in ACTIVE_TASKS


def is_disabled_task(task: str | None) -> bool:
    return str(task or "") in DISABLED_TASKS


def is_active_finding(finding: dict[str, object]) -> bool:
    """Keep all non-vulnerability evidence and sourced rule vulnerabilities."""

    risk_type = str(finding.get("risk_type") or finding.get("behavior") or "")
    if risk_type != "vulnerable":
        return True
    return str(finding.get("source") or "") in RULE_VULNERABILITY_SOURCES
