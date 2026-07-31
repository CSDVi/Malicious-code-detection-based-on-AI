"""Stable detection contracts shared by engines, orchestration, and storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EngineStatus = Literal["completed", "skipped", "unavailable", "failed"]
Decision = Literal["benign", "malicious", "vulnerable", "unknown"]
DetectionMode = Literal["quick", "standard", "deep", "auto"]


@dataclass(frozen=True)
class Finding:
    source: str
    category: str
    severity: int
    line: int | None = None
    cwe: str | None = None
    behavior: str | None = None
    rule_id: str | None = None
    snippet: str | None = None
    description: str | None = None
    repair_advice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineResult:
    name: str
    status: EngineStatus
    decision: Decision = "unknown"
    probability: float | None = None
    threshold: float | None = None
    risk_score: int | None = None
    model_version: str | None = None
    duration_ms: int | None = None
    reason: str | None = None
    error: str | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionRequest:
    filename: str
    content: str
    selected_mode: DetectionMode = "auto"
    is_project: bool = False


@dataclass(frozen=True)
class DetectionResult:
    selected_mode: DetectionMode
    effective_mode: DetectionMode
    final_decision: Decision
    risk_score: int
    confidence: float | None
    engines: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    escalation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
