"""Resolve legacy and organized practiceset directory layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VULNERABILITY_DIRECTORY = "vulnerability_detection"
MALWARE_DIRECTORY = "malware_detection"


@dataclass(frozen=True)
class PracticesetLayout:
    root: Path
    vulnerability: Path
    java: Path
    javascript: Path
    php: Path
    python: Path
    other: Path
    organized: bool


def resolve_practiceset_layout(root: Path) -> PracticesetLayout:
    """Return language/category roots while retaining legacy-layout support."""

    resolved = root.resolve()
    vulnerability = resolved / VULNERABILITY_DIRECTORY
    malware = resolved / MALWARE_DIRECTORY
    organized = vulnerability.is_dir() and malware.is_dir()
    if not organized:
        return PracticesetLayout(
            root=resolved,
            vulnerability=resolved,
            java=resolved,
            javascript=resolved,
            php=resolved,
            python=resolved,
            other=resolved,
            organized=False,
        )
    return PracticesetLayout(
        root=resolved,
        vulnerability=vulnerability,
        java=malware / "java",
        javascript=malware / "javascript",
        php=malware / "php",
        python=malware / "python",
        other=malware / "other",
        organized=True,
    )
