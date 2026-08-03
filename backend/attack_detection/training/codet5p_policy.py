"""Operational decision policy shared by CodeT5+ training and inference."""

from __future__ import annotations

from typing import Any


ABSOLUTE_MIN_DECISION_THRESHOLD = 0.01
LANGUAGE_THRESHOLD_MIN = 0.70
LANGUAGE_THRESHOLD_CENTER = 0.80
MAX_DECISION_THRESHOLD = 0.85
DECISION_THRESHOLD_STEP = 0.01


def effective_decision_threshold(value: Any) -> float:
    """Clamp trained thresholds to the supported production range."""

    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = MAX_DECISION_THRESHOLD
    return min(
        MAX_DECISION_THRESHOLD,
        max(ABSOLUTE_MIN_DECISION_THRESHOLD, threshold),
    )


def decision_threshold_candidates() -> list[float]:
    """Return the validation-search band centered around 0.80."""

    lower = round(LANGUAGE_THRESHOLD_MIN / DECISION_THRESHOLD_STEP)
    upper = round(MAX_DECISION_THRESHOLD / DECISION_THRESHOLD_STEP)
    return [
        round(index * DECISION_THRESHOLD_STEP, 2)
        for index in range(lower, upper + 1)
    ]
