"""Shared per-language deployment eligibility checks for classical models."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from attack_detection.dataset import CodeSample


DEFAULT_MINIMUMS = {"train": 20, "validation": 5, "test": 10}


def eligible_task_languages(
    partitions: Mapping[str, Sequence[CodeSample]],
    positive: str,
    negative: str,
    minimums: Mapping[str, int] | None = None,
) -> tuple[list[str], dict[str, dict[str, dict[str, int] | bool]]]:
    """Require both task classes in every split before claiming support."""

    required = dict(minimums or DEFAULT_MINIMUMS)
    languages = sorted({sample.language for values in partitions.values() for sample in values})
    coverage: dict[str, dict[str, dict[str, int] | bool]] = {}
    eligible = []
    for language in languages:
        language_report: dict[str, dict[str, int] | bool] = {}
        passed = True
        for split, minimum in required.items():
            counts = Counter(
                sample.label for sample in partitions.get(split, [])
                if sample.language == language and sample.label in {negative, positive}
            )
            row = {
                negative: counts[negative],
                positive: counts[positive],
                "minimum_per_class": int(minimum),
            }
            language_report[split] = row
            if counts[negative] < minimum or counts[positive] < minimum:
                passed = False
        language_report["eligible"] = passed
        coverage[language] = language_report
        if passed:
            eligible.append(language)
    return eligible, coverage
