"""Build a compact vulnerability dataset with entire sources held out.

The test partition contains no source seen during training/validation.  Benign
training rows containing security-sensitive APIs are tagged as hard negatives
so the trainer can up-weight them without duplicating source code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


HOLDOUT_SOURCE = {
    "c": "zenodo_13870382",
    "cpp": "zenodo_13870382",
    "csharp": "nist_juliet_csharp_1.3",
    "go": "zenodo_13870382",
    "java": "owasp_benchmark_java",
    "javascript": "zenodo_13870382",
    "php": "zenodo_13870382",
    "python": "zenodo_13870382",
    "ruby": "zenodo_13870382",
}
SENSITIVE_PATTERNS = (
    r"\b(?:exec|eval|system|popen|shell_exec|passthru|subprocess|processbuilder)\b",
    r"\b(?:socket|requests?|httpx|fetch|curl|urlopen|webclient)\b",
    r"\b(?:select|insert|update|delete|query|execute|cursor)\b",
    r"\b(?:base64|b64decode|fromcharcode|unescape|decode|fromhex)\b",
    r"\b(?:open|write|writefile|file_put_contents|fopen|unlink|rename|chmod)\b",
    r"\b(?:md5|sha1|sha256|crypto|cipher|encrypt|decrypt)\b",
    r"\b(?:getenv|environ|process\.env|request|params?|argv|stdin)\b",
)


def _identity(row: dict[str, Any]) -> str:
    value = str(row.get("sample_hash") or "")
    if value:
        return value
    return hashlib.sha256(
        str(row.get("code") or "").encode("utf-8", errors="ignore")
    ).hexdigest()


def _family_bucket(row: dict[str, Any]) -> int:
    family = str(row.get("family") or row.get("package_name") or _identity(row))
    return int(hashlib.sha256(family.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % 100


def _hard_negative_score(code: str) -> int:
    lowered = code.lower()
    return sum(bool(re.search(pattern, lowered)) for pattern in SENSITIVE_PATTERNS)


def _eligible(row: dict[str, Any], languages: set[str]) -> bool:
    if str(row.get("language") or "") not in languages:
        return False
    if str(row.get("label") or "") not in {"benign", "vulnerable"}:
        return False
    scopes = row.get("label_scopes")
    return not scopes or "vulnerability_risk" in scopes


def _assigned_split(row: dict[str, Any]) -> str:
    language = str(row.get("language") or "")
    if str(row.get("source") or "") == HOLDOUT_SOURCE[language]:
        return "test"
    return "validation" if _family_bucket(row) < 25 else "train"


def _bucket(row: dict[str, Any]) -> tuple[str, str, str, str]:
    split = _assigned_split(row)
    label = str(row.get("label") or "")
    priority = "regular"
    if split == "train" and label == "benign":
        priority = "hard" if _hard_negative_score(str(row.get("code") or "")) >= 2 else "regular"
    return str(row.get("language") or ""), split, label, priority


def _cap(key: tuple[str, str, str, str], train_cap: int, eval_cap: int, hard_cap: int) -> int:
    _, split, label, priority = key
    if split in {"validation", "test"}:
        return eval_cap
    if label == "benign" and priority == "hard":
        return min(train_cap, hard_cap)
    if label == "benign":
        return max(1, train_cap - min(train_cap, hard_cap))
    return train_cap


def build(
    source: Path,
    output: Path,
    report_path: Path,
    languages: set[str],
    train_cap: int,
    eval_cap: int,
    hard_cap: int,
    max_code_chars: int,
) -> dict[str, Any]:
    rng = random.Random(20260725)
    reservoirs: dict[tuple[str, str, str, str], list[str]] = {}
    seen: dict[tuple[str, str, str, str], int] = {}
    source_counts: dict[tuple[str, str, str, str], int] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not _eligible(row, languages):
                continue
            key = _bucket(row)
            source_key = (
                key[0],
                key[1],
                str(row.get("label") or ""),
                str(row.get("source") or ""),
            )
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            cap = _cap(key, train_cap, eval_cap, hard_cap)
            count = seen.get(key, 0) + 1
            seen[key] = count
            values = reservoirs.setdefault(key, [])
            identity = _identity(row)
            if len(values) < cap:
                values.append(identity)
            else:
                replacement = rng.randrange(count)
                if replacement < cap:
                    values[replacement] = identity

    selected = {identity for values in reservoirs.values() for identity in values}
    output.parent.mkdir(parents=True, exist_ok=True)
    output_counts: dict[tuple[str, str, str], int] = {}
    hard_counts: dict[str, int] = {}
    sources_by_language_split: dict[tuple[str, str], set[str]] = {}
    families_by_language_split: dict[tuple[str, str], set[str]] = {}
    written: set[str] = set()
    with source.open("r", encoding="utf-8") as stream, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = _identity(row)
            if identity not in selected or identity in written or not _eligible(row, languages):
                continue
            split = _assigned_split(row)
            code = str(row.get("code") or "")
            hard_score = _hard_negative_score(code) if row.get("label") == "benign" else 0
            row["split"] = split
            row["source_heldout_protocol"] = True
            row["heldout_source"] = HOLDOUT_SOURCE[str(row.get("language") or "")]
            row["hard_negative_score"] = hard_score
            row["hard_negative"] = split == "train" and row.get("label") == "benign" and hard_score >= 2
            if row["hard_negative"]:
                language = str(row.get("language") or "")
                hard_counts[language] = hard_counts.get(language, 0) + 1
            if len(code) > max_code_chars:
                row["original_code_length"] = len(code)
                row["code"] = code[:max_code_chars]
                row["normalized_code"] = row["code"]
                row["training_truncated"] = True
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            written.add(identity)
            count_key = (
                str(row.get("language") or ""),
                split,
                str(row.get("label") or ""),
            )
            output_counts[count_key] = output_counts.get(count_key, 0) + 1
            language = str(row.get("language") or "")
            sources_by_language_split.setdefault((language, split), set()).add(str(row.get("source") or ""))
            families_by_language_split.setdefault((language, split), set()).add(
                str(row.get("family") or row.get("package_name") or identity)
            )

    source_leaks: dict[str, list[str]] = {}
    family_leaks: dict[str, list[str]] = {}
    for language in sorted(languages):
        development_sources = (
            sources_by_language_split.get((language, "train"), set())
            | sources_by_language_split.get((language, "validation"), set())
        )
        leaked_sources = development_sources & sources_by_language_split.get((language, "test"), set())
        if leaked_sources:
            source_leaks[language] = sorted(leaked_sources)
        leaked_families = (
            families_by_language_split.get((language, "train"), set())
            & families_by_language_split.get((language, "validation"), set())
        )
        if leaked_families:
            family_leaks[language] = sorted(leaked_families)[:20]
    if source_leaks or family_leaks:
        raise ValueError(
            f"held-out split leakage detected: sources={source_leaks}, families={family_leaks}"
        )

    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "task": "vulnerability_risk",
        "protocol": "entire source held out for test; repository family held together in train/validation",
        "heldout_sources": {
            language: HOLDOUT_SOURCE[language]
            for language in sorted(languages)
        },
        "rows": len(written),
        "hard_negative_rows": hard_counts,
        "source_isolation_verified": True,
        "train_validation_family_isolation_verified": True,
        "counts": [
            {"language": language, "split": split, "label": label, "rows": rows}
            for (language, split, label), rows in sorted(output_counts.items())
        ],
        "input_source_counts": [
            {"language": language, "split": split, "label": label, "source": source_name, "rows": rows}
            for (language, split, label, source_name), rows in sorted(source_counts.items())
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--languages",
        default=",".join(HOLDOUT_SOURCE),
        help="comma-separated canonical languages",
    )
    parser.add_argument("--train-cap", type=int, default=2500)
    parser.add_argument("--eval-cap", type=int, default=500)
    parser.add_argument("--hard-negative-cap", type=int, default=800)
    parser.add_argument("--max-code-chars", type=int, default=12000)
    args = parser.parse_args()
    languages = {value.strip() for value in args.languages.split(",") if value.strip()}
    unknown = languages - set(HOLDOUT_SOURCE)
    if unknown:
        raise SystemExit(f"no holdout source configured for: {sorted(unknown)}")
    print(json.dumps(build(
        args.source,
        args.output,
        args.report,
        languages,
        max(1, args.train_cap),
        max(1, args.eval_cap),
        max(0, args.hard_negative_cap),
        max(1000, args.max_code_chars),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
