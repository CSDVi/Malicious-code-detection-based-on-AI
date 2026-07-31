"""Build a bounded-memory, deterministic training view of a JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _identity(row: dict[str, Any]) -> str:
    value = str(row.get("sample_hash") or "")
    if value:
        return value
    code = str(row.get("code") or "")
    return hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("language") or "unknown"),
        str(row.get("split") or "train"),
        str(row.get("label") or ""),
    )


def _reservoir(
    source: Path,
    caps: dict[str, int],
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    selected: dict[tuple[str, str, str], list[str]] = {}
    seen: dict[tuple[str, str, str], int] = {}
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _key(row)
            label = key[2]
            if label not in {"benign", "malicious", "vulnerable"}:
                continue
            cap = int(caps.get(key[1], 0))
            if cap <= 0:
                continue
            identity = _identity(row)
            bucket = selected.setdefault(key, [])
            count = seen.get(key, 0) + 1
            seen[key] = count
            if len(bucket) < cap:
                bucket.append(identity)
                continue
            replacement = rng.randrange(count)
            if replacement < cap:
                bucket[replacement] = identity
    return {
        identity
        for bucket in selected.values()
        for identity in bucket
    }


def build(source: Path, output: Path, max_code_chars: int, caps: dict[str, int]) -> dict[str, Any]:
    selected = _reservoir(source, caps, seed=20260725)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    total = 0
    truncated = 0
    with source.open("r", encoding="utf-8") as stream, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if _identity(row) not in selected:
                continue
            code = str(row.get("code") or "")
            if len(code) > max_code_chars:
                row["original_code_length"] = len(code)
                row["code"] = code[:max_code_chars]
                row["normalized_code"] = row["code"]
                row["training_truncated"] = True
                truncated += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            language = str(row.get("language") or "unknown")
            counts[language] = counts.get(language, 0) + 1
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "selected_rows": total,
        "truncated_rows": truncated,
        "max_code_chars": max_code_chars,
        "caps": caps,
        "by_language": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-code-chars", type=int, default=12000)
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument("--validation-cap", type=int, default=300)
    parser.add_argument("--test-cap", type=int, default=300)
    args = parser.parse_args()
    summary = build(
        args.source,
        args.output,
        max(1000, args.max_code_chars),
        {
            "train": max(1, args.train_cap),
            "validation": max(1, args.validation_cap),
            "test": max(1, args.test_cap),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
