"""Audit false positives/negatives without modifying training data.

This script is intentionally read-only with respect to source datasets and
model candidates. It writes a JSON error inventory that can be used to design
the next dataset/feature revision without leaking held-out samples into train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import load_dataset
from attack_detection.features.behavior_tokens import (
    BEHAVIOR_TOKEN_VERSION,
    behavior_token_text,
)
from attack_detection.features.static_features import feature_vector


def _matrix(rows: list[Any], bundle: dict[str, Any]):
    import numpy
    from scipy.sparse import csr_matrix, hstack

    structured = numpy.asarray(
        [
            feature_vector(
                row.code,
                row.language,
                feature_names=bundle.get("feature_names"),
                include_rules=False,
            )
            for row in rows
        ],
        dtype="float32",
    )
    if bundle.get("feature_mode") == "structured_static":
        return csr_matrix(structured, dtype="float32")
    texts = [
        (
            behavior_token_text(row.code, row.language)
            if bundle.get("text_transform") == BEHAVIOR_TOKEN_VERSION
            else row.code
        )
        for row in rows
    ]
    return hstack(
        [
            csr_matrix(structured),
            bundle["word_vectorizer"].transform(texts),
            bundle["char_vectorizer"].transform(texts),
        ],
        format="csr",
        dtype="float32",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _row_summary(row: Any, score: float, threshold: float, predicted: str) -> dict[str, Any]:
    return {
        "sample_hash": row.sample_hash,
        "code_sha256": _sha256_text(row.code),
        "language": row.language,
        "split": row.split,
        "label": row.label,
        "predicted": predicted,
        "score": round(float(score), 8),
        "threshold": threshold,
        "family": row.family,
        "source": row.source,
        "file_path": row.file_path,
        "category": row.category,
        "review_status": row.review_status,
        "behavior_labels": list(row.behavior_labels),
        "code_chars": len(row.code),
        "preview": row.code[:500].replace("\x00", "").replace("\r", " ").replace("\n", " "),
    }


def audit(candidate: Path, dataset: Path) -> dict[str, Any]:
    from joblib import load

    bundle = load(candidate.with_suffix(".joblib"))
    metrics = json.loads(candidate.with_suffix(".json").read_text(encoding="utf-8"))
    threshold = float(bundle["threshold"])
    rows = [
        row
        for row in load_dataset(dataset)
        if row.label in {"benign", "malicious"}
    ]
    result: dict[str, Any] = {
        "candidate": str(candidate.resolve()),
        "candidate_metrics": str(candidate.with_suffix(".json").resolve()),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256_file(dataset),
        "language": metrics["language"],
        "feature_mode": metrics["feature_mode"],
        "threshold": threshold,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row.split == split]
        if not split_rows:
            continue
        scores = bundle["model"].predict_proba(_matrix(split_rows, bundle))[:, 1]
        errors: list[dict[str, Any]] = []
        confusion = Counter()
        by_family: dict[str, Counter[str]] = defaultdict(Counter)
        by_source: dict[str, Counter[str]] = defaultdict(Counter)
        for row, score in zip(split_rows, scores):
            predicted = "malicious" if float(score) >= threshold else "benign"
            confusion[(row.label, predicted)] += 1
            by_family[row.family][(row.label, predicted)] += 1
            by_source[row.source][(row.label, predicted)] += 1
            if predicted != row.label:
                errors.append(_row_summary(row, float(score), threshold, predicted))
        result["splits"][split] = {
            "rows": len(split_rows),
            "label_counts": dict(Counter(row.label for row in split_rows)),
            "confusion": {
                f"{actual}->{predicted}": count
                for (actual, predicted), count in sorted(confusion.items())
            },
            "errors": sorted(errors, key=lambda item: item["score"], reverse=True),
            "family_confusion": {
                family: {
                    f"{actual}->{predicted}": count
                    for (actual, predicted), count in sorted(counts.items())
                }
                for family, counts in sorted(by_family.items())
            },
            "source_confusion": {
                source: {
                    f"{actual}->{predicted}": count
                    for (actual, predicted), count in sorted(counts.items())
                }
                for source, counts in sorted(by_source.items())
            },
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    # PowerShell on Windows may expose a GBK stdout codec.  The audit contains
    # source previews, so console output must never make the analysis fail
    # after the JSON evidence has already been written.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate metrics prefix, e.g. powershell=backend/models/candidates/v12/xgb_malicious_powershell",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="language=dataset JSONL path",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidates = {
        value.split("=", 1)[0]: Path(value.split("=", 1)[1]).resolve()
        for value in args.candidate
    }
    datasets = {
        value.split("=", 1)[0]: Path(value.split("=", 1)[1]).resolve()
        for value in args.dataset
    }
    if set(candidates) != set(datasets):
        raise SystemExit("candidate and dataset language keys must match")
    result = {
        "schema_version": 1,
        "created_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(timespec="seconds"),
        "offline_text_only": True,
        "candidates": {
            language: audit(candidates[language], datasets[language])
            for language in sorted(candidates)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
