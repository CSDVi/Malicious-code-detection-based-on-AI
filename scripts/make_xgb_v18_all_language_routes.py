"""Build per-language malicious-intent route files for the v18 dataset."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from attack_detection.dataset import is_task_training_eligible, load_dataset


DATASET = Path(
    os.environ.get(
        "XGB_ROUTE_DATASET",
        str(ROOT / "backend/data/processed/xgb_multilingual_malicious_20260727_v18.jsonl"),
    )
)
OUT_DIR = Path(
    os.environ.get(
        "XGB_ROUTE_OUT_DIR",
        str(ROOT / "backend/data/splits/xgb_v18_all_language_routes"),
    )
)
MANIFEST = Path(
    os.environ.get(
        "XGB_ROUTE_MANIFEST",
        str(
            ROOT
            / "artifacts/xgb_multilingual_optimization_20260727/evidence/v18_all_language_routes_manifest.json"
        ),
    )
)
TASK = "malicious_intent"
POSITIVE = "malicious"
NEGATIVE = "benign"
SPLITS = ("train", "validation", "test")

ELIGIBLE_LANGUAGES = {
    "bash",
    "c",
    "config",
    "cpp",
    "go",
    "html",
    "java",
    "javascript",
    "php",
    "powershell",
    "python",
    "ruby",
    "rust",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[object]] = defaultdict(list)
    for sample in load_dataset(DATASET):
        language = sample.language.lower()
        split = sample.split.lower()
        if language not in ELIGIBLE_LANGUAGES:
            continue
        if split not in SPLITS:
            continue
        if sample.label not in {NEGATIVE, POSITIVE}:
            continue
        if not is_task_training_eligible(sample, TASK):
            continue
        rows[language].append(sample)

    manifest = {
        "dataset": str(DATASET),
        "task": TASK,
        "positive_label": POSITIVE,
        "negative_label": NEGATIVE,
        "output_dir": str(OUT_DIR),
        "languages": {},
    }

    for language in sorted(ELIGIBLE_LANGUAGES):
        records = rows.get(language, [])
        route_path = OUT_DIR / f"{language}.jsonl"
        with route_path.open("w", encoding="utf-8") as handle:
            for sample in records:
                handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

        split_counts: dict[str, dict[str, int]] = {}
        family_counts: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            split_rows = [sample for sample in records if sample.split.lower() == split]
            labels = Counter(sample.label for sample in split_rows)
            families = {
                label: len({
                    sample.family or sample.package_name or sample.source or sample.sample_hash
                    for sample in split_rows
                    if sample.label == label
                })
                for label in (NEGATIVE, POSITIVE)
            }
            split_counts[split] = {
                NEGATIVE: labels.get(NEGATIVE, 0),
                POSITIVE: labels.get(POSITIVE, 0),
            }
            family_counts[split] = families

        trainable = all(
            split_counts[split][NEGATIVE] > 0 and split_counts[split][POSITIVE] > 0
            for split in SPLITS
        )
        manifest["languages"][language] = {
            "route": str(route_path),
            "rows": len(records),
            "split_label_counts": split_counts,
            "split_family_counts": family_counts,
            "trainable": trainable,
            "low_positive_test_support": split_counts["test"][POSITIVE] < 30,
        }
        print(language, len(records), split_counts, route_path)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest {MANIFEST}")


if __name__ == "__main__":
    main()
