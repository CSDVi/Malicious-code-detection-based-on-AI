"""Summarize v18 per-language XGBoost candidate runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/xgb_multilingual_optimization_20260727/evidence"
SUMMARY_GLOB = "v18_all_language_training_summary_20260727T*.json"
OUT_JSON = EVIDENCE / "v18_all_language_best_candidates_latest.json"
OUT_CSV = EVIDENCE / "v18_all_language_best_candidates_latest.csv"


def _rank(result: dict[str, Any]) -> tuple[bool, float, float, float, float]:
    return (
        bool(result.get("quality_gate_passed")),
        -float(result.get("metric_deficit", 999.0)),
        float(result.get("f1") or 0.0),
        float(result.get("precision") or 0.0),
        float(result.get("recall") or 0.0),
    )


def main() -> None:
    summaries = sorted(EVIDENCE.glob(SUMMARY_GLOB))
    if not summaries:
        raise SystemExit(f"no summaries matched {SUMMARY_GLOB}")

    all_results: list[dict[str, Any]] = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = summary.get("run_id") or summary_path.stem.rsplit("_", 1)[-1]
        for result in summary.get("results", []):
            item = dict(result)
            item["run_id"] = run_id
            item["summary"] = str(summary_path)
            all_results.append(item)

    best: dict[str, dict[str, Any]] = {}
    for result in all_results:
        if result.get("returncode") != 0:
            continue
        language = str(result["language"])
        current = best.get(language)
        if current is None or _rank(result) > _rank(current):
            best[language] = result

    rows = []
    for language, result in sorted(best.items()):
        rows.append({
            "language": language,
            "pass": bool(result.get("quality_gate_passed")),
            "config": result.get("config"),
            "run_id": result.get("run_id"),
            "accuracy": result.get("accuracy"),
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "false_positive_rate": result.get("false_positive_rate"),
            "false_negative_rate": result.get("false_negative_rate"),
            "f1": result.get("f1"),
            "metric_deficit": result.get("metric_deficit"),
            "low_positive_test_support": result.get("low_positive_test_support"),
            "model": result.get("model"),
            "metrics": result.get("metrics"),
        })

    output = {
        "summaries": [str(path) for path in summaries],
        "candidate_results": len(all_results),
        "best_by_language": {row["language"]: row for row in rows},
        "passed_languages": [row["language"] for row in rows if row["pass"]],
        "failed_languages": [row["language"] for row in rows if not row["pass"]],
    }
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "output_json": str(OUT_JSON),
        "output_csv": str(OUT_CSV),
        "candidate_results": len(all_results),
        "passed_languages": output["passed_languages"],
        "failed_languages": output["failed_languages"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
