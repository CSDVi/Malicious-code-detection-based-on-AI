"""Create language-scoped GATv2 graph datasets from an existing graph JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def graph_languages(record: dict[str, object]) -> set[str]:
    return {
        str(node.get("language") or "").strip().lower()
        for node in (record.get("nodes") or [])
        if isinstance(node, dict) and node.get("type") == "file" and node.get("language")
    }


def build(source: Path, output_root: Path, languages: list[str]) -> dict[str, object]:
    requested = {value.strip().lower() for value in languages if value.strip()}
    if not requested:
        raise ValueError("at least one language is required")
    output_root.mkdir(parents=True, exist_ok=True)
    streams = {
        language: (output_root / f"gatv2_{language}_graphs.jsonl").open("w", encoding="utf-8")
        for language in sorted(requested)
    }
    counts = {language: Counter() for language in requested}
    try:
        with source.open("r", encoding="utf-8") as input_stream:
            for line in input_stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                matched = requested.intersection(graph_languages(record))
                if not matched:
                    continue
                rendered = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for language in matched:
                    streams[language].write(rendered)
                    counts[language][
                        f"{record.get('split', 'unknown')}:{record.get('label', 'unknown')}"
                    ] += 1
    finally:
        for stream in streams.values():
            stream.close()

    report = {
        "source": str(source.resolve()),
        "outputs": {
            language: {
                "path": str((output_root / f"gatv2_{language}_graphs.jsonl").resolve()),
                "counts": dict(sorted(values.items())),
                "graphs": sum(values.values()),
            }
            for language, values in sorted(counts.items())
        },
    }
    (output_root / "gatv2_language_graphs_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build language-scoped GATv2 graph JSONL files")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--languages", required=True, nargs="+")
    args = parser.parse_args()
    print(json.dumps(
        build(args.source.resolve(), args.output_root.resolve(), args.languages),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
