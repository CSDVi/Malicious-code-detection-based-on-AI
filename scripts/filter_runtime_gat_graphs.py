"""Keep only graphs that the production GATv2 engine is able to execute."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def filter_graphs(
    source: Path,
    output: Path,
    report: Path,
    feature_schema_version: int = 1,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    minimum_nodes = 2 if feature_schema_version >= 7 else 3
    minimum_edges = 1 if feature_schema_version >= 7 else 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as input_stream, output.open(
        "w", encoding="utf-8", newline="\n",
    ) as output_stream:
        for line in input_stream:
            if not line.strip():
                continue
            graph = json.loads(line)
            key = f"{graph.get('split', 'unknown')}:{graph.get('label', 'unknown')}"
            if (
                int(graph.get("node_count") or 0) < minimum_nodes
                or int(graph.get("edge_count") or 0) < minimum_edges
            ):
                counts[f"excluded:{key}"] += 1
                continue
            output_stream.write(json.dumps(graph, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[f"included:{key}"] += 1
    result = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "runtime_minimum": {
            "node_count": minimum_nodes,
            "edge_count": minimum_edges,
            "feature_schema_version": feature_schema_version,
        },
        "counts": dict(sorted(counts.items())),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--feature-schema-version", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(
        filter_graphs(
            args.source,
            args.output,
            args.report,
            args.feature_schema_version,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
