"""Build versioned package graphs for GATv2 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attack_detection.dataset import CodeSample, load_dataset
from attack_detection.features.graph_builder import build_project_graph

PACKAGE_SOURCES = {
    "npm_official_registry",
    "pypi_official_registry",
    "pypi_malregistry_ase2023",
    "pypi_popular_official",
    "codesearchnet",
    "javascript_malware_collection",
    "php_webshell_collection",
    "android_malware_source",
    "mascot_human_reviewed",
    "crossvul",
    "zenodo_13870382",
    "nist_juliet_csharp_1.3",
    "github_popular_benign_source",
    "the_stack_permissive_benign",
}


def build_graph_dataset(dataset_path: str | Path, output_path: str | Path, report_path: str | Path) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[CodeSample]] = defaultdict(list)
    for sample in load_dataset(dataset_path):
        if sample.source not in PACKAGE_SOURCES or not sample.family:
            continue
        groups[(sample.family, sample.version, sample.split)].append(sample)
    graphs = []
    for key in sorted(groups):
        graph = build_project_graph(groups[key])
        graph["content_sha256"] = _graph_hash(graph)
        graphs.append(graph)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for graph in graphs:
            stream.write(json.dumps(graph, ensure_ascii=False) + "\n")
    node_types = Counter()
    edge_types = Counter()
    for graph in graphs:
        node_types.update(graph["node_types"])
        edge_types.update(graph["edge_types"])
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(Path(dataset_path).resolve()),
        "output": str(output.resolve()),
        "graphs": len(graphs),
        "labels": dict(Counter(graph["label"] for graph in graphs)),
        "splits": dict(Counter(graph["split"] for graph in graphs)),
        "node_type_graphs": dict(node_types),
        "edge_type_graphs": dict(edge_types),
        "complete_schema_graphs": sum(
            {"file", "function", "package", "dangerous_api"}.issubset(graph["node_types"])
            and {"call", "import", "dependency", "version_diff"}.issubset(graph["edge_types"])
            for graph in graphs
        ),
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _graph_hash(graph: dict[str, Any]) -> str:
    value = json.dumps(graph, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build package dependency graphs for GATv2")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    print(json.dumps(build_graph_dataset(args.dataset, args.output, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
