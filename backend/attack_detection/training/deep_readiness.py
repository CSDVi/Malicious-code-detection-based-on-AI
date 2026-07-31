"""Build an auditable readiness report for ByteCNN-TCN and GATv2 training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attack_detection.dataset import is_training_eligible, load_dataset


def build_readiness_report(dataset_path: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_path).resolve()
    samples = load_dataset(dataset)
    eligible = [sample for sample in samples if is_training_eligible(sample)]
    malicious_train = [sample for sample in eligible if sample.split == "train" and sample.label == "malicious"]
    malicious_train_original = [sample for sample in malicious_train if sample.review_status != "generated_variant"]
    vulnerable_train = [sample for sample in eligible if sample.split == "train" and sample.label == "vulnerable"]
    benign_train = [sample for sample in eligible if sample.split == "train" and sample.label == "benign"]
    positive_languages = Counter(sample.language for sample in malicious_train_original)
    behavior_labeled = sum(bool(sample.behavior_labels) for sample in malicious_train_original)
    cwe_labeled = sum(bool(sample.cwe_labels) for sample in vulnerable_train)
    line_labeled = sum(bool(sample.line_labels) for sample in malicious_train + vulnerable_train)
    graphs = _load_graphs(dataset.parent / "project_graphs.jsonl")
    project_graph_records = len(graphs)
    graph_splits = {str(graph.get("split") or "") for graph in graphs}
    graph_labels = Counter(str(graph.get("label") or "") for graph in graphs)
    graph_split_labels: dict[str, set[str]] = {}
    for graph in graphs:
        graph_split_labels.setdefault(str(graph.get("split") or ""), set()).add(str(graph.get("label") or ""))
    node_types = {str(value) for graph in graphs for value in graph.get("node_types", [])}
    edge_types = {str(value) for graph in graphs for value in graph.get("edge_types", [])}

    bytetcn_checks = {
        "minimum_malicious_train_1000": len(malicious_train_original) >= 1000,
        "minimum_malicious_families_50": len({sample.family for sample in malicious_train_original}) >= 50,
        "minimum_behavior_labeled_malicious_500": behavior_labeled >= 500,
        "minimum_line_labeled_samples_500": line_labeled >= 500,
        "minimum_one_language_with_1000_malicious_positives": any(count >= 1000 for count in positive_languages.values()),
        "benign_train_present": bool(benign_train),
        "vulnerable_train_present": bool(vulnerable_train),
    }
    gat_checks = {
        "minimum_project_graphs_1000": project_graph_records >= 1000,
        "graph_nodes_include_file_function_package_api": {"file", "function", "package", "dangerous_api"}.issubset(node_types),
        "graph_edges_include_call_import_dependency_version_diff": {"call", "import", "dependency", "version_diff"}.issubset(edge_types),
        "graph_labels_have_train_validation_test": {"train", "validation", "test"}.issubset(graph_splits),
        "minimum_malicious_project_graphs_500": graph_labels["malicious"] >= 500,
        "minimum_benign_project_graphs_500": graph_labels["benign"] >= 500,
        "both_labels_present_in_each_split": all(
            {"malicious", "benign"}.issubset(graph_split_labels.get(split, set()))
            for split in ("train", "validation", "test")
        ),
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "samples": len(samples),
        "training_eligible_samples": len(eligible),
        "bytetcn": {
            "ready": all(bytetcn_checks.values()),
            "checks": bytetcn_checks,
            "malicious_train_samples": len(malicious_train_original),
            "malicious_train_augmented_samples": len(malicious_train),
            "malicious_train_families": len({sample.family for sample in malicious_train_original}),
            "malicious_train_by_language": dict(sorted(positive_languages.items())),
            "supported_malicious_languages": sorted(
                language for language, count in positive_languages.items() if count >= 100
            ),
            "behavior_labeled_malicious_train": behavior_labeled,
            "cwe_labeled_vulnerable_train": cwe_labeled,
            "line_labeled_train_samples": line_labeled,
            "required_outputs": [
                "malicious_intent",
                "vulnerability_risk",
                "behavior_labels",
                "line_localization",
            ],
            "architecture_target": "ByteCNN-TCN CPU",
        },
        "gatv2": {
            "ready": all(gat_checks.values()),
            "checks": gat_checks,
            "project_graph_records": project_graph_records,
            "project_graph_labels": dict(sorted(graph_labels.items())),
            "project_graph_split_labels": {
                split: sorted(labels) for split, labels in sorted(graph_split_labels.items())
            },
            "required_node_types": ["file", "function", "package", "dangerous_api"],
            "required_edge_types": ["call", "import", "dependency", "version_diff"],
        },
    }


def _load_graphs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return output


def write_readiness_report(dataset_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    report = build_readiness_report(dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check deep-model training data readiness")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(write_readiness_report(args.dataset, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
