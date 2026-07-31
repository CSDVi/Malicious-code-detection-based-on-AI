"""Re-evaluate an existing GATv2 artifact on language-scoped graph subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attack_detection.training.gat_trainer import (
    EDGE_TYPES,
    LEGACY_LANGUAGES,
    _language_coverage,
    _metrics,
    _predict,
    _record_languages,
    _to_data,
    feature_dimension,
    graph_feature_dimension,
)


def evaluate(graphs_path: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    from attack_detection.training.gat_model import GATv2GraphClassifier

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_dir = manifest_path.parent
    records = [
        json.loads(line) for line in graphs_path.open(encoding="utf-8") if line.strip()
    ]
    positive_label = str(manifest.get("positive_label") or "malicious")
    accepted_labels = {positive_label, "benign"}
    records_by_split = {split: [] for split in ("train", "validation", "test")}
    for record in records:
        split = str(record.get("split") or "")
        if split in records_by_split and record.get("label") in accepted_labels:
            records_by_split[split].append(record)

    languages = list(manifest.get("languages") or LEGACY_LANGUAGES)
    training = manifest["training"]
    feature_schema_version = int(training.get("feature_schema_version") or 1)
    model = GATv2GraphClassifier(
        input_dim=feature_dimension(languages, feature_schema_version),
        edge_dim=len(EDGE_TYPES), hidden=int(training["hidden"]), heads=int(training["heads"]),
        dropout=float(training["dropout"]), pooling=str(training.get("pooling") or "mean"),
        graph_feature_dim=graph_feature_dimension(feature_schema_version),
    )
    weights_name = str(
        manifest.get("artifact") or (manifest.get("files") or ["gatv2_classifier.pt"])[0]
    )
    model.load_state_dict(torch.load(model_dir / weights_name, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(1)

    test_records = records_by_split["test"]
    dataset = [
        _to_data(
            record,
            torch,
            positive_label,
            languages,
            feature_schema_version=feature_schema_version,
        )
        for record in test_records
    ]
    logits, labels = _predict(model, DataLoader(dataset, batch_size=64), torch.device("cpu"), torch)
    global_threshold = float(manifest["threshold"])
    temperature = float(manifest["temperature"])
    by_language = {}
    observed = sorted({language for record in test_records for language in _record_languages(record)})
    for language in observed:
        indices = [
            index for index, record in enumerate(test_records)
            if language in _record_languages(record)
        ]
        threshold = float(
            (manifest.get("language_thresholds") or {}).get(
                language,
                global_threshold,
            )
        )
        metrics = _metrics(
            [logits[index] for index in indices], [labels[index] for index in indices],
            threshold, temperature,
        )
        metrics["samples"] = len(indices)
        by_language[language] = metrics

    report = {
        "model_version": manifest.get("model_version"),
        "graphs_sha256": manifest.get("dataset_sha256"),
        "language_coverage": _language_coverage(records_by_split, positive_label),
        "test_metrics_by_language": by_language,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GATv2 artifact by graph language")
    parser.add_argument("--graphs", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.graphs, args.manifest, args.output), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
