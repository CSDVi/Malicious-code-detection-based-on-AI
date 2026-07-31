"""Single-project GATv2 inference subprocess."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


def infer(graph: dict[str, object], model_dir: str | Path) -> dict[str, object]:
    import torch
    from torch_geometric.data import Batch

    from attack_detection.training.gat_model import GATv2GraphClassifier
    from attack_detection.training.gat_trainer import (
        EDGE_TYPES,
        LEGACY_LANGUAGES,
        _to_data,
        feature_dimension,
        graph_feature_dimension,
    )

    root = Path(model_dir)
    manifest = json.loads((root / "gatv2_manifest.json").read_text(encoding="utf-8"))
    supported_languages = set(manifest.get("supported_languages") or [])
    language_counts = Counter(
        str(node.get("language") or "").lower()
        for node in (graph.get("nodes") or [])
        if node.get("type") == "file" and node.get("language")
    )
    eligible_counts = {
        language: count for language, count in language_counts.items()
        if not supported_languages or language in supported_languages
    }
    if supported_languages and not eligible_counts:
        return {
            "status": "unavailable",
            "reason": "Project has no language validated by this GATv2 artifact",
            "model_version": manifest.get("model_version"),
        }
    route_language = max(
        eligible_counts, key=lambda language: (eligible_counts[language], language),
        default=None,
    )
    language_models = manifest.get("language_models") or {}
    route_settings = language_models.get(route_language) or {}
    languages = list(
        route_settings.get("languages") or manifest.get("languages") or LEGACY_LANGUAGES
    )
    graph = dict(graph)
    graph["label"] = "benign"
    training = route_settings.get("training") or manifest["training"]
    feature_schema_version = int(training.get("feature_schema_version") or 1)
    data = _to_data(
        graph,
        torch,
        languages=languages,
        feature_schema_version=feature_schema_version,
    )
    model = GATv2GraphClassifier(
        input_dim=feature_dimension(languages, feature_schema_version),
        edge_dim=len(EDGE_TYPES), hidden=int(training["hidden"]), heads=int(training["heads"]),
        dropout=float(training["dropout"]), pooling=str(training.get("pooling") or "mean"),
        graph_feature_dim=graph_feature_dimension(feature_schema_version),
    )
    weights_name = str(
        route_settings.get("file") or manifest.get("artifact")
        or (manifest.get("files") or ["gatv2_classifier.pt"])[0]
    )
    model.load_state_dict(torch.load(root / weights_name, map_location="cpu", weights_only=True))
    requested_device = os.environ.get(
        "XIEZHI_GAT_DEVICE",
        "cpu",
    ).strip().lower()
    device = torch.device(
        "cuda"
        if requested_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    model.to(device).eval()
    torch.set_num_threads(1)
    with torch.inference_mode():
        logits = model(
            Batch.from_data_list([data]).to(device)
        )[0].detach().cpu().tolist()
    temperature = float(route_settings.get("temperature", manifest["temperature"]))
    probability = _sigmoid((float(logits[1]) - float(logits[0])) / max(temperature, 1e-4))
    threshold = float(
        route_settings.get(
            "threshold",
            (manifest.get("language_thresholds") or {}).get(route_language, manifest["threshold"]),
        )
    )
    return {
        "status": "completed",
        "decision": "malicious" if probability >= threshold else "benign",
        "probability": probability,
        "threshold": threshold,
        "route_language": route_language,
        "artifact_version": route_settings.get("model_version", manifest["model_version"]),
        "model_version": manifest["model_version"],
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GATv2 project inference request")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(infer(json.loads(sys.stdin.read()), args.model_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
