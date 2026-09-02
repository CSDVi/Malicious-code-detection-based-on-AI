"""Single-project GATv2 inference subprocess."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


MAX_ATTRIBUTION_FILES = max(
    0,
    int(os.environ.get("XIEZHI_GAT_ATTRIBUTION_MAX_FILES", "24")),
)
ATTRIBUTION_BATCH_SIZE = max(
    1,
    int(os.environ.get("XIEZHI_GAT_ATTRIBUTION_BATCH_SIZE", "4")),
)


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
    temperature = float(route_settings.get("temperature", manifest["temperature"]))
    ablations = _file_component_ablations(
        graph,
        limit=MAX_ATTRIBUTION_FILES,
    )
    inference_graphs = [graph, *(item["graph"] for item in ablations)]
    logits_rows = []
    with torch.inference_mode():
        for start in range(0, len(inference_graphs), ATTRIBUTION_BATCH_SIZE):
            inference_data = [
                _to_data(
                    dict(item, label="benign"),
                    torch,
                    languages=languages,
                    feature_schema_version=feature_schema_version,
                )
                for item in inference_graphs[start:start + ATTRIBUTION_BATCH_SIZE]
            ]
            logits_rows.extend(
                model(Batch.from_data_list(inference_data).to(device))
                .detach().cpu().tolist()
            )
    probabilities = [
        _sigmoid(
            (float(logits[1]) - float(logits[0]))
            / max(temperature, 1e-4)
        )
        for logits in logits_rows
    ]
    probability = probabilities[0]
    node_attributions = _shape_attributions(
        ablations,
        baseline_probability=probability,
        ablated_probabilities=probabilities[1:],
        total_file_count=sum(
            node.get("type") == "file"
            for node in (graph.get("nodes") or [])
        ),
    )
    most_suspicious_component = next(
        (
            item for item in node_attributions
            if item.get("supports_malicious_decision")
        ),
        None,
    )
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
        "attribution_method": "leave_one_file_component_out",
        "node_attributions": node_attributions,
        "most_suspicious_component": most_suspicious_component,
        "attributed_file_count": len(node_attributions),
        "total_file_count": sum(
            node.get("type") == "file"
            for node in (graph.get("nodes") or [])
        ),
        "attribution_coverage_complete": len(node_attributions) == sum(
            node.get("type") == "file"
            for node in (graph.get("nodes") or [])
        ),
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _file_component_ablations(
    graph: dict[str, object],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Build bounded leave-one-file-out graphs without changing the model schema."""

    if limit <= 0:
        return []
    nodes = [dict(node) for node in (graph.get("nodes") or [])]
    edges = [dict(edge) for edge in (graph.get("edges") or [])]
    file_nodes = [node for node in nodes if node.get("type") == "file"]
    if len(file_nodes) <= 1:
        return []
    by_id = {str(node.get("id")): node for node in nodes}
    outgoing = Counter(str(edge.get("source")) for edge in edges)
    api_calls: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source"))
        target = by_id.get(str(edge.get("target"))) or {}
        if edge.get("type") == "call" and target.get("type") == "dangerous_api":
            api_calls[source] += 1
    ranked = sorted(
        file_nodes,
        key=lambda node: (
            -api_calls[str(node.get("id"))],
            -outgoing[str(node.get("id"))],
            -_lexical_magnitude(node.get("lexical_buckets")),
            str(node.get("name") or node.get("id") or "").casefold(),
        ),
    )[:limit]
    output = []
    for node in ranked:
        file_id = str(node.get("id"))
        removed = {file_id}
        removed.update(
            str(edge.get("target"))
            for edge in edges
            if edge.get("type") == "declares" and str(edge.get("source")) == file_id
        )
        kept_edges = [
            edge for edge in edges
            if str(edge.get("source")) not in removed
            and str(edge.get("target")) not in removed
        ]
        connected = {
            str(value)
            for edge in kept_edges
            for value in (edge.get("source"), edge.get("target"))
        }
        kept_nodes = [
            item for item in nodes
            if str(item.get("id")) not in removed
            and (
                item.get("type") not in {"function", "dangerous_api"}
                or str(item.get("id")) in connected
            )
        ]
        ablated = dict(graph)
        ablated.update({
            "nodes": kept_nodes,
            "edges": kept_edges,
            "node_count": len(kept_nodes),
            "edge_count": len(kept_edges),
            "node_types": sorted({str(item.get("type")) for item in kept_nodes}),
            "edge_types": sorted({str(item.get("type")) for item in kept_edges}),
            "label": "benign",
        })
        output.append({
            "node_id": file_id,
            "path": str(node.get("name") or file_id),
            "language": str(node.get("language") or "unknown"),
            "graph": ablated,
            "selection_signals": {
                "dangerous_api_edges": api_calls[file_id],
                "outgoing_edges": outgoing[file_id],
            },
        })
    return output


def _shape_attributions(
    ablations: list[dict[str, object]],
    *,
    baseline_probability: float,
    ablated_probabilities: list[float],
    total_file_count: int,
) -> list[dict[str, object]]:
    rows = []
    for item, probability_without in zip(ablations, ablated_probabilities):
        drop = float(baseline_probability) - float(probability_without)
        rows.append({
            "node_id": item["node_id"],
            "path": item["path"],
            "language": item["language"],
            "baseline_probability": round(float(baseline_probability), 8),
            "probability_without_component": round(float(probability_without), 8),
            "probability_drop": round(drop, 8),
            "supports_malicious_decision": drop > 0.0,
            "selection_signals": item["selection_signals"],
            "method": "leave_one_file_component_out",
            "meaning": "移除该文件及其声明函数后的项目恶意概率变化，不是漏洞概率",
            "attribution_scope": {
                "attributed_files": len(ablations),
                "total_files": total_file_count,
            },
        })
    positive_total = sum(max(0.0, float(row["probability_drop"])) for row in rows)
    for row in rows:
        row["contribution_percent"] = round(
            max(0.0, float(row["probability_drop"]))
            / max(positive_total, 1e-12)
            * 100.0,
            2,
        ) if positive_total > 0 else 0.0
    rows.sort(key=lambda row: (-float(row["probability_drop"]), str(row["path"]).casefold()))
    for index, row in enumerate(rows):
        row["rank"] = index + 1
    return rows


def _lexical_magnitude(value: object) -> float:
    if not isinstance(value, list):
        return 0.0
    return sum(abs(float(item)) for item in value if isinstance(item, (int, float)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GATv2 project inference request")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(infer(json.loads(sys.stdin.read()), args.model_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
