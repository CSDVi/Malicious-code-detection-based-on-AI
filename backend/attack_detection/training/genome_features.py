"""Shared graph schema for source/artifact software-genome models."""

from __future__ import annotations

import hashlib
import math
from typing import Any


NODE_TYPES = [
    "package",
    "file",
    "function",
    "dangerous_api",
    "binary",
    "section",
    "import_library",
    "native_api",
    "string_signal",
]
EDGE_TYPES = [
    "contains",
    "declares",
    "import",
    "dependency",
    "call",
    "version_diff",
    "has_section",
    "imports",
    "exposes_string",
]
LEXICAL_BUCKETS = 64
NUMERIC_FEATURES = 4


def feature_dimension() -> int:
    return len(NODE_TYPES) + 2 + LEXICAL_BUCKETS + NUMERIC_FEATURES


def graph_to_pyg(graph: dict[str, Any], modality: str, torch: Any) -> Any:
    """Convert one JSON graph into a PyG Data value using a stable schema."""

    from torch_geometric.data import Data

    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    if not nodes:
        raise ValueError("software-genome graph must contain at least one node")
    index = {str(node.get("id")): position for position, node in enumerate(nodes)}
    sources: list[int] = []
    targets: list[int] = []
    edge_features: list[list[float]] = []
    indegree = [0] * len(nodes)
    outdegree = [0] * len(nodes)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = index.get(str(edge.get("source")))
        target = index.get(str(edge.get("target")))
        edge_type = str(edge.get("type") or "")
        if source is None or target is None or edge_type not in EDGE_TYPES:
            continue
        encoded = [float(edge_type == value) for value in EDGE_TYPES]
        for left, right in ((source, target), (target, source)):
            sources.append(left)
            targets.append(right)
            edge_features.append(encoded)
            outdegree[left] += 1
            indegree[right] += 1
    if not sources:
        for position in range(len(nodes)):
            sources.append(position)
            targets.append(position)
            edge_features.append([0.0] * len(EDGE_TYPES))

    maximum_degree = max(1, len(nodes) - 1)
    features = []
    for position, node in enumerate(nodes):
        node_type = str(node.get("type") or "")
        lexical = _lexical_features(node)
        features.append(
            [float(node_type == value) for value in NODE_TYPES]
            + [float(modality == "source"), float(modality == "artifact")]
            + lexical
            + [
                indegree[position] / maximum_degree,
                outdegree[position] / maximum_degree,
                _bounded_entropy(node.get("entropy")),
                float(bool(node.get("suspicious"))),
            ]
        )
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
    )


def _lexical_features(node: dict[str, Any]) -> list[float]:
    raw = node.get("lexical_buckets")
    if isinstance(raw, list):
        values = [float(value) for value in raw[:LEXICAL_BUCKETS]]
        return values + [0.0] * (LEXICAL_BUCKETS - len(values))
    value = str(node.get("name") or node.get("id") or "").strip().lower()
    features = [0.0] * LEXICAL_BUCKETS
    if value:
        bucket = int(hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        features[bucket % LEXICAL_BUCKETS] = 1.0
    return features


def _bounded_entropy(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value) / 8.0))
    except (TypeError, ValueError):
        return 0.0


def graph_size_feature(graph: dict[str, Any]) -> list[float]:
    """Small graph-level feature used for debugging and dataset audits."""

    node_count = len(graph.get("nodes") or [])
    edge_count = len(graph.get("edges") or [])
    return [
        min(1.0, math.log1p(node_count) / math.log1p(1000)),
        min(1.0, math.log1p(edge_count) / math.log1p(2000)),
    ]
