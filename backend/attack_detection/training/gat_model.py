"""Shared GATv2 graph classifier architecture for training and inference."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class GATv2GraphClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden: int,
        heads: int,
        dropout: float,
        pooling: str = "mean",
        graph_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling if pooling in {"mean", "mean_max"} else "mean"
        self.graph_feature_dim = max(0, int(graph_feature_dim))
        self.conv1 = GATv2Conv(input_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, edge_dim=edge_dim, dropout=dropout)
        pooled_dim = hidden * 2 if self.pooling == "mean_max" else hidden
        pooled_dim += self.graph_feature_dim
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pooled_dim, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 2),
        )

    def forward(self, batch: object) -> torch.Tensor:
        value = functional.elu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        value = functional.dropout(value, p=self.dropout, training=self.training)
        value = functional.elu(self.conv2(value, batch.edge_index, batch.edge_attr))
        pooled = global_mean_pool(value, batch.batch)
        if self.pooling == "mean_max":
            pooled = torch.cat((pooled, global_max_pool(value, batch.batch)), dim=1)
        if self.graph_feature_dim:
            pooled = torch.cat((pooled, batch.graph_features), dim=1)
        return self.classifier(pooled)
