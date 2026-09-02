"""Siamese GATv2 architecture for source-to-artifact integrity verification."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class GenomeGraphEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden: int = 96,
        heads: int = 4,
        dropout: float = 0.2,
        embedding_dim: int = 96,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(
            input_dim,
            hidden,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.conv2 = GATv2Conv(
            hidden * heads,
            hidden,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, embedding_dim),
        )

    def forward(self, batch: object) -> torch.Tensor:
        value = functional.elu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        value = functional.dropout(value, p=self.dropout, training=self.training)
        value = functional.elu(self.conv2(value, batch.edge_index, batch.edge_attr))
        pooled = torch.cat(
            (global_mean_pool(value, batch.batch), global_max_pool(value, batch.batch)),
            dim=1,
        )
        return functional.normalize(self.projection(pooled), p=2, dim=1)


class TwinGATVerifier(torch.nn.Module):
    """Encode both modalities with shared weights and predict tampering."""

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden: int = 96,
        heads: int = 4,
        dropout: float = 0.2,
        embedding_dim: int = 96,
    ) -> None:
        super().__init__()
        self.encoder = GenomeGraphEncoder(
            input_dim,
            edge_dim,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2 + 1, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, source_batch: object, artifact_batch: object) -> torch.Tensor:
        source = self.encoder(source_batch)
        artifact = self.encoder(artifact_batch)
        cosine = functional.cosine_similarity(source, artifact, dim=1).unsqueeze(1)
        pair_features = torch.cat((torch.abs(source - artifact), source * artifact, cosine), dim=1)
        return self.classifier(pair_features).squeeze(1)
