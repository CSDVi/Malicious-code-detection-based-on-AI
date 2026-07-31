"""CPU-friendly byte-level CNN and temporal convolution network."""

from __future__ import annotations

from typing import Any

import torch


class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.conv(value)
        return output[:, :, : value.shape[-1]] if self.padding else output


class TemporalBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(channels, channels, kernel_size, dilation)
        self.pointwise = torch.nn.Conv1d(channels, channels, 1)
        self.norm = torch.nn.GroupNorm(1, channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.depthwise(value)
        value = self.pointwise(value)
        value = self.norm(value)
        value = torch.nn.functional.gelu(value)
        return residual + self.dropout(value)


class ByteCNNTCN(torch.nn.Module):
    def __init__(
        self,
        channels: int = 64,
        embedding_dim: int = 48,
        layers: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.2,
        behavior_labels: int = 1,
        cwe_labels: int = 1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "mean_max"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.config = {
            "channels": channels,
            "embedding_dim": embedding_dim,
            "layers": layers,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "behavior_labels": behavior_labels,
            "cwe_labels": cwe_labels,
            "pooling": pooling,
        }
        self.pooling = pooling
        self.embedding = torch.nn.Embedding(260, embedding_dim, padding_idx=0)
        self.stem = CausalConv1d(embedding_dim, channels, kernel_size=7)
        self.blocks = torch.nn.ModuleList([
            TemporalBlock(channels, kernel_size, dilation=2 ** index, dropout=dropout)
            for index in range(layers)
        ])
        self.final_norm = torch.nn.GroupNorm(1, channels)
        pooled_channels = channels * 2 if pooling == "mean_max" else channels
        self.malicious_head = torch.nn.Linear(pooled_channels, 1)
        self.vulnerability_head = torch.nn.Linear(pooled_channels, 1)
        self.behavior_head = torch.nn.Linear(pooled_channels, max(1, behavior_labels))
        self.cwe_head = torch.nn.Linear(pooled_channels, max(1, cwe_labels))
        self.line_head = torch.nn.Conv1d(channels, 1, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.embedding(input_ids).transpose(1, 2)
        value = torch.nn.functional.gelu(self.stem(value))
        for block in self.blocks:
            value = block(value)
        value = torch.nn.functional.gelu(self.final_norm(value))
        token_features = value.transpose(1, 2)
        mask = attention_mask.unsqueeze(-1)
        pooled = (token_features * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        if self.pooling == "mean_max":
            maximum = token_features.masked_fill(mask == 0, float("-inf")).amax(1)
            pooled = torch.cat((pooled, maximum), dim=-1)
        return {
            "malicious_intent": self.malicious_head(pooled).squeeze(-1),
            "vulnerability_risk": self.vulnerability_head(pooled).squeeze(-1),
            "behavior_labels": self.behavior_head(pooled),
            "cwe_labels": self.cwe_head(pooled),
            "line_localization": self.line_head(value).squeeze(1),
        }


def from_config(config: dict[str, Any]) -> ByteCNNTCN:
    return ByteCNNTCN(
        channels=int(config.get("channels", 64)),
        embedding_dim=int(config.get("embedding_dim", 48)),
        layers=int(config.get("layers", 5)),
        kernel_size=int(config.get("kernel_size", 5)),
        dropout=float(config.get("dropout", 0.2)),
        behavior_labels=int(config.get("behavior_labels", 1)),
        cwe_labels=int(config.get("cwe_labels", 1)),
        pooling=str(config.get("pooling", "mean")),
    )
