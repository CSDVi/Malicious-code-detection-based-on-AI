"""CodeT5+ encoder classifier shared by training and inference."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import T5Config, T5EncoderModel


class CodeT5PClassifier(nn.Module):
    def __init__(self, encoder: T5EncoderModel, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(encoder.config.d_model), 1)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        cache_dir: str | None = None,
        dropout: float = 0.15,
    ) -> "CodeT5PClassifier":
        encoder = T5EncoderModel.from_pretrained(checkpoint, cache_dir=cache_dir)
        return cls(encoder, dropout=dropout)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | T5Config,
        *,
        dropout: float = 0.15,
    ) -> "CodeT5PClassifier":
        parsed = config if isinstance(config, T5Config) else T5Config.from_dict(config)
        return cls(T5EncoderModel(parsed), dropout=dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits, pooled

