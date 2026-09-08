"""Standard Transformer-encoder baseline for multivariate SCADA windows."""

from __future__ import annotations

import torch
from torch import nn


class BaselineTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        lookback: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_emb[:, : x.size(1), :]
        h = self.encoder(h)
        return self.head(h[:, -1, :]).view(x.size(0), self.horizon, 2)
