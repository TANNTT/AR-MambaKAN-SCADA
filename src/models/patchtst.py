"""PatchTST-style baseline (multivariate patches + Transformer encoder)."""

from __future__ import annotations

import torch
from torch import nn


class BaselinePatchTST(nn.Module):
    """Channel-mixed patching over the lookback window (PatchTST-inspired)."""

    def __init__(
        self,
        in_dim: int,
        lookback: int,
        horizon: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        n_patches = (lookback - patch_len) // stride + 1
        if n_patches < 1:
            raise ValueError(f"lookback={lookback} too small for patch_len={patch_len}")

        patch_dim = patch_len * in_dim
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))
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
            nn.LayerNorm(d_model * n_patches),
            nn.Linear(d_model * n_patches, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * 2),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, n_patches, patch_len * D]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches.reshape(patches.size(0), patches.size(1), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(self._patchify(x)) + self.pos_emb
        h = self.encoder(h)
        h = h.reshape(h.size(0), -1)
        return self.head(h).view(x.size(0), self.horizon, 2)
