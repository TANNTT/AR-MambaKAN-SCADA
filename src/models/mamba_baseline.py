"""Mamba (selective SSM) baseline when `mamba_ssm` is installed.

Falls back to a stacked gated-convolution stem (same family as AR-MambaKAN stem)
plus BiLSTM when the CUDA-optimized `mamba_ssm` package is unavailable — useful
for CPU-only smoke tests but not a substitute for the full SSM baseline in the paper.
"""

from __future__ import annotations

import torch
from torch import nn

from models.armambakan import GatedConvStem

try:
    from mamba_ssm import Mamba

    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class BaselineMamba(nn.Module):
    def __init__(
        self,
        in_dim: int,
        horizon: int,
        d_model: int = 96,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_fallback: bool = False,
        use_local_stem: bool = False,
    ):
        super().__init__()
        self.horizon = horizon
        self.uses_fallback = use_fallback or not HAS_MAMBA_SSM
        self.input_proj = nn.Linear(in_dim, d_model)

        self.drop = nn.Dropout(dropout)
        if not self.uses_fallback:
            # `use_local_stem`: prepend a gated depthwise-conv local-persistence stem
            # (same module as AR-MambaKAN's) before the real selective-SSM blocks.
            # Untested combination otherwise: the paper's Mamba baseline is bare SSM,
            # and the gated-conv stem is only ever tested in front of a BiLSTM.
            self.use_local_stem = use_local_stem
            if use_local_stem:
                self.local_stem = GatedConvStem(d_model, dropout=dropout)
            self.blocks = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)]
            )
            self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)
        else:
            self.use_local_stem = False
            self.stem = nn.Sequential(
                GatedConvStem(d_model, dropout=dropout),
                GatedConvStem(d_model, dropout=dropout),
            )
            self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)

        head_in = d_model * 2 if self.uses_fallback else d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        if self.uses_fallback:
            h = self.stem(h)
            out, _ = self.lstm(h)
            z = out[:, -1, :]
        else:
            if self.use_local_stem:
                h = self.local_stem(h)
            for norm, block in zip(self.block_norms, self.blocks):
                residual = h
                h = norm(h)
                h = block(h)
                h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
                h = residual + self.drop(h)
            h = self.norm(h)
            z = h[:, -1, :]
        return self.head(z).view(x.size(0), self.horizon, 2)


def mamba_backend_label() -> str:
    return "mamba_ssm" if HAS_MAMBA_SSM else "gated_conv_fallback"
