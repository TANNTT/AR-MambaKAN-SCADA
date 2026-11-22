# Copied from AR-MambaKAN-SCADA @ 8582ecc937b45081f5c4116694c73d685f6c8508
# Source file: src/models/armambakan.py
# Local change: add readout={"last","full"} (default "last" matches lines 187-188).

"""AR-MambaKAN: local gated convolution + bidirectional recurrence + KAN fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from g_chung.readout import bilstm_state_readout


class KANLinear(nn.Module):
    """KAN-style layer: learnable RBF spline activations + linear base path."""

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        self.base = nn.Linear(in_dim, out_dim)
        self.spline_weight = nn.Parameter(torch.randn(in_dim, out_dim, grid_size) * (1.0 / (in_dim ** 0.5)))

        grid = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer("grid", grid.view(1, 1, grid_size))
        spacing = 2.0 / max(grid_size - 1, 1)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(spacing)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = torch.tanh(x)
        base = self.base(xb)
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3)
        dist = (xb.unsqueeze(-1) - self.grid) / sigma
        basis = torch.exp(-0.5 * dist.pow(2))
        spline = torch.einsum("big,iog->bo", basis, self.spline_weight)
        return base + spline


class GatedConvStem(nn.Module):
    """Mamba-style gated depthwise conv: local features, x * SiLU(z)."""

    def __init__(self, d_model: int, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel, padding=pad, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, z = self.in_proj(h).chunk(2, dim=-1)
        a = self.dwconv(a.transpose(1, 2)).transpose(1, 2)
        a = a * F.silu(z)
        return x + torch.tanh(self.gate) * self.dropout(self.out_proj(a))


class KANFusionHead(nn.Module):
    """BiLSTM-equivalent linear head + gated KAN refinement."""

    def __init__(
        self,
        in_dim: int,
        horizon: int,
        hidden: int = 96,
        dropout: float = 0.1,
        use_kan: bool = True,
        use_regime_gate: bool = False,
    ):
        super().__init__()
        self.horizon = horizon
        self.use_kan = use_kan
        self.use_regime_gate = use_regime_gate
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * 2),
        )
        if self.use_kan:
            self.kan_pt = KANLinear(in_dim, horizon)
            self.kan_qt = KANLinear(in_dim, horizon)
            self.alpha = nn.Parameter(torch.zeros(1))
            if self.use_regime_gate:
                self.regime_proj = nn.Linear(1, 1)
                nn.init.zeros_(self.regime_proj.weight)
                nn.init.zeros_(self.regime_proj.bias)

    def forward(self, h: torch.Tensor, regime_signal: torch.Tensor | None = None) -> torch.Tensor:
        base = self.base(h).view(h.size(0), self.horizon, 2)
        if not self.use_kan:
            return base
        kan = torch.stack([self.kan_pt(h), self.kan_qt(h)], dim=-1)
        gate_logit = self.alpha
        if self.use_regime_gate and regime_signal is not None:
            gate_logit = gate_logit + self.regime_proj(regime_signal.unsqueeze(-1))
        gate = torch.tanh(gate_logit).unsqueeze(-1)
        return base + gate * kan


class ARMambaKAN(nn.Module):
    """AR-MambaKAN with optional last/full BiLSTM readout."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 96,
        hidden: int = 96,
        horizon: int = 48,
        kan_hidden: int = 96,
        dropout: float = 0.1,
        use_stem: bool = True,
        use_kan: bool = True,
        use_regime_gate: bool = False,
        regime_window: int = 48,
        qt_channel_idx: int = 1,
        readout: str = "last",
    ):
        super().__init__()
        if readout not in ("last", "full"):
            raise ValueError(f"readout must be 'last' or 'full', got {readout!r}")
        self.horizon = horizon
        self.hidden = hidden
        self.readout = readout
        self.use_stem = use_stem
        self.use_regime_gate = use_regime_gate
        self.regime_window = regime_window
        self.qt_channel_idx = qt_channel_idx
        self.input_proj = nn.Linear(in_dim, d_model)
        if self.use_stem:
            self.stem = GatedConvStem(d_model, dropout=dropout)
        self.lstm = nn.LSTM(d_model, hidden, batch_first=True, bidirectional=True)
        self.head = KANFusionHead(
            hidden * 2,
            horizon,
            hidden=kan_hidden,
            dropout=dropout,
            use_kan=use_kan,
            use_regime_gate=use_regime_gate,
        )

    def _recent_q_volatility(self, x: torch.Tensor) -> torch.Tensor:
        q = x[:, -self.regime_window :, self.qt_channel_idx]
        return torch.diff(q, dim=1).std(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        regime_signal = self._recent_q_volatility(x) if self.use_regime_gate else None
        h = self.input_proj(x)
        if self.use_stem:
            h = self.stem(h)
        out, _ = self.lstm(h)
        last = bilstm_state_readout(out, self.hidden, self.readout)
        return self.head(last, regime_signal=regime_signal)
