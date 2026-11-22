# Copied/simplified from AR-MambaKAN-SCADA @ 8582ecc937b45081f5c4116694c73d685f6c8508
# Source file: src/models/model_factory.py
# Kept G-chung builders; added readout kwarg; added ablation variants A1–A6.

"""Construct forecast models and optimizers for G-chung / G2b / G3 / G4."""

from __future__ import annotations

import torch
from torch import nn

from .armambakan import ARMambaKAN
from .rnn_baselines import BaselineBiLSTM, BaselineLSTM, BaselineMLP

# Models whose BiLSTM state is affected by readout={"last","full"}.
BILSTM_READOUT_MODELS = frozenset({"bilstm", "armambakan", "armambakan_nostem", "armambakan_nokan"})
G_CHUNG_MODELS = frozenset({"bilstm", "armambakan"})
ALL_TRAINABLE = frozenset(
    {"mlp", "lstm", "bilstm", "armambakan", "armambakan_nostem", "armambakan_nokan"}
)
ARMAMBAKAN_FAMILY = frozenset({"armambakan", "armambakan_nostem", "armambakan_nokan"})


def build_model(
    model_name: str,
    in_dim: int,
    lookback: int,
    horizon: int,
    device: str | torch.device,
    readout: str = "last",
) -> nn.Module:
    if model_name == "mlp":
        model = BaselineMLP(in_dim=in_dim, lookback=lookback, hidden_dim=256, horizon=horizon, readout=readout)
    elif model_name == "lstm":
        model = BaselineLSTM(in_dim=in_dim, hidden_dim=128, horizon=horizon, readout=readout)
    elif model_name == "bilstm":
        model = BaselineBiLSTM(in_dim=in_dim, hidden_dim=128, horizon=horizon, readout=readout)
    elif model_name == "armambakan":
        model = ARMambaKAN(in_dim=in_dim, d_model=96, horizon=horizon, readout=readout)
    elif model_name == "armambakan_nostem":
        # model_factory.py:65-66 @ 8582ecc
        model = ARMambaKAN(
            in_dim=in_dim, d_model=96, horizon=horizon, use_stem=False, use_kan=True, readout=readout
        )
    elif model_name == "armambakan_nokan":
        # model_factory.py:67-68 @ 8582ecc
        model = ARMambaKAN(
            in_dim=in_dim, d_model=96, horizon=horizon, use_stem=True, use_kan=False, readout=readout
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def default_grad_clip(model_name: str) -> float:
    # Source: model_factory.py:84-90 @ 8582ecc
    if model_name in ARMAMBAKAN_FAMILY:
        return 1.0
    return 1.0


def build_optimizer(model_name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    # Source: model_factory.py:93-105 @ 8582ecc
    if model_name in ARMAMBAKAN_FAMILY:
        tuned_lr = min(lr, 5e-4)
        return torch.optim.AdamW(model.parameters(), lr=tuned_lr, weight_decay=3e-4)
    if model_name in ("bilstm", "lstm", "mlp"):
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unknown model for optimizer: {model_name}")
