"""Construct forecast models and optimizers for training / profiling."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.armambakan import ARMambaKAN
from models.mamba_baseline import BaselineMamba, mamba_backend_label
from models.patchtst import BaselinePatchTST
from models.transencoder import BaselineTransformerEncoder

from models.rnn_baselines import BaselineBiLSTM, BaselineLSTM, BaselineMLP

LEARNED_MODELS = [
    "mlp",
    "lstm",
    "bilstm",
    "armambakan",
    "patchtst",
    "transencoder",
    "mamba",
]

ALL_MODELS = ["lastvalue", *LEARNED_MODELS]

# Ablation-only variants of AR-MambaKAN: isolate each component's contribution.
# Deliberately excluded from LEARNED_MODELS/ALL_MODELS so they never leak into
# the main model comparison (Table 1/2); referenced explicitly by name from
# run_ablation.py's ABLATIONS dict.
ARMAMBAKAN_ABLATION_VARIANTS = ["armambakan_nostem", "armambakan_nokan"]

# Exploratory architecture-search variants, not yet validated results: deliberately
# excluded from LEARNED_MODELS/ALL_MODELS. "mamba_stem" tests the untested design-space
# cell (local gated-conv stem in front of the *real* selective-SSM Mamba backbone,
# rather than BiLSTM) to see whether the strong lag-1 local persistence in this SCADA
# series helps a true SSM the same way it helps BiLSTM in AR-MambaKAN. "armambakan_regime"
# conditions the KAN gate on a causal recent-Q-volatility signal (see
# ARMambaKAN.use_regime_gate docstring); motivated by a measured leakage-free
# correlation between recent volatility and coming forecast error.
EXPLORATORY_VARIANTS = ["mamba_stem", "armambakan_regime"]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    model_name: str,
    in_dim: int,
    lookback: int,
    horizon: int,
    device: str | torch.device,
) -> nn.Module:
    if model_name == "mlp":
        model = BaselineMLP(in_dim=in_dim, lookback=lookback, hidden_dim=256, horizon=horizon)
    elif model_name == "lstm":
        model = BaselineLSTM(in_dim=in_dim, hidden_dim=128, horizon=horizon)
    elif model_name == "bilstm":
        model = BaselineBiLSTM(in_dim=in_dim, hidden_dim=128, horizon=horizon)
    elif model_name == "armambakan":
        model = ARMambaKAN(in_dim=in_dim, d_model=96, horizon=horizon)
    elif model_name == "armambakan_nostem":
        model = ARMambaKAN(in_dim=in_dim, d_model=96, horizon=horizon, use_stem=False, use_kan=True)
    elif model_name == "armambakan_nokan":
        model = ARMambaKAN(in_dim=in_dim, d_model=96, horizon=horizon, use_stem=True, use_kan=False)
    elif model_name == "patchtst":
        model = BaselinePatchTST(in_dim=in_dim, lookback=lookback, horizon=horizon)
    elif model_name == "transencoder":
        model = BaselineTransformerEncoder(in_dim=in_dim, lookback=lookback, horizon=horizon)
    elif model_name == "mamba":
        model = BaselineMamba(in_dim=in_dim, horizon=horizon, d_model=96, n_layers=2)
    elif model_name == "mamba_stem":
        model = BaselineMamba(in_dim=in_dim, horizon=horizon, d_model=96, n_layers=2, use_local_stem=True)
    elif model_name == "armambakan_regime":
        model = ARMambaKAN(in_dim=in_dim, d_model=96, horizon=horizon, use_stem=True, use_kan=True, use_regime_gate=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def default_grad_clip(model_name: str) -> float:
    """Per-model gradient clipping (tighter for full Mamba SSM)."""
    if model_name in ("mamba", "mamba_stem"):
        return 0.5
    if model_name in ("patchtst", "transencoder", "armambakan", "armambakan_regime", *ARMAMBAKAN_ABLATION_VARIANTS):
        return 1.0
    return 1.0


def build_optimizer(model_name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if model_name in ("mamba", "mamba_stem"):
        tuned_lr = min(lr, 2e-4)
    elif model_name in ("armambakan", "armambakan_regime", "patchtst", "transencoder", *ARMAMBAKAN_ABLATION_VARIANTS):
        tuned_lr = min(lr, 5e-4)
    else:
        tuned_lr = lr
    if model_name in (
        "armambakan", "armambakan_regime", "patchtst", "transencoder", "mamba", "mamba_stem",
        *ARMAMBAKAN_ABLATION_VARIANTS,
    ):
        return torch.optim.AdamW(model.parameters(), lr=tuned_lr, weight_decay=3e-4)
    return torch.optim.Adam(model.parameters(), lr=lr)


def model_display_name(model_name: str) -> str:
    from scada_exp.figure_labels import MODEL_LABELS

    return MODEL_LABELS.get(model_name, model_name)


def model_metadata(model_name: str, model: nn.Module | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"model": model_name, "display_name": model_display_name(model_name)}
    if model_name == "mamba":
        meta["mamba_backend"] = mamba_backend_label()
        if model is not None and isinstance(model, BaselineMamba):
            meta["mamba_uses_fallback"] = bool(model.uses_fallback)
    return meta
