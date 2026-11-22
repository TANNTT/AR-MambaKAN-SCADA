# Copied from AR-MambaKAN-SCADA @ 8582ecc937b45081f5c4116694c73d685f6c8508
# Source files: src/evaluation/train.py (split_indices, WindowDataset, standardize,
# train_epoch, eval_epoch) and experiments/pooled/common_pooled.py (hyperparams, pooling).
# Local additions: TimedWindowDataset, predict_destandardized, fit returning best_epoch.

"""Training primitives for G-chung (local and pooled)."""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Paper hyperparameters (common_pooled.py:33-40 @ 8582ecc)
LOOKBACK = 336
EPOCHS = 100
BATCH = 128
LR = 1e-3
PATIENCE = 10
MIN_DELTA = 0.001
MAPE_EPS = 1.0
NUM_THREADS = 8
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
SEEDS = (42, 43, 44)
HORIZONS = (4, 48, 96, 336)
STEP_NS = int(30 * 60 * 1e9)  # 30-minute SCADA step in int64 ns


def limit_threads(n: int = NUM_THREADS) -> None:
    n = max(1, n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class SplitIndex:
    train_end: int
    val_end: int


def split_indices(n: int, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO) -> SplitIndex:
    # train.py:84-87 @ 8582ecc
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return SplitIndex(train_end=train_end, val_end=val_end)


def standardize_by_train(train_arr: np.ndarray, val_arr: np.ndarray, test_arr: np.ndarray):
    # train.py:56-66 @ 8582ecc
    mean = np.mean(train_arr, axis=0, keepdims=True)
    std = np.std(train_arr, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        (train_arr - mean) / std,
        (val_arr - mean) / std,
        (test_arr - mean) / std,
        mean,
        std,
    )


class WindowDataset(Dataset):
    # train.py:69-81 @ 8582ecc
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, :2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TimedWindowDataset(Dataset):
    """WindowDataset plus origin/target timestamps (int64 ns) for prediction export."""

    def __init__(self, data: np.ndarray, times_ns: np.ndarray, lookback: int, horizon: int):
        if len(data) != len(times_ns):
            raise ValueError("data and times_ns length mismatch")
        self.data = data
        self.times_ns = times_ns.astype(np.int64, copy=False)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, :2]
        t_origin = self.times_ns[idx + self.lookback - 1]
        t_target = self.times_ns[idx + self.lookback : idx + self.lookback + self.horizon]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(t_origin, dtype=torch.int64),
            torch.tensor(t_target, dtype=torch.int64),
        )


def _finite_mean(values: list[float], default: float = float("nan")) -> float:
    good = [v for v in values if math.isfinite(v)]
    return float(np.mean(good)) if good else default


def _masked_mape_pct(err: torch.Tensor, y: torch.Tensor, eps: float) -> float:
    mask = torch.abs(y) >= eps
    if mask.sum() == 0:
        return 0.0
    return float(torch.mean(torch.abs(err[mask] / y[mask])).item() * 100.0)


def train_epoch(model, loader, opt, device, grad_clip: float = 0.0):
    # train.py:102-120 @ 8582ecc
    model.train()
    loss_fn = nn.L1Loss()
    losses = []
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        if not torch.isfinite(pred).all():
            continue
        loss = loss_fn(pred, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(loss.item())
    return _finite_mean(losses)


@torch.no_grad()
def eval_epoch(model, loader, device, y_mean: torch.Tensor, y_std: torch.Tensor, mape_eps: float):
    # train.py:123-148 @ 8582ecc (mean of per-batch means)
    model.eval()
    mae_vals = []
    rmse_vals = []
    mape_vals = []
    smape_vals = []
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        p = model(x)
        if not torch.isfinite(p).all():
            continue
        p = p * y_std + y_mean
        y = y * y_std + y_mean
        err = p - y
        mae_vals.append(torch.mean(torch.abs(err)).item())
        rmse_vals.append(torch.sqrt(torch.mean(err * err)).item())
        mape_vals.append(_masked_mape_pct(err, y, mape_eps))
        smape_denom = torch.clamp(torch.abs(p) + torch.abs(y), min=1e-3)
        smape_vals.append(torch.mean(200.0 * torch.abs(err) / smape_denom).item())
    return (
        _finite_mean(mae_vals),
        _finite_mean(rmse_vals),
        _finite_mean(mape_vals),
        _finite_mean(smape_vals),
    )


@torch.no_grad()
def predict_destandardized(model, loader, device, y_mean: torch.Tensor, y_std: torch.Tensor):
    """Collect destandardized predictions/targets and timestamps from TimedWindowDataset."""
    model.eval()
    preds, trues, origins, targets = [], [], [], []
    for x, y, t_origin, t_target in loader:
        x = x.to(device)
        p = model(x)
        if not torch.isfinite(p).all():
            raise RuntimeError("non-finite prediction encountered during export")
        p = p * y_std + y_mean
        y = y.to(device) * y_std + y_mean
        preds.append(p.cpu().numpy().astype(np.float32))
        trues.append(y.cpu().numpy().astype(np.float32))
        origins.append(t_origin.numpy().astype(np.int64))
        targets.append(t_target.numpy().astype(np.int64))
    if not preds:
        raise RuntimeError("empty prediction loader")
    return {
        "pred": np.concatenate(preds, axis=0),
        "y": np.concatenate(trues, axis=0),
        "t_origin": np.concatenate(origins, axis=0),
        "t_target": np.concatenate(targets, axis=0),
    }


def mae_from_arrays(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Micro-average MAE (joint / P / Q). Verifiable from saved npz."""
    err = np.abs(pred.astype(np.float64) - y.astype(np.float64))
    return {
        "mae_joint": float(err.mean()),
        "mae_p": float(err[:, :, 0].mean()),
        "mae_q": float(err[:, :, 1].mean()),
    }


def fit_model(
    model,
    train_loader,
    val_loader,
    device,
    y_mean,
    y_std,
    model_name: str,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    min_delta: float = MIN_DELTA,
):
    """Mirror common_pooled._fit (lines 79-100) with best_epoch tracking from train.py."""
    from g_chung.models.model_factory import build_optimizer, default_grad_clip

    opt = build_optimizer(model_name, model, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    grad_clip = default_grad_clip(model_name)
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    stopped_epoch = 0
    for ep in range(1, epochs + 1):
        train_epoch(model, train_loader, opt, device, grad_clip)
        va_mae, _, _, _ = eval_epoch(
            model, val_loader, device, y_mean.to(device), y_std.to(device), MAPE_EPS
        )
        scheduler.step(va_mae if math.isfinite(va_mae) else best_val_mae)
        stopped_epoch = ep
        if math.isfinite(va_mae) and va_mae < (best_val_mae - min_delta):
            best_val_mae = va_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_mae": float(best_val_mae) if math.isfinite(best_val_mae) else None,
    }
