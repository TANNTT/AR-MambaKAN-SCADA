"""Shared training helpers for the pooled/local AR-MambaKAN experiments (matched-budget pooling,
LOSO, calendar ablation, weather ablation). Mirrors the paper's production hyperparameters exactly:
lookback=336, epochs<=100 w/ early-stopping patience=10, min_delta=0.001, batch=128, lr=1e-3,
mape_eps=1.0, seeds {42,43,44}.

Data layout: each meter is one CSV with at least the columns `pc_time`, `pt_kw`, `qt_kvar`
(plus optionally `ua_v`, `ub_v`, `uc_v`, `cos_phi`, `freq_hz`); see docs/data_schema.md. Point
`--data-dir` at a folder of such CSVs -- by default this is the bundled `demo_data/` synthetic
set, so every script in this repository runs end-to-end without the proprietary EVN telemetry
used in the paper. Point it at your own real data (same schema) to reproduce the paper's numbers.
"""
import glob
import os
import copy
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

from features.features import CALENDAR_FEATURE_COLS, add_calendar_features
from models.model_factory import build_model, build_optimizer, default_grad_clip
from evaluation.train import (
    WindowDataset,
    split_indices,
    standardize_by_train,
    train_epoch,
    eval_epoch,
    _set_seed,
    _limit_threads,
)

LOOKBACK = 336
EPOCHS = 100
BATCH = 128
LR = 1e-3
PATIENCE = 10
MIN_DELTA = 0.001
MAPE_EPS = 1.0
NUM_THREADS = 8


def discover_meters(data_dir: str) -> dict:
    """Map meter code -> CSV path for every `<CODE>.csv` file in `data_dir`. The paper's 10-meter
    benchmark used anonymized codes such as M01 (see results/ and docs/data_schema.md); the
    bundled demo_data/ ships a handful of synthetic meters with the same naming convention so
    these scripts run unmodified on either."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def load_meter_windows(path: str, horizon: int, calendar_features: bool, lookback: int = LOOKBACK):
    df = pd.read_csv(path, parse_dates=["pc_time"])
    feat_cols = [c for c in ["pt_kw", "qt_kvar", "ua_v", "ub_v", "uc_v", "cos_phi", "freq_hz"] if c in df.columns]
    if calendar_features:
        df = add_calendar_features(df, time_col="pc_time")
        feat_cols = feat_cols + [c for c in CALENDAR_FEATURE_COLS if c in df.columns]
    arr = df[feat_cols].astype(np.float32).to_numpy()
    idx = split_indices(len(arr))
    train_arr = arr[: idx.train_end]
    val_arr = arr[idx.train_end : idx.val_end]
    test_arr = arr[idx.val_end :]
    train_arr, val_arr, test_arr, mean, std = standardize_by_train(train_arr, val_arr, test_arr)
    y_mean = torch.tensor(mean[0, :2], dtype=torch.float32).view(1, 1, 2)
    y_std = torch.tensor(std[0, :2], dtype=torch.float32).view(1, 1, 2)
    train_ds = WindowDataset(train_arr, lookback, horizon)
    val_ds = WindowDataset(val_arr, lookback, horizon)
    test_ds = WindowDataset(test_arr, lookback, horizon)
    return {
        "n_features": len(feat_cols),
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _fit(model, train_loader, val_loader, device, y_mean, y_std):
    opt = build_optimizer("armambakan", model, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    grad_clip = default_grad_clip("armambakan")
    best_val_mae = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, opt, device, grad_clip)
        va_mae, _, _, _ = eval_epoch(model, val_loader, device, y_mean.to(device), y_std.to(device), MAPE_EPS)
        scheduler.step(va_mae if math.isfinite(va_mae) else best_val_mae)
        if math.isfinite(va_mae) and va_mae < (best_val_mae - MIN_DELTA):
            best_val_mae = va_mae
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_local_armambakan(meter_path: str, horizon: int, calendar_features: bool, seed: int, device: str = "cpu"):
    _limit_threads(NUM_THREADS)
    _set_seed(seed)
    m = load_meter_windows(meter_path, horizon, calendar_features)
    train_loader = DataLoader(m["train_ds"], batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(m["val_ds"], batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(m["test_ds"], batch_size=BATCH, shuffle=False)
    model = build_model("armambakan", m["n_features"], LOOKBACK, horizon, device)
    model = _fit(model, train_loader, val_loader, device, m["y_mean"], m["y_std"])
    te_mae, _, _, _ = eval_epoch(model, test_loader, device, m["y_mean"].to(device), m["y_std"].to(device), MAPE_EPS)
    return te_mae


def train_pooled_armambakan(
    pool_codes: list,
    meters_index: dict,
    horizon: int,
    calendar_features: bool,
    seed: int,
    eval_codes: list = None,
    per_meter_train_budget: int = None,
    device: str = "cpu",
):
    """Pool training windows across `pool_codes` (each standardized with its own train stats),
    train one shared AR-MambaKAN, then evaluate separately on each of `eval_codes` (default:
    same as pool_codes) using that meter's own test split / own destandardization stats.
    `meters_index` is a {code: csv_path} map, e.g. from `discover_meters(data_dir)`."""
    _limit_threads(NUM_THREADS)
    _set_seed(seed)
    eval_codes = eval_codes or pool_codes
    meters = {c: load_meter_windows(meters_index[c], horizon, calendar_features) for c in set(pool_codes) | set(eval_codes)}
    n_features = meters[pool_codes[0]]["n_features"]

    rng = np.random.default_rng(seed)
    train_subsets = []
    val_subsets = []
    for c in pool_codes:
        tds = meters[c]["train_ds"]
        if per_meter_train_budget is not None and per_meter_train_budget < len(tds):
            chosen = rng.choice(len(tds), size=per_meter_train_budget, replace=False)
            train_subsets.append(Subset(tds, chosen.tolist()))
        else:
            train_subsets.append(tds)
        val_subsets.append(meters[c]["val_ds"])
    pooled_train = ConcatDataset(train_subsets)
    pooled_val = ConcatDataset(val_subsets)

    train_loader = DataLoader(pooled_train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(pooled_val, batch_size=BATCH, shuffle=False)

    # Pooled val evaluation needs a single (y_mean,y_std) for eval_epoch; since we only use val
    # MAE for early-stopping/scheduler comparisons (not for reporting), use the first pool meter's
    # stats -- consistent across epochs, monotonic-comparable, does not affect final per-meter
    # test evaluation, which always uses each meter's OWN y_mean/y_std below.
    ref_mean, ref_std = meters[pool_codes[0]]["y_mean"], meters[pool_codes[0]]["y_std"]
    model = build_model("armambakan", n_features, LOOKBACK, horizon, device)
    model = _fit(model, train_loader, val_loader, device, ref_mean, ref_std)

    results = {}
    for c in eval_codes:
        test_loader = DataLoader(meters[c]["test_ds"], batch_size=BATCH, shuffle=False)
        te_mae, _, _, _ = eval_epoch(
            model, test_loader, device, meters[c]["y_mean"].to(device), meters[c]["y_std"].to(device), MAPE_EPS
        )
        results[c] = te_mae
    return results, len(pooled_train)
