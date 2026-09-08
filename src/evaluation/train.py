import argparse
import copy
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from features.features import (
    CALENDAR_FEATURE_COLS,
    WEATHER_FEATURE_COLS,
    add_calendar_features,
    add_weather_features,
)
from models.model_factory import (
    ALL_MODELS,
    ARMAMBAKAN_ABLATION_VARIANTS,
    EXPLORATORY_VARIANTS,
    build_model,
    build_optimizer,
    default_grad_clip,
)


def _limit_threads(n: int) -> None:
    """Cap CPU threads so a single run never oversubscribes the machine."""
    n = max(1, n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class SplitIndex:
    train_end: int
    val_end: int


def standardize_by_train(train_arr: np.ndarray, val_arr: np.ndarray, test_arr: np.ndarray):
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
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, :2]  # pt, qt
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> SplitIndex:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return SplitIndex(train_end=train_end, val_end=val_end)


def _masked_mape_pct(err: torch.Tensor, y: torch.Tensor, eps: float) -> float:
    mask = torch.abs(y) >= eps
    if mask.sum() == 0:
        return 0.0
    return float(torch.mean(torch.abs(err[mask] / y[mask])).item() * 100.0)


def _finite_mean(values: list[float], default: float = float("nan")) -> float:
    good = [v for v in values if math.isfinite(v)]
    return float(np.mean(good)) if good else default


def train_epoch(model, loader, opt, device, grad_clip: float = 0.0):
    model.train()
    loss_fn = nn.L1Loss()
    losses = []
    for x, y in loader:
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
    model.eval()
    mae_vals = []
    rmse_vals = []
    mape_vals = []
    smape_vals = []
    for x, y in loader:
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
def eval_last_value(loader, device, y_mean: torch.Tensor, y_std: torch.Tensor, mape_eps: float):
    mae_vals = []
    rmse_vals = []
    mape_vals = []
    smape_vals = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Repeat the latest observed P,Q across forecast horizon.
        last = x[:, -1:, :2]
        p = last.repeat(1, y.shape[1], 1)
        p = p * y_std + y_mean
        y = y * y_std + y_mean
        err = p - y
        mae_vals.append(torch.mean(torch.abs(err)).item())
        rmse_vals.append(torch.sqrt(torch.mean(err * err)).item())
        mape_vals.append(_masked_mape_pct(err, y, mape_eps))
        smape_denom = torch.clamp(torch.abs(p) + torch.abs(y), min=1e-3)
        smape_vals.append(torch.mean(200.0 * torch.abs(err) / smape_denom).item())
    return (
        float(np.mean(mae_vals)),
        float(np.mean(rmse_vals)),
        float(np.mean(mape_vals)),
        float(np.mean(smape_vals)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--model",
        choices=[*ALL_MODELS, *ARMAMBAKAN_ABLATION_VARIANTS, *EXPLORATORY_VARIANTS],
        default="bilstm",
    )
    ap.add_argument("--early-stopping-patience", type=int, default=0)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--mape-eps", type=float, default=1.0, help="Min |y| (kW/kVAR) for MAPE")
    ap.add_argument("--num-threads", type=int, default=8, help="CPU thread cap to avoid oversubscription")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=-1.0,
        help="Max grad norm (0 disables; default -1 = per-model preset)",
    )
    ap.add_argument(
        "--calendar-features",
        action="store_true",
        help="Add deterministic calendar features (time-of-day/week/year, weekend, VN holiday).",
    )
    ap.add_argument(
        "--weather-csv",
        default="",
        help="Optional weather CSV (from fetch_weather) to join as exogenous wx_* features.",
    )
    ap.add_argument("--save-best", default="")
    args = ap.parse_args()
    if args.grad_clip < 0:
        args.grad_clip = default_grad_clip(args.model)

    _limit_threads(args.num_threads)
    _set_seed(args.seed)

    df = pd.read_csv(args.dataset, parse_dates=["pc_time"])
    # pt_kw, qt_kvar must stay first: targets are taken as the first two columns.
    feat_cols = [c for c in ["pt_kw", "qt_kvar", "ua_v", "ub_v", "uc_v", "cos_phi", "freq_hz"] if c in df.columns]
    if args.calendar_features:
        df = add_calendar_features(df, time_col="pc_time")
        feat_cols = feat_cols + [c for c in CALENDAR_FEATURE_COLS if c in df.columns]
    if args.weather_csv:
        df = add_weather_features(df, args.weather_csv, time_col="pc_time")
        feat_cols = feat_cols + [c for c in WEATHER_FEATURE_COLS if c in df.columns]
    arr = df[feat_cols].astype(np.float32).to_numpy()

    idx = split_indices(len(arr))
    train_arr = arr[: idx.train_end]
    val_arr = arr[idx.train_end : idx.val_end]
    test_arr = arr[idx.val_end :]
    train_arr, val_arr, test_arr, mean, std = standardize_by_train(train_arr, val_arr, test_arr)
    y_mean = torch.tensor(mean[0, :2], dtype=torch.float32).view(1, 1, 2)
    y_std = torch.tensor(std[0, :2], dtype=torch.float32).view(1, 1, 2)

    train_ds = WindowDataset(train_arr, args.lookback, args.horizon)
    val_ds = WindowDataset(val_arr, args.lookback, args.horizon)
    test_ds = WindowDataset(test_arr, args.lookback, args.horizon)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "lastvalue":
        va_mae, va_rmse, va_mape, va_smape = eval_last_value(
            val_loader, device, y_mean.to(device), y_std.to(device), args.mape_eps
        )
        te_mae, te_rmse, te_mape, te_smape = eval_last_value(
            test_loader, device, y_mean.to(device), y_std.to(device), args.mape_eps
        )
        print(
            f"val_mae={va_mae:.4f} val_rmse={va_rmse:.4f} "
            f"val_mape={va_mape:.2f}% val_smape={va_smape:.2f}%"
        )
        print(
            f"test_mae={te_mae:.4f} test_rmse={te_rmse:.4f} "
            f"test_mape={te_mape:.2f}% test_smape={te_smape:.2f}%"
        )
        return

    model = build_model(args.model, len(feat_cols), args.lookback, args.horizon, device)
    opt = build_optimizer(args.model, model, args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    no_improve_count = 0

    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, device, args.grad_clip)
        va_mae, va_rmse, va_mape, va_smape = eval_epoch(
            model, val_loader, device, y_mean.to(device), y_std.to(device), args.mape_eps
        )
        print(
            f"epoch={ep} train_mae={tr:.4f} val_mae={va_mae:.4f} val_rmse={va_rmse:.4f} "
            f"val_mape={va_mape:.2f}% val_smape={va_smape:.2f}%"
        )
        scheduler.step(va_mae if math.isfinite(va_mae) else best_val_mae)
        if not math.isfinite(va_mae):
            print(f"epoch={ep} val_metrics_invalid=true")
            no_improve_count += 1
        elif va_mae < (best_val_mae - args.min_delta):
            best_val_mae = va_mae
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if args.early_stopping_patience > 0 and no_improve_count >= args.early_stopping_patience:
            print(
                f"early_stop_triggered=true epoch={ep} "
                f"best_epoch={best_epoch} best_val_mae={best_val_mae:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored_best_model=true best_epoch={best_epoch} best_val_mae={best_val_mae:.4f}")
        if args.save_best:
            ckpt_path = args.save_best
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_name": args.model,
                    "best_epoch": best_epoch,
                    "best_val_mae": best_val_mae,
                    "lookback": args.lookback,
                    "horizon": args.horizon,
                    "feature_columns": feat_cols,
                },
                ckpt_path,
            )
            print(f"saved_best_checkpoint={ckpt_path}")

    te_mae, te_rmse, te_mape, te_smape = eval_epoch(
        model, test_loader, device, y_mean.to(device), y_std.to(device), args.mape_eps
    )
    print(
        f"test_mae={te_mae:.4f} test_rmse={te_rmse:.4f} "
        f"test_mape={te_mape:.2f}% test_smape={te_smape:.2f}%"
    )
    if not all(math.isfinite(v) for v in (te_mae, te_rmse, te_mape, te_smape)):
        print("test_metrics_invalid=true", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

