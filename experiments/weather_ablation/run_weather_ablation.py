"""Weather-feature ablation, AR-MambaKAN, local training, H=48 (24h). Compares calendar-only
against calendar+ERA5-weather inputs (Table 18 / Section 7.9 of the paper -- a scoped negative
result: adding weather on top of calendar features does not improve, and can slightly harm, the
strongest baselines).

Weather CSVs are matched by meter code: for meter `<CODE>.csv` in --data-dir, this script looks for
`<CODE>.csv` in --weather-dir (one row per timestamp, joined to the nearest SCADA timestamp; see
`features.features.add_weather_features` and `features.fetch_weather` for the ERA5 fetch/join
pipeline). Meters without a matching weather file are skipped with a warning rather than failing
the whole run, since ERA5 fetch requires internet access and is not bundled with this repository.

Usage:
    PYTHONPATH=../../src:../pooled python run_weather_ablation.py \
        --data-dir ../../demo_data --weather-dir ../../demo_data/weather \
        --out-csv ../../results/weather_ablation_demo.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pooled"))
import common_pooled as cp

from features.features import CALENDAR_FEATURE_COLS, WEATHER_FEATURE_COLS, add_calendar_features, add_weather_features
from evaluation.train import WindowDataset, split_indices, standardize_by_train, eval_epoch, _set_seed, _limit_threads
from models.model_factory import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "demo_data"))
    ap.add_argument("--weather-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "demo_data", "weather"))
    ap.add_argument("--out-csv", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "weather_ablation.csv"))
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meters = cp.discover_meters(args.data_dir)
    rows = []
    for code, path in meters.items():
        wx_csv = os.path.join(args.weather_dir, f"{code}.csv")
        if not os.path.exists(wx_csv):
            print(f"[weather] no weather file for {code} at {wx_csv}, skipping", flush=True)
            continue

        _limit_threads(cp.NUM_THREADS)
        _set_seed(args.seed)
        df = pd.read_csv(path, parse_dates=["pc_time"])
        feat_cols = [c for c in ["pt_kw", "qt_kvar", "ua_v", "ub_v", "uc_v", "cos_phi", "freq_hz"] if c in df.columns]
        df = add_calendar_features(df, time_col="pc_time")
        feat_cols = feat_cols + [c for c in CALENDAR_FEATURE_COLS if c in df.columns]
        df = add_weather_features(df, wx_csv, time_col="pc_time")
        feat_cols = feat_cols + [c for c in WEATHER_FEATURE_COLS if c in df.columns]
        arr = df[feat_cols].astype(np.float32).to_numpy()
        idx = split_indices(len(arr))
        train_arr, val_arr, test_arr = arr[: idx.train_end], arr[idx.train_end: idx.val_end], arr[idx.val_end:]
        train_arr, val_arr, test_arr, mean, std = standardize_by_train(train_arr, val_arr, test_arr)
        y_mean = torch.tensor(mean[0, :2], dtype=torch.float32).view(1, 1, 2)
        y_std = torch.tensor(std[0, :2], dtype=torch.float32).view(1, 1, 2)
        train_ds = WindowDataset(train_arr, cp.LOOKBACK, args.horizon)
        val_ds = WindowDataset(val_arr, cp.LOOKBACK, args.horizon)
        test_ds = WindowDataset(test_arr, cp.LOOKBACK, args.horizon)
        train_loader = DataLoader(train_ds, batch_size=cp.BATCH, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cp.BATCH, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=cp.BATCH, shuffle=False)
        model = build_model("armambakan", len(feat_cols), cp.LOOKBACK, args.horizon, args.device)
        t0 = time.time()
        model = cp._fit(model, train_loader, val_loader, args.device, y_mean, y_std)
        te_mae, _, _, _ = eval_epoch(model, test_loader, args.device, y_mean.to(args.device), y_std.to(args.device), cp.MAPE_EPS)
        dt = time.time() - t0
        print(f"[weather] {code} mae={te_mae:.2f} ({dt:.0f}s)", flush=True)
        rows.append({"code": code, "seed": args.seed, "horizon": args.horizon, "mae": te_mae})

    if not rows:
        print("No meters had a matching weather file -- nothing to do. "
              "See features/fetch_weather.py to build demo_data/weather/<CODE>.csv files.")
        return
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"DONE weather ablation experiment, saved {args.out_csv}")


if __name__ == "__main__":
    main()
