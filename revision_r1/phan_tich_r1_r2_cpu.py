#!/usr/bin/env python3
"""R1.1 / R1.5 / R2 CPU analysis from saved pred_*.npz (read-only results).

Assumptions (checked at start; abort if violated):
1. Each results/g-chung|g4/<run>/pred_<M>.npz has pred_{val,test}, y_{val,test},
   t_origin_{val,test}, t_target_{val,test}; values are destandardized original units;
   channel 0=P (kW), 1=Q (kvar).
2. Recomputed test MAE from npz matches summary per_meter[M].test_mae (rel < 1e-3).
3. Daily-naive on the same test windows as models, rule matching kiem_tai_lap.py:
   with window start i and lookback L=336, y_hat[j] = series[i+L+(j%48)-48].
   Equivalently with NPZ origin index o = i+L-1 (last lookback point):
   y_hat[j] = series[o+1+(j%48)-48]. Mean joint MAE over 10 meters at H=48 must be
   within 1% of 390.45 (Table 17 / results/kiem-tai-lap).
4. No exact clock 00:00 origins in SCADA stamps (~:03/:33). For R2 "one target once"
   at H=48, select per calendar day the window whose first target is the earliest
   sample that day (k=1..48 covers ~24h forward). Documented in summary.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from g_chung.data import MAIN_METERS, ROLLING_FOLDS, resolve_meter_index
from g_chung.train_core import LOOKBACK, mae_from_arrays

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "r1-r2-cpu"
G_CHUNG = ROOT / "results" / "g-chung"
G4 = ROOT / "results" / "g4"
SEEDS = (42, 43, 44)
METERS = tuple(MAIN_METERS)
MODELS = ("bilstm", "armambakan")
REGIMES = ("pooled", "local")
READOUTS = ("last", "full")
HORIZONS = (4, 48, 96, 336)
DAILY_PERIOD = 48
DAILY_NAIVE_H48_REF = 390.45
MAPE_FLOOR_FRAC = 0.05
MIN_SEASON_DAYS = 21
CHANNELS = ("joint", "P", "Q")
MODEL_CFGS = [
    ("pooled", "bilstm"),
    ("pooled", "armambakan"),
    ("local", "bilstm"),
    ("local", "armambakan"),
]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
    except Exception:
        return "unknown"


def season_of_month(month: int) -> str:
    if month in (12, 1, 2):
        return "dong"
    if month in (3, 4):
        return "xuan"
    if month in (5, 6, 7, 8):
        return "he"
    return "thu"


def second_week_wednesday(year: int, month: int) -> pd.Timestamp:
    """Wednesday of days 8..14 in the given month."""
    for day in range(8, 15):
        ts = pd.Timestamp(year=year, month=month, day=day)
        if ts.dayofweek == 2:  # Wednesday
            return ts.normalize()
    raise RuntimeError(f"no Wednesday in days 8-14 of {year}-{month:02d}")


def second_week_sunday(year: int, month: int) -> pd.Timestamp:
    wed = second_week_wednesday(year, month)
    return (wed + pd.Timedelta(days=4)).normalize()


class SeriesCache:
    """Read-only meter PQ series + timestamps (anonymized labels only)."""

    def __init__(self) -> None:
        self._paths = resolve_meter_index(use_demo=False, meters=list(METERS))
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, meter: str) -> tuple[np.ndarray, np.ndarray]:
        if meter not in self._data:
            path = self._paths[meter]
            df = pd.read_csv(path, usecols=["pc_time", "pt_kw", "qt_kvar"], parse_dates=["pc_time"])
            df = (
                df.dropna(subset=["pc_time", "pt_kw", "qt_kvar"])
                .sort_values("pc_time")
                .reset_index(drop=True)
            )
            arr = df[["pt_kw", "qt_kvar"]].to_numpy(np.float64)
            times = df["pc_time"].to_numpy("datetime64[ns]").astype(np.int64)
            self._data[meter] = (arr, times)
        return self._data[meter]


def run_dir_gchung(model: str, regime: str, readout: str, horizon: int, seed: int, meter: str | None) -> Path:
    if regime == "pooled":
        return G_CHUNG / f"{model}_{regime}_{readout}_h{horizon}_s{seed}"
    return G_CHUNG / f"{model}_{regime}_{readout}_h{horizon}_s{seed}_{meter}"


def run_dir_g4(
    model: str, regime: str, horizon: int, seed: int, fold: str, meter: str | None
) -> Path:
    if regime == "pooled":
        return G4 / f"{model}_{regime}_last_h{horizon}_s{seed}_{fold}"
    return G4 / f"{model}_{regime}_last_h{horizon}_s{seed}_{meter}_{fold}"


def load_npz(path: Path, meter: str) -> dict[str, np.ndarray]:
    z = np.load(path / f"pred_{meter}.npz")
    return {k: z[k] for k in z.files}


def verify_mae_samples() -> list[dict]:
    samples = [
        G_CHUNG / "armambakan_local_full_h48_s42_M01",
        G_CHUNG / "bilstm_pooled_last_h48_s42",
        G_CHUNG / "armambakan_pooled_full_h336_s43",
        G4 / "bilstm_local_last_h48_s42_M05_F2",
        G4 / "armambakan_pooled_last_h336_s42_F3",
    ]
    rows = []
    for run in samples:
        sm = json.loads((run / "summary.json").read_text())
        for pred_path in sorted(run.glob("pred_*.npz")):
            meter = pred_path.stem.replace("pred_", "")
            z = np.load(pred_path)
            keys = list(z.files)
            required = {
                "pred_test",
                "y_test",
                "t_origin_test",
                "t_target_test",
                "pred_val",
                "y_val",
                "t_origin_val",
                "t_target_val",
            }
            if not required.issubset(keys):
                raise RuntimeError(f"{pred_path}: missing keys {required - set(keys)}")
            recomputed = mae_from_arrays(z["pred_test"], z["y_test"])
            reported = sm["per_meter"][meter]["test_mae"]
            rel = {
                k: abs(recomputed[k] - reported[k]) / max(abs(reported[k]), 1e-12)
                for k in ("mae_joint", "mae_p", "mae_q")
            }
            ok = all(v < 1e-3 for v in rel.values())
            rows.append(
                {
                    "run": run.name,
                    "meter": meter,
                    "keys": keys,
                    "pred_test_shape": list(z["pred_test"].shape),
                    "y_range_joint": [
                        float(np.min(z["y_test"])),
                        float(np.max(z["y_test"])),
                    ],
                    "recomputed": recomputed,
                    "reported": reported,
                    "rel_diff": rel,
                    "ok": ok,
                }
            )
            if not ok:
                raise RuntimeError(f"MAE mismatch {run.name} {meter}: {rel}")
    return rows


def daily_naive_pred_from_origin(
    series: np.ndarray, times: np.ndarray, t_origin: np.ndarray, horizon: int
) -> np.ndarray:
    """Daily-naive predictions aligned to NPZ windows (kiem_tai_lap rule)."""
    o = np.searchsorted(times, t_origin)
    if not np.array_equal(times[o], t_origin):
        n_miss = int(np.sum(times[o] != t_origin))
        raise RuntimeError(f"t_origin not found in series ({n_miss} misses); stop, do not guess")
    preds = np.empty((len(o), horizon, 2), dtype=np.float64)
    for j in range(horizon):
        preds[:, j, :] = series[o + 1 + (j % DAILY_PERIOD) - DAILY_PERIOD]
    return preds


def mae_channels(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.abs(pred.astype(np.float64) - y.astype(np.float64))
    return {
        "joint": float(err.mean()),
        "P": float(err[:, :, 0].mean()),
        "Q": float(err[:, :, 1].mean()),
    }


def mean_abs_y(y: np.ndarray) -> dict[str, float]:
    y64 = y.astype(np.float64)
    return {
        "joint": float(np.mean(np.abs(y64))),
        "P": float(np.mean(np.abs(y64[:, :, 0]))),
        "Q": float(np.mean(np.abs(y64[:, :, 1]))),
    }


def select_daily_window_indices(t_target: np.ndarray) -> tuple[np.ndarray, dict]:
    """One window per calendar day: earliest first-target that day.

    No exact 00:00 origins exist; this covers ~24h from the day's first sample.
    """
    t0 = pd.to_datetime(t_target[:, 0])
    day = t0.normalize()
    # stable: first occurrence of min timestamp per day
    order = np.argsort(t_target[:, 0], kind="mergesort")
    chosen = []
    seen = set()
    for i in order:
        d = day[i]
        key = d.to_datetime64()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(int(i))
    idx = np.array(sorted(chosen), dtype=np.int64)
    first_targets = pd.to_datetime(t_target[idx, 0]) if len(idx) else pd.DatetimeIndex([])
    meta = {
        "rule": (
            "no exact 00:00 origin; per calendar day take window with earliest t_target[:,0] "
            "(first sample of that day); H=48 steps cover ~24h forward"
        ),
        "n_days": int(len(idx)),
        "selected_first_target_hour_median": (
            float(pd.Series(first_targets.hour).median()) if len(idx) else None
        ),
        "first_target_time_sample": str(first_targets[0]) if len(idx) else None,
        "window_stride_note": "underlying windows are mostly 30-min stride (step 1 in samples)",
    }
    return idx, meta


def wilcoxon_wins(model_by_meter: dict[str, float], base_by_meter: dict[str, float]) -> dict:
    meters = [m for m in METERS if m in model_by_meter and m in base_by_meter]
    diff = np.array([base_by_meter[m] - model_by_meter[m] for m in meters], dtype=np.float64)
    wins = int(np.sum([model_by_meter[m] < base_by_meter[m] for m in meters]))
    # positive diff => model better (lower MAE)
    if np.allclose(diff, 0):
        return {
            "n": len(meters),
            "n_model_wins": wins,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "note": "all differences zero",
        }
    try:
        stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        return {
            "n": len(meters),
            "n_model_wins": wins,
            "wilcoxon_stat": float(stat),
            "wilcoxon_pvalue": float(p),
        }
    except ValueError as e:
        return {
            "n": len(meters),
            "n_model_wins": wins,
            "wilcoxon_stat": None,
            "wilcoxon_pvalue": None,
            "note": str(e),
        }


def seed_mean_mae_gchung(
    model: str, regime: str, readout: str, horizon: int, meter: str
) -> dict[str, float]:
    vals = {"joint": [], "P": [], "Q": []}
    for seed in SEEDS:
        run = run_dir_gchung(model, regime, readout, horizon, seed, meter if regime == "local" else None)
        sm = json.loads((run / "summary.json").read_text())
        tm = sm["per_meter"][meter]["test_mae"]
        vals["joint"].append(tm["mae_joint"])
        vals["P"].append(tm["mae_p"])
        vals["Q"].append(tm["mae_q"])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def seed_mean_mae_g4(
    model: str, regime: str, horizon: int, fold: str, meter: str
) -> dict[str, float]:
    vals = {"joint": [], "P": [], "Q": []}
    for seed in SEEDS:
        run = run_dir_g4(model, regime, horizon, seed, fold, meter if regime == "local" else None)
        sm = json.loads((run / "summary.json").read_text())
        tm = sm["per_meter"][meter]["test_mae"]
        vals["joint"].append(tm["mae_joint"])
        vals["P"].append(tm["mae_p"])
        vals["Q"].append(tm["mae_q"])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def seed_mean_pred_gchung(
    model: str, regime: str, readout: str, horizon: int, meter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    y_ref = None
    t_o = t_t = None
    for seed in SEEDS:
        run = run_dir_gchung(model, regime, readout, horizon, seed, meter if regime == "local" else None)
        z = load_npz(run, meter)
        preds.append(z["pred_test"].astype(np.float64))
        if y_ref is None:
            y_ref = z["y_test"].astype(np.float64)
            t_o = z["t_origin_test"]
            t_t = z["t_target_test"]
        else:
            if not np.allclose(y_ref, z["y_test"].astype(np.float64), rtol=1e-5, atol=1e-3):
                raise RuntimeError(f"y_test mismatch across seeds for {run.name}")
    return np.mean(np.stack(preds, axis=0), axis=0), y_ref, t_o, t_t


def seed_mean_pred_g4(
    model: str, regime: str, horizon: int, fold: str, meter: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds = []
    y_ref = None
    t_o = t_t = None
    for seed in SEEDS:
        run = run_dir_g4(model, regime, horizon, seed, fold, meter if regime == "local" else None)
        z = load_npz(run, meter)
        preds.append(z["pred_test"].astype(np.float64))
        if y_ref is None:
            y_ref = z["y_test"].astype(np.float64)
            t_o = z["t_origin_test"]
            t_t = z["t_target_test"]
        else:
            if not np.allclose(y_ref, z["y_test"].astype(np.float64), rtol=1e-5, atol=1e-3):
                raise RuntimeError(f"y_test mismatch across seeds for {run.name}")
    return np.mean(np.stack(preds, axis=0), axis=0), y_ref, t_o, t_t


def masked_mape(pred: np.ndarray, y: np.ndarray, floor: float) -> tuple[float, int, int]:
    """MAPE% on |y|>=floor; returns (mape_pct, n_used, n_excluded)."""
    y64 = y.astype(np.float64)
    e = np.abs(pred.astype(np.float64) - y64)
    mask = np.abs(y64) >= floor
    n_ex = int((~mask).sum())
    n_ok = int(mask.sum())
    if n_ok == 0:
        return float("nan"), 0, n_ex
    return float(np.mean(e[mask] / np.abs(y64[mask])) * 100.0), n_ok, n_ex


def ratio_sum(pred: np.ndarray, y: np.ndarray) -> float:
    y64 = y.astype(np.float64)
    denom = np.sum(np.abs(y64))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(pred.astype(np.float64) - y64)) / denom)


def cfg_name(regime: str, model: str) -> str:
    return f"{regime}_{model}"


def build_r11_gchung(series_cache: SeriesCache) -> dict:
    """R1.1 for G-chung: all horizons, channels, readouts."""
    daily_by = {}  # (H, meter, ch) -> mae ; also store mean_abs for R1.5
    mean_abs_by = {}  # (H, meter, ch) -> mean|y|
    daily_check = {}

    # daily naive once per horizon/meter from any reference npz (pooled last s42)
    for H in HORIZONS:
        meter_maes = []
        for meter in METERS:
            ref = run_dir_gchung("bilstm", "pooled", "last", H, 42, None)
            z = load_npz(ref, meter)
            series, times = series_cache.get(meter)
            dpred = daily_naive_pred_from_origin(series, times, z["t_origin_test"], H)
            y = z["y_test"]
            # sanity: y matches series
            o = np.searchsorted(times, z["t_origin_test"])
            y_from = np.stack([series[o[i] + 1 : o[i] + 1 + H] for i in range(len(o))])
            if not np.allclose(y.astype(np.float64), y_from, rtol=1e-4, atol=1e-2):
                raise RuntimeError(f"y_test vs series mismatch {meter} H={H}")
            m = mae_channels(dpred, y)
            may = mean_abs_y(y)
            for ch in CHANNELS:
                daily_by[(H, meter, ch)] = m[ch]
                mean_abs_by[(H, meter, ch)] = may[ch]
            meter_maes.append(m["joint"])
        mean_joint = float(np.mean(meter_maes))
        daily_check[H] = {
            "mean_joint_10meters": mean_joint,
            "per_meter_joint": {m: daily_by[(H, m, "joint")] for m in METERS},
        }
        if H == 48:
            rel = abs(mean_joint - DAILY_NAIVE_H48_REF) / DAILY_NAIVE_H48_REF
            daily_check[H]["ref_390_45"] = DAILY_NAIVE_H48_REF
            daily_check[H]["rel_diff_vs_ref"] = rel
            if rel > 0.01:
                raise RuntimeError(
                    f"daily naive H48 mean {mean_joint} differs >1% from {DAILY_NAIVE_H48_REF}"
                )

    rows = []
    for H in HORIZONS:
        for readout in READOUTS:
            for ch in CHANNELS:
                base = {m: daily_by[(H, m, ch)] for m in METERS}
                cell = {
                    "horizon": H,
                    "readout": readout,
                    "channel": ch,
                    "daily_naive_mean": float(np.mean(list(base.values()))),
                    "daily_naive_per_meter": base,
                    "models": {},
                }
                for regime, model in MODEL_CFGS:
                    per = {}
                    for meter in METERS:
                        per[meter] = seed_mean_mae_gchung(model, regime, readout, H, meter)[ch]
                    cell["models"][cfg_name(regime, model)] = {
                        "mean": float(np.mean(list(per.values()))),
                        "per_meter": per,
                        "vs_daily_naive": wilcoxon_wins(per, base),
                    }
                rows.append(cell)
    return {"daily_naive_check": daily_check, "cells": rows, "mean_abs_y": {
        f"H{H}|{m}|{ch}": mean_abs_by[(H, m, ch)]
        for H in HORIZONS
        for m in METERS
        for ch in CHANNELS
    }}


def build_r11_g4(series_cache: SeriesCache) -> dict:
    rows = []
    daily_check = {}
    for fold in ("F2", "F3"):
        for H in (48, 336):
            base = {}
            for meter in METERS:
                ref = run_dir_g4("bilstm", "pooled", H, 42, fold, None)
                z = load_npz(ref, meter)
                series, times = series_cache.get(meter)
                dpred = daily_naive_pred_from_origin(series, times, z["t_origin_test"], H)
                base[meter] = mae_channels(dpred, z["y_test"])
            daily_check[f"{fold}|H{H}"] = {
                "mean_joint": float(np.mean([base[m]["joint"] for m in METERS])),
                "per_meter_joint": {m: base[m]["joint"] for m in METERS},
            }
            for ch in CHANNELS:
                base_ch = {m: base[m][ch] for m in METERS}
                cell = {
                    "fold": fold,
                    "horizon": H,
                    "readout": "last",
                    "channel": ch,
                    "daily_naive_mean": float(np.mean(list(base_ch.values()))),
                    "daily_naive_per_meter": base_ch,
                    "models": {},
                }
                for regime, model in MODEL_CFGS:
                    per = {
                        m: seed_mean_mae_g4(model, regime, H, fold, m)[ch] for m in METERS
                    }
                    cell["models"][cfg_name(regime, model)] = {
                        "mean": float(np.mean(list(per.values()))),
                        "per_meter": per,
                        "vs_daily_naive": wilcoxon_wins(per, base_ch),
                    }
                rows.append(cell)
    return {"daily_naive_check": daily_check, "cells": rows}


def build_r15(r11: dict) -> dict:
    """Rankings on MAE / nMAE / MASE from R1.1 cells + mean_abs_y."""
    mean_abs = r11["mean_abs_y"]
    rankings = []
    rank_flip_notes = []
    for cell in r11["cells"]:
        H = cell["horizon"]
        readout = cell["readout"]
        ch = cell["channel"]
        # per-meter metrics for ranking configs
        cfg_meter_mae = {"daily_naive": cell["daily_naive_per_meter"]}
        for name, md in cell["models"].items():
            cfg_meter_mae[name] = md["per_meter"]

        def agg(metric_by_cfg_meter):
            out = {}
            for cfg, per in metric_by_cfg_meter.items():
                vals = [per[m] for m in METERS]
                out[cfg] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "per_meter": per,
                }
            # rank by mean ascending
            order = sorted(out.keys(), key=lambda c: out[c]["mean"])
            for i, c in enumerate(order, 1):
                out[c]["rank_by_mean"] = i
            return out, order

        mae_tbl, mae_order = agg(cfg_meter_mae)

        # nMAE and MASE only meaningful per channel; for joint use joint mean|y|
        nmae_by = {}
        mase_by = {}
        for cfg, per in cfg_meter_mae.items():
            nmae_by[cfg] = {}
            mase_by[cfg] = {}
            for m in METERS:
                denom = mean_abs[f"H{H}|{m}|{ch}"]
                nmae_by[cfg][m] = per[m] / denom if denom > 0 else float("nan")
                base = cell["daily_naive_per_meter"][m]
                mase_by[cfg][m] = per[m] / base if base > 0 else float("nan")
        nmae_tbl, nmae_order = agg(nmae_by)
        mase_tbl, mase_order = agg(mase_by)

        flip = mae_order != nmae_order
        rankings.append(
            {
                "horizon": H,
                "readout": readout,
                "channel": ch,
                "mae": mae_tbl,
                "nmae": nmae_tbl,
                "mase_vs_daily_naive": mase_tbl,
                "rank_order_mae": mae_order,
                "rank_order_nmae": nmae_order,
                "rank_order_mase": mase_order,
                "rank_flip_mae_vs_nmae": flip,
            }
        )
        if flip:
            rank_flip_notes.append(
                {
                    "horizon": H,
                    "readout": readout,
                    "channel": ch,
                    "mae_order": mae_order,
                    "nmae_order": nmae_order,
                }
            )

    # cross-channel rank comparison for same H/readout
    cross = []
    by_hr = defaultdict(list)
    for r in rankings:
        by_hr[(r["horizon"], r["readout"])].append(r)
    for (H, readout), group in sorted(by_hr.items()):
        orders = {r["channel"]: r["rank_order_mae"] for r in group}
        cross.append(
            {
                "horizon": H,
                "readout": readout,
                "mae_rank_orders": orders,
                "joint_vs_P_same": orders.get("joint") == orders.get("P"),
                "joint_vs_Q_same": orders.get("joint") == orders.get("Q"),
                "P_vs_Q_same": orders.get("P") == orders.get("Q"),
            }
        )
    return {"rankings": rankings, "rank_flips_mae_vs_nmae": rank_flip_notes, "cross_channel": cross}


def flatten_points(pred_w: np.ndarray, y_w: np.ndarray, t_w: np.ndarray):
    """Windows (N,H,2) + targets (N,H) -> point arrays."""
    return (
        pred_w.reshape(-1, 2),
        y_w.reshape(-1, 2),
        t_w.reshape(-1),
    )


def channel_slice(pred, y, ch: str):
    if ch == "P":
        return pred[:, 0], y[:, 0]
    if ch == "Q":
        return pred[:, 1], y[:, 1]
    raise ValueError(ch)


def build_r2_gchung(series_cache: SeriesCache) -> dict:
    """Season / week / day analysis on G-chung H=48, last and full."""
    day_rule_meta = None
    seasons_out = []
    weeks_out = []
    days_out = []
    rep_days_out = []
    friedman_out = []
    wilcoxon_season_out = []

    rep_specs = [
        ("2022-10", second_week_wednesday(2022, 10)),
        ("2023-01", second_week_wednesday(2023, 1)),
        ("2023-04", second_week_wednesday(2023, 4)),
    ]

    for readout in READOUTS:
        # collect per-meter daily-window arrays for each config
        for meter in METERS:
            series, times = series_cache.get(meter)
            # reference windows from pooled bilstm
            ref_pred, y, t_o, t_t = seed_mean_pred_gchung("bilstm", "pooled", readout, 48, meter)
            day_idx, meta = select_daily_window_indices(t_t)
            if day_rule_meta is None:
                day_rule_meta = meta

            dpred_full = daily_naive_pred_from_origin(series, times, t_o, 48)
            cfg_preds = {"daily_naive": dpred_full}
            for regime, model in MODEL_CFGS:
                pred, y2, t_o2, t_t2 = seed_mean_pred_gchung(model, regime, readout, 48, meter)
                if not np.array_equal(t_o2, t_o):
                    raise RuntimeError(f"origin mismatch {meter} {regime} {model}")
                cfg_preds[cfg_name(regime, model)] = pred
                if not np.allclose(y2, y, rtol=1e-5, atol=1e-3):
                    raise RuntimeError(f"y mismatch {meter} {regime} {model}")

            # restrict to daily windows
            y_d = y[day_idx]
            t_d = t_t[day_idx]
            cfg_d = {k: v[day_idx] for k, v in cfg_preds.items()}

            # mean |y| on full test (for MAPE floor) per channel
            floors = {
                "P": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 0]))),
                "Q": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 1]))),
            }

            # --- seasons ---
            t_flat = t_d.reshape(-1)
            months = pd.to_datetime(t_flat).month.to_numpy()
            season_labels = np.array([season_of_month(int(m)) for m in months])
            # days count: unique calendar days of targets in season
            day_of_point = pd.to_datetime(t_flat).normalize()
            for season in ("dong", "xuan", "he", "thu"):
                mask_pts = season_labels == season
                n_days = int(pd.Series(day_of_point[mask_pts]).nunique()) if mask_pts.any() else 0
                it_mau = n_days < MIN_SEASON_DAYS
                for ch in ("P", "Q"):
                    row = {
                        "readout": readout,
                        "meter": meter,
                        "season": season,
                        "channel": ch,
                        "n_days": n_days,
                        "it_mau": it_mau,
                        "mape_floor": floors[ch],
                        "configs": {},
                    }
                    for cfg, pred_w in cfg_d.items():
                        p_flat, y_flat, _ = flatten_points(pred_w, y_d, t_d)
                        pc, yc = channel_slice(p_flat, y_flat, ch)
                        pc_s, yc_s = pc[mask_pts], yc[mask_pts]
                        if len(yc_s):
                            mae = float(np.mean(np.abs(pc_s - yc_s)))
                            mask = np.abs(yc_s) >= floors[ch]
                            n_ex = int((~mask).sum())
                            n_ok = int(mask.sum())
                            mape = (
                                float(
                                    np.mean(np.abs(pc_s[mask] - yc_s[mask]) / np.abs(yc_s[mask]))
                                    * 100.0
                                )
                                if n_ok
                                else float("nan")
                            )
                            rsum = float(
                                np.sum(np.abs(pc_s - yc_s)) / max(np.sum(np.abs(yc_s)), 1e-12)
                            )
                        else:
                            mae, mape, n_ok, n_ex, rsum = (
                                float("nan"),
                                float("nan"),
                                0,
                                0,
                                float("nan"),
                            )
                        row["configs"][cfg] = {
                            "mae": mae,
                            "mape_pct": mape,
                            "mape_n_used": n_ok,
                            "mape_n_excluded": n_ex,
                            "ratio_sum_abs_e_over_abs_y": rsum,
                        }
                    seasons_out.append(row)

            # --- weeks ---
            t_pts = pd.to_datetime(t_flat)
            iso = t_pts.isocalendar()
            week_id = (
                iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
            ).to_numpy()
            for week in sorted(set(week_id.tolist())):
                mask_pts = week_id == week
                seas = season_of_month(int(pd.to_datetime(t_flat[mask_pts][0]).month))
                for ch in ("P", "Q"):
                    wrow = {
                        "readout": readout,
                        "meter": meter,
                        "iso_week": week,
                        "season": seas,
                        "channel": ch,
                        "configs": {},
                    }
                    for cfg, pred_w in cfg_d.items():
                        p_flat, y_flat, _ = flatten_points(pred_w, y_d, t_d)
                        pc, yc = channel_slice(p_flat, y_flat, ch)
                        pc_s, yc_s = pc[mask_pts], yc[mask_pts]
                        mae = float(np.mean(np.abs(pc_s - yc_s)))
                        mask = np.abs(yc_s) >= floors[ch]
                        n_ok = int(mask.sum())
                        mape = (
                            float(np.mean(np.abs(pc_s[mask] - yc_s[mask]) / np.abs(yc_s[mask])) * 100.0)
                            if n_ok
                            else float("nan")
                        )
                        wrow["configs"][cfg] = {"mae": mae, "mape_pct": mape, "mape_n_used": n_ok}
                    weeks_out.append(wrow)

            # --- fixed calendar days (Wed + Sun of second week each month in test) ---
            # discover months present in daily windows
            day_starts = pd.to_datetime(t_d[:, 0]).normalize()
            months_present = sorted({(ts.year, ts.month) for ts in day_starts})
            for year, month in months_present:
                for kind, ts in (
                    ("wed", second_week_wednesday(year, month)),
                    ("sun", second_week_sunday(year, month)),
                ):
                    # find daily window whose first target date == ts
                    hits = np.where(day_starts == ts)[0]
                    drow = {
                        "readout": readout,
                        "meter": meter,
                        "kind": kind,
                        "date": str(ts.date()),
                        "year": year,
                        "month": month,
                        "found": bool(len(hits)),
                        "configs": {},
                    }
                    if not len(hits):
                        days_out.append(drow)
                        continue
                    wi = int(hits[0])
                    for ch in ("P", "Q"):
                        for cfg, pred_w in cfg_d.items():
                            pc = pred_w[wi, :, 0 if ch == "P" else 1]
                            yc = y_d[wi, :, 0 if ch == "P" else 1]
                            mae = float(np.mean(np.abs(pc - yc)))
                            mask = np.abs(yc) >= floors[ch]
                            n_ok = int(mask.sum())
                            mape = (
                                float(np.mean(np.abs(pc[mask] - yc[mask]) / np.abs(yc[mask])) * 100.0)
                                if n_ok
                                else float("nan")
                            )
                            drow["configs"].setdefault(cfg, {})[ch] = {
                                "mae": mae,
                                "mape_pct": mape,
                                "mape_n_used": n_ok,
                                "mape_n_excluded": int((~mask).sum()),
                            }
                    days_out.append(drow)

            # representative three Wednesdays
            for label, ts in rep_specs:
                hits = np.where(day_starts == ts)[0]
                rrow = {
                    "readout": readout,
                    "meter": meter,
                    "label": label,
                    "date": str(ts.date()),
                    "found": bool(len(hits)),
                    "configs": {},
                }
                if len(hits):
                    wi = int(hits[0])
                    for ch in ("P", "Q"):
                        for cfg, pred_w in cfg_d.items():
                            pc = pred_w[wi, :, 0 if ch == "P" else 1]
                            yc = y_d[wi, :, 0 if ch == "P" else 1]
                            mae = float(np.mean(np.abs(pc - yc)))
                            mask = np.abs(yc) >= floors[ch]
                            n_ok = int(mask.sum())
                            mape = (
                                float(np.mean(np.abs(pc[mask] - yc[mask]) / np.abs(yc[mask])) * 100.0)
                                if n_ok
                                else float("nan")
                            )
                            rrow["configs"].setdefault(cfg, {})[ch] = {
                                "mae": mae,
                                "mape_pct": mape,
                                "mape_n_used": n_ok,
                            }
                rep_days_out.append(rrow)

        # Friedman / Wilcoxon after all meters for this readout (P and Q, using season MAE)
        for ch in ("P", "Q"):
            # season MAE matrix meters x seasons for each cfg
            for cfg in ["daily_naive"] + [cfg_name(r, m) for r, m in MODEL_CFGS]:
                mat = {s: [] for s in ("dong", "xuan", "he", "thu")}
                meters_ok = []
                for meter in METERS:
                    vals = []
                    ok = True
                    for season in ("dong", "xuan", "he", "thu"):
                        rows = [
                            r
                            for r in seasons_out
                            if r["readout"] == readout
                            and r["meter"] == meter
                            and r["season"] == season
                            and r["channel"] == ch
                        ]
                        if not rows or rows[0]["it_mau"] or not np.isfinite(rows[0]["configs"][cfg]["mae"]):
                            ok = False
                            break
                        vals.append(rows[0]["configs"][cfg]["mae"])
                    if ok:
                        meters_ok.append(meter)
                        for season, v in zip(("dong", "xuan", "he", "thu"), vals):
                            mat[season].append(v)
                if len(meters_ok) >= 3:
                    try:
                        stat, p = friedmanchisquare(
                            mat["dong"], mat["xuan"], mat["he"], mat["thu"]
                        )
                        friedman_out.append(
                            {
                                "readout": readout,
                                "channel": ch,
                                "config": cfg,
                                "n_meters": len(meters_ok),
                                "meters": meters_ok,
                                "stat": float(stat),
                                "pvalue": float(p),
                            }
                        )
                    except ValueError as e:
                        friedman_out.append(
                            {
                                "readout": readout,
                                "channel": ch,
                                "config": cfg,
                                "n_meters": len(meters_ok),
                                "error": str(e),
                            }
                        )

            for season in ("dong", "xuan", "he", "thu"):
                for regime_pair in (("pooled_bilstm", "local_bilstm"), ("pooled_armambakan", "local_armambakan")):
                    a, b = regime_pair
                    pa, pb = {}, {}
                    for meter in METERS:
                        rows = [
                            r
                            for r in seasons_out
                            if r["readout"] == readout
                            and r["meter"] == meter
                            and r["season"] == season
                            and r["channel"] == ch
                        ]
                        if not rows or rows[0]["it_mau"]:
                            continue
                        pa[meter] = rows[0]["configs"][a]["mae"]
                        pb[meter] = rows[0]["configs"][b]["mae"]
                    if len(pa) >= 3:
                        # wilcoxon: does pooled beat local? (lower MAE)
                        wilcoxon_season_out.append(
                            {
                                "readout": readout,
                                "channel": ch,
                                "season": season,
                                "pooled": a,
                                "local": b,
                                **wilcoxon_wins(pa, pb),
                                # here base=local, model=pooled for "wins"
                                "note": "n_model_wins = pooled MAE < local MAE",
                            }
                        )

    # days-per-season summary table
    days_per_cell = []
    for readout in READOUTS:
        for season in ("dong", "xuan", "he", "thu"):
            for meter in METERS:
                rows = [
                    r
                    for r in seasons_out
                    if r["readout"] == readout
                    and r["season"] == season
                    and r["meter"] == meter
                    and r["channel"] == "P"
                ]
                if rows:
                    days_per_cell.append(
                        {
                            "readout": readout,
                            "season": season,
                            "meter": meter,
                            "n_days": rows[0]["n_days"],
                            "it_mau": rows[0]["it_mau"],
                        }
                    )

    return {
        "horizon": 48,
        "daily_window_rule": day_rule_meta,
        "seasons": seasons_out,
        "days_per_season_meter": days_per_cell,
        "weeks": weeks_out,
        "fixed_calendar_days": days_out,
        "representative_wednesdays": rep_days_out,
        "friedman_seasons": friedman_out,
        "wilcoxon_pooled_vs_local_by_season": wilcoxon_season_out,
    }


def build_r2_g4(series_cache: SeriesCache) -> dict:
    """Same-calendar-day 2022 (G4) vs 2023 (G-chung), plus F3 summer and load-shift stats."""
    # calendar compare dates
    compare = {
        "F2": [
            ("2022-01", second_week_wednesday(2022, 1)),
            ("2022-04", second_week_wednesday(2022, 4)),
        ],
        "F3": [
            ("2022-05", second_week_wednesday(2022, 5)),
            ("2022-06", second_week_wednesday(2022, 6)),
            ("2022-07", second_week_wednesday(2022, 7)),
            ("2022-08", second_week_wednesday(2022, 8)),
        ],
    }
    gchung_2023 = [
        ("2023-01", second_week_wednesday(2023, 1)),
        ("2023-04", second_week_wednesday(2023, 4)),
    ]

    daily_compare = []
    # G4 days
    for fold, specs in compare.items():
        for meter in METERS:
            series, times = series_cache.get(meter)
            pred_ref, y, t_o, t_t = seed_mean_pred_g4("bilstm", "pooled", 48, fold, meter)
            day_idx, _ = select_daily_window_indices(t_t)
            dpred = daily_naive_pred_from_origin(series, times, t_o, 48)
            cfg_preds = {"daily_naive": dpred}
            for regime, model in MODEL_CFGS:
                pred, y2, t_o2, _ = seed_mean_pred_g4(model, regime, 48, fold, meter)
                if not np.array_equal(t_o2, t_o):
                    raise RuntimeError("g4 origin mismatch")
                cfg_preds[cfg_name(regime, model)] = pred
            y_d = y[day_idx]
            t_d = t_t[day_idx]
            cfg_d = {k: v[day_idx] for k, v in cfg_preds.items()}
            day_starts = pd.to_datetime(t_d[:, 0]).normalize()
            floors = {
                "P": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 0]))),
                "Q": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 1]))),
            }
            for label, ts in specs:
                hits = np.where(day_starts == ts)[0]
                row = {
                    "source": f"g4_{fold}",
                    "meter": meter,
                    "label": label,
                    "date": str(ts.date()),
                    "found": bool(len(hits)),
                    "configs": {},
                }
                if len(hits):
                    wi = int(hits[0])
                    for ch in ("P", "Q"):
                        for cfg, pred_w in cfg_d.items():
                            pc = pred_w[wi, :, 0 if ch == "P" else 1]
                            yc = y_d[wi, :, 0 if ch == "P" else 1]
                            mask = np.abs(yc) >= floors[ch]
                            n_ok = int(mask.sum())
                            row["configs"].setdefault(cfg, {})[ch] = {
                                "mae": float(np.mean(np.abs(pc - yc))),
                                "mape_pct": (
                                    float(
                                        np.mean(np.abs(pc[mask] - yc[mask]) / np.abs(yc[mask])) * 100.0
                                    )
                                    if n_ok
                                    else float("nan")
                                ),
                            }
                daily_compare.append(row)

    # G-chung 2023 same months
    for meter in METERS:
        series, times = series_cache.get(meter)
        pred_ref, y, t_o, t_t = seed_mean_pred_gchung("bilstm", "pooled", "last", 48, meter)
        day_idx, _ = select_daily_window_indices(t_t)
        dpred = daily_naive_pred_from_origin(series, times, t_o, 48)
        cfg_preds = {"daily_naive": dpred}
        for regime, model in MODEL_CFGS:
            pred, _, t_o2, _ = seed_mean_pred_gchung(model, regime, "last", 48, meter)
            cfg_preds[cfg_name(regime, model)] = pred
        y_d = y[day_idx]
        t_d = t_t[day_idx]
        cfg_d = {k: v[day_idx] for k, v in cfg_preds.items()}
        day_starts = pd.to_datetime(t_d[:, 0]).normalize()
        floors = {
            "P": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 0]))),
            "Q": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 1]))),
        }
        for label, ts in gchung_2023:
            hits = np.where(day_starts == ts)[0]
            row = {
                "source": "gchung_last",
                "meter": meter,
                "label": label,
                "date": str(ts.date()),
                "found": bool(len(hits)),
                "configs": {},
            }
            if len(hits):
                wi = int(hits[0])
                for ch in ("P", "Q"):
                    for cfg, pred_w in cfg_d.items():
                        pc = pred_w[wi, :, 0 if ch == "P" else 1]
                        yc = y_d[wi, :, 0 if ch == "P" else 1]
                        mask = np.abs(yc) >= floors[ch]
                        n_ok = int(mask.sum())
                        row["configs"].setdefault(cfg, {})[ch] = {
                            "mae": float(np.mean(np.abs(pc - yc))),
                            "mape_pct": (
                                float(
                                    np.mean(np.abs(pc[mask] - yc[mask]) / np.abs(yc[mask])) * 100.0
                                )
                                if n_ok
                                else float("nan")
                            ),
                        }
            daily_compare.append(row)

    # F3 full summer season metrics
    summer = []
    wilcoxon_summer = []
    for meter in METERS:
        series, times = series_cache.get(meter)
        _, y, t_o, t_t = seed_mean_pred_g4("bilstm", "pooled", 48, "F3", meter)
        day_idx, _ = select_daily_window_indices(t_t)
        dpred = daily_naive_pred_from_origin(series, times, t_o, 48)
        cfg_preds = {"daily_naive": dpred}
        for regime, model in MODEL_CFGS:
            pred, _, _, _ = seed_mean_pred_g4(model, regime, 48, "F3", meter)
            cfg_preds[cfg_name(regime, model)] = pred
        y_d = y[day_idx]
        t_d = t_t[day_idx]
        cfg_d = {k: v[day_idx] for k, v in cfg_preds.items()}
        t_flat = t_d.reshape(-1)
        months = pd.to_datetime(t_flat).month.to_numpy()
        # F3 is May-Aug = he entirely
        floors = {
            "P": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 0]))),
            "Q": MAPE_FLOOR_FRAC * float(np.mean(np.abs(y[:, :, 1]))),
        }
        n_days = int(pd.to_datetime(t_d[:, 0]).normalize().nunique())
        for ch in ("P", "Q"):
            row = {
                "meter": meter,
                "season": "he",
                "fold": "F3",
                "channel": ch,
                "n_days": n_days,
                "configs": {},
            }
            for cfg, pred_w in cfg_d.items():
                p_flat, y_flat, _ = flatten_points(pred_w, y_d, t_d)
                pc, yc = channel_slice(p_flat, y_flat, ch)
                mask = np.abs(yc) >= floors[ch]
                n_ok = int(mask.sum())
                row["configs"][cfg] = {
                    "mae": float(np.mean(np.abs(pc - yc))),
                    "mape_pct": (
                        float(np.mean(np.abs(pc[mask] - yc[mask]) / np.abs(yc[mask])) * 100.0)
                        if n_ok
                        else float("nan")
                    ),
                    "mape_n_used": n_ok,
                    "mape_n_excluded": int((~mask).sum()),
                }
            summer.append(row)

    for ch in ("P", "Q"):
        for a, b in (("pooled_bilstm", "local_bilstm"), ("pooled_armambakan", "local_armambakan")):
            pa, pb = {}, {}
            for meter in METERS:
                rows = [r for r in summer if r["meter"] == meter and r["channel"] == ch]
                pa[meter] = rows[0]["configs"][a]["mae"]
                pb[meter] = rows[0]["configs"][b]["mae"]
            wilcoxon_summer.append(
                {
                    "channel": ch,
                    "pooled": a,
                    "local": b,
                    **wilcoxon_wins(pa, pb),
                    "note": "n_model_wins = pooled < local",
                }
            )

    # load-shift stats: mean daily load Wed week-2 of Jan/Apr/Oct across 2020,2021,2022
    load_shift = []
    for meter in METERS:
        series, times = series_cache.get(meter)
        ts = pd.to_datetime(times)
        df = pd.DataFrame({"t": ts, "P": series[:, 0], "Q": series[:, 1]})
        df["date"] = df["t"].dt.normalize()
        per_year = {}
        for year in (2020, 2021, 2022):
            year_rows = {}
            for month in (1, 4, 10):
                wed = second_week_wednesday(year, month)
                day = df[df["date"] == wed]
                year_rows[f"{month:02d}"] = {
                    "n_points": int(len(day)),
                    "mean_P": float(day["P"].mean()) if len(day) else None,
                    "mean_Q": float(day["Q"].mean()) if len(day) else None,
                    "date": str(wed.date()),
                }
            per_year[str(year)] = year_rows
        # aggregate diffs
        diffs = {}
        for month in ("01", "04", "10"):
            vals_p = [per_year[y][month]["mean_P"] for y in ("2020", "2021", "2022")]
            vals_q = [per_year[y][month]["mean_Q"] for y in ("2020", "2021", "2022")]
            if all(v is not None for v in vals_p):
                diffs[month] = {
                    "mean_P_by_year": {y: per_year[y][month]["mean_P"] for y in ("2020", "2021", "2022")},
                    "mean_Q_by_year": {y: per_year[y][month]["mean_Q"] for y in ("2020", "2021", "2022")},
                    "range_P": float(max(vals_p) - min(vals_p)),
                    "range_Q": float(max(vals_q) - min(vals_q)),
                    "max_abs_yoy_P": float(
                        max(abs(vals_p[1] - vals_p[0]), abs(vals_p[2] - vals_p[1]))
                    ),
                    "max_abs_yoy_Q": float(
                        max(abs(vals_q[1] - vals_q[0]), abs(vals_q[2] - vals_q[1]))
                    ),
                }
        load_shift.append({"meter": meter, "per_year": per_year, "diffs": diffs})

    # aggregate only (no raw)
    agg = {
        "n_meters": 10,
        "months": ["01", "04", "10"],
        "mean_range_P_across_meters": {
            m: float(np.mean([ls["diffs"][m]["range_P"] for ls in load_shift if m in ls["diffs"]]))
            for m in ("01", "04", "10")
        },
        "mean_range_Q_across_meters": {
            m: float(np.mean([ls["diffs"][m]["range_Q"] for ls in load_shift if m in ls["diffs"]]))
            for m in ("01", "04", "10")
        },
        "mean_max_abs_yoy_P": {
            m: float(
                np.mean([ls["diffs"][m]["max_abs_yoy_P"] for ls in load_shift if m in ls["diffs"]])
            )
            for m in ("01", "04", "10")
        },
    }

    return {
        "daily_2022_vs_2023": daily_compare,
        "f3_summer": summer,
        "wilcoxon_f3_summer_pooled_vs_local": wilcoxon_summer,
        "load_shift_wed_week2": load_shift,
        "load_shift_aggregate": agg,
    }


def write_csv_helpers(summary: dict, out: Path) -> list[str]:
    paths = []
    # R1.1 means
    rows = []
    for cell in summary["r1_1_gchung"]["cells"]:
        rows.append(
            {
                "grid": "g-chung",
                "horizon": cell["horizon"],
                "readout": cell["readout"],
                "channel": cell["channel"],
                "daily_naive": cell["daily_naive_mean"],
                **{k: v["mean"] for k, v in cell["models"].items()},
                **{
                    f"{k}_n_wins_vs_naive": v["vs_daily_naive"]["n_model_wins"]
                    for k, v in cell["models"].items()
                },
                **{
                    f"{k}_wilcoxon_p": v["vs_daily_naive"].get("wilcoxon_pvalue")
                    for k, v in cell["models"].items()
                },
            }
        )
    p = out / "r1_1_gchung_means.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths.append(str(p.relative_to(ROOT)))

    rows4 = []
    for cell in summary["r1_1_g4"]["cells"]:
        rows4.append(
            {
                "grid": "g4",
                "fold": cell["fold"],
                "horizon": cell["horizon"],
                "channel": cell["channel"],
                "daily_naive": cell["daily_naive_mean"],
                **{k: v["mean"] for k, v in cell["models"].items()},
                **{
                    f"{k}_n_wins_vs_naive": v["vs_daily_naive"]["n_model_wins"]
                    for k, v in cell["models"].items()
                },
            }
        )
    p = out / "r1_1_g4_means.csv"
    pd.DataFrame(rows4).to_csv(p, index=False)
    paths.append(str(p.relative_to(ROOT)))

    # R1.5 ranks
    r15rows = []
    for r in summary["r1_5"]["rankings"]:
        for metric in ("mae", "nmae", "mase_vs_daily_naive"):
            for cfg, vals in r[metric].items():
                r15rows.append(
                    {
                        "horizon": r["horizon"],
                        "readout": r["readout"],
                        "channel": r["channel"],
                        "metric": metric,
                        "config": cfg,
                        "mean": vals["mean"],
                        "median": vals["median"],
                        "rank": vals["rank_by_mean"],
                    }
                )
    p = out / "r1_5_rankings.csv"
    pd.DataFrame(r15rows).to_csv(p, index=False)
    paths.append(str(p.relative_to(ROOT)))

    # season days
    p = out / "r2_days_per_season.csv"
    pd.DataFrame(summary["r2_gchung"]["days_per_season_meter"]).to_csv(p, index=False)
    paths.append(str(p.relative_to(ROOT)))

    # rep days
    rep_flat = []
    for r in summary["r2_gchung"]["representative_wednesdays"]:
        if not r["found"]:
            rep_flat.append(
                {
                    "readout": r["readout"],
                    "meter": r["meter"],
                    "label": r["label"],
                    "date": r["date"],
                    "found": False,
                }
            )
            continue
        for cfg, chs in r["configs"].items():
            for ch, mets in chs.items():
                rep_flat.append(
                    {
                        "readout": r["readout"],
                        "meter": r["meter"],
                        "label": r["label"],
                        "date": r["date"],
                        "found": True,
                        "config": cfg,
                        "channel": ch,
                        **mets,
                    }
                )
    p = out / "r2_representative_days.csv"
    pd.DataFrame(rep_flat).to_csv(p, index=False)
    paths.append(str(p.relative_to(ROOT)))

    # 2022 vs 2023
    cmp_flat = []
    for r in summary["r2_g4"]["daily_2022_vs_2023"]:
        if not r["found"]:
            cmp_flat.append(
                {
                    "source": r["source"],
                    "meter": r["meter"],
                    "label": r["label"],
                    "date": r["date"],
                    "found": False,
                }
            )
            continue
        for cfg, chs in r["configs"].items():
            for ch, mets in chs.items():
                cmp_flat.append(
                    {
                        "source": r["source"],
                        "meter": r["meter"],
                        "label": r["label"],
                        "date": r["date"],
                        "found": True,
                        "config": cfg,
                        "channel": ch,
                        **mets,
                    }
                )
    p = out / "r2_daily_2022_vs_2023.csv"
    pd.DataFrame(cmp_flat).to_csv(p, index=False)
    paths.append(str(p.relative_to(ROOT)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("ASSUMPTIONS: see module docstring", flush=True)
    print("1) Verifying NPZ keys + MAE vs summary ...", flush=True)
    mae_checks = verify_mae_samples()
    print(f"   MAE checks OK: {len(mae_checks)} meter-files", flush=True)

    series_cache = SeriesCache()
    print("2) R1.1 G-chung ...", flush=True)
    r11 = build_r11_gchung(series_cache)
    h48 = r11["daily_naive_check"][48]["mean_joint_10meters"]
    print(f"   daily naive H48 mean={h48:.6f} (ref 390.45)", flush=True)

    print("3) R1.1 G4 ...", flush=True)
    r11_g4 = build_r11_g4(series_cache)

    print("4) R1.5 rankings ...", flush=True)
    r15 = build_r15(r11)

    print("5) R2 G-chung seasons/weeks/days ...", flush=True)
    r2 = build_r2_gchung(series_cache)

    print("6) R2 G4 calendar + load-shift ...", flush=True)
    r2_g4 = build_r2_g4(series_cache)

    elapsed = time.perf_counter() - t0
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "host": platform.node(),
        "platform": platform.platform(),
    }
    (out / "env.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in env.items())
        + "\n"
        + subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ),
        encoding="utf-8",
    )

    summary = {
        "run": "r1-r2-cpu",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "host": platform.node(),
        "device": "cpu",
        "gpu_name": None,
        "wall_sec": elapsed,
        "peak_rss_mb": peak_rss_mb(),
        "assumptions": __doc__.strip().splitlines(),
        "mae_npz_checks": mae_checks,
        "r1_1_gchung": r11,
        "r1_1_g4": r11_g4,
        "r1_5": r15,
        "r2_gchung": r2,
        "r2_g4": r2_g4,
        "env": env,
    }
    csv_paths = write_csv_helpers(summary, out)
    summary["csv_outputs"] = csv_paths

    # drop bulky mean_abs from printed path but keep in file
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    # concise stdout report
    print("\n=== REPORT ===", flush=True)
    print(f"MAE npz checks: all OK ({len(mae_checks)})", flush=True)
    print(
        f"Daily naive H48 mean={h48:.6f} vs 390.45 rel={r11['daily_naive_check'][48]['rel_diff_vs_ref']:.4e}",
        flush=True,
    )
    # R1.1 h48 joint last/full means
    for cell in r11["cells"]:
        if cell["horizon"] == 48 and cell["channel"] == "joint":
            print(
                f"R1.1 H48 {cell['readout']} joint means: naive={cell['daily_naive_mean']:.2f} "
                + " ".join(f"{k}={v['mean']:.2f}(wins {v['vs_daily_naive']['n_model_wins']}/10)"
                           for k, v in cell["models"].items()),
                flush=True,
            )
    # average over horizons for joint last
    for readout in READOUTS:
        subset = [c for c in r11["cells"] if c["readout"] == readout and c["channel"] == "joint"]
        print(
            f"R1.1 mean-over-H {readout} joint: naive={np.mean([c['daily_naive_mean'] for c in subset]):.2f} "
            + " ".join(
                f"{cfg}={np.mean([c['models'][cfg]['mean'] for c in subset]):.2f}"
                for cfg in [cfg_name(r, m) for r, m in MODEL_CFGS]
            ),
            flush=True,
        )
    # R1.5 sample
    for r in r15["rankings"]:
        if r["horizon"] == 48 and r["readout"] == "last" and r["channel"] in ("joint", "P", "Q"):
            print(f"R1.5 rank MAE H48 last {r['channel']}: {r['rank_order_mae']}", flush=True)
            print(f"         nMAE: {r['rank_order_nmae']} flip={r['rank_flip_mae_vs_nmae']}", flush=True)
    print(f"R1.5 n rank flips mae vs nmae: {len(r15['rank_flips_mae_vs_nmae'])}", flush=True)
    # season days
    dps = r2["days_per_season_meter"]
    for season in ("dong", "xuan", "he", "thu"):
        sub = [d for d in dps if d["readout"] == "last" and d["season"] == season]
        print(
            f"R2 days last|{season}: "
            + ", ".join(f"{d['meter']}={d['n_days']}{'*' if d['it_mau'] else ''}" for d in sub),
            flush=True,
        )
    # rep days found count
    for label, _ in [
        ("2022-10", None),
        ("2023-01", None),
        ("2023-04", None),
    ]:
        sub = [r for r in r2["representative_wednesdays"] if r["label"] == label and r["readout"] == "last"]
        n_found = sum(1 for r in sub if r["found"])
        print(f"R2 rep {label}: found {n_found}/10 meters", flush=True)
    print(f"commit={summary['git_commit'][:12]} wall_sec={elapsed:.1f} peak_rss_mb={summary['peak_rss_mb']:.1f}", flush=True)
    print(f"wrote {out / 'summary.json'}", flush=True)
    for p in csv_paths:
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
