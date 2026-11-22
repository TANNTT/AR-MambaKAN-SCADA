#!/usr/bin/env python3
"""Reproducibility check for IET GTD Table 5 and Table 17 (CPU-only, read-only data).

Assumptions (this run):
1. Table 5 printed cells are seed-mean MAE_P / MAE_Q over 40 meter-horizon tasks,
   rounded to 1 decimal, from pq_split_summary.csv (independent P/Q rerun).
2. Table 17 daily-naive is mean joint MAE over 10 meters of the fixed rule
   y_hat[j] = y[lookback + (j % 48) - 48] on test-segment WindowDataset windows
   (lookback=336, split 0.7/0.1 by row index), matching artifacts/seasonal_naive.csv.
3. Table 17 BiLSTM numbers come from pooled_multihorizon_*; AR-MambaKAN from
   stem-fix local/pooled CSVs (also mirrored in AR-MambaKAN-SCADA/results/).

Scripts named seasonal_naive.py / analyze_seasonal_naive.py / analyze_pq_split.py
are documented in PAPER_CLAIMS.md but are not present in either code tree
(PAPER_CLAIMS notes scratch scripts under /tmp). Logic below reconstructs the
rule from PAPER_CLAIMS Task #72 and matches the stored CSV to ~1e-5.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import sys as _sys
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from _paths import require_env  # noqa: E402

import numpy as np
import pandas as pd

LOOKBACK = 336
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
DAILY_PERIOD = 48
HORIZONS = (4, 48, 96, 336)

PAPER_TABLE5 = {
    # (model, regime, channel) -> printed 1-decimal
    ("bilstm", "local", "P"): 839.4,
    ("bilstm", "pooled", "P"): 786.5,
    ("bilstm", "local", "Q"): 508.8,
    ("bilstm", "pooled", "Q"): 368.7,
    ("armambakan", "local", "P"): 850.3,
    ("armambakan", "pooled", "P"): 787.2,
    ("armambakan", "local", "Q"): 540.6,
    ("armambakan", "pooled", "Q"): 379.6,
    ("mlp", "local", "P"): 1300.8,
    ("mlp", "pooled", "P"): 868.4,
    ("lstm", "local", "P"): 852.3,
    ("lstm", "pooled", "P"): 801.7,
}

PAPER_TABLE17_DAILY = {4: 391.6, 48: 390.5, 96: 484.0, 336: 767.2}
PAPER_TABLE17_MODELS = {
    ("pooled", "bilstm", 4): 245.7,
    ("pooled", "bilstm", 48): 490.5,
    ("pooled", "bilstm", 96): 607.2,
    ("pooled", "bilstm", 336): 952.7,
    ("local", "bilstm", 4): 376.5,
    ("local", "bilstm", 48): 639.3,
    ("local", "bilstm", 96): 722.7,
    ("local", "bilstm", 336): 970.4,
    ("pooled", "armambakan", 4): 285.9,
    ("pooled", "armambakan", 48): 481.8,
    ("pooled", "armambakan", 96): 607.4,
    ("pooled", "armambakan", 336): 941.6,
    ("local", "armambakan", 4): 408.7,
    ("local", "armambakan", 48): 636.9,
    ("local", "armambakan", 96): 737.3,
    ("local", "armambakan", 336): 1004.6,
}


def split_indices(n: int, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO):
    """Mirror AR-MambaKAN-SCADA/src/evaluation/train.py::split_indices."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def half_ulp_printed(printed: float) -> float:
    """Half unit in the last printed decimal place (paper uses 1 decimal)."""
    return 0.05


def verdict(printed: float, recomputed: float) -> str:
    return "KHOP" if abs(recomputed - printed) < half_ulp_printed(printed) else "LECH"


def seasonal_naive_mae(
    series_pq: np.ndarray,
    lookback: int,
    horizon: int,
    period: int = DAILY_PERIOD,
) -> float:
    """Joint MAE of daily/weekly seasonal-naive on a contiguous segment.

    For window origin i and lead j in 0..H-1:
        y_hat[j] = series[i + lookback + (j % period) - period]
    Matches PAPER_CLAIMS Task #72 ("cyclically repeated for h>m") and
    reproduces artifacts/seasonal_naive.csv on the test split.
    """
    n = len(series_pq)
    n_win = n - lookback - horizon + 1
    if n_win <= 0:
        raise ValueError(f"segment too short: n={n}, L={lookback}, H={horizon}")
    errs = []
    for i in range(n_win):
        target = series_pq[i + lookback : i + lookback + horizon]
        pred = np.empty_like(target)
        for j in range(horizon):
            pred[j] = series_pq[i + lookback + (j % period) - period]
        errs.append(np.abs(pred - target))
    return float(np.mean(np.stack(errs, axis=0)))


def seasonal_naive_mae_fast(
    series_pq: np.ndarray,
    lookback: int,
    horizon: int,
    period: int = DAILY_PERIOD,
) -> tuple[float, int]:
    """Vectorized form of seasonal_naive_mae (same numeric result)."""
    n = len(series_pq)
    n_win = n - lookback - horizon + 1
    if n_win <= 0:
        raise ValueError(f"segment too short: n={n}, L={lookback}, H={horizon}")
    targets = np.stack(
        [series_pq[i + lookback : i + lookback + horizon] for i in range(n_win)]
    )
    preds = np.empty_like(targets)
    for j in range(horizon):
        src = lookback + (j % period) - period
        preds[:, j, :] = series_pq[np.arange(n_win) + src]
    return float(np.mean(np.abs(preds - targets))), n_win


def self_test_periodic() -> dict:
    """Synthetic period-48 series must give daily-naive MAE == 0 on test windows."""
    n = 5000
    t = np.arange(n)
    day = np.sin(2 * np.pi * t / 48.0)
    series = np.stack([day, 0.3 * day], axis=1)
    _, val_end = split_indices(n)
    test = series[val_end:]
    mae4, nw4 = seasonal_naive_mae_fast(test, LOOKBACK, 4, DAILY_PERIOD)
    mae48, nw48 = seasonal_naive_mae_fast(test, LOOKBACK, 48, DAILY_PERIOD)
    mae96, nw96 = seasonal_naive_mae_fast(test, LOOKBACK, 96, DAILY_PERIOD)
    ok = max(abs(mae4), abs(mae48), abs(mae96)) < 1e-12
    return {
        "name": "synthetic_period48_mae_zero",
        "passed": bool(ok),
        "mae_H4": mae4,
        "mae_H48": mae48,
        "mae_H96": mae96,
        "n_windows_H4": nw4,
        "n_windows_H48": nw48,
        "n_windows_H96": nw96,
    }


def self_test_demo_data(demo_dir: Path) -> dict:
    """Sanity: demo CSVs load; daily-naive MAE is finite and non-negative."""
    demo = demo_dir / "DEMO0001.csv"
    if not demo.is_file():
        return {"name": "demo_data_daily_naive", "passed": False, "error": "missing DEMO0001.csv"}
    df = pd.read_csv(demo, usecols=["pt_kw", "qt_kvar"])
    arr = df[["pt_kw", "qt_kvar"]].to_numpy(np.float64)
    _, val_end = split_indices(len(arr))
    test = arr[val_end:]
    mae, nw = seasonal_naive_mae_fast(test, LOOKBACK, 48, DAILY_PERIOD)
    return {
        "name": "demo_data_daily_naive",
        "passed": bool(np.isfinite(mae) and mae >= 0 and nw > 0),
        "mae_H48": mae,
        "n_windows": nw,
        "n_rows_read": int(len(arr)),
        "demo_file": demo.name,
    }


def load_anon_map(path: Path) -> dict[str, str]:
    """real_code -> anon_label (M01..)."""
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["real_code"]] = row["anon_label"]
    return out


def choose_meter_csv(search_roots: list[Path], real_code: str) -> Path | None:
    cands: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        cands.extend(root.rglob(f"*_{real_code}_2020_2023.csv"))
    if not cands:
        return None

    def score(p: Path) -> float:
        s = str(p)
        sc = 0.0
        if "smoke" in s:
            sc -= 200
        if "/q1_multihorizon_sota/h48/batch_datasets/" in s:
            sc += 100
        elif "/artifacts/batch_datasets/" in s and "/q1_" not in s:
            sc += 80
        elif "/artifacts/multistation_datasets/" in s:
            sc += 60
        sc -= len(s) * 0.001
        return sc

    return max(cands, key=score)


def check_table5(art: Path, pub: Path) -> list[dict]:
    loc = pd.read_csv(art / "pq_split_local.csv")
    poo = pd.read_csv(art / "pq_split_pooled.csv")
    summary_art = pd.read_csv(art / "pq_split_summary.csv").set_index("model")
    summary_pub = pd.read_csv(pub / "pq_split_summary.csv").set_index("model")

    cells = []
    # Primary cell: BiLSTM local P, recomputed from per-seed task rows
    model, regime, channel = "bilstm", "local", "P"
    printed = PAPER_TABLE5[(model, regime, channel)]
    df = loc if regime == "local" else poo
    col = "mae_p" if channel == "P" else "mae_q"
    sub = df[df["model"] == model]
    seed_means = sub.groupby("seed")[col].mean()
    recomputed = float(seed_means.mean())
    summary_val = float(summary_art.loc[model, f"{regime}_mae_{channel.lower()}_mean"])
    pub_val = float(summary_pub.loc[model, f"{regime}_mae_{channel.lower()}_mean"])
    cells.append(
        {
            "table": 5,
            "cell": f"{model}|{regime}|{channel}",
            "row": "BiLSTM",
            "column": f"{regime}_{channel}",
            "printed": printed,
            "recomputed": recomputed,
            "abs_diff": abs(recomputed - printed),
            "rel_diff": abs(recomputed - printed) / abs(printed) if printed else None,
            "verdict": verdict(printed, recomputed),
            "aggregation": "mean over 3 seeds of (mean MAE over 40 meter-horizon tasks)",
            "source_detail": {
                "file": str(art / "pq_split_local.csv"),
                "column": col,
                "n_rows_used": int(len(sub)),
                "seed_means": {str(k): float(v) for k, v in seed_means.items()},
            },
            "source_summary": {
                "file": str(art / "pq_split_summary.csv"),
                "column": f"{regime}_mae_{channel.lower()}_mean",
                "value": summary_val,
                "matches_recomputed": abs(summary_val - recomputed) < 1e-9,
            },
            "public_repo_summary": {
                "file": str(pub / "pq_split_summary.csv"),
                "value": pub_val,
                "identical_to_artifacts": abs(pub_val - summary_val) < 1e-12,
            },
            "code_generator": {
                "documented": "PAPER_CLAIMS.md Task pq_split: pq_split_metrics.py + analyze_pq_split.py",
                "scripts_present_in_trees": False,
                "note": "scratch scripts under /tmp per PAPER_CLAIMS; outputs retained in artifacts/ and results/",
            },
            "level": "below_summary_csv",
        }
    )

    # Also confirm summary rounding for a second printed cell without detail re-agg claim
    model2, regime2, ch2 = "armambakan", "pooled", "Q"
    printed2 = PAPER_TABLE5[(model2, regime2, ch2)]
    summary_val2 = float(summary_art.loc[model2, f"{regime2}_mae_{ch2.lower()}_mean"])
    cells.append(
        {
            "table": 5,
            "cell": f"{model2}|{regime2}|{ch2}",
            "row": "AR-MambaKAN",
            "column": f"{regime2}_{ch2}",
            "printed": printed2,
            "recomputed": summary_val2,
            "abs_diff": abs(summary_val2 - printed2),
            "rel_diff": abs(summary_val2 - printed2) / abs(printed2),
            "verdict": verdict(printed2, summary_val2),
            "aggregation": "pq_split_summary mean (same as seed-agg of 40 tasks)",
            "source_summary": {
                "file": str(art / "pq_split_summary.csv"),
                "column": f"{regime2}_mae_{ch2.lower()}_mean",
                "value": summary_val2,
            },
            "level": "summary_csv_rounding_check",
        }
    )
    return cells


def check_table17_daily_from_data(
    art: Path,
    map_path: Path,
    search_roots: list[Path],
    anon_label: str,
    horizon: int,
) -> dict:
    real_to_anon = load_anon_map(map_path)
    anon_to_real = {v: k for k, v in real_to_anon.items()}
    if anon_label not in anon_to_real:
        raise KeyError(f"unknown anon label {anon_label}")
    real_code = anon_to_real[anon_label]
    csv_path = choose_meter_csv(search_roots, real_code)
    if csv_path is None:
        raise FileNotFoundError(f"no built CSV for {anon_label}")

    df = pd.read_csv(csv_path, usecols=["pc_time", "pt_kw", "qt_kvar"], parse_dates=["pc_time"])
    df = df.dropna(subset=["pc_time", "pt_kw", "qt_kvar"]).sort_values("pc_time").reset_index(drop=True)
    arr = df[["pt_kw", "qt_kvar"]].to_numpy(np.float64)
    n = len(arr)
    _, val_end = split_indices(n)
    test = arr[val_end:]
    mae, n_win = seasonal_naive_mae_fast(test, LOOKBACK, horizon, DAILY_PERIOD)

    sn = pd.read_csv(art / "seasonal_naive.csv")
    # Compare to CSV using real_code only in memory; never write real_code to summary
    csv_mae = float(
        sn[(sn["code"] == real_code) & (sn["horizon"] == horizon) & (sn["baseline"] == "daily_naive")][
            "mae"
        ].iloc[0]
    )
    # Aggregate printed cell is mean over 10 meters
    printed_agg = PAPER_TABLE17_DAILY[horizon]
    sn_h = sn[(sn["horizon"] == horizon) & (sn["baseline"] == "daily_naive")]
    recomputed_agg = float(sn_h["mae"].mean())

    return {
        "table": 17,
        "cell": f"daily_naive|H{horizon}|mean_over_10_meters",
        "row": "Daily-naive",
        "column": f"H{horizon}",
        "printed": printed_agg,
        "recomputed": recomputed_agg,
        "abs_diff": abs(recomputed_agg - printed_agg),
        "rel_diff": abs(recomputed_agg - printed_agg) / printed_agg,
        "verdict": verdict(printed_agg, recomputed_agg),
        "aggregation": "mean of per-meter daily_naive MAE (10 meters)",
        "source_summary": {
            "file": str(art / "seasonal_naive.csv"),
            "column": "mae",
            "filter": f"baseline=daily_naive, horizon={horizon}",
            "n_meters": int(len(sn_h)),
        },
        "per_meter_from_built_data": {
            "anon_label": anon_label,
            "horizon": horizon,
            "lookback": LOOKBACK,
            "period": DAILY_PERIOD,
            "split": "test = rows[int(0.8*n):] after sort by pc_time",
            "n_rows_full": n,
            "n_rows_test": int(len(test)),
            "n_windows": n_win,
            "mae_from_data": mae,
            "mae_in_seasonal_naive_csv": csv_mae,
            "abs_diff_vs_csv": abs(mae - csv_mae),
            "verdict_vs_csv": "KHOP" if abs(mae - csv_mae) < 1e-4 else "LECH",
            "source_dataset_kind": "built_*_2020_2023.csv under q1_multihorizon_sota/h48/batch_datasets",
            "code_rule": (
                "PAPER_CLAIMS.md:656-657; reconstructed: "
                "y_hat[j]=y[L+(j%48)-48] on test WindowDataset; "
                "scripts seasonal_naive.py missing from trees"
            ),
        },
        "level": "built_data_plus_summary_csv",
    }


def check_table17_model(art: Path, pub: Path, regime: str, model: str, horizon: int) -> dict:
    printed = PAPER_TABLE17_MODELS[(regime, model, horizon)]
    if model == "armambakan":
        if regime == "local":
            path = art / "armambakan_stemfix_local.csv"
            col = "mae"
            df = pd.read_csv(path)
            task = df[df["horizon"] == horizon].groupby("code")[col].mean()
            # also pub mirror
            pub_path = pub / "local_vs_pooled_armambakan_local.csv"
            pub_df = pd.read_csv(pub_path)
            pub_task = pub_df[pub_df["horizon"] == horizon].groupby("code")["mae"].mean()
        else:
            path = art / "armambakan_stemfix_pooled.csv"
            col = "pooled_mae"
            df = pd.read_csv(path)
            task = df[df["horizon"] == horizon].groupby("code")[col].mean()
            pub_path = pub / "local_vs_pooled_armambakan_pooled.csv"
            pub_df = pd.read_csv(pub_path)
            pub_task = pub_df[pub_df["horizon"] == horizon].groupby("code")["pooled_mae"].mean()
        recomputed = float(task.mean())
        pub_recomputed = float(pub_task.mean())
        source = {
            "file": str(path),
            "column": col,
            "aggregation": "seed-mean per meter then mean over 10 meters",
            "n_seed_rows": int(len(df[df["horizon"] == horizon])),
            "public_mirror_file": str(pub_path),
            "public_recomputed": pub_recomputed,
            "matches_public": abs(recomputed - pub_recomputed) < 1e-9,
        }
        code_gen = {
            "documented": "PAPER_CLAIMS stem-fix cascade; recompute_seasonal_naive_stemfix.py (scratch, missing)",
            "scripts_present_in_trees": False,
            "csv_present": True,
        }
    elif model == "bilstm" and regime == "pooled":
        path = art / "pooled_multihorizon_results.csv"
        df = pd.read_csv(path)
        sub = df[(df["model"] == "bilstm") & (df["horizon"] == horizon)]
        task = sub.groupby("code")["pooled_mae"].mean()
        recomputed = float(task.mean())
        # summary file
        summ = pd.read_csv(art / "pooled_multihorizon_summary.csv")
        summ_val = float(
            summ[(summ["model"] == "bilstm") & (summ["horizon"] == horizon)]["pooled_mae"].iloc[0]
        )
        source = {
            "file": str(path),
            "column": "pooled_mae",
            "aggregation": "seed-mean per meter then mean over 10 meters",
            "n_seed_rows": int(len(sub)),
            "summary_file": str(art / "pooled_multihorizon_summary.csv"),
            "summary_value": summ_val,
            "matches_summary": abs(recomputed - summ_val) < 1e-9,
        }
        code_gen = {
            "documented": "pooled multihorizon batch runs; summarize_multihorizon.py family",
            "scripts_present_in_trees": True,
            "note": "upstream summarize_multihorizon; public repo has results only for AR-MambaKAN local/pooled",
        }
    elif model == "bilstm" and regime == "local":
        # Local BiLSTM in Table 17 matches Table 11 / friedman9 Local BiLSTM column
        fr = pd.read_csv(art / "friedman9_stemfix_data.csv")
        # friedman uses anonymized? check
        col = "Local BiLSTM"
        recomputed = float(fr[fr["horizon"] == horizon][col].mean())
        source = {
            "file": str(art / "friedman9_stemfix_data.csv"),
            "column": col,
            "aggregation": "mean over 10 meters (values already seed-averaged)",
            "note": "same numbers as pooled_multihorizon_summary local_mae for bilstm",
        }
        summ = pd.read_csv(art / "pooled_multihorizon_summary.csv")
        summ_val = float(
            summ[(summ["model"] == "bilstm") & (summ["horizon"] == horizon)]["local_mae"].iloc[0]
        )
        source["summary_file"] = str(art / "pooled_multihorizon_summary.csv")
        source["summary_local_mae"] = summ_val
        source["matches_summary"] = abs(recomputed - summ_val) < 1e-6
        code_gen = {
            "documented": "multihorizon local benchmark + friedman9_stemfix assembly",
            "scripts_present_in_trees": "partial",
        }
    else:
        raise ValueError(f"unsupported {regime}/{model}")

    return {
        "table": 17,
        "cell": f"{regime}|{model}|H{horizon}",
        "row": f"{regime} {model}",
        "column": f"H{horizon}",
        "printed": printed,
        "recomputed": recomputed,
        "abs_diff": abs(recomputed - printed),
        "rel_diff": abs(recomputed - printed) / printed,
        "verdict": verdict(printed, recomputed),
        "source": source,
        "code_generator": code_gen,
        "level": "below_or_equal_summary_csv",
    }


def git_rev(path: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=None, help="default: $SCADA_ARTIFACTS_DIR")
    ap.add_argument("--pub", default=None, help="default: $SCADA_PUBLIC_REPO/results")
    ap.add_argument("--map", default=None, help="default: $SCADA_ANON_MAP")
    ap.add_argument("--demo-dir", default=None, help="default: $SCADA_DEMO_DATA")
    ap.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Roots to search for built *_2020_2023.csv (repeatable)",
    )
    ap.add_argument("--out", default="results/kiem-tai-lap")
    ap.add_argument("--anon-probe", default="M01", help="Anonymous meter for data-level daily-naive probe")
    ap.add_argument("--horizon-probe", type=int, default=48)
    args = ap.parse_args()
    if args.art is None:
        args.art = str(require_env("SCADA_ARTIFACTS_DIR"))
    if args.pub is None:
        args.pub = str(require_env("SCADA_PUBLIC_REPO") / "results")
    if args.map is None:
        args.map = str(require_env("SCADA_ANON_MAP"))
    if args.demo_dir is None:
        args.demo_dir = str(require_env("SCADA_DEMO_DATA"))

    art = Path(args.art)
    pub = Path(args.pub)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_roots = (
        [Path(p) for p in args.search_root]
        if args.search_root
        else [require_env("SCADA_ARTIFACTS_DIR")]
    )

    self_tests = [self_test_periodic(), self_test_demo_data(Path(args.demo_dir))]
    if not all(t["passed"] for t in self_tests):
        print("SELF-TEST FAILED", self_tests, file=sys.stderr)
        return 2

    cells: list[dict] = []
    cells.extend(check_table5(art, pub))
    cells.append(
        check_table17_daily_from_data(
            art, Path(args.map), search_roots, args.anon_probe, args.horizon_probe
        )
    )
    cells.append(check_table17_model(art, pub, "pooled", "armambakan", 4))
    cells.append(check_table17_model(art, pub, "pooled", "bilstm", 4))

    paper_dir = Path(__file__).resolve().parent
    summary = {
        "run_name": "kiem-tai-lap",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "python": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "git_commit_paper_dir": git_rev(paper_dir),
        "git_commit_armambakan": git_rev(require_env("SCADA_PUBLIC_REPO")),
        "assumptions": [
            "Table 5 = independent P/Q MAE means over 40 tasks, 3 seeds (pq_split_*).",
            "Table 17 daily-naive = mean over 10 meters of test-split WindowDataset seasonal rule period=48, L=336.",
            "Table 17 AR-MambaKAN = stem-fix local/pooled CSVs; BiLSTM pooled = pooled_multihorizon_results.",
            "No raw SCADA values or real meter codes written to this summary.",
        ],
        "self_tests": self_tests,
        "cells": cells,
        "sources": {
            "table5": {
                "summary_csv": [
                    str(art / "pq_split_summary.csv"),
                    str(pub / "pq_split_summary.csv"),
                ],
                "detail_csv": [
                    str(art / "pq_split_local.csv"),
                    str(art / "pq_split_pooled.csv"),
                ],
                "generator_scripts": {
                    "named_in_PAPER_CLAIMS": ["pq_split_metrics.py", "analyze_pq_split.py"],
                    "present_in_code_trees": False,
                },
            },
            "table17": {
                "daily_naive_csv": str(art / "seasonal_naive.csv"),
                "daily_naive_summary_csv": str(art / "seasonal_naive_summary.csv"),
                "bilstm_pooled_detail": str(art / "pooled_multihorizon_results.csv"),
                "bilstm_summary": str(art / "pooled_multihorizon_summary.csv"),
                "armambakan_local": str(art / "armambakan_stemfix_local.csv"),
                "armambakan_pooled": str(art / "armambakan_stemfix_pooled.csv"),
                "public_armambakan_local": str(pub / "local_vs_pooled_armambakan_local.csv"),
                "public_armambakan_pooled": str(pub / "local_vs_pooled_armambakan_pooled.csv"),
                "friedman9": str(art / "friedman9_stemfix_data.csv"),
                "generator_scripts": {
                    "named_in_PAPER_CLAIMS": [
                        "seasonal_naive.py",
                        "analyze_seasonal_naive.py",
                        "recompute_seasonal_naive_stemfix.py",
                    ],
                    "present_in_code_trees": False,
                    "note": "PAPER_CLAIMS: scratch scripts under /tmp, outputs kept in artifacts/",
                },
            },
            "checkpoints_predictions": {
                "AR-MambaKAN-SCADA": "none (.pt/.npy predictions absent)",
                "artifacts/pilot_h48_s42": (
                    "265 .pt checkpoints exist but belong to locked pilot protocol "
                    "(H=48 epoch-10, representatives), NOT confirmed as IET BiLSTM/"
                    "AR-MambaKAN multihorizon pooled/local runs"
                ),
            },
        },
        "other_mae_rank_tables_in_submission": [
            {"table": 3, "note": "pooled vs local joint MAE, 4 architectures"},
            {"table": 4, "note": "per-horizon pooling benefit BiLSTM/AR-MambaKAN"},
            {"table": 6, "note": "data-budget-matched pooling MAE"},
            {"table": 7, "note": "LOSO MAE BiLSTM/AR-MambaKAN (and others)"},
            {"table": 9, "note": "quality-strata pooled vs local MAE"},
            {"table": 10, "note": "Friedman 9-condition mean ranks (incl. BiLSTM/AR-MambaKAN)"},
            {"table": 11, "note": "local multihorizon MAE/sMAPE BiLSTM/AR-MambaKAN/LSTM/MLP/Naive"},
            {"table": 12, "note": "Wilcoxon AR-MambaKAN vs baselines (local)"},
            {"table": 13, "note": "modern baselines MAE (not BiLSTM/AR)"},
            {"table": 14, "note": "pairwise among 6 learned models incl. BiLSTM/AR-MambaKAN"},
            {"table": 16, "note": "naive baselines by horizon (daily/weekly/last-value)"},
            {"table": 18, "note": "component ablation MAE (A3 BiLSTM, A4 AR-MambaKAN)"},
            {"table": 19, "note": "calendar ablation local MAE incl. BiLSTM/AR-MambaKAN"},
        ],
        "lstm_readout_impact_note": (
            "Any change to BiLSTM/AR-MambaKAN LSTM readout would affect Tables "
            "3,4,5,6,7,9,10,11,12,14,17,18,19 (and possibly 8 hierarchical tests). "
            "Tables 13,15,16 are baselines/compute/naive and are less directly tied "
            "to BiLSTM readout; Table 16 naive is independent."
        ),
    }

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env_path = out_dir / "env.txt"
    env_path.write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"pandas={pd.__version__}",
                f"numpy={np.__version__}",
                f"host={platform.node()}",
                f"platform={platform.platform()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("self_tests:", json.dumps(self_tests, ensure_ascii=False))
    for c in cells:
        print(
            f"T{c['table']} {c['cell']}: printed={c['printed']} recomputed={c['recomputed']:.6f} "
            f"diff={c['abs_diff']:.6g} {c['verdict']}"
        )
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
