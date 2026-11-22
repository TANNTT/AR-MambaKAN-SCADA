"""Meter CSV discovery for G-chung (anonymized labels only in outputs).

Assumptions:
1. Real experiment CSVs are ``*_CODE_2020_2023.csv`` under ``SCADA_ARTIFACTS_DIR``
   (same preference order as do_phu_du_lieu.py / kiem_tai_lap.py).
2. Demo uses ``SCADA_DEMO_DATA``/DEMO*.csv (synthetic; labels DEMO0001..).
3. Anonymization map maps real_code -> M01..; real codes never written to summary/npz keys.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from g_chung.features.features import (
    CALENDAR_FEATURE_COLS,
    WEATHER_FEATURE_COLS,
    add_calendar_features,
    add_weather_features,
    station_slug,
)
from g_chung.train_core import (
    LOOKBACK,
    SplitIndex,
    TimedWindowDataset,
    WindowDataset,
    split_indices,
    standardize_by_train,
)

import sys as _sys
_PAPER_ROOT = Path(__file__).resolve().parents[1]
if str(_PAPER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PAPER_ROOT))
from _paths import require_env, env_or_none  # noqa: E402


def _map_path() -> Path:
    return require_env("SCADA_ANON_MAP")


def _demo_path() -> Path:
    return require_env("SCADA_DEMO_DATA")


def _search_roots() -> list[Path]:
    roots = [require_env("SCADA_ARTIFACTS_DIR")]
    nocal = env_or_none("SCADA_ARTIFACTS_NOCAL_DIR")
    if nocal is not None:
        roots.append(nocal)
    return roots


def _weather_dir() -> Path:
    return require_env("SCADA_WEATHER_DIR")


def _weather_station_ref() -> Path:
    return require_env("SCADA_WEATHER_STATION_REF")


MAIN_METERS = tuple(f"M{i:02d}" for i in range(1, 11))
# Table 20 weather A/B meters (8 stations, mapped from armambakan_stemfix_weather.csv codes).
WEATHER_STUDY_METERS = ("M16", "M01", "M05", "M22", "M28", "M03", "M23", "M24")
ELEC_COLS = ["pt_kw", "qt_kvar", "ua_v", "ub_v", "uc_v", "cos_phi", "freq_hz"]


def load_anon_map(path: Path | None = None) -> dict[str, str]:
    """anon_label -> real_code (real_code used only to locate files)."""
    path = path or _map_path()
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["anon_label"].strip()] = row["real_code"].strip()
    return out


def _score_csv(p: Path) -> float:
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
    if "artifacts_nocal" in s:
        sc -= 20
    sc -= len(s) * 0.001
    return sc


def choose_meter_csv(search_roots: list[Path], real_code: str) -> Path | None:
    cands: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        cands.extend(root.rglob(f"*_{real_code}_2020_2023.csv"))
    if not cands:
        return None
    return max(cands, key=_score_csv)


def discover_demo_meters(demo_dir: Path | None = None) -> dict[str, Path]:
    demo_dir = demo_dir or _demo_path()
    paths = sorted(demo_dir.glob("DEMO*.csv"))
    return {p.stem: p for p in paths}


def resolve_meter_index(
    *,
    use_demo: bool,
    meters: list[str] | None,
    map_path: Path | None = None,
    search_roots: list[Path] | None = None,
    demo_dir: Path | None = None,
) -> dict[str, Path]:
    """Return {anon_or_demo_label: csv_path}. Never exposes real codes as keys."""
    if use_demo:
        all_demo = discover_demo_meters(demo_dir)
        if meters:
            missing = [m for m in meters if m not in all_demo]
            if missing:
                raise FileNotFoundError(f"demo meters not found: {missing}")
            return {m: all_demo[m] for m in meters}
        return all_demo

    anon_to_real = load_anon_map(map_path)
    roots = search_roots or _search_roots()
    wanted = meters or list(MAIN_METERS)
    out: dict[str, Path] = {}
    for label in wanted:
        if label not in anon_to_real:
            raise KeyError(f"unknown anon label {label}")
        real = anon_to_real[label]
        path = choose_meter_csv(roots, real)
        if path is None:
            raise FileNotFoundError(f"no built CSV for {label}")
        out[label] = path
    return out


# Rolling-origin folds for G4 (PA-3 bỏ F1). Inclusive calendar dates (UTC-naive local).
ROLLING_FOLDS: dict[str, tuple[str, str]] = {
    "F2": ("2022-01-01", "2022-04-30"),
    "F3": ("2022-05-01", "2022-08-31"),
}


def _anon_to_station_name(map_path: Path | None = None, ref_csv: Path | None = None) -> dict[str, str]:
    """anon_label -> station display name (for weather slug only; never exported)."""
    map_path = map_path or _map_path()
    ref_csv = ref_csv or _weather_station_ref()
    if not ref_csv.is_file():
        return {}
    real_to_anon = {v: k for k, v in load_anon_map(map_path).items()}
    out: dict[str, str] = {}
    with ref_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            code = row.get("code", "").strip()
            station = row.get("station", "").strip()
            anon = real_to_anon.get(code)
            if anon and station and anon not in out:
                out[anon] = station
    return out


def resolve_weather_csv(
    weather_dir: Path,
    anon_label: str,
    *,
    map_path: Path | None = None,
) -> Path | None:
    """Return per-meter ERA5 CSV path, or None if missing.

    Mirrors features.features.station_slug + benchmark_common.resolve_weather_csv.
    """
    if not weather_dir or not Path(weather_dir).is_dir():
        return None
    names = _anon_to_station_name(map_path=map_path)
    station = names.get(anon_label)
    if not station:
        return None
    path = Path(weather_dir) / f"{station_slug(station)}.csv"
    return path if path.is_file() else None


def _prepare_frame(
    path: Path,
    calendar_features: bool,
    weather_csv: Path | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, parse_dates=["pc_time"])
    df = df.dropna(subset=["pc_time"]).sort_values("pc_time").reset_index(drop=True)
    feat_cols = [c for c in ELEC_COLS if c in df.columns]
    if calendar_features:
        df = add_calendar_features(df, time_col="pc_time")
        feat_cols = feat_cols + [c for c in CALENDAR_FEATURE_COLS if c in df.columns]
    if weather_csv is not None:
        df = add_weather_features(df, str(weather_csv), time_col="pc_time")
        feat_cols = feat_cols + [c for c in WEATHER_FEATURE_COLS if c in df.columns]
    return df, feat_cols


def _bundle_from_arrays(
    train_arr,
    val_arr,
    test_arr,
    train_t,
    val_t,
    test_t,
    feat_cols: list[str],
    horizon: int,
    lookback: int,
    split_meta: dict,
) -> dict:
    train_arr, val_arr, test_arr, mean, std = standardize_by_train(train_arr, val_arr, test_arr)
    import torch

    y_mean = torch.tensor(mean[0, :2], dtype=torch.float32).view(1, 1, 2)
    y_std = torch.tensor(std[0, :2], dtype=torch.float32).view(1, 1, 2)
    n_rows = int(len(train_arr) + len(val_arr) + len(test_arr))
    return {
        "n_features": len(feat_cols),
        "n_rows": n_rows,
        "feat_cols": feat_cols,
        "train_ds": TimedWindowDataset(train_arr, train_t, lookback, horizon),
        "val_ds": TimedWindowDataset(val_arr, val_t, lookback, horizon),
        "test_ds": TimedWindowDataset(test_arr, test_t, lookback, horizon),
        "train_plain": WindowDataset(train_arr, lookback, horizon),
        "y_mean": y_mean,
        "y_std": y_std,
        "split": split_meta,
        "boundaries": {
            "train_t_min": int(train_t[0]) if len(train_t) else None,
            "train_t_max": int(train_t[-1]) if len(train_t) else None,
            "val_t_min": int(val_t[0]) if len(val_t) else None,
            "val_t_max": int(val_t[-1]) if len(val_t) else None,
            "test_t_min": int(test_t[0]) if len(test_t) else None,
            "test_t_max": int(test_t[-1]) if len(test_t) else None,
        },
    }


def load_meter_bundle(
    path: Path,
    horizon: int,
    *,
    calendar_features: bool = True,
    weather_csv: Path | str | None = None,
    lookback: int = LOOKBACK,
) -> dict:
    """Load one meter: per-split TimedWindowDatasets + destandardization stats.

    Mirrors common_pooled.load_meter_windows (lines 52-76) with timestamps retained.
    Calendar default True matches experiments/pooled/run_pooled.py:39 @ 8582ecc.
    Optional weather_csv joins ERA5 wx_* columns (run_weather_ablation.py:55-60).
    """
    df, feat_cols = _prepare_frame(path, calendar_features, weather_csv=weather_csv)
    arr = df[feat_cols].astype(np.float32).to_numpy()
    times_ns = df["pc_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)

    idx = split_indices(len(arr))
    train_arr = arr[: idx.train_end]
    val_arr = arr[idx.train_end : idx.val_end]
    test_arr = arr[idx.val_end :]
    train_t = times_ns[: idx.train_end]
    val_t = times_ns[idx.train_end : idx.val_end]
    test_t = times_ns[idx.val_end :]

    return _bundle_from_arrays(
        train_arr,
        val_arr,
        test_arr,
        train_t,
        val_t,
        test_t,
        feat_cols,
        horizon,
        lookback,
        {"train_end": idx.train_end, "val_end": idx.val_end, "n": int(len(arr)), "mode": "ratio_70_10_20"},
    )


def rolling_split_indices(n_pre: int) -> SplitIndex:
    """Split pre-test rows into train/val with val/train ≈ 1/7 (paper 70/10).

    Of the n_pre rows before the test calendar cut, use 7/8 for train and 1/8 for
    val so val_len / train_len = 1/7 when n_pre is divisible by 8.
    """
    if n_pre < 2:
        raise ValueError(f"need at least 2 pre-test rows, got {n_pre}")
    train_end = int(n_pre * 7 / 8)
    if train_end <= 0 or train_end >= n_pre:
        # Extremely short series: keep at least 1 val row.
        train_end = max(1, n_pre - 1)
    return SplitIndex(train_end=train_end, val_end=n_pre)


def load_meter_bundle_fold(
    path: Path,
    horizon: int,
    fold: str,
    *,
    calendar_features: bool = True,
    weather_csv: Path | str | None = None,
    lookback: int = LOOKBACK,
) -> dict:
    """Rolling-origin bundle: test = fold calendar window; train/val from earlier rows only."""
    if fold not in ROLLING_FOLDS:
        raise KeyError(f"unknown fold {fold}; expected {list(ROLLING_FOLDS)}")
    test_start_s, test_end_s = ROLLING_FOLDS[fold]
    test_start = np.datetime64(test_start_s, "ns")
    # Inclusive end-of-day: last 30-min stamp on test_end_s is 23:30.
    test_end = np.datetime64(test_end_s, "ns") + np.timedelta64(1, "D") - np.timedelta64(1, "ns")

    df, feat_cols = _prepare_frame(path, calendar_features, weather_csv=weather_csv)
    times = df["pc_time"].to_numpy(dtype="datetime64[ns]")
    arr = df[feat_cols].astype(np.float32).to_numpy()
    times_ns = times.astype(np.int64)

    pre_mask = times < test_start
    test_mask = (times >= test_start) & (times <= test_end)
    n_pre = int(pre_mask.sum())
    n_test = int(test_mask.sum())
    if n_pre < lookback + horizon + 2:
        raise RuntimeError(
            f"{path.name} fold {fold}: too few pre-test rows ({n_pre}) for lookback+horizon"
        )
    if n_test < lookback + horizon:
        raise RuntimeError(f"{path.name} fold {fold}: too few test rows ({n_test})")

    idx = rolling_split_indices(n_pre)
    pre_arr = arr[pre_mask]
    pre_t = times_ns[pre_mask]
    test_arr = arr[test_mask]
    test_t = times_ns[test_mask]
    train_arr = pre_arr[: idx.train_end]
    val_arr = pre_arr[idx.train_end : idx.val_end]
    train_t = pre_t[: idx.train_end]
    val_t = pre_t[idx.train_end : idx.val_end]

    return _bundle_from_arrays(
        train_arr,
        val_arr,
        test_arr,
        train_t,
        val_t,
        test_t,
        feat_cols,
        horizon,
        lookback,
        {
            "mode": "rolling_origin",
            "fold": fold,
            "test_start": test_start_s,
            "test_end": test_end_s,
            "n_pre": n_pre,
            "train_end": idx.train_end,
            "val_end": idx.val_end,
            "n_test_rows": n_test,
            "val_over_train": float(len(val_arr) / max(len(train_arr), 1)),
        },
    )


def assert_no_leakage(bundle: dict, horizon: int, lookback: int = LOOKBACK) -> dict:
    """Mandatory G4 checks: window origin/targets never cross split boundaries.

    Real SCADA series may contain gaps, so we do NOT require a perfect 30-min
    arithmetic grid. We only require: (1) origin and all targets lie inside the
    split's row times, (2) last train target < val start, (3) last val target <
    test start, (4) targets strictly increasing within each window.
    """
    b = bundle["boundaries"]
    checks = {}
    for split_name, ds_key, t_max_key, next_min in (
        ("train", "train_ds", "train_t_max", b["val_t_min"]),
        ("val", "val_ds", "val_t_max", b["test_t_min"]),
        ("test", "test_ds", "test_t_max", None),
    ):
        ds = bundle[ds_key]
        if len(ds) == 0:
            raise RuntimeError(f"{split_name} dataset empty")
        split_max = b[t_max_key]
        split_min = b[t_max_key.replace("_max", "_min")]
        # Probe first, mid, last windows
        probe = sorted({0, len(ds) // 2, len(ds) - 1})
        for idx in probe:
            _x, _y, t_origin, t_target = ds[idx]
            t_origin = int(t_origin)
            t_target = np.asarray(t_target).astype(np.int64)
            if t_target.shape[0] != horizon:
                raise AssertionError(f"{split_name} horizon length mismatch at idx={idx}")
            if t_origin < split_min or t_origin > split_max:
                raise AssertionError(f"{split_name} origin {t_origin} outside split bounds")
            if int(t_target.min()) < split_min or int(t_target.max()) > split_max:
                raise AssertionError(f"{split_name} targets outside split bounds")
            if not np.all(t_target[1:] > t_target[:-1]):
                raise AssertionError(f"{split_name} targets not strictly increasing at idx={idx}")
            if int(t_target[0]) <= t_origin:
                raise AssertionError(f"{split_name} first target not after origin at idx={idx}")
        last = ds[len(ds) - 1]
        last_target_max = int(np.asarray(last[3]).max())
        if next_min is not None and last_target_max >= next_min:
            raise AssertionError(
                f"{split_name} last target {last_target_max} >= next split start {next_min}"
            )
        checks[f"{split_name}_last_target_max"] = last_target_max
        checks[f"{split_name}_n_windows"] = len(ds)
    checks["lookback"] = lookback
    checks["horizon"] = horizon
    checks["ok"] = True
    return checks
