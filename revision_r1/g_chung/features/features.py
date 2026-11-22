# Copied from AR-MambaKAN-SCADA @ 8582ecc937b45081f5c4116694c73d685f6c8508
# Source file: src/features/features.py (read-only original; do not edit upstream)
"""Offline, reproducible calendar feature engineering for SCADA forecasting.

These features are *known in advance* (unlike measured electrical signals), so they are
legitimate exogenous inputs for forecasting. We hard-code Vietnamese public holidays for
2020-2023 (the span of the dataset) to avoid any network/lunar-calendar dependency and to
guarantee exact reproducibility.
"""
from __future__ import annotations

import datetime as _dt
import re as _re
import unicodedata as _ud

import numpy as np
import pandas as pd

# Vietnamese public holidays 2020-2023 (incl. Tet windows and observed/compensatory days).
# Source: official Vietnamese government holiday announcements for each year.
_VN_HOLIDAY_DATES = [
    # --- 2020 ---
    (2020, 1, 1),  # New Year's Day
    *[(2020, 1, d) for d in range(23, 30)],  # Tet (mung 1 = Jan 25)
    (2020, 4, 2),  # Hung Kings (10/3 lunar)
    (2020, 4, 30),  # Reunification Day
    (2020, 5, 1),  # Labour Day
    (2020, 9, 2),  # National Day
    # --- 2021 ---
    (2021, 1, 1),
    *[(2021, 2, d) for d in range(10, 17)],  # Tet (mung 1 = Feb 12)
    (2021, 4, 21),  # Hung Kings
    (2021, 4, 30),
    (2021, 5, 1),
    (2021, 5, 2),  # observed
    (2021, 5, 3),  # observed
    (2021, 9, 1),
    (2021, 9, 2),
    # --- 2022 ---
    (2022, 1, 1),
    (2022, 1, 2),  # observed
    (2022, 1, 3),  # observed
    *[(2022, 1, d) for d in range(29, 32)],  # Tet weekend + start (mung 1 = Feb 1)
    *[(2022, 2, d) for d in range(1, 7)],  # Tet core + weekend
    (2022, 4, 10),  # Hung Kings (10/3 lunar)
    (2022, 4, 11),  # observed
    (2022, 4, 30),
    (2022, 5, 1),
    (2022, 5, 2),  # observed
    (2022, 5, 3),  # observed
    (2022, 9, 1),
    (2022, 9, 2),
    # --- 2023 ---
    (2023, 1, 1),
    (2023, 1, 2),  # observed
    *[(2023, 1, d) for d in range(20, 27)],  # Tet (mung 1 = Jan 22)
    (2023, 4, 29),  # Hung Kings + long break start
    (2023, 4, 30),
    (2023, 5, 1),
    (2023, 5, 2),  # observed
    (2023, 5, 3),  # observed
    *[(2023, 9, d) for d in range(1, 5)],  # National Day 4-day break
]

VN_HOLIDAYS = frozenset(_dt.date(y, m, d) for (y, m, d) in _VN_HOLIDAY_DATES)

CALENDAR_FEATURE_COLS = [
    "cal_hour_sin",
    "cal_hour_cos",
    "cal_dow_sin",
    "cal_dow_cos",
    "cal_month_sin",
    "cal_month_cos",
    "cal_is_weekend",
    "cal_is_holiday",
]


def add_calendar_features(df: pd.DataFrame, time_col: str = "pc_time") -> pd.DataFrame:
    """Append cyclical time-of-day/week/year + weekend/holiday flags derived from `time_col`.

    All features are deterministic functions of the timestamp, hence safe to use for
    forecasting (no leakage of the target). Cyclical encodings keep continuity across the
    24h/7d/12mo boundaries.
    """
    if time_col not in df.columns:
        raise KeyError(f"time column {time_col!r} not found in dataframe")
    t = pd.to_datetime(df[time_col])
    hour = t.dt.hour + t.dt.minute / 60.0
    dow = t.dt.dayofweek
    month = t.dt.month

    out = df.copy()
    out["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    out["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    out["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    out["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    out["cal_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    out["cal_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    out["cal_is_weekend"] = (dow >= 5).astype(np.float32)
    out["cal_is_holiday"] = t.dt.date.isin(VN_HOLIDAYS).astype(np.float32)
    return out


# --- Weather (exogenous) features ---------------------------------------------------------
# Canonical weather columns (single source of truth shared with fetch_weather.py).
WEATHER_FEATURE_COLS = [
    "wx_temp",
    "wx_humidity",
    "wx_cloud",
    "wx_radiation",
    "wx_wind",
    "wx_precip",
    "wx_apptemp",
]


def station_slug(name: str) -> str:
    """ASCII, filesystem-safe slug for a station name (diacritics stripped).

    Used by both the weather fetcher and the join so file names match deterministically.
    e.g. 'E10.4 Tía' -> 'e10_4_tia'.
    """
    s = _ud.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = _re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s


def add_weather_features(
    df: pd.DataFrame,
    weather_csv: str,
    time_col: str = "pc_time",
    tolerance_min: int = 90,
) -> pd.DataFrame:
    """Join hourly weather (nearest timestamp) onto the SCADA frame.

    The result is sorted chronologically by `time_col` (required for windowing). Weather gaps
    beyond `tolerance_min` are interpolated/filled so there are no NaNs in the feature columns.
    """
    w = pd.read_csv(weather_csv, parse_dates=["time"]).sort_values("time")
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values(time_col).reset_index(drop=True)
    merged = pd.merge_asof(
        out,
        w,
        left_on=time_col,
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )
    for c in WEATHER_FEATURE_COLS:
        if c in merged.columns:
            merged[c] = merged[c].astype("float32").interpolate(limit_direction="both")
    if "time" in merged.columns:
        merged = merged.drop(columns=["time"])
    return merged
