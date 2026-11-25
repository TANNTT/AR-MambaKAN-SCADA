"""Fetch real historical weather (Open-Meteo ERA5 archive) per station.

Run this ONCE from a terminal that has internet access (a sandboxed environment cannot reach the
network). It reads station coordinates from a CSV and writes one weather CSV per station to
`--out-dir`, with the canonical `wx_*` columns expected by `scada_exp.features`.

Example:
    PYTHONPATH=src python -m scada_exp.fetch_weather \
        --coords-csv config/station_coords.csv \
        --out-dir artifacts/weather \
        --start 2020-01-01 --end 2023-12-31

Data source: Open-Meteo Historical Weather API (ERA5 reanalysis), free, no API key.
NOTE (honesty): ERA5 reanalysis is used as a proxy for weather conditions. Using it as a
forecast input assumes a near-perfect weather forecast (an oracle assumption that should be
documented in the manuscript).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from features.features import WEATHER_FEATURE_COLS, station_slug

# Open-Meteo hourly variable -> canonical wx_* column
_OPEN_METEO_VARS = {
    "temperature_2m": "wx_temp",
    "relative_humidity_2m": "wx_humidity",
    "cloud_cover": "wx_cloud",
    "shortwave_radiation": "wx_radiation",
    "wind_speed_10m": "wx_wind",
    "precipitation": "wx_precip",
    "apparent_temperature": "wx_apptemp",
}

_API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_one(lat: float, lon: float, start: str, end: str, timezone: str, retries: int = 4) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(_OPEN_METEO_VARS.keys()),
        "timezone": timezone,
    }
    url = _API + "?" + urllib.parse.urlencode(params)
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return json.load(resp)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 5 * (i + 1)
            print(f"  retry {i + 1}/{retries} after error: {e} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed for ({lat},{lon}): {last_err}")


def payload_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for src, dst in _OPEN_METEO_VARS.items():
        df[dst] = hourly.get(src)
    # ensure all canonical columns present
    for c in WEATHER_FEATURE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[["time", *WEATHER_FEATURE_COLS]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords-csv", default="config/station_coords.csv")
    ap.add_argument("--out-dir", default="artifacts/weather")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--timezone", default="Asia/Bangkok", help="UTC+7 for Vietnam local time")
    ap.add_argument("--sleep", type=float, default=1.0, help="Pause between requests (politeness)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = pd.read_csv(args.coords_csv)
    if not {"station", "lat", "lon"}.issubset(coords.columns):
        raise ValueError("coords CSV must have columns: station, lat, lon")

    n_ok = 0
    for _, row in coords.iterrows():
        slug = station_slug(row["station"])
        dst = out_dir / f"{slug}.csv"
        if dst.exists() and not args.overwrite:
            print(f"skip (exists): {dst}")
            n_ok += 1
            continue
        print(f"fetch {row['station']}  ({row['lat']},{row['lon']}) ...", flush=True)
        payload = fetch_one(float(row["lat"]), float(row["lon"]), args.start, args.end, args.timezone)
        df = payload_to_df(payload)
        df.to_csv(dst, index=False)
        print(f"  saved {dst}  ({len(df)} hourly rows)")
        n_ok += 1
        time.sleep(args.sleep)

    print(f"done: {n_ok}/{len(coords)} stations -> {out_dir}")


if __name__ == "__main__":
    main()
