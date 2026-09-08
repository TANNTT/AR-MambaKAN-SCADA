"""Generates a small SYNTHETIC demo dataset with the same column schema as the real EVN SCADA
telemetry used in the paper, so the pipeline in this repository can be run end-to-end without
access to the (proprietary, non-public) real data. All values below are randomly generated with
plausible diurnal shape and magnitude -- they are NOT derived from, subsampled from, or otherwise
related to any real substation's telemetry.

Run once to (re)populate demo_data/*.csv:
    python generate_synthetic_demo.py
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_METERS = 4
N_ROWS = 4000  # ~83 days at 30-min cadence -- enough for lookback=336 + horizon=336 windows
CADENCE_MIN = 30

METER_CODES = ["DEMO0001", "DEMO0002", "DEMO0003", "DEMO0004"]


def make_meter(code: str, base_load_kw: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-01", periods=N_ROWS, freq=f"{CADENCE_MIN}min")
    hour = t.hour + t.minute / 60.0
    dow = t.dayofweek

    # Diurnal + weekly shape + noise, clipped to stay positive -- purely synthetic.
    diurnal = 0.5 + 0.5 * np.sin((hour - 6) / 24 * 2 * np.pi - np.pi / 2)
    weekday_factor = np.where(dow < 5, 1.0, 0.85)
    noise = rng.normal(0, 0.05, N_ROWS)
    pt_kw = np.clip(base_load_kw * diurnal * weekday_factor * (1 + noise) + rng.normal(0, 2, N_ROWS), 1, None)
    qt_kvar = np.clip(pt_kw * rng.uniform(0.25, 0.45) + rng.normal(0, 1, N_ROWS), -50, None)

    ua_v = 220 + rng.normal(0, 2, N_ROWS)
    ub_v = 220 + rng.normal(0, 2, N_ROWS)
    uc_v = 220 + rng.normal(0, 2, N_ROWS)
    cos_phi = np.clip(pt_kw / np.sqrt(pt_kw ** 2 + qt_kvar ** 2 + 1e-6), 0.7, 1.0)
    freq_hz = 50 + rng.normal(0, 0.02, N_ROWS)

    return pd.DataFrame({
        "pc_time": t,
        "pt_kw": pt_kw.round(2),
        "qt_kvar": qt_kvar.round(2),
        "ua_v": ua_v.round(2),
        "ub_v": ub_v.round(2),
        "uc_v": uc_v.round(2),
        "cos_phi": cos_phi.round(4),
        "freq_hz": freq_hz.round(3),
    })


def main():
    import os
    out_dir = os.path.dirname(__file__)
    for i, code in enumerate(METER_CODES):
        df = make_meter(code, base_load_kw=200 + 150 * i, seed=100 + i)
        path = os.path.join(out_dir, f"{code}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
