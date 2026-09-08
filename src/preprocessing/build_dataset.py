import argparse
import re
from pathlib import Path

import pandas as pd

from preprocessing.parser import parse_html_xls


def _year_overlap(file_name: str, y0: int, y1: int) -> bool:
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{4})-(\d{1,2})_(\d{1,2})_(\d{4})\.xls$", file_name)
    if not m:
        return False
    start_y, end_y = int(m.group(3)), int(m.group(6))
    return not (end_y < y0 or start_y > y1)


def _collect_stream(data_root: Path, station: str, code: str, prefix: str, folder: str, kind: str, y0: int, y1: int) -> pd.DataFrame:
    stream_dir = data_root / folder
    files = [
        p
        for p in stream_dir.glob(f"{prefix}_{station}*_{code}(*)_*.xls")
        if _year_overlap(p.name, y0, y1)
    ]
    frames = [parse_html_xls(str(p), kind=kind) for p in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    tcol = "pc_time" if kind != "billing" else "meter_time"
    df = df[(df[tcol].dt.year >= y0) & (df[tcol].dt.year <= y1)]
    return df.sort_values(tcol).drop_duplicates(subset=[tcol]).reset_index(drop=True)


def _safe_station_name(station: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", station).strip("_").lower()


def build_dataset(data_root: Path, station: str, code: str, out_dir: Path, y0: int, y1: int) -> Path:
    instant = _collect_stream(data_root, station, code, "Instant", "CSTT_Instant", "instant", y0, y1)
    energy = _collect_stream(data_root, station, code, "Energy", "CSTT_Energy", "energy", y0, y1)
    billing = _collect_stream(data_root, station, code, "Billing", "CSC_Billing", "billing", y0, y1)

    if instant.empty:
        raise RuntimeError("No instant data collected.")

    df = instant.copy()
    if not energy.empty:
        df = pd.merge_asof(
            df.sort_values("pc_time"),
            energy.sort_values("pc_time"),
            on="pc_time",
            tolerance=pd.Timedelta(seconds=120),
            direction="nearest",
        )

    if not billing.empty:
        billing2 = billing.rename(columns={"meter_time": "pc_time"}).sort_values("pc_time")
        df = pd.merge_asof(
            df.sort_values("pc_time"),
            billing2,
            on="pc_time",
            tolerance=pd.Timedelta(days=32),
            direction="backward",
        )

    # Simple quality cleaning for baseline reproducibility.
    for c in ["pt_kw", "qt_kvar", "ua_v", "ub_v", "uc_v", "cos_phi", "freq_hz"]:
        if c in df.columns:
            df[c] = df[c].interpolate(limit_direction="both")
    df = df.dropna(subset=["pt_kw", "qt_kvar", "pc_time"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_station_name(station)}_{code}_{y0}_{y1}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--station", required=True, help="Substation name/label as it appears in your raw data export.")
    ap.add_argument("--code", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--year-start", type=int, default=2020)
    ap.add_argument("--year-end", type=int, default=2023)
    args = ap.parse_args()

    out = build_dataset(
        data_root=Path(args.data_root),
        station=args.station,
        code=args.code,
        out_dir=Path(args.out_dir),
        y0=args.year_start,
        y1=args.year_end,
    )
    print(f"Saved dataset: {out}")


if __name__ == "__main__":
    main()

