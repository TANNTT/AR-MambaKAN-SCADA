import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

from preprocessing.parser import parse_html_xls


def _extract_code(file_name: str) -> str:
    m = re.search(r"_(D[^_(]+)\(", file_name)
    return m.group(1) if m else "UNKNOWN"


def _extract_station(file_name: str) -> str:
    m = re.match(r"Instant_(.*?)\s*\(\s*\d+\s*\)_", file_name)
    return m.group(1).strip() if m else "UNKNOWN"


def _year_overlap(file_name: str, y0: int, y1: int) -> bool:
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{4})-(\d{1,2})_(\d{1,2})_(\d{4})\.xls$", file_name)
    if not m:
        return False
    start_y, end_y = int(m.group(3)), int(m.group(6))
    return not (end_y < y0 or start_y > y1)


def _safe_ratio(n: float, d: float) -> float:
    return 0.0 if d <= 0 else n / d


def compute_quality_scores(
    data_root: Path,
    y0: int,
    y1: int,
    min_rows: int = 5000,
    max_codes: int = 0,
    max_files_per_code: int = 8,
) -> list[Dict]:
    instant_dir = data_root / "CSTT_Instant"
    files = [p for p in instant_dir.glob("Instant_*.xls") if _year_overlap(p.name, y0, y1)]
    files.sort()
    grouped_files = defaultdict(list)
    for p in files:
        grouped_files[_extract_code(p.name)].append(p)
    codes = sorted(grouped_files.keys())
    if max_codes > 0:
        codes = codes[:max_codes]

    stats = defaultdict(
        lambda: {
            "station": "UNKNOWN",
            "n_rows": 0,
            "pt_zero": 0,
            "qt_zero": 0,
            "missing": 0,
            "interval_good": 0,
            "interval_total": 0,
            "years": set(),
            "last_time": None,
            "first_time": None,
        }
    )

    for code in codes:
        code_files = grouped_files[code][:max_files_per_code] if max_files_per_code > 0 else grouped_files[code]
        for p in code_files:
            station = _extract_station(p.name)
            df = parse_html_xls(str(p), kind="instant")
            if df.empty:
                continue
            df = df[(df["pc_time"].dt.year >= y0) & (df["pc_time"].dt.year <= y1)]
            if df.empty:
                continue

            s = stats[code]
            s["station"] = station
            s["n_rows"] += len(df)
            s["pt_zero"] += int((df["pt_kw"] == 0).sum())
            s["qt_zero"] += int((df["qt_kvar"] == 0).sum())
            s["missing"] += int(df[["pt_kw", "qt_kvar", "ua_v", "ub_v", "uc_v"]].isna().sum().sum())
            s["years"].update(df["pc_time"].dt.year.unique().tolist())

            tmin = df["pc_time"].min()
            tmax = df["pc_time"].max()
            if s["first_time"] is None or tmin < s["first_time"]:
                s["first_time"] = tmin
            if s["last_time"] is None or tmax > s["last_time"]:
                s["last_time"] = tmax

            deltas = df["pc_time"].sort_values().diff().dt.total_seconds().dropna() / 60.0
            if len(deltas) > 0:
                s["interval_good"] += int(((deltas >= 29) & (deltas <= 31)).sum())
                s["interval_total"] += int(len(deltas))

    rows = []
    target_years = (y1 - y0 + 1)
    for code, s in stats.items():
        if s["n_rows"] < min_rows:
            continue
        pt_zero_rate = _safe_ratio(s["pt_zero"], s["n_rows"])
        qt_zero_rate = _safe_ratio(s["qt_zero"], s["n_rows"])
        miss_rate = _safe_ratio(s["missing"], s["n_rows"] * 5)
        cadence_rate = _safe_ratio(s["interval_good"], s["interval_total"])
        year_cov = _safe_ratio(len(s["years"]), target_years)

        # Weighted quality score in [0,100]
        score = 100.0 * (
            0.35 * (1.0 - pt_zero_rate)
            + 0.30 * (1.0 - qt_zero_rate)
            + 0.15 * (1.0 - miss_rate)
            + 0.10 * cadence_rate
            + 0.10 * year_cov
        )
        rows.append(
            {
                "station": s["station"],
                "code": code,
                "n_rows": s["n_rows"],
                "years": len(s["years"]),
                "pt_zero_pct": 100.0 * pt_zero_rate,
                "qt_zero_pct": 100.0 * qt_zero_rate,
                "missing_pct": 100.0 * miss_rate,
                "cadence_pct": 100.0 * cadence_rate,
                "quality_score": score,
                "time_start": s["first_time"],
                "time_end": s["last_time"],
            }
        )
    rows.sort(key=lambda r: (-r["quality_score"], -r["n_rows"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--year-start", type=int, default=2020)
    ap.add_argument("--year-end", type=int, default=2023)
    ap.add_argument("--min-rows", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-codes", type=int, default=0)
    ap.add_argument("--max-files-per-code", type=int, default=8)
    ap.add_argument("--out-csv", default="artifacts/meter_quality_scores.csv")
    args = ap.parse_args()

    rows = compute_quality_scores(
        data_root=Path(args.data_root),
        y0=args.year_start,
        y1=args.year_end,
        min_rows=args.min_rows,
        max_codes=args.max_codes,
        max_files_per_code=args.max_files_per_code,
    )
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "station",
        "code",
        "n_rows",
        "years",
        "pt_zero_pct",
        "qt_zero_pct",
        "missing_pct",
        "cadence_pct",
        "quality_score",
        "time_start",
        "time_end",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved quality ranking: {out_path}")
    print("station,code,quality_score,n_rows,pt_zero_pct,qt_zero_pct")
    for r in rows[: args.top_k]:
        print(
            f"{r['station']},{r['code']},{r['quality_score']:.2f},"
            f"{r['n_rows']},{r['pt_zero_pct']:.2f},{r['qt_zero_pct']:.2f}"
        )


if __name__ == "__main__":
    main()

