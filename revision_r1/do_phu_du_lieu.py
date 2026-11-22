#!/usr/bin/env python3
"""Time-coverage audit for IET GTD S1 meters (CPU-only, summary stats).

Assumptions (this run):
1. Experiment inputs are built per-meter CSVs ``*_2020_2023.csv`` under
   $SCADA_ARTIFACTS_DIR trees (not raw SCADA XLS). Schema: pc_time, pt_kw, ...
2. Anonymized labels M01..M33 come from meter_anonymization_map.csv (real codes
   used only to locate files; never written to summary.json).
3. Temporal split matches src/evaluation/train.py split_indices: train 0.7,
   val 0.1 by row index on the CSV order (already sorted by pc_time at build).
4. Northern-Vietnam seasons: dong={12,1,2}, xuan={3,4}, he={5,6,7,8}, thu={9,10,11}.

Two interpretations considered for "which file":
  A) Prefer q1_multihorizon_sota/h48/batch_datasets for the 10 main meters, else
     artifacts/batch_datasets, else multistation_datasets (chosen: matches paper
     experiment layout).
  B) Always use artifacts/batch_datasets / multistation only.
Chose A because M01-M10 live in the h48 SOTA batch used for multi-horizon runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys as _sys
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from _paths import require_env, env_or_none  # noqa: E402

import pandas as pd

SEASON_DEF = {
    "dong": [12, 1, 2],
    "xuan": [3, 4],
    "he": [5, 6, 7, 8],
    "thu": [9, 10, 11],
}
MONTH_TO_SEASON = {m: s for s, months in SEASON_DEF.items() for m in months}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
GAP_THRESHOLD = pd.Timedelta(days=1)
GOOD_DAY_MIN_SAMPLES = 40  # of 48 half-hour slots


def split_indices(n: int, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO):
    """Mirror AR-MambaKAN-SCADA/src/evaluation/train.py::split_indices."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def load_anonymization_map(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def index_dataset_csvs(search_roots: list[Path]) -> dict[str, list[Path]]:
    """Map real meter code -> candidate CSV paths (*_CODE_2020_2023.csv)."""
    by_code: dict[str, list[Path]] = defaultdict(list)
    seen: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*_2020_2023.csv"):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            stem = p.name[: -len("_2020_2023.csv")]
            code = stem.split("_")[-1]
            by_code[code].append(p)
    return by_code


def choose_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None

    def score(p: Path) -> float:
        s = str(p)
        sc = 0.0
        if "smoke" in s:
            sc -= 200
        if "/q1_multihorizon_sota/h48/batch_datasets/" in s:
            sc += 100
        if "/artifacts/batch_datasets/" in s and "/q1_" not in s:
            sc += 80
        if "/artifacts/multistation_datasets/" in s and "/q1_" not in s:
            sc += 60
        if "artifacts_nocal" in s:
            sc -= 20
        sc -= len(s) * 0.001
        return sc

    return max(paths, key=score)


def find_gaps(times: pd.Series, threshold: pd.Timedelta = GAP_THRESHOLD) -> list[dict]:
    """Gaps where consecutive pc_time differ by more than threshold."""
    if len(times) < 2:
        return []
    t = times.sort_values().reset_index(drop=True)
    deltas = t.diff()
    gaps = []
    for i in deltas.index[1:]:
        d = deltas.iloc[i]
        if pd.isna(d) or d <= threshold:
            continue
        start = t.iloc[i - 1]
        end = t.iloc[i]
        gaps.append(
            {
                "start": start.isoformat(sep=" "),
                "end": end.isoformat(sep=" "),
                "duration_hours": round(d.total_seconds() / 3600.0, 3),
                "duration_days": round(d.total_seconds() / 86400.0, 3),
            }
        )
    return gaps


def analyze_meter(anon_label: str, path: Path) -> dict:
    df = pd.read_csv(path, usecols=["pc_time"], parse_dates=["pc_time"])
    df = df.dropna(subset=["pc_time"]).sort_values("pc_time").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {
            "anon_label": anon_label,
            "source_path": str(path),
            "n_rows": 0,
            "error": "empty_after_parse",
        }

    train_end, val_end = split_indices(n)
    t0 = df["pc_time"].iloc[0]
    t1 = df["pc_time"].iloc[-1]
    train_end_time = df["pc_time"].iloc[train_end - 1] if train_end > 0 else None
    val_end_time = df["pc_time"].iloc[val_end - 1] if val_end > 0 else None
    # first test timestamp (or None if empty test)
    test_start_time = df["pc_time"].iloc[val_end] if val_end < n else None

    ym = df["pc_time"].dt.to_period("M").astype(str)
    rows_per_month = ym.value_counts().sort_index().to_dict()
    rows_per_month = {k: int(v) for k, v in rows_per_month.items()}

    day = df["pc_time"].dt.floor("D")
    day_counts = df.groupby([ym, day]).size()
    good_days = day_counts[day_counts >= GOOD_DAY_MIN_SAMPLES]
    good_days_per_month = good_days.groupby(level=0).size().to_dict()
    good_days_per_month = {str(k): int(v) for k, v in good_days_per_month.items()}

    gaps = find_gaps(df["pc_time"])

    test = df.iloc[val_end:].copy()
    test_ym = test["pc_time"].dt.to_period("M").astype(str)
    test_rows_per_month = test_ym.value_counts().sort_index().to_dict()
    test_rows_per_month = {k: int(v) for k, v in test_rows_per_month.items()}

    test_season_rows = {s: 0 for s in SEASON_DEF}
    if len(test):
        seasons = test["pc_time"].dt.month.map(MONTH_TO_SEASON)
        for s, c in seasons.value_counts().items():
            test_season_rows[str(s)] = int(c)

    return {
        "anon_label": anon_label,
        "source_path": str(path),
        "n_rows": int(n),
        "time_start": t0.isoformat(sep=" "),
        "time_end": t1.isoformat(sep=" "),
        "n_months_with_data": len(rows_per_month),
        "rows_per_month": rows_per_month,
        "good_days_per_month_ge40of48": good_days_per_month,
        "gaps_gt_1_day": gaps,
        "n_gaps_gt_1_day": len(gaps),
        "split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "n_train": int(train_end),
            "n_val": int(val_end - train_end),
            "n_test": int(n - val_end),
            "train_end_index": int(train_end),
            "val_end_index": int(val_end),
            "pc_time_at_train_end": None
            if train_end_time is None
            else train_end_time.isoformat(sep=" "),
            "pc_time_at_val_end": None
            if val_end_time is None
            else val_end_time.isoformat(sep=" "),
            "pc_time_test_start": None
            if test_start_time is None
            else test_start_time.isoformat(sep=" "),
        },
        "test_rows_per_month": test_rows_per_month,
        "test_rows_per_season": test_season_rows,
    }


def self_check_demo(demo_dir: Path) -> None:
    """Verify monthly row counts on DEMO0001 against a hand count for one month."""
    path = demo_dir / "DEMO0001.csv"
    if not path.is_file():
        raise SystemExit(f"self-check failed: missing {path}")

    df = pd.read_csv(path, usecols=["pc_time"], parse_dates=["pc_time"])
    df = df.dropna(subset=["pc_time"]).sort_values("pc_time").reset_index(drop=True)
    target_month = "2023-01"
    mask = df["pc_time"].dt.to_period("M").astype(str) == target_month
    hand = int(mask.sum())
    by_script = int((df["pc_time"].dt.to_period("M").astype(str) == target_month).sum())
    # independent hand count: iterate rows
    hand2 = 0
    for ts in df["pc_time"]:
        if f"{ts.year:04d}-{ts.month:02d}" == target_month:
            hand2 += 1
    if hand != by_script or hand != hand2:
        raise SystemExit(
            f"self-check FAILED month {target_month}: hand={hand} hand2={hand2} script={by_script}"
        )
    # also exercise analyze_meter path (rename temporarily via copy-free: analyze then overwrite label)
    stats = analyze_meter("DEMO0001", path)
    if stats["rows_per_month"].get(target_month) != hand:
        raise SystemExit(
            f"self-check FAILED analyze_meter: got {stats['rows_per_month'].get(target_month)} want {hand}"
        )
    print(
        f"[self-check OK] DEMO0001 {target_month}: {hand} rows "
        f"(n_rows={stats['n_rows']}, months={stats['n_months_with_data']})",
        flush=True,
    )


def git_rev(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def print_table(meters: dict[str, dict]) -> None:
    header = (
        f"{'label':<5} {'start':<20} {'end':<20} {'n_mo':>4} "
        f"{'val_end':<20} {'test_months'}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label in sorted(meters):
        m = meters[label]
        if m.get("error") or m.get("missing"):
            print(f"{label:<5} MISSING/ERROR {m.get('note', m.get('error', ''))}", flush=True)
            continue
        test_months = ",".join(m["test_rows_per_month"].keys())
        print(
            f"{label:<5} {m['time_start']:<20} {m['time_end']:<20} "
            f"{m['n_months_with_data']:>4} {m['split']['pc_time_at_val_end']:<20} "
            f"{test_months}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="SCADA meter time-coverage audit (anonymized).")
    ap.add_argument(
        "--map",
        default=None,
        help="CSV with real_code,anon_label,used_in (default: $SCADA_ANON_MAP)",
    )
    ap.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Root(s) to search for *_2020_2023.csv (repeatable). Default: $SCADA_ARTIFACTS_DIR [+$SCADA_ARTIFACTS_NOCAL_DIR].",
    )
    ap.add_argument(
        "--demo-dir",
        default=None,
        help="Synthetic demo_data for self-check (default: $SCADA_DEMO_DATA)",
    )
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--demo-only", action="store_true", help="Only self-check demo_data, then exit")
    ap.add_argument(
        "--out",
        default=None,
        help="summary.json path (default: results/do-phu-du-lieu/summary.json under this script)",
    )
    args = ap.parse_args()

    paper_root = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else paper_root / "results" / "do-phu-du-lieu" / "summary.json"

    if not args.skip_self_check:
        self_check_demo(Path(args.demo_dir))
    if args.demo_only:
        return

    map_path = Path(args.map)
    if not map_path.is_file():
        raise SystemExit(
            f"Anonymization map not found at {map_path}. "
            "Also checked nothing else. Stop."
        )
    rows = load_anonymization_map(map_path)

    if args.map is None:
        args.map = str(require_env("SCADA_ANON_MAP"))
    if args.demo_dir is None:
        args.demo_dir = str(require_env("SCADA_DEMO_DATA"))

    if args.search_root:
        search_roots = [Path(p) for p in args.search_root]
    else:
        search_roots = [require_env("SCADA_ARTIFACTS_DIR")]
        nocal = env_or_none("SCADA_ARTIFACTS_NOCAL_DIR")
        if nocal is not None:
            search_roots.append(nocal)

    searched = [str(p) for p in search_roots]
    by_code = index_dataset_csvs(search_roots)

    meters: dict[str, dict] = {}
    missing: list[str] = []
    for row in rows:
        anon = row["anon_label"].strip()
        code = row["real_code"].strip()
        used_in = row.get("used_in", "")
        paths = by_code.get(code, [])
        chosen = choose_path(paths)
        if chosen is None:
            missing.append(anon)
            meters[anon] = {
                "anon_label": anon,
                "missing": True,
                "used_in": used_in,
                "note": "no *_2020_2023.csv found under search roots",
                "n_candidate_paths": 0,
            }
            continue
        stats = analyze_meter(anon, chosen)
        stats["used_in"] = used_in
        stats["n_candidate_paths"] = len(paths)
        meters[anon] = stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    env_path = out_path.parent / "env.txt"
    env_path.write_text(
        f"python={sys.version.replace(chr(10), ' ')}\n"
        f"pandas={pd.__version__}\n"
        f"platform={platform.platform()}\n"
        f"hostname={platform.node()}\n"
    )

    summary = {
        "run_name": "do-phu-du-lieu",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "git_commit_paper_dir": git_rev(paper_root),
        "git_commit_armambakan": git_rev(require_env("SCADA_PUBLIC_REPO")),
        "season_definition": {
            "region": "Northern Vietnam climate seasons (mien Bac)",
            "dong": SEASON_DEF["dong"],
            "xuan": SEASON_DEF["xuan"],
            "he": SEASON_DEF["he"],
            "thu": SEASON_DEF["thu"],
            "note": "dong=Dec-Jan-Feb; xuan=Mar-Apr; he=May-Aug; thu=Sep-Nov",
        },
        "split_definition": {
            "source": "AR-MambaKAN-SCADA/src/evaluation/train.py::split_indices",
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": round(1.0 - TRAIN_RATIO - VAL_RATIO, 6),
            "by": "row_index",
        },
        "good_day_rule": f">={GOOD_DAY_MIN_SAMPLES} samples in a calendar day (of 48 half-hour slots)",
        "gap_rule": "consecutive pc_time delta > 1 day",
        "search_roots": searched,
        "map_path": str(map_path),
        "n_meters_in_map": len(rows),
        "n_meters_found": len(rows) - len(missing),
        "n_meters_missing": len(missing),
        "missing_labels": missing,
        "quality_score_note": (
            "artifacts/meter_quality_scores.csv n_rows is from quality_score.py with "
            "max_files_per_code=8 (sampled XLS), NOT the full built *_2020_2023.csv. "
            "Do not equate quality n_rows with experiment coverage."
        ),
        "meters": meters,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out_path}", flush=True)
    print(
        f"Found {summary['n_meters_found']}/{summary['n_meters_in_map']} meters; "
        f"missing={missing}",
        flush=True,
    )
    print_table(meters)


if __name__ == "__main__":
    main()
