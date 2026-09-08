"""Calendar-feature ablation, AR-MambaKAN, local training, all 4 horizons. Trains once with and
once without the deterministic calendar/holiday features and reports the local-training MAE change
(Table 17 / Section 7.8 of the paper -- the paper's headline calendar-interaction finding, -9.9%).
Resumable.

Usage:
    PYTHONPATH=../../src:../pooled python run_calendar_ablation.py \
        --data-dir ../../demo_data --out-csv ../../results/calendar_ablation_demo.csv
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pooled"))
import common_pooled as cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "demo_data"))
    ap.add_argument("--out-csv", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "calendar_ablation.csv"))
    ap.add_argument("--horizons", nargs="+", type=int, default=[4, 48, 96, 336])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meters = cp.discover_meters(args.data_dir)
    if not meters:
        raise SystemExit(f"No meter CSVs found in {args.data_dir}")

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for row in csv.DictReader(f):
                done.add((row["code"], int(row["seed"]), int(row["horizon"]), row["calendar"]))
    else:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["code", "seed", "horizon", "calendar", "mae"])

    for h in args.horizons:
        for calendar_on in (False, True):
            cal_label = "with_calendar" if calendar_on else "no_calendar"
            for code, path in meters.items():
                for seed in args.seeds:
                    if (code, seed, h, cal_label) in done:
                        continue
                    t0 = time.time()
                    mae = cp.train_local_armambakan(path, horizon=h, calendar_features=calendar_on, seed=seed, device=args.device)
                    dt = time.time() - t0
                    with open(args.out_csv, "a", newline="") as f:
                        csv.writer(f).writerow([code, seed, h, cal_label, mae])
                    print(f"[calendar-ablation] H={h} {cal_label} {code} seed={seed} mae={mae:.2f} ({dt:.0f}s)", flush=True)

    print("DONE calendar ablation experiment")


if __name__ == "__main__":
    main()
