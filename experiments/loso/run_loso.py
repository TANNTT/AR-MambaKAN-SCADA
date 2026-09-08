"""Leave-One-Substation-Out (LOSO) generalization test, AR-MambaKAN, H=48 (24h). For each meter in
turn, trains a pooled model on every OTHER meter only, then evaluates on the held-out meter's own
test split. Tests whether pooling generalizes to a substation with no history in the training pool,
as opposed to only benefiting substations that are part of the pool (Table 5 / Section 7.1.3 of the
paper). Resumable.

Usage:
    PYTHONPATH=../../src:../pooled python run_loso.py \
        --data-dir ../../demo_data --out-csv ../../results/loso_demo.csv
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
    ap.add_argument("--out-csv", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "loso.csv"))
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meters = cp.discover_meters(args.data_dir)
    codes = list(meters.keys())
    if len(codes) < 3:
        raise SystemExit(f"LOSO needs at least 3 meter CSVs in {args.data_dir}, found {len(codes)}")

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for row in csv.DictReader(f):
                done.add((row["heldout_code"], int(row["seed"])))
    else:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["heldout_code", "seed", "horizon", "mae", "pooled_n"])

    for heldout in codes:
        pool_codes = [c for c in codes if c != heldout]
        for seed in args.seeds:
            if (heldout, seed) in done:
                continue
            t0 = time.time()
            results, pooled_n = cp.train_pooled_armambakan(
                pool_codes, meters, horizon=args.horizon, calendar_features=True, seed=seed,
                eval_codes=[heldout], per_meter_train_budget=None, device=args.device,
            )
            dt = time.time() - t0
            mae = results[heldout]
            with open(args.out_csv, "a", newline="") as f:
                csv.writer(f).writerow([heldout, seed, args.horizon, mae, pooled_n])
            print(f"[loso] heldout={heldout} seed={seed} mae={mae:.2f} pooled_n={pooled_n} ({dt:.0f}s)", flush=True)

    print("DONE loso experiment")


if __name__ == "__main__":
    main()
