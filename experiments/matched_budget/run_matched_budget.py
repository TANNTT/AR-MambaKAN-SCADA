"""Data-budget-matched pooling, AR-MambaKAN. For each horizon, subsamples the pooled training set
down to the same total window count as a single local model, so the pooled-vs-local comparison
cannot be explained by sample size alone (Table 4 / Section 7.1.2 of the paper). Resumable: reruns
skip (horizon, seed) pairs already present in --out-csv.

Usage:
    PYTHONPATH=../../src:../pooled python run_matched_budget.py \
        --data-dir ../../demo_data --out-csv ../../results/matched_budget_demo.csv
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
    ap.add_argument("--out-csv", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "matched_budget.csv"))
    ap.add_argument("--horizons", nargs="+", type=int, default=[4, 48, 96, 336])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meters = cp.discover_meters(args.data_dir)
    codes = list(meters.keys())
    if len(codes) < 2:
        raise SystemExit(f"Need at least 2 meter CSVs in {args.data_dir}, found {len(codes)}")

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv) as f:
            for row in csv.DictReader(f):
                done.add((int(row["horizon"]), int(row["seed"])))
    else:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(["horizon", "seed", "code", "mae", "pooled_n", "per_meter_budget"])

    for h in args.horizons:
        # Average local training-window count across all discovered meters at this horizon
        # (calendar features on, matching the local protocol used everywhere else in the paper).
        local_ns = [len(cp.load_meter_windows(meters[c], h, calendar_features=True)["train_ds"]) for c in codes]
        avg_local_n = sum(local_ns) / len(local_ns)
        per_meter_budget = round(avg_local_n / len(codes))
        print(f"H={h}: avg_local_train_n={avg_local_n:.0f} per_meter_budget={per_meter_budget} "
              f"target_total={per_meter_budget * len(codes)}", flush=True)

        for seed in args.seeds:
            if (h, seed) in done:
                continue
            t0 = time.time()
            results, pooled_n = cp.train_pooled_armambakan(
                codes, meters, horizon=h, calendar_features=True, seed=seed,
                eval_codes=codes, per_meter_train_budget=per_meter_budget, device=args.device,
            )
            dt = time.time() - t0
            with open(args.out_csv, "a", newline="") as f:
                w = csv.writer(f)
                for code, mae in results.items():
                    w.writerow([h, seed, code, mae, pooled_n, per_meter_budget])
            print(f"[matched-budget] H={h} seed={seed} pooled_n={pooled_n} "
                  f"mean_mae={sum(results.values()) / len(results):.2f} ({dt:.0f}s)", flush=True)

    print("DONE matched-budget experiment")


if __name__ == "__main__":
    main()
