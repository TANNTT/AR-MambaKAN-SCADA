"""Pooled multi-substation training: one shared model fit on the concatenated training windows of
every discovered meter, evaluated separately per meter. Reproduces the 'Pooled' columns of Table 1
-- the paper's headline comparison.

Usage:
    PYTHONPATH=../../src:. python run_pooled.py \
        --data-dir ../../demo_data --out-csv ../../results/pooled_demo.csv
"""
import argparse
import csv
import os
import time

import common_pooled as cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "demo_data"))
    ap.add_argument("--out-csv", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "pooled.csv"))
    ap.add_argument("--horizons", nargs="+", type=int, default=[4, 48, 96, 336])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meters = cp.discover_meters(args.data_dir)
    codes = list(meters.keys())
    if len(codes) < 2:
        raise SystemExit(f"Need at least 2 meter CSVs in {args.data_dir}, found {len(codes)}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "code", "seed", "pooled_mae"])
        for h in args.horizons:
            for seed in args.seeds:
                t0 = time.time()
                results, pooled_n = cp.train_pooled_armambakan(
                    codes, meters, horizon=h, calendar_features=True, seed=seed,
                    eval_codes=codes, per_meter_train_budget=None, device=args.device,
                )
                for code, mae in results.items():
                    w.writerow([h, code, seed, mae])
                f.flush()
                print(f"[pooled] H={h} seed={seed} pooled_n={pooled_n} "
                      f"mean_mae={sum(results.values())/len(results):.2f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"DONE pooled training, saved {args.out_csv}")


if __name__ == "__main__":
    main()
