"""Per-meter local training baseline: one model fit and evaluated independently per meter, the
predominant paradigm in prior joint P-Q forecasting work (Section 2/4.2 of the paper). Reproduces
the 'Local' columns of Table 1.

Usage:
    PYTHONPATH=../../src:../pooled python run_local.py \
        --data-dir ../../demo_data --out-csv ../../results/local_demo.csv
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
    ap.add_argument("--out-csv", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "local.csv"))
    ap.add_argument("--horizons", nargs="+", type=int, default=[4, 48, 96, 336])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meters = cp.discover_meters(args.data_dir)
    if not meters:
        raise SystemExit(f"No meter CSVs found in {args.data_dir}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "code", "seed", "mae"])
        for h in args.horizons:
            for code, path in meters.items():
                for seed in args.seeds:
                    t0 = time.time()
                    mae = cp.train_local_armambakan(path, horizon=h, calendar_features=True, seed=seed, device=args.device)
                    w.writerow([h, code, seed, mae])
                    f.flush()
                    print(f"[local] H={h} {code} seed={seed} mae={mae:.2f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"DONE local training, saved {args.out_csv}")


if __name__ == "__main__":
    main()
