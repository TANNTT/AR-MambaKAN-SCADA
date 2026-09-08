import argparse
import csv
import subprocess
import time
from pathlib import Path


def run_command(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def parse_test_metrics(output: str) -> tuple[float, float, float, float]:
    # Expected line:
    # test_mae=... test_rmse=... test_mape=...% test_smape=...%
    last = [ln.strip() for ln in output.splitlines() if ln.strip()]
    target = [ln for ln in last if ln.startswith("test_mae=")]
    if not target:
        raise RuntimeError(f"Cannot parse test metrics from output:\n{output}")
    line = target[-1].replace("%", "")
    parts = dict(p.split("=") for p in line.split(" "))
    return (
        float(parts["test_mae"]),
        float(parts["test_rmse"]),
        float(parts["test_mape"]),
        float(parts["test_smape"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-csv", default="artifacts/benchmark_results.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--models", nargs="+", default=["lastvalue", "mlp", "bilstm"])
    args = ap.parse_args()

    rows = []
    for model in args.models:
        cmd = [
            "python",
            "-m",
            "scada_exp.train",
            "--dataset",
            args.dataset,
            "--model",
            model,
            "--lookback",
            str(args.lookback),
            "--horizon",
            str(args.horizon),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        t0 = time.perf_counter()
        out = run_command(cmd)
        runtime_sec = time.perf_counter() - t0
        mae, rmse, mape, smape = parse_test_metrics(out)
        rows.append(
            {
                "model": model,
                "mae": mae,
                "rmse": rmse,
                "mape_pct": mape,
                "smape_pct": smape,
                "runtime_sec": runtime_sec,
            }
        )
        print(
            f"[{model}] mae={mae:.4f} rmse={rmse:.4f} mape={mape:.2f}% "
            f"smape={smape:.2f}% runtime={runtime_sec:.2f}s"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mae", "rmse", "mape_pct", "smape_pct", "runtime_sec"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved benchmark table: {out_path}")


if __name__ == "__main__":
    main()

