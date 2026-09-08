"""Shared helpers for benchmark / ablation subprocess training."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import pandas as pd

from features.features import station_slug
from models.model_factory import default_grad_clip


def run_subprocess(cmd: list[str], *, stream: bool = True) -> str:
    """Run a command; when stream=True, print output live and return captured text."""
    if stream:
        print("$ " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        chunks: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return "".join(chunks)
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def resolve_weather_csv(weather_dir: str, station: str) -> str:
    """Return the per-station weather CSV path inside `weather_dir`, or "" if not found."""
    if not weather_dir:
        return ""
    path = Path(weather_dir) / f"{station_slug(station)}.csv"
    return str(path) if path.exists() else ""


def add_train_hparams(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=0.001)
    ap.add_argument("--mape-eps", type=float, default=1.0, help="|y| threshold (kW/kVAR) for masked MAPE")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="One or more random seeds")
    ap.add_argument("--num-threads", type=int, default=8, help="CPU thread cap per training run")
    ap.add_argument(
        "--calendar-features",
        action="store_true",
        help="Add deterministic calendar features (time-of-day/week/year, weekend, VN holiday).",
    )
    ap.add_argument(
        "--weather-dir",
        default="",
        help="Dir of per-station weather CSVs (from fetch_weather); joined as wx_* features.",
    )


def build_train_cmd(
    dataset: str,
    model: str,
    lookback: int,
    horizon: int,
    args: argparse.Namespace,
    seed: int = 42,
    weather_csv: str = "",
    *,
    grad_clip: float | None = None,
    lr: float | None = None,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "scada_exp.train",
        "--dataset",
        dataset,
        "--model",
        model,
        "--lookback",
        str(lookback),
        "--horizon",
        str(horizon),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(lr if lr is not None else args.lr),
        "--mape-eps",
        str(args.mape_eps),
        "--seed",
        str(seed),
        "--num-threads",
        str(getattr(args, "num_threads", 8)),
        "--grad-clip",
        str(grad_clip if grad_clip is not None else default_grad_clip(model)),
    ]
    if getattr(args, "early_stopping_patience", 0) > 0:
        cmd.extend(
            [
                "--early-stopping-patience",
                str(args.early_stopping_patience),
                "--min-delta",
                str(args.min_delta),
            ]
        )
    if getattr(args, "calendar_features", False):
        cmd.append("--calendar-features")
    if weather_csv:
        cmd.extend(["--weather-csv", weather_csv])
    return cmd


def parse_test_metrics(output: str) -> tuple[float, float, float, float]:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    target = [ln for ln in lines if ln.startswith("test_mae=")]
    if not target:
        tail = "\n".join(lines[-30:]) if lines else "(empty output)"
        raise RuntimeError(f"Cannot parse test metrics from output (last lines):\n{tail}")
    line = target[-1].replace("%", "")
    parts = dict(p.split("=") for p in line.split(" "))
    metrics = (
        float(parts["test_mae"]),
        float(parts["test_rmse"]),
        float(parts["test_mape"]),
        float(parts["test_smape"]),
    )
    if not metrics_are_finite(metrics):
        raise RuntimeError(f"Non-finite test metrics: {line}")
    return metrics


def metrics_are_finite(metrics: tuple[float, float, float, float]) -> bool:
    return all(math.isfinite(v) for v in metrics)


def load_valid_benchmark_rows(out_csv: Path) -> list[dict]:
    """Rows with finite MAE (invalid / NaN runs are dropped on resume)."""
    if not out_csv.exists():
        return []
    df = pd.read_csv(out_csv)
    if "mae" not in df.columns:
        return []
    mae = pd.to_numeric(df["mae"], errors="coerce")
    valid = df[mae.map(lambda v: math.isfinite(v) if pd.notna(v) else False)]
    return valid.to_dict(orient="records")


def benchmark_done_keys(out_csv: Path) -> set[tuple[str, str, str, int]]:
    rows = load_valid_benchmark_rows(out_csv)
    return {
        (str(r["station"]), str(r["code"]), str(r["model"]), int(r["seed"]))
        for r in rows
    }
