#!/usr/bin/env python3
"""CPU smoke for G-chung: readout tests + tiny train + npz/time checks. No GPU."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[1]
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from _paths import require_env  # noqa: E402

from g_chung.test_readout import main as readout_main
from g_chung.train_core import STEP_NS, mae_from_arrays

PYTHON = sys.executable
OUT_ROOT = _PAPER / "results" / "g-chung" / "smoke-cpu"
DEMO = require_env("SCADA_DEMO_DATA")


def run_chay(readout: str, out: Path, epochs: int = 2) -> dict:
    cmd = [
        PYTHON,
        str(_PAPER / "g_chung" / "chay.py"),
        "--model",
        "bilstm",
        "--regime",
        "local",
        "--readout",
        readout,
        "--horizon",
        "4",
        "--seed",
        "42",
        "--meter",
        "DEMO0001",
        "--demo",
        "--demo-dir",
        str(DEMO),
        "--device",
        "cpu",
        "--epochs",
        str(epochs),
        "--patience",
        "10",
        "--out",
        str(out),
        "--num-threads",
        "4",
    ]
    # Allow overwrite for smoke: remove summary if present so we actually train
    if out.exists():
        for p in out.glob("*"):
            if p.is_file():
                p.unlink()
    print("CMD", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.check_call(cmd, cwd=_PAPER)
    wall = time.perf_counter() - t0
    summary = json.loads((out / "summary.json").read_text())
    summary["_smoke_wall_sec"] = wall
    return summary


def check_npz(out: Path, summary: dict) -> dict:
    lab = summary["meter_labels"][0]
    z = np.load(out / f"pred_{lab}.npz")
    pred = z["pred_test"]
    y = z["y_test"]
    t_o = z["t_origin_test"]
    t_t = z["t_target_test"]
    assert pred.shape == y.shape
    assert pred.ndim == 3 and pred.shape[-1] == 2
    H = pred.shape[1]
    # t_target[:, k] == t_origin + (k+1)*30min
    expected = t_o[:, None] + (np.arange(1, H + 1, dtype=np.int64)[None, :] * STEP_NS)
    max_dt = int(np.max(np.abs(t_t - expected)))
    recomputed = mae_from_arrays(pred, y)
    reported = summary["per_meter"][lab]["test_mae"]
    mae_ok = all(abs(recomputed[k] - reported[k]) < 1e-6 for k in ("mae_joint", "mae_p", "mae_q"))
    return {
        "label": lab,
        "pred_shape": list(pred.shape),
        "max_abs_time_delta_ns": max_dt,
        "time_alignment_ok": max_dt == 0,
        "mae_recomputed": recomputed,
        "mae_reported": reported,
        "mae_match": mae_ok,
        "npz_bytes": (out / f"pred_{lab}.npz").stat().st_size,
    }


def measure_epoch_and_h336_size() -> dict:
    """One-epoch timing on demo + measure compressed npz size for one meter at h336 shapes."""
    import tempfile

    from g_chung.data import load_meter_bundle, resolve_meter_index
    from g_chung.models.model_factory import build_model
    from g_chung.train_core import LOOKBACK, fit_model, limit_threads, set_seed
    from torch.utils.data import DataLoader

    limit_threads(4)
    set_seed(42)
    meters = resolve_meter_index(use_demo=True, meters=["DEMO0001"], demo_dir=DEMO)
    b = load_meter_bundle(meters["DEMO0001"], horizon=4, calendar_features=True)
    loader = DataLoader(b["train_ds"], batch_size=128, shuffle=True)
    val = DataLoader(b["val_ds"], batch_size=128, shuffle=False)
    model = build_model("bilstm", b["n_features"], LOOKBACK, 4, "cpu", readout="last")
    t0 = time.perf_counter()
    fit_model(
        model, loader, val, "cpu", b["y_mean"], b["y_std"], "bilstm", epochs=1, patience=10
    )
    epoch_sec = time.perf_counter() - t0

    try:
        real = resolve_meter_index(use_demo=False, meters=["M01"])
        br = load_meter_bundle(real["M01"], horizon=336, calendar_features=True)
        n_val, n_test = len(br["val_ds"]), len(br["test_ds"])
        H = 336
        rng = np.random.default_rng(0)

        def pack(n):
            return {
                "pred": rng.standard_normal((n, H, 2), dtype=np.float32),
                "y": rng.standard_normal((n, H, 2), dtype=np.float32),
                "t_origin": np.arange(n, dtype=np.int64) * STEP_NS,
                "t_target": (
                    np.arange(n, dtype=np.int64)[:, None] * STEP_NS
                    + (np.arange(1, H + 1, dtype=np.int64)[None, :] * STEP_NS)
                ),
            }

        val_p, test_p = pack(n_val), pack(n_test)
        with tempfile.TemporaryDirectory(prefix="gchung_h336_") as td:
            npz_path = Path(td) / "pred_M01.npz"
            np.savez_compressed(
                npz_path,
                pred_val=val_p["pred"],
                y_val=val_p["y"],
                t_origin_val=val_p["t_origin"],
                t_target_val=val_p["t_target"],
                pred_test=test_p["pred"],
                y_test=test_p["y"],
                t_origin_test=test_p["t_origin"],
                t_target_test=test_p["t_target"],
            )
            npz_bytes = npz_path.stat().st_size

        raw_one = (
            2 * (n_val * H * 2 * 4 + n_test * H * 2 * 4)
            + (n_val + n_test) * 8
            + (n_val + n_test) * H * 8
        )
        estimate = {
            "meter": "M01",
            "n_val_windows": n_val,
            "n_test_windows": n_test,
            "measured_npz_bytes_one_meter_h336": npz_bytes,
            "measured_npz_gib_one_meter": npz_bytes / (1024**3),
            "raw_bytes_one_meter_h336": raw_one,
            "extrapolated_npz_gib_one_pooled_run_10meters": 10 * npz_bytes / (1024**3),
            "extrapolated_npz_gib_full_grid": (48 * 10 + 480) * npz_bytes / (1024**3),
            "note": (
                "npz size measured via savez_compressed on random float32 arrays with "
                "true M01 window counts at H=336 (no GPU train). Real preds may compress differently."
            ),
        }
    except Exception as exc:
        estimate = {"error": str(exc)}

    return {"one_epoch_cpu_sec_demo_bilstm_h4": epoch_sec, "h336_size": estimate}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=== (0) readout unit tests ===", flush=True)
    readout_main()

    print("=== (a) smoke last/full demo local h4 2 epoch ===", flush=True)
    checks = {}
    for readout in ("last", "full"):
        out = OUT_ROOT / f"bilstm_local_{readout}_h4_s42_DEMO0001"
        summary = run_chay(readout, out, epochs=2)
        chk = check_npz(out, summary)
        checks[readout] = {
            "summary_mean_test_mae_joint": summary["mean_test_mae_joint"],
            "best_epoch": summary["best_epoch"],
            "stopped_epoch": summary["stopped_epoch"],
            "train_sec": summary["train_sec"],
            "wall_sec": summary["_smoke_wall_sec"],
            "npz_check": chk,
        }
        assert chk["time_alignment_ok"], chk
        assert chk["mae_match"], chk
        print(f"OK readout={readout} mae_joint={summary['mean_test_mae_joint']:.4f} npz={chk['npz_bytes']}B")

    print("=== (b/c) epoch time + h336 size estimate ===", flush=True)
    sizing = measure_epoch_and_h336_size()
    print(json.dumps(sizing, indent=2))

    report = {
        "readout_tests": "PASS",
        "smoke": checks,
        "sizing": sizing,
        "device": "cpu",
        "python": PYTHON,
    }
    report_path = OUT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
