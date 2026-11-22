#!/usr/bin/env python3
"""CPU smoke for G2b / G3 / G4: unit checks + 1-epoch tiny runs. No GPU.

Assumptions:
1. G2b: local, no-calendar, readout full, ablation A5 on demo (BiLSTM-affected).
2. G3: matched p=0.5 on demo meters, check budget windows in summary.
3. G4: F2 local bilstm on real M01 (demo has no 2022 dates), 1 epoch, leakage on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from _paths import require_env  # noqa: E402

from g_chung.test_budget import main as budget_main
from g_chung.test_readout import main as readout_main

PYTHON = sys.executable
OUT = _PAPER / "results" / "smoke-g2b-g3-g4"
DEMO = require_env("SCADA_DEMO_DATA")


def _wipe(p: Path) -> None:
    if p.exists():
        for f in p.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)


def run(cmd: list[str]) -> None:
    print("CMD", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=_PAPER)


def smoke_g2b() -> dict:
    out = OUT / "g2b_A5_demo"
    _wipe(out)
    t0 = time.perf_counter()
    run(
        [
            PYTHON,
            str(_PAPER / "g_chung" / "chay.py"),
            "--model",
            "armambakan_nostem",
            "--regime",
            "local",
            "--readout",
            "full",
            "--horizon",
            "48",
            "--seed",
            "42",
            "--meter",
            "DEMO0001",
            "--demo",
            "--demo-dir",
            str(DEMO),
            "--no-calendar",
            "--ablation",
            "A5",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--patience",
            "10",
            "--out",
            str(out),
            "--num-threads",
            "4",
        ]
    )
    s = json.loads((out / "summary.json").read_text())
    return {
        "wall_sec": time.perf_counter() - t0,
        "mean_test_mae_joint": s["mean_test_mae_joint"],
        "ablation": s.get("ablation"),
        "calendar_features": s["calendar_features"],
        "readout": s["readout"],
        "run": s["run"],
    }


def smoke_g3() -> dict:
    out = OUT / "g3_p0.5_demo"
    _wipe(out)
    meters = ["DEMO0001", "DEMO0002", "DEMO0003", "DEMO0004"]
    t0 = time.perf_counter()
    run(
        [
            PYTHON,
            str(_PAPER / "g_chung" / "chay.py"),
            "--model",
            "armambakan",
            "--regime",
            "matched",
            "--readout",
            "last",
            "--horizon",
            "48",
            "--seed",
            "42",
            "--target-meter",
            "DEMO0001",
            "--target-frac",
            "0.5",
            "--meters",
            *meters,
            "--demo",
            "--demo-dir",
            str(DEMO),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--out",
            str(out),
            "--num-threads",
            "4",
        ]
    )
    s = json.loads((out / "summary.json").read_text())
    b = s["budget"]
    assert b["total"] == b["n_local_target"]
    assert b["budgets"]["DEMO0001"] == round(0.5 * b["n_local_target"])
    # equal-share cross-check on same demo meters
    out2 = OUT / "g3_equal_share_demo"
    _wipe(out2)
    run(
        [
            PYTHON,
            str(_PAPER / "g_chung" / "chay.py"),
            "--model",
            "armambakan",
            "--regime",
            "matched",
            "--readout",
            "last",
            "--horizon",
            "4",
            "--seed",
            "42",
            "--target-meter",
            "DEMO0001",
            "--equal-share",
            "--meters",
            *meters,
            "--demo",
            "--demo-dir",
            str(DEMO),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--out",
            str(out2),
            "--num-threads",
            "4",
        ]
    )
    s2 = json.loads((out2 / "summary.json").read_text())
    b2 = s2["budget"]
    per = list(b2["budgets"].values())[0]
    assert all(v == per for v in b2["budgets"].values())
    return {
        "wall_sec": time.perf_counter() - t0,
        "p05_total": b["total"],
        "p05_target": b["budgets"]["DEMO0001"],
        "equal_share_per_meter": per,
        "equal_share_total": b2["total"],
        "mean_test_mae_joint": s["mean_test_mae_joint"],
    }


def smoke_g4() -> dict:
    out = OUT / "g4_F2_M01"
    _wipe(out)
    t0 = time.perf_counter()
    run(
        [
            PYTHON,
            str(_PAPER / "g_chung" / "chay.py"),
            "--model",
            "bilstm",
            "--regime",
            "local",
            "--readout",
            "last",
            "--horizon",
            "48",
            "--seed",
            "42",
            "--meter",
            "M01",
            "--fold",
            "F2",
            "--check-leakage",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--out",
            str(out),
            "--num-threads",
            "4",
        ]
    )
    s = json.loads((out / "summary.json").read_text())
    assert s["fold"] == "F2"
    assert s["leakage_check"]["ok"] is True
    return {
        "wall_sec": time.perf_counter() - t0,
        "mean_test_mae_joint": s["mean_test_mae_joint"],
        "leakage_ok": s["leakage_check"]["ok"],
        "n_train": s["per_meter"]["M01"]["n_train_windows"],
        "n_val": s["per_meter"]["M01"]["n_val_windows"],
        "n_test": s["per_meter"]["M01"]["n_test_windows"],
    }


def count_plan() -> dict:
    # G2b default: A3-A6 × 10 × 3 = 120; with --all-codes: 180
    g2b = 4 * 10 * 3
    g2b_all = 6 * 10 * 3
    g3 = 10 * 3 * 3  # meters × p × seeds
    # model × fold × horizon × seed; local also × meters
    g4_pooled = 2 * 2 * 2 * 3  # = 24
    g4_local = 2 * 2 * 10 * 2 * 3  # = 240
    g4 = g4_pooled + g4_local  # = 264
    unit_pooled_min = 6.0
    unit_local_min = 0.5
    return {
        "g2b_bilstm_variants_runs": g2b,
        "g2b_all_A1_to_A6_runs": g2b_all,
        "g2b_gpu_hours_est": round(g2b * unit_local_min / 60, 2),
        "g3_runs": g3,
        "g3_gpu_hours_est": round(g3 * unit_local_min / 60, 2),
        "g4_pooled_runs": g4_pooled,
        "g4_local_runs": g4_local,
        "g4_runs": g4,
        "g4_gpu_hours_est": round((g4_pooled * unit_pooled_min + g4_local * unit_local_min) / 60, 2),
        "unit_pooled_min": unit_pooled_min,
        "unit_local_min": unit_local_min,
        "note": "hours use pooled=6 min, local=0.5 min; revise after G0 timing",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    readout_main()
    budget_main()
    # Rolling leakage on real meter (imported here to avoid delaying budget-only fails)
    from g_chung.test_rolling import main as rolling_main

    rolling_main()

    report = {
        "g2b": smoke_g2b(),
        "g3": smoke_g3(),
        "g4": smoke_g4(),
        "run_counts": count_plan(),
        "bilstm_ablations": ["A3", "A4", "A5", "A6"],
        "non_bilstm_ablations": ["A1", "A2"],
        "sources": {
            "ablation_codes": "upstream ablation script (variant codes A3-A6)",
            "table18_protocol": "PAPER_CLAIMS.md Task #71 (local, h48, no-calendar, 10 meters, 3 seeds)",
            "matched_budget": "experiments/matched_budget/run_matched_budget.py:44-60",
            "sampling": "experiments/pooled/common_pooled.py:136-143",
        },
    }
    path = OUT / "smoke_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["run_counts"], indent=2))
    print(f"SMOKE OK -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
