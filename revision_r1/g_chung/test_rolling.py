#!/usr/bin/env python3
"""Leakage checks for G4 rolling-origin folds (must pass before GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[1]
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from g_chung.data import (  # noqa: E402
    assert_no_leakage,
    load_meter_bundle_fold,
    resolve_meter_index,
    rolling_split_indices,
)


def test_rolling_ratio():
    idx = rolling_split_indices(8000)
    train_n = idx.train_end
    val_n = idx.val_end - idx.train_end
    assert abs(val_n / train_n - 1 / 7) < 0.01


def test_leakage_real_meter_f2_f3():
    idx = resolve_meter_index(use_demo=False, meters=["M01"])
    path = idx["M01"]
    for fold in ("F2", "F3"):
        for horizon in (48, 336):
            bundle = load_meter_bundle_fold(path, horizon, fold, calendar_features=True)
            checks = assert_no_leakage(bundle, horizon)
            assert checks["ok"]
            # val/train ≈ 1/7
            split = bundle["split"]
            train_rows = split["train_end"]
            val_rows = split["val_end"] - split["train_end"]
            assert abs(val_rows / train_rows - 1 / 7) < 0.02
            # last train target < val start; last val target < test start
            b = bundle["boundaries"]
            assert checks["train_last_target_max"] < b["val_t_min"]
            assert checks["val_last_target_max"] < b["test_t_min"]
            # test window inside fold calendar
            test_ds = bundle["test_ds"]
            _x, _y, t_o, t_t = test_ds[0]
            assert int(t_o) >= b["test_t_min"]
            assert int(np.asarray(t_t).max()) <= b["test_t_max"]
            print(
                f"  {fold} h{horizon}: train_win={len(bundle['train_ds'])} "
                f"val_win={len(bundle['val_ds'])} test_win={len(bundle['test_ds'])} OK"
            )


def main() -> int:
    test_rolling_ratio()
    print("rolling ratio OK; loading M01 folds (may take ~10s)...")
    test_leakage_real_meter_f2_f3()
    print("PASS test_rolling: no boundary leakage on F2/F3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
