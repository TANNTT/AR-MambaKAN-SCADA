"""Summary of G4 (rolling origin F2/F3, 264 runs). Reads results/g4/*/summary.json only.

Writes results/g4-tong/summary.json. CPU.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict

import scipy.stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METERS = [f"M{i:02d}" for i in range(1, 11)]
METERS_NO_M08 = [m for m in METERS if m != "M08"]
SEEDS = [42, 43, 44]
MODELS = ["bilstm", "armambakan"]
REGIMES = ["pooled", "local"]
FOLDS_EXPECTED = ["F2", "F3"]
N_EXPECTED_RUNS = 264
METRICS = ("mae_joint", "mae_p", "mae_q")
METRIC_TAG = {"mae_joint": "joint", "mae_p": "p", "mae_q": "q"}


def wilcoxon_two_sided(diffs: list[float]) -> float | None:
    nonzero = [x for x in diffs if x != 0]
    if len(nonzero) < 1:
        return None
    try:
        res = scipy.stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
        return float(res.pvalue)
    except ValueError:
        return None


def mean_or_none(xs: list[float]) -> float | None:
    return st.mean(xs) if xs else None


def per_meter_seed_mean(
    cell: dict[tuple[int, str], dict], metric: str, meters: list[str]
) -> dict[str, float]:
    out = {}
    for m in meters:
        vals = [cell[(s, m)][metric] for s in SEEDS if (s, m) in cell]
        if vals:
            out[m] = st.mean(vals)
    return out


def compare_pair(
    a: dict[str, float],
    b: dict[str, float],
    meters: list[str],
    *,
    a_name: str,
    b_name: str,
) -> dict:
    common = [m for m in meters if m in a and m in b]
    diffs = [a[m] - b[m] for m in common]
    a_wins = sum(1 for d in diffs if d < 0)
    rel = [(a[m] / b[m] - 1.0) if b[m] != 0 else None for m in common]
    rel_ok = [r for r in rel if r is not None]
    return {
        f"mean_{a_name}": mean_or_none([a[m] for m in common]),
        f"mean_{b_name}": mean_or_none([b[m] for m in common]),
        "n_meters": len(common),
        f"{a_name}_wins": a_wins,
        "wilcoxon_p": wilcoxon_two_sided(diffs),
        "median_rel_diff": (st.median(rel_ok) if rel_ok else None),
        "per_meter": {
            m: {
                a_name: a[m],
                b_name: b[m],
                "diff": a[m] - b[m],
                "rel": (a[m] / b[m] - 1.0) if b[m] != 0 else None,
            }
            for m in common
        },
    }


def main() -> None:
    # cell[(fold,model,regime,readout,h)][(seed,meter)] = metrics + windows
    cells: dict[tuple, dict[tuple[int, str], dict]] = defaultdict(dict)
    leakage_by_run = []
    readout_c = Counter()
    horizon_c = Counter()
    fold_c = Counter()
    n_runs = 0

    for f in sorted(glob.glob(os.path.join(ROOT, "results/g4/*/summary.json"))):
        s = json.load(open(f))
        n_runs += 1
        fold = s["fold"]
        model = s["model"]
        regime = s["regime"]
        readout = s["readout"]
        h = int(s["horizon"])
        seed = int(s["seed"])
        readout_c[readout] += 1
        horizon_c[h] += 1
        fold_c[fold] += 1
        k = (fold, model, regime, readout, h)
        if s.get("leakage_check") is not None:
            lc = s["leakage_check"]
            # local: {..., ok: bool}; pooled: {M01: {..., ok}, M02: ...}
            if isinstance(lc, dict) and "ok" in lc:
                all_ok = bool(lc["ok"])
            elif isinstance(lc, dict) and lc and all(
                isinstance(v, dict) and "ok" in v for v in lc.values()
            ):
                all_ok = all(bool(v["ok"]) for v in lc.values())
            else:
                all_ok = False
            leakage_by_run.append(
                {
                    "run": s.get("run") or os.path.basename(os.path.dirname(f)),
                    "fold": fold,
                    "model": model,
                    "regime": regime,
                    "readout": readout,
                    "horizon": h,
                    "seed": seed,
                    "leakage_all_ok": all_ok,
                    "leakage_check": lc,
                }
            )
        for m, v in s["per_meter"].items():
            tm = v["test_mae"]
            cells[k][(seed, m)] = {
                "mae_joint": float(tm["mae_joint"]),
                "mae_p": float(tm["mae_p"]),
                "mae_q": float(tm["mae_q"]),
                "n_train_windows": int(v["n_train_windows"]),
                "n_val_windows": int(v["n_val_windows"]),
                "n_test_windows": int(v["n_test_windows"]),
            }

    readouts_present = sorted(readout_c.keys())
    horizons_present = sorted(horizon_c.keys())
    folds_present = sorted(fold_c.keys())

    # Kiem du: moi o (fold,model,regime,ro,h) can 3 seed x 10 meter
    missing = []
    expected_keys = []
    for fold in FOLDS_EXPECTED:
        for model in MODELS:
            for regime in REGIMES:
                for ro in readouts_present or ["last"]:
                    for h in horizons_present or [48, 336]:
                        expected_keys.append((fold, model, regime, ro, h))
                        for seed in SEEDS:
                            for m in METERS:
                                if (seed, m) not in cells.get(
                                    (fold, model, regime, ro, h), {}
                                ):
                                    missing.append(
                                        {
                                            "fold": fold,
                                            "model": model,
                                            "regime": regime,
                                            "readout": ro,
                                            "horizon": h,
                                            "seed": seed,
                                            "meter": m,
                                        }
                                    )

    pool_vs_local = {}
    ar_vs_bilstm = {}
    window_counts = {}

    for fold in folds_present:
        for ro in readouts_present:
            for h in horizons_present:
                for metric in METRICS:
                    tag = METRIC_TAG[metric]
                    # (a) pooled vs local theo fold
                    for model in MODELS:
                        kp = (fold, model, "pooled", ro, h)
                        kl = (fold, model, "local", ro, h)
                        P = per_meter_seed_mean(cells.get(kp, {}), metric, METERS)
                        L = per_meter_seed_mean(cells.get(kl, {}), metric, METERS)
                        for suffix, meters in (("", METERS), ("_noM08", METERS_NO_M08)):
                            key = f"{fold}|{model}|{ro}|{h}|{tag}{suffix}"
                            pool_vs_local[key] = compare_pair(
                                P, L, meters, a_name="pooled", b_name="local"
                            )
                    # (b) AR vs BiLSTM theo fold
                    for regime in REGIMES:
                        ka = (fold, "armambakan", regime, ro, h)
                        kb = (fold, "bilstm", regime, ro, h)
                        A = per_meter_seed_mean(cells.get(ka, {}), metric, METERS)
                        B = per_meter_seed_mean(cells.get(kb, {}), metric, METERS)
                        for suffix, meters in (("", METERS), ("_noM08", METERS_NO_M08)):
                            key = f"{fold}|{regime}|{ro}|{h}|{tag}{suffix}"
                            ar_vs_bilstm[key] = compare_pair(
                                A, B, meters, a_name="armambakan", b_name="bilstm"
                            )

        # Number of train / validation / test windows per meter per fold
        # Taken from seed 42 (window counts do not depend on the seed by design)
        for model in MODELS:
            for regime in REGIMES:
                for ro in readouts_present:
                    for h in horizons_present:
                        k = (fold, model, regime, ro, h)
                        cell = cells.get(k, {})
                        per_m = {}
                        for m in METERS:
                            # uu tien seed 42; neu thieu lay seed bat ky
                            entry = None
                            for seed in SEEDS:
                                if (seed, m) in cell:
                                    entry = cell[(seed, m)]
                                    if seed == 42:
                                        break
                            if entry is None:
                                continue
                            per_m[m] = {
                                "n_train_windows": entry["n_train_windows"],
                                "n_val_windows": entry["n_val_windows"],
                                "n_test_windows": entry["n_test_windows"],
                            }
                        window_counts[f"{fold}|{model}|{regime}|{ro}|{h}"] = per_m

    leakage_ok = (
        all(r["leakage_all_ok"] for r in leakage_by_run) if leakage_by_run else None
    )
    leakage_failures = [r for r in leakage_by_run if not r["leakage_all_ok"]]

    out = {
        "n_runs": n_runs,
        "n_expected_runs": N_EXPECTED_RUNS,
        "n_missing_seed_meter_cells": len(missing),
        "missing": missing,
        "readouts_present": readouts_present,
        "horizons_present": horizons_present,
        "folds_present": folds_present,
        "readout_counts": dict(readout_c),
        "horizon_counts": {str(k): v for k, v in horizon_c.items()},
        "fold_counts": dict(fold_c),
        "pool_vs_local": pool_vs_local,
        "ar_vs_bilstm": ar_vs_bilstm,
        "window_counts": window_counts,
        "leakage_check_n_runs": len(leakage_by_run),
        "leakage_check_all_ok": leakage_ok,
        "leakage_check_failures": [
            {k: v for k, v in r.items() if k != "leakage_check"} for r in leakage_failures
        ],
        "note": (
            "G4 goc truot F2/F3. Moi cong to = trung binh 3 seed. "
            "pool_vs_local / ar_vs_bilstm giong G-chung, tach theo fold. "
            "window_counts lay n_train/val/test_windows tu per_meter (uu tien seed 42). "
            "leakage_check: local co khoa ok o dinh; pooled la dict theo meter."
        ),
    }

    out_dir = os.path.join(ROOT, "results/g4-tong")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    json.dump(out, open(out_path, "w"), indent=2)

    print(
        "G4 tong: n_runs=%d / %d expected, missing_cells=%d, readout=%s horizon=%s folds=%s -> %s"
        % (
            n_runs,
            N_EXPECTED_RUNS,
            len(missing),
            readouts_present,
            horizons_present,
            folds_present,
            out_path,
        )
    )
    print(
        "leakage_check: n=%d all_ok=%s failures=%d"
        % (len(leakage_by_run), leakage_ok, len(leakage_failures))
    )
    if missing:
        print("O thieu (%d):" % len(missing))
        for x in missing[:40]:
            print(" ", x)
        if len(missing) > 40:
            print("  ... va %d o nua" % (len(missing) - 40))

    print("\n== (a) pooled vs local (joint) theo fold")
    print(
        "%-4s %-12s %-5s %4s %10s %10s %8s %8s"
        % ("fold", "model", "ro", "h", "pooled", "local", "wins", "p")
    )
    for fold in folds_present:
        for model in MODELS:
            for ro in readouts_present:
                for h in horizons_present:
                    v = pool_vs_local[f"{fold}|{model}|{ro}|{h}|joint"]
                    print(
                        "%-4s %-12s %-5s %4d %10.1f %10.1f %3d/%d %8s"
                        % (
                            fold,
                            model,
                            ro,
                            h,
                            v["mean_pooled"],
                            v["mean_local"],
                            v["pooled_wins"],
                            v["n_meters"],
                            ("%.4f" % v["wilcoxon_p"]) if v["wilcoxon_p"] is not None else "NA",
                        )
                    )

    print("\n== (b) AR vs BiLSTM (joint) theo fold")
    print(
        "%-4s %-8s %-5s %4s %10s %10s %8s %8s"
        % ("fold", "regime", "ro", "h", "AR", "BiLSTM", "ARwins", "p")
    )
    for fold in folds_present:
        for regime in REGIMES:
            for ro in readouts_present:
                for h in horizons_present:
                    v = ar_vs_bilstm[f"{fold}|{regime}|{ro}|{h}|joint"]
                    print(
                        "%-4s %-8s %-5s %4d %10.1f %10.1f %3d/%d %8s"
                        % (
                            fold,
                            regime,
                            ro,
                            h,
                            v["mean_armambakan"],
                            v["mean_bilstm"],
                            v["armambakan_wins"],
                            v["n_meters"],
                            ("%.4f" % v["wilcoxon_p"]) if v["wilcoxon_p"] is not None else "NA",
                        )
                    )

    print("\n== so cua so (vi du bilstm pooled last, seed uu tien 42)")
    for fold in folds_present:
        for h in horizons_present:
            key = f"{fold}|bilstm|pooled|last|{h}"
            wc = window_counts.get(key, {})
            if not wc:
                continue
            trains = [wc[m]["n_train_windows"] for m in METERS if m in wc]
            vals = [wc[m]["n_val_windows"] for m in METERS if m in wc]
            tests = [wc[m]["n_test_windows"] for m in METERS if m in wc]
            print(
                "%s h=%d train[%d..%d] val[%d..%d] test[%d..%d]"
                % (
                    fold,
                    h,
                    min(trains),
                    max(trains),
                    min(vals),
                    max(vals),
                    min(tests),
                    max(tests),
                )
            )
            for m in METERS:
                if m in wc:
                    w = wc[m]
                    print(
                        "  %s train=%d val=%d test=%d"
                        % (m, w["n_train_windows"], w["n_val_windows"], w["n_test_windows"])
                    )


if __name__ == "__main__":
    main()
