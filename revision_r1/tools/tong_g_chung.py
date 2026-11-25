"""Summary of G-chung (528 runs): pooled/local x last/full x 4 horizons x 3 seeds.

Reads results/g-chung/*/summary.json only (fold/ablation/target_meter all null).
Writes results/g-chung-tong/summary.json. CPU.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

import scipy.stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METERS = [f"M{i:02d}" for i in range(1, 11)]
METERS_NO_M08 = [m for m in METERS if m != "M08"]
SEEDS = [42, 43, 44]
MODELS = ["bilstm", "armambakan"]
REGIMES = ["pooled", "local"]
READOUTS = ["last", "full"]
HORIZONS = [4, 48, 96, 336]
METRICS = ("mae_joint", "mae_p", "mae_q")
METRIC_TAG = {"mae_joint": "joint", "mae_p": "p", "mae_q": "q"}

# G0 reference (mean over 10 stations, last, h48, seed 42, pooled)
G0_LAST_H48_S42 = {
    "bilstm": 484.079,
    "armambakan": 473.438,
}
G0_TOL = 0.5  # versus the figure printed in the request (3 decimal places)


def is_null(v) -> bool:
    return v is None


def wilcoxon_two_sided(diffs: list[float]) -> float | None:
    """Wilcoxon signed-rank, two-sided. None neu khong du so khac 0."""
    nonzero = [x for x in diffs if x != 0]
    if len(nonzero) < 1:
        return None
    # n=1..: scipy can it nhat 1; exact khi n nho
    try:
        res = scipy.stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
        return float(res.pvalue)
    except ValueError:
        return None


def mean_or_none(xs: list[float]) -> float | None:
    return st.mean(xs) if xs else None


def per_meter_seed_mean(
    cell: dict[tuple[int, str], dict[str, float]], metric: str, meters: list[str]
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
    diffs = [a[m] - b[m] for m in common]  # a - b; a thang khi diff < 0
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
    # cell[(model,regime,readout,h)][(seed,meter)] = {mae_joint, mae_p, mae_q}
    cells: dict[tuple, dict[tuple[int, str], dict[str, float]]] = defaultdict(dict)
    n_runs = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "results/g-chung/*/summary.json"))):
        s = json.load(open(f))
        if not (is_null(s.get("fold")) and is_null(s.get("ablation")) and is_null(s.get("target_meter"))):
            continue
        n_runs += 1
        k = (s["model"], s["regime"], s["readout"], int(s["horizon"]))
        seed = int(s["seed"])
        for m, v in s["per_meter"].items():
            tm = v["test_mae"]
            cells[k][(seed, m)] = {
                "mae_joint": float(tm["mae_joint"]),
                "mae_p": float(tm["mae_p"]),
                "mae_q": float(tm["mae_q"]),
            }

    expected_keys = [
        (model, regime, ro, h)
        for model in MODELS
        for regime in REGIMES
        for ro in READOUTS
        for h in HORIZONS
    ]
    missing = []
    for k in expected_keys:
        for seed in SEEDS:
            for m in METERS:
                if (seed, m) not in cells.get(k, {}):
                    missing.append(
                        {
                            "model": k[0],
                            "regime": k[1],
                            "readout": k[2],
                            "horizon": k[3],
                            "seed": seed,
                            "meter": m,
                        }
                    )

    pool_vs_local = {}
    ar_vs_bilstm = {}
    full_vs_last = {}
    ket_luan_dao = {}

    for ro in READOUTS:
        for h in HORIZONS:
            dao_entry = {"readout": ro, "horizon": h, "pooled_hon_local": {}, "bilstm_hon_armambakan": {}}
            for metric in METRICS:
                tag = METRIC_TAG[metric]
                # (a) pooled vs local
                for model in MODELS:
                    kp = (model, "pooled", ro, h)
                    kl = (model, "local", ro, h)
                    P = per_meter_seed_mean(cells.get(kp, {}), metric, METERS)
                    L = per_meter_seed_mean(cells.get(kl, {}), metric, METERS)
                    for suffix, meters in (("", METERS), ("_noM08", METERS_NO_M08)):
                        key = f"{model}|{ro}|{h}|{tag}{suffix}"
                        pool_vs_local[key] = compare_pair(
                            P, L, meters, a_name="pooled", b_name="local"
                        )
                    if metric == "mae_joint":
                        cmp10 = pool_vs_local[f"{model}|{ro}|{h}|joint"]
                        dao_entry["pooled_hon_local"][model] = bool(
                            cmp10["mean_pooled"] is not None
                            and cmp10["mean_local"] is not None
                            and cmp10["mean_pooled"] < cmp10["mean_local"]
                            and cmp10["pooled_wins"] >= 6
                        )

                # (b) AR-MambaKAN vs BiLSTM
                for regime in REGIMES:
                    ka = ("armambakan", regime, ro, h)
                    kb = ("bilstm", regime, ro, h)
                    A = per_meter_seed_mean(cells.get(ka, {}), metric, METERS)
                    B = per_meter_seed_mean(cells.get(kb, {}), metric, METERS)
                    for suffix, meters in (("", METERS), ("_noM08", METERS_NO_M08)):
                        key = f"{regime}|{ro}|{h}|{tag}{suffix}"
                        ar_vs_bilstm[key] = compare_pair(
                            A, B, meters, a_name="armambakan", b_name="bilstm"
                        )
                    if metric == "mae_joint":
                        cmp10 = ar_vs_bilstm[f"{regime}|{ro}|{h}|joint"]
                        # bilstm hon = mean bilstm thap hon (tot hon)
                        dao_entry["bilstm_hon_armambakan"][regime] = bool(
                            cmp10["mean_bilstm"] is not None
                            and cmp10["mean_armambakan"] is not None
                            and cmp10["mean_bilstm"] < cmp10["mean_armambakan"]
                        )
            ket_luan_dao[f"{ro}|{h}"] = dao_entry

    # (c) full vs last — theo (model, regime, horizon), joint/P/Q
    for model in MODELS:
        for regime in REGIMES:
            for h in HORIZONS:
                for metric in METRICS:
                    tag = METRIC_TAG[metric]
                    kf = (model, regime, "full", h)
                    kl = (model, regime, "last", h)
                    F = per_meter_seed_mean(cells.get(kf, {}), metric, METERS)
                    Lt = per_meter_seed_mean(cells.get(kl, {}), metric, METERS)
                    common = [m for m in METERS if m in F and m in Lt]
                    mean_f = mean_or_none([F[m] for m in common])
                    mean_l = mean_or_none([Lt[m] for m in common])
                    pct = None
                    if mean_f is not None and mean_l is not None and mean_l != 0:
                        pct = 100.0 * (mean_f / mean_l - 1.0)
                    full_vs_last[f"{model}|{regime}|{h}|{tag}"] = {
                        "mean_last": mean_l,
                        "mean_full": mean_f,
                        "pct_change_full_vs_last": pct,
                        "n_meters": len(common),
                    }

    # Doi chieu G0: last h48 seed 42 pooled, mean 10 tram mae_joint
    g0_check = {"ok": True, "rows": []}
    for model in MODELS:
        k = (model, "pooled", "last", 48)
        vals = []
        for m in METERS:
            if (42, m) in cells.get(k, {}):
                vals.append(cells[k][(42, m)]["mae_joint"])
        observed = st.mean(vals) if vals else None
        expected = G0_LAST_H48_S42[model]
        delta = None if observed is None else observed - expected
        ok = observed is not None and abs(delta) <= G0_TOL
        if not ok:
            g0_check["ok"] = False
        g0_check["rows"].append(
            {
                "model": model,
                "regime": "pooled",
                "readout": "last",
                "horizon": 48,
                "seed": 42,
                "metric": "mae_joint_mean10",
                "observed": observed,
                "expected_g0": expected,
                "delta": delta,
                "ok": ok,
            }
        )

    # Bang trung binh thuan tien in
    cell_means = {}
    for k in expected_keys:
        pm = per_meter_seed_mean(cells.get(k, {}), "mae_joint", METERS)
        cell_means["|".join(map(str, k))] = {
            "n_seed_meter": len(cells.get(k, {})),
            "mean10_joint": mean_or_none(list(pm.values())) if pm else None,
            "per_meter_joint": pm,
        }

    out = {
        "n_runs_read": n_runs,
        "n_expected_seed_meter_cells": len(MODELS)
        * len(REGIMES)
        * len(READOUTS)
        * len(HORIZONS)
        * len(SEEDS)
        * len(METERS),
        "n_missing": len(missing),
        "missing": missing,
        "models": MODELS,
        "regimes": REGIMES,
        "readouts": READOUTS,
        "horizons": HORIZONS,
        "seeds": SEEDS,
        "meters": METERS,
        "pool_vs_local": pool_vs_local,
        "ar_vs_bilstm": ar_vs_bilstm,
        "full_vs_last": full_vs_last,
        "ket_luan_dao": ket_luan_dao,
        "g0_check": g0_check,
        "cell_means_joint": cell_means,
        "note": (
            "Moi cong to = trung binh 3 seed. "
            "pool_vs_local: pooled thang khi MAE pooled < local. "
            "ar_vs_bilstm: armambakan thang khi MAE AR < BiLSTM. "
            "median_rel_diff = median(pooled/local - 1) hoac (AR/BiLSTM - 1). "
            "ket_luan_dao.pooled_hon_local: mean pooled thap hon VA thang >=6/10. "
            "ket_luan_dao.bilstm_hon_armambakan: mean BiLSTM joint thap hon AR o regime do. "
            "Ban _noM08 bo M08 khoi trung binh va Wilcoxon."
        ),
    }

    out_dir = os.path.join(ROOT, "results/g-chung-tong")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    json.dump(out, open(out_path, "w"), indent=2)

    # In bang ngan
    print(
        "G-chung tong: n_runs=%d missing_cells=%d g0_ok=%s -> %s"
        % (n_runs, len(missing), g0_check["ok"], out_path)
    )
    if missing:
        print("O thieu (%d):" % len(missing))
        for x in missing[:50]:
            print(" ", x)
        if len(missing) > 50:
            print("  ... va %d o nua" % (len(missing) - 50))

    print("\n== (a) pooled vs local (joint, 10 tram)")
    print(
        "%-12s %-5s %4s %10s %10s %8s %8s %10s"
        % ("model", "ro", "h", "pooled", "local", "wins", "p", "med_rel")
    )
    for model in MODELS:
        for ro in READOUTS:
            for h in HORIZONS:
                v = pool_vs_local[f"{model}|{ro}|{h}|joint"]
                print(
                    "%-12s %-5s %4d %10.1f %10.1f %3d/10 %8s %10.4f"
                    % (
                        model,
                        ro,
                        h,
                        v["mean_pooled"],
                        v["mean_local"],
                        v["pooled_wins"],
                        ("%.4f" % v["wilcoxon_p"]) if v["wilcoxon_p"] is not None else "NA",
                        v["median_rel_diff"] if v["median_rel_diff"] is not None else float("nan"),
                    )
                )

    print("\n== (a) pooled vs local (joint, no M08)")
    print(
        "%-12s %-5s %4s %10s %10s %8s %8s"
        % ("model", "ro", "h", "pooled", "local", "wins", "p")
    )
    for model in MODELS:
        for ro in READOUTS:
            for h in HORIZONS:
                v = pool_vs_local[f"{model}|{ro}|{h}|joint_noM08"]
                print(
                    "%-12s %-5s %4d %10.1f %10.1f %3d/9 %8s"
                    % (
                        model,
                        ro,
                        h,
                        v["mean_pooled"],
                        v["mean_local"],
                        v["pooled_wins"],
                        ("%.4f" % v["wilcoxon_p"]) if v["wilcoxon_p"] is not None else "NA",
                    )
                )

    print("\n== (b) AR-MambaKAN vs BiLSTM (joint, 10 tram)")
    print(
        "%-8s %-5s %4s %10s %10s %8s %8s"
        % ("regime", "ro", "h", "AR", "BiLSTM", "ARwins", "p")
    )
    for regime in REGIMES:
        for ro in READOUTS:
            for h in HORIZONS:
                v = ar_vs_bilstm[f"{regime}|{ro}|{h}|joint"]
                print(
                    "%-8s %-5s %4d %10.1f %10.1f %3d/10 %8s"
                    % (
                        regime,
                        ro,
                        h,
                        v["mean_armambakan"],
                        v["mean_bilstm"],
                        v["armambakan_wins"],
                        ("%.4f" % v["wilcoxon_p"]) if v["wilcoxon_p"] is not None else "NA",
                    )
                )

    print("\n== (b) AR vs BiLSTM (joint, no M08)")
    for regime in REGIMES:
        for ro in READOUTS:
            for h in HORIZONS:
                v = ar_vs_bilstm[f"{regime}|{ro}|{h}|joint_noM08"]
                print(
                    "%-8s %-5s %4d AR %.1f Bi %.1f wins %d/9 p=%s"
                    % (
                        regime,
                        ro,
                        h,
                        v["mean_armambakan"],
                        v["mean_bilstm"],
                        v["armambakan_wins"],
                        ("%.4f" % v["wilcoxon_p"]) if v["wilcoxon_p"] is not None else "NA",
                    )
                )

    print("\n== (c) full vs last (joint)")
    print(
        "%-12s %-8s %4s %10s %10s %8s"
        % ("model", "regime", "h", "last", "full", "pct")
    )
    for model in MODELS:
        for regime in REGIMES:
            for h in HORIZONS:
                v = full_vs_last[f"{model}|{regime}|{h}|joint"]
                print(
                    "%-12s %-8s %4d %10.1f %10.1f %+7.1f%%"
                    % (
                        model,
                        regime,
                        h,
                        v["mean_last"],
                        v["mean_full"],
                        v["pct_change_full_vs_last"],
                    )
                )

    print("\n== ket_luan_dao (joint)")
    for ro in READOUTS:
        for h in HORIZONS:
            e = ket_luan_dao[f"{ro}|{h}"]
            print(
                "ro=%s h=%d pooled_hon_local=%s bilstm_hon_AR=%s"
                % (ro, h, e["pooled_hon_local"], e["bilstm_hon_armambakan"])
            )

    print("\n== doi chieu G0 (last h48 seed 42 pooled mean10 joint)")
    for row in g0_check["rows"]:
        print(
            "%s observed=%.6f expected=%.3f delta=%+.6f ok=%s"
            % (row["model"], row["observed"], row["expected_g0"], row["delta"], row["ok"])
        )


if __name__ == "__main__":
    main()
