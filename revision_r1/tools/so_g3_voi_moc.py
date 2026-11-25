"""Compare G3 (target_frac) with the equal-split matched-budget reference. Read-only, CPU.

Reference preference (read in order, take the first file that exists):
  1) results/matched_budget_armambakan.csv (in the paper directory)
  2) $SCADA_PUBLIC_REPO/results/matched_budget_armambakan.csv
     (submitted version run_matched_budget.py, read-only)

Filter horizon=48, MAE by station (codes M01..M10), mean over 3 seeds.
G3: read results/g3/*/summary.json, target-station mae_joint, mean over 3 seeds by p.
Paired Wilcoxon by station (3-seed mean): p0.5 vs p0.75, p0.25 vs p0.5.
Output: results/g3-so-moc/summary.json
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METERS = [f"M{i:02d}" for i in range(1, 11)]
PS = [0.25, 0.5, 0.75]
SEEDS = [42, 43, 44]
HORIZON = 48

def _moc_candidates():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from _paths import require_env, env_or_none
    out = [ROOT / "results" / "matched_budget_armambakan.csv"]
    pub = env_or_none("SCADA_PUBLIC_REPO")
    if pub is not None:
        out.append(pub / "results" / "matched_budget_armambakan.csv")
    return out

MOC_CANDIDATES = None  # resolved in main via _moc_candidates()


def load_g3():
    """(meter, p) -> {seed: mae_target}."""
    d = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "g3" / "*" / "summary.json"))):
        s = json.load(open(f))
        tm = s["target_meter"]
        p = float(s["target_frac"])
        seed = int(s["seed"])
        if tm not in s["per_meter"]:
            raise SystemExit(f"missing per_meter[{tm}] in {f}")
        mae = float(s["per_meter"][tm]["test_mae"]["mae_joint"])
        d.setdefault((tm, p), {})[seed] = mae
    return d


def load_moc_equal_share():
    """Return (path_or_None, {meter: {seed: mae}})."""
    path = next((p for p in _moc_candidates() if p.is_file()), None)
    if path is None:
        return None, {}
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["horizon"]) != HORIZON:
                continue
            m = r["code"]
            if m not in METERS:
                continue
            seed = int(r["seed"])
            by.setdefault(m, {})[seed] = float(r["mae"])
    return str(path), by


def mean_seeds(seed_map):
    vals = [seed_map[s] for s in SEEDS if s in seed_map]
    if len(vals) != len(SEEDS):
        return None, vals
    return st.mean(vals), vals


def wilcoxon_paired(a, b, label):
    """a, b: lists of per-meter means, same order. Two-sided Wilcoxon signed-rank."""
    diffs = [x - y for x, y in zip(a, b)]
    out = {
        "label": label,
        "n_meters": len(a),
        "diffs": diffs,
        "n_positive": sum(1 for d in diffs if d > 0),
        "n_negative": sum(1 for d in diffs if d < 0),
        "n_zero": sum(1 for d in diffs if d == 0),
    }
    if len(a) < 2:
        out["pvalue"] = None
        out["statistic"] = None
        out["note"] = "too few meters"
        return out
    if all(d == 0 for d in diffs):
        out["pvalue"] = 1.0
        out["statistic"] = 0.0
        out["note"] = "all differences zero"
        return out
    # zero_method='wilcox' drops zeros; alternative two-sided
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    out["statistic"] = float(stat)
    out["pvalue"] = float(p)
    out["note"] = None
    return out


def main():
    g3 = load_g3()
    moc_path, moc = load_moc_equal_share()
    co_moc = moc_path is not None and bool(moc)

    per_meter = {}
    for m in METERS:
        row = {"target_meter": m, "g3": {}, "moc_equal_share": None}
        for p in PS:
            mean_p, vals = mean_seeds(g3.get((m, p), {}))
            row["g3"][f"{p:g}"] = {
                "n_seeds": len(g3.get((m, p), {})),
                "seeds": sorted(g3.get((m, p), {})),
                "mean_mae_target": mean_p,
                "per_seed": {str(s): g3[(m, p)][s] for s in sorted(g3.get((m, p), {}))},
            }
        if co_moc and m in moc:
            mean_m, _ = mean_seeds(moc[m])
            row["moc_equal_share"] = {
                "n_seeds": len(moc[m]),
                "mean_mae": mean_m,
                "per_seed": {str(s): moc[m][s] for s in sorted(moc[m])},
            }
            if mean_m is not None:
                for p in PS:
                    g = row["g3"][f"{p:g}"]["mean_mae_target"]
                    if g is not None:
                        row["g3"][f"{p:g}"]["delta_vs_moc"] = g - mean_m
                        row["g3"][f"{p:g}"]["rel_vs_moc"] = (g - mean_m) / mean_m
        per_meter[m] = row

    # Mean over meters that have all 3 seeds for that p
    by_frac = {}
    for p in PS:
        means = []
        for m in METERS:
            v = per_meter[m]["g3"][f"{p:g}"]["mean_mae_target"]
            if v is not None and per_meter[m]["g3"][f"{p:g}"]["n_seeds"] == len(SEEDS):
                means.append((m, v))
        by_frac[f"{p:g}"] = {
            "target_frac": p,
            "n_meters": len(means),
            "mean10_of_target_mae": st.mean(v for _, v in means) if means else None,
            "per_meter": {m: v for m, v in means},
        }

    # Count stations where p=0.75 worse (higher MAE) than p=0.5
    worse_075 = []
    for m in METERS:
        a = per_meter[m]["g3"]["0.5"]["mean_mae_target"]
        b = per_meter[m]["g3"]["0.75"]["mean_mae_target"]
        if a is not None and b is not None and b > a:
            worse_075.append(m)

    # Wilcoxon on meters with both means
    def series(p):
        out_m, out_v = [], []
        for m in METERS:
            v = per_meter[m]["g3"][f"{p:g}"]["mean_mae_target"]
            if v is not None and per_meter[m]["g3"][f"{p:g}"]["n_seeds"] == len(SEEDS):
                out_m.append(m)
                out_v.append(v)
        return out_m, out_v

    m50, v50 = series(0.5)
    m75, v75 = series(0.75)
    m25, v25 = series(0.25)
    common_5075 = [m for m in m50 if m in m75]
    common_2550 = [m for m in m25 if m in m50]
    v50_c = [per_meter[m]["g3"]["0.5"]["mean_mae_target"] for m in common_5075]
    v75_c = [per_meter[m]["g3"]["0.75"]["mean_mae_target"] for m in common_5075]
    v25_c = [per_meter[m]["g3"]["0.25"]["mean_mae_target"] for m in common_2550]
    v50_c2 = [per_meter[m]["g3"]["0.5"]["mean_mae_target"] for m in common_2550]

    wil = {
        "p0.5_vs_p0.75": {
            **wilcoxon_paired(v50_c, v75_c, "mean_mae p0.5 vs p0.75"),
            "meters": common_5075,
            # positive diff means p0.5 higher MAE than p0.75
        },
        "p0.25_vs_p0.5": {
            **wilcoxon_paired(v25_c, v50_c2, "mean_mae p0.25 vs p0.5"),
            "meters": common_2550,
        },
    }
    # Also report how many times p0.75 > p0.5 among paired
    wil["p0.5_vs_p0.75"]["n_p075_worse_than_p05"] = sum(
        1 for a, b in zip(v50_c, v75_c) if b > a
    )

    out = {
        "horizon": HORIZON,
        "n_g3_runs": sum(len(v) for v in g3.values()),
        "n_expected_g3": len(METERS) * len(PS) * len(SEEDS),
        "moc_equal_share": {
            "found": co_moc,
            "path": moc_path,
            "note": (
                "run_matched_budget.py equal per-meter budget (chia deu); "
                "mae = eval_epoch joint MAE; horizon=48; mean over seeds 42,43,44"
                if co_moc
                else "khong co moc"
            ),
            "n_meters": len(moc) if co_moc else 0,
            "mean10_mae": (
                st.mean(mean_seeds(moc[m])[0] for m in METERS if m in moc and mean_seeds(moc[m])[0] is not None)
                if co_moc
                else None
            ),
        },
        "by_frac": by_frac,
        "n_meters_p075_worse_than_p05": len(worse_075),
        "meters_p075_worse_than_p05": worse_075,
        "wilcoxon": wil,
        "per_meter": per_meter,
    }

    out_dir = ROOT / "results" / "g3-so-moc"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / "summary.json", "w"), indent=2, ensure_ascii=False)

    # Console report
    print(
        "G3 so moc: n_g3=%d / %d, moc=%s"
        % (
            out["n_g3_runs"],
            out["n_expected_g3"],
            moc_path if co_moc else "khong co moc",
        )
    )
    for p in PS:
        bf = by_frac[f"{p:g}"]
        print(
            "p=%g n_meters=%d mean10_target_mae %s"
            % (
                p,
                bf["n_meters"],
                ("%.2f" % bf["mean10_of_target_mae"]) if bf["mean10_of_target_mae"] is not None else "NA",
            )
        )
    if co_moc:
        print(
            "moc equal_share mean10_mae %.2f path=%s"
            % (out["moc_equal_share"]["mean10_mae"], moc_path)
        )
    else:
        print("moc equal_share: khong co moc (chi so giua cac p)")

    print("bang tram (mean 3 seed):")
    hdr = "meter   moc_eq"
    for p in PS:
        hdr += f"   p={p:g}"
    print(hdr)
    for m in METERS:
        moc_s = per_meter[m]["moc_equal_share"]
        moc_v = moc_s["mean_mae"] if moc_s and moc_s["mean_mae"] is not None else None
        line = f"{m}  {moc_v:8.2f}" if moc_v is not None else f"{m}  {'NA':>8}"
        for p in PS:
            g = per_meter[m]["g3"][f"{p:g}"]["mean_mae_target"]
            line += f"  {g:8.2f}" if g is not None else f"  {'NA':>8}"
        print(line)

    print(
        "so tram p=0.75 xau hon p=0.5: %d / %d  %s"
        % (len(worse_075), len(common_5075), ",".join(worse_075) if worse_075 else "-")
    )
    for key in ("p0.5_vs_p0.75", "p0.25_vs_p0.5"):
        w = wil[key]
        print(
            "Wilcoxon %s: n=%d statistic=%s pvalue=%s n_pos=%d n_neg=%d note=%s"
            % (
                key,
                w["n_meters"],
                ("%.4g" % w["statistic"]) if w["statistic"] is not None else "NA",
                ("%.4g" % w["pvalue"]) if w["pvalue"] is not None else "NA",
                w["n_positive"],
                w["n_negative"],
                w["note"] or "-",
            )
        )


if __name__ == "__main__":
    main()
