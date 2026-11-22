"""Tong hop G3 (matched budget, target_frac). Chi doc results/g3/*/summary.json, CPU.

Moi run G3 (g_chung/chay.py, regime=matched) chi cham tap kiem TRAM DICH, nen
per_meter chi co dung target_meter. mean_test_mae_joint == mae tram dich.

Voi moi (target_meter, target_frac): n_seeds, mean MAE joint tren tram dich.
"Trung binh 10 tram" = trung binh cac mean MAE tram dich theo target_frac
(cac tram da co du lieu). Ghi results/g3-tong/summary.json.
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METERS = [f"M{i:02d}" for i in range(1, 11)]
PS = [0.25, 0.5, 0.75]
SEEDS = [42, 43, 44]
N_EXPECTED = len(METERS) * len(PS) * len(SEEDS)  # 90, lo_g3.sh

d = {}  # (target_meter, target_frac) -> list of (seed, mae_target, mae_mean_in_summary)
have = set()
for f in sorted(glob.glob(os.path.join(ROOT, "results/g3/*/summary.json"))):
    s = json.load(open(f))
    tm = s["target_meter"]
    p = float(s["target_frac"])
    seed = int(s["seed"])
    have.add((tm, p, seed))
    pm = s["per_meter"]
    if tm not in pm:
        raise SystemExit(f"missing per_meter[{tm}] in {f}")
    mae_target = float(pm[tm]["test_mae"]["mae_joint"])
    mae_mean_in = float(st.mean(m["test_mae"]["mae_joint"] for m in pm.values()))
    d.setdefault((tm, p), []).append((seed, mae_target, mae_mean_in))

missing = [
    {"target_meter": m, "target_frac": p, "seed": seed}
    for m in METERS
    for p in PS
    for seed in SEEDS
    if (m, p, seed) not in have
]

n_meters_max = 0
for f in glob.glob(os.path.join(ROOT, "results/g3/*/summary.json")):
    n_meters_max = max(n_meters_max, len(json.load(open(f))["per_meter"]))

by_pair = {}
for m in METERS:
    for p in PS:
        rows = d.get((m, p), [])
        if not rows:
            continue
        seeds = sorted(r[0] for r in rows)
        by_pair[f"{m}|{p:g}"] = {
            "target_meter": m,
            "target_frac": p,
            "n_seeds": len(rows),
            "seeds": seeds,
            "mean_mae_target": st.mean(r[1] for r in rows),
            # G3 chi luu tram dich trong per_meter; bang mae_target khi n_meters=1
            "mean_mae_over_meters_in_summary": st.mean(r[2] for r in rows),
            "n_meters_in_summary_max": n_meters_max,
        }

by_frac = {}
for p in PS:
    meter_means = {}
    for m in METERS:
        key = f"{m}|{p:g}"
        if key in by_pair:
            meter_means[m] = by_pair[key]["mean_mae_target"]
    if not meter_means:
        continue
    by_frac[f"{p:g}"] = {
        "target_frac": p,
        "n_meters_with_data": len(meter_means),
        "n_seeds_min": min(by_pair[f"{m}|{p:g}"]["n_seeds"] for m in meter_means),
        "mean10_of_target_mae": st.mean(meter_means.values()),
        "per_meter_mean_mae_target": meter_means,
    }

out = {
    "n_runs": sum(len(v) for v in d.values()),
    "n_expected": N_EXPECTED,
    "n_missing": len(missing),
    "missing": missing,
    "note": (
        "G3 matched (chay.py) evaluates target meter only; per_meter has 1 key. "
        "mean_mae_target is the metric of interest. "
        "mean10_of_target_mae averages mean_mae_target across meters that have data "
        "for that target_frac (not a within-run 10-meter mean)."
    ),
    "by_pair": by_pair,
    "by_frac": by_frac,
}

os.makedirs(os.path.join(ROOT, "results/g3-tong"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/g3-tong/summary.json"), "w"), indent=2)

print(
    "G3 tong: n_runs=%d / %d expected, missing=%d"
    % (out["n_runs"], out["n_expected"], out["n_missing"])
)
for p in PS:
    bf = by_frac.get(f"{p:g}")
    if not bf:
        print("p=%g no data" % p)
        continue
    print(
        "p=%g n_meters=%d n_seeds_min=%d mean10_target_mae %.2f"
        % (p, bf["n_meters_with_data"], bf["n_seeds_min"], bf["mean10_of_target_mae"])
    )
for m in METERS:
    for p in PS:
        key = f"{m}|{p:g}"
        if key not in by_pair:
            print("%s p=%g MISSING all seeds" % (m, p))
            continue
        v = by_pair[key]
        print(
            "%s p=%g n_seeds=%d mae_target %.2f mae_over_pm %.2f"
            % (m, p, v["n_seeds"], v["mean_mae_target"], v["mean_mae_over_meters_in_summary"])
        )
if missing:
    print("missing pairs:")
    for x in missing:
        print("  %s p=%g seed=%d" % (x["target_meter"], x["target_frac"], x["seed"]))
