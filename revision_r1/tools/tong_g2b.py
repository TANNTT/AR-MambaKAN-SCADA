"""Tong hop G2b (ablation, readout=full) so voi component_ablation_summary.csv (readout=last, ban nop).
Ghi results/g2b-tong/summary.json. Chi doc, CPU.
"""
import csv
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from _paths import require_env
REF = str(require_env("SCADA_PUBLIC_REPO") / "results" / "component_ablation_summary.csv")
V = {"A3": "A3_bilstm", "A4": "A4_full", "A5": "A5_no_stem", "A6": "A6_no_kan"}

ref = {r["code"]: r for r in csv.DictReader(open(REF))}
d = {}
for f in glob.glob(os.path.join(ROOT, "results/g2b/*/summary.json")):
    s = json.load(open(f))
    d.setdefault((s["ablation"], s["meter_labels"][0]), []).append(s["mean_test_mae_joint"])
ms = sorted({k[1] for k in d})
out = {"n_runs": sum(len(v) for v in d.values()), "meters": ms, "variants": {}, "per_meter": {}}
for a, col in V.items():
    full = {m: st.mean(d[(a, m)]) for m in ms}
    last = {m: float(ref[m][col]) for m in ms}
    out["variants"][a] = {
        "n_seeds_min": min(len(d[(a, m)]) for m in ms),
        "full_mean10": st.mean(full.values()),
        "last_mean10": st.mean(last.values()),
        "full_mean_noM08": st.mean(v for m, v in full.items() if m != "M08"),
        "last_mean_noM08": st.mean(v for m, v in last.items() if m != "M08"),
        "full_per_meter": full,
        "last_per_meter": last,
    }
for m in ms:
    f = {a: out["variants"][a]["full_per_meter"][m] for a in V}
    l = {a: out["variants"][a]["last_per_meter"][m] for a in V}
    out["per_meter"][m] = {"best_full": min(f, key=f.get), "best_last": min(l, key=l.get)}
for tag in ("full", "last"):
    out["wins_" + tag] = {a: sum(1 for m in ms if out["per_meter"][m]["best_" + tag] == a) for a in V}
os.makedirs(os.path.join(ROOT, "results/g2b-tong"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/g2b-tong/summary.json"), "w"), indent=2)
for a in V:
    v = out["variants"][a]
    print(a, "n_seeds>=%d full %.2f last %.2f | noM08 full %.2f last %.2f" % (
        v["n_seeds_min"], v["full_mean10"], v["last_mean10"], v["full_mean_noM08"], v["last_mean_noM08"]))
print("wins_full", out["wins_full"], "wins_last", out["wins_last"])
for m in ms:
    print(m, out["per_meter"][m], " ".join("%s=%.1f/%.1f" % (a, out["variants"][a]["full_per_meter"][m], out["variants"][a]["last_per_meter"][m]) for a in V))
