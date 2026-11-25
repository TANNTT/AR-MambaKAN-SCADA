"""Compare G0 (readout=last, h48, seed 42) with the reference CSV of the submitted version. Written 2026-11-21.

Reference: pooled_multihorizon_results.csv (bilstm pooled), armambakan_stemfix_{pooled,local}.csv,
pq_split_{local,pooled}.csv (mae_p, mae_q). Joint MAE = (mae_p + mae_q)/2.
Seed spread = max-min range across seeds 42, 43, 44 of the same task in the reference CSV.
Output: results/g0-so-moc/summary.json
"""
import csv, json, statistics, pathlib, sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from _paths import require_env

A = require_env("SCADA_ARTIFACTS_DIR")
MAP = require_env("SCADA_ANON_MAP")
R = _ROOT / "results"
G = R / "g-chung"
OUT = R / "g0-so-moc"

code2m = {r["real_code"]: r["anon_label"] for r in csv.DictReader(open(MAP))}


def load(fn, regime=None, model=None, val=None):
    d = {}
    for r in csv.DictReader(open(A / fn)):
        if int(r["horizon"]) != 48:
            continue
        if regime and r.get("regime") != regime:
            continue
        if model and r["model"] != model:
            continue
        m = code2m.get(r["code"])
        if m is None:
            continue
        if val:
            v = float(r[val])
        else:
            v = (float(r["mae_p"]) + float(r["mae_q"])) / 2
        d.setdefault(m, {})[int(r["seed"])] = v
    return d


moc = {
    ("bilstm", "pooled", "joint"): load("pooled_multihorizon_results.csv", model="bilstm", val="pooled_mae"),
    ("bilstm", "pooled", "pq"): load("pq_split_pooled.csv", regime="pooled", model="bilstm"),
    ("bilstm", "local", "pq"): load("pq_split_local.csv", regime="local", model="bilstm"),
    ("armambakan", "pooled", "joint"): load("armambakan_stemfix_pooled.csv", model="armambakan", val="pooled_mae"),
    ("armambakan", "pooled", "pq"): load("pq_split_pooled.csv", regime="pooled", model="armambakan"),
    ("armambakan", "local", "joint"): load("armambakan_stemfix_local.csv", model="armambakan", val="mae"),
    ("armambakan", "local", "pq"): load("pq_split_local.csv", regime="local", model="armambakan"),
}


def g0(model, regime, readout, meter):
    if regime == "pooled":
        p = G / f"{model}_pooled_{readout}_h48_s42" / "summary.json"
        s = json.load(open(p))
        return s["per_meter"][meter]["test_mae"]["mae_joint"], s.get("git_commit"), s.get("best_epoch")
    p = G / f"{model}_local_{readout}_h48_s42_{meter}" / "summary.json"
    s = json.load(open(p))
    pm = s["per_meter"][meter]
    return pm["test_mae"]["mae_joint"], s.get("git_commit"), s.get("best_epoch")


rows = []
for (model, regime, src), d in moc.items():
    meters = [f"M{i:02d}" for i in range(1, 11)] if regime == "pooled" else ["M01", "M02"]
    for m in meters:
        if m not in d or 42 not in d[m]:
            continue
        seeds = d[m]
        spread = max(seeds.values()) - min(seeds.values()) if len(seeds) > 1 else None
        row = {"model": model, "regime": regime, "moc": src, "meter": m,
               "moc_s42": seeds[42], "moc_seeds": seeds, "spread_seed": spread}
        for ro in ("last", "full"):
            v, c, be = g0(model, regime, ro, m)
            row[f"g0_{ro}"] = v
            row[f"commit_{ro}"] = c
            row[f"best_epoch_{ro}"] = be
        row["lech_last"] = row["g0_last"] - row["moc_s42"]
        row["lech_tuong_doi_last"] = row["lech_last"] / row["moc_s42"]
        row["trong_dao_dong"] = (spread is not None and abs(row["lech_last"]) <= spread)
        row["lech_full_vs_last"] = row["g0_full"] - row["g0_last"]
        rows.append(row)


def mean(xs):
    return statistics.fmean(xs) if xs else None


tong = {}
for (model, regime, src) in moc:
    rs = [r for r in rows if (r["model"], r["regime"], r["moc"]) == (model, regime, src)]
    if not rs:
        continue
    seeds_mean = {}
    for s in (42, 43, 44):
        vals = [r["moc_seeds"].get(s) for r in rs]
        if all(v is not None for v in vals):
            seeds_mean[s] = mean(vals)
    tong[f"{model}_{regime}_{src}"] = {
        "n_meter": len(rs),
        "moc_s42_mean": mean([r["moc_s42"] for r in rs]),
        "moc_mean_theo_seed": seeds_mean,
        "g0_last_mean": mean([r["g0_last"] for r in rs]),
        "g0_full_mean": mean([r["g0_full"] for r in rs]),
        "lech_tuong_doi_last_mean": mean([r["lech_tuong_doi_last"] for r in rs]),
        "so_meter_trong_dao_dong": sum(r["trong_dao_dong"] for r in rs),
    }

# Commit of each G0 run (read from the run's summary.json)
g0_commits = {}
for p in sorted(G.glob("*/summary.json")):
    s = json.load(open(p))
    g0_commits[p.parent.name] = {
        "git_commit": s.get("git_commit"),
        "best_epoch": s.get("best_epoch"),
        "stopped_epoch": s.get("stopped_epoch"),
        "model": s.get("model"),
        "regime": s.get("regime"),
        "readout": s.get("readout"),
        "horizon": s.get("horizon"),
        "seed": s.get("seed"),
    }

OUT.mkdir(parents=True, exist_ok=True)
json.dump(
    {"tong": tong, "g0_commits": g0_commits, "rows": rows},
    open(OUT / "summary.json", "w"),
    indent=2,
    ensure_ascii=False,
)
for k, v in tong.items():
    print(k, json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "moc_mean_theo_seed"}),
          {s: round(x, 2) for s, x in v["moc_mean_theo_seed"].items()})
print()
for r in rows:
    print(f'{r["model"]:10s} {r["regime"]:6s} {r["moc"]:5s} {r["meter"]} moc42={r["moc_s42"]:9.2f} '
          f'spread={(r["spread_seed"] or 0):7.2f} last={r["g0_last"]:9.2f} full={r["g0_full"]:9.2f} '
          f'lech={100*r["lech_tuong_doi_last"]:6.1f}% ok={r["trong_dao_dong"]} be={r["best_epoch_last"]}/{r["best_epoch_full"]} '
          f'c={str(r["commit_last"])[:7]}/{str(r["commit_full"])[:7]}')
