"""PA-B: bang so hai cach doc (last vs full) tu results/g-chung.

Chi doc results/g-chung/*/summary.json. Ghi results/bang-hai-cach-doc/.
Cung cach tinh mean 10 cong to (trung binh 3 seed) nhu tong_g_chung.py.
Wilcoxon ghep theo cong to, HAI PHIA. CPU.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as st
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import scipy.stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METERS = [f"M{i:02d}" for i in range(1, 11)]
SEEDS = [42, 43, 44]
MODELS = ["bilstm", "armambakan"]
REGIMES = ["pooled", "local"]
READOUTS = ["last", "full"]
HORIZONS = [4, 48, 96, 336]
METRICS = ("mae_joint", "mae_p", "mae_q")
METRIC_TAG = {"mae_joint": "joint", "mae_p": "p", "mae_q": "q"}
MODEL_LABEL = {"bilstm": "BiLSTM", "armambakan": "AR-MambaKAN"}

# So ban nop (OMML tu _Nga.docx): Table 3 aggregate 40 task, Table 4/17 theo horizon.
BAN_NOP_T3 = {
    "bilstm": {"local": 677.2, "pooled": 574.0, "pct": -15.2, "tasks_improved": "32/40", "p": 3.7e-4},
    "armambakan": {"local": 696.9, "pooled": 579.2, "pct": -16.9, "tasks_improved": "35/40", "p": 8.1e-7},
}
BAN_NOP_T4 = {
    # pct change local->pooled, p Wilcoxon n=10
    ("bilstm", 4): {"pct": -34.7, "p": 0.005, "sig": True},
    ("bilstm", 48): {"pct": -23.3, "p": 0.022, "sig": True},
    ("bilstm", 96): {"pct": -16.0, "p": 0.093, "sig": False},
    ("bilstm", 336): {"pct": -1.8, "p": 0.333, "sig": False},
    ("armambakan", 4): {"pct": -30.1, "p": 0.009, "sig": True},
    ("armambakan", 48): {"pct": -24.3, "p": 0.007, "sig": True},
    ("armambakan", 96): {"pct": -17.6, "p": 0.009, "sig": True},
    ("armambakan", 336): {"pct": -6.3, "p": 0.028, "sig": True},
}
BAN_NOP_T5 = {
    "bilstm": {
        "local_p": 839.4, "pooled_p": 786.5, "pct_p": -6.3, "p_p": 1.1e-2,
        "local_q": 508.8, "pooled_q": 368.7, "pct_q": -27.5, "p_q": 5.8e-5,
    },
    "armambakan": {
        "local_p": 850.3, "pooled_p": 787.2, "pct_p": -7.4, "p_p": 2.3e-3,
        "local_q": 540.6, "pooled_q": 379.6, "pct_q": -29.8, "p_q": 2.6e-5,
    },
}
BAN_NOP_T17 = {
    # joint MAE theo horizon (seed-mean roi mean 10 tram)
    ("daily_naive", 4): 391.6,
    ("daily_naive", 48): 390.5,
    ("daily_naive", 96): 484.0,
    ("daily_naive", 336): 767.2,
    ("pooled", "bilstm", 4): 245.7,
    ("pooled", "bilstm", 48): 490.5,
    ("pooled", "bilstm", 96): 607.2,
    ("pooled", "bilstm", 336): 952.7,
    ("local", "bilstm", 4): 376.5,
    ("local", "bilstm", 48): 639.3,
    ("local", "bilstm", 96): 722.7,
    ("local", "bilstm", 336): 970.4,
    ("pooled", "armambakan", 4): 285.9,
    ("pooled", "armambakan", 48): 481.8,
    ("pooled", "armambakan", 96): 607.4,
    ("pooled", "armambakan", 336): 941.6,
    ("local", "armambakan", 4): 408.7,
    ("local", "armambakan", 48): 636.9,
    ("local", "armambakan", 96): 737.3,
    ("local", "armambakan", 336): 1004.6,
}


def is_null(v) -> bool:
    return v is None


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
    diffs = [a[m] - b[m] for m in common]  # a thang khi diff < 0
    a_wins = sum(1 for d in diffs if d < 0)
    return {
        f"mean_{a_name}": mean_or_none([a[m] for m in common]),
        f"mean_{b_name}": mean_or_none([b[m] for m in common]),
        "n_meters": len(common),
        f"{a_name}_wins": a_wins,
        "wilcoxon_p_two_sided": wilcoxon_two_sided(diffs),
        "per_meter": {
            m: {a_name: a[m], b_name: b[m], "diff": a[m] - b[m]} for m in common
        },
    }


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_cells() -> tuple[dict, int]:
    cells: dict[tuple, dict[tuple[int, str], dict[str, float]]] = defaultdict(dict)
    n_runs = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "results/g-chung/*/summary.json"))):
        s = json.load(open(f))
        if not (
            is_null(s.get("fold"))
            and is_null(s.get("ablation"))
            and is_null(s.get("target_meter"))
        ):
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
    return cells, n_runs


def main() -> None:
    cells, n_runs = load_cells()
    gchung = json.load(open(os.path.join(ROOT, "results/g-chung-tong/summary.json")))

    rows = []
    cell_means = {}
    check_last_vs_gchung = {"ok": True, "mismatches": []}

    for regime in REGIMES:
        for model in MODELS:
            for h in HORIZONS:
                row = {
                    "regime": regime,
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "horizon": h,
                }
                for metric in METRICS:
                    tag = METRIC_TAG[metric]
                    per = {}
                    for ro in READOUTS:
                        k = (model, regime, ro, h)
                        pm = per_meter_seed_mean(cells.get(k, {}), metric, METERS)
                        mean_v = mean_or_none(list(pm.values())) if pm else None
                        row[f"mae_{tag}_{ro}"] = mean_v
                        row[f"per_meter_{tag}_{ro}"] = pm
                        per[ro] = pm
                        cell_means[f"{model}|{regime}|{ro}|{h}|{tag}"] = mean_v

                    mean_l = row[f"mae_{tag}_last"]
                    mean_f = row[f"mae_{tag}_full"]
                    pct = None
                    if mean_l is not None and mean_f is not None and mean_l != 0:
                        pct = 100.0 * (mean_f / mean_l - 1.0)
                    row[f"pct_full_vs_last_{tag}"] = pct

                    # (a) full vs last
                    cmp_fl = compare_pair(
                        per["full"], per["last"], METERS, a_name="full", b_name="last"
                    )
                    row[f"full_vs_last_{tag}"] = {
                        "full_wins": cmp_fl["full_wins"],
                        "wilcoxon_p_two_sided": cmp_fl["wilcoxon_p_two_sided"],
                        "n_meters": cmp_fl["n_meters"],
                    }

                    # doi chieu last voi g-chung-tong.full_vs_last
                    ref = gchung["full_vs_last"].get(f"{model}|{regime}|{h}|{tag}")
                    if ref is not None and mean_l is not None:
                        delta = mean_l - ref["mean_last"]
                        if abs(delta) > 1e-9:
                            check_last_vs_gchung["ok"] = False
                            check_last_vs_gchung["mismatches"].append(
                                {
                                    "key": f"{model}|{regime}|{h}|{tag}",
                                    "this_last": mean_l,
                                    "gchung_last": ref["mean_last"],
                                    "delta": delta,
                                }
                            )

                # (b) AR vs BiLSTM pooled, chi ghi o hang pooled+armambakan (tranh trung)
                # (c) pooled vs local: ghi o moi hang theo model/horizon/readout
                for ro in READOUTS:
                    for metric in METRICS:
                        tag = METRIC_TAG[metric]
                        kp = (model, "pooled", ro, h)
                        kl = (model, "local", ro, h)
                        P = per_meter_seed_mean(cells.get(kp, {}), metric, METERS)
                        L = per_meter_seed_mean(cells.get(kl, {}), metric, METERS)
                        cmp_pl = compare_pair(
                            P, L, METERS, a_name="pooled", b_name="local"
                        )
                        row[f"pooled_vs_local_{ro}_{tag}"] = {
                            "mean_pooled": cmp_pl["mean_pooled"],
                            "mean_local": cmp_pl["mean_local"],
                            "pooled_wins": cmp_pl["pooled_wins"],
                            "wilcoxon_p_two_sided": cmp_pl["wilcoxon_p_two_sided"],
                        }

                    if regime == "pooled":
                        ka = ("armambakan", "pooled", ro, h)
                        kb = ("bilstm", "pooled", ro, h)
                        for metric in METRICS:
                            tag = METRIC_TAG[metric]
                            A = per_meter_seed_mean(cells.get(ka, {}), metric, METERS)
                            B = per_meter_seed_mean(cells.get(kb, {}), metric, METERS)
                            cmp_ab = compare_pair(
                                A, B, METERS, a_name="armambakan", b_name="bilstm"
                            )
                            row[f"ar_vs_bilstm_pooled_{ro}_{tag}"] = {
                                "mean_armambakan": cmp_ab["mean_armambakan"],
                                "mean_bilstm": cmp_ab["mean_bilstm"],
                                "armambakan_wins": cmp_ab["armambakan_wins"],
                                "wilcoxon_p_two_sided": cmp_ab["wilcoxon_p_two_sided"],
                            }

                rows.append(row)

    # Cac khoa Wilcoxon phang de verify_numbers.py do
    tests_a = {}
    tests_b = {}
    tests_c = {}
    for regime in REGIMES:
        for model in MODELS:
            for h in HORIZONS:
                for metric in METRICS:
                    tag = METRIC_TAG[metric]
                    kf = (model, regime, "full", h)
                    kl = (model, regime, "last", h)
                    F = per_meter_seed_mean(cells.get(kf, {}), metric, METERS)
                    Lt = per_meter_seed_mean(cells.get(kl, {}), metric, METERS)
                    cmp_fl = compare_pair(F, Lt, METERS, a_name="full", b_name="last")
                    tests_a[f"{model}|{regime}|{h}|{tag}"] = {
                        "mean_last": cmp_fl["mean_last"],
                        "mean_full": cmp_fl["mean_full"],
                        "pct_change_full_vs_last": (
                            100.0 * (cmp_fl["mean_full"] / cmp_fl["mean_last"] - 1.0)
                            if cmp_fl["mean_last"]
                            else None
                        ),
                        "full_wins": cmp_fl["full_wins"],
                        "wilcoxon_p_two_sided": cmp_fl["wilcoxon_p_two_sided"],
                    }

                    kp = (model, "pooled", "last", h)  # placeholder unused
                    _ = kp
                    for ro in READOUTS:
                        P = per_meter_seed_mean(
                            cells.get((model, "pooled", ro, h), {}), metric, METERS
                        )
                        L = per_meter_seed_mean(
                            cells.get((model, "local", ro, h), {}), metric, METERS
                        )
                        cmp_pl = compare_pair(
                            P, L, METERS, a_name="pooled", b_name="local"
                        )
                        tests_c[f"{model}|{ro}|{h}|{tag}"] = {
                            "mean_pooled": cmp_pl["mean_pooled"],
                            "mean_local": cmp_pl["mean_local"],
                            "pooled_wins": cmp_pl["pooled_wins"],
                            "wilcoxon_p_two_sided": cmp_pl["wilcoxon_p_two_sided"],
                        }

    for ro in READOUTS:
        for h in HORIZONS:
            for metric in METRICS:
                tag = METRIC_TAG[metric]
                A = per_meter_seed_mean(
                    cells.get(("armambakan", "pooled", ro, h), {}), metric, METERS
                )
                B = per_meter_seed_mean(
                    cells.get(("bilstm", "pooled", ro, h), {}), metric, METERS
                )
                cmp_ab = compare_pair(
                    A, B, METERS, a_name="armambakan", b_name="bilstm"
                )
                tests_b[f"pooled|{ro}|{h}|{tag}"] = {
                    "mean_armambakan": cmp_ab["mean_armambakan"],
                    "mean_bilstm": cmp_ab["mean_bilstm"],
                    "armambakan_wins": cmp_ab["armambakan_wins"],
                    "wilcoxon_p_two_sided": cmp_ab["wilcoxon_p_two_sided"],
                }

    # So last G-chung voi ban nop Table 3, 4, 5, 17
    lech_ban_nop = {"table3": [], "table4": [], "table5": [], "table17": [], "note": (
        "Table 3/5 ban nop = trung binh 40 task (10 tram x 4 horizon), seed-mean roi gop. "
        "G-chung so day la mean 4 horizon cua mean 10 tram (last). "
        "Table 4/17 so truc tiep theo horizon. "
        "Chenh lech tai lap = G-chung last - ban nop."
    )}

    for model in MODELS:
        means_l = [
            cell_means[f"{model}|local|last|{h}|joint"] for h in HORIZONS
        ]
        means_p = [
            cell_means[f"{model}|pooled|last|{h}|joint"] for h in HORIZONS
        ]
        g_local = st.mean(means_l)
        g_pooled = st.mean(means_p)
        bn = BAN_NOP_T3[model]
        lech_ban_nop["table3"].append(
            {
                "model": model,
                "gchung_mean4h_local_last": g_local,
                "gchung_mean4h_pooled_last": g_pooled,
                "ban_nop_local": bn["local"],
                "ban_nop_pooled": bn["pooled"],
                "delta_local": g_local - bn["local"],
                "delta_pooled": g_pooled - bn["pooled"],
                "pct_gchung": 100.0 * (g_pooled / g_local - 1.0),
                "pct_ban_nop": bn["pct"],
            }
        )

    for model in MODELS:
        for h in HORIZONS:
            g_p = cell_means[f"{model}|pooled|last|{h}|joint"]
            g_l = cell_means[f"{model}|local|last|{h}|joint"]
            pct = 100.0 * (g_p / g_l - 1.0)
            bn = BAN_NOP_T4[(model, h)]
            # Ban nop Table 4 khong in MAE thô, chi pct: doi chieu pct va p
            lech_ban_nop["table4"].append(
                {
                    "model": model,
                    "horizon": h,
                    "gchung_pooled_last": g_p,
                    "gchung_local_last": g_l,
                    "gchung_pct_pooled_vs_local": pct,
                    "ban_nop_pct": bn["pct"],
                    "delta_pct": pct - bn["pct"],
                    "gchung_wilcoxon_p_two_sided": tests_c[f"{model}|last|{h}|joint"][
                        "wilcoxon_p_two_sided"
                    ],
                    "ban_nop_p": bn["p"],
                    "ban_nop_sig_0.05": bn["sig"],
                }
            )

    for model in MODELS:
        means_lp = [cell_means[f"{model}|local|last|{h}|p"] for h in HORIZONS]
        means_pp = [cell_means[f"{model}|pooled|last|{h}|p"] for h in HORIZONS]
        means_lq = [cell_means[f"{model}|local|last|{h}|q"] for h in HORIZONS]
        means_pq = [cell_means[f"{model}|pooled|last|{h}|q"] for h in HORIZONS]
        bn = BAN_NOP_T5[model]
        lech_ban_nop["table5"].append(
            {
                "model": model,
                "gchung_mean4h_local_p": st.mean(means_lp),
                "gchung_mean4h_pooled_p": st.mean(means_pp),
                "gchung_mean4h_local_q": st.mean(means_lq),
                "gchung_mean4h_pooled_q": st.mean(means_pq),
                "ban_nop_local_p": bn["local_p"],
                "ban_nop_pooled_p": bn["pooled_p"],
                "ban_nop_local_q": bn["local_q"],
                "ban_nop_pooled_q": bn["pooled_q"],
                "delta_local_p": st.mean(means_lp) - bn["local_p"],
                "delta_pooled_p": st.mean(means_pp) - bn["pooled_p"],
                "delta_local_q": st.mean(means_lq) - bn["local_q"],
                "delta_pooled_q": st.mean(means_pq) - bn["pooled_q"],
            }
        )

    for regime in REGIMES:
        for model in MODELS:
            for h in HORIZONS:
                g = cell_means[f"{model}|{regime}|last|{h}|joint"]
                bn = BAN_NOP_T17[(regime, model, h)]
                lech_ban_nop["table17"].append(
                    {
                        "regime": regime,
                        "model": model,
                        "horizon": h,
                        "gchung_last_joint": g,
                        "ban_nop_joint": bn,
                        "delta": g - bn,
                        "rel_pct": 100.0 * (g / bn - 1.0) if bn else None,
                    }
                )

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "n_runs_read": n_runs,
        "models": MODELS,
        "regimes": REGIMES,
        "horizons": HORIZONS,
        "seeds": SEEDS,
        "meters": METERS,
        "check_last_vs_gchung_tong": check_last_vs_gchung,
        "rows": [
            {
                k: v
                for k, v in r.items()
                if not k.startswith("per_meter_")
            }
            for r in rows
        ],
        "rows_with_per_meter": rows,
        "tests_a_full_vs_last": tests_a,
        "tests_b_ar_vs_bilstm_pooled": tests_b,
        "tests_c_pooled_vs_local": tests_c,
        "lech_ban_nop": lech_ban_nop,
        "assumptions": [
            "Moi cong to = trung binh 3 seed (42,43,44).",
            "Mae in bang = trung binh 10 cong to.",
            "Wilcoxon signed-rank ghep theo cong to, alternative=two-sided, scipy method=auto.",
            "pct_full_vs_last = 100*(mean_full/mean_last - 1).",
            "Khong dung results/tb2-*.",
        ],
        "note": (
            "Bang so hai cach doc (PA-B). Khoa tests_a/b/c de verify_numbers.py do. "
            "lech_ban_nop ghi chenh lech tai lap last G-chung so voi Table 3,4,5,17."
        ),
    }

    out_dir = os.path.join(ROOT, "results/bang-hai-cach-doc")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    json.dump(out, open(out_path, "w"), indent=2)

    # bang.csv: hang gon de in
    csv_path = os.path.join(out_dir, "bang.csv")
    fieldnames = [
        "regime",
        "model",
        "horizon",
        "mae_joint_last",
        "mae_joint_full",
        "pct_full_vs_last_joint",
        "full_wins_joint",
        "p_full_vs_last_joint",
        "mae_p_last",
        "mae_p_full",
        "pct_full_vs_last_p",
        "mae_q_last",
        "mae_q_full",
        "pct_full_vs_last_q",
        "pooled_wins_last_joint",
        "p_pooled_vs_local_last_joint",
        "pooled_wins_full_joint",
        "p_pooled_vs_local_full_joint",
        "ar_wins_pooled_last_joint",
        "p_ar_vs_bilstm_pooled_last_joint",
        "ar_wins_pooled_full_joint",
        "p_ar_vs_bilstm_pooled_full_joint",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "regime": r["regime"],
                    "model": r["model"],
                    "horizon": r["horizon"],
                    "mae_joint_last": r["mae_joint_last"],
                    "mae_joint_full": r["mae_joint_full"],
                    "pct_full_vs_last_joint": r["pct_full_vs_last_joint"],
                    "full_wins_joint": r["full_vs_last_joint"]["full_wins"],
                    "p_full_vs_last_joint": r["full_vs_last_joint"][
                        "wilcoxon_p_two_sided"
                    ],
                    "mae_p_last": r["mae_p_last"],
                    "mae_p_full": r["mae_p_full"],
                    "pct_full_vs_last_p": r["pct_full_vs_last_p"],
                    "mae_q_last": r["mae_q_last"],
                    "mae_q_full": r["mae_q_full"],
                    "pct_full_vs_last_q": r["pct_full_vs_last_q"],
                    "pooled_wins_last_joint": r["pooled_vs_local_last_joint"][
                        "pooled_wins"
                    ],
                    "p_pooled_vs_local_last_joint": r["pooled_vs_local_last_joint"][
                        "wilcoxon_p_two_sided"
                    ],
                    "pooled_wins_full_joint": r["pooled_vs_local_full_joint"][
                        "pooled_wins"
                    ],
                    "p_pooled_vs_local_full_joint": r["pooled_vs_local_full_joint"][
                        "wilcoxon_p_two_sided"
                    ],
                    "ar_wins_pooled_last_joint": (
                        r.get("ar_vs_bilstm_pooled_last_joint", {}) or {}
                    ).get("armambakan_wins"),
                    "p_ar_vs_bilstm_pooled_last_joint": (
                        r.get("ar_vs_bilstm_pooled_last_joint", {}) or {}
                    ).get("wilcoxon_p_two_sided"),
                    "ar_wins_pooled_full_joint": (
                        r.get("ar_vs_bilstm_pooled_full_joint", {}) or {}
                    ).get("armambakan_wins"),
                    "p_ar_vs_bilstm_pooled_full_joint": (
                        r.get("ar_vs_bilstm_pooled_full_joint", {}) or {}
                    ).get("wilcoxon_p_two_sided"),
                }
            )

    # env
    try:
        env = subprocess.check_output(
            ["uv", "pip", "freeze"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        try:
            env = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL,
            ).decode()
        except Exception:
            env = "pip freeze unavailable\n"
    open(os.path.join(out_dir, "env.txt"), "w").write(env)

    print(
        "bang_hai_cach_doc: n_runs=%d last_vs_gchung_ok=%s -> %s"
        % (n_runs, check_last_vs_gchung["ok"], out_path)
    )
    if check_last_vs_gchung["mismatches"]:
        print("MISMATCHES:", check_last_vs_gchung["mismatches"][:10])

    print("\n== joint last | full (pct)  pooled/local x model x h")
    print(
        "%-8s %-12s %4s %10s %10s %8s %6s %8s"
        % ("regime", "model", "h", "last", "full", "pct", "Fwin", "p_FL")
    )
    for r in rows:
        fl = r["full_vs_last_joint"]
        print(
            "%-8s %-12s %4d %10.1f %10.1f %+7.1f%% %3d/10 %8s"
            % (
                r["regime"],
                r["model"],
                r["horizon"],
                r["mae_joint_last"],
                r["mae_joint_full"],
                r["pct_full_vs_last_joint"],
                fl["full_wins"],
                (
                    "%.4f" % fl["wilcoxon_p_two_sided"]
                    if fl["wilcoxon_p_two_sided"] is not None
                    else "NA"
                ),
            )
        )

    print("\n== lech Table 17 (joint last)")
    for e in lech_ban_nop["table17"]:
        if abs(e["delta"]) >= 0.5:
            print(
                "%s %s h%d gchung=%.2f ban_nop=%.1f delta=%+.2f"
                % (
                    e["regime"],
                    e["model"],
                    e["horizon"],
                    e["gchung_last_joint"],
                    e["ban_nop_joint"],
                    e["delta"],
                )
            )


if __name__ == "__main__":
    main()
