#!/usr/bin/env python3
"""Export print tables and R2 / two-readout figures from existing CPU results (option B).

Assumptions:
1. Source numbers are read-only: results/bang-hai-cach-doc, r1-r2-cpu, p-hai-phia (not modified).
2. The main R2 table uses readout last (option B). Every p is a TWO-SIDED Wilcoxon paired by meter.
3. MAE printed with 1 decimal, MAPE with 2, p with 3 significant digits; each cell keeps the raw_* values alongside.
4. Figure (a) meter: M01 = first meter in the quality ranking (MAIN_METERS) that has the 3 representative days.
5. Figure size: IET/Wiley single column = 84 mm (same as Springer column=1 in kieu-trinh-bay).
6. MAPE threshold 5% |y|_mean fixed in phan_tich_r1_r2_cpu; unchanged.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phan_tich_r1_r2_cpu import (  # noqa: E402
    SeriesCache,
    daily_naive_pred_from_origin,
    second_week_wednesday,
    seed_mean_pred_gchung,
    select_daily_window_indices,
)

OUT = ROOT / "results" / "bang-sua"
HINH = OUT / "hinh"
METERS = [f"M{i:02d}" for i in range(1, 11)]
SEASONS = ("dong", "xuan", "he", "thu")
SEASON_EN = {"dong": "winter", "xuan": "spring", "he": "summer", "thu": "autumn"}
CFGS = (
    "daily_naive",
    "pooled_bilstm",
    "pooled_armambakan",
    "local_bilstm",
    "local_armambakan",
)
CFG_LABEL = {
    "daily_naive": "daily-naive",
    "pooled_bilstm": "pooled BiLSTM",
    "pooled_armambakan": "pooled AR-MambaKAN",
    "local_bilstm": "local BiLSTM",
    "local_armambakan": "local AR-MambaKAN",
}
REP_LABELS = ("2022-10", "2023-01", "2023-04")
# IET GTD (Wiley) single-column artwork width commonly 84 mm
IET_COT1_MM = 84.0
MM_PER_IN = 25.4


def _load_kieu():
    import os, sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from _paths import require_env
    path = require_env("SCADA_KIEU_TRINH_BAY")
    spec = importlib.util.spec_from_file_location("kieu_trinh_bay", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def round_mae(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), 1)


def round_mape(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), 2)


def round_pct(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), 2)


def round_p(x: float | None) -> float | None:
    """3 significant figures."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    v = float(x)
    if v == 0.0:
        return 0.0
    return float(f"{v:.3g}")


def mean_finite(vals: list[float]) -> float | None:
    arr = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not arr:
        return None
    return float(np.mean(arr))


def median_finite(vals: list[float]) -> float | None:
    arr = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not arr:
        return None
    return float(np.median(arr))


def wilcoxon_two_sided(model: dict[str, float], base: dict[str, float]) -> dict:
    meters = [m for m in METERS if m in model and m in base]
    diffs = np.array([base[m] - model[m] for m in meters], dtype=np.float64)
    wins = int(sum(model[m] < base[m] for m in meters))
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return {
            "n": len(meters),
            "n_model_wins": wins,
            "p_two_sided": None,
            "note": "all differences zero",
        }
    try:
        res = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return {
            "n": len(meters),
            "n_model_wins": wins,
            "wilcoxon_stat": float(res.statistic),
            "p_two_sided": float(res.pvalue),
        }
    except ValueError as e:
        return {
            "n": len(meters),
            "n_model_wins": wins,
            "p_two_sided": None,
            "note": str(e),
        }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def build_bang_hai_cach_doc(src: dict) -> list[dict]:
    rows = []
    for r in src["rows"]:
        regime, model, H = r["regime"], r["model"], r["horizon"]
        cell = {
            "regime": regime,
            "model": r["model_label"],
            "model_key": model,
            "horizon": H,
            "mae_joint_last": round_mae(r["mae_joint_last"]),
            "mae_joint_full": round_mae(r["mae_joint_full"]),
            "pct_full_vs_last": round_pct(r["pct_full_vs_last_joint"]),
            "p_full_vs_last": round_p(r["full_vs_last_joint"]["wilcoxon_p_two_sided"]),
            "full_wins": r["full_vs_last_joint"]["full_wins"],
            "p_pooled_vs_local_last": round_p(
                r["pooled_vs_local_last_joint"]["wilcoxon_p_two_sided"]
            ),
            "pooled_wins_last": r["pooled_vs_local_last_joint"]["pooled_wins"],
            "p_pooled_vs_local_full": round_p(
                r["pooled_vs_local_full_joint"]["wilcoxon_p_two_sided"]
            ),
            "pooled_wins_full": r["pooled_vs_local_full_joint"]["pooled_wins"],
            "raw_mae_joint_last": r["mae_joint_last"],
            "raw_mae_joint_full": r["mae_joint_full"],
            "raw_pct_full_vs_last": r["pct_full_vs_last_joint"],
            "raw_p_full_vs_last": r["full_vs_last_joint"]["wilcoxon_p_two_sided"],
            "raw_p_pooled_vs_local_last": r["pooled_vs_local_last_joint"][
                "wilcoxon_p_two_sided"
            ],
            "raw_p_pooled_vs_local_full": r["pooled_vs_local_full_joint"][
                "wilcoxon_p_two_sided"
            ],
        }
        if regime == "pooled":
            ar_last = r["ar_vs_bilstm_pooled_last_joint"]
            ar_full = r["ar_vs_bilstm_pooled_full_joint"]
            cell.update(
                {
                    "p_ar_vs_bilstm_pooled_last": round_p(
                        ar_last["wilcoxon_p_two_sided"]
                    ),
                    "ar_wins_pooled_last": ar_last["armambakan_wins"],
                    "p_ar_vs_bilstm_pooled_full": round_p(
                        ar_full["wilcoxon_p_two_sided"]
                    ),
                    "ar_wins_pooled_full": ar_full["armambakan_wins"],
                    "raw_p_ar_vs_bilstm_pooled_last": ar_last[
                        "wilcoxon_p_two_sided"
                    ],
                    "raw_p_ar_vs_bilstm_pooled_full": ar_full[
                        "wilcoxon_p_two_sided"
                    ],
                }
            )
        else:
            cell.update(
                {
                    "p_ar_vs_bilstm_pooled_last": None,
                    "ar_wins_pooled_last": None,
                    "p_ar_vs_bilstm_pooled_full": None,
                    "ar_wins_pooled_full": None,
                    "raw_p_ar_vs_bilstm_pooled_last": None,
                    "raw_p_ar_vs_bilstm_pooled_full": None,
                }
            )
        rows.append(cell)
    return rows


def build_r2_mua(r2: dict) -> list[dict]:
    """Mean over meters, last readout, h48, P/Q by season."""
    seasons = [s for s in r2["seasons"] if s["readout"] == "last"]
    rows = []
    for season in SEASONS:
        for ch in ("P", "Q"):
            subset = [s for s in seasons if s["season"] == season and s["channel"] == ch]
            n_days_list = [s["n_days"] for s in subset]
            it_mau = any(s["it_mau"] for s in subset)
            n_meters = len(subset)
            for cfg in CFGS:
                maes = [s["configs"][cfg]["mae"] for s in subset]
                mapes = [s["configs"][cfg]["mape_pct"] for s in subset]
                n_excl = [s["configs"][cfg]["mape_n_excluded"] for s in subset]
                n_used = [s["configs"][cfg]["mape_n_used"] for s in subset]
                ratios = [
                    s["configs"][cfg]["ratio_sum_abs_e_over_abs_y"] for s in subset
                ]
                rows.append(
                    {
                        "season": season,
                        "season_en": SEASON_EN[season],
                        "channel": ch,
                        "config": cfg,
                        "config_label": CFG_LABEL[cfg],
                        "mae": round_mae(mean_finite(maes)),
                        "mape_pct": round_mape(mean_finite(mapes)),
                        "ratio_sum": (
                            round(mean_finite(ratios), 4)
                            if mean_finite(ratios) is not None
                            else None
                        ),
                        "n_days_mean": round(mean_finite(n_days_list) or 0, 1),
                        "n_days_min": int(min(n_days_list)) if n_days_list else 0,
                        "n_days_max": int(max(n_days_list)) if n_days_list else 0,
                        "n_meters": n_meters,
                        "it_mau": it_mau,
                        "mape_n_excluded_sum": int(sum(n_excl)),
                        "mape_n_used_sum": int(sum(n_used)),
                        "raw_mae": mean_finite(maes),
                        "raw_mape_pct": mean_finite(mapes),
                        "raw_ratio_sum": mean_finite(ratios),
                    }
                )
    return rows


def build_r2_tuan(r2: dict) -> list[dict]:
    weeks = [w for w in r2["weeks"] if w["readout"] == "last"]
    rows = []
    for season in SEASONS:
        for ch in ("P", "Q"):
            subset = [w for w in weeks if w["season"] == season and w["channel"] == ch]
            iso_weeks = sorted({w["iso_week"] for w in subset})
            for cfg in CFGS:
                mapes = [w["configs"][cfg]["mape_pct"] for w in subset]
                rows.append(
                    {
                        "season": season,
                        "season_en": SEASON_EN[season],
                        "channel": ch,
                        "config": cfg,
                        "config_label": CFG_LABEL[cfg],
                        "wmape_mean": round_mape(mean_finite(mapes)),
                        "wmape_median": round_mape(median_finite(mapes)),
                        "n_meter_weeks": len(mapes),
                        "n_unique_iso_weeks": len(iso_weeks),
                        "raw_wmape_mean": mean_finite(mapes),
                        "raw_wmape_median": median_finite(mapes),
                    }
                )
    return rows


def build_r2_ngay(r2: dict, r2g4: dict) -> dict:
    rep = [r for r in r2["representative_wednesdays"] if r["readout"] == "last"]
    rep_rows = []
    for label in REP_LABELS:
        for ch in ("P", "Q"):
            found = [r for r in rep if r["label"] == label and r["found"]]
            missing = [
                r["meter"] for r in rep if r["label"] == label and not r["found"]
            ]
            date = next((r["date"] for r in rep if r["label"] == label), None)
            for cfg in CFGS:
                mapes = [r["configs"][cfg][ch]["mape_pct"] for r in found]
                maes = [r["configs"][cfg][ch]["mae"] for r in found]
                n_used = [r["configs"][cfg][ch].get("mape_n_used") for r in found]
                rep_rows.append(
                    {
                        "kind": "representative",
                        "label": label,
                        "date": date,
                        "channel": ch,
                        "config": cfg,
                        "config_label": CFG_LABEL[cfg],
                        "dmape_pct": round_mape(mean_finite(mapes)),
                        "dmae": round_mae(mean_finite(maes)),
                        "n_meters": len(found),
                        "meters_missing": ",".join(sorted(set(missing))),
                        "mape_n_used_sum": int(sum(x or 0 for x in n_used)),
                        "raw_dmape_pct": mean_finite(mapes),
                        "raw_dmae": mean_finite(maes),
                    }
                )

    # Same calendar Wed week-2: 2022 (G4) and 2023 (main test) already in one file
    cmp_all = r2g4["daily_2022_vs_2023"]
    by_label: dict[str, list] = defaultdict(list)
    for r in cmp_all:
        by_label[r["label"]].append(r)

    pair_rows = []
    for label_2022 in sorted(lb for lb in by_label if lb.startswith("2022-")):
        month_i = int(label_2022.split("-")[1])
        label_2023 = f"2023-{month_i:02d}"
        rows22 = by_label[label_2022]
        rows23 = by_label.get(label_2023, [])
        date_2022 = next((r["date"] for r in rows22 if r.get("found")), None)
        date_2023 = next((r["date"] for r in rows23 if r.get("found")), None)
        src22 = next((r.get("source") for r in rows22), None)
        for ch in ("P", "Q"):
            found22 = [r for r in rows22 if r.get("found")]
            miss22 = [r["meter"] for r in rows22 if not r.get("found")]
            found23 = [r for r in rows23 if r.get("found")]
            miss23 = [r["meter"] for r in rows23 if not r.get("found")]
            for cfg in CFGS:
                mapes22 = [
                    r["configs"][cfg][ch]["mape_pct"]
                    for r in found22
                    if cfg in r.get("configs", {}) and ch in r["configs"][cfg]
                ]
                mapes23 = [
                    r["configs"][cfg][ch]["mape_pct"]
                    for r in found23
                    if cfg in r.get("configs", {}) and ch in r["configs"][cfg]
                ]
                pair_rows.append(
                    {
                        "kind": "calendar_2022_vs_2023",
                        "label_2022": label_2022,
                        "date_2022": date_2022,
                        "source_2022": src22,
                        "label_2023": label_2023 if rows23 else None,
                        "date_2023": date_2023,
                        "channel": ch,
                        "config": cfg,
                        "config_label": CFG_LABEL[cfg],
                        "dmape_2022": round_mape(mean_finite(mapes22)),
                        "dmape_2023": round_mape(mean_finite(mapes23))
                        if rows23
                        else None,
                        "n_meters_2022": len(found22),
                        "n_meters_2023": len(found23),
                        "meters_missing_2022": ",".join(sorted(set(miss22))),
                        "meters_missing_2023": (
                            ",".join(sorted(set(miss23)))
                            if rows23
                            else "no_2023_counterpart_in_main_test"
                        ),
                        "raw_dmape_2022": mean_finite(mapes22),
                        "raw_dmape_2023": mean_finite(mapes23) if rows23 else None,
                    }
                )

    return {"representative": rep_rows, "calendar_2022_vs_2023": pair_rows}


def build_r2_he_f3(r2g4: dict, p_hai: dict) -> list[dict]:
    summer = r2g4["f3_summer"]
    pmap = {
        (x["cell_id"].split("|")[2], x["cell_id"].split("|")[3]): x
        for x in p_hai["rebuttal_two_sided"]["r2_f3_summer"]
    }
    # cell_id like R2|F3summer|P|bilstm
    rows = []
    for ch in ("P", "Q"):
        subset = [s for s in summer if s["channel"] == ch]
        n_days = [s["n_days"] for s in subset]
        for arch in ("bilstm", "armambakan"):
            pooled = f"pooled_{arch}"
            local = f"local_{arch}"
            mae_p = [s["configs"][pooled]["mae"] for s in subset]
            mae_l = [s["configs"][local]["mae"] for s in subset]
            mape_p = [s["configs"][pooled]["mape_pct"] for s in subset]
            mape_l = [s["configs"][local]["mape_pct"] for s in subset]
            n_excl_p = [s["configs"][pooled].get("mape_n_excluded", 0) for s in subset]
            n_excl_l = [s["configs"][local].get("mape_n_excluded", 0) for s in subset]
            # wins from per-meter
            wins = int(sum(a < b for a, b in zip(mae_p, mae_l)))
            # two-sided from p-hai-phia
            pcell = pmap.get((ch, arch))
            p_two = pcell["p_two_sided"] if pcell else None
            rows.append(
                {
                    "fold": "F3",
                    "season": "he",
                    "channel": ch,
                    "architecture": arch,
                    "mae_pooled": round_mae(mean_finite(mae_p)),
                    "mae_local": round_mae(mean_finite(mae_l)),
                    "mape_pooled": round_mape(mean_finite(mape_p)),
                    "mape_local": round_mape(mean_finite(mape_l)),
                    "pooled_wins": wins,
                    "n_meters": len(subset),
                    "n_days_mean": round(mean_finite(n_days) or 0, 1),
                    "p_two_sided": round_p(p_two),
                    "mape_n_excluded_pooled_sum": int(sum(n_excl_p)),
                    "mape_n_excluded_local_sum": int(sum(n_excl_l)),
                    "raw_mae_pooled": mean_finite(mae_p),
                    "raw_mae_local": mean_finite(mae_l),
                    "raw_mape_pooled": mean_finite(mape_p),
                    "raw_mape_local": mean_finite(mape_l),
                    "raw_p_two_sided": p_two,
                    "conclusion_flips_vs_one_sided": pcell["conclusion_flips_at_0.05"]
                    if pcell
                    else None,
                }
            )
    return rows


def build_r1_1_hai_phia(r11: dict, p_hai: dict) -> list[dict]:
    """Models vs daily-naive, two-sided p, last readout, joint/P/Q."""
    # joint from p-hai-phia for cross-check; P/Q recomputed from per_meter
    joint_lookup = {
        x["cell_id"]: x for x in p_hai["rebuttal_two_sided"]["r1_1_gchung_joint"]
    }
    rows = []
    for cell in r11["cells"]:
        if cell["readout"] != "last":
            continue
        H = cell["horizon"]
        ch = cell["channel"]
        naive = cell["daily_naive_per_meter"]
        for cfg, mdat in cell["models"].items():
            per = mdat["per_meter"]
            w = wilcoxon_two_sided(per, naive)
            cell_id = f"R1.1|gchung|last|h{H}|{ch}|{cfg}"
            p_joint_ref = joint_lookup.get(
                f"R1.1|gchung|last|h{H}|joint|{cfg}"
            )
            p_two = w["p_two_sided"]
            # Prefer p-hai-phia value for joint to match published rebuttal file
            if ch == "joint" and p_joint_ref is not None:
                p_two = p_joint_ref["p_two_sided"]
                # sanity: recompute should match closely
                if w["p_two_sided"] is not None and abs(
                    w["p_two_sided"] - p_two
                ) > 1e-9:
                    raise RuntimeError(
                        f"p mismatch joint {cell_id}: "
                        f"{w['p_two_sided']} vs {p_two}"
                    )
            rows.append(
                {
                    "horizon": H,
                    "channel": ch,
                    "config": cfg,
                    "config_label": CFG_LABEL.get(cfg, cfg),
                    "mae_model": round_mae(mdat["mean"]),
                    "mae_daily_naive": round_mae(cell["daily_naive_mean"]),
                    "n_model_wins": w["n_model_wins"],
                    "p_two_sided": round_p(p_two),
                    "raw_mae_model": mdat["mean"],
                    "raw_mae_daily_naive": cell["daily_naive_mean"],
                    "raw_p_two_sided": p_two,
                    "raw_p_one_sided_from_r1r2": mdat["vs_daily_naive"].get(
                        "wilcoxon_pvalue"
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def choose_meter_for_fig(r2: dict) -> str:
    """First MAIN_METERS entry with all three representative days present."""
    rep = [r for r in r2["representative_wednesdays"] if r["readout"] == "last"]
    by_m: dict[str, dict] = defaultdict(dict)
    for r in rep:
        by_m[r["meter"]][r["label"]] = r["found"]
    for m in METERS:
        if all(by_m[m].get(lab, False) for lab in REP_LABELS):
            return m
    raise RuntimeError("no meter has all three representative days")


def ve_hinh_du_bao_ba_ngay(meter: str, kieu, out_stem: Path) -> dict:
    """(a) Forecast vs measured on three representative days, P and Q."""
    cache = SeriesCache()
    series, times = cache.get(meter)
    pred_b, y, t_o, t_t = seed_mean_pred_gchung("bilstm", "pooled", "last", 48, meter)
    pred_a, y2, t_o2, t_t2 = seed_mean_pred_gchung(
        "armambakan", "pooled", "last", 48, meter
    )
    if not np.array_equal(t_o, t_o2) or not np.allclose(y, y2, rtol=1e-5, atol=1e-3):
        raise RuntimeError("origin/y mismatch bilstm vs armambakan for figure")
    dpred = daily_naive_pred_from_origin(series, times, t_o, 48)
    day_idx, day_meta = select_daily_window_indices(t_t)
    day_starts = pd.to_datetime(t_t[day_idx, 0]).normalize()

    rep_specs = [
        ("2022-10", second_week_wednesday(2022, 10)),
        ("2023-01", second_week_wednesday(2023, 1)),
        ("2023-04", second_week_wednesday(2023, 4)),
    ]

    so_lieu: dict[str, Any] = {
        "figure": "r2_du_bao_ba_ngay",
        "meter_anonymous": meter,
        "meter_rule": "first MAIN_METERS (quality rank order) with all three days",
        "readout": "last",
        "horizon": 48,
        "seed_mean": [42, 43, 44],
        "daily_window_rule": day_meta,
        "days": {},
        "source_runs": {
            "pooled_bilstm": "results/g-chung/bilstm_pooled_last_h48_s{42,43,44}/pred_{meter}.npz",
            "pooled_armambakan": "results/g-chung/armambakan_pooled_last_h48_s{42,43,44}/pred_{meter}.npz",
            "daily_naive": "phan_tich_r1_r2_cpu.daily_naive_pred_from_origin",
        },
    }

    kieu.ap_dung("springer", co_chu_pt=8.0)
    w = IET_COT1_MM / MM_PER_IN
    h = w * 1.15
    fig, axes = plt.subplots(2, 3, figsize=(w, h), sharex=False)
    styles = kieu.CHUOI_STYLE

    for col, (label, ts) in enumerate(rep_specs):
        hits = np.where(day_starts == ts)[0]
        if len(hits) == 0:
            raise RuntimeError(f"{meter} missing day {label}")
        wi = int(day_idx[int(hits[0])])
        t_steps = pd.to_datetime(t_t[wi])
        hours = (t_steps - t_steps[0]).total_seconds() / 3600.0
        day_pack = {
            "label": label,
            "date": str(ts.date()),
            "window_index": int(wi),
            "n_steps": 48,
            "channels": {},
        }
        for row, ch_name, ch_idx in ((0, "P", 0), (1, "Q", 1)):
            ax = axes[row, col]
            y_ch = y[wi, :, ch_idx]
            curves = {
                "measured": y_ch.tolist(),
                "daily_naive": dpred[wi, :, ch_idx].tolist(),
                "pooled_bilstm": pred_b[wi, :, ch_idx].tolist(),
                "pooled_armambakan": pred_a[wi, :, ch_idx].tolist(),
            }
            day_pack["channels"][ch_name] = {
                "hours_from_start": hours.tolist(),
                "series": curves,
            }
            plot_order = [
                ("measured", y_ch, styles[0]),
                ("daily-naive", dpred[wi, :, ch_idx], styles[1]),
                ("pooled BiLSTM", pred_b[wi, :, ch_idx], styles[2]),
                ("pooled AR-MambaKAN", pred_a[wi, :, ch_idx], styles[3]),
            ]
            for name, vals, sty in plot_order:
                ax.plot(
                    hours,
                    vals,
                    color=sty["color"],
                    linestyle=sty["linestyle"],
                    marker=sty["marker"],
                    markersize=2.0,
                    markevery=8,
                    linewidth=1.0,
                    label=name if col == 0 and row == 0 else None,
                )
            if row == 0:
                ax.set_title(ts.strftime("%Y-%m-%d"))
            if col == 0:
                ax.set_ylabel("P (kW)" if ch_name == "P" else "Q (kvar)")
            if row == 1:
                ax.set_xlabel("Hour")
            ax.grid(True, alpha=0.3, linewidth=0.4)
        so_lieu["days"][label] = day_pack

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(f"Meter {meter} (anonymous)", y=1.08, fontsize=8)

    pdf_path = out_stem.with_suffix(".pdf")
    eps_path = out_stem.with_suffix(".eps")
    info = kieu.luu_pdf(fig, pdf_path)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)

    so_path = OUT / "so-lieu-r2-du-bao-ba-ngay.json"
    so_path.write_text(json.dumps(so_lieu, indent=2))
    return {
        "pdf": str(pdf_path),
        "eps": str(eps_path),
        "so_lieu": str(so_path),
        "pdffonts": info,
        "meter": meter,
    }


def ve_hinh_wmape_tuan(r2: dict, kieu, out_stem: Path) -> dict:
    """(b) Weekly MAPE on main test, seasonal background shading."""
    weeks = [w for w in r2["weeks"] if w["readout"] == "last"]
    # mean across meters per iso_week x channel x config
    agg: dict[tuple, list] = defaultdict(list)
    season_of_week: dict[str, str] = {}
    for w in weeks:
        season_of_week[w["iso_week"]] = w["season"]
        for cfg in CFGS:
            key = (w["iso_week"], w["channel"], cfg)
            agg[key].append(w["configs"][cfg]["mape_pct"])

    iso_weeks = sorted(set(season_of_week), key=lambda s: s)
    # map iso week to an ordinal for x axis
    x = np.arange(len(iso_weeks))
    week_to_x = {w: i for i, w in enumerate(iso_weeks)}

    so_lieu: dict[str, Any] = {
        "figure": "r2_wmape_tuan",
        "readout": "last",
        "horizon": 48,
        "aggregation": "mean WMAPE across meters per ISO week",
        "iso_weeks": iso_weeks,
        "season_of_week": season_of_week,
        "series": {},
        "source": "results/r1-r2-cpu/summary.json -> r2_gchung.weeks",
    }

    kieu.ap_dung("springer", co_chu_pt=8.0)
    w = IET_COT1_MM / MM_PER_IN
    h = w * 1.0
    fig, axes = plt.subplots(2, 1, figsize=(w, h), sharex=True)

    # season background colors (light, distinguishable in grayscale by hatch)
    season_spans = []
    if iso_weeks:
        cur = season_of_week[iso_weeks[0]]
        start = 0
        for i, iw in enumerate(iso_weeks):
            seas = season_of_week[iw]
            if seas != cur:
                season_spans.append((start, i - 1, cur))
                cur = seas
                start = i
        season_spans.append((start, len(iso_weeks) - 1, cur))

    season_face = {
        "dong": "#D9D9D9",
        "xuan": "#F0F0F0",
        "he": "#E0E0E0",
        "thu": "#FAFAFA",
    }
    season_hatch = {"dong": "///", "xuan": "...", "he": "xxx", "thu": "\\\\\\"}

    plot_cfgs = [
        ("daily_naive", kieu.CHUOI_STYLE[0]),
        ("pooled_bilstm", kieu.CHUOI_STYLE[1]),
        ("pooled_armambakan", kieu.CHUOI_STYLE[2]),
        ("local_bilstm", kieu.CHUOI_STYLE[3]),
        ("local_armambakan", kieu.CHUOI_STYLE[4]),
    ]

    for ax, ch in zip(axes, ("P", "Q")):
        for start, end, seas in season_spans:
            ax.axvspan(
                start - 0.5,
                end + 0.5,
                facecolor=season_face[seas],
                hatch=season_hatch[seas],
                edgecolor="0.6",
                linewidth=0.3,
                alpha=1.0,
                zorder=0,
            )
        so_lieu["series"][ch] = {}
        for cfg, sty in plot_cfgs:
            ys = []
            for iw in iso_weeks:
                vals = agg.get((iw, ch, cfg), [])
                ys.append(mean_finite(vals))
            so_lieu["series"][ch][cfg] = {
                "wmape_pct_per_week": ys,
                "label": CFG_LABEL[cfg],
            }
            yplot = [np.nan if v is None else v for v in ys]
            ax.plot(
                x,
                yplot,
                color=sty["color"],
                linestyle=sty["linestyle"],
                marker=sty["marker"],
                markersize=2.2,
                markevery=max(1, len(x) // 12),
                linewidth=1.0,
                label=CFG_LABEL[cfg],
                zorder=2,
            )
        ax.set_ylabel(f"WMAPE {ch} (%)")
        ax.grid(True, alpha=0.3, linewidth=0.4, zorder=1)
        # season labels at midpoints
        for start, end, seas in season_spans:
            ax.text(
                (start + end) / 2,
                0.92,
                SEASON_EN[seas][:3],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                color="0.3",
            )

    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=6.5)
    axes[1].set_xlabel("ISO week index (main test)")
    # sparse xticks
    tick_idx = list(range(0, len(iso_weeks), max(1, len(iso_weeks) // 6)))
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([iso_weeks[i][2:] for i in tick_idx], rotation=45, ha="right")

    pdf_path = out_stem.with_suffix(".pdf")
    eps_path = out_stem.with_suffix(".eps")
    info = kieu.luu_pdf(fig, pdf_path)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)

    so_path = OUT / "so-lieu-r2-wmape-tuan.json"
    so_path.write_text(json.dumps(so_lieu, indent=2))
    return {
        "pdf": str(pdf_path),
        "eps": str(eps_path),
        "so_lieu": str(so_path),
        "pdffonts": info,
        "n_weeks": len(iso_weeks),
    }


def pdftoppm_png(pdf: Path) -> Path:
    out_prefix = pdf.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-r", "200", "-png", str(pdf), str(out_prefix)],
        check=True,
    )
    png = Path(str(out_prefix) + "-1.png")
    if not png.exists():
        # single-page may be without -1
        alt = Path(str(out_prefix) + ".png")
        if alt.exists():
            return alt
        raise RuntimeError(f"pdftoppm did not create PNG for {pdf}")
    return png


def print_compact(tables: dict) -> None:
    print("=== bang_hai_cach_doc (joint MAE last/full) ===")
    for r in tables["bang_hai_cach_doc"]:
        print(
            f"  {r['regime']:6} {r['model']:12} h{r['horizon']:<3} "
            f"last={r['mae_joint_last']} full={r['mae_joint_full']} "
            f"d%={r['pct_full_vs_last']} p_fvl={r['p_full_vs_last']} "
            f"p_pool={r['p_pooled_vs_local_last']}({r['pooled_wins_last']}/10) "
            f"p_ar={r['p_ar_vs_bilstm_pooled_last']}"
        )
    print("=== r2_mua (mean MAE/MAPE, last, h48) ===")
    for r in tables["r2_mua"]:
        if r["config"] not in (
            "daily_naive",
            "pooled_bilstm",
            "pooled_armambakan",
        ):
            continue
        flag = " *it_mau*" if r["it_mau"] else ""
        print(
            f"  {r['season']:4} {r['channel']} {r['config_label']:22} "
            f"MAE={r['mae']} MAPE={r['mape_pct']} "
            f"days={r['n_days_min']}-{r['n_days_max']} nM={r['n_meters']} "
            f"excl={r['mape_n_excluded_sum']}{flag}"
        )
    print("=== r2_tuan (WMAPE mean/median by season) ===")
    for r in tables["r2_tuan"]:
        if r["config"] != "pooled_bilstm":
            continue
        print(
            f"  {r['season']:4} {r['channel']} pooled BiLSTM "
            f"mean={r['wmape_mean']} med={r['wmape_median']} "
            f"nMW={r['n_meter_weeks']} nW={r['n_unique_iso_weeks']}"
        )
    print("=== r2_ngay representative ===")
    for r in tables["r2_ngay"]["representative"]:
        if r["config"] not in ("daily_naive", "pooled_bilstm", "pooled_armambakan"):
            continue
        print(
            f"  {r['label']} {r['channel']} {r['config_label']:22} "
            f"DMAPE={r['dmape_pct']} miss={r['meters_missing'] or '-'}"
        )
    print("=== r2_he_f3 ===")
    for r in tables["r2_he_f3"]:
        print(
            f"  {r['channel']} {r['architecture']:11} "
            f"pool={r['mae_pooled']}/{r['mape_pooled']} "
            f"loc={r['mae_local']}/{r['mape_local']} "
            f"wins={r['pooled_wins']}/10 p={r['p_two_sided']} "
            f"flip={r['conclusion_flips_vs_one_sided']}"
        )
    print("=== r1_1_hai_phia (last) ===")
    for r in tables["r1_1_hai_phia"]:
        if r["channel"] != "joint":
            continue
        print(
            f"  h{r['horizon']:<3} {r['config_label']:22} "
            f"MAE={r['mae_model']} naive={r['mae_daily_naive']} "
            f"wins={r['n_model_wins']}/10 p={r['p_two_sided']}"
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HINH.mkdir(parents=True, exist_ok=True)

    bang = load_json(ROOT / "results" / "bang-hai-cach-doc" / "summary.json")
    r1r2 = load_json(ROOT / "results" / "r1-r2-cpu" / "summary.json")
    p_hai = load_json(ROOT / "results" / "p-hai-phia" / "summary.json")

    tables = {
        "bang_hai_cach_doc": build_bang_hai_cach_doc(bang),
        "r2_mua": build_r2_mua(r1r2["r2_gchung"]),
        "r2_tuan": build_r2_tuan(r1r2["r2_gchung"]),
        "r2_ngay": build_r2_ngay(r1r2["r2_gchung"], r1r2["r2_g4"]),
        "r2_he_f3": build_r2_he_f3(r1r2["r2_g4"], p_hai),
        "r1_1_hai_phia": build_r1_1_hai_phia(r1r2["r1_1_gchung"], p_hai),
    }

    write_csv(OUT / "bang_hai_cach_doc.csv", tables["bang_hai_cach_doc"])
    write_csv(OUT / "r2_mua.csv", tables["r2_mua"])
    write_csv(OUT / "r2_tuan.csv", tables["r2_tuan"])
    write_csv(OUT / "r2_ngay_dai_dien.csv", tables["r2_ngay"]["representative"])
    write_csv(
        OUT / "r2_ngay_2022_vs_2023.csv", tables["r2_ngay"]["calendar_2022_vs_2023"]
    )
    write_csv(OUT / "r2_he_f3.csv", tables["r2_he_f3"])
    write_csv(OUT / "r1_1_hai_phia.csv", tables["r1_1_hai_phia"])

    kieu = _load_kieu()
    meter = choose_meter_for_fig(r1r2["r2_gchung"])
    fig_a = ve_hinh_du_bao_ba_ngay(meter, kieu, HINH / "r2-du-bao-ba-ngay")
    fig_b = ve_hinh_wmape_tuan(r1r2["r2_gchung"], kieu, HINH / "r2-wmape-tuan")

    png_a = pdftoppm_png(Path(fig_a["pdf"]))
    png_b = pdftoppm_png(Path(fig_b["pdf"]))
    fig_a["png"] = str(png_a)
    fig_b["png"] = str(png_b)
    # strip bulky stdout from pdffonts in summary
    for fig in (fig_a, fig_b):
        fonts = fig["pdffonts"]
        fig["pdffonts"] = {
            "type3": fonts.get("type3"),
            "all_embedded": fonts.get("all_embedded"),
            "rong_mm": fonts.get("rong_mm"),
            "fonts": [
                {"type": f.get("type"), "emb": f.get("emb")} for f in fonts.get("fonts", [])
            ],
        }

    summary = {
        "run": "bang-sua",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "host": platform.node(),
        "device": "cpu",
        "assumptions": [
            "PA-B: main readout last; all p two-sided Wilcoxon paired by meter",
            "Numbers from results/bang-hai-cach-doc, r1-r2-cpu, p-hai-phia only",
            "Print rounding: MAE 1 dp, MAPE 2 dp, p 3 significant figures; raw_* kept",
            "Figure meter: first MAIN_METERS with three representative days (M01)",
            f"IET single-column width {IET_COT1_MM} mm (Wiley artwork; matches Springer cot=1)",
            "MAPE floor 5% of channel mean |y| on meter test set (locked rule)",
        ],
        "sources": {
            "bang_hai_cach_doc": "results/bang-hai-cach-doc/summary.json",
            "r1_r2_cpu": "results/r1-r2-cpu/summary.json",
            "p_hai_phia": "results/p-hai-phia/summary.json",
            "g_chung_preds_for_fig_a": "results/g-chung/*_pooled_last_h48_s{42,43,44}",
        },
        "bang_hai_cach_doc": tables["bang_hai_cach_doc"],
        "r2_mua": tables["r2_mua"],
        "r2_tuan": tables["r2_tuan"],
        "r2_ngay": tables["r2_ngay"],
        "r2_he_f3": tables["r2_he_f3"],
        "r1_1_hai_phia": tables["r1_1_hai_phia"],
        "hinh": {"r2_du_bao_ba_ngay": fig_a, "r2_wmape_tuan": fig_b},
        "csv_outputs": [
            "bang_hai_cach_doc.csv",
            "r2_mua.csv",
            "r2_tuan.csv",
            "r2_ngay_dai_dien.csv",
            "r2_ngay_2022_vs_2023.csv",
            "r2_he_f3.csv",
            "r1_1_hai_phia.csv",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # env
    env_lines = [
        f"python={sys.version.split()[0]}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"host={platform.node()}",
        f"platform={platform.platform()}",
    ]
    try:
        import matplotlib

        env_lines.append(f"matplotlib={matplotlib.__version__}")
    except Exception:
        pass
    (OUT / "env.txt").write_text("\n".join(env_lines) + "\n")

    print_compact(tables)
    print("=== hinh ===")
    print(f"  meter_fig_a={meter}")
    print(f"  A pdf={fig_a['pdf']} png={fig_a['png']} fonts={fig_a['pdffonts']}")
    print(f"  B pdf={fig_b['pdf']} png={fig_b['png']} fonts={fig_b['pdffonts']}")
    print(f"Wrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
