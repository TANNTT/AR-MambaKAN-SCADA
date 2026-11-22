"""PA-B: liet ke kiem dinh ban nop (mot/hai phia) va tinh lai p hai phia.

Chi doc ma goc AR-MambaKAN-SCADA, CSV results/, va summary da co.
Ghi results/p-hai-phia/summary.json. CPU. Khong GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import scipy.stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from _paths import require_env
SRC = str(require_env("SCADA_PUBLIC_REPO"))
SRC_ALT = str(require_env("SCADA_CODE_ROOT"))
METERS = [f"M{i:02d}" for i in range(1, 11)]
ALPHA = 0.05


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


def wilcoxon_both(diffs: list[float]) -> dict:
    """diffs = a - b; positive means a worse if metric is MAE and we compare a vs better-b.
    Tra ve p one-sided greater (diffs>0), less (diffs<0), two-sided.
    """
    nonzero = [x for x in diffs if x != 0]
    if len(nonzero) < 1:
        return {
            "n_nonzero": 0,
            "p_greater": None,
            "p_less": None,
            "p_two_sided": None,
            "note": "all differences zero",
        }
    try:
        pg = float(
            scipy.stats.wilcoxon(
                nonzero, alternative="greater", method="auto"
            ).pvalue
        )
        pl = float(
            scipy.stats.wilcoxon(nonzero, alternative="less", method="auto").pvalue
        )
        pt = float(
            scipy.stats.wilcoxon(
                nonzero, alternative="two-sided", method="auto"
            ).pvalue
        )
        return {
            "n_nonzero": len(nonzero),
            "p_greater": pg,
            "p_less": pl,
            "p_two_sided": pt,
            "n_pos": sum(1 for x in nonzero if x > 0),
            "n_neg": sum(1 for x in nonzero if x < 0),
        }
    except ValueError as e:
        return {
            "n_nonzero": len(nonzero),
            "p_greater": None,
            "p_less": None,
            "p_two_sided": None,
            "note": str(e),
        }


def sig(p: float | None) -> bool | None:
    if p is None:
        return None
    return bool(p < ALPHA)


def flip_row(
    *,
    cell_id: str,
    p_one: float | None,
    p_two: float | None,
    one_sided_label: str,
) -> dict:
    s1 = sig(p_one)
    s2 = sig(p_two)
    return {
        "cell_id": cell_id,
        "one_sided_label": one_sided_label,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "sig_one_0.05": s1,
        "sig_two_0.05": s2,
        "conclusion_flips_at_0.05": bool(s1 is not None and s2 is not None and s1 != s2),
    }


def catalog_ban_nop_tests() -> list[dict]:
    """Liet ke kiem dinh Table 3-14, 17-20 theo ma goc (chi doc)."""
    # Hai implement Wilcoxon trong ma goc deu hai phia.
    two_sided_src = [
        {
            "file": f"{SRC}/scripts/statistical_tests.py",
            "lines": "25-39",
            "fn": "wilcoxon_signed_rank",
            "evidence": "p = 2*(1-Phi(|z|)) normal approx, two-sided",
        },
        {
            "file": f"{SRC}/src/evaluation/statistical_tests.py",
            "lines": "69-84",
            "fn": "_wilcoxon",
            "evidence": 'scipy.stats.wilcoxon(..., alternative="two-sided")',
        },
        {
            "file": f"{SRC_ALT}/src/scada_exp/statistical_tests.py",
            "lines": "69-84",
            "fn": "_wilcoxon",
            "evidence": 'scipy.stats.wilcoxon(..., alternative="two-sided")',
        },
    ]
    tables = [
        {
            "table": 3,
            "test": "Wilcoxon paired pooled vs local, 40 meter-horizon tasks, per architecture",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/local_vs_pooled_armambakan_{{local,pooled}}.csv (va tuong duong MLP/LSTM/BiLSTM trong artifacts)",
            "bonferroni": False,
            "recompute": "co the tu CSV task-level",
        },
        {
            "table": 4,
            "test": "Wilcoxon paired pooled vs local per horizon, n=10 meters",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC_ALT}/artifacts/pooled_multihorizon_summary.csv (cot wilcoxon_p)",
            "bonferroni": False,
            "recompute": "co CSV horizon x model",
        },
        {
            "table": 5,
            "test": "Wilcoxon paired P and Q separately, 40 tasks; Holm across 8 tests (text)",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/pq_split_summary.csv",
            "bonferroni": False,
            "holm": True,
            "recompute": "co CSV tong hop; chi tiet task trong pq_split_{local,pooled}.csv",
        },
        {
            "table": 6,
            "test": "Wilcoxon paired local vs pooled-balanced vs full-pooled, 40 tasks",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/matched_budget_armambakan.csv (+ artifacts data_budget_*)",
            "bonferroni": False,
            "recompute": "co CSV",
        },
        {
            "table": 7,
            "test": "Wilcoxon paired LOSO, n=10 meters (seed-mean), H=48",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/loso_armambakan.csv",
            "bonferroni": False,
            "recompute": "co CSV",
        },
        {
            "table": 8,
            "test": "Meter-level Wilcoxon n=10 + cluster bootstrap (not a sided Wilcoxon alternative)",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/hierarchical_test_summary.csv",
            "bonferroni": False,
            "recompute": "co CSV tong; bootstrap khong doi sided",
        },
        {
            "table": 9,
            "test": "Wilcoxon quality strata, 20 tasks per stratum",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/quality_strata_summary.csv",
            "bonferroni": False,
            "recompute": "co CSV tong",
        },
        {
            "table": 10,
            "test": "Friedman omnibus 9 conditions (khong co mot/hai phia)",
            "sided": "n/a-friedman",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "81-90",
            "csv": f"{SRC}/results/friedman9_stemfix_data.csv",
            "bonferroni": False,
            "recompute": "Friedman khong chuyen sided",
        },
        {
            "table": 11,
            "test": "Descriptive MAE/sMAPE table (khong p trong bang; Friedman trong van ban)",
            "sided": "n/a-descriptive",
            "source_file": None,
            "source_lines": None,
            "csv": None,
            "bonferroni": False,
            "recompute": "khong co kiem dinh trong bang",
        },
        {
            "table": 12,
            "test": "Pairwise Wilcoxon AR-MambaKAN vs baselines, Bonferroni",
            "sided": "two-sided",
            "source_file": f"{SRC}/src/evaluation/statistical_tests.py",
            "source_lines": "69-84,146-156",
            "csv": f"{SRC_ALT}/artifacts/submission_bundle/tables/stat_tests/stat_tests_multihorizon/wilcoxon_pairwise.csv",
            "bonferroni": True,
            "recompute": "co CSV pairwise + p_value_bonferroni",
        },
        {
            "table": 13,
            "test": "Descriptive modern baselines (khong p trong bang)",
            "sided": "n/a-descriptive",
            "source_file": None,
            "source_lines": None,
            "csv": None,
            "bonferroni": False,
            "recompute": "khong co kiem dinh trong bang",
        },
        {
            "table": 14,
            "test": "Pairwise Wilcoxon 6 models, Bonferroni over 15 pairs",
            "sided": "two-sided",
            "source_file": f"{SRC}/src/evaluation/statistical_tests.py",
            "source_lines": "69-84,146-156",
            "csv": f"{SRC_ALT}/artifacts/stat_tests_six/wilcoxon_pairwise.csv (neu co) / submission_bundle",
            "bonferroni": True,
            "recompute": "co CSV pairwise",
        },
        {
            "table": 17,
            "test": "Wilcoxon model vs daily-naive, n=10 meters per cell",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": "Khong co CSV task rieng trong results/; so p khop two-sided (vd BiLSTM h4 p=0.005 = 10/10 wins)",
            "bonferroni": False,
            "recompute": "tinh lai tu results/r1-r2-cpu per_meter (G-chung)",
            "note": (
                "Ban nop p=0.059 cho pooled AR h4 (n.s.). "
                "Phan tich R1.1 CPU (phan_tich_r1_r2_cpu.py) dung alternative=greater (mot phia)."
            ),
        },
        {
            "table": 18,
            "test": "Wilcoxon ablation vs A4, n=10; Holm on A5/A6; Friedman 4 variants",
            "sided": "two-sided",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/component_ablation_summary.csv",
            "bonferroni": False,
            "holm": True,
            "recompute": "co CSV tong",
        },
        {
            "table": 19,
            "test": "Calendar ablation descriptive deltas (van ban/bang: doi chieu MAE; Wilcoxon neu co trong CSV)",
            "sided": "two-sided-if-tested",
            "source_file": f"{SRC}/scripts/statistical_tests.py",
            "source_lines": "25-39",
            "csv": f"{SRC}/results/calendar_ablation_armambakan_nocal.csv",
            "bonferroni": False,
            "recompute": "CSV co thi tinh; bang chu yeu descriptive",
        },
        {
            "table": 20,
            "test": "Weather vs calendar descriptive (8 stations, 1 seed)",
            "sided": "n/a-descriptive",
            "source_file": None,
            "source_lines": None,
            "csv": f"{SRC}/results/weather_ablation_armambakan.csv",
            "bonferroni": False,
            "recompute": "n=8, 1 seed: khong Wilcoxon ban nop",
        },
    ]
    return {
        "two_sided_implementations": two_sided_src,
        "tables": tables,
        "one_sided_in_ban_nop_tables": [],
        "note": (
            "Khong tim thay alternative=greater/less trong ma goc sinh bang ban nop. "
            "Moi Wilcoxon/Friedman dung cho Table 3-14,17-18 la hai phia (hoac n/a). "
            "Kiem dinh MOT PHIA chi xuat hien o phan tich CPU R1.1/R2 "
            "(phan_tich_r1_r2_cpu.py:261-280, alternative=greater)."
        ),
    }


def recompute_from_csv_two_sided() -> dict:
    """Tinh lai p hai phia cho bang ban nop co CSV task/meter."""
    import pandas as pd

    out = {"tables": {}, "flips": []}

    # Table 4: pooled_multihorizon_summary da la two-sided; xac nhan bang tinh lai
    path = os.path.join(SRC_ALT, "artifacts/pooled_multihorizon_summary.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "model": r["model"],
                    "horizon": int(r["horizon"]),
                    "ban_nop_or_csv_p": float(r["wilcoxon_p"]),
                    "sided_in_source": "two-sided",
                    "recompute_needed": False,
                    "reason": "CSV wilcoxon_p da tu wilcoxon_signed_rank two-sided",
                }
            )
        out["tables"]["table4"] = rows
    else:
        out["tables"]["table4"] = {"error": f"missing {path}"}

    # Table 3 / 8 hierarchical: already two-sided
    hpath = os.path.join(SRC, "results/hierarchical_test_summary.csv")
    if os.path.exists(hpath):
        hdf = pd.read_csv(hpath)
        out["tables"]["table3_table8"] = [
            {
                "model": r["model"],
                "p_40task": float(r["p_40task"]),
                "p_meter_level_n10": float(r["p_meter_level_n10"]),
                "sided_in_source": "two-sided",
                "recompute_needed": False,
            }
            for _, r in hdf.iterrows()
        ]

    # Table 5
    pq = os.path.join(SRC, "results/pq_split_summary.csv")
    if os.path.exists(pq):
        pdf = pd.read_csv(pq)
        out["tables"]["table5"] = [
            {
                "model": r["model"],
                "p_mae_p": float(r["p_mae_p"]),
                "p_mae_q": float(r["p_mae_q"]),
                "sided_in_source": "two-sided",
                "recompute_needed": False,
            }
            for _, r in pdf.iterrows()
        ]

    # Table 12 pairwise: double raw p for one->two only if were one-sided; here already two
    wpath = os.path.join(
        SRC_ALT,
        "artifacts/submission_bundle/tables/stat_tests/stat_tests_multihorizon/wilcoxon_pairwise.csv",
    )
    if os.path.exists(wpath):
        wdf = pd.read_csv(wpath)
        out["tables"]["table12"] = {
            "n_pairs": len(wdf),
            "sided_in_source": "two-sided",
            "bonferroni": True,
            "recompute_needed": False,
            "sample": wdf.head(5).to_dict(orient="records"),
        }
    else:
        out["tables"]["table12"] = {"error": f"missing {wpath}"}

    return out


def recompute_rebuttal() -> dict:
    """p hai phia cho so dung trong thu tra loi: R1.1, G3, G4, R2."""
    flips = []
    out = {}

    # --- R1.1 gchung: mot phia (greater: naive-model > 0) -> tinh lai hai phia
    r12 = json.load(open(os.path.join(ROOT, "results/r1-r2-cpu/summary.json")))
    r11_cells = []
    for cell in r12["r1_1_gchung"]["cells"]:
        if cell["channel"] != "joint":
            continue
        base = cell["daily_naive_per_meter"]
        for mname, md in cell["models"].items():
            per = md["per_meter"]
            # diff = naive - model: >0 model better (same as wilcoxon_wins)
            diffs = [base[m] - per[m] for m in METERS if m in base and m in per]
            both = wilcoxon_both(diffs)
            p_one = md["vs_daily_naive"].get("wilcoxon_pvalue")
            p_two = both["p_two_sided"]
            # one-sided in source was alternative=greater on (naive-model)
            cell_id = (
                f"R1.1|gchung|{cell['readout']}|h{cell['horizon']}|joint|{mname}"
            )
            fr = flip_row(
                cell_id=cell_id,
                p_one=p_one,
                p_two=p_two,
                one_sided_label="greater(naive-model): model better than daily-naive",
            )
            fr.update(
                {
                    "mean_model": md["mean"],
                    "mean_naive": cell["daily_naive_mean"],
                    "n_model_wins": md["vs_daily_naive"].get("n_model_wins"),
                    "p_greater": both.get("p_greater"),
                    "p_less": both.get("p_less"),
                }
            )
            r11_cells.append(fr)
            if fr["conclusion_flips_at_0.05"]:
                flips.append(fr)
    out["r1_1_gchung_joint"] = r11_cells

    # R1.1 g4
    r11_g4 = []
    for cell in r12["r1_1_g4"]["cells"]:
        if cell["channel"] != "joint":
            continue
        base = cell["daily_naive_per_meter"]
        for mname, md in cell["models"].items():
            per = md["per_meter"]
            diffs = [base[m] - per[m] for m in METERS if m in base and m in per]
            both = wilcoxon_both(diffs)
            p_one = md["vs_daily_naive"].get("wilcoxon_pvalue")
            cell_id = (
                f"R1.1|g4|{cell['fold']}|h{cell['horizon']}|joint|{mname}"
            )
            fr = flip_row(
                cell_id=cell_id,
                p_one=p_one,
                p_two=both["p_two_sided"],
                one_sided_label="greater(naive-model)",
            )
            fr.update(
                {
                    "mean_model": md["mean"],
                    "mean_naive": cell["daily_naive_mean"],
                    "n_model_wins": md["vs_daily_naive"].get("n_model_wins"),
                }
            )
            r11_g4.append(fr)
            if fr["conclusion_flips_at_0.05"]:
                flips.append(fr)
    out["r1_1_g4_joint"] = r11_g4

    # --- G3: da hai phia trong so_g3_voi_moc; van ghi p_two va doi chieu
    g3 = json.load(open(os.path.join(ROOT, "results/g3-so-moc/summary.json")))
    g3_out = {}
    for key, label_one in (
        ("p0.5_vs_p0.75", "two-sided already (report both)"),
        ("p0.25_vs_p0.5", "two-sided already (report both)"),
    ):
        w = g3["wilcoxon"][key]
        diffs = w["diffs"]
        both = wilcoxon_both(diffs)
        # stored p is two-sided; also compute one-sided for flip check
        # directional: for p0.5 vs p0.75, "p0.75 worse" <=> diff = p0.5 - p0.75 < 0 when values are mae at those fracs
        # so_g3 stores diffs as listed; use p_less/p_greater for completeness
        p_two_src = float(w["pvalue"])
        # Choose the one-sided that matches the claim direction used in nhat ky
        if key == "p0.5_vs_p0.75":
            # claim: p0.75 worse than p0.5 => mae_0.75 > mae_0.5 => diff (0.5-0.75) < 0 => alternative less
            p_one = both["p_less"]
            one_label = "less(mae_p0.5 - mae_p0.75): p0.75 worse"
        else:
            # claim exploratory either way; use greater on | | no: report less as "p0.25 better"
            p_one = both["p_less"]
            one_label = "less(mae_p0.25 - mae_p0.5)"
        fr = flip_row(
            cell_id=f"G3|{key}",
            p_one=p_one,
            p_two=both["p_two_sided"],
            one_sided_label=one_label,
        )
        fr["p_two_sided_source_file"] = p_two_src
        fr["n_positive"] = w.get("n_positive")
        fr["n_negative"] = w.get("n_negative")
        g3_out[key] = fr
        # Only flag flip if someone had used one-sided as primary; still record
        if fr["conclusion_flips_at_0.05"]:
            flips.append({**fr, "note": "G3 source was already two-sided; flip is one-vs-two hypothetically"})
    out["g3"] = g3_out

    # --- G4: tong_g4 da two-sided; tinh them one-sided de danh dau flip gia dinh
    g4 = json.load(open(os.path.join(ROOT, "results/g4-tong/summary.json")))
    g4_rows = []
    for key, cmp in g4["pool_vs_local"].items():
        if key.endswith("_noM08") or not key.endswith("|joint"):
            continue
        # per_meter: pooled - local
        pm = cmp.get("per_meter") or {}
        diffs = [pm[m]["diff"] for m in METERS if m in pm]
        both = wilcoxon_both(diffs)
        # claim pooled better: pooled - local < 0 => alternative less
        p_one = both["p_less"]
        p_two = cmp.get("wilcoxon_p")
        fr = flip_row(
            cell_id=f"G4|pool_vs_local|{key}",
            p_one=p_one,
            p_two=p_two if p_two is not None else both["p_two_sided"],
            one_sided_label="less(pooled-local): pooled better",
        )
        fr["pooled_wins"] = cmp.get("pooled_wins")
        fr["mean_pooled"] = cmp.get("mean_pooled")
        fr["mean_local"] = cmp.get("mean_local")
        g4_rows.append(fr)
        if fr["conclusion_flips_at_0.05"]:
            flips.append({**fr, "note": "G4 source already two-sided; hypothetical flip one->two"})
    for key, cmp in g4["ar_vs_bilstm"].items():
        if key.endswith("_noM08") or not key.endswith("|joint"):
            continue
        pm = cmp.get("per_meter") or {}
        diffs = [pm[m]["diff"] for m in METERS if m in pm]
        both = wilcoxon_both(diffs)
        p_one = both["p_less"]  # AR better when AR-BiLSTM < 0
        p_two = cmp.get("wilcoxon_p")
        fr = flip_row(
            cell_id=f"G4|ar_vs_bilstm|{key}",
            p_one=p_one,
            p_two=p_two if p_two is not None else both["p_two_sided"],
            one_sided_label="less(AR-BiLSTM): AR better",
        )
        fr["armambakan_wins"] = cmp.get("armambakan_wins")
        g4_rows.append(fr)
        if fr["conclusion_flips_at_0.05"]:
            flips.append({**fr, "note": "G4 source already two-sided; hypothetical flip"})
    out["g4"] = g4_rows

    # --- R2 F3 summer: mot phia via wilcoxon_wins
    summer_rows = []
    # rebuild per-meter from f3_summer
    by = {}
    for row in r12["r2_g4"]["f3_summer"]:
        by.setdefault(row["channel"], {})[row["meter"]] = row["configs"]
    for ch, meters in by.items():
        for pooled_k, local_k, label in (
            ("pooled_bilstm", "local_bilstm", "bilstm"),
            ("pooled_armambakan", "local_armambakan", "armambakan"),
        ):
            diffs = []
            wins = 0
            for m in METERS:
                if m not in meters:
                    continue
                pv = meters[m][pooled_k]["mae"]
                lv = meters[m][local_k]["mae"]
                diffs.append(lv - pv)  # >0 pooled better (same as wilcoxon_wins model=pooled base=local)
                if pv < lv:
                    wins += 1
            both = wilcoxon_both(diffs)
            # find stored one-sided
            stored = None
            for w in r12["r2_g4"]["wilcoxon_f3_summer_pooled_vs_local"]:
                if w["channel"] == ch and w["pooled"] == pooled_k:
                    stored = w
                    break
            p_one = stored["wilcoxon_pvalue"] if stored else both["p_greater"]
            fr = flip_row(
                cell_id=f"R2|F3summer|{ch}|{label}",
                p_one=p_one,
                p_two=both["p_two_sided"],
                one_sided_label="greater(local-pooled): pooled better",
            )
            fr["n_pooled_wins"] = wins
            fr["p_greater"] = both.get("p_greater")
            summer_rows.append(fr)
            if fr["conclusion_flips_at_0.05"]:
                flips.append(fr)
    out["r2_f3_summer"] = summer_rows

    out["conclusion_flips_at_0.05"] = [
        f for f in flips if f.get("conclusion_flips_at_0.05")
    ]
    out["n_flips"] = len(out["conclusion_flips_at_0.05"])
    return out


def main() -> None:
    catalog = catalog_ban_nop_tests()
    csv_re = recompute_from_csv_two_sided()
    rebuttal = recompute_rebuttal()

    # Tom tat o doi ket luan: chi nhung o THAT SU chuyen tu mot phia (R1.1, R2)
    real_flips = [
        f
        for f in rebuttal["conclusion_flips_at_0.05"]
        if f["cell_id"].startswith("R1.1") or f["cell_id"].startswith("R2")
    ]
    hypo_flips = [
        f
        for f in rebuttal["conclusion_flips_at_0.05"]
        if f["cell_id"].startswith("G3") or f["cell_id"].startswith("G4")
    ]

    out = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "alpha": ALPHA,
        "assumptions": [
            "Ban nop Table 3-14,17-18: Wilcoxon/Friedman trong ma goc la hai phia.",
            "R1.1 va R2 summer trong results/r1-r2-cpu dung alternative=greater (mot phia).",
            "G3 (so_g3_voi_moc) va G4 (tong_g4) da hai phia; van tinh one-sided de doi chieu gia dinh.",
            "Ket luan doi o alpha=0.05 khi sig(one) != sig(two).",
        ],
        "ban_nop_catalog": catalog,
        "ban_nop_csv_recompute": csv_re,
        "rebuttal_two_sided": rebuttal,
        "flips_real_one_to_two": real_flips,
        "flips_hypothetical_g3_g4": hypo_flips,
        "n_flips_real": len(real_flips),
        "n_flips_hypothetical": len(hypo_flips),
        "one_sided_ban_nop_tables": catalog["one_sided_in_ban_nop_tables"],
        "note": (
            "Khong co bang ban nop nao dung Wilcoxon mot phia trong ma goc. "
            "O doi ket luan that su chi den tu R1.1/R2 (CPU mot phia -> hai phia)."
        ),
    }

    out_dir = os.path.join(ROOT, "results/p-hai-phia")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    json.dump(out, open(path, "w"), indent=2)

    try:
        env = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        env = "pip freeze unavailable\n"
    open(os.path.join(out_dir, "env.txt"), "w").write(env)

    print("p_hai_phia ->", path)
    print(
        "ban_nop one-sided tables:",
        catalog["one_sided_in_ban_nop_tables"] or "(none)",
    )
    print("n_flips_real (R1.1/R2):", len(real_flips))
    for f in real_flips:
        print(
            " FLIP",
            f["cell_id"],
            "p1=%.4g p2=%.4g" % (f["p_one_sided"], f["p_two_sided"]),
        )
    print("n_flips_hypothetical (G3/G4):", len(hypo_flips))
    for f in hypo_flips[:20]:
        print(
            " HYPO",
            f["cell_id"],
            "p1=%.4g p2=%.4g" % (f["p_one_sided"], f["p_two_sided"]),
        )


if __name__ == "__main__":
    main()
