"""Regenerates the paper's headline pooled-vs-local numbers (Table 1) and the nine-condition
Friedman ranking (Table 8) directly from the CSVs in results/, using the manual Wilcoxon/Friedman
implementations in statistical_tests.py. This is the same computation used to produce the numbers
reported in the manuscript; run it to verify the results/ CSVs are self-consistent with the paper.

Usage (from this directory):
    python reproduce_tables.py --results-dir ../results
"""
import argparse
import os

import pandas as pd

from statistical_tests import wilcoxon_signed_rank, friedman_test


def reproduce_table1(results_dir: str):
    local = pd.read_csv(os.path.join(results_dir, "local_vs_pooled_armambakan_local.csv"))
    pooled = pd.read_csv(os.path.join(results_dir, "local_vs_pooled_armambakan_pooled.csv"))

    local_agg = local.groupby(["horizon", "code"])["mae"].mean()
    pooled_agg = pooled.groupby(["horizon", "code"])["pooled_mae"].mean()
    idx = sorted(local_agg.index)
    l = local_agg.loc[idx].values
    p = pooled_agg.loc[idx].values

    delta_pct = 100 * (p.mean() - l.mean()) / l.mean()
    p_value, n = wilcoxon_signed_rank(p, l)
    n_improved = int((p < l).sum())
    print("=== Table 1 (AR-MambaKAN row): pooled vs. local ===")
    print(f"Local MAE:  {l.mean():.1f}")
    print(f"Pooled MAE: {p.mean():.1f}")
    print(f"Delta:      {delta_pct:+.1f}%")
    print(f"Tasks improved: {n_improved}/{n}")
    print(f"Wilcoxon p: {p_value:.3g}")


def reproduce_friedman9(results_dir: str):
    path = os.path.join(results_dir, "friedman9_stemfix_data.csv")
    if not os.path.exists(path):
        print(f"\n(skipping Table 8 -- {path} not found)")
        return
    df = pd.read_csv(path).drop(columns=["code", "horizon"], errors="ignore")
    chi2, dof, p, mean_ranks = friedman_test(df)
    print("\n=== Table 8: nine-condition Friedman ranking ===")
    print(f"chi2={chi2:.2f}  df={dof}  p={p:.3g}")
    print(mean_ranks.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    reproduce_table1(args.results_dir)
    reproduce_friedman9(args.results_dir)


if __name__ == "__main__":
    main()
