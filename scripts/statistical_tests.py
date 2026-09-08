"""Non-parametric significance tests used throughout the paper: a manual (no-SciPy) Wilcoxon
signed-rank test and Friedman omnibus test, implemented via `pandas.Series.rank()` plus the
closed-form normal / chi-square approximations. These are the exact implementations used to
produce the paper's Friedman/Wilcoxon numbers (Tables 1, 2, 8, 15, etc.) -- reproduced here as a
standalone, reusable module/CLI so the tests can be independently re-run on any wide-format CSV
(one row per task/block, one column per condition).

CLI usage (Friedman + all pairwise Wilcoxon vs. a reference column) on a wide CSV, e.g.
results/friedman9_stemfix_data.csv (one row per meter*horizon task, one column per
model x training-regime condition):

    python statistical_tests.py --input-csv ../results/friedman9_stemfix_data.csv \
        --id-cols code horizon --reference "Pooled AR-MambaKAN"

As a library:
    from statistical_tests import wilcoxon_signed_rank, friedman_test
"""
import argparse
from math import erf, sqrt

import numpy as np
import pandas as pd


def wilcoxon_signed_rank(x, y):
    """Two-sided paired Wilcoxon signed-rank test via the normal approximation.
    Returns (p_value, n_nonzero_pairs)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    ranks = pd.Series(np.abs(d)).rank(method="average").values
    w = min(ranks[d > 0].sum(), ranks[d < 0].sum())
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sigma if sigma > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return p, n


def _chi2_sf(x, k):
    """Upper tail of the chi-square distribution via the regularized incomplete gamma function
    (continued-fraction / series expansion; no SciPy dependency)."""
    import math
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1.0:
        term = 1.0 / a
        summ = term
        m = a
        for _ in range(200):
            m += 1
            term *= xx / m
            summ += term
            if abs(term) < abs(summ) * 1e-12:
                break
        return 1 - summ * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    else:
        tiny = 1e-300
        b = xx + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


def friedman_test(df: pd.DataFrame):
    """Friedman omnibus test over a wide DataFrame (rows = blocks/tasks, columns = conditions).
    Returns (chi2, dof, p_value, mean_ranks: pd.Series sorted ascending)."""
    n, k = df.shape
    ranks = df.rank(axis=1)
    mean_ranks = ranks.mean().sort_values()
    R = ranks.sum().values
    chi2 = (12 / (n * k * (k + 1))) * np.sum(R ** 2) - 3 * n * (k + 1)
    p = _chi2_sf(chi2, k - 1)
    return chi2, k - 1, p, mean_ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", required=True, help="Wide CSV: one row per task, one column per condition (plus optional id columns).")
    ap.add_argument("--id-cols", nargs="*", default=[], help="Non-condition columns to drop before running the tests (e.g. code, horizon).")
    ap.add_argument("--reference", default=None, help="Condition column to run pairwise Wilcoxon tests against every other column.")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    conditions = df.drop(columns=args.id_cols, errors="ignore")

    chi2, dof, p, mean_ranks = friedman_test(conditions)
    print(f"Friedman: chi2={chi2:.3f}  df={dof}  p={p:.4g}  n_tasks={len(conditions)}")
    print("Mean ranks (1 = best):")
    print(mean_ranks.round(3).to_string())

    if args.reference:
        print(f"\nPairwise Wilcoxon vs. '{args.reference}':")
        ref = conditions[args.reference]
        for col in conditions.columns:
            if col == args.reference:
                continue
            p_pair, n = wilcoxon_signed_rank(ref, conditions[col])
            delta = (ref - conditions[col]).mean()
            print(f"  vs {col:30s} mean_delta={delta:+9.2f}  p={p_pair:.4g}  n={n}")


if __name__ == "__main__":
    main()
