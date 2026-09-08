import argparse
import math
from itertools import combinations
from pathlib import Path

import pandas as pd

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


def _task_id(df: pd.DataFrame) -> pd.Series:
    """Build a task identifier. Includes horizon when present so multi-horizon
    runs are treated as distinct operating conditions (more statistical power)."""
    tid = df["station"].astype(str) + "::" + df["code"].astype(str)
    if "horizon" in df.columns:
        tid = tid + "::h" + df["horizon"].astype(str)
    return tid


def _rank_by_task(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    x = df.copy()
    x["task_id"] = _task_id(x)
    x["rank"] = x.groupby("task_id")[metric_col].rank(method="average", ascending=True)
    return x


def _chi2_sf(chi2: float, dof: int) -> float:
    """Upper tail of chi-square; uses scipy if available, else a series fallback."""
    if _scipy_stats is not None:
        return float(_scipy_stats.chi2.sf(chi2, dof))
    # Regularized upper incomplete gamma via simple series (dof even or odd small).
    # Fallback only; scipy is expected in this project.
    k = dof / 2.0
    x = chi2 / 2.0
    if x <= 0:
        return 1.0
    term = 1.0
    summation = 0.0
    for n in range(0, 200):
        if n == 0:
            term = 1.0 / math.gamma(k + 1.0)
        else:
            term *= x / (k + n)
        summation += term
    lower_reg = summation * math.exp(-x + k * math.log(x) - math.lgamma(k + 1.0) + math.lgamma(k + 1.0))
    lower_reg = min(max(lower_reg, 0.0), 1.0)
    return 1.0 - lower_reg


def _friedman(wide: pd.DataFrame) -> tuple[float, float, int]:
    """Friedman test on a tasks x models matrix (no missing)."""
    mat = wide.dropna(axis=0, how="any")
    n_tasks, k = mat.shape
    if _scipy_stats is not None and n_tasks >= 2 and k >= 3:
        cols = [mat[c].to_numpy() for c in mat.columns]
        chi2, p = _scipy_stats.friedmanchisquare(*cols)
        return float(chi2), float(p), int(n_tasks)
    # Manual chi-square from average ranks.
    ranks = mat.rank(axis=1, method="average", ascending=True)
    rj = ranks.mean(axis=0)
    sum_r2 = float((rj**2).sum())
    chi2 = (12.0 * n_tasks) / (k * (k + 1.0)) * (sum_r2 - (k * (k + 1.0) ** 2) / 4.0)
    return float(chi2), _chi2_sf(chi2, k - 1), int(n_tasks)


def _wilcoxon(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    sub = pd.concat([a, b], axis=1).dropna()
    diff = (sub.iloc[:, 0] - sub.iloc[:, 1]).to_numpy()
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0, 1.0
    if _scipy_stats is not None:
        try:
            res = _scipy_stats.wilcoxon(
                sub.iloc[:, 0].to_numpy(),
                sub.iloc[:, 1].to_numpy(),
                zero_method="wilcox",
                alternative="two-sided",
                mode="auto",
            )
            return float(res.statistic), float(res.pvalue)
        except Exception:
            pass
    # Normal-approximation fallback.
    import numpy as np

    order = np.argsort(np.abs(nz))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(nz) + 1)
    signs = np.sign(nz)
    w_plus = ranks[signs > 0].sum()
    w_minus = ranks[signs < 0].sum()
    w = min(w_plus, w_minus)
    n = len(nz)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma <= 0:
        return float(w), 1.0
    z = (w - mu) / sigma
    p = 2.0 * 0.5 * (1.0 - math.erf(abs(z) / math.sqrt(2.0)))
    return float(w), float(p)


def run_tests(input_csv: Path, metric_col: str, out_dir: Path, label: str = "") -> dict:
    df = pd.read_csv(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"station", "code", "model", metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ranked = _rank_by_task(df, metric_col)
    models = sorted(ranked["model"].unique().tolist())

    df = df.copy()
    df["task_id"] = _task_id(df)
    wide = df.pivot_table(index="task_id", columns="model", values=metric_col, aggfunc="mean")
    wide = wide[[m for m in models if m in wide.columns]]

    chi2, p_friedman, n_tasks = _friedman(wide)
    backend = "scipy" if _scipy_stats is not None else "fallback"
    friedman_df = pd.DataFrame(
        [
            {
                "label": label or input_csv.stem,
                "n_tasks": n_tasks,
                "n_models": len(models),
                "metric": metric_col,
                "friedman_chi2": chi2,
                "friedman_pvalue": p_friedman,
                "backend": backend,
            }
        ]
    )
    friedman_df.to_csv(out_dir / "friedman_test.csv", index=False)

    pairs = []
    n_pairs = len(models) * (len(models) - 1) // 2
    for a, b in combinations(models, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        stat, p = _wilcoxon(wide[a], wide[b])
        sub = wide[[a, b]].dropna()
        diff = sub[a] - sub[b]
        pairs.append(
            {
                "model_a": a,
                "model_b": b,
                "n_tasks": int(len(sub)),
                "wilcoxon_stat": stat,
                "p_value": p,
                "p_value_bonferroni": min(1.0, p * n_pairs),
                "median_diff_a_minus_b": float(diff.median()) if len(diff) else float("nan"),
            }
        )
    pair_df = pd.DataFrame(pairs).sort_values("p_value_bonferroni")
    pair_df.to_csv(out_dir / "wilcoxon_pairwise.csv", index=False)

    avg_rank_df = wide.rank(axis=1, method="average", ascending=True).mean(axis=0).reset_index()
    avg_rank_df.columns = ["model", "avg_rank"]
    avg_rank_df = avg_rank_df.sort_values("avg_rank")
    avg_rank_df.to_csv(out_dir / "average_ranks.csv", index=False)

    (out_dir / "average_ranks.tex").write_text(avg_rank_df.to_latex(index=False, escape=False), encoding="utf-8")
    (out_dir / "wilcoxon_pairwise.tex").write_text(pair_df.to_latex(index=False, escape=False), encoding="utf-8")

    print(
        f"[{label or input_csv.stem}] Friedman chi2={chi2:.3f} p={p_friedman:.3g} "
        f"(n_tasks={n_tasks}, k={len(models)}, backend={backend}) -> {out_dir}"
    )
    return {"chi2": chi2, "p": p_friedman, "n_tasks": n_tasks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--metric-col", default="mae")
    ap.add_argument("--out-dir", default="artifacts/stat_tests")
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--group-col",
        default="",
        help="If set (e.g. 'horizon'), also run tests separately per group value.",
    )
    args = ap.parse_args()

    run_tests(Path(args.input_csv), args.metric_col, Path(args.out_dir), args.label)

    if args.group_col:
        df = pd.read_csv(args.input_csv)
        if args.group_col not in df.columns:
            raise ValueError(f"--group-col {args.group_col} not in columns")
        for val, sub in df.groupby(args.group_col):
            sub_dir = Path(args.out_dir) / f"{args.group_col}_{val}"
            tmp = sub_dir / "_input.csv"
            sub_dir.mkdir(parents=True, exist_ok=True)
            sub.to_csv(tmp, index=False)
            run_tests(tmp, args.metric_col, sub_dir, label=f"{args.group_col}={val}")


if __name__ == "__main__":
    main()
