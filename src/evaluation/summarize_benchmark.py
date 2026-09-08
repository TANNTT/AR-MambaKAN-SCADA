import argparse
from pathlib import Path

import pandas as pd


def _rank_per_station(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    x = df.copy()
    x["rank"] = x.groupby(["station", "code"])[metric].rank(method="min")
    return x


def _collapse_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds so each (station, code, model) is one stable estimate."""
    if "seed" not in df.columns:
        return df
    metric_cols = [c for c in ["mae", "rmse", "mape_pct", "smape_pct", "runtime_sec"] if c in df.columns]
    keep = [c for c in ["station", "code", "model", "horizon", "quality_score"] if c in df.columns]
    agg = df.groupby([c for c in ["station", "code", "model"] if c in df.columns], as_index=False)[metric_cols].mean()
    extra = df[keep].drop_duplicates(subset=[c for c in ["station", "code", "model"] if c in df.columns])
    return agg.merge(extra, on=[c for c in ["station", "code", "model"] if c in df.columns], how="left")


def summarize(input_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    df = _collapse_seeds(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mean/Std summary by model
    summary = (
        df.groupby("model")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            smape_mean=("smape_pct", "mean"),
            smape_std=("smape_pct", "std"),
            n_cases=("model", "count"),
        )
        .reset_index()
        .sort_values("mae_mean")
    )
    summary.to_csv(out_dir / "model_summary.csv", index=False)

    # 2) Win-rate by station-code task
    ranked = _rank_per_station(df, "mae")
    winners = ranked[ranked["rank"] == 1]
    total_tasks = df[["station", "code"]].drop_duplicates().shape[0]
    win_rate = winners.groupby("model").size().reset_index(name="wins")
    win_rate["total_tasks"] = total_tasks
    win_rate["win_rate_pct"] = 100.0 * win_rate["wins"] / total_tasks
    win_rate = win_rate[["model", "wins", "total_tasks", "win_rate_pct"]].sort_values("win_rate_pct", ascending=False)
    win_rate.to_csv(out_dir / "model_win_rate.csv", index=False)

    # 3) Detailed ranking table for appendix
    detailed = ranked.sort_values(["station", "code", "rank", "model"])
    detailed.to_csv(out_dir / "detailed_rankings.csv", index=False)

    # 4) LaTeX-ready main table
    latex_df = summary.copy()
    latex_df["MAE (mean±std)"] = latex_df.apply(lambda r: f"{r['mae_mean']:.2f} ± {r['mae_std']:.2f}", axis=1)
    latex_df["RMSE (mean±std)"] = latex_df.apply(lambda r: f"{r['rmse_mean']:.2f} ± {r['rmse_std']:.2f}", axis=1)
    latex_df["sMAPE% (mean±std)"] = latex_df.apply(lambda r: f"{r['smape_mean']:.2f} ± {r['smape_std']:.2f}", axis=1)
    latex_df = latex_df[["model", "MAE (mean±std)", "RMSE (mean±std)", "sMAPE% (mean±std)", "n_cases"]]
    latex_content = latex_df.to_latex(index=False, escape=False)
    (out_dir / "model_summary.tex").write_text(latex_content, encoding="utf-8")

    print(f"Saved: {out_dir / 'model_summary.csv'}")
    print(f"Saved: {out_dir / 'model_win_rate.csv'}")
    print(f"Saved: {out_dir / 'detailed_rankings.csv'}")
    print(f"Saved: {out_dir / 'model_summary.tex'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-dir", default="artifacts/paper_tables")
    args = ap.parse_args()
    summarize(Path(args.input_csv), Path(args.out_dir))


if __name__ == "__main__":
    main()

