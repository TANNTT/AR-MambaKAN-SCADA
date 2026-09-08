import argparse
import csv
from pathlib import Path

import pandas as pd


def summarize_multihorizon(input_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mean metric by (horizon, model)
    hm = (
        df.groupby(["horizon", "model"])
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            smape_mean=("smape_pct", "mean"),
            n_cases=("model", "count"),
        )
        .reset_index()
        .sort_values(["horizon", "mae_mean"])
    )
    hm.to_csv(out_dir / "horizon_model_summary.csv", index=False)

    # Best model per horizon by mean MAE
    best = hm.sort_values(["horizon", "mae_mean"]).groupby("horizon").head(1).reset_index(drop=True)
    best.to_csv(out_dir / "best_model_per_horizon.csv", index=False)

    # Pivot table for paper (rows=model, cols=horizon, value=MAE)
    pivot_mae = hm.pivot(index="model", columns="horizon", values="mae_mean")
    pivot_mae.to_csv(out_dir / "pivot_mae_by_horizon.csv")
    pivot_smape = hm.pivot(index="model", columns="horizon", values="smape_mean")
    pivot_smape.to_csv(out_dir / "pivot_smape_by_horizon.csv")

    # latex friendly row-wise horizon table
    latex_df = hm.copy()
    latex_df["MAE"] = latex_df["mae_mean"].map(lambda v: f"{v:.2f}")
    latex_df["RMSE"] = latex_df["rmse_mean"].map(lambda v: f"{v:.2f}")
    latex_df["sMAPE%"] = latex_df["smape_mean"].map(lambda v: f"{v:.2f}")
    latex_df = latex_df[["horizon", "model", "MAE", "RMSE", "sMAPE%", "n_cases"]]
    (out_dir / "horizon_model_summary.tex").write_text(latex_df.to_latex(index=False, escape=False), encoding="utf-8")

    # plain-text markdown takeaway
    lines = ["# Multi-Horizon Summary", ""]
    for _, r in best.iterrows():
        lines.append(
            f"- Horizon `{int(r['horizon'])}`: best model `{r['model']}` "
            f"(MAE={r['mae_mean']:.2f}, sMAPE={r['smape_mean']:.2f}%)."
        )
    (out_dir / "multihorizon_takeaways.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_dir / 'horizon_model_summary.csv'}")
    print(f"Saved: {out_dir / 'best_model_per_horizon.csv'}")
    print(f"Saved: {out_dir / 'pivot_mae_by_horizon.csv'}")
    print(f"Saved: {out_dir / 'pivot_smape_by_horizon.csv'}")
    print(f"Saved: {out_dir / 'horizon_model_summary.tex'}")
    print(f"Saved: {out_dir / 'multihorizon_takeaways.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-dir", default="artifacts/multihorizon_tables")
    args = ap.parse_args()
    summarize_multihorizon(Path(args.input_csv), Path(args.out_dir))


if __name__ == "__main__":
    main()

