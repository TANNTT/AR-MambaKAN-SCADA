#!/usr/bin/env python3
"""G-chung single-run entry: BiLSTM / AR-MambaKAN × pooled/local × last/full readout.

Assumptions (this run):
1. Protocol mirrors AR-MambaKAN-SCADA @ 8582ecc common_pooled.py + model_factory.py:
   lookback 336, epochs<=100, patience 10, min_delta 0.001, batch 128, lr 1e-3,
   split 70/10/20 by row index, standardize from train only, calendar_features=True
   (as run_pooled.py / run_local.py).
2. Pooled mixing: ConcatDataset of per-meter windows each standardized with that
   meter's train stats; early-stop val uses first pool meter's y_mean/y_std
   (common_pooled.py:153-159); test MAE always per-meter own stats.
3. Optimizer: bilstm=Adam(lr), armambakan=AdamW(min(lr,5e-4), wd=3e-4) via model_factory
   (common_pooled._fit hardcodes armambakan only because that helper trains AR-MambaKAN).
4. Reported mae_* in summary.json are micro-averages over saved npz arrays (exact match
   on recomputation). Early stopping still uses eval_epoch mean-of-batch-means.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

# Allow `python g_chung/chay.py` from paper root
_PAPER = Path(__file__).resolve().parents[1]
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from g_chung.budget import (  # noqa: E402
    assert_plan_windows,
    equal_share_budget,
    subsample_train_datasets,
    target_frac_budget,
)
from g_chung.data import (  # noqa: E402
    MAIN_METERS,
    ROLLING_FOLDS,
    _demo_path,
    _map_path,
    _search_roots,
    assert_no_leakage,
    load_meter_bundle,
    load_meter_bundle_fold,
    resolve_meter_index,
    resolve_weather_csv,
)
from g_chung.models.model_factory import ALL_TRAINABLE, build_model  # noqa: E402
from g_chung.train_core import (  # noqa: E402
    BATCH,
    EPOCHS,
    LOOKBACK,
    LR,
    PATIENCE,
    fit_model,
    limit_threads,
    mae_from_arrays,
    predict_destandardized,
    set_seed,
)


def git_commit(cwd: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_name(model: str, regime: str, readout: str, horizon: int, seed: int, meter: str | None) -> str:
    base = f"{model}_{regime}_{readout}_h{horizon}_s{seed}"
    if regime == "local" and meter:
        return f"{base}_{meter}"
    return base


def save_env(out_dir: Path) -> None:
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception as exc:
        freeze = f"# pip freeze failed: {exc}\n"
    (out_dir / "env.txt").write_text(
        f"python={sys.executable}\n"
        f"platform={platform.platform()}\n"
        f"torch={torch.__version__}\n"
        f"numpy={np.__version__}\n"
        f"cuda_available={torch.cuda.is_available()}\n"
        f"--- pip freeze ---\n{freeze}"
    )


def _loaders(ds, batch: int, shuffle: bool, device: str):
    # pin_memory only useful on CUDA
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )


def _load_bundle(
    path: Path,
    horizon: int,
    calendar_features: bool,
    fold: str | None,
    weather_csv: Path | None = None,
):
    if fold is None:
        return load_meter_bundle(
            path, horizon, calendar_features=calendar_features, weather_csv=weather_csv
        )
    return load_meter_bundle_fold(
        path, horizon, fold, calendar_features=calendar_features, weather_csv=weather_csv
    )


def train_local(
    *,
    model_name: str,
    readout: str,
    meter_label: str,
    meter_path: Path,
    horizon: int,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch: int,
    calendar_features: bool,
    fold: str | None = None,
    check_leakage: bool = False,
    weather_csv: Path | None = None,
) -> dict:
    set_seed(seed)
    bundle = _load_bundle(meter_path, horizon, calendar_features, fold, weather_csv=weather_csv)
    leakage = assert_no_leakage(bundle, horizon) if check_leakage else None
    train_loader = _loaders(bundle["train_ds"], batch, True, device)
    val_loader = _loaders(bundle["val_ds"], batch, False, device)
    test_loader = _loaders(bundle["test_ds"], batch, False, device)

    model = build_model(model_name, bundle["n_features"], LOOKBACK, horizon, device, readout=readout)
    t0 = time.perf_counter()
    fit = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        bundle["y_mean"],
        bundle["y_std"],
        model_name,
        epochs=epochs,
        patience=patience,
        lr=LR,
    )
    train_sec = time.perf_counter() - t0
    model = fit["model"]

    val_pack = predict_destandardized(
        model, val_loader, device, bundle["y_mean"].to(device), bundle["y_std"].to(device)
    )
    test_pack = predict_destandardized(
        model, test_loader, device, bundle["y_mean"].to(device), bundle["y_std"].to(device)
    )
    val_mae = mae_from_arrays(val_pack["pred"], val_pack["y"])
    test_mae = mae_from_arrays(test_pack["pred"], test_pack["y"])
    return {
        "meters": {
            meter_label: {
                "n_rows": bundle["n_rows"],
                "n_train_windows": len(bundle["train_ds"]),
                "n_val_windows": len(bundle["val_ds"]),
                "n_test_windows": len(bundle["test_ds"]),
                "val": val_pack,
                "test": test_pack,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "split": bundle.get("split"),
            }
        },
        "best_epoch": fit["best_epoch"],
        "stopped_epoch": fit["stopped_epoch"],
        "best_val_mae_train_obj": fit["best_val_mae"],
        "train_sec": train_sec,
        "n_features": bundle["n_features"],
        "pooled_train_windows": None,
        "leakage_check": leakage,
        "budget": None,
    }


def train_pooled(
    *,
    model_name: str,
    readout: str,
    meter_index: dict[str, Path],
    horizon: int,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch: int,
    calendar_features: bool,
    fold: str | None = None,
    check_leakage: bool = False,
    heldout_meter: str | None = None,
    weather_dir: Path | None = None,
) -> dict:
    # common_pooled.train_pooled_armambakan lines 116-168 @ 8582ecc
    # LOSO (run_loso.py): pool_codes = all except heldout; eval_codes = [heldout].
    set_seed(seed)
    all_labels = list(meter_index.keys())
    if heldout_meter is not None:
        if heldout_meter not in meter_index:
            raise KeyError(f"heldout {heldout_meter} not in meter_index")
        pool_labels = [lab for lab in all_labels if lab != heldout_meter]
        eval_labels = [heldout_meter]
        if len(pool_labels) < 2:
            raise RuntimeError(f"LOSO needs >=2 pool meters, got {len(pool_labels)}")
    else:
        pool_labels = all_labels
        eval_labels = all_labels

    def _wx(lab: str) -> Path | None:
        if weather_dir is None:
            return None
        p = resolve_weather_csv(weather_dir, lab)
        if p is None:
            raise FileNotFoundError(f"no weather CSV for {lab} in {weather_dir}")
        return p

    need_labels = list(dict.fromkeys(pool_labels + eval_labels))
    bundles = {
        lab: _load_bundle(
            meter_index[lab], horizon, calendar_features, fold, weather_csv=_wx(lab)
        )
        for lab in need_labels
    }
    n_features = bundles[pool_labels[0]]["n_features"]
    for lab, b in bundles.items():
        if b["n_features"] != n_features:
            raise RuntimeError(
                f"feature dim mismatch: {pool_labels[0]}={n_features} vs {lab}={b['n_features']}"
            )

    leakage = None
    if check_leakage:
        leakage = {lab: assert_no_leakage(bundles[lab], horizon) for lab in need_labels}

    train_subsets = [bundles[lab]["train_ds"] for lab in pool_labels]
    val_subsets = [bundles[lab]["val_ds"] for lab in pool_labels]
    pooled_train = ConcatDataset(train_subsets)
    pooled_val = ConcatDataset(val_subsets)
    train_loader = _loaders(pooled_train, batch, True, device)
    val_loader = _loaders(pooled_val, batch, False, device)

    ref_mean, ref_std = bundles[pool_labels[0]]["y_mean"], bundles[pool_labels[0]]["y_std"]
    model = build_model(model_name, n_features, LOOKBACK, horizon, device, readout=readout)
    t0 = time.perf_counter()
    fit = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        ref_mean,
        ref_std,
        model_name,
        epochs=epochs,
        patience=patience,
    )
    train_sec = time.perf_counter() - t0
    model = fit["model"]

    meters_out = {}
    for lab in eval_labels:
        b = bundles[lab]
        val_loader_m = _loaders(b["val_ds"], batch, False, device)
        test_loader_m = _loaders(b["test_ds"], batch, False, device)
        val_pack = predict_destandardized(
            model, val_loader_m, device, b["y_mean"].to(device), b["y_std"].to(device)
        )
        test_pack = predict_destandardized(
            model, test_loader_m, device, b["y_mean"].to(device), b["y_std"].to(device)
        )
        # For LOSO heldout: n_train_windows is 0 in the pool (excluded from weight training).
        n_train_reported = (
            0 if (heldout_meter is not None and lab == heldout_meter) else len(b["train_ds"])
        )
        meters_out[lab] = {
            "n_rows": b["n_rows"],
            "n_train_windows": n_train_reported,
            "n_val_windows": len(b["val_ds"]),
            "n_test_windows": len(b["test_ds"]),
            "val": val_pack,
            "test": test_pack,
            "val_mae": mae_from_arrays(val_pack["pred"], val_pack["y"]),
            "test_mae": mae_from_arrays(test_pack["pred"], test_pack["y"]),
            "split": b.get("split"),
        }
    return {
        "meters": meters_out,
        "best_epoch": fit["best_epoch"],
        "stopped_epoch": fit["stopped_epoch"],
        "best_val_mae_train_obj": fit["best_val_mae"],
        "train_sec": train_sec,
        "n_features": n_features,
        "pooled_train_windows": len(pooled_train),
        "leakage_check": leakage,
        "budget": None,
        "heldout_meter": heldout_meter,
        "pool_meters": pool_labels,
    }


def train_matched(
    *,
    model_name: str,
    readout: str,
    meter_index: dict[str, Path],
    target: str | None,
    horizon: int,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch: int,
    calendar_features: bool,
    target_frac: float | None,
    equal_share: bool = False,
) -> dict:
    """Matched-budget train.

    equal_share (Table 6 / run_matched_budget.py): one model, eval ALL meters.
    target_frac (G3): one model per target, eval target only.
    Sampling: np.random.default_rng(seed).choice without replacement
    (common_pooled.py:136-143 @ 8582ecc).
    """
    set_seed(seed)
    labels = list(meter_index.keys())
    bundles = {
        lab: load_meter_bundle(path, horizon, calendar_features=calendar_features)
        for lab, path in meter_index.items()
    }
    local_ns = {lab: len(bundles[lab]["train_ds"]) for lab in labels}
    if equal_share:
        plan = equal_share_budget(local_ns)
        eval_labels = labels
    else:
        if target is None:
            raise ValueError("target required unless equal_share=True")
        if target not in meter_index:
            raise KeyError(f"target {target} not in meter_index")
        if target_frac is None:
            raise ValueError("target_frac required unless equal_share=True")
        plan = target_frac_budget(local_ns, target=target, p=target_frac)
        eval_labels = [target]
    assert_plan_windows(plan)

    # Preserve label order for ConcatDataset; budgets may omit nothing
    train_map = {lab: bundles[lab]["train_ds"] for lab in labels}
    # Budget plan may only list meters in local_ns; ensure all labels present
    for lab in labels:
        if lab not in plan.budgets:
            raise RuntimeError(f"missing budget for {lab}")
    train_subsets, pooled_n = subsample_train_datasets(train_map, plan.budgets, seed)
    if pooled_n != plan.total:
        raise RuntimeError(f"pooled_n {pooled_n} != plan.total {plan.total}")

    val_subsets = [bundles[lab]["val_ds"] for lab in labels]
    pooled_train = ConcatDataset(train_subsets)
    pooled_val = ConcatDataset(val_subsets)
    train_loader = _loaders(pooled_train, batch, True, device)
    val_loader = _loaders(pooled_val, batch, False, device)

    n_features = bundles[labels[0]]["n_features"]
    ref_mean, ref_std = bundles[labels[0]]["y_mean"], bundles[labels[0]]["y_std"]
    model = build_model(model_name, n_features, LOOKBACK, horizon, device, readout=readout)
    t0 = time.perf_counter()
    fit = fit_model(
        model,
        train_loader,
        val_loader,
        device,
        ref_mean,
        ref_std,
        model_name,
        epochs=epochs,
        patience=patience,
    )
    train_sec = time.perf_counter() - t0
    model = fit["model"]

    meters_out = {}
    for lab in eval_labels:
        b = bundles[lab]
        val_loader_m = _loaders(b["val_ds"], batch, False, device)
        test_loader_m = _loaders(b["test_ds"], batch, False, device)
        val_pack = predict_destandardized(
            model, val_loader_m, device, b["y_mean"].to(device), b["y_std"].to(device)
        )
        test_pack = predict_destandardized(
            model, test_loader_m, device, b["y_mean"].to(device), b["y_std"].to(device)
        )
        meters_out[lab] = {
            "n_rows": b["n_rows"],
            "n_train_windows": plan.budgets[lab],
            "n_val_windows": len(b["val_ds"]),
            "n_test_windows": len(b["test_ds"]),
            "val": val_pack,
            "test": test_pack,
            "val_mae": mae_from_arrays(val_pack["pred"], val_pack["y"]),
            "test_mae": mae_from_arrays(test_pack["pred"], test_pack["y"]),
            "split": b.get("split"),
        }
    return {
        "meters": meters_out,
        "best_epoch": fit["best_epoch"],
        "stopped_epoch": fit["stopped_epoch"],
        "best_val_mae_train_obj": fit["best_val_mae"],
        "train_sec": train_sec,
        "n_features": n_features,
        "pooled_train_windows": pooled_n,
        "leakage_check": None,
        "budget": {
            "mode": plan.mode,
            "target": plan.target if not equal_share else None,
            "p": plan.p,
            "n_local_target": plan.n_local_target,
            "budgets": plan.budgets,
            "total": plan.total,
            "local_ns": local_ns,
        },
    }


def write_outputs(out_dir: Path, result: dict, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_meter = {}
    total_rows = 0
    for lab, m in result["meters"].items():
        npz_path = out_dir / f"pred_{lab}.npz"
        np.savez_compressed(
            npz_path,
            pred_val=m["val"]["pred"],
            y_val=m["val"]["y"],
            t_origin_val=m["val"]["t_origin"],
            t_target_val=m["val"]["t_target"],
            pred_test=m["test"]["pred"],
            y_test=m["test"]["y"],
            t_origin_test=m["test"]["t_origin"],
            t_target_test=m["test"]["t_target"],
        )
        total_rows += int(m["n_rows"])
        per_meter[lab] = {
            "n_rows": m["n_rows"],
            "n_train_windows": m["n_train_windows"],
            "n_val_windows": m["n_val_windows"],
            "n_test_windows": m["n_test_windows"],
            "val_mae": m["val_mae"],
            "test_mae": m["test_mae"],
            "npz": npz_path.name,
            "npz_bytes": npz_path.stat().st_size,
        }

    summary = {
        **meta,
        "best_epoch": result["best_epoch"],
        "stopped_epoch": result["stopped_epoch"],
        "best_val_mae_train_obj": result["best_val_mae_train_obj"],
        "train_sec": result["train_sec"],
        "n_features": result["n_features"],
        "pooled_train_windows": result["pooled_train_windows"],
        "n_rows_total": total_rows,
        "per_meter": per_meter,
        "mean_test_mae_joint": float(
            np.mean([m["test_mae"]["mae_joint"] for m in result["meters"].values()])
        ),
        "budget": result.get("budget"),
        "leakage_check": result.get("leakage_check"),
        "heldout_meter": result.get("heldout_meter"),
        "pool_meters": result.get("pool_meters"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    save_env(out_dir)
    return summary_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="G-chung single training run")
    # Default choices unchanged for G0; extra models are additive for G2b.
    ap.add_argument("--model", choices=sorted(ALL_TRAINABLE), required=True)
    ap.add_argument("--regime", choices=["pooled", "local", "matched"], required=True)
    ap.add_argument("--readout", choices=["last", "full"], required=True)
    ap.add_argument("--horizon", type=int, required=True, choices=[4, 48, 96, 336])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--meter", default=None, help="Anon/demo label for local (required if local)")
    ap.add_argument("--meters", nargs="+", default=None, help="Override meter list (pooled or subset)")
    ap.add_argument("--demo", action="store_true", help="Use synthetic demo_data")
    ap.add_argument("--demo-dir", type=Path, default=None, help="default: $SCADA_DEMO_DATA")
    ap.add_argument("--map", type=Path, default=None, help="default: $SCADA_ANON_MAP")
    ap.add_argument("--search-root", action="append", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True, help="results/<job>/<run-name>")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--no-calendar", action="store_true", help="Disable calendar features (default ON)")
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--lo", action="store_true", help="Ignored here; use g_chung/lo.sh")
    # --- Additive flags (defaults preserve G0 / G-chung behaviour) ---
    ap.add_argument(
        "--fold",
        default=None,
        choices=list(ROLLING_FOLDS),
        help="G4 rolling-origin fold (default: None = ratio 70/10/20 split)",
    )
    ap.add_argument(
        "--check-leakage",
        action="store_true",
        help="G4: assert window timestamps never cross train/val/test bounds",
    )
    ap.add_argument(
        "--target-meter",
        default=None,
        help="G3 matched regime: target meter label (also used as eval meter)",
    )
    ap.add_argument(
        "--target-frac",
        type=float,
        default=None,
        help="G3: fraction p of n_local(target) reserved for the target meter",
    )
    ap.add_argument(
        "--equal-share",
        action="store_true",
        help="G3 check: use original equal-share budget (run_matched_budget.py)",
    )
    ap.add_argument(
        "--ablation",
        default=None,
        help="Optional ablation code tag stored in summary (A1..A6)",
    )
    ap.add_argument(
        "--heldout-meter",
        default=None,
        help="LOSO: exclude this meter from pooled training; eval on it only (regime=pooled)",
    )
    ap.add_argument(
        "--weather-dir",
        type=Path,
        default=None,
        help="Join ERA5 wx_* from this dir (station_slug CSV per meter). Implies calendar ON.",
    )
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.map is None:
        args.map = _map_path()
    if args.demo_dir is None:
        args.demo_dir = _demo_path()
    if args.lookback != LOOKBACK:
        raise SystemExit(f"lookback must be {LOOKBACK} per locked design (got {args.lookback})")
    if args.regime == "local" and not args.meter and not (args.meters and len(args.meters) == 1):
        raise SystemExit("local regime requires --meter LABEL")
    if args.regime == "matched":
        if not args.equal_share and not args.target_meter and not args.meter:
            raise SystemExit("matched regime requires --target-meter (or --equal-share)")
        if not args.equal_share and args.target_frac is None:
            raise SystemExit("matched regime requires --target-frac or --equal-share")
    if args.heldout_meter and args.regime != "pooled":
        raise SystemExit("--heldout-meter only valid with --regime pooled")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    if args.fold is not None and args.regime == "matched":
        raise SystemExit("fold and matched regime are mutually exclusive")
    if args.weather_dir is not None and args.no_calendar:
        raise SystemExit("--weather-dir requires calendar features (do not pass --no-calendar)")

    limit_threads(args.num_threads)
    calendar_features = not args.no_calendar
    search_roots = args.search_root or _search_roots()
    weather_dir = args.weather_dir

    if args.regime == "local":
        meter = args.meter or args.meters[0]
        meter_index = resolve_meter_index(
            use_demo=args.demo,
            meters=[meter],
            map_path=args.map,
            search_roots=search_roots,
            demo_dir=args.demo_dir,
        )
    else:
        meters = args.meters or (
            list(resolve_meter_index(use_demo=True, meters=None, demo_dir=args.demo_dir).keys())
            if args.demo
            else list(MAIN_METERS)
        )
        meter_index = resolve_meter_index(
            use_demo=args.demo,
            meters=meters,
            map_path=args.map,
            search_roots=search_roots,
            demo_dir=args.demo_dir,
        )

    out_dir = args.out
    if out_dir.exists() and (out_dir / "summary.json").is_file():
        print(f"SKIP existing {out_dir / 'summary.json'}", flush=True)
        return 0

    device = args.device
    target = args.target_meter or args.meter
    run_meter = None
    if args.regime == "local":
        run_meter = args.meter or args.meters[0]
    elif args.regime == "matched" and not args.equal_share:
        run_meter = target
    elif args.heldout_meter:
        run_meter = args.heldout_meter
    name = run_name(args.model, args.regime, args.readout, args.horizon, args.seed, run_meter)
    if args.fold:
        name = f"{name}_{args.fold}"
    if args.regime == "matched" and args.equal_share:
        name = f"{args.model}_matched_{args.readout}_h{args.horizon}_s{args.seed}_eqshare"
    if args.regime == "matched" and args.target_frac is not None and not args.equal_share:
        name = f"{name}_p{args.target_frac:g}"
    if args.heldout_meter:
        name = f"{args.model}_loso_{args.readout}_h{args.horizon}_s{args.seed}_{args.heldout_meter}"
    if args.weather_dir is not None:
        name = f"{name}_weather"
    if args.ablation:
        name = f"{args.ablation}_{name}"

    wx_joined = False
    if weather_dir is not None:
        for lab in meter_index:
            if resolve_weather_csv(weather_dir, lab) is None:
                raise SystemExit(f"missing weather CSV for {lab} under {weather_dir}")
        wx_joined = True

    meta = {
        "run": name,
        "model": args.model,
        "regime": args.regime,
        "readout": args.readout,
        "horizon": args.horizon,
        "seed": args.seed,
        "lookback": LOOKBACK,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "calendar_features": calendar_features,
        "weather_features": wx_joined,
        "weather_dir": str(weather_dir) if weather_dir else None,
        "fold": args.fold,
        "ablation": args.ablation,
        "heldout_meter": args.heldout_meter,
        "target_meter": target if args.regime == "matched" and not args.equal_share else None,
        "target_frac": args.target_frac if args.regime == "matched" else None,
        "equal_share": bool(args.equal_share) if args.regime == "matched" else None,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "host": platform.node(),
        "git_commit": git_commit(_PAPER),
        "source_upstream_commit": "8582ecc937b45081f5c4116694c73d685f6c8508",
        "meter_labels": list(meter_index.keys()),
        "demo": bool(args.demo),
    }

    print(
        f"RUN {meta['run']} device={device} meters={meta['meter_labels']} "
        f"epochs<={args.epochs} calendar={calendar_features} weather={wx_joined} "
        f"fold={args.fold} heldout={args.heldout_meter}",
        flush=True,
    )

    if args.regime == "local":
        lab = list(meter_index.keys())[0]
        wx = resolve_weather_csv(weather_dir, lab) if weather_dir else None
        result = train_local(
            model_name=args.model,
            readout=args.readout,
            meter_label=lab,
            meter_path=meter_index[lab],
            horizon=args.horizon,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch_size,
            calendar_features=calendar_features,
            fold=args.fold,
            check_leakage=args.check_leakage,
            weather_csv=wx,
        )
    elif args.regime == "pooled":
        result = train_pooled(
            model_name=args.model,
            readout=args.readout,
            meter_index=meter_index,
            horizon=args.horizon,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch_size,
            calendar_features=calendar_features,
            fold=args.fold,
            check_leakage=args.check_leakage,
            heldout_meter=args.heldout_meter,
            weather_dir=weather_dir,
        )
    else:
        result = train_matched(
            model_name=args.model,
            readout=args.readout,
            meter_index=meter_index,
            target=target,
            horizon=args.horizon,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch_size,
            calendar_features=calendar_features,
            target_frac=args.target_frac,
            equal_share=args.equal_share,
        )

    summary_path = write_outputs(out_dir, result, meta)
    print(
        f"DONE {meta['run']} mean_test_mae_joint={json.loads(summary_path.read_text())['mean_test_mae_joint']:.4f} "
        f"best_epoch={result['best_epoch']} train_sec={result['train_sec']:.1f} -> {summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
