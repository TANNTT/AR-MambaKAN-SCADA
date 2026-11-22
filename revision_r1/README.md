# Revision R1 analyses (IET GTD-2026-11-0840)

Code and **aggregated** results added for the IET Generation, Transmission & Distribution revision R1:

- full-window BiLSTM readout (vs final-step)
- target-share control (matched budget)
- rolling-origin windows
- seasonal and weekly evaluation
- two-sided Wilcoxon tests

Raw SCADA telemetry, the meter anonymization map, checkpoints, and per-timestep forecasts are **not** included.

## Environment variables

Set these before running any script that loads local data. There are **no** machine-local defaults.

| Variable | Role |
| --- | --- |
| `SCADA_ANON_MAP` | Path to meter anonymization CSV (`real_code`, `anon_label`) |
| `SCADA_ARTIFACTS_DIR` | Root of built meter CSVs (`*_CODE_2020_2023.csv`) |
| `SCADA_ARTIFACTS_NOCAL_DIR` | Optional second artifacts tree (calendar-ablation builds) |
| `SCADA_WEATHER_DIR` | Directory of per-station ERA5 weather CSVs |
| `SCADA_WEATHER_STATION_REF` | Reference CSV with `code`/`station` for weather slug lookup |
| `SCADA_DEMO_DATA` | Synthetic `DEMO*.csv` directory |
| `SCADA_PUBLIC_REPO` | Checkout of the public AR-MambaKAN-SCADA tree (for reference CSVs) |
| `SCADA_CODE_ROOT` | Private experiment code tree (reference CSVs / scripts only) |
| `SCADA_KIEU_TRINH_BAY` | Path to `kieu-trinh-bay.py` (figure style helper; for `xuat_bang_sua.py`) |
| `SCADA_PAPER_ROOT` | This `revision_r1` directory (shell wrappers) |
| `SCADA_PYTHON` | Project Python interpreter (shell wrappers) |

Missing variables raise a clear error at run time.

## Scripts → paper tables / figures

| Script | Output under `results/` | Paper |
| --- | --- | --- |
| `tools/bang_hai_cach_doc.py` | `bang-hai-cach-doc/` | Table 18 (final-step vs full-window) |
| `tools/tong_g_chung.py` | `g-chung-tong/` | Supports Table 18 / pooled–local tests |
| `tools/tong_g2b.py` | `g2b-tong/` | Ablation under full-window readout |
| `tools/tong_g3.py` / `tools/so_g3_voi_moc.py` | `g3-tong/`, `g3-so-moc/` | Target-share control |
| `tools/tong_g4.py` | `g4-tong/` | Rolling-origin windows |
| `tools/p_hai_phia.py` | `p-hai-phia/` | Two-sided p-values |
| `phan_tich_r1_r2_cpu.py` | `r1-r2-cpu/` | Seasonal / weekly / daily scores |
| `tools/xuat_bang_sua.py` | `bang-sua/` | Tables 18, 22, 23; Fig. 10–11 exports |
| `g_chung/chay.py` + `g_chung/lo*.sh` | per-run dirs (not shipped) | Training entry for the analyses above |

Shipped under `results/` are multi-meter aggregates only (means, p-values, seasonal/weekly tables). Per-run `npz` / checkpoints and Fig. 10 timestep series are omitted.
