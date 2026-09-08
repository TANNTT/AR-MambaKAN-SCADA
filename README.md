# AR-MambaKAN-SCADA

Reproducibility package for:

> Thi-Thanh-Tan Nguyen. **Pooled Multi-Substation Training Outperforms Per-Meter Local Training for
> Joint Active-Reactive Power Forecasting on Real Distribution Supervisory Control and Data
> Acquisition Telemetry.** Submitted to *Sustainable Energy, Grids and Networks*.

This repository contains the full model implementation, preprocessing/feature pipeline,
experiment scripts, statistical-analysis code, and non-confidential aggregated result files needed
to reproduce every table and statistical test in the paper. It does **not** contain the raw EVN
SCADA telemetry used in the paper, which is proprietary (see [Data availability](#data-availability)
below) -- a small synthetic dataset with the same schema is bundled instead so the pipeline can be
run end-to-end by anyone.

## What this paper found

Using real SCADA telemetry from 10-20 primary distribution substations of Vietnam Electricity
(EVN), we compare **local training** (one model per substation, the paradigm used throughout the
prior joint P-Q forecasting literature) against **pooled training** (one shared model fit on
concatenated multi-substation data, evaluated per substation), across four architectures (MLP,
LSTM, BiLSTM, and AR-MambaKAN, a gated-convolution + BiLSTM + Kolmogorov-Arnold-network hybrid).
Pooled training significantly outperforms local training for every architecture tested
(-14% to -36% MAE, all Wilcoxon p < 0.001), a result that survives matched-training-budget,
leave-one-substation-out, meter-level, and cluster-bootstrap robustness checks. This is a
**training-protocol** finding, not an architecture contest.

## Repository structure

```
AR-MambaKAN-SCADA/
├── README.md                    <- you are here
├── LICENSE                      <- MIT (code only; see note in LICENSE)
├── requirements.txt
├── CITATION.cff
├── src/
│   ├── models/                  <- AR-MambaKAN + all baseline architectures
│   ├── preprocessing/           <- raw SCADA parsing, per-meter data-quality scoring
│   ├── features/                <- calendar-feature + ERA5-weather-feature generation
│   └── evaluation/               <- training loop, metrics, statistical tests
├── experiments/
│   ├── local/                   <- per-meter local training (Table 1, "Local" columns)
│   ├── pooled/                  <- shared helpers + pooled training (Table 1, "Pooled" columns)
│   ├── matched_budget/          <- data-budget-matched pooling (Table 4)
│   ├── loso/                    <- leave-one-substation-out (Table 5)
│   ├── calendar_ablation/       <- calendar-feature ablation (Table 17)
│   └── weather_ablation/        <- ERA5 weather-feature ablation (Table 18)
├── scripts/
│   ├── run_all.sh               <- end-to-end demo run on the synthetic data
│   ├── statistical_tests.py     <- Wilcoxon / Friedman implementations used throughout the paper
│   └── reproduce_tables.py      <- regenerates Table 1 / Table 8 from results/*.csv
├── results/                     <- real, non-confidential aggregated result CSVs from the paper
├── demo_data/                   <- synthetic SCADA data (same schema as the real data)
└── docs/
    └── data_schema.md           <- exact column schema expected by every script
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python >= 3.10 and PyTorch. A GPU is not required (all scripts accept `--device cpu`),
but training is considerably faster on one.

## Quick start: run everything on the bundled synthetic data

```bash
bash scripts/run_all.sh cpu   # or: bash scripts/run_all.sh cuda
```

This trains local and pooled AR-MambaKAN models on the 4 synthetic demo meters, runs the
matched-budget, LOSO, and calendar-ablation experiments, and finally regenerates Table 1 / Table 8
from the paper's **real** result files in `results/` (not from the demo run) to confirm the
statistical-test code reproduces the numbers reported in the manuscript. Expected output for the
last step:

```
=== Table 1 (AR-MambaKAN row): pooled vs. local ===
Local MAE:  696.9
Pooled MAE: 579.2
Delta:      -16.9%
Tasks improved: 35/40
Wilcoxon p: 8.1e-07

=== Table 8: nine-condition Friedman ranking ===
chi2=154.35  df=8  p=2.43e-29
Pooled AR-MambaKAN    2.725
...
```

## Reproducing on your own data

Point any experiment script's `--data-dir` at a folder of per-meter CSVs following
[`docs/data_schema.md`](docs/data_schema.md), e.g.:

```bash
export PYTHONPATH=src:experiments/pooled
python experiments/local/run_local.py   --data-dir /path/to/your/meters --out-csv results/local.csv
python experiments/pooled/run_pooled.py --data-dir /path/to/your/meters --out-csv results/pooled.csv
python experiments/matched_budget/run_matched_budget.py --data-dir /path/to/your/meters
python experiments/loso/run_loso.py                     --data-dir /path/to/your/meters
python experiments/calendar_ablation/run_calendar_ablation.py --data-dir /path/to/your/meters
```

The paper's exact protocol uses `--horizons 4 48 96 336 --seeds 42 43 44` (the defaults in every
script) over 10 real EVN meters.

## Data availability

The raw SCADA telemetry used in this study is proprietary to Vietnam Electricity (EVN) and cannot
be publicly released under the applicable data-access restrictions. This repository therefore does
**not** include:

- Timestamp-by-timestamp substation P/Q trajectories, three-phase voltage, or other raw
  operational telemetry.
- Any mapping from meter code to a real-world substation name or location.
- Internal EVN metadata, credentials, API keys, or network paths.

What **is** included: the complete model implementations, preprocessing and experiment scripts,
statistical-analysis code, configuration/hyperparameters, non-confidential aggregated result files
(`results/`), and a synthetic example dataset (`demo_data/`) with the same input schema so the
end-to-end pipeline can be exercised without the real data.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please cite the accompanying paper.

## License

Code is released under the [MIT License](LICENSE). This does not extend to any proprietary data.
