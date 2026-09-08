#!/usr/bin/env bash
# End-to-end demo run: executes every experiment against the bundled synthetic demo_data/, then
# regenerates the headline tables from the results this produces.
#
# This is a QUICK, small-scale smoke run (4 synthetic meters, default hyperparameters) meant to
# prove the pipeline works end-to-end, not to reproduce the paper's exact numbers -- for that,
# point --data-dir at your own real data with the same schema (docs/data_schema.md) and use the
# full 10+ meter, 3-seed, 4-horizon protocol described in the paper's Section 6.
#
# Usage: bash run_all.sh [cpu|cuda]
set -euo pipefail
DEVICE="${1:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/6] Local training ==="
PYTHONPATH="$ROOT/src:$ROOT/experiments/pooled" python3 experiments/local/run_local.py \
  --data-dir demo_data --out-csv results/local_demo.csv --device "$DEVICE"

echo "=== [2/6] Pooled training ==="
PYTHONPATH="$ROOT/src" python3 experiments/pooled/run_pooled.py \
  --data-dir demo_data --out-csv results/pooled_demo.csv --device "$DEVICE"

echo "=== [3/6] Data-budget-matched pooling ==="
PYTHONPATH="$ROOT/src:$ROOT/experiments/pooled" python3 experiments/matched_budget/run_matched_budget.py \
  --data-dir demo_data --out-csv results/matched_budget_demo.csv --device "$DEVICE"

echo "=== [4/6] Leave-one-substation-out ==="
PYTHONPATH="$ROOT/src:$ROOT/experiments/pooled" python3 experiments/loso/run_loso.py \
  --data-dir demo_data --out-csv results/loso_demo.csv --device "$DEVICE"

echo "=== [5/6] Calendar-feature ablation ==="
PYTHONPATH="$ROOT/src:$ROOT/experiments/pooled" python3 experiments/calendar_ablation/run_calendar_ablation.py \
  --data-dir demo_data --out-csv results/calendar_ablation_demo.csv --device "$DEVICE"

echo "=== [6/6] Statistical tests on the paper's real result CSVs ==="
python3 scripts/reproduce_tables.py --results-dir results

echo "Done. See results/*_demo.csv for this run's outputs, and results/*.csv for the paper's real,"
echo "already-computed result files (reproduced from proprietary EVN data, not from demo_data)."
