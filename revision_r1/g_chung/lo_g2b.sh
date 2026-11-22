#!/usr/bin/env bash
# G2b: component ablation with readout=full (Table 18 protocol).
# Protocol (upstream): local, NO calendar, h48, 3 seeds, 10 meters.
# Codes: A1–A6 from scada_exp/run_ablation.py; BiLSTM-affected = A3,A4,A5,A6.
# Default batch runs A3–A6 with --readout full (120 runs). Pass --all-codes for A1–A6 (180).
# Skips if summary.json exists. Output: results/g2b/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="cuda"
OUT_ROOT="$ROOT/results/g2b"
DEMO_FLAG=()
METERS=(M01 M02 M03 M04 M05 M06 M07 M08 M09 M10)
# BiLSTM readout variants (default G2b GPU scope)
CODES=(A3 A4 A5 A6)
SEEDS=(42 43 44)
HORIZON=48
READOUT=full

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --demo) DEMO_FLAG=(--demo); METERS=(DEMO0001 DEMO0002); shift ;;
    --all-codes) CODES=(A1 A2 A3 A4 A5 A6); shift ;;
    --meters) shift; METERS=(); while [[ $# -gt 0 && "$1" != --* ]]; do METERS+=("$1"); shift; done ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

declare -A MODEL_OF=(
  [A1]=mlp
  [A2]=lstm
  [A3]=bilstm
  [A4]=armambakan
  [A5]=armambakan_nostem
  [A6]=armambakan_nokan
)

mkdir -p "$OUT_ROOT"
total=$(( ${#CODES[@]} * ${#METERS[@]} * ${#SEEDS[@]} ))
done_n=0; skip_n=0; fail_n=0; job_i=0
echo "G2b lo: total_jobs=$total codes=${CODES[*]} readout=$READOUT no-calendar local h$HORIZON out=$OUT_ROOT"

for code in "${CODES[@]}"; do
  model="${MODEL_OF[$code]}"
  for meter in "${METERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      name="${code}_${model}_local_${READOUT}_h${HORIZON}_s${seed}_${meter}"
      out="$OUT_ROOT/$name"
      job_i=$((job_i + 1))
      if [[ -f "$out/summary.json" ]]; then
        skip_n=$((skip_n + 1))
        echo "[$job_i/$total] SKIP $name"
        continue
      fi
      echo "[$job_i/$total] RUN  $name"
      if "$PYTHON" "$ROOT/g_chung/chay.py" \
          --model "$model" --regime local --readout "$READOUT" \
          --horizon "$HORIZON" --seed "$seed" --meter "$meter" \
          --device "$DEVICE" --no-calendar --ablation "$code" \
          --out "$out" "${DEMO_FLAG[@]}"; then
        done_n=$((done_n + 1))
      else
        fail_n=$((fail_n + 1))
        echo "FAIL $name" >&2
      fi
    done
  done
done

echo "G2b lo finished: ran=$done_n skip=$skip_n fail=$fail_n total=$total"
[[ "$fail_n" -eq 0 ]]
