#!/usr/bin/env bash
# G-chung batch: 2 models x 2 regimes x 2 readouts x 4 horizons x 3 seeds
# pooled: 48 runs; local: 480 runs (10 meters). Skips if summary.json exists.
# Usage:
#   bash g_chung/lo.sh --python /path/to/python --device cuda [--demo] [--out-root results/g-chung]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="cuda"
OUT_ROOT="$ROOT/results/g-chung"
DEMO_FLAG=()
METERS=(M01 M02 M03 M04 M05 M06 M07 M08 M09 M10)
MODELS=(bilstm armambakan)
REGIMES=(pooled local)
READOUTS=(last full)
HORIZONS=(4 48 96 336)
SEEDS=(42 43 44)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --demo) DEMO_FLAG=(--demo); METERS=(DEMO0001 DEMO0002 DEMO0003 DEMO0004); shift ;;
    --meters) shift; METERS=(); while [[ $# -gt 0 && "$1" != --* ]]; do METERS+=("$1"); shift; done ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_ROOT"
total=0
done_n=0
skip_n=0
fail_n=0

# Count planned jobs
for model in "${MODELS[@]}"; do
  for regime in "${REGIMES[@]}"; do
    for readout in "${READOUTS[@]}"; do
      for h in "${HORIZONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if [[ "$regime" == pooled ]]; then
            total=$((total + 1))
          else
            total=$((total + ${#METERS[@]}))
          fi
        done
      done
    done
  done
done

echo "G-chung lo: total_jobs=$total device=$DEVICE python=$PYTHON out=$OUT_ROOT demo_flags=${DEMO_FLAG[*]:-}"

job_i=0
run_one() {
  local model="$1" regime="$2" readout="$3" h="$4" seed="$5" meter="${6:-}"
  local name="${model}_${regime}_${readout}_h${h}_s${seed}"
  local extra=()
  if [[ -n "$meter" ]]; then
    name="${name}_${meter}"
    extra=(--meter "$meter")
  fi
  local out="$OUT_ROOT/$name"
  job_i=$((job_i + 1))
  if [[ -f "$out/summary.json" ]]; then
    skip_n=$((skip_n + 1))
    echo "[$job_i/$total] SKIP $name"
    return 0
  fi
  echo "[$job_i/$total] RUN  $name"
  if "$PYTHON" "$ROOT/g_chung/chay.py" \
      --model "$model" --regime "$regime" --readout "$readout" \
      --horizon "$h" --seed "$seed" --device "$DEVICE" \
      --out "$out" "${DEMO_FLAG[@]}" "${extra[@]}"; then
    done_n=$((done_n + 1))
  else
    fail_n=$((fail_n + 1))
    echo "FAIL $name" >&2
    return 1
  fi
}

for model in "${MODELS[@]}"; do
  for regime in "${REGIMES[@]}"; do
    for readout in "${READOUTS[@]}"; do
      for h in "${HORIZONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if [[ "$regime" == pooled ]]; then
            run_one "$model" "$regime" "$readout" "$h" "$seed" || true
          else
            for meter in "${METERS[@]}"; do
              run_one "$model" "$regime" "$readout" "$h" "$seed" "$meter" || true
            done
          fi
        done
      done
    done
  done
done

echo "G-chung lo finished: ran=$done_n skip=$skip_n fail=$fail_n total=$total"
[[ "$fail_n" -eq 0 ]]
