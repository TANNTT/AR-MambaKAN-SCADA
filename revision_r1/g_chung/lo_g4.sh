#!/usr/bin/env bash
# G4: rolling-origin F2 and F3 (PA-3 bỏ F1).
# F2 test 2022-01-01..2022-04-30; F3 test 2022-05-01..2022-08-31.
# Train = data before val; val = last 1/8 of pre-test (val/train≈1/7).
# BiLSTM + AR-MambaKAN, readout last, pooled + local, h48 + h336, seeds 42/43/44.
# Skips if summary.json exists. Output: results/g4/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="cuda"
OUT_ROOT="$ROOT/results/g4"
DEMO_FLAG=()
METERS=(M01 M02 M03 M04 M05 M06 M07 M08 M09 M10)
MODELS=(bilstm armambakan)
REGIMES=(pooled local)
FOLDS=(F2 F3)
HORIZONS=(48 336)
SEEDS=(42 43 44)
READOUT=last

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --demo) DEMO_FLAG=(--demo); METERS=(DEMO0001 DEMO0002); echo "WARN: demo dates are 2023-only; F2/F3 need real meters" >&2; shift ;;
    --meters) shift; METERS=(); while [[ $# -gt 0 && "$1" != --* ]]; do METERS+=("$1"); shift; done ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_ROOT"
total=0
for model in "${MODELS[@]}"; do
  for regime in "${REGIMES[@]}"; do
    for fold in "${FOLDS[@]}"; do
      for h in "${HORIZONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if [[ "$regime" == pooled ]]; then total=$((total + 1)); else total=$((total + ${#METERS[@]})); fi
        done
      done
    done
  done
done

done_n=0; skip_n=0; fail_n=0; job_i=0
echo "G4 lo: total_jobs=$total folds=${FOLDS[*]} out=$OUT_ROOT"

run_one() {
  local model="$1" regime="$2" fold="$3" h="$4" seed="$5" meter="${6:-}"
  local name="${model}_${regime}_${READOUT}_h${h}_s${seed}"
  local extra=()
  if [[ -n "$meter" ]]; then
    name="${name}_${meter}"
    extra=(--meter "$meter")
  fi
  name="${name}_${fold}"
  local out="$OUT_ROOT/$name"
  job_i=$((job_i + 1))
  if [[ -f "$out/summary.json" ]]; then
    skip_n=$((skip_n + 1))
    echo "[$job_i/$total] SKIP $name"
    return 0
  fi
  echo "[$job_i/$total] RUN  $name"
  if "$PYTHON" "$ROOT/g_chung/chay.py" \
      --model "$model" --regime "$regime" --readout "$READOUT" \
      --horizon "$h" --seed "$seed" --fold "$fold" --check-leakage \
      --device "$DEVICE" --out "$out" "${DEMO_FLAG[@]}" "${extra[@]}"; then
    done_n=$((done_n + 1))
  else
    fail_n=$((fail_n + 1))
    echo "FAIL $name" >&2
    return 1
  fi
}

for model in "${MODELS[@]}"; do
  for regime in "${REGIMES[@]}"; do
    for fold in "${FOLDS[@]}"; do
      for h in "${HORIZONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          if [[ "$regime" == pooled ]]; then
            run_one "$model" "$regime" "$fold" "$h" "$seed" || true
          else
            for meter in "${METERS[@]}"; do
              run_one "$model" "$regime" "$fold" "$h" "$seed" "$meter" || true
            done
          fi
        done
      done
    done
  done
done

echo "G4 lo finished: ran=$done_n skip=$skip_n fail=$fail_n total=$total"
[[ "$fail_n" -eq 0 ]]
