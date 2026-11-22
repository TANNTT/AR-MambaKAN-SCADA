#!/usr/bin/env bash
# G3: target-fraction matched budget (R1.3).
# Protocol: AR-MambaKAN, readout last, h48, seeds 42/43/44, calendar ON (as matched_budget).
# p in {0.25, 0.5, 0.75}; one model per target meter; eval on target test.
# Sampling: rng.choice without replacement (common_pooled.py:141-143).
# Skips if summary.json exists. Output: results/g3/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="cuda"
OUT_ROOT="$ROOT/results/g3"
DEMO_FLAG=()
METERS=(M01 M02 M03 M04 M05 M06 M07 M08 M09 M10)
PS=(0.25 0.5 0.75)
SEEDS=(42 43 44)
HORIZON=48
MODEL=armambakan
READOUT=last

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
total=$(( ${#METERS[@]} * ${#PS[@]} * ${#SEEDS[@]} ))
done_n=0; skip_n=0; fail_n=0; job_i=0
echo "G3 lo: total_jobs=$total model=$MODEL h$HORIZON p=${PS[*]} out=$OUT_ROOT"

for target in "${METERS[@]}"; do
  for p in "${PS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      name="${MODEL}_matched_${READOUT}_h${HORIZON}_s${seed}_${target}_p${p}"
      out="$OUT_ROOT/$name"
      job_i=$((job_i + 1))
      if [[ -f "$out/summary.json" ]]; then
        skip_n=$((skip_n + 1))
        echo "[$job_i/$total] SKIP $name"
        continue
      fi
      echo "[$job_i/$total] RUN  $name"
      if "$PYTHON" "$ROOT/g_chung/chay.py" \
          --model "$MODEL" --regime matched --readout "$READOUT" \
          --horizon "$HORIZON" --seed "$seed" \
          --target-meter "$target" --target-frac "$p" \
          --meters "${METERS[@]}" \
          --device "$DEVICE" --out "$out" "${DEMO_FLAG[@]}"; then
        done_n=$((done_n + 1))
      else
        fail_n=$((fail_n + 1))
        echo "FAIL $name" >&2
      fi
    done
  done
done

echo "G3 lo finished: ran=$done_n skip=$skip_n fail=$fail_n total=$total"
[[ "$fail_n" -eq 0 ]]
