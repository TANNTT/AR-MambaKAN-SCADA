#!/usr/bin/env bash
# G0: trial run h48 seed 42, 2 models x 2 readouts, pooled (10 meters) + local M01, M02.
# Purpose: measure real runtime, check that readout=last reproduces the reference CSV, and check the npz outputs. Written 2026-11-21 following the proposed queue-command script.
set -euo pipefail
PAPER="${SCADA_PAPER_ROOT:?Set SCADA_PAPER_ROOT to the revision_r1 directory}"
PY="${SCADA_PYTHON:?Set SCADA_PYTHON to the project Python interpreter}"
OUT="$PAPER/results/g-chung"
for m in bilstm armambakan; do
  for r in last full; do
    t0=$(date +%s)
    "$PY" "$PAPER/g_chung/chay.py" --model "$m" --regime pooled --readout "$r" --horizon 48 --seed 42 \
      --device cuda --out "$OUT/${m}_pooled_${r}_h48_s42"
    echo "G0 pooled $m $r xong sau $(( $(date +%s) - t0 )) giay"
    for meter in M01 M02; do
      t0=$(date +%s)
      "$PY" "$PAPER/g_chung/chay.py" --model "$m" --regime local --readout "$r" --horizon 48 --seed 42 \
        --meter "$meter" --device cuda --out "$OUT/${m}_local_${r}_h48_s42_${meter}"
      echo "G0 local $m $r $meter xong sau $(( $(date +%s) - t0 )) giay"
    done
  done
done
