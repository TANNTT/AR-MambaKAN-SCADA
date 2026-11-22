#!/usr/bin/env bash
# G0: chạy thử h48 seed 42, 2 mô hình x 2 cách đọc, pooled (10 công tơ) + local M01, M02.
# Mục đích: đo giờ thật, kiểm tái lập readout=last với CSV mốc, kiểm npz. ARIS viết 2026-11-21 theo lenh-hang-doi-de-xuat.sh của Cursor.
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
