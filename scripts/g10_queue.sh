#!/usr/bin/env bash
set -euo pipefail
for t in $(seq 1 180); do
   if grep -q 'Successfully installed' g10-data/install.log; then break; fi
   if grep -q '^ERROR:' g10-data/install.log; then cat g10-data/install.log; exit 1; fi
   sleep 5
done
export PYTHONPATH="g10-env:src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
binary=$(find g10-env -path '*/bin/sumo' -type f | sort | tail -1)
test -n "$binary"
export SUMO_HOME=$(dirname "$(dirname "$(realpath "$binary")")")
"$binary" --version | head -1
sim(){ .venv/bin/python scripts/g10_sumo.py "$1" "$binary" > "g10-data/sim$1.log" 2>&1; }
for batch in 260101 260105; do
 pids=()
 for seed in $(seq "$batch" "$((batch+3))"); do sim "$seed" & pids+=($!); done
 for pid in "${pids[@]}"; do wait "$pid"; done
done
.venv/bin/python scripts/g10_prepare.py
pids=()
for seed in 42 43 44; do CUDA_VISIBLE_DEVICES=$((seed-42)) .venv/bin/python scripts/g10_sample.py "$seed" > "g10-data/model$seed.log" 2>&1 & pids+=($!); done
for pid in "${pids[@]}"; do wait "$pid"; done
.venv/bin/python scripts/g10_eval.py
