#!/usr/bin/env bash
# Reproduce from the existing frozen experiment assets and SUMO 1.28.0.
# Set CNR_SUMO_BIN to the SUMO binary, and include sumolib in PYTHONPATH.
set -euo pipefail
: "${CNR_SUMO_BIN:?Set the path to SUMO 1.28.0}"
export CNR_RUN_ROOT=g11-data CNR_FIRST_SEED=260201 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir "$CNR_RUN_ROOT"
for s in $(seq 260201 260208); do
  .venv/bin/python scripts/g10_sumo.py "$s" "$CNR_SUMO_BIN"
done
.venv/bin/python scripts/g10_prepare.py
pids=()
for s in 42 43 44; do
 CUDA_VISIBLE_DEVICES=$((s-42)) .venv/bin/python scripts/g11_sample.py "$s" > "g11-data/model$s.log" 2>&1 &
 pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
.venv/bin/python scripts/g11_eval.py
.venv/bin/python scripts/g11_groups.py
