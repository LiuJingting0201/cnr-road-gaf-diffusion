#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:scripts:../scripts:../vendor/original_cnr/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>g29-data/grid-queue.lock
flock -n 9 || exit 1
trap 'echo "failure at $LINENO" >g29-data/grid-failed.txt' ERR
for (( attempt=0; attempt<240; attempt++ )); do
    [[ -f g29-data/training-complete.txt ]] && break
    [[ -f g29-data/failed.txt ]] && exit 1
    sleep 10
done
test -f g29-data/training-complete.txt
.venv/bin/python scripts/g29_quantized.py >g29-data/grid-prepare.log 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g29_train.py cnr full grid_xy 42 >g29-data/cnr-full-grid_xy-42.log 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/g29_train.py cnr full grid_xy 43 >g29-data/cnr-full-grid_xy-43.log 2>&1 & b=$!
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/g29_train.py sumo full grid_xy 42 >g29-data/sumo-full-grid_xy-42.log 2>&1 & c=$!
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u scripts/g29_train.py sumo full grid_xy 43 >g29-data/sumo-full-grid_xy-43.log 2>&1 & d=$!
failed=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || failed=1; done
test "$failed" = 0
echo complete >g29-data/grid-training-complete.txt
