#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:../g22-job/scripts:../g23-job/scripts:../g17-job/scripts:../g21-job/scripts:../g19-job/scripts:../scripts:../vendor/original_cnr/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p g28-data/models
exec 9>g28-data/diffusion-queue.lock
flock -n 9 || exit 1
trap 'echo "failure at line $LINENO" >g28-data/diffusion-failed.txt' ERR

worker() {
    local gpu=$1
    shift
    local task arm seed
    for task in "$@"; do
        arm=${task%-*}
        seed=${task#*-}
        CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g28_train.py "$arm" "$seed" \
            >>"g28-data/models/$task.log" 2>&1 || return 1
    done
}

for view in spatial speed; do
    test -f "g28-data/teachers/$view-42/complete.json"
done
worker 0 B-42 S-42 SV-42 & first=$!
worker 1 P-42 V-42 SH-42 & second=$!
worker 2 B-43 S-43 SV-43 & third=$!
worker 3 P-43 V-43 SH-43 & fourth=$!
failed=0
for worker_pid in "$first" "$second" "$third" "$fourth"; do
    wait "$worker_pid" || failed=1
done
test "$failed" = 0
echo complete >g28-data/diffusion-complete.txt
