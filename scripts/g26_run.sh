#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:../g22-job/scripts:../g23-job/scripts:../g17-job/scripts:../g21-job/scripts:../g19-job/scripts:../scripts:../vendor/original_cnr/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=g26-data
if [[ "${CNR_G26_LAYOUT:-hierarchical}" == flat ]]; then root=g27-data; fi
exec 9>"$root/queue.lock"
flock -n 9 || exit 1
trap 'echo "failure at $LINENO" >"$root/failed.txt"' ERR
worker(){
    local arm=$1 gpu=$2
    for seed in 42 43; do
        CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g26_train.py "$arm" "$seed" >"$root/$arm-$seed.log" 2>&1 || return 1
    done
}
worker H 0 & a=$!
worker HM 1 & b=$!
worker HV 2 & c=$!
worker HG 3 & d=$!
bad=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || bad=1; done
test "$bad" = 0
echo complete >"$root/training-complete.txt"
