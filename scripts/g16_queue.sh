#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:g10-env
base=${CNR_G16_DATA_DIR:-g16-data}
mkdir -p "$base"
exec 9>"$base/queue.lock"
flock -n 9 || exit 1
trap 'echo "failure at line $LINENO" > "$base/train-failed.txt"' ERR
for tick in $(seq 1 180); do
    test ! -e "$base/data-complete.json" || break
    if [[ -f prep.log ]] && grep -q Traceback prep.log; then cat prep.log; exit 1; fi
    sleep 5
done
test -s "$base/data-complete.json"
worker() {
    local gpu=$1 seed=$2
    shift 2
    for arm in "$@"; do
        CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/g16_train.py "$arm" "$seed" > "$base/$arm-$seed.log" 2>&1 || return 1
    done
}
worker 0 42 A C & a=$!
worker 1 42 B D & b=$!
worker 2 43 A C & c=$!
worker 3 43 B D & d=$!
failed=0
for job in "$a" "$b" "$c" "$d"; do wait "$job" || failed=1; done
test "$failed" = 0
echo done >"$base/train-complete.txt"
