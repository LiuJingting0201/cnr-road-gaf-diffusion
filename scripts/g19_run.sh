#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:g10-env:../g17-job/scripts:../scripts:../vendor/original_cnr/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>g19-data/queue.lock
flock -n 9 || exit 1
trap 'echo "failed at $LINENO" >g19-data/train-failed.txt' ERR
run(){
    local arm=$1 seed=$2 gpu=$3
    if [[ "$arm" == road ]]; then
        for _ in $(seq 1 600); do
            [[ ! -s g19-data/field.json ]] || break
            if grep -q Traceback field.log; then return 1; fi
            sleep 2
        done
        test -s g19-data/field.json || return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g19_train.py "$arm" "$seed" >"g19-data/$arm-$seed.log" 2>&1
}
run plain 42 0 & a=$!
run plain 43 1 & b=$!
run road 42 2 & c=$!
run road 43 3 & d=$!
bad=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || bad=1; done
test "$bad" = 0
echo complete >g19-data/train-complete.txt
