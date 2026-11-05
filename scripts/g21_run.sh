#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:../g17-job/scripts:../g19-job/scripts:../scripts:../vendor/original_cnr/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>queue.lock
flock -n 9 || exit 1
trap 'echo "failed at $LINENO" > failed.txt' ERR
for i in $(seq 1 600); do
    test ! -s g21-data/validation/complete.json || break
    if grep -q Traceback prep.log; then exit 1; fi
    sleep 3
done
test -s g21-data/validation/complete.json
job(){
    local gpu=$1 arm=$2 seed=$3
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g21_train.py "$arm" "$seed" > "$arm-$seed.log" 2>&1
}
(run_pair(){ job "$1" physical "$2" && job "$1" temporal "$2"; }; run_pair 2 42) & c=$!
(job 3 temporal 43 && job 3 physical 43) & d=$!
job 0 original 42 & a=$!
job 1 original 43 & b=$!
failed=0
for pid in "$c" "$d" "$a" "$b"; do wait "$pid" || failed=1; done
test "$failed" = 0
echo done >training-complete.txt
