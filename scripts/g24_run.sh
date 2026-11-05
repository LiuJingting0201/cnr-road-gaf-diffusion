#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:../g23-job/scripts:../g22-job/scripts:../g17-job/scripts:../g21-job/scripts:../g19-job/scripts:../scripts:../vendor/original_cnr/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>g24-data/queue.lock
flock -n 9 || exit 1
trap 'echo "failed at $LINENO" >g24-data/failed.txt' ERR
for i in $(seq 1 300); do test ! -s g24-data/references.npz || break; sleep 2; done
test -s g24-data/references.npz
job(){ CUDA_VISIBLE_DEVICES=$1 .venv/bin/python -u scripts/g24_sample.py "$2" "$3" >g24-data/"$2-$3".log 2>&1; }
(job 0 plain 42 && job 0 guided 42) & a=$!
(job 1 plain 43 && job 1 guided 43) & b=$!
job 2 permuted 42 & c=$!
job 3 permuted 43 & d=$!
failed=0
for id in "$a" "$b" "$c" "$d"; do wait "$id" || failed=1; done
test "$failed" = 0
echo done > g24-data/sample-complete.txt
