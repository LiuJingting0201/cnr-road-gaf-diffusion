#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:scripts:../g22-job/scripts:../g23-job/scripts:../g17-job/scripts:../g21-job/scripts:../g19-job/scripts:../scripts:../vendor/original_cnr/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 9>g28-data/teachers.lock
flock -n 9 || exit 1
trap 'echo "failure at $LINENO" >g28-data/teachers-failed.txt' ERR
# Wait only for the already running, finite preparation job.
while [[ ! -f g28-data/cache/prepared.json ]]; do
    pgrep -f '[p]ython -u scripts/g28_prepare.py' >/dev/null || { echo 'Preparation stopped without marker'; exit 1; }
    sleep 15
done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g28_teachers.py spatial 42 >g28-data/spatial-42.log 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/g28_teachers.py speed 42 >g28-data/speed-42.log 2>&1 & b=$!
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/g28_teachers.py spatial 43 >g28-data/spatial-43.log 2>&1 & c=$!
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u scripts/g28_teachers.py speed 43 >g28-data/speed-43.log 2>&1 & d=$!
bad=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || bad=1; done
test "$bad" = 0
echo complete >g28-data/teachers-complete.txt
