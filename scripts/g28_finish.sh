#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:scripts:../g22-job/scripts:../g23-job/scripts:../g17-job/scripts:../g21-job/scripts:../g19-job/scripts:../scripts:../vendor/original_cnr/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec 8>g28-data/finish.lock
flock -n 8 || exit 1
trap 'echo "failure at $LINENO" >g28-data/finish-failed.txt' ERR
for (( attempt=0; attempt<240; attempt++ )); do
    if [[ -f g28-data/teachers-complete.txt && -f g28-data/roundtrip.json && -f g28-data/ready.txt ]]; then break; fi
    sleep 15
done
test -f g28-data/teachers-complete.txt
test -f g28-data/roundtrip.json
test -f g28-data/ready.txt
.venv/bin/python -u scripts/g28_probes.py >g28-data/probes.log 2>&1
bash scripts/g28_run.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g28_eval.py >g28-data/evaluation.log 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g28_independent_check.py >g28-data/checks.log 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g28_fresh_probes.py >g28-data/fresh-probes.log 2>&1
echo complete >g28-data/complete.txt
