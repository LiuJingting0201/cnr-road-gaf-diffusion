#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>comparison.lock
flock -n 9 || exit 0
[[ ! -f COMPARISON_COMPLETE ]] || exit 0
trap 'rc=$?; if (( rc != 0 )); then echo "status $rc" >COMPARISON_FAILED; fi' EXIT
exp=/workspace/cnr-experiments
old="$exp/taxi-repro-debug-v1"
new="$exp/taxi20k-stabilized-v1"
while [[ ! -f "$new/COMPLETE" ]]; do
    if [[ -f "$new/FAILED" ]]; then echo 'Training queue reported failure' >&2; exit 1; fi
    sleep 30
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$new:$exp/difftraj-nov11"
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
"$py" taxi_stability_comparison.py "$old/20k" "$new" "$old/10k" \
 "$exp/gaf-inloop-v1/g30-job/g30-data/source/scenario/in/most.net.xml" output
touch COMPARISON_COMPLETE
