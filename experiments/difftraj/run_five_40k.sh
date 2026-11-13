#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 0
[[ ! -f COMPLETE ]] || exit 0
if [[ -f FAILED ]]; then mv FAILED "FAILED.previous.$(date -u +%s)"; fi
trap 'status=$?; if (( status != 0 )); then date -u >FAILED; fi' EXIT
export PYTHONPATH="$PWD:/workspace/cnr-experiments/gaf-inloop-v1/g10-env" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
date -u > STARTED
echo "$$" > queue.pid
"$py" -u expand_five_40k.py --workers 12
