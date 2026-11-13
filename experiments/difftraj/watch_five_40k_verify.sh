#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 8>verification.lock
flock -n 8 || exit 0
[[ ! -f VERIFIED ]] || exit 0
trap 'rc=$?; if (( rc != 0 )); then date -u > VERIFY_FAILED; fi' EXIT
echo "$$" > verification.pid
while [[ ! -f COMPLETE ]]; do
  if [[ -f FAILED ]]; then exit 1; fi
  sleep 30
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/workspace/cnr-road-gaf-diffusion/.venv/bin/python -u verify_five_40k.py . ../difftraj-five-20k-localclock-v1
date -u >VERIFIED
