#!/usr/bin/env bash
# One bounded interface audit, after the output-only replay. No model fitting.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src:g10-env:scripts:../g30-job/scripts:../g29-job/scripts:../scripts:../vendor/original_cnr/src
mkdir -p g31-data
exec 9>g31-data/interface.lock
flock -n 9 || exit 1
test ! -e g31-data/interface-started.txt
date -u +%FT%TZ >g31-data/interface-started.txt
trap 'echo "failed line $LINENO" >g31-data/interface-failed.txt' ERR
comparison=g31-data/compare/pilot-300001-attempt2
deadline=$((SECONDS+900))
while [[ ! -f "$comparison/comparison-complete.json" ]]; do
    test ! -f "$comparison/comparison-failed.json"
    test "$SECONDS" -lt "$deadline"
    sleep 5
done
.venv/bin/python -u scripts/g31_legs.py --comparison "$comparison/comparison-complete.json" >g31-data/legs.log 2>&1
.venv/bin/python -u scripts/g31_paths.py >g31-data/paths.log 2>&1
.venv/bin/python -u scripts/g31_roundtrip.py >g31-data/roundtrip.log 2>&1
echo complete >g31-data/interface-complete.txt
