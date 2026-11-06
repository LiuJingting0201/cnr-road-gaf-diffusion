#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src:g10-env:scripts:../g32-job/scripts:../g31-job/scripts:../g30-job/scripts:../g29-job/scripts:../g28-job/scripts:../g23-job/scripts:../g22-job/scripts:../scripts:../vendor/original_cnr/src
exec 9>g33-data/postprocess.lock
flock -n 9 || exit 1
test ! -e g33-data/postprocess-started.txt
date -u > g33-data/postprocess-started.txt
trap 'date -u > g33-data/postprocess-failed.txt' ERR
begin=$SECONDS
while [[ ! -f g33-data/evaluation-complete.json ]]; do
    test ! -f g33-data/evaluation-failed.json
    test ! -f g33-data/evaluation-queue-failed.txt
    (( SECONDS - begin < 14400 ))
    sleep 10
done
.venv/bin/python -u scripts/g33_report.py --output g33-data/report
.venv/bin/python -u scripts/g33_ranking_diagnostic.py \
    --selected g33-data/selected/native_0_25s-42 g33-data/selected/native_0_25s-43 \
    --output g33-data/gaf-ranking-diagnostic.json
.venv/bin/python -u scripts/g33_plots.py --output g33-data/figures
.venv/bin/python - <<'PY'
import hashlib,json,time
from pathlib import Path
names=['g33_report.py','g33_ranking_diagnostic.py','g33_plots.py','g33_postprocess.sh']
Path('g33-data/postprocess-complete.json').write_text(json.dumps({
    'complete':True,'finished_unix':time.time(),
    'source_sha256':{n:hashlib.sha256((Path('scripts')/n).read_bytes()).hexdigest() for n in names},
    'scope':'read-only aggregation and figures; scientific evaluation unchanged'
},indent=2)+'\n')
PY
