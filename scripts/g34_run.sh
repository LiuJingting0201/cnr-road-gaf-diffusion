#!/usr/bin/env bash
# Finite G34 development diagnostic: no fitting, request change or new simulator.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src:g10-env:scripts:../g32-job/scripts:../g31-job/scripts:../g30-job/scripts:../g29-job/scripts:../g28-job/scripts:../g23-job/scripts:../g22-job/scripts:../scripts:../vendor/original_cnr/src
mkdir -p g34-data
exec 9>g34-data/queue.lock
flock -n 9 || exit 1
test ! -e g34-data/queue-started.json
trap 'date -u > g34-data/queue-failed.txt' ERR
.venv/bin/python - <<'PY'
import hashlib,json,time
from pathlib import Path
old=json.loads(Path('g33-data/evaluation-complete.json').read_text())
assert old['complete'] is True
names=['g34_fixed_duration.py','g34_summary.py','g34_run.sh','g33_model.py','g33_evaluate.py','g33_feedback.py']
Path('g34-data/queue-started.json').write_text(json.dumps({
 'started_unix':time.time(),'experiment_phase':'G34_fixed_request_duration_development_diagnostic',
 'gpu_slots':{'42':2,'43':3},'model_seeds':[42,43],'K':8,'DDIM_steps':100,
 'source_sha256':{n:hashlib.sha256((Path('scripts')/n).read_bytes()).hexdigest() for n in names},
 'scope':'Reuse inspected G33 development reference; no new training or simulation'
},indent=2)+'\n')
PY
.venv/bin/python -u scripts/g34_fixed_duration.py checks > g34-data/sampler-checks.log 2>&1
.venv/bin/python -u scripts/g34_summary.py checks > g34-data/summary-checks.log 2>&1
slot() {
    local gpu=$1 seed=$2 name="native_0_25s-$2"
    export CUDA_VISIBLE_DEVICES="$gpu"
    .venv/bin/python -u scripts/g34_fixed_duration.py generate \
        --maps g33-data/maps --model "g33-data/models/$name" \
        --g33-baseline "g33-data/pools/$name" --seed "$seed" \
        --output "g34-data/pools/$name" > "g34-data/generate-$name.log" 2>&1
    .venv/bin/python -u scripts/g33_evaluate.py evaluate \
        --generated "g34-data/pools/$name" --reference g33-data/reference-extracted \
        --maps g33-data/maps --training g33-data/direct-legs \
        --output "g34-data/evaluation/$name" > "g34-data/evaluate-$name.log" 2>&1
    .venv/bin/python -u scripts/g33_feedback.py score \
        --generated "g34-data/pools/$name" --teachers g33-data/teachers --cache g33-data/gaf-cache \
        --output "g34-data/selected/$name" > "g34-data/score-$name.log" 2>&1
    .venv/bin/python -u scripts/g33_feedback.py evaluate \
        --selected "g34-data/selected/$name" --reference g33-data/reference-extracted \
        --maps g33-data/maps --training g33-data/direct-legs \
        --output "g34-data/evaluation/selected-$name" > "g34-data/selected-evaluate-$name.log" 2>&1
}
slot 2 42 & p0=$!
slot 3 43 & p1=$!
bad=0
for pid in "$p0" "$p1"; do
    wait "$pid" || bad=1
done
if [[ "$bad" != 0 ]]; then
    date -u > g34-data/queue-failed.txt
    exit 1
fi
.venv/bin/python -u scripts/g34_summary.py summarize \
    --root g34-data --g33-root g33-data --output g34-data/report > g34-data/summary.log 2>&1
.venv/bin/python - <<'PY'
import hashlib,json,time
from pathlib import Path
report=Path('g34-data/report/summary.json')
assert json.loads(report.read_text())['complete'] is True
Path('g34-data/queue-complete.json').write_text(json.dumps({
 'complete':True,'finished_unix':time.time(),
 'experiment_phase':'G34_fixed_request_duration_development_diagnostic',
 'expected_evaluations':10,'summary_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
 'in_loop_guidance_evaluated':False,'new_blind_reference':False
},indent=2)+'\n')
PY
