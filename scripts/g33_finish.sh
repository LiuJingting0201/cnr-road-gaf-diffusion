#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src:g10-env:scripts:../g32-job/scripts:../g31-job/scripts:../g30-job/scripts:../g29-job/scripts:../g28-job/scripts:../g23-job/scripts:../g22-job/scripts:../scripts:../vendor/original_cnr/src
exec 9>g33-data/evaluation.lock
flock -n 9 || exit 1
test ! -e g33-data/evaluation-started.txt
date -u > g33-data/evaluation-started.txt
trap 'date -u > g33-data/evaluation-queue-failed.txt' ERR
mkdir -p g33-data/pools g33-data/evaluation g33-data/selected
.venv/bin/python -u scripts/g33_evaluate.py prepare_maps --output g33-data/maps > g33-data/maps.log 2>&1
wait_component() {
    local begin=$SECONDS
    while [[ ! -f "$1/complete.json" ]]; do
        [[ ! -f "$1/failed.json" ]] || return 1
        (( SECONDS - begin < 14400 )) || return 1
        sleep 10
    done
}
pool() {
    local gpu=$1 arm=$2 seed=$3 family=$4 name=$5
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g33_evaluate.py generate \
        --model "g33-data/models/$arm-$seed" --seed "$seed" --family "$family" \
        --output "g33-data/pools/$name" > "g33-data/generate-$name.log" 2>&1
    .venv/bin/python -u scripts/g33_evaluate.py evaluate --generated "g33-data/pools/$name" \
        --output "g33-data/evaluation/$name" > "g33-data/evaluate-$name.log" 2>&1
}
slot() {
    local gpu=$1 arm=$2 seed=$3 name="$2-$3"
    wait_component "g33-data/models/$name"
    # Slots 2/3 must wait for their queued readers before reusing these GPUs.
    wait_component g33-data/teachers/spatial-42
    wait_component g33-data/teachers/speed-42
    pool "$gpu" "$arm" "$seed" diffusion "$name"
    if [[ "$arm" == native_0_25s ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g33_feedback.py score \
            --generated "g33-data/pools/$name" --output "g33-data/selected/$name" > "g33-data/score-$name.log" 2>&1
        .venv/bin/python -u scripts/g33_feedback.py evaluate --selected "g33-data/selected/$name" \
            --output "g33-data/evaluation/selected-$name" > "g33-data/selected-evaluate-$name.log" 2>&1
    else
        pool "$gpu" "$arm" "$seed" empirical "empirical-$seed"
    fi
}
wait_component g33-data/reference-extracted
slot 0 native_0_25s 42 & p0=$!
slot 1 native_0_25s 43 & p1=$!
slot 2 relative_257 42 & p2=$!
slot 3 relative_257 43 & p3=$!
bad=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || bad=1
done
.venv/bin/python - "$bad" <<'PY'
import json,sys,time
from pathlib import Path
names=[f'{arm}-{seed}' for arm in ['relative_257','native_0_25s','empirical'] for seed in [42,43]]+[f'selected-native_0_25s-{s}' for s in [42,43]]
missing=[n for n in names if not (Path('g33-data/evaluation')/n/'complete.json').exists() or json.loads((Path('g33-data/evaluation')/n/'complete.json').read_text()).get('complete') is not True]
ok=sys.argv[1]=='0' and not missing
Path('g33-data/evaluation-complete.json' if ok else 'g33-data/evaluation-failed.json').write_text(json.dumps({'complete':ok,'finished_unix':time.time(),'missing_components':missing,'in_loop_guidance_evaluated':False},indent=2)+'\n')
if not ok:raise SystemExit(1)
PY
