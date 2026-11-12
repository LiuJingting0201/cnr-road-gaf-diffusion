#!/usr/bin/env bash
set -eu
root=$PWD
base=/workspace/cnr-experiments/beijing-geolife-difftraj
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11 OMP_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
jobs=()
for arm in B V C VC; do
(
    for seed in 42 43; do
        dir="$root/bj-$arm-$seed"
        mkdir -p "$dir"
        cd "$dir"
        ln -sfn "$base/pilot.npz" .
        for f in "$base/seed-$seed/$arm-8000.pt" "$base/seed-$seed/$arm-8000-class.pt"; do ln -sfn "$f" .; done
        CUDA_VISIBLE_DEVICES=0 DIFFTRAJ_TRAINING_SEED=$seed "$py" "$root/pilot.py" "$arm" 8000 --evaluate --all-validation --save-draws > replay.log 2>&1
        "$py" - "$base/seed-$seed/$arm-8000.json" "$arm-8000.json" <<'PY'
import json,sys,math
a,b=[json.load(open(p)) for p in sys.argv[1:]]
assert all(math.isclose(a[k],b[k],rel_tol=1e-5,abs_tol=1e-7) for k in ('speed_w1_mps','length_w1_m','density_js'))
PY
    done
) & jobs+=($!)
done
for pid in "${jobs[@]}"; do wait "$pid"; done
touch BJ_DRAWS_COMPLETE
