#!/usr/bin/env bash
# Each seed trains once; both checkpoint rules are evaluated on the original development requests.
set -euo pipefail
cd "$(dirname "$0")"
exec 9>training.lock
flock -n 9 || exit 0
[[ ! -f COMPLETE ]] || exit 0
root="$PWD"
base=/workspace/cnr-experiments/taxi-repro-debug-v1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$root:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DIFFTRAJ_GLOBAL_BATCH=440
trap 'rc=$?; if (( rc != 0 )); then date -u > "$root/FAILED"; fi' EXIT
if [[ -f FAILED ]]; then mv FAILED "FAILED.previous.$(date -u +%s)"; fi
for f in prepared.npz observations.npz; do
    [[ -e "$f" ]] || ln -s "$base/20k/$f" "$f"
done
date -u > STARTED
"$py" -c 'import numpy as np; z=np.load("prepared.npz"); s=z["split"]; assert (s<8).sum()==20000 and (s==8).sum()==1000 and not (s==9).any()'
fit() (
    set -e
    seed="$1"
    export PORTO_TRAIN_SEED="$seed" CUDA_VISIBLE_DEVICES="$2"
    mkdir -p "$seed"
    cd "$seed"
    for f in prepared.npz observations.npz; do [[ -e "$f" ]] || ln -s "../$f" "$f"; done
    if [[ ! -f TRAIN_COMPLETE ]]; then
        resume=()
        if [[ -f last.pt ]]; then resume=(--resume); fi
        "$py" -m torch.distributed.run --standalone --nproc_per_node=1 ../porto.py train --large-microbatch --stabilized "${resume[@]}" >> train.log 2>&1
        "$py" -c 'import torch; assert torch.load("checkpoint.pt", map_location="cpu", weights_only=True)["epoch"] == 200'
        touch TRAIN_COMPLETE
    fi
    for method in final best; do
        mkdir -p "$method"
        (
            cd "$method"
            if [[ -f COMPLETE ]]; then exit 0; fi
            for f in prepared.npz observations.npz checkpoint.pt best.pt; do [[ -e "$f" ]] || ln -s "../$f" "$f"; done
            flag=()
            if [[ "$method" == best ]]; then flag=(--best); fi
            "$py" ../../porto.py score --development --all-validation --save-draws "${flag[@]}" >score.log 2>&1
            "$py" -c 'import numpy as np,sys; np.testing.assert_array_equal(np.load("draws.npz")["ids"],np.load(sys.argv[1])["ids"])' "$base/20k/$seed/draws.npz"
            "$py" ../../taxi_od_diagnostic.py . --common-pack "$base/10k" --check development.json --output od-diagnostic.json >audit.log 2>&1
            touch COMPLETE
        )
    done
    touch COMPLETE
)
fit 42 2 & pid42=$!
fit 43 3 & pid43=$!
status=0
wait "$pid42" || status=1
wait "$pid43" || status=1
if (( status )); then exit 1; fi
date -u > COMPLETE
