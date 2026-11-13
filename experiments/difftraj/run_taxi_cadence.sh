#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>cadence.lock
flock -n 9 || exit 0
[[ ! -f CADENCE_COMPLETE ]] || exit 0
trap 'status=$?; if (( status != 0 )); then echo "$status" >CADENCE_FAILED; fi' EXIT
base=/workspace/cnr-experiments/taxi-repro-debug-v1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ -e one ]] || ln -s "$base/10k" one
for filename in porto.py taxi_od_diagnostic.py; do
    [[ -e $filename ]] || cp "$base/$filename" "$filename"
done
mkdir -p fifteen
if [[ ! -f fifteen/READY ]]; then
    "$py" taxi_cadence.py pack
    (cd fifteen && "$py" ../porto.py prepare >split.json)
    "$py" taxi_cadence.py align
    touch fifteen/READY
fi
fit() (
    seed=$1
    export CUDA_VISIBLE_DEVICES=$2 PORTO_TRAIN_SEED=$seed DIFFTRAJ_GLOBAL_BATCH=220
    mkdir -p "fifteen/$seed"
    cd "fifteen/$seed"
    [[ ! -f COMPLETE ]] || exit 0
    for f in prepared.npz observations.npz; do [[ -e $f ]] || ln -s "../$f" "$f"; done
    "$py" -m torch.distributed.run --standalone --nproc_per_node=1 ../../porto.py train --large-microbatch >train.log 2>&1
    "$py" ../../porto.py score --development --all-validation --save-draws >score.log 2>&1
    touch COMPLETE
)
fit 42 2 & a=$!
fit 43 3 & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
(( rc == 0 )) || exit 1
"$py" taxi_cadence.py summary
touch CADENCE_COMPLETE
