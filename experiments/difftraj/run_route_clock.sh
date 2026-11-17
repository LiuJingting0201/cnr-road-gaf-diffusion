#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 0
export PYTHONPATH="$PWD:$PWD/vendor" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
data=/workspace/cnr-experiments/difftraj-five-40k-v1
base=/workspace/cnr-experiments/pooled-velocity-class-v1
arm() (
    seed=$1
    gpu=$2
    auxgpu=$3
    dir="seed-$seed"
    features="$PWD/features-$seed"
    mkdir -p "$dir" "$features"
    args=(--base "$base" --data "$data" --out "$features" --seed "$seed")
    CUDA_VISIBLE_DEVICES=$gpu "$py" route_clock_features.py generate "${args[@]}" --part 0 > "$dir/geometry-0.log" 2>&1 & p0=$!
    CUDA_VISIBLE_DEVICES=$auxgpu "$py" route_clock_features.py generate "${args[@]}" --part 1 > "$dir/geometry-1.log" 2>&1 & p1=$!
    rc=0; wait "$p0" || rc=1; wait "$p1" || rc=1; (( rc==0 ))
    "$py" route_clock_features.py combine "${args[@]}"
    export CUDA_VISIBLE_DEVICES=$gpu
    "$py" five_class_train.py --data prepared --speed-dir prepared --arm C --mode all --out "$dir" --seed "$seed" --epochs 10 --batch-size 205 --microbatch 205 --route-features "$features" > "$dir/train.log" 2>&1
    "$py" route_clock_eval.py --data "$data" --base "$base" --run "$PWD" --seed "$seed" > "$dir/eval.log" 2>&1
)
arm 42 0 2 & p=$!
arm 43 1 3 & q=$!
rc=0; wait "$p" || rc=1; wait "$q" || rc=1
if (( rc )); then touch FAILED; exit 1; fi
touch COMPLETE
