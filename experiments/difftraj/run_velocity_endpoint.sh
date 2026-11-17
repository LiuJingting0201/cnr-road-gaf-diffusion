#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 0
export PYTHONPATH="$PWD:$PWD/vendor" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
data=/workspace/cnr-experiments/difftraj-five-40k-v1
fit() (
    export CUDA_VISIBLE_DEVICES=$2
    dir="velocity-$1"
    mkdir -p "$dir"
    "$py" five_class_train.py --data prepared --speed-dir prepared --out "$dir" --seed "$1" --arm C --mode all --epochs 10 --batch-size 205 --microbatch 205 --endpoint-weight 0.1 > "$dir/train.log" 2>&1
    "$py" pooled_velocity_evaluate.py --data "$data" --prepared prepared --checkpoint "$dir/final.pt" --output "$dir/eval" --representation velocity > "$dir/eval.log" 2>&1
)
fit 42 0 & a=$!
fit 43 1 & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
(( rc == 0 )) || { touch FAILED; exit 1; }
"$py" velocity_label_balance.py --data "$data" --run "$PWD" --output report > balance.log
touch COMPLETE
