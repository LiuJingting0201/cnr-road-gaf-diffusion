#!/usr/bin/env bash
set -euo pipefail
exec 9>review.lock
flock -n 9 || exit 1
trap 'touch FAILED' ERR
root=$PWD
old=/workspace/cnr-experiments/porto-stabilized-v1
export PYTHONPATH="$root:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
ln -sfn "$old/shared-bins.npz" shared-bins.npz
score(){
    local arm=$1 cohort=$2 seed=$3 gpu=$4
    local name="$arm-on-$cohort"
    local dir="$root/$seed-$name"
    local weights="$old/$arm"
    if [[ $seed == 43 ]]; then weights="$root/fit43"; fi
    mkdir -p "$dir"
    (cd "$dir"
    for file in observations.npz prepared.npz; do ln -sfn "$old/$name/$file" "$file"; done
    for file in best.pt checkpoint.pt; do ln -sfn "$weights/$file" "$file"; done
    CUDA_VISIBLE_DEVICES=$gpu "$py" ../porto.py score --best --cutoff-comparison --save-draws > score.log 2>&1)
}
pids=()
gpu=0
for arm in long mixed; do
    for cohort in long mixed; do
        score "$arm" "$cohort" 42 "$gpu" & pids+=($!)
        gpu=$((gpu+1))
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done
touch REPLAY_COMPLETE
mkdir -p fit43
(cd fit43
ln -sfn "$old/mixed/prepared.npz" prepared.npz
PORTO_TRAIN_SEED=43 "$py" -m torch.distributed.run --standalone --nproc_per_node=4 ../porto.py train --large-microbatch --stabilized > train.log 2>&1)
score mixed mixed 43 0 & first=$!
score mixed long 43 1 & second=$!
wait "$first"
wait "$second"
touch COMPLETE
