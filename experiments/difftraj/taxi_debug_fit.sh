#!/usr/bin/env bash
set -euo pipefail
root=$PWD; arm=$1; seed=$2; gpu=$3
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$root:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1 PORTO_TRAIN_SEED=$seed CUDA_VISIBLE_DEVICES=$gpu
export DIFFTRAJ_GLOBAL_BATCH=220
if [[ "$arm" == 20k ]]; then export DIFFTRAJ_GLOBAL_BATCH=440;fi
mkdir -p "$arm"; cd "$arm"
(flock -x 9; if ! test -f READY; then "$py" ../taxi_debug_data.py "$arm"; "$py" ../porto.py prepare >split.json; touch READY; fi) 9>prepare.lock
mkdir "$seed"; cd "$seed"
ln -s ../prepared.npz ../observations.npz .
"$py" -m torch.distributed.run --standalone --nproc_per_node=1 ../../porto.py train --large-microbatch >train.log 2>&1
if [[ "$arm" == porto ]]; then "$py" ../../porto.py score --matched-reserve --save-draws >score.log 2>&1; else "$py" ../../porto.py score --development --all-validation --save-draws >score.log 2>&1;fi
touch COMPLETE
