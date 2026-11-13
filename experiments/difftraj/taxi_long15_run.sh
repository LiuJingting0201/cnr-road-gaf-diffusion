#!/usr/bin/env bash
set -euo pipefail
exec 9>queue.lock
flock -n 9 || exit 1
trap 'touch FAILED' ERR
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11:/workspace/cnr-experiments/gaf-inloop-v1/g10-env"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DIFFTRAJ_GLOBAL_BATCH=220
if [[ ! -f DATA_READY ]]; then
  "$py" -u taxi_long15.py
  "$py" porto.py prepare >split.json
  touch DATA_READY
fi
fit() (
  set -e
  seed=$1; export PORTO_TRAIN_SEED=$seed CUDA_VISIBLE_DEVICES=$2
  mkdir -p "$seed"; cd "$seed"
  [[ -f COMPLETE ]] && return 0
  ln -sfn ../prepared.npz prepared.npz; ln -sfn ../observations.npz observations.npz
  "$py" -m torch.distributed.run --standalone --nproc_per_node=1 ../porto.py train --large-microbatch >train.log 2>&1
  "$py" ../porto.py score --development --all-validation --save-draws >development.log 2>&1
  mv draws.npz development-draws.npz
  "$py" ../porto.py score --all-holdout --save-draws >score.log 2>&1
  touch COMPLETE
)
fit 42 0 & a=$!
fit 43 1 & b=$!
s=0; wait "$a" || s=1; wait "$b" || s=1; test "$s" -eq 0
touch COMPLETE
