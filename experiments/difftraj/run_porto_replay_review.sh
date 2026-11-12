#!/usr/bin/env bash
set -eu
exec 8>review.lock
flock -n 8 || exit 1
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
until test -f COMPLETE; do kill -0 "$1" || exit 1; sleep 30; done
for ck in raw-140 raw-147 raw-165 clip-147 clip-165; do
    mkdir -p "dev-$ck"
    (cd "dev-$ck"; ln -sf ../"$ck".pt checkpoint.pt; for f in prepared.npz observations.npz; do ln -sf ../"$f" "$f"; done; CUDA_VISIBLE_DEVICES=0 "$py" ../porto.py score --development > eval.log)
done
touch REVIEW_COMPLETE
