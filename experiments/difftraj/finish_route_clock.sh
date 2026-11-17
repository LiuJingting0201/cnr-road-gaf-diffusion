#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>finish.lock
flock -n 9 || exit 0
while ! test -f COMPLETE; do test ! -f FAILED || exit 1; sleep 12; done
export PYTHONPATH="$PWD:$PWD/vendor" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
"$py" route_clock_eval.py --data ../difftraj-five-40k-v1 --base ../pooled-velocity-class-v1 --run "$PWD" --seed 43 --route-seed 42
for s in 42 43; do
    cp "seed-$s/config.json" "report/config-$s.json"
    cp "seed-$s/epochs.jsonl" "report/loss-$s.jsonl"
done
touch report/FINISHED
