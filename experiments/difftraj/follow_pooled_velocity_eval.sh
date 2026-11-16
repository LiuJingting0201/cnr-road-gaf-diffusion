#!/usr/bin/env bash
set -eu
run=$(cd "$(dirname "$0")" && pwd)
cd "$run"
exec 9>evaluation.lock
flock -n 9 || exit 0
data=${DATA_DIR:-/workspace/cnr-experiments/difftraj-five-40k-v1}
python=${PYTHON_BIN:-/workspace/cnr-road-gaf-diffusion/.venv/bin/python}
export PYTHONPATH="$run:$run/vendor" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
score() {
    local rep=$1 seed=$2 gpu=$3
    local dir="$run/$rep-$seed"
    if [[ -s "$dir/eval/metrics.json" && -f "$dir/EVAL_COMPLETE" ]]; then return 0; fi
    while [[ ! -f "$dir/TRAIN_COMPLETE" ]]; do
        if ! kill -0 "$(cat "$dir/pid")" 2>/dev/null; then return 1; fi
        sleep 12
    done
    CUDA_VISIBLE_DEVICES=$gpu "$python" pooled_velocity_evaluate.py \
        --data "$data" --prepared "$run/prepared" --checkpoint "$dir/final.pt" \
        --representation "$rep" --output "$dir/eval" > "$dir/eval.log" 2>&1 || return 1
    test -s "$dir/eval/metrics.json" && touch "$dir/EVAL_COMPLETE"
}
pids=()
for spec in 'velocity 42 0' 'velocity 43 1' 'coordinate 42 2' 'coordinate 43 3'; do
    read -r rep seed gpu <<< "$spec"
    score "$rep" "$seed" "$gpu" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then touch EVAL_FAILED; exit 1; fi
if [[ -f EVAL_FAILED ]]; then mv EVAL_FAILED "EVAL_FAILED.previous.$(date +%s)"; fi
touch COMPLETE
