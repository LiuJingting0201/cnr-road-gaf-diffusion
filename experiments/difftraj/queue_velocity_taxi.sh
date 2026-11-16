#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 0
test ! -e COMPLETE || exit 0
trap 'rc=$?; if (( rc )); then date -Iseconds > FAILED; fi' EXIT
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$PWD:$PWD/vendor" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
data=/workspace/cnr-experiments/difftraj-five-40k-v1
prior=/workspace/cnr-experiments/five-class-difftraj-200k-v1/runs
net=/workspace/cnr-experiments/gaf-inloop-v1/g27-job/g27-data/kit-check/assets/most.net.xml
"$py" velocity_difftraj.py prepare --data "$data" --out prepared > prepare.log 2>&1
fit() (
    set -e
    s=$1
    export CUDA_VISIBLE_DEVICES=$2
    run="velocity-$s"
    mkdir -p "$run"
    "$py" velocity_difftraj.py train --data prepared --out "$run" --seed "$s" --epochs 50 > "$run/train.log" 2>&1
    "$py" velocity_difftraj.py sample --data "$data" --prepared prepared --checkpoint "$run/final.pt" --representation velocity --output "$run/draws.npz" > "$run/sample.log" 2>&1
    for _ in $(seq 1 180); do
        [[ ! -f "coordinate-$s/FAILED" ]] || exit 2
        [[ ! -f "coordinate-$s/metrics.json" ]] || break
        sleep 2
    done
    "$py" score_velocity_pilot.py --data "$data" --draws "$run/draws.npz" --mode taxi --output "$run/metrics.json" --baseline "coordinate-$s/draws.npz" --net "$net" --map "$run/map.png" --per-trip "$run/trip-metrics.csv" > "$run/eval.log" 2>&1
)
coordinate() (
    set -e
    s=$1
    export CUDA_VISIBLE_DEVICES=$2
    run="coordinate-$s"
    mkdir -p "$run"
    trap 'rc=$?; if (( rc )); then date -Iseconds > "$run/FAILED"; fi' EXIT
    "$py" velocity_difftraj.py sample --data "$data" --prepared prepared --checkpoint "$prior/taxi-A-$s/final.pt" --representation coordinate --output "$run/draws.npz" > "$run/sample.log" 2>&1
    "$py" score_velocity_pilot.py --data "$data" --mode taxi --draws "$run/draws.npz" --output "$run/metrics.json" --per-trip "$run/trip-metrics.csv" > "$run/eval.log" 2>&1
)
fit 42 0 & a=$!
fit 43 1 & b=$!
coordinate 42 2 & c=$!
coordinate 43 3 & d=$!
rc=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || rc=1; done
(( rc == 0 ))
date -Iseconds > COMPLETE
