#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9>finish.lock
flock -n 9 || exit 0
while [[ ! -f COMPLETE ]]; do
    [[ ! -f FAILED ]] || exit 1
    sleep 12
done
export PYTHONPATH="$PWD:$PWD/vendor" MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
data=/workspace/cnr-experiments/difftraj-five-40k-v1
prev=/workspace/cnr-experiments/pooled-velocity-class-v1
net=/workspace/cnr-experiments/gaf-inloop-v1/g27-job/g27-data/kit-check/assets/most.net.xml
"$py" plot_endpoint_comparison.py --run "$PWD" --baseline "$prev" --data "$data" --net "$net" --output report
"$py" debug_velocity_direction.py --run "$PWD" --data "$data" > report/geometry.json
for s in 42 43; do
    cp "velocity-$s/eval/metrics.json" "report/velocity-$s.json"
    cp "velocity-$s/config.json" "report/config-$s.json"
    cp "velocity-$s/epochs.jsonl" "report/loss-$s.jsonl"
done
touch report/VERIFIED_OUTPUTS
