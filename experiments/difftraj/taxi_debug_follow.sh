#!/usr/bin/env bash
set -euo pipefail
exec 9>follow.lock
flock -n 9 || exit 1
trap 'touch FOLLOW_FAILED' ERR
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for tick in {1..300}; do
    if [[ -f 10k/42/COMPLETE && -f 10k/43/COMPLETE && -f porto/42/COMPLETE && -f porto/43/COMPLETE ]]; then break; fi
    sleep 10
done
for arm in 10k porto; do for seed in 42 43; do test -f "$arm/$seed/COMPLETE"; done; done
# Only fit expanded arm when a prespecified development deficiency remains.
"$py" - <<'PY'
import json
from pathlib import Path
scores=[json.loads(Path(f'10k/{s}/development.json').read_text()) for s in [42,43]]
assert any(d['trip_js']>.0132 or d['length_js']>.0242 or d['density_js']>.0087 for d in scores), 'No prespecified expansion trigger; review the two development scores.'
PY
pids=()
for spec in 20k:42:0 20k:43:1 oldclock:42:2 oldclock:43:3; do
 IFS=: read -r arm seed gpu <<< "$spec"
 bash taxi_debug_fit.sh "$arm" "$seed" "$gpu" >"$arm-$seed.log" 2>&1 & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
for arm in 10k 20k oldclock; do
 for seed in 42 43; do
  "$py" taxi_od_diagnostic.py "$arm/$seed" --common-pack 10k --check "$arm/$seed/development.json"
 done
done
"$py" decide_taxi_size.py
arm=$(cat SELECTED)
mkdir "fresh-$arm"
(cd "fresh-$arm"; "$py" ../taxi_debug_data.py "$arm" fresh; "$py" ../porto.py prepare >split.json)
pids=()
for seed in 42 43; do
 gpu=$((seed-42))
 (
  cd "fresh-$arm"; mkdir "$seed"; cd "$seed"
  ln -s ../observations.npz ../prepared.npz "../../$arm/$seed/checkpoint.pt" .
  CUDA_VISIBLE_DEVICES=$gpu "$py" ../../porto.py score --all-holdout --save-draws >score.log 2>&1
 ) & pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
for seed in 42 43; do
 "$py" taxi_od_diagnostic.py "fresh-$arm/$seed" --common-pack 10k --check "fresh-$arm/$seed/geolife-final.json"
done
"$py" taxi_fresh_reference_compare.py --output fresh-reference-compare.json "fresh-$arm/42" "fresh-$arm/43"
touch FINAL_COMPLETE
