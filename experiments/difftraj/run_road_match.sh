#!/usr/bin/env bash
set -euo pipefail
exec 9>match.lock
flock -n 9 || exit 1
trap 'touch MATCH_FAILED' ERR
export PYTHONPATH="$PWD/routing" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
for city in Porto Beijing; do
    for i in {1..120}; do if test -f "$city.ready"; then break; fi; sleep 5; done
    test -f "$city.ready"
    for f in tracks/"$city"-*.npz; do
        [[ "$f" == *-status.npz ]] && continue
        name=$(basename "$f" .npz)
        test -s "metrics/$name.json" || "$py" road_match.py "$name" 12
    done
done
touch MATCH_COMPLETE
