#!/usr/bin/env bash
set -euo pipefail
export CNR_RUN_ROOT=g14-data PYTHONPATH=g10-env:src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
binary=g10-env/sumo/bin/sumo
export SUMO_HOME="$(realpath g10-env/sumo)"
mkdir -p "$CNR_RUN_ROOT"
exec 9>"$CNR_RUN_ROOT/queue.lock"
flock -n 9 || exit 1
if [[ -s "$CNR_RUN_ROOT/complete.json" ]]; then exit 0; fi
mkdir -p "$CNR_RUN_ROOT/interrupted"
if [[ -f "$CNR_RUN_ROOT/failed.txt" ]]; then
    mv "$CNR_RUN_ROOT/failed.txt" "$CNR_RUN_ROOT/interrupted/failed.$(date +%s)"
fi
trap 'echo "failed at line $LINENO" > "$CNR_RUN_ROOT/failed.txt"' ERR
simulate_episode() {
    local seed="$1"
    if [[ ! -s "$CNR_RUN_ROOT/$seed/manifest.json" ]]; then
        if [[ -d "$CNR_RUN_ROOT/$seed" ]]; then
            mv "$CNR_RUN_ROOT/$seed" "$CNR_RUN_ROOT/interrupted/$seed.$(date +%s)"
        fi
        .venv/bin/python scripts/g10_sumo.py "$seed" "$binary" > "$CNR_RUN_ROOT/$seed.log" 2>&1
    fi
}
for ((first=261001; first<=261080; first+=8)); do
    pids=()
    for ((seed=first; seed<first+8; seed++)); do
        simulate_episode "$seed" & pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    test "$failed" = 0
done
.venv/bin/python scripts/g14_export.py "$CNR_RUN_ROOT"
