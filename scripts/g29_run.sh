#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src:g10-env:scripts:../scripts:../vendor/original_cnr/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>g29-data/queue.lock
flock -n 9 || exit 1
trap 'echo "failure at $LINENO" >g29-data/failed.txt' ERR
.venv/bin/python -u scripts/g29_encode.py --meta ../data/processed/original-hilbert64grid224/dataset.json >g29-data/encode.log 2>&1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g29_positive.py >g29-data/positive.log 2>&1
.venv/bin/python -u scripts/g29_structure.py >g29-data/structure.log 2>&1 & diagnostic=$!
worker(){
    local gpu=$1 count=0 domain view arm seed task
    for domain in cnr sumo; do
        for arm in xy native_h squeezed_h unit_h gaf; do
            for seed in 42 43; do
                if (( count % 4 == gpu )); then
                    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g29_train.py "$domain" full "$arm" "$seed" >"g29-data/$domain-full-$arm-$seed.log" 2>&1 || return 1
                fi
                count=$((count+1))
            done
        done
    done
    for domain in cnr sumo; do
        for seed in 42 43; do
            if (( count % 4 == gpu )); then
                CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g29_train.py "$domain" window512 gaf "$seed" >"g29-data/$domain-window512-gaf-$seed.log" 2>&1 || return 1
            fi
            count=$((count+1))
        done
    done
    for seed in 42 43; do
        if (( count % 4 == gpu )); then
            CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g29_train.py sumo-duration full gaf "$seed" >"g29-data/sumo-duration-full-gaf-$seed.log" 2>&1 || return 1
        fi
        count=$((count+1))
    done
}
worker 0 & a=$!
worker 1 & b=$!
worker 2 & c=$!
worker 3 & d=$!
failed=0
for pid in "$a" "$b" "$c" "$d" "$diagnostic"; do wait "$pid" || failed=1; done
test "$failed" = 0
echo complete >g29-data/training-complete.txt
