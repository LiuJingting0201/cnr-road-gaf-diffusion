#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:g10-env:scripts:../scripts:../vendor/original_cnr/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>g30-data/cohort.lock
flock -n 9 || exit 1
trap 'echo "failed line $LINENO" >g30-data/cohort-failed.txt' ERR
test -f g30-data/pilot-complete.txt
.venv/bin/python -u scripts/g30_native.py cohort >g30-data/cohort-simulations.log 2>&1
extract_worker(){
    local worker=$1 index=0 seed split folder
    for seed in {300101..300106} {300201..300203}; do
        split=train
        (( seed >= 300200 )) && split=val
        if (( index % 3 == worker )); then
            folder=g30-data/native/$split/$seed
            .venv/bin/python -u scripts/g30_extract.py \
                --fcd "$folder/fcd.xml.gz" \
                --types g30-data/source/scenario/in/add/basic.vType.xml g30-data/source/scenario/in/route/*.xml \
                --tripinfo "$folder/tripinfo.xml" --routes "$folder/vehroute.xml.gz" \
                --episode "$seed" --output "$folder/extracted" >"$folder/extract.log" 2>&1 || return 1
        fi
        index=$((index+1))
    done
}
extract_worker 0 & a=$!
extract_worker 1 & b=$!
extract_worker 2 & c=$!
failed=0
for pid in "$a" "$b" "$c"; do wait "$pid" || failed=1; done
test "$failed" = 0
.venv/bin/python -u scripts/g30_prepare.py >g30-data/prepare.log 2>&1
.venv/bin/python -u scripts/g29_structure.py --raw g30-data/raw --domains cnr sumo native --output g30-data/structure.json >g30-data/structure.log 2>&1
.venv/bin/python -u scripts/g30_quality.py >g30-data/quality.log 2>&1
.venv/bin/python -u scripts/g30_fit.py encode >g30-data/encoding.log 2>&1
fit_worker(){
    local gpu=$1 index=0 domain seed
    for domain in cnr sumo native; do
        for seed in 42 43; do
            if (( index % 4 == gpu )); then
                CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u scripts/g30_fit.py fit \
                    --domain "$domain" --seed "$seed" >"g30-data/fit-$domain-$seed.log" 2>&1 || return 1
            fi
            index=$((index+1))
        done
    done
}
fit_worker 0 & a=$!
fit_worker 1 & b=$!
fit_worker 2 & c=$!
fit_worker 3 & d=$!
failed=0
for pid in "$a" "$b" "$c" "$d"; do wait "$pid" || failed=1; done
test "$failed" = 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/g30_fit.py transfer >g30-data/transfer.log 2>&1
.venv/bin/python -u scripts/g30_summarize.py >g30-data/summary.log 2>&1
echo complete >g30-data/cohort-complete.txt
