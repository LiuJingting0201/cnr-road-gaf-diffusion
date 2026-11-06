#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:g10-env:scripts:../scripts:../vendor/original_cnr/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>g30-data/pilot.lock
flock -n 9 || exit 1
trap 'echo "failed line $LINENO" >g30-data/pilot-failed.txt' ERR
.venv/bin/python -u scripts/g30_native.py pilot
.venv/bin/python -u scripts/g30_extract.py \
    --fcd g30-data/native/pilot/300001/fcd.xml.gz \
    --types g30-data/source/scenario/in/add/basic.vType.xml g30-data/source/scenario/in/route/*.xml \
    --tripinfo g30-data/native/pilot/300001/tripinfo.xml \
    --routes g30-data/native/pilot/300001/vehroute.xml.gz \
    --episode 300001 --output g30-data/native/pilot/300001/extracted
echo complete >g30-data/pilot-complete.txt
