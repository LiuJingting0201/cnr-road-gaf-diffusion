#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>summary.lock
flock -n 9 || exit 0
[[ ! -f SUMMARY_COMPLETE ]] || exit 0
trap 'rc=$?; if (( rc != 0 )); then date -u > SUMMARY_FAILED; fi' EXIT
while [[ ! -f COMPLETE ]]; do
    [[ ! -f FAILED.json ]] || exit 1
    kill -0 "$(cat queue.pid)" || exit 1
    sleep 30
done
/workspace/cnr-road-gaf-diffusion/.venv/bin/python aggregate_five_motion.py
test -s validation-summary.json
test -s validation-tables.md
date -u > SUMMARY_COMPLETE
