#!/usr/bin/env bash
set -eu
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
exec 9>launcher.lock
flock -n 9 || exit 1
if test -e COMPLETE; then exit 0; fi
for i in {1..360}; do
    size=$(du -cb block-* 2>/dev/null | tail -1 | cut -f1)
    if test "$size" = "$(cat expected-bytes)"; then break; fi
    sleep 5
done
cat block-* > pilot.npz
sha256sum -c expected-sha
pids=()
for spec in 0:B 1:V 2:C 3:VC; do
    (
        arm=${spec#*:}
        export CUDA_VISIBLE_DEVICES=${spec%:*}
        for seed in 42 43; do
            dir="seed-$seed"
            mkdir -p "$dir"
            ln -sfn ../pilot.npz "$dir/pilot.npz"
            (cd "$dir"; DIFFTRAJ_TRAINING_SEED=$seed "$py" -u ../pilot.py "$arm" 8000 --all-validation > "$arm.log" 2>&1)
        done
    ) &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if test "$status" = 0; then touch COMPLETE; else touch FAILED; fi
exit "$status"
