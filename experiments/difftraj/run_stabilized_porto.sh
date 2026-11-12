set -eu
exec 9>run.lock
flock -n 9 || exit 1
export PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
base=/workspace/cnr-experiments/porto-cutoff-v1
for arm in long mixed; do
    mkdir -p "$arm"
    (cd "$arm"
    "$py" - "$arm" <<'PY'
import numpy as np,sys
from porto_cutoff_setup_utils import make_train_dev
make_train_dev(sys.argv[1])
PY
    opts=()
    if test -e last.pt; then opts+=(--resume); fi
    "$py" -m torch.distributed.run --standalone --nproc_per_node=4 ../porto.py train --large-microbatch --stabilized "${opts[@]}" > train.log 2>&1)
    for cohort in long mixed; do
        dir="$arm-on-$cohort"; mkdir -p "$dir"
        (cd "$dir"; for f in observations.npz prepared.npz; do ln -sfn "$base/$dir/$f" "$f"; done
        for f in best.pt checkpoint.pt; do ln -sfn "../$arm/$f" "$f"; done
        "$py" ../porto.py score --best --cutoff-comparison > score.log)
    done
done
touch COMPLETE
