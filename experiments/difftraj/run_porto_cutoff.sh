#!/usr/bin/env bash
set -eu
exec 9>cutoff.lock
flock -n 9 || exit 1
export OMP_NUM_THREADS=1 PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11"
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
sha256sum -c inputs.sha256
"$py" porto_cutoff_setup.py > setup.log
# Long-arm training started independently in this run directory.
until grep -q '"epoch": 200,' long/epochs.jsonl; do test -n "$(pgrep -f 'porto.py train --large-microbatch')" || exit 1; sleep 20; done
while ! "$py" -c "import torch; assert torch.load('long/checkpoint.pt',weights_only=True,map_location='cpu')['epoch']==200"; do sleep 3; done
for cohort in long mixed; do
    (cd long-on-$cohort; ln -sf ../long/checkpoint.pt checkpoint.pt; "$py" ../porto.py score --cutoff-comparison > score.log)
done
(cd mixed-on-long; "$py" -m torch.distributed.run --standalone --nproc_per_node=4 ../porto.py train --large-microbatch > train.log 2>&1)
ln -sf ../mixed-on-long/checkpoint.pt mixed-on-mixed/checkpoint.pt
for cohort in long mixed; do
    (cd mixed-on-$cohort; "$py" ../porto.py score --cutoff-comparison > score.log)
done
touch COMPLETE
