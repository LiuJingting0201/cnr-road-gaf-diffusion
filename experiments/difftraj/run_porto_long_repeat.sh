set -euo pipefail
exec 9>fit.lock
flock -n 9 || exit 1
trap 'touch FAILED' ERR
old=/workspace/cnr-experiments/porto-stabilized-v1
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
export PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1
mkdir fit score
ln -s "$old/shared-bins.npz" .
(cd fit
ln -s "$old/long/prepared.npz" .
PORTO_TRAIN_SEED=43 "$py" -m torch.distributed.run --standalone --nproc_per_node=4 ../porto.py train --large-microbatch --stabilized >train.log 2>&1)
(cd score
ln -s "$old/long-on-long/observations.npz" "$old/long-on-long/prepared.npz" ../fit/best.pt ../fit/checkpoint.pt .
"$py" ../porto.py score --best --cutoff-comparison --save-draws >score.log 2>&1)
"$py" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('score/porto-final.json').read_text())
old=json.loads(Path('/workspace/cnr-experiments/porto-stabilized-v1/long-on-long/porto-final.json').read_text())
rows=[json.loads(s) for s in Path('fit/epochs.jsonl').read_text().splitlines()]
assert len(rows)==200 and all(x['train_seed']==43 for x in rows)
assert r['requests']==3000 and r['request_index_sha256']==old['request_index_sha256']
assert r['epoch']==min(rows,key=lambda x:x['val_loss'])['epoch']
PY
touch COMPLETE
