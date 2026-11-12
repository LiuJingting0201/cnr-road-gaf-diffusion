#!/usr/bin/env bash
set -euo pipefail
exec 9>repeat.lock
flock -n 9 || exit 1
trap 'touch FAILED' ERR
export PYTHONPATH="$PWD:/workspace/cnr-experiments/difftraj-nov11" OMP_NUM_THREADS=1 PORTO_TRAIN_SEED=43
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
base=/workspace/cnr-experiments/erik-geolife
old=/workspace/cnr-experiments/erik-issue11-15
sha256sum "$base/observations.npz" "$base/prepared.npz" > input.sha256
for arm in conditional unconditional; do
    mkdir "$arm"
    opts=()
    if [[ "$arm" == unconditional ]]; then opts+=(--unconditional); fi
    (
      cd "$arm"
      ln -s "$base/prepared.npz" "$base/observations.npz" .
      "$py" -m torch.distributed.run --standalone --nproc_per_node=4 ../porto.py train "${opts[@]}" > train.log 2>&1
      CUDA_VISIBLE_DEVICES=0 "$py" ../porto.py score "${opts[@]}" > score.log 2>&1
    )
    "$py" - "$arm" "$old" <<'PY'
import json,sys
from pathlib import Path
arm=Path(sys.argv[1])
new,old=[json.loads((d/'geolife-final.json').read_text()) for d in [arm,Path(sys.argv[2])/arm]]
rows=[json.loads(line) for line in (arm/'epochs.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in rows]==list(range(1,201))
assert all(r['train_seed']==43 for r in rows)
assert new['requests']==old['requests']==3000 and new['epoch']==200
assert new['request_index_sha256']==old['request_index_sha256'] and new['model']==old['model']==arm.name
(arm/'paired.json').write_text(json.dumps({'seed42':old,'seed43':new,'train_seconds':rows[-1]['elapsed_s']},indent=2))
PY
    touch "$arm/COMPLETE"
done
touch COMPLETE
