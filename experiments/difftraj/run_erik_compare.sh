set -eu
exec 9>compare.lock
flock -n 9 || exit 1
export PYTHONPATH=/workspace/cnr-experiments/difftraj-nov11
py=/workspace/cnr-road-gaf-diffusion/.venv/bin/python
base=/workspace/cnr-experiments/erik-geolife
mkdir -p conditional unconditional
for dir in conditional unconditional; do
    for f in observations.npz prepared.npz; do ln -sfn "$base/$f" "$dir/$f"; done
done
cd conditional
cp "$base/checkpoint.pt" checkpoint.pt
CUDA_VISIBLE_DEVICES=0 "$py" ../porto.py score --development
cd ..
while ! test -f "$base/COMPLETE"; do sleep 30; done
cp "$base/checkpoint.pt" conditional/checkpoint.pt
(cd conditional && CUDA_VISIBLE_DEVICES=0 "$py" ../porto.py score)
cd unconditional
OMP_NUM_THREADS=1 "$py" -m torch.distributed.run --standalone --nproc_per_node=4 ../porto.py train --unconditional
CUDA_VISIBLE_DEVICES=0 "$py" ../porto.py score --unconditional
cd ..
"$py" - <<'PY'
import json
from pathlib import Path
a,b=[json.loads(Path(d,'geolife-final.json').read_text()) for d in ['conditional','unconditional']]
assert a['request_index_sha256']==b['request_index_sha256'] and a['requests']==b['requests']==3000
Path('comparison.json').write_text(json.dumps([a,b],indent=2))
PY
touch COMPLETE
