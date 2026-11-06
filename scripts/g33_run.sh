#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=src:g10-env:scripts:../g31-job/scripts:../g30-job/scripts:../g29-job/scripts:../g28-job/scripts:../g23-job/scripts:../g22-job/scripts:../scripts:../vendor/original_cnr/src
mkdir -p g33-data/models g33-data/teachers
exec 9>g33-data/training.lock
flock -n 9 || exit 1
test -f g33-data/direct-legs/complete.json
test ! -e g33-data/training-started.json
.venv/bin/python - <<'PY'
import hashlib,json,time
from pathlib import Path
names=['g33_model.py','g33_train.py','g33_teachers.py','g33_prepare.py','g33_run.sh']
Path('g33-data/training-started.json').write_text(json.dumps({'started_unix':time.time(),'steps':10000,'seeds':[42,43],'effective_batch':64,'arms':['relative_257','native_0_25s'],'teacher_epochs':50,'source_sha256':{n:hashlib.sha256((Path('scripts')/n).read_bytes()).hexdigest() for n in names}},indent=2)+'\n')
PY
fit() {
    CUDA_VISIBLE_DEVICES="$1" .venv/bin/python -u scripts/g33_train.py \
        --data g33-data/direct-legs/dataset.npz --arm "$2" --seed "$3" \
        --steps 10000 --output "g33-data/models/$2-$3" > "g33-data/$2-$3.log" 2>&1
}
teacher() {
    while [[ ! -f g33-data/gaf-cache/complete.json ]]; do
        [[ ! -f g33-data/gaf-cache/failed.json ]] || return 1
        [[ ! -f g33-data/gaf-cache-launch-failed.txt ]] || return 1
        sleep 2
    done
    CUDA_VISIBLE_DEVICES="$1" .venv/bin/python -u scripts/g33_teachers.py train \
        --view "$2" --output g33-data/teachers > "g33-data/teacher-$2.log" 2>&1
}
fit 0 native_0_25s 42 & p0=$!
fit 1 native_0_25s 43 & p1=$!
(fit 2 relative_257 42 && teacher 2 spatial) & p2=$!
(fit 3 relative_257 43 && teacher 3 speed) & p3=$!
(.venv/bin/python -u scripts/g33_teachers.py prepare > g33-data/gaf-prepare.log 2>&1 || { touch g33-data/gaf-cache-launch-failed.txt; exit 1; }) & pc=$!
bad=0
for pid in "$p0" "$p1" "$p2" "$p3" "$pc"; do
    wait "$pid" || bad=1
done
.venv/bin/python - "$bad" <<'PY'
import json,sys,time
from pathlib import Path
expected=[Path('g33-data/models')/f'{arm}-{seed}'/'complete.json' for arm in ['relative_257','native_0_25s'] for seed in [42,43]]+[Path('g33-data/teachers')/f'{view}-42'/'complete.json' for view in ['spatial','speed']]
missing=[str(p) for p in expected if not p.exists() or json.loads(p.read_text()).get('complete') is not True]
ok=sys.argv[1]=='0' and not missing
Path('g33-data/training-complete.json' if ok else 'g33-data/training-failed.json').write_text(json.dumps({'complete':ok,'finished_unix':time.time(),'missing_completed_components':missing,'generation_evaluated':False},indent=2)+'\n')
if not ok: raise SystemExit(1)
PY
