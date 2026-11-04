"""Expand only the road-audit search capacity; reuse all saved models/candidates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

tag='long-selection-e4-cap64';root=Path('runs')/tag
root.mkdir(parents=True,exist_ok=False)
env=os.environ|{'PYTHONPATH':'src','OMP_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'0'}
(root/'plan.json').write_text(json.dumps({'pid':os.getpid(),'max_candidates':64,'radius_m':5,'seeds':[42,43,44],'scope':'post-hoc audit search-cap sensitivity; unchanged models, samples and selection rule','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
try:
    original=Path('runs/long-selection-e4')
    deadline=time.monotonic()+1800
    while not (original/'complete.json').exists():
        if (original/'failed.json').exists() or time.monotonic()>deadline:
            raise RuntimeError('Original E4 failed or did not complete before sensitivity timeout')
        time.sleep(5)
    for seed in [42,43,44]:
        with (root/f'eval{seed}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/evaluate_long_selection.py','--seed',str(seed),'--max-candidates','64','--tag',tag],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'scripts/summarize_long_selection.py','--tag',tag],env=env,check=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}));raise
