"""Finite E4 evaluation queue with a matched-data raw-trajectory comparator."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

root=Path('runs/long-selection-e4');root.mkdir(parents=True,exist_ok=False)
selected=json.loads(Path('runs/long-path-e3/selected_data.json').read_text())
length=selected['length'];data=f'data/processed/gaf-duration-e2/length{length}'
count=len(np.load(Path(data)/'train.npz')['labels']);steps=math.ceil(count/64)*50
(root/'plan.json').write_text(json.dumps({'pid':os.getpid(),'length':length,'data':data,'steps':steps,'batch_size':64,'equivalent_epochs':50,
    'selector_seed':42,'auditor_seeds':[43,44],'checkpoint':'final','K':8,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
env=os.environ|{'PYTHONPATH':'src','OMP_NUM_THREADS':'4'}
def worker(item):
    gpu,seed=item;e=env|{'CUDA_VISIBLE_DEVICES':str(gpu)}
    with (root/f'seed{seed}.log').open('x') as log:
        subprocess.run([sys.executable,'-u','scripts/train_representation.py','--variant','xy_motion_tcn','--data',data,'--output',str(root/f'tcn-seed{seed}'),'--seed',str(seed),'--steps',str(steps),'--batch-size','64','--primary-checkpoint','final'],env=e,stdout=log,stderr=subprocess.STDOUT,check=True)
    return seed
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,enumerate([42,43,44])))
    # Serialize CPU-intensive road graph audits; GPUs are no longer the bottleneck.
    for seed in [42,43,44]:
        with (root/f'eval{seed}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/evaluate_long_selection.py','--seed',str(seed)],env=env|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}));raise
