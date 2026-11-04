"""Bounded route-motion candidate experiment with frozen generator and classifiers."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/route-diverse-e4r');root.mkdir(parents=True,exist_ok=False)
env=os.environ|{'PYTHONPATH':'src','OMP_NUM_THREADS':'4'}
(root/'plan.json').write_text(json.dumps({'pid':os.getpid(),'K':8,'diffusion_seeds':[42,43,44],'new_training':False,'route_seed':20261107,'paired_noise_seed':2026,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
try:
    def generate(item):
        gpu,seed=item
        with (root/f'sample{seed}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/sample_route_diverse.py','--seed',str(seed)],env=env|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(generate,enumerate([42,43,44])))
    for seed in [42,43,44]:
        with (root/f'eval{seed}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/evaluate_long_selection.py','--seed',str(seed),'--max-candidates','64','--tag','route-diverse-e4r','--generator-root',str(root)],env=env|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'scripts/summarize_long_selection.py','--tag','route-diverse-e4r'],env=env,check=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}));raise
