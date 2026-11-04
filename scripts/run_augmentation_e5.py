"""Bounded E5 pipeline: training pool, six controls, two downstream families, three seeds."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/augmentation-e5');root.mkdir(parents=True,exist_ok=False)
env=os.environ|{'PYTHONPATH':'src','OMP_NUM_THREADS':'4'}
groups=['real_only','repeat','timewarp','synthetic_random','synthetic_gaf','synthetic_tcn']
(root/'plan.json').write_text(json.dumps({'pid':os.getpid(),'diffusion_seed':42,'training_conditions':425,'K':8,'extra_per_group':425,'groups':groups,'downstream_arch':['tcn','gru'],'downstream_seeds':[100,101,102],'steps':1150,'batch_size':64,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
try:
    for script in ['sample_training_pool.py','prepare_augmentation_e5.py']:
        with (root/(script+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/'+script],env=env|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    for arch in ['tcn','gru']:
        with (root/f'smoke-{arch}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/train_augmentation_downstream.py','--group','real_only','--arch',arch,'--seed','999','--steps','2'],env=env|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    def worker(item):
        gpu,seed=item
        for arch in ['tcn','gru']:
            for group in groups:
                with (root/f'{arch}-{group}-seed{seed}.log').open('x') as log:
                    subprocess.run([sys.executable,'-u','scripts/train_augmentation_downstream.py','--group',group,'--arch',arch,'--seed',str(seed)],env=env|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,check=True)
        return seed
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:completed=list(pool.map(worker,enumerate([100,101,102])))
    subprocess.run([sys.executable,'scripts/summarize_augmentation_e5.py'],env=env,check=True)
    (root/'complete.json').write_text(json.dumps({'complete':True,'seeds':completed,'time':time.time()}))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}));raise
