"""Finite E3 queue; coverage gates precede 3-seed diffusion training, no GAF filtering."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/long-path-e3')
root.mkdir(parents=True,exist_ok=False)
env=os.environ|{'PYTHONPATH':'src','OMP_NUM_THREADS':'4'}
(root/'plan.json').write_text(json.dumps({'pid':os.getpid(),'preferred_length':512,'fallback_length':224,
 'budget_m':4096,'seeds':[42,43,44],'steps':20000,'K':8,'minimum_training_per_class':16,'minimum_validation_per_class':3,
 'minimum_retention':.2,'maximum_mean_reconstruction_error_m':2,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
try:
    selected=None
    for length in [512,224]:
        data=f'data/processed/long-path-e3-{length}'
        source=f'data/processed/gaf-duration-e2/length{length}'
        with (root/f'prepare{length}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/prepare_long_route_data.py','--source',source,'--output',data,'--route-budget','4096'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        m=json.loads((Path(data)/'dataset.json').read_text())
        parent=json.loads((Path(source)/'dataset.json').read_text())
        eligible=True;reasons=[]
        for split,minimum in [('train',16),('val',3)]:
            counts=m['summary'][split]['class_counts']
            if min(counts[c] for c in ['bus','motorcycle','private_car','bycicle','taxi_or_uber'])<minimum:
                eligible=False;reasons.append(f'{split}: insufficient class support')
            retained=m['summary'][split]['windows']/parent['summary'][split]['vehicles']
            if retained<.2 or m['summary'][split]['projection_mean_error_m']>2:
                eligible=False;reasons.append(f'{split}: retention/reconstruction gate')
        (root/f'gate{length}.json').write_text(json.dumps({'eligible':eligible,'reasons':reasons,'dataset':m},indent=2))
        if eligible:
            selected=data;break
    if selected is None:raise RuntimeError('Both horizons failed coverage/reconstruction gates; inspect before changing protocol')
    (root/'selected_data.json').write_text(json.dumps({'data':selected,'length':length},indent=2))
    with (root/'smoke.log').open('x') as log:
        subprocess.run([sys.executable,'-u','scripts/train_long_path_diffusion.py','--data',selected,'--output',str(root/'smoke'),'--steps','20','--samples-per-class','1','--repeats','2'],env=env|{'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
    def worker(item):
        gpu,seed=item
        with (root/f'seed{seed}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','scripts/train_long_path_diffusion.py','--data',selected,'--output',str(root/f'seed{seed}'),'--seed',str(seed),'--steps','20000','--samples-per-class','8','--repeats','8'],env=env|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=log,stderr=subprocess.STDOUT,check=True)
        return seed
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        completed=list(pool.map(worker,enumerate([42,43,44])))
    (root/'training_complete.json').write_text(json.dumps({'complete':True,'seeds':completed,'time':time.time(),'next':'Independent topology/dynamics evaluation and equal-pool GAF selection; no test access'}))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}));raise
