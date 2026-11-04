"""Three independent GPU runs, followed by exact vector-road diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--name',default='road-g2');p.add_argument('--steps',type=int,default=20000)
a=p.parse_args();out=Path('runs')/a.name;out.mkdir(parents=True,exist_ok=False)

def worker(gpu,variant):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src',OMP_NUM_THREADS='4')
    cmd=[sys.executable,'-u','scripts/train_road_diffusion.py','--variant',variant,
         '--steps',str(a.steps),'--output',str(out/variant)]
    with (out/f'{variant}.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/f'{variant}-job.json').write_text(json.dumps({'gpu':gpu,'pid':child.pid,'command':cmd}))
        code=child.wait()
        if code==0:
            code=subprocess.run([sys.executable,'scripts/evaluate_geometry.py','--samples',str(out/variant/'validation_samples.npz'),
                '--output',str(out/variant/'geometry.json')],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    print(json.dumps({'variant':variant,'exit_code':code}),flush=True)
    return {'variant':variant,'exit_code':code}

with ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(worker,i,v) for i,v in enumerate(['relative','map','map_roadloss'])]
    jobs=[f.result() for f in futures]
(out/'study_complete.json').write_text(json.dumps({'jobs':jobs,'complete':all(j['exit_code']==0 for j in jobs)},indent=2))
if any(j['exit_code'] for j in jobs):sys.exit(1)
