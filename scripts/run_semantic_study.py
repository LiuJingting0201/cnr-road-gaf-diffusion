"""Four GPU diffusion training interventions, then independent validation audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--name',default='semantic-g4');p.add_argument('--steps',type=int,default=20000)
a=p.parse_args();out=Path('runs')/a.name;out.mkdir(parents=True,exist_ok=False)
variants=['none','gaf','tcn','shuffled_gaf']
def worker(gpu,variant):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src',OMP_NUM_THREADS='4')
    cmd=[sys.executable,'-u','scripts/train_road_diffusion.py','--variant','map','--steps',str(a.steps),
         '--seed','42','--noise-seed','2042','--output',str(out/variant)]
    if variant!='none':
        path='runs/representation-r1-delta/delta_tcn-seed42' if variant=='tcn' else 'runs/representation-r1/delta_gaf-seed42'
        cmd+=['--semantic-teacher',path,'--semantic-weight','0.01']
    if variant=='shuffled_gaf':cmd+=['--shuffle-semantic-labels']
    with (out/f'{variant}.log').open('x') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/f'{variant}-job.json').write_text(json.dumps({'pid':proc.pid,'gpu':gpu,'command':cmd}))
        code=proc.wait()
    return {'variant':variant,'exit_code':code}
with ThreadPoolExecutor(max_workers=4) as pool:
    jobs=list(pool.map(lambda item:worker(*item),enumerate(variants)))
(out/'study_complete.json').write_text(json.dumps({'jobs':jobs,'complete':all(x['exit_code']==0 for x in jobs)},indent=2))
if any(x['exit_code'] for x in jobs):raise SystemExit('Training failed; inspect per-run logs')
env=os.environ.copy();env['PYTHONPATH']='src'
subprocess.run([sys.executable,'scripts/summarize_road_study.py','--name',a.name,'--semantic','--variants',*variants],env=env,check=True)
