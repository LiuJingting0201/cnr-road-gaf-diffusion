"""Four independent GPU queues; immutable run directory, explicit failure records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from cnr.representations import VARIANTS

p=argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--steps',type=int,default=4000)
p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
p.add_argument('--data',default='data/processed/most64')
a=p.parse_args()
out=Path('runs')/a.name;out.mkdir(parents=True,exist_ok=False)
tasks=[{'variant':v,'seed':s} for s in a.seeds for v in VARIANTS]
(out/'study.json').write_text(json.dumps(vars(a)|{'tasks':tasks,'created_utc':datetime.now(timezone.utc).isoformat()},indent=2))

def worker(gpu,tasks):
    history=[]
    for task in tasks:
        name=f"{task['variant']}-seed{task['seed']}"
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src',OMP_NUM_THREADS='4')
        cmd=[sys.executable,'-u','scripts/train_representation.py','--variant',task['variant'],
             '--seed',str(task['seed']),'--steps',str(a.steps),'--data',a.data,'--output',str(out/name)]
        with (out/f'{name}.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            (out/f'gpu{gpu}-current.json').write_text(json.dumps({'pid':child.pid,'gpu':gpu,'task':task,'command':cmd}))
            code=child.wait()
        history.append(task|{'exit_code':code})
        (out/f'gpu{gpu}-history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({'gpu':gpu,'task':task,'exit_code':code}),flush=True)
        if code:
            # A systematic failure should not silently consume the rest of the queue.
            break
    return history

with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
    futures=[pool.submit(worker,gpu,tasks[i::len(a.gpus)]) for i,gpu in enumerate(a.gpus)]
    result=[f.result() for f in futures]
done=sum(len(h) for h in result);failed=sum(r['exit_code']!=0 for h in result for r in h)
(out/'study_complete.json').write_text(json.dumps({'completed_tasks':done,'expected_tasks':len(tasks),'failed_tasks':failed},indent=2))
if failed or done!=len(tasks):sys.exit(1)
