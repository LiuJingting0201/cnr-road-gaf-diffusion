"""E2 paired-duration screening on inherited 64x64-grid RGB; no new architecture sweep."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/gaf-duration-e2')
root.mkdir(parents=True,exist_ok=False)
(root/'plan.json').write_text(json.dumps({'lengths':[64,224,512],'seeds':[42,43,44],'epochs':50,
    'initialization':'imagenet','grid':64,'pid':os.getpid(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},indent=2))
def worker(item):
    gpu,length=item
    env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'src','OMP_NUM_THREADS':'4'}
    for seed in [42,43,44]:
        name=f'length{length}-seed{seed}'
        cmd=[sys.executable,'-u','scripts/train_gaf_resnet.py','--data',f'data/processed/gaf-duration-e2/length{length}','--output',str(root/name),'--seed',str(seed),'--initialization','imagenet']
        with (root/f'{name}.log').open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert json.loads((root/name/'complete.json').read_text())['complete']
    return length
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        done=list(pool.map(worker,enumerate([64,224,512])))
    subprocess.run([sys.executable,'scripts/evaluate_gaf_duration.py'],env=os.environ|{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':'src'},check=True)
    (root/'study_complete.json').write_text(json.dumps({'complete':True,'lengths':done,'time':time.time()}))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}))
    raise
