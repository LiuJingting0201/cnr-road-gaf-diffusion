"""Finite G5 queue: full preprocessing, three seeds, complete validation reports."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=Path('runs/path-g5');p.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(PYTHONPATH='src',OMP_NUM_THREADS='4')
try:
 with (p/'prepare.log').open('x') as f:
  subprocess.run([sys.executable,'-u','scripts/prepare_route_data.py'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 meta=json.loads(Path('data/processed/route512/dataset.json').read_text())
 if min(meta['summary']['train']['class_counts'][c] for c in ['bus','motorcycle','private_car','bycicle','taxi_or_uber'])<32:raise RuntimeError('Insufficient training support')
 (p/'data_ready.json').write_text(json.dumps(meta,indent=2))
 def worker(item):
  gpu,seed=item;e=env|{'CUDA_VISIBLE_DEVICES':str(gpu)};run=p/f'seed{seed}'
  with (p/f'seed{seed}.log').open('x') as f:
   subprocess.run([sys.executable,'-u','scripts/train_path_diffusion.py','--output',str(run),'--seed',str(seed)],env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
   subprocess.run([sys.executable,'-u','scripts/evaluate_path_diffusion.py','--run',str(run)],env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
  return {'seed':seed,'complete':True}
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(worker,enumerate([42,43,44])))
 (p/'complete.json').write_text(json.dumps({'status':'complete','results':results,'time':time.time()}))
except Exception as ex:
 (p/'failed.json').write_text(json.dumps({'status':'failed','error':str(ex),'time':time.time()}));raise
