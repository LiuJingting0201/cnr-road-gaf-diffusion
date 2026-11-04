"""Matched coordinate baselines plus scale-preserving GAF sampling, finite queue."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path('runs/g6-queue');root.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(PYTHONPATH='src',OMP_NUM_THREADS='4')
def command(cmd,log,gpu):
 with log.open('a') as f:subprocess.run([sys.executable,'-u',*cmd],env=env|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=f,stderr=subprocess.STDOUT,check=True)
def baseline(seed,gpu):
 run=f'runs/coordinate-g6/seed{seed}';log=root/f'coordinate{seed}.log'
 command(['scripts/train_road_diffusion.py','--variant','map','--data','data/processed/most64-route-cohort','--output',run,'--seed',str(seed),'--noise-seed',str(seed+2000),'--samples-per-class','16','--repeats','4'],log,gpu)
 command(['scripts/evaluate_path_diffusion.py','--run',run],log,gpu)
def guided():
 for seed in [42,43,44]:
  run=f'runs/preserved-g6/seed{seed}';log=root/f'preserved{seed}.log'
  command(['scripts/train_path_diffusion.py','--seed',str(seed),'--checkpoint',f'runs/path-g5/seed{seed}/latest.pt','--output',run,'--samples-per-class','16','--repeats','4','--guidance-weights','0.3','1','3','--guidance-mode','distance_preserving'],log,3)
  command(['scripts/evaluate_path_diffusion.py','--run',run],log,3)
try:
 command(['scripts/prepare_matched_coordinates.py'],root/'prepare.log',0)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  tasks=[pool.submit(baseline,seed,gpu) for gpu,seed in enumerate([42,43,44])]+[pool.submit(guided)]
  for task in tasks:task.result()
 (root/'complete.json').write_text(json.dumps({'status':'complete','time':time.time()}))
except Exception as ex:
 (root/'failed.json').write_text(json.dumps({'status':'failed','error':str(ex),'time':time.time()}));raise
