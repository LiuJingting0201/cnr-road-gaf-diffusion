import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path('runs/original-gaf-alignment');root.mkdir(parents=True,exist_ok=False)
def work(item):
 gpu,seed=item;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='src',OMP_NUM_THREADS='4');run=root/f'seed{seed}'
 with (root/f'seed{seed}.log').open('x') as f:
  subprocess.run([sys.executable,'-u','scripts/train_original_gaf_classifier.py','--output',str(run),'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run([sys.executable,'-u','scripts/evaluate_original_gaf_classifier.py','--run',str(run)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 return seed
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:seeds=list(pool.map(work,enumerate([42,43,44])))
 (root/'study_complete.json').write_text(json.dumps({'complete':True,'seeds':seeds,'time':time.time()}))
except Exception as ex:
 (root/'failed.json').write_text(json.dumps({'error':str(ex),'time':time.time()}));raise
