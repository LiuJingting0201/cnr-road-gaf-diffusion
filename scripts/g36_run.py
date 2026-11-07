"""Finite cloud queue: two new references, paired native generators and teachers."""
import concurrent.futures as cf
import json, os, pathlib, shutil, signal, subprocess, threading, time, hashlib
R=pathlib.Path('g36-data'); PY='.venv/bin/python'; EPISODES=(360201,360202)
LOCK=threading.Lock(); ACTIVE={}; STOP=threading.Event(); START=time.monotonic()
ROOTS={'old':pathlib.Path('../g33-job/g33-data'),'new':pathlib.Path('../g35d-job/g35-data')}
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def guard():
 free=shutil.disk_usage('.').free
 used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
 if free<8*1024**3 or used>10*1024**3 or time.monotonic()-START>12*3600:raise RuntimeError(f'G36 resource/time limit: free={free}, used={used}')
 return {'free_bytes':free,'g36_bytes':used}
def run(name,script,args,gpu=None):
 if STOP.is_set():raise RuntimeError('Queue halted')
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='src:g10-env:scripts:../scripts:../vendor/original_cnr/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 with (R/(name+'.log')).open('xb') as log:
  p=subprocess.Popen([PY,'-u','scripts/'+script,*map(str,args)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with LOCK:ACTIVE[name]=p
  try:
   while p.poll() is None:
    if STOP.wait(10):raise RuntimeError('Queue halted')
    guard()
   if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}; see preserved log')
  except BaseException:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   raise
  finally:
   with LOCK:ACTIVE.pop(name,None)
def scene(e):return R/str(e)
def reference(e):
 b=scene(e);a=['--episode',e,'--prepared',b/'requests','--simulation',b/'simulation']
 run(f'reference-{e}','g36_requests.py',['run',*a])
 # Successful process exit alone is insufficient for a complete simulation.
 import xml.etree.ElementTree as ET
 last=None
 for _,n in ET.iterparse(b/'simulation/summary.xml',events=('end',)):
  if n.tag=='step':last=float(n.get('time'));n.clear()
 expected=(23000 if e==360201 else 37400)-.25
 if last!=expected:raise RuntimeError(f'incomplete reference clock {last}, expected {expected}')
 run(f'extract-{e}','g36_requests.py',['extract',*a,'--output',b/'reference'])
def slot(gpu,era,seed):
 root=ROOTS[era]
 for e in EPISODES:
  b=scene(e);name=f'{era}-{seed}';pool=b/'pools'/name
  run(f'generate-{e}-{name}','g36_generate.py',['generate','--maps',b/'maps','--model',root/'models'/f'native_0_25s-{seed}','--seed',seed,'--output',pool],gpu)
  for teacher in (('old','new') if era=='new' else ('old',)):
   t=ROOTS[teacher]
   run(f'score-{e}-{name}-{teacher}','g36_feedback.py',['score','--generated',pool,'--teachers',t/'teachers','--cache',t/'gaf-cache','--output',b/'selected'/f'{name}-{teacher}'],gpu)
def main():
 R.mkdir(exist_ok=True)
 with (R/'queue-started.json').open('x') as f:json.dump({'started_unix':time.time(),'episodes':EPISODES,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('scripts').glob('*.py')},'new_root_budget_gib':10,'min_free_gib':8},f)
 try:
  for e in EPISODES:
   b=scene(e);b.mkdir()
   run(f'prepare-{e}','g36_requests.py',['prepare','--episode',e,'--prepared',b/'requests'])
   from g31_paths import source_inputs
   net,_=source_inputs(pathlib.Path('../g30-job/g30-data/source'))
   run(f'maps-{e}','g33_evaluate.py',['prepare_maps','--prepared',b/'requests','--network',net,'--output',b/'maps'])
  with cf.ThreadPoolExecutor(max_workers=6) as pool:
   futures=[pool.submit(reference,e) for e in EPISODES]+[pool.submit(slot,i,era,seed) for i,(era,seed) in enumerate([(e,s) for e in ('new','old') for s in (42,43)])]
   try:
    while futures:
     done,_=cf.wait(futures,timeout=10,return_when=cf.FIRST_COMPLETED)
     for f in done:f.result();futures.remove(f)
     with LOCK:active={k:v.pid for k,v in ACTIVE.items()}
     save(R/'progress.json',{'stage':'reference_generation_scoring','active':active,'seconds':time.monotonic()-START,'storage':guard()})
   except BaseException:STOP.set();raise
  # Teacher changes must never change the unscored Uniform control.
  import numpy as np
  for e in EPISODES:
   for seed in (42,43):
    a=scene(e)/'selected'/f'new-{seed}-old'/'Uniform'/'samples.npz'
    b=scene(e)/'selected'/f'new-{seed}-new'/'Uniform'/'samples.npz'
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
     if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):
      raise RuntimeError('Old/new teacher Uniform outputs differ')
  save(R/'uniform-controls.json',{'complete':True,'paired_pools':4,'all_arrays_equal':True})
  results={}
  for e in EPISODES:
   b=scene(e)
   for era in ('new','old'):
    for seed in (42,43):
     name=f'{era}-{seed}';shared=['--reference',b/'reference','--maps',b/'maps','--training',ROOTS[era]/'direct-legs']
     out=b/'evaluation'/name
     run(f'eval-{e}-{name}','g33_evaluate.py',['evaluate','--generated',b/'pools'/name,*shared,'--output',out])
     results[f'{e}/{name}/pool']=json.loads((out/'complete.json').read_text())
     for teacher in (('old','new') if era=='new' else ('old',)):
      tag=f'{name}-{teacher}';out=b/'evaluation'/f'selected-{tag}'
      run(f'eval-selection-{e}-{tag}','g36_feedback.py',['evaluate','--selected',b/'selected'/tag,*shared,'--output',out])
      for method in ('Uniform','S','V','SV'):
       results[f'{e}/{tag}/{method}']=json.loads((out/method/'complete.json').read_text())
     save(R/'progress.json',{'stage':'evaluation','completed_pools':len(results),'seconds':time.monotonic()-START,'storage':guard()})
  save(R/'complete.json',{'complete':True,'seconds':time.monotonic()-START,'results':results,'in_loop_guidance':False,'limits':['Two new simulator episodes and shared public road plans; no unseen-city claim.','Terminal selection, no denoising gradient guidance.','Generator scale comparison includes changed training-side duration prior.','Same new candidate pool with old/new teachers isolates teacher change.']})
 except BaseException as ex:
  STOP.set();save(R/'failed.json',{'complete':False,'error':repr(ex),'seconds':time.monotonic()-START});raise
if __name__=='__main__':main()
