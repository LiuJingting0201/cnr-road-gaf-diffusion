"""Finite isolated supervisor. No restart or adaptive sweep."""
import json,os,subprocess,time,shutil,hashlib
from pathlib import Path
R=Path('g37-data');start=time.monotonic()
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def run(name,script,args):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='src:g10-env:scripts:../scripts:../vendor/original_cnr/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
 with (R/(name+'.log')).open('x') as f:
  p=subprocess.Popen(['.venv/bin/python','-u','scripts/'+script,*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  try:
   while p.poll() is None:
    free=shutil.disk_usage('.').free;used=sum(x.stat().st_size for x in R.rglob('*') if x.is_file())
    if free<12*1024**3 or used>8*1024**3 or time.monotonic()-start>7200:raise RuntimeError('G37 resource/time bound')
    save(R/'queue-progress.json',{'phase':name,'pid':p.pid,'seconds':time.monotonic()-start,'free_bytes':free,'output_bytes':used});time.sleep(5)
   if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}')
  except BaseException:
   if p.poll() is None:p.terminate();p.wait(timeout=30)
   raise
if __name__=='__main__':
 with (R/'queue-started.json').open('x') as f:json.dump({'started':time.time(),'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('scripts').glob('g37_*.py')}},f)
 try:
  run('routes','g37_routes.py',['routes','--source',R/'od.json','--network','../g30-job/g30-data/source/scenario/in/most.net.xml','--output',R/'routes'])
  t=Path('../g35d-job/g35-data')
  run('generate','g37_generate.py',['generate','--catalog',R/'routes/routes.json','--model',t/'models/native_0_25s-42','--teachers',t/'teachers','--cache',t/'gaf-cache','--output',R/'generated'])
  run('evaluate','g37_generate.py',['evaluate','--generated',R/'generated','--reference','../g36-job/g36-data/360201/reference','--output',R/'evaluation'])
  for p in ['routes','generated','evaluation']:assert json.loads((R/p/'complete.json').read_text())['complete']
  save(R/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'new_simulation':False,'new_training':False,'scope':'finite OD routing development feasibility; reused completed diffusion/GAF, existing reference only after generation'})
 except BaseException as e:save(R/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise
