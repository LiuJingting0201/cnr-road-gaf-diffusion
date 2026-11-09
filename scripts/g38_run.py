"""Finite, detached cloud dependency queue; no outcome-adaptive configurations."""
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

R=Path('g38-data');PY='.venv/bin/python';SOURCE=Path('../g30-job/g30-data/source')
NETWORK=SOURCE/'scenario/in/most.net.xml';FROZEN=Path('../g35d-job/g35-data')
EPISODES=(380201,380202);ACTIVE={};LOCK=threading.Lock();STOP=threading.Event();START=time.monotonic()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def guard():
    free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    if free<12*1024**3 or used>12*1024**3 or time.monotonic()-START>12*3600:
        STOP.set();raise RuntimeError(f'G38 resource limit free={free},used={used}')
    return {'free_bytes':free,'output_bytes':used}


def run(name,script,args,gpu=None):
    if STOP.is_set():raise RuntimeError('Queue halted')
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='src:g10-env:scripts:../scripts:../vendor/original_cnr/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
    with (R/(name+'.log')).open('xb') as log:
        proc=subprocess.Popen([PY,'-u','scripts/'+script,*map(str,args)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK:ACTIVE[name]=proc
        try:
            while proc.poll() is None:
                if STOP.wait(5):raise RuntimeError('Queue halted')
                storage=guard()
                with LOCK:active={n:p.pid for n,p in ACTIVE.items()}
                # Individual status files avoid multiple writers to one temporary file.
                save(R/'workers'/f'{name}.json',{'name':name,'pid':proc.pid,'active':active,'queue_seconds':time.monotonic()-START,**storage})
            if proc.returncode:raise RuntimeError(f'{name}:exit{proc.returncode}; preserved log')
        except BaseException:
            STOP.set()
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            raise
        finally:
            with LOCK:ACTIVE.pop(name,None)


def scene(e):return R/str(e)


def reference(e):
    b=scene(e);args=['--episode',e,'--prepared',b/'requests','--simulation',b/'simulation']
    run(f'reference-{e}','g38_requests.py',['run',*args])
    run(f'extract-{e}','g38_requests.py',['extract',*args,'--output',b/'reference'])


def main():
    R.mkdir(exist_ok=True)
    lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (R/'queue-started.json').open('x') as f:
        json.dump({'started_unix':time.time(),'pid':os.getpid(),'episodes':EPISODES,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('scripts').glob('g38*.py')},
            'max_hours':12,'min_free_gib':12,'max_output_gib':12,'route_fits':4,'new_diffusion_fits':0},f,indent=2)
    try:
        for name,script in [('policy','g38_policy.py'),('pool','g38_pool.py'),('metrics','g38_evaluate.py')]:run('checks-'+name,script,['checks'])
        run('checks-reference','g38_requests.py',['checks','--episode',380201])
        run('checks-geometry','g32_route_checks.py',[])
        run('prepare-policy','g38_policy.py',['prepare','--data',FROZEN/'direct-legs','--source',SOURCE,'--output',R/'policy-data'])
        for e in EPISODES:
            scene(e).mkdir();run(f'requests-{e}','g38_requests.py',['prepare','--episode',e,'--prepared',scene(e)/'requests'])
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            fits=[pool.submit(run,f'train-{kind}-{seed}','g38_policy.py',['train','--data',R/'policy-data','--kind',kind,'--seed',seed,'--output',R/'models'/f'{kind}-{seed}'],gpu)
                  for gpu,(kind,seed) in enumerate([(k,s) for k in ('linear','neural') for s in (42,43)])]
            refs=[pool.submit(reference,e) for e in EPISODES]
            maps=[pool.submit(run,f'catalog-{e}','g38_pool.py',['catalog','--prepared',scene(e)/'requests','--network',NETWORK,'--output',scene(e)/'catalog']) for e in EPISODES]
            for f in fits:f.result()
            save(R/'training-complete.json',{'complete':True,'fits':{f'{k}-{s}':json.loads((R/'models'/f'{k}-{s}'/'complete.json').read_text()) for k in ('linear','neural') for s in (42,43)}})
            run('evaluate-policy','g38_policy.py',['evaluate','--data',R/'policy-data','--models',R/'models','--output',R/'policy-evaluation'],0)
            for f in maps:f.result()
            draws=[pool.submit(run,f'generate-{e}-{seed}','g38_pool.py',['generate','--catalog',scene(e)/'catalog','--network',NETWORK,'--models',R/'models','--frozen',FROZEN,'--seed',seed,'--output',scene(e)/'generated'/str(seed)],gpu)
                   for gpu,(e,seed) in enumerate([(e,s) for e in EPISODES for s in (42,43)])]
            for f in draws:f.result()
            save(R/'generation-complete.json',{'complete':True,'cells':4,'references_opened_by_generation':False})
            for f in refs:f.result()
        results={}
        # Bound evaluation memory; GPU fits and sampling have finished.
        for e in EPISODES:
            for seed in (42,43):
                b=scene(e);output=b/'evaluation'/str(seed)
                run(f'evaluate-{e}-{seed}','g38_evaluate.py',['evaluate','--generated',b/'generated'/str(seed),'--catalog',b/'catalog','--reference',b/'reference','--simulation',b/'simulation','--network',NETWORK,'--output',output])
                results[f'{e}/{seed}']=json.loads((output/'complete.json').read_text())
        save(R/'complete.json',{'complete':True,'seconds':time.monotonic()-START,'results':results,
            'policy_evaluation':json.loads((R/'policy-evaluation/complete.json').read_text()),
            'limits':['Two independent new simulator scenes sharing OD/service plans; four cells are not four independent scenes.','Fixed catalog truncates learned route probabilities.','Class score improvements alone do not imply better trajectories.']})
    except BaseException as e:
        STOP.set();save(R/'failed.json',{'complete':False,'error':repr(e),'seconds':time.monotonic()-START});raise


if __name__=='__main__':main()
