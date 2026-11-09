"""Finite detached G39: four GPU fits, fresh SUMO, matched direct generation."""
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

R=Path('g39-data');PY='.venv/bin/python';FROZEN=Path('../g35d-job/g35-data')
SOURCE=Path('../g30-job/g30-data/source');NETWORK=SOURCE/'scenario/in/most.net.xml'
POLICY=Path('../g38-job/g38-data/models');EPISODES=tuple(range(390201,390209))
ARMS=('B','S','V','SV','P','SH','RAW');ACTIVE={};LOCK=threading.Lock();STOP=threading.Event();START=time.monotonic()


def digest(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def guard():
    free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    if free<12*1024**3 or used>24*1024**3 or time.monotonic()-START>24*3600:
        STOP.set();raise RuntimeError(f'G39 resource limit: free={free},used={used},seconds={time.monotonic()-START}')
    return {'free_bytes':free,'output_bytes':used}


def run(name,script,args,gpu=None):
    if STOP.is_set():raise RuntimeError('Queue halted')
    guard()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='src:g10-env:scripts:../scripts:../vendor/original_cnr/src',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    with (R/(name+'.log')).open('xb') as f:
        proc=subprocess.Popen([PY,'-u','scripts/'+script,*map(str,args)],env=env,stdin=subprocess.DEVNULL,
                              stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK:ACTIVE[name]=proc
        print(json.dumps({'started':name,'pid':proc.pid,'gpu':gpu}),flush=True)
        try:
            while proc.poll() is None:
                if STOP.wait(10):raise RuntimeError('Queue halted')
                storage=guard()
                save(R/'workers'/f'{name}.json',{'pid':proc.pid,'gpu':gpu,'seconds':time.monotonic()-START,**storage})
            if proc.returncode:raise RuntimeError(f'{name} exited{proc.returncode}; log preserved')
        except BaseException:
            STOP.set()
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(10)
            raise
        finally:
            with LOCK:ACTIVE.pop(name,None)


def verify(folder,sources=None):
    """Verify local outputs separately from hashes of upstream inputs."""
    folder=Path(folder)
    r=json.loads((folder/'complete.json').read_text())
    assert r['complete'] and not (folder/'failed.json').exists()
    artifacts=r.get('artifacts')
    if artifacts is not None:
        assert isinstance(artifacts,dict) and artifacts,'Empty artifact manifest'
        for name,spec in artifacts.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert digest(folder/name)==(spec['sha256'] if isinstance(spec,dict) else spec)
    # Older model/utility receipts have no manifest. In generated-pool receipts
    # model_sha256 identifies the source model, which is deliberately elsewhere.
    for field,name in (('model_sha256','model.pt'),('config_sha256','config.json'),
                       ('stats_sha256','stats.json'),('predictions_sha256','predictions.npz')):
        if field in r and (artifacts is None or name in artifacts):
            assert digest(folder/name)==r[field]
    for field,path in (sources or {}).items():assert digest(path)==r[field],field
    return r


def train(gpu,arm,seed):
    folder=R/'models'/f'{arm}-{seed}'
    teachers=R/'raw-teachers' if arm=='RAW' else FROZEN/'teachers'
    run(f'train-{arm}-{seed}','g39_train.py',['--data',FROZEN/'direct-legs/dataset.npz','--teachers',teachers,
        '--arm',arm,'--seed',seed,'--steps',10000,'--output',folder],gpu)
    return verify(folder)


def teacher(gpu,view,ready):
    ready.result();folder=R/'raw-teachers'/f'{view}-42'
    run('raw-teacher-'+view,'g39_raw.py',['train','--cache',R/'raw-cache','--view',view,'--output',folder],gpu)
    return verify(folder)


def training():
    jobs=[('fit',a,s) for a,s in [('B',42),('B',43),('S',42),('V',42)]]
    jobs += [('raw',v,None) for v in ('spatial','speed')]
    jobs += [('fit',a,s) for a,s in [('S',43),('V',43),('SV',42),('SV',43),('P',42),('P',43),('SH',42),('SH',43)]]
    jobs += [('fit','RAW',s) for s in (42,43)]
    finished={};running={};free=list(range(4));ready_raw=set()
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        prep=pool.submit(run,'prepare-raw','g39_raw.py',['prepare','--data',FROZEN/'direct-legs','--cache',FROZEN/'gaf-cache','--output',R/'raw-cache'])
        while jobs or running:
            for gpu in free.copy():
                available=next((i for i,j in enumerate(jobs) if j[1]!='RAW' or len(ready_raw)==2),None)
                if available is None:continue
                kind,arg,seed=jobs.pop(available);name=f'{kind}-{arg}-{seed}'
                f=pool.submit(train,gpu,arg,seed) if kind=='fit' else pool.submit(teacher,gpu,arg,prep)
                running[f]=(gpu,name,kind,arg);free.remove(gpu)
            done,_=cf.wait(running,timeout=10,return_when=cf.FIRST_COMPLETED)
            for f in done:
                gpu,name,kind,arg=running.pop(f);finished[name]=f.result();free.append(gpu);free.sort()
                if kind=='raw':ready_raw.add(arg)
            with LOCK:active={k:v.pid for k,v in ACTIVE.items()}
            save(R/'training-progress.json',{'complete':False,'finished':list(finished),'pending':jobs,'active':active,'seconds':time.monotonic()-START,**guard()})
        prep.result()
    assert len(finished)==16
    # Common initialization/statistics/data/number of seen valid tokens, per seed.
    for seed in (42,43):
        configs=[json.loads((R/'models'/f'{a}-{seed}'/'config.json').read_text()) for a in ARMS]
        for field in ('base_initial_sha256','data_sha256','effective_batch','learning_rate'):
            assert len({json.dumps(c[field],sort_keys=True) for c in configs})==1
        receipts=[verify(R/'models'/f'{a}-{seed}') for a in ARMS]
        for field in ('valid_tokens_seen','stats_sha256'):assert len({r[field] for r in receipts})==1
    save(R/'training-complete.json',{'complete':True,'components':finished,'fits':14,'raw_teacher_fits':2,
                                    'paired_initialization_data_tokens_stats_verified':True,'seconds':time.monotonic()-START})


def reference(e):
    b=R/str(e);args=['--episode',e,'--prepared',b/'requests','--simulation',b/'simulation']
    run(f'reference-{e}','g39_requests.py',['run',*args])
    run(f'extract-{e}','g39_requests.py',['extract',*args,'--output',b/'reference']);verify(b/'reference')


def references():
    for e in EPISODES:
        b=R/str(e);b.mkdir()
        run(f'requests-{e}','g39_requests.py',['prepare','--episode',e,'--prepared',b/'requests'])
    save(R/'requests-frozen.json',{'complete':True,'manifest_sha256':{str(e):digest(R/str(e)/'requests/requests.json') for e in EPISODES}})
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        refs=[pool.submit(reference,e) for e in EPISODES]
        # Map construction is sequential to bound graph memory.
        for e in EPISODES:
            b=R/str(e)
            run(f'catalog-{e}','g38_pool.py',['catalog','--prepared',b/'requests','--network',NETWORK,'--output',b/'catalog'])
        for f in refs:f.result()
    save(R/'references-complete.json',{'complete':True,'episodes':list(EPISODES)})


def generation_slot(gpu,cells):
    for e,a,s in cells:
        b=R/str(e);name=f'{a}-{s}';g=b/'generated'/name;out=b/'evaluation'/name
        run(f'generate-{e}-{name}','g39_generate.py',['--catalog',b/'catalog','--network',NETWORK,
            '--policy',POLICY/f'neural-{s}','--model',R/'models'/name,'--seed',s,'--output',g],gpu)
        verify(g,{'model_sha256':R/'models'/name/'model.pt','catalog_sha256':b/'catalog/routes.json'})
        run(f'evaluate-{e}-{name}','g39_evaluate.py',['--generated',g,'--catalog',b/'catalog','--reference',b/'reference',
            '--simulation',b/'simulation','--network',NETWORK,'--raw-cache',R/'raw-cache','--output',out]);verify(out)


def main():
    R.mkdir(exist_ok=True);lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (R/'queue-started.json').open('x') as f:
        json.dump({'pid':os.getpid(),'started_unix':time.time(),'episodes':EPISODES,'training_fits':14,
            'source_sha256':{str(p):digest(p) for root in ('scripts','src') for p in Path(root).rglob('*.py')},
            'protocol_sha256':digest('docs/G39_PROTOCOL.md'),'max_hours':24,'min_free_gib':12,'max_output_gib':24},f,indent=2)
    def terminate(*_):STOP.set();raise RuntimeError('G39 supervisor interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    try:
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:raise RuntimeError('GPUs occupied; preserve other jobs')
        for name in ('model','targets'):run('checks-'+name,f'g39_{name}.py',[])
        run('checks-base','g33_model.py',[])
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fits=pool.submit(training);refs=pool.submit(references);fits.result();refs.result()
        cells=[(e,a,s) for e in EPISODES for a in ARMS for s in (42,43)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(generation_slot,g,cells[g::4]) for g in range(4)]
            for f in futures:f.result()
        for e in EPISODES:
            for s in (42,43):
                hashes=[verify(R/str(e)/'generated'/f'{a}-{s}')['route_duration_pairing_sha256'] for a in ARMS]
                assert len(set(hashes))==1,'Route/duration differs across arms'
        run('summarize','g39_report.py',['--root',R,'--output',R/'report'])
        save(R/'complete.json',{'complete':True,'seconds':time.monotonic()-START,'fits':14,'evaluation_cells':len(cells),
            'route_duration_equal_across_arms':True,'report_sha256':digest(R/'report/complete.json'),
            'utility_stage_complete':False,'mechanism_stage_complete':False})
    except BaseException as e:
        STOP.set();save(R/'failed.json',{'complete':False,'error':repr(e),'seconds':time.monotonic()-START});raise


if __name__=='__main__':main()
