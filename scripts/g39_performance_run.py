"""Bounded performance-only benchmark after all G39 scientific queues close."""
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

ROOT=Path('performance-data');BASE=Path('..')
DEPENDENCIES=[BASE/'g39r-job/g39-data',BASE/'g39ur-job/g39u-data',BASE/'g39fr-job/g39f-data']
STOP=threading.Event();ACTIVE={};LOCK=threading.Lock();START=None


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def guard():
    if STOP.is_set():raise RuntimeError('Performance queue halted')
    if START is not None and time.monotonic()-START>3600:raise TimeoutError('60 minute active benchmark limit')
    free=shutil.disk_usage('.').free
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    if free<12*1024**3 or used>3*1024**3:raise RuntimeError('Performance storage budget')


def run(name,script,args,gpu=None):
    guard();env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH='src:g10-env:scripts:../scripts:../vendor/original_cnr/src',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    started=time.monotonic();observations=[];next_sample=started
    with (ROOT/(name+'.log')).open('xb') as log:
        proc=subprocess.Popen(['.venv/bin/python','-u',script,*map(str,args)],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with LOCK:ACTIVE[name]=proc
        try:
            while proc.poll() is None:
                time.sleep(2);guard()
                if time.monotonic()>=next_sample:
                    sample={'elapsed_s':time.monotonic()-started}
                    try:
                        status=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        sample['rss_kib']=next(int(v.split()[1]) for v in status if v.startswith('VmRSS:'))
                        if gpu is not None:
                            result=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                                '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
                            sample['gpu_utilization_percent'],sample['driver_memory_mib']=map(int,result.strip().split(','))
                    except (OSError,StopIteration,subprocess.SubprocessError):pass
                    observations.append(sample);next_sample=time.monotonic()+10
            if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}')
        except BaseException:
            STOP.set()
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise
        finally:
            with LOCK:ACTIVE.pop(name,None)
    save(ROOT/'timing'/f'{name}.json',{'complete':True,'wall_s':time.monotonic()-started,'gpu':gpu,
        'resource_samples':observations,'resource_note':'Approximately10s snapshots; RSS may miss shorter peaks.'})


def fit_slot(gpu,arm,seed):
    order=['original','cached','cached','original'] if gpu%2==0 else ['cached','original','original','cached']
    counts={'original':0,'cached':0}
    for variant in order:
        repeat=counts[variant];counts[variant]+=1
        name=f'{arm}-{seed}-{variant}-{repeat}';out=ROOT/'fits'/name
        script=('original/scripts/' if variant=='original' else 'scripts/')+'g39_train.py'
        args=['--data',BASE/'g35d-job/g35-data/direct-legs/dataset.npz',
              '--teachers',BASE/'g35d-job/g35-data/teachers','--arm',arm,'--seed',seed,
              '--steps',300,'--output',out]
        if variant=='cached':args+=['--input-pipeline','cached']
        run(name,script,args,gpu)
        receipt=json.loads((out/'complete.json').read_text());assert receipt['complete']
        for field,filename in [('model_sha256','model.pt'),('config_sha256','config.json'),('stats_sha256','stats.json')]:
            assert digest(out/filename)==receipt[field]
        save(ROOT/'slots'/f'{gpu}.json',{'last_fit':name,'seconds':time.monotonic()-START})


def main():
    global START
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:
        json.dump({'pid':os.getpid(),'started_unix':time.time(),'awaiting':[str(p) for p in DEPENDENCIES],
            'max_wait_hours':24,'max_active_minutes':60,'steps_per_fit':300,'fits':16,
            'scientific_validation_used_for_selection':False,'source_manifest_sha256':digest('source-manifest.json')},f,indent=2)
    def terminate(*_):STOP.set();raise RuntimeError('Performance queue interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,terminate)
    wait_started=time.monotonic()
    try:
        while not all((p/'complete.json').exists() for p in DEPENDENCIES):
            if any((p/'failed.json').exists() for p in DEPENDENCIES):raise RuntimeError('Scientific dependency failed; no benchmark')
            if time.monotonic()-wait_started>24*3600:raise TimeoutError('24 hour dependency limit')
            save(ROOT/'progress.json',{'stage':'waiting_for_entire_G39_batch','seconds':time.monotonic()-wait_started})
            time.sleep(30)
        receipts=[json.loads((p/'complete.json').read_text()) for p in DEPENDENCIES]
        assert all(r['complete'] for r in receipts)
        assert receipts[0]['evaluation_cells']==112 and receipts[0]['route_duration_equal_across_arms']
        assert receipts[1]['fit_count']==26 and len(receipts[2]['results'])==4
        assert digest(DEPENDENCIES[0]/'report/complete.json')==receipts[0]['report_sha256']
        assert digest(DEPENDENCIES[0]/'complete.json')==receipts[1]['source_G39_sha256']
        # Scientific interpretation is an agent review, not a marker watcher.
        # Queue preparation may precede that review; benchmarks may not.
        while not (ROOT/'scientific-review.json').exists():
            if time.monotonic()-wait_started>24*3600:raise TimeoutError('24 hour scientific review limit')
            save(ROOT/'progress.json',{'stage':'awaiting_completed_batch_review','seconds':time.monotonic()-wait_started})
            time.sleep(30)
        reviewed=json.loads((ROOT/'scientific-review.json').read_text())
        assert reviewed['reviewed_main_utility_factorial']
        assert reviewed['receipt_sha256']=={str(p):digest(p/'complete.json') for p in DEPENDENCIES}
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Other GPU jobs active; no takeover')
        START=time.monotonic();guard()
        save(ROOT/'dependencies-verified.json',{'complete':True,'receipts':{str(p):digest(p/'complete.json') for p in DEPENDENCIES}})
        save(ROOT/'progress.json',{'stage':'exact-input-checks'})
        run('input-checks','scripts/g39_fast_inputs.py',[])
        run('cuda-input-checks','scripts/g39_performance_report.py',['checks'],0)
        jobs=[('B',42),('SV',42),('B',43),('SV',43)]
        save(ROOT/'progress.json',{'stage':'paired_training_benchmarks','jobs':jobs})
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(fit_slot,g,*job) for g,job in enumerate(jobs)]
            for future in futures:future.result()
        save(ROOT/'progress.json',{'stage':'evaluation_io_benchmark'})
        # Fixed already-completed case; both repetitions keep the same order.
        source=DEPENDENCIES[0];episode=source/'390201'
        for repeat,order in enumerate([['original','cached'],['cached','original']]):
            for variant in order:
                name=f'evaluate-{variant}-{repeat}'
                script=('original/scripts/' if variant=='original' else 'scripts/')+'g39_evaluate.py'
                run(name,script,['--generated',episode/'generated/B-42','--catalog',episode/'catalog',
                    '--reference',episode/'reference','--simulation',episode/'simulation',
                    '--network',BASE/'g30-job/g30-data/source/scenario/in/most.net.xml',
                    '--raw-cache',source/'raw-cache','--output',ROOT/'evaluation'/name])
        run('summarize','scripts/g39_performance_report.py',['report','--root',ROOT])
        save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-START,
            'report_sha256':digest(ROOT/'report.json'),'all_scientific_jobs_completed_before_benchmark':True,
            'new_full_training':False,'no_frozen_test_access':True})
    except BaseException as error:
        STOP.set();save(ROOT/'failed.json',{'complete':False,'error':repr(error),
            'seconds_since_launch':time.monotonic()-wait_started});raise


if __name__=='__main__':main()
