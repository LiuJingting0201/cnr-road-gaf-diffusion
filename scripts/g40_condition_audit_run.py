"""One finite final-model audit after completed scientific and figure queues."""
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import g39_run as runner
from g40_model import ARMS

ROOT=Path('g40m-data');SOURCE=Path('../g40-job/g40-data');REVIEW=Path('../g40-review-job/g40-review-data')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'source_manifest_sha256':runner.digest('source-manifest.json'),'new_fitting':False,'max_wait_hours':24,'max_active_minutes':30},f,indent=2)
    start=time.monotonic()
    def terminate(*_):runner.STOP.set();raise RuntimeError('G40 condition audit interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    try:
        while not (REVIEW/'complete.json').exists():
            for root in (SOURCE,REVIEW,Path('../g40-job/g40u-data')):
                if (root/'failed.json').exists():raise RuntimeError('Upstream failed; preserve all results')
            if time.monotonic()-start>24*3600:raise TimeoutError('24 hour audit dependency limit')
            runner.save(ROOT/'progress.json',{'stage':'waiting_for_closed_science_and_visuals'});time.sleep(30)
        assert json.loads((SOURCE/'all-stages-complete.json').read_text())['complete']
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Other GPU jobs active')
        stage=time.monotonic();runner.R=ROOT;runner.START=stage
        def guard():
            free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if free<12*1024**3 or used>1024**3 or time.monotonic()-stage>1800:
                runner.STOP.set();raise RuntimeError('Final-model audit resource bound')
            return {'free_bytes':free,'output_bytes':used}
        runner.guard=guard
        def slot(gpu,jobs):
            for arm,seed in jobs:
                name=f'{arm}-{seed}';out=ROOT/'models'/name
                runner.run(name,'g40_condition_audit.py',['--data','../g35d-job/g35-data/direct-legs/dataset.npz',
                    '--model',SOURCE/'models'/name,'--prior',SOURCE/'priors'/name,'--output',out],gpu)
                runner.verify(out)
        jobs=[(a,s) for a in ARMS for s in (42,43)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(slot,g,jobs[g::4]) for g in range(4)]
            for f in futures:f.result()
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-stage,
            'results':{f'{a}-{s}':runner.verify(ROOT/'models'/f'{a}-{s}') for a,s in jobs},
            'formal_source_receipt_sha256':runner.digest(SOURCE/'all-stages-complete.json'),
            'limits':'25 fixed training parents, five predetermined noise levels; mechanism only, no held-out synthesis quality claim.'})
    except BaseException as error:runner.STOP.set();runner.save(ROOT/'failed.json',{'error':repr(error)});raise


if __name__=='__main__':main()
