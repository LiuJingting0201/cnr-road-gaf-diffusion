"""Cloud dependency continuation: fixed downstream utility after G40 closure."""
import concurrent.futures as cf
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import g39_run as runner
from g40_utility import REGIMES

ROOT=Path('g40u-data');SOURCE=Path('../g40-job/g40-data')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'awaiting':'G40 complete.json','fits':30,'max_wait_hours':24,'max_fit_hours':4,
        'source_hashes':{str(p):runner.digest(p) for p in Path('scripts').glob('g40*.py')}},f,indent=2)
    started=time.monotonic()
    try:
        while not (SOURCE/'complete.json').exists():
            if (SOURCE/'failed.json').exists():raise RuntimeError('G40 failed; utility does not proceed')
            if time.monotonic()-started>24*3600:raise TimeoutError('G40 dependency exceeded24h; no speculative training')
            runner.save(ROOT/'progress.json',{'stage':'waiting_G40','seconds':time.monotonic()-started});time.sleep(30)
        receipt=json.loads((SOURCE/'complete.json').read_text());assert receipt['complete'] and receipt['route_duration_equal_across_arms']
        stage=time.monotonic();runner.R=ROOT;runner.START=stage
        def guard():
            free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if free<12*1024**3 or used>4*1024**3 or time.monotonic()-stage>4*3600:
                runner.STOP.set();raise RuntimeError('G40 utility storage/time budget exceeded')
            return {'free_bytes':free,'output_bytes':used}
        runner.guard=guard
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy:raise RuntimeError('GPU occupied after G40; preserve other work, do not start')
        runner.run('prepare','g40_utility.py',['prepare','--root',SOURCE,'--data','../g35d-job/g35-data/direct-legs','--output',ROOT/'cache'])
        def slot(gpu,jobs):
            for regime,seed in jobs:
                name=f'{regime}-{seed}';out=ROOT/'fits'/name
                runner.run(name,'g40_utility.py',['train','--cache',ROOT/'cache','--regime',regime,'--seed',seed,'--output',out],gpu)
                runner.verify(out)
        jobs=[(r,s) for r in REGIMES for s in (4001,4002)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(slot,g,jobs[g::4]) for g in range(4)]
            for f in futures:f.result()
        results={f'{r}-{s}':runner.verify(ROOT/'fits'/f'{r}-{s}') for r,s in jobs}
        for s in (4001,4002):
            configs=[json.loads((ROOT/'fits'/f'{r}-{s}'/'config.json').read_text()) for r in REGIMES]
            assert len({c['initial_sha256'] for c in configs})==1
        runner.save(ROOT/'complete.json',{'complete':True,'fit_count':30,'seconds':time.monotonic()-stage,'results':results,
            'source_G40_sha256':runner.digest(SOURCE/'complete.json'),'utility_protocol_sha256':runner.digest('docs/G40_EXECUTION_PROTOCOL.md'),
            'limits':['Two held-out OD cohorts; shared bus templates.','Independent raw-series classifier is a utility endpoint, not a replacement for physical quality.']})
    except BaseException as e:
        runner.STOP.set();runner.save(ROOT/'failed.json',{'complete':False,'error':repr(e),'seconds':time.monotonic()-started});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE)
    SOURCE=parser.parse_args().source
    main()
