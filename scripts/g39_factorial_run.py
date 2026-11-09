"""Finite post-utility factorial sampling; never contend with main four GPUs."""
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

ROOT=Path('g39f-data');DEPEND=Path('../g39u-job/g39u-data');OLD=Path('../g38-job/g38-data')
NETWORK=Path('../g30-job/g30-data/source/scenario/in/most.net.xml')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'awaiting':'G39 utility completion and free GPUs','max_wait_hours':28,'max_sampling_hours':6,
        'source_hashes':{str(p):runner.digest(p) for p in Path('scripts').glob('g39*.py')}},f,indent=2)
    start=time.monotonic()
    try:
        while not (DEPEND/'complete.json').exists():
            if (DEPEND/'failed.json').exists():raise RuntimeError('Utility dependency failed; do not compete for GPUs')
            if time.monotonic()-start>28*3600:raise TimeoutError('Dependency wait exceeded28h')
            runner.save(ROOT/'progress.json',{'stage':'waiting_for_GPU_dependency','seconds':time.monotonic()-start});time.sleep(30)
        assert json.loads((DEPEND/'complete.json').read_text())['complete'];stage=time.monotonic();runner.R=ROOT;runner.START=stage
        def guard():
            free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if free<12*1024**3 or used>6*1024**3 or time.monotonic()-stage>6*3600:
                runner.STOP.set();raise RuntimeError('Factorial resource/time limit')
            return {'free_bytes':free,'output_bytes':used}
        runner.guard=guard
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPUs occupied; no takeover')
        runner.run('checks','g39_factorial.py',['checks'])
        def cell(gpu,e,seed):
            b=OLD/str(e);g=ROOT/str(e)/'generated'/str(seed);out=ROOT/str(e)/'evaluation'/str(seed)
            runner.run(f'generate-{e}-{seed}','g39_factorial.py',['generate','--catalog',b/'catalog','--network',NETWORK,
                '--policy',OLD/'models'/f'neural-{seed}','--frozen','../g35d-job/g35-data','--seed',seed,'--output',g],gpu)
            runner.verify(g,{'model_sha256':Path('../g35d-job/g35-data/models')/f'native_0_25s-{seed}'/'model.pt',
                             'catalog_sha256':b/'catalog/routes.json'})
            runner.run(f'evaluate-{e}-{seed}','g39_factorial.py',['evaluate','--generated',g,'--catalog',b/'catalog','--reference',b/'reference',
                '--simulation',b/'simulation','--network',NETWORK,'--output',out]);return runner.verify(out)
        jobs=[(e,s) for e in (380201,380202) for s in (42,43)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={f'{e}/{s}':pool.submit(cell,g,e,s) for g,(e,s) in enumerate(jobs)}
            results={k:f.result() for k,f in futures.items()}
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-stage,'results':results,
            'exposed_development_scenes':True,'same_expanded_pool_for_all_methods':True,'reference_duration_used':False,
            'limit':'Eight candidates per available route; not an equal-budget comparison with historical K8. Distribution reweighting, not diffusion in-loop gradients.'})
    except BaseException as e:runner.STOP.set();runner.save(ROOT/'failed.json',{'complete':False,'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dependency',type=Path,default=DEPEND)
    DEPEND=parser.parse_args().dependency
    main()
