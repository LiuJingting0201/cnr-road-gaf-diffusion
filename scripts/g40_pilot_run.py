"""Four pilots only; formal experiments require explicit recorded review."""
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import g39_run as runner

ROOT=Path('g40p-data');PREP=Path('../g40a-job/g40a-data')
FROZEN=Path('../g35d-job/g35-data');BASE=Path('../g39r-job/g39-data')
ARMS=('B','F-SV','J-SV','J-RAW')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'source_manifest_sha256':runner.digest('source-manifest.json'),'formal_fits':False},f,indent=2)
    start=time.monotonic()
    try:
        prepared=json.loads((PREP/'complete.json').read_text());assert prepared['complete']
        assert prepared['input_pipeline']=='original'
        assert not (PREP/'failed.json').exists()
        for view,h in prepared['prefixes'].items():assert runner.digest(PREP/'prefixes'/view/'complete.json')==h
        review=json.loads((PREP/'review.json').read_text());assert review['release_finite_pilots']
        assert review['preparation_receipt_sha256']==runner.digest(PREP/'complete.json')
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        runner.R=ROOT;runner.START=start
        def guard():
            free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if free<12*1024**3 or used>4*1024**3 or time.monotonic()-start>3*3600:
                runner.STOP.set();raise RuntimeError('Pilot storage/time bound')
            return {'free_bytes':free,'output_bytes':used}
        runner.guard=guard;runner.run('implementation-checks','g40_checks.py',[])
        def slot(gpu,arm):
            name=arm+'-42';out=ROOT/'models'/name;prior=ROOT/'priors'/name
            runner.run('train-'+name,'g40_train.py',['--data',FROZEN/'direct-legs/dataset.npz','--base-models',BASE/'models',
                '--teachers',FROZEN/'teachers','--raw-teachers',BASE/'raw-teachers','--prefixes',PREP/'prefixes',
                '--output',out,'--arm',arm,'--seed',42,'--steps',1000,'--input-pipeline','original'],gpu)
            runner.verify(out)
            runner.run('prior-'+name,'g40_prior.py',['--model',out,'--output',prior,'--steps',1000],gpu);runner.verify(prior)
            runner.run('diagnostic-'+name,'g40_pilot_diagnostic.py',['--model',out,'--prior',prior,
                '--data',FROZEN/'direct-legs/dataset.npz','--output',ROOT/'diagnostics'/name],gpu)
            diagnostic=ROOT/'diagnostics'/name
            result=json.loads((diagnostic/'complete.json').read_text())
            assert result['complete'] and not (diagnostic/'failed.json').exists()
            assert result['model_sha256']==runner.digest(out/'model.pt')
            assert result['prior_sha256']==runner.digest(prior/'model.pt')
            return result
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={a:pool.submit(slot,g,a) for g,a in enumerate(ARMS)}
            results={a:f.result() for a,f in futures.items()}
        receipts=[runner.verify(ROOT/'models'/(a+'-42')) for a in ARMS]
        for field in ('valid_tokens_seen','stats_sha256','final_cuda_rng_sha256','final_batch_rng_sha256'):
            assert len({r[field] for r in receipts})==1,field
        configs=[json.loads((ROOT/'models'/(a+'-42')/'config.json').read_text()) for a in ARMS]
        assert len({c['initial_motion_sha256'] for c in configs})==1
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'pilots':results,
            'paired_initialization_tokens_rng_stats_verified':True,'source_manifest_sha256':runner.digest('source-manifest.json'),
            'next_action':'review code usage and prior behavior before formal fresh-start fits',
            'new_reference_outcomes_opened':False})
    except BaseException as error:runner.STOP.set();runner.save(ROOT/'failed.json',{'error':repr(error)});raise


if __name__=='__main__':main()
