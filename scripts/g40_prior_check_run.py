"""Recover pilot receipt review and test the already-declared full prior budget."""
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import g39_run as runner

ROOT=Path('g40pr-data');OLD=Path('../g40p-job/g40p-data');ARMS=('B','F-SV','J-SV','J-RAW')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'source_manifest_sha256':runner.digest('source-manifest.json'),'reuse_completed_motion_fits':True,
        'prior_updates':5000,'formal_source_prior_budget':5000},f,indent=2)
    start=time.monotonic();runner.R=ROOT;runner.START=start
    try:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        rec=[runner.verify(OLD/'models'/(a+'-42')) for a in ARMS]
        for key in ('valid_tokens_seen','stats_sha256','final_cuda_rng_sha256','final_batch_rng_sha256'):
            assert len({r[key] for r in rec})==1,key
        configs=[json.loads((OLD/'models'/(a+'-42')/'config.json').read_text()) for a in ARMS]
        assert len({c['initial_motion_sha256'] for c in configs})==1
        def slot(gpu,arm):
            model=OLD/'models'/(arm+'-42');prior=ROOT/'priors'/(arm+'-42');out=ROOT/'diagnostics'/(arm+'-42')
            runner.run('prior-'+arm,'g40_prior.py',['--model',model,'--output',prior,'--steps',5000],gpu);runner.verify(prior)
            runner.run('diagnostic-'+arm,'g40_pilot_diagnostic.py',['--model',model,'--prior',prior,
                '--data','../g35d-job/g35-data/direct-legs/dataset.npz','--output',out],gpu)
            r=json.loads((out/'complete.json').read_text());assert r['complete'] and not (out/'failed.json').exists()
            assert r['model_sha256']==runner.digest(model/'model.pt') and r['prior_sha256']==runner.digest(prior/'model.pt')
            return r
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={a:pool.submit(slot,g,a) for g,a in enumerate(ARMS)}
            results={a:f.result() for a,f in futures.items()}
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'pilots':results,
            'paired_initialization_tokens_rng_stats_verified':True,'old_failure_preserved':True,
            'reused_motion_receipts':{a:runner.digest(OLD/'models'/(a+'-42')/'complete.json') for a in ARMS},
            'no_motion_retraining':True,'prior_steps':5000,'new_reference_outcomes_opened':False,
            'next_action':'review full-budget prior support and code use before formal release'})
    except BaseException as error:runner.STOP.set();runner.save(ROOT/'failed.json',{'error':repr(error)});raise


if __name__=='__main__':main()
