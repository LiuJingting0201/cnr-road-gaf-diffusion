"""Bounded four-GPU formal fits, prospective SUMO, generation and utility."""
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
from g40_requests import EPISODES

ROOT=Path('g40-data');PREP=Path('../g40a-job/g40a-data');PILOT=Path('../g40p-job/g40p-data')
FROZEN=Path('../g35d-job/g35-data');BASE=Path('../g39r-job/g39-data')
NETWORK=Path('../g30-job/g30-data/source/scenario/in/most.net.xml');POLICY=Path('../g38-job/g38-data/models')


def train_slot(gpu,jobs):
    for arm,seed in jobs:
        name=f'{arm}-{seed}';model=ROOT/'models'/name;prior=ROOT/'priors'/name
        runner.run('train-'+name,'g40_train.py',['--data',FROZEN/'direct-legs/dataset.npz','--base-models',BASE/'models',
            '--teachers',FROZEN/'teachers','--raw-teachers',BASE/'raw-teachers','--prefixes',PREP/'prefixes',
            '--output',model,'--arm',arm,'--seed',seed,'--steps',5000,'--input-pipeline','original'],gpu)
        runner.verify(model)
        runner.run('prior-'+name,'g40_prior.py',['--model',model,'--output',prior,'--steps',5000],gpu)
        runner.verify(prior)


def training():
    jobs=[(a,s) for a in ARMS for s in (42,43)]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(train_slot,g,jobs[g::4]) for g in range(4)]
        for f in futures:f.result()
    results={f'{a}-{s}':runner.verify(ROOT/'models'/f'{a}-{s}') for a,s in jobs}
    for seed in (42,43):
        receipts=[results[f'{a}-{seed}'] for a in ARMS]
        for field in ('valid_tokens_seen','stats_sha256','final_cuda_rng_sha256','final_batch_rng_sha256'):
            assert len({r[field] for r in receipts})==1,field
        configs=[json.loads((ROOT/'models'/f'{a}-{seed}'/'config.json').read_text()) for a in ARMS]
        for key in ('initial_motion_sha256','data_sha256','base_model_sha256'):
            assert len({c[key] for c in configs})==1,key
    runner.save(ROOT/'training-complete.json',{'complete':True,'fits':16,'learned_prior_fits':12,
        'paired_initialization_tokens_rng_stats_verified':True,'models':results,
        'prior_receipts':{f'{a}-{s}':runner.digest(ROOT/'priors'/f'{a}-{s}'/'complete.json') for a,s in jobs},
        'no_pilot_checkpoint_reused':True})


def reference(e):
    b=ROOT/str(e);args=['--episode',e,'--root',ROOT,'--prepared',b/'requests','--simulation',b/'simulation']
    runner.run(f'reference-{e}','g40_requests.py',['run',*args])
    runner.run(f'extract-{e}','g40_requests.py',['extract',*args,'--output',b/'reference']);runner.verify(b/'reference')


def references():
    # All request lists freeze before the first new simulator outcome is made.
    for e in EPISODES:
        b=ROOT/str(e);b.mkdir()
        runner.run(f'requests-{e}','g40_requests.py',['prepare','--episode',e,'--root',ROOT,'--prepared',b/'requests'])
        assert json.loads((b/'requests/g40-cohort.json').read_text())['training_nonbus_parent_overlap']==0
    runner.save(ROOT/'requests-frozen.json',{'complete':True,
        'manifest_sha256':{str(e):runner.digest(ROOT/str(e)/'requests/requests.json') for e in EPISODES}})
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        refs=[pool.submit(reference,e) for e in EPISODES]
        for e in EPISODES:
            b=ROOT/str(e)
            runner.run(f'catalog-{e}','g38_pool.py',['catalog','--prepared',b/'requests','--network',NETWORK,'--output',b/'catalog'])
        for f in refs:f.result()
    runner.save(ROOT/'references-complete.json',{'complete':True,'episodes':list(EPISODES)})


def generation_slot(gpu,cells):
    for e,arm,seed in cells:
        b=ROOT/str(e);name=f'{arm}-{seed}';generated=b/'generated'/name;out=b/'evaluation'/name
        runner.run(f'generate-{e}-{name}','g40_generate.py',['--catalog',b/'catalog','--network',NETWORK,
            '--policy',POLICY/f'neural-{seed}','--model',ROOT/'models'/name,'--prior',ROOT/'priors'/name,
            '--seed',seed,'--output',generated],gpu)
        runner.verify(generated,{'model_sha256':ROOT/'models'/name/'model.pt','prior_sha256':ROOT/'priors'/name/'model.pt',
            'catalog_sha256':b/'catalog/routes.json'})
        runner.run(f'evaluate-{e}-{name}','g39_evaluate.py',['--generated',generated,'--catalog',b/'catalog',
            '--reference',b/'reference','--simulation',b/'simulation','--network',NETWORK,'--raw-cache',BASE/'raw-cache','--output',out])
        runner.verify(out);assert json.loads((out/'diagnostics.json').read_text())['complete']


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'source_manifest_sha256':runner.digest('source-manifest.json'),'protocol_sha256':runner.digest('docs/G40_EXECUTION_PROTOCOL.md'),
        'max_hours':24,'max_new_gib':24,'min_free_gib':12,'episodes':EPISODES,'arms':ARMS,'motion_fits':16},f,indent=2)
    start=time.monotonic();runner.R=ROOT;runner.START=start
    def guard():
        free=shutil.disk_usage('.').free
        used=sum(p.stat().st_size for root in (ROOT,Path('g40u-data')) for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if free<12*1024**3 or used>24*1024**3 or time.monotonic()-start>24*3600:
            runner.STOP.set();raise RuntimeError('G40 combined formal/utility resource bound')
        return {'free_bytes':free,'output_bytes':used}
    runner.guard=guard
    def terminate(*_):runner.STOP.set();raise RuntimeError('G40 supervisor interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    try:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        preparation=json.loads((PREP/'complete.json').read_text());assert preparation['complete']
        pilot=json.loads((PILOT/'complete.json').read_text());assert pilot['complete'] and not (PILOT/'failed.json').exists()
        review=json.loads((PILOT/'review.json').read_text());assert review['release_formal_experiments']
        assert review['pilot_receipt_sha256']==runner.digest(PILOT/'complete.json')
        assert review['formal_source_manifest_sha256']==runner.digest('source-manifest.json')
        runner.run('implementation-checks','g40_checks.py',[])
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            fits=pool.submit(training);refs=pool.submit(references);fits.result();refs.result()
        cells=[(e,a,s) for e in EPISODES for a in ARMS for s in (42,43)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(generation_slot,g,cells[g::4]) for g in range(4)]
            for f in futures:f.result()
        for e in EPISODES:
            for seed in (42,43):
                hashes=[runner.verify(ROOT/str(e)/'generated'/f'{a}-{seed}')['route_duration_pairing_sha256'] for a in ARMS]
                assert len(set(hashes))==1,'Route/duration differs across arms'
        runner.run('summarize','g40_report.py',['--root',ROOT,'--output',ROOT/'report'])
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-start,'motion_fits':16,'prior_fits':12,
            'evaluation_cells':128,'route_duration_equal_across_arms':True,'report_sha256':runner.digest(ROOT/'report/complete.json'),
            'utility_stage_complete':False,'receipt_scope':'immutable primary-stage closure; all-stages-complete.json closes utility'})
        runner.run('utility','g40_utility_run.py',['--source',ROOT])
        utility=runner.verify(Path('g40u-data'))
        assert utility['fit_count']==30
        runner.save(ROOT/'all-stages-complete.json',{'complete':True,'seconds':time.monotonic()-start,
            'primary_receipt_sha256':runner.digest(ROOT/'complete.json'),'utility_receipt_sha256':runner.digest('g40u-data/complete.json'),
            'next_action':'scientific evidence review then thesis update; no autonomous success claim'})
    except BaseException as error:runner.STOP.set();runner.save(ROOT/'failed.json',{'error':repr(error),'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
