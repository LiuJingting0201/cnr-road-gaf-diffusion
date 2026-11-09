"""Isolated continuation after receipt-verifier failure; never retrain or overwrite."""
import argparse
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import g39_run as runner


def link(source,target):
    assert source.exists(),source
    assert not target.exists() and not target.is_symlink(),target
    target.parent.mkdir(parents=True,exist_ok=True)
    target.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def prepare(old):
    root=runner.R;old=old.resolve()
    assert json.loads((old/'failed.json').read_text())['complete'] is False
    training=json.loads((old/'training-complete.json').read_text())
    assert training['complete'] and training['fits']==14 and training['paired_initialization_data_tokens_stats_verified']
    origin=json.loads((old/'queue-started.json').read_text())
    unchanged={}
    for name,expected in origin['source_sha256'].items():
        if name=='scripts/g39_run.py':continue
        assert runner.digest(name)==expected,('Scientific source changed',name)
        unchanged[name]=expected
    assert runner.digest('docs/G39_PROTOCOL.md')==origin['protocol_sha256']
    for arm in runner.ARMS:
        for seed in (42,43):runner.verify(old/'models'/f'{arm}-{seed}')
    for view in ('spatial','speed'):runner.verify(old/'raw-teachers'/f'{view}-42')
    raw=runner.verify(old/'raw-cache');assert runner.digest(old/'raw-cache/raw.npz')==raw['raw_sha256']
    assert json.loads((old/'references-complete.json').read_text())['complete']
    for name in ('models','raw-teachers','raw-cache','training-complete.json','references-complete.json','requests-frozen.json'):
        link(old/name,root/name)
    reused=[]
    for episode in runner.EPISODES:
        source=old/str(episode);target=root/str(episode)
        runner.verify(source/'reference')
        catalog=runner.verify(source/'catalog')
        for field,name in [('routes_sha256','routes.json'),('od_sha256','od.json')]:
            assert runner.digest(source/'catalog'/name)==catalog[field]
        assert runner.digest(source/'requests/requests.json')==catalog['request_manifest_sha256']
        for name in ('requests','catalog','reference','simulation'):link(source/name,target/name)
        for generated in sorted((source/'generated').glob('*')):
            assert generated.is_dir() and (generated/'complete.json').exists(),('Partial old pool',str(generated))
            receipt=runner.verify(generated,{'model_sha256':old/'models'/generated.name/'model.pt',
                                            'catalog_sha256':source/'catalog/routes.json'})
            assert receipt['reference_opened'] is False and receipt['GAF_used_at_sampling'] is False
            assert not (source/'evaluation'/generated.name).exists(),'Unexpected previous evaluation'
            link(generated,target/'generated'/generated.name)
            reused.append({'cell':f'{episode}/{generated.name}','receipt_sha256':runner.digest(generated/'complete.json')})
    runner.save(root/'recovery-inputs.json',{'complete':True,'old_failure_sha256':runner.digest(old/'failed.json'),
        'old_queue_log_sha256':runner.digest(old/'queue.log'),'old_training_receipt_sha256':runner.digest(old/'training-complete.json'),
        'reused_pools':reused,'unchanged_scientific_sources':unchanged,'new_diffusion_training':False,
        'old_tree_written':False,'repair':'Separate local artifact manifests from upstream provenance hashes'})


def slot(gpu,cells):
    root=runner.R
    for episode,arm,seed in cells:
        b=root/str(episode);name=f'{arm}-{seed}';generated=b/'generated'/name;out=b/'evaluation'/name
        if not generated.exists():
            runner.run(f'generate-{episode}-{name}','g39_generate.py',['--catalog',b/'catalog','--network',runner.NETWORK,
                '--policy',runner.POLICY/f'neural-{seed}','--model',root/'models'/name,'--seed',seed,'--output',generated],gpu)
        runner.verify(generated,{'model_sha256':root/'models'/name/'model.pt','catalog_sha256':b/'catalog/routes.json'})
        runner.run(f'evaluate-{episode}-{name}','g39_evaluate.py',['--generated',generated,'--catalog',b/'catalog',
            '--reference',b/'reference','--simulation',b/'simulation','--network',runner.NETWORK,
            '--raw-cache',root/'raw-cache','--output',out])
        runner.verify(out)
        diagnostic=json.loads((out/'diagnostics.json').read_text());assert diagnostic['complete']
        assert diagnostic['generated_sha256']==runner.digest(generated/'complete.json')
        runner.save(root/'slots'/f'{gpu}.json',{'last_completed_cell':f'{episode}/{name}',
            'seconds':time.monotonic()-runner.START})


def main(old):
    root=runner.R;root.mkdir(exist_ok=True)
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (root/'queue-started.json').open('x') as f:
        json.dump({'pid':os.getpid(),'started_unix':time.time(),'recovery_from':str(old),'max_hours':24,
            'new_diffusion_training':False,'source_sha256':{str(p):runner.digest(p) for p in Path('scripts').glob('g39*.py')}},f,indent=2)
    def terminate(*_):runner.STOP.set();raise RuntimeError('Recovery interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,terminate)
    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPUs occupied; preserve other work')
        runner.guard();prepare(old)
        runner.run('checks-receipts','g39_receipt_checks.py',[])
        cells=[(e,a,s) for e in runner.EPISODES for a in runner.ARMS for s in (42,43)]
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(slot,g,cells[g::4]) for g in range(4)]
            for future in futures:future.result()
        for e in runner.EPISODES:
            for s in (42,43):
                hashes=[runner.verify(root/str(e)/'generated'/f'{a}-{s}')['route_duration_pairing_sha256'] for a in runner.ARMS]
                assert len(set(hashes))==1,'Route/duration differs across arms'
        runner.run('summarize','g39_report.py',['--root',root,'--output',root/'report'])
        runner.save(root/'complete.json',{'complete':True,'seconds':time.monotonic()-runner.START,'fits':14,
            'new_diffusion_training':False,'evaluation_cells':len(cells),'route_duration_equal_across_arms':True,
            'report_sha256':runner.digest(root/'report/complete.json'),
            'recovery_receipt_sha256':runner.digest(root/'recovery-inputs.json'),
            'utility_stage_complete':False,'mechanism_stage_complete':False})
    except BaseException as error:
        runner.STOP.set();runner.save(root/'failed.json',{'complete':False,'error':repr(error),
            'seconds':time.monotonic()-runner.START});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path,default=Path('../g39-job/g39-data'))
    main(parser.parse_args().previous)
