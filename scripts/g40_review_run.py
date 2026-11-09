"""Finite post-completion evidence export and predeclared figures; no new fitting."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import g39_run as runner

ROOT=Path('g40-review-data');SOURCE=Path('../g40-job/g40-data');UTILITY=Path('../g40-job/g40u-data')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time(),
        'source_manifest_sha256':runner.digest('source-manifest.json'),'max_wait_hours':24,'max_active_hours':1,
        'new_fitting':False,'automatic_thesis_success_claim':False},f,indent=2)
    start=time.monotonic()
    def terminate(*_):runner.STOP.set();raise RuntimeError('G40 evidence supervisor interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    try:
        while not (SOURCE/'all-stages-complete.json').exists():
            for root in (SOURCE,UTILITY):
                if (root/'failed.json').exists():raise RuntimeError('Upstream failed; do not summarize partial results as complete')
            if time.monotonic()-start>24*3600:raise TimeoutError('24 hour wait limit')
            runner.save(ROOT/'progress.json',{'stage':'waiting_for_primary_and_utility_closure'});time.sleep(30)
        stage=time.monotonic();runner.R=ROOT;runner.START=stage
        def guard():
            free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if free<12*1024**3 or used>4*1024**3 or time.monotonic()-stage>3600:
                runner.STOP.set();raise RuntimeError('G40 evidence budget')
            return {'free_bytes':free,'output_bytes':used}
        runner.guard=guard
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Other GPU jobs active; preserve them')
        runner.save(ROOT/'progress.json',{'stage':'full_artifact_and_utility_verification'})
        runner.run('verify','g40_export.py',['--root',SOURCE,'--utility',UTILITY,'--output',ROOT/'aggregates'])
        r=json.loads((ROOT/'aggregates/complete.json').read_text());assert r['complete']
        assert r['artifact_sha256']==runner.digest(ROOT/'aggregates/verified-aggregates.json')
        evidence=ROOT/'evidence';shutil.copytree('paper-evidence',evidence)
        shutil.copy2(ROOT/'aggregates/verified-aggregates.json',evidence/'g40-verified.json')
        runner.save(ROOT/'progress.json',{'stage':'fixed_all_class_figures'})
        runner.run('visuals','g40_visual_assets.py',['--root',SOURCE,'--cache','../g35d-job/g35-data/gaf-cache',
            '--network','../g30-job/g30-data/source/scenario/in/most.net.xml','--output',ROOT/'journey-figures'],0)
        runner.verify(ROOT/'journey-figures')
        runner.run('tables-figures','../thesis/tools/build_g40_assets.py',['--evidence',evidence,'--output',ROOT/'paper-assets'])
        runner.verify(ROOT/'paper-assets')
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-stage,
            'formal_all_stages_sha256':runner.digest(SOURCE/'all-stages-complete.json'),
            'aggregate_receipt_sha256':runner.digest(ROOT/'aggregates/complete.json'),
            'journey_figure_receipt_sha256':runner.digest(ROOT/'journey-figures/complete.json'),
            'paper_asset_receipt_sha256':runner.digest(ROOT/'paper-assets/complete.json'),
            'scientific_interpretation_complete':False,'thesis_revised':False,
            'next_action':'inspect full evidence and rendered figures, then revise and build thesis'})
    except BaseException as error:runner.STOP.set();runner.save(ROOT/'failed.json',{'error':repr(error)});raise


if __name__=='__main__':main()
