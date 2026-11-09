"""Finite G40 preparation after the existing performance queue closes."""
import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import g39_run as runner
from g40_data import VIEWS

ROOT=Path('g40a-data');PERF=Path('../g39perf-job/performance-data')


def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (ROOT/'queue-started.json').open('x') as f:
        json.dump({'pid':os.getpid(),'started_unix':time.time(),'purpose':'frozen prefix preparation and train-only sensitivity audit',
            'max_wait_hours':24,'max_active_hours':2,'source_manifest_sha256':runner.digest('source-manifest.json')},f,indent=2)
    start=time.monotonic()
    try:
        while not (PERF/'complete.json').exists():
            if (PERF/'failed.json').exists():raise RuntimeError('Performance dependency failed; inspect before preparation')
            if time.monotonic()-start>24*3600:raise TimeoutError('24 hour dependency limit')
            runner.save(ROOT/'progress.json',{'stage':'waiting_for_performance_completion'});time.sleep(30)
        report=json.loads((PERF/'report.json').read_text())
        assert report['complete'] and report['evaluation_json_exactly_equal']
        assert runner.digest(PERF/'report.json')==json.loads((PERF/'complete.json').read_text())['report_sha256']
        cached=report['training_state_bitwise_equal'] and all(v['whole_run_speedup']>1 for v in report['training'].values())
        runner.save(ROOT/'performance-adoption.json',{'input_pipeline':'cached' if cached else 'original',
            'training_bitwise_equal':report['training_state_bitwise_equal'],'report_sha256':runner.digest(PERF/'report.json'),
            'evaluation_one_load_adopted':True})
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Other GPU jobs active; no takeover')
        stage=time.monotonic();runner.R=ROOT;runner.START=stage
        def guard():
            free=shutil.disk_usage('.').free;used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if free<12*1024**3 or used>16*1024**3 or time.monotonic()-stage>2*3600:
                runner.STOP.set();raise RuntimeError('G40 preparation storage/time limit')
            return {'free_bytes':free,'output_bytes':used}
        runner.guard=guard
        runner.save(ROOT/'progress.json',{'stage':'model_preflight'})
        runner.run('model-checks','g40_model.py',[])
        runner.save(ROOT/'progress.json',{'stage':'four_view_prefix_preparation'})
        def prepare(gpu,view):
            out=ROOT/'prefixes'/view
            runner.run('prefix-'+view,'g40_data.py',['--view',view,'--cache','../g35d-job/g35-data/gaf-cache',
                '--raw-cache','../g39r-job/g39-data/raw-cache','--teachers','../g35d-job/g35-data/teachers',
                '--raw-teachers','../g39r-job/g39-data/raw-teachers','--output',out],gpu)
            return runner.verify(out)
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures={v:pool.submit(prepare,i,v) for i,v in enumerate(VIEWS)}
            prefixes={v:f.result() for v,f in futures.items()}
        assert len({r['rows'] for r in prefixes.values()})==1
        runner.save(ROOT/'progress.json',{'stage':'train_only_sensitivity_audit'})
        runner.run('sensitivity','g40_audit.py',['--data','../g35d-job/g35-data/direct-legs',
            '--cache','../g35d-job/g35-data/gaf-cache','--teachers','../g35d-job/g35-data/teachers',
            '--output',ROOT/'sensitivity'],0)
        audit=runner.verify(ROOT/'sensitivity')
        runner.save(ROOT/'complete.json',{'complete':True,'seconds':time.monotonic()-stage,
            'input_pipeline':'cached' if cached else 'original','prefixes':{v:runner.digest(ROOT/'prefixes'/v/'complete.json') for v in VIEWS},
            'audit_sha256':runner.digest(ROOT/'sensitivity/complete.json'),'training_parents_audited':audit['training_parents'],
            'new_diffusion_fit_started':False,'next_action':'review preparation and finite pilot before formal G40 fits'})
    except BaseException as e:
        runner.STOP.set();runner.save(ROOT/'failed.json',{'complete':False,'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':main()
