"""One-shot G3 selection audit followed by G2 generator seed replication.

No test-set access; stop on any error. Run after GPUs 0,1,2 are available.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
os.chdir(root)
logdir=root/'logs/followup-g3-g2'
logdir.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env.update(PYTHONPATH='src',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
commands=[
    [sys.executable,'-u','scripts/evaluate_candidate_selection.py',
     '--generator','runs/road-g2/map','--delta-study','runs/representation-r1-delta',
     '--output','runs/candidate-selection-g3','--repetitions','8'],
]
for seed in [43,44]:
    commands += [[sys.executable,'-u','scripts/run_road_study.py','--name',f'road-g2-seed{seed}','--seed',str(seed)],
                 [sys.executable,'-u','scripts/summarize_road_study.py','--name',f'road-g2-seed{seed}']]
(logdir/'commands.json').write_text(json.dumps(commands,indent=2))
completed=[]
for i,cmd in enumerate(commands):
    with (logdir/f'{i:02d}.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (logdir/'active.json').write_text(json.dumps({'index':i,'pid':child.pid,'command':cmd}))
        code=child.wait()
    completed.append({'index':i,'exit_code':code,'finished_unix':time.time()})
    (logdir/'progress.json').write_text(json.dumps(completed,indent=2))
    if code:raise SystemExit(f'Task {i} failed; inspect log')
    if i==0:
        shutil.copy2('runs/candidate-selection-g3/report.json','docs/results/candidate-selection-g3.json')
(logdir/'complete.json').write_text(json.dumps({'status':'complete','jobs':completed},indent=2))
