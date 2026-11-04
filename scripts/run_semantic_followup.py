"""Bounded follow-up: two CE replication seeds and one matched KL grid."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];os.chdir(root)
logdir=Path('logs/semantic-followup');logdir.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env['PYTHONPATH']='src'
commands=[]
for seed in [43,44]:
    commands.append([sys.executable,'-u','scripts/run_semantic_study.py','--name',f'semantic-g4-seed{seed}','--seed',str(seed)])
commands.append([sys.executable,'-u','scripts/run_semantic_study.py','--name','semantic-g4b-kl','--study','consistency'])
(logdir/'commands.json').write_text(json.dumps(commands,indent=2));completed=[]
for index,command in enumerate(commands):
    with (logdir/f'{index}.log').open('x') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        (logdir/'active.json').write_text(json.dumps({'index':index,'pid':child.pid,'command':command}))
        code=child.wait()
    completed.append({'index':index,'exit_code':code,'finished_unix':time.time()})
    (logdir/'progress.json').write_text(json.dumps(completed,indent=2))
    if code:raise SystemExit(f'Follow-up {index} failed')
(logdir/'complete.json').write_text(json.dumps({'complete':True,'jobs':completed},indent=2))
