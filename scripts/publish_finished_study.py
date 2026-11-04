"""Publish a completed R2 report on a separate result branch without touching main.

One-shot experiment callback, not a recurring monitor. Only report artifacts are
staged. Raw data, model weights and incomplete/failed results are never committed.
"""
import json
from pathlib import Path
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[1]
name='representation-r2-long'
source=root/'docs/results'/name
marker=root/'runs'/name/'study_complete.json'
deadline=time.monotonic()+12*3600
while time.monotonic()<deadline:
    if marker.exists():
        d=json.loads(marker.read_text())
        if d['failed_tasks'] or d['completed_tasks']!=d['expected_tasks']:
            raise SystemExit('R2 incomplete/failed; publication refused')
        result=source/'results.json';svg=source/'validation_f1.svg'
        if result.exists() and svg.exists() and svg.read_text().rstrip().endswith('</svg>'):
            if json.loads(result.read_text())['complete']:break
    time.sleep(30)
else:raise SystemExit('Report readiness timeout')
branch=f'results/{name}'
worktree=root.parent/'cnr-result-worktrees'/name
worktree.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(['git','worktree','add','-b',branch,str(worktree),'HEAD'],cwd=root,check=True)
destination=worktree/'docs/results'/name
shutil.copytree(source,destination,dirs_exist_ok=True)
subprocess.run(['git','add',str(destination.relative_to(worktree))],cwd=worktree,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
                'Record completed long-window validation study and figures'],cwd=worktree,check=True)
subprocess.run(['git','push','origin',branch],cwd=worktree,check=True)
record={'branch':branch,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=worktree,text=True).strip(),
        'remote':'origin (developer-server Git remote)','artifacts':str(destination)}
(root/'logs/r2-publication.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
