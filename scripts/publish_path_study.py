"""One-shot local callback: publish completed reports in an isolated Git branch.

Never stages data/checkpoints, never changes the caller's checkout, and never
force-pushes. Intended to accompany the explicitly authorized experiment batch.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',default='7777')
p.add_argument('--remote-dir',required=True);p.add_argument('--worktree',required=True)
p.add_argument('--timeout-hours',type=float,default=3);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
log=root/'logs/path-g5-publication.json';log.parent.mkdir(parents=True,exist_ok=True)
def record(status,**kwargs):log.write_text(json.dumps({'status':status,'time_unix':time.time(),**kwargs},indent=2))
def remote(command):
    return subprocess.run(['ssh','-p',a.port,'-o','BatchMode=yes','-o','ConnectTimeout=15',a.host,
        'cd '+shlex.quote(a.remote_dir)+' && '+command],capture_output=True,text=True,timeout=180)
deadline=time.monotonic()+a.timeout_hours*3600;record('waiting_for_training')
try:
    while time.monotonic()<deadline:
        check=remote('cat runs/path-g5/complete.json')
        if check.returncode==0:
            marker=json.loads(check.stdout)
            if marker.get('status')!='complete':raise RuntimeError('Training reported incomplete')
            break
        progress=remote('cat runs/path-g5/failed.json')
        if progress.returncode==0:
            raise RuntimeError('A training task failed; publication refused')
        time.sleep(30)
    else:raise TimeoutError('Training completion deadline expired')
    branch='results/path-g5';destination=Path(a.worktree)
    subprocess.run(['git','worktree','add','-b',branch,str(destination),'HEAD'],cwd=root,check=True)
    names=['path-g5']
    for name in names:
        subprocess.run(['rsync','-az','--no-owner','--no-group','-e',f'ssh -p {a.port}',
            f'{a.host}:{a.remote_dir}/docs/results/{name}',str(destination/'docs/results/')],check=True)
    for seed in [42,43,44]:
        report=json.loads((destination/f'docs/results/path-g5/seed{seed}/results.json').read_text())
        assert len(report['rows'])==6
        assert report['config']['steps']==20000
    subprocess.run(['git','add',*[f'docs/results/{n}' for n in names]],cwd=destination,check=True)
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
                    'Publish completed map-path diffusion and GAF guidance validation'],cwd=destination,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=destination,text=True).strip()
    for target in ['origin','server']:
        for attempt in range(5):
            push=subprocess.run(['git','push',target,branch],cwd=destination,capture_output=True,text=True,timeout=180)
            if push.returncode==0:break
            record('retrying_push',remote=target,attempt=attempt+1,commit=commit,error=push.stderr[-1500:])
            time.sleep(30)
        else:raise RuntimeError(f'Push to {target} failed after five attempts')
    record('published',branch=branch,commit=commit,worktree=str(destination))
except Exception as exc:
    record('failed',error=str(exc));raise
