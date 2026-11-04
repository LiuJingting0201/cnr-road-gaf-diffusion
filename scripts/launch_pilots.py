"""Launch two independent GPU pilots, preserving outputs and process IDs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--data', default='data/processed/most64')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
target = root/'runs'/a.name
target.mkdir(parents=True, exist_ok=False)
jobs = []
for gpu, kind in enumerate(('gaf_classifier', 'raw_diffusion')):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(root/'src'), OMP_NUM_THREADS='4')
    command = [sys.executable, '-u', 'scripts/train_baseline.py', '--kind', kind,
               '--data', a.data, '--output', str(target/kind), '--steps', str(a.steps)]
    with (target/f'{kind}.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    jobs.append({'kind': kind, 'gpu': gpu, 'pid': process.pid, 'command': command})
record = {'launched_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs}
(target/'jobs.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
