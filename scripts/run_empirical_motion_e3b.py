"""Finite baseline queue, refusing to overwrite any existing run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/empirical-motion-e3b')
root.mkdir(parents=True, exist_ok=False)
env = os.environ | {'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': '0'}
(root / 'plan.json').write_text(json.dumps({'pid': os.getpid(), 'K': 8, 'points': 512,
    'resampling_seeds': [42, 43, 44], 'new_training': False,
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}, indent=2))
try:
    for seed in [42, 43, 44]:
        for stage, cmd in [
            ('sample', [sys.executable, '-u', 'scripts/sample_empirical_motion.py', '--seed', str(seed)]),
            ('eval', [sys.executable, '-u', 'scripts/evaluate_long_selection.py', '--seed', str(seed),
                '--max-candidates', '64', '--tag', 'empirical-motion-e3b',
                '--generator-root', str(root), '--prediction-subdir', 'selection'])]:
            with (root / f'{stage}{seed}.log').open('x') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, 'scripts/summarize_empirical_motion.py'], env=env, check=True)
    (root / 'complete.json').write_text(json.dumps({'complete': True, 'time': time.time()}))
except Exception as ex:
    (root / 'failed.json').write_text(json.dumps({'error': str(ex), 'time': time.time()}))
    raise
