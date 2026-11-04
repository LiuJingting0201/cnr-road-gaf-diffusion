"""One bounded locked-test queue; no model fitting, no automatic protocol changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cnr.final_protocol import verify_lock, sha256

p = argparse.ArgumentParser(); p.add_argument('--lock', required=True); a = p.parse_args()
verify_lock(a.lock)
root = Path('runs/final-test-e7'); root.mkdir(parents=True, exist_ok=False)
env = os.environ | {'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': '0'}
(root / 'plan.json').write_text(json.dumps({'pid': os.getpid(), 'lock': a.lock,
    'lock_sha256': sha256(a.lock), 'seeds': [42, 43, 44], 'new_training': False}, indent=2))
def run(name, script, *args):
    with (root / (name + '.log')).open('x') as log:
        subprocess.run([sys.executable, '-u', 'scripts/'+script, '--lock', a.lock, *args], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
try:
    run('prepare', 'prepare_locked_test.py')
    run('plan', 'plan_locked_test.py')
    for family in ['diffusion', 'empirical']:
        for seed in [42, 43, 44]:
            run(f'sample-{family}-{seed}', 'sample_locked_test.py', '--family', family, '--seed', str(seed))
            run(f'eval-{family}-{seed}', 'evaluate_long_selection.py', '--seed', str(seed), '--max-candidates', '64',
                '--tag', f'final-{family}-e7', '--generator-root', f'runs/final-{family}-e7',
                '--prediction-subdir', 'selection', '--test-reference', 'data/processed/final-test-e7/reference')
    run('auxiliary', 'evaluate_final_auxiliary.py')
    run('downstream', 'evaluate_final_downstream.py')
    run('summary', 'summarize_final_test.py')
    done = {'complete': True, 'time': time.time(), 'lock_sha256': sha256(a.lock), 'next': 'Inspect all results, create fixed figures, publish and pause'}
    (root / 'complete.json').write_text(json.dumps(done))
    Path('docs/results/final-test-e7/complete.json').write_text(json.dumps(done))
except Exception as ex:
    (root / 'failed.json').write_text(json.dumps({'error': str(ex), 'time': time.time()})); raise
