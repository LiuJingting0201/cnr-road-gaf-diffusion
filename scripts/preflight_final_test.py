"""Inventory frozen inputs and run checks without accessing test trajectories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from cnr.final_protocol import frozen_inputs, sha256

p = argparse.ArgumentParser()
p.add_argument('--output', default='runs/final-preflight')
p.add_argument('--lock', action='store_true')
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
try:
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], check=True, stdout=subprocess.PIPE)
    tests = ['tests/test_final_protocol.py', 'tests/test_original_gaf_alignment.py',
             'tests/test_path_diffusion.py', 'tests/test_cluster_wasserstein.py',
             'tests/test_candidate_selection.py', 'tests/test_augmentation.py']
    with (out / 'checks.log').open('x') as log:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    tracked = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
    files = [f for f in tracked if f.endswith('.py') or f.startswith('requirements')]
    files += frozen_inputs()
    if a.lock:
        if '状态：已锁定' not in Path('docs/FINAL_TEST_PROTOCOL.md').read_text():
            raise ValueError('Protocol remains a draft')
        required = ['scripts/run_final_test.py', 'scripts/evaluate_final_downstream.py', 'scripts/evaluate_final_auxiliary.py',
                    'scripts/summarize_final_test.py']
        if not all(f in tracked for f in required):
            raise ValueError('Final execution pipeline is not complete')
    manifest = {'status': 'locked' if a.lock else 'preflight_only', 'checks_passed': True,
                'created_unix': time.time(), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'files': {f: sha256(f) for f in sorted(set(files))},
                'canonical_expected_sha256': json.loads(Path('data/processed/gaf-duration-e2/length512/dataset.json').read_text())['source_package_sha256'],
                'test_access': 'none; canonical package is not opened by preflight'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'complete.json').write_text(json.dumps({'complete': True, 'status': manifest['status']}))
    print(json.dumps({'complete': True, 'status': manifest['status'], 'files': len(manifest['files'])}), flush=True)
except Exception as ex:
    (out / 'failed.json').write_text(json.dumps({'error': str(ex), 'time': time.time()}))
    raise
