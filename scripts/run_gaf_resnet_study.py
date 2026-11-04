"""Fixed E1 factorial queue: four GPUs, three seeds per cell; fails visibly."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/gaf-resnet-e1')
root.mkdir(parents=True, exist_ok=False)
cells = [(32, 'random'), (32, 'imagenet'), (64, 'random'), (64, 'imagenet')]
plan = {'cells': cells, 'seeds': [42, 43, 44], 'epochs': 50, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'pid': os.getpid(), 'started_unix': time.time()}
(root/'plan.json').write_text(json.dumps(plan, indent=2))


def worker(item):
    gpu, (grid, init) = item
    env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': 'src', 'OMP_NUM_THREADS': '4'}
    data = 'data/processed/original-hilbert224' if grid == 32 else 'data/processed/original-hilbert64grid224'
    if init == 'imagenet':
        # Preflight caches the official weights once; avoid concurrent downloads.
        import torch
        cache = Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
        deadline = time.monotonic() + 1800
        (root/f'gpu{gpu}-waiting-weights.json').write_text(json.dumps({'cache': str(cache), 'started_unix': time.time()}))
        while not cache.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError('Official ResNet18 weight preflight did not finish within 30 minutes')
            time.sleep(5)
        # Torch hub verifies the hash on download; independently verify cached bytes.
        import hashlib
        if not hashlib.sha256(cache.read_bytes()).hexdigest().startswith('f37072fd'):
            raise RuntimeError('Official weight cache hash mismatch')
        (root/f'gpu{gpu}-weights-ready.json').write_text(json.dumps({'time': time.time()}))
    completed = []
    for seed in [42, 43, 44]:
        name = f'grid{grid}-{init}-seed{seed}'
        cmd = [sys.executable, '-u', 'scripts/train_gaf_resnet.py', '--data', data, '--initialization', init, '--seed', str(seed), '--output', str(root/name)]
        (root/f'{name}.job.json').write_text(json.dumps({'cmd': cmd, 'gpu': gpu, 'started_unix': time.time()}, indent=2))
        with (root/f'{name}.log').open('x') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            (root/f'{name}.failed.json').write_text(json.dumps({'returncode': result.returncode, 'time': time.time()}))
            raise RuntimeError(f'{name} failed: {result.returncode}')
        assert json.loads((root/name/'complete.json').read_text())['complete']
        completed.append(name)
    return completed


try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(worker, enumerate(cells)))
    (root/'study_complete.json').write_text(json.dumps({'complete': True, 'runs': done, 'time': time.time()}, indent=2))
except Exception as ex:
    (root/'failed.json').write_text(json.dumps({'error': str(ex), 'time': time.time()}))
    raise
