"""One-shot launch after a complete, checksum-verified dataset arrives."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--steps', type=int, default=10000)
p.add_argument('--data', default='data/processed/most64')
p.add_argument('--manifest', default='docs/processed_hashes.json')
p.add_argument('--timeout-hours', type=float, default=12)
a = p.parse_args()
expected = json.loads(Path(a.manifest).read_text())
deadline = time.monotonic()+a.timeout_hours*3600
while time.monotonic() < deadline:
    missing = []
    for name, digest in expected.items():
        f = Path(a.data)/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            missing.append(name)
    if not missing:
        print('All processed dataset checksums verified; launching baselines.', flush=True)
        subprocess.run([sys.executable, 'scripts/launch_pilots.py', '--name', a.name,
                        '--steps', str(a.steps), '--data', a.data], check=True)
        break
    print(json.dumps({'awaiting_verified_files': missing}), flush=True)
    time.sleep(30)
else:
    raise SystemExit('Dataset readiness timeout; no training launched')
