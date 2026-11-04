"""Wait for canonical data and G2 to release GPUs, then run the registered R2."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

source=Path('data/canonical/most.npz.xz')
expected='9e733b8e97ab9d0bfb3e8f101c8ce2bae65daaa308cc626be447544c838e9d3a'
deadline=time.monotonic()+12*3600
while time.monotonic()<deadline:
    source_ok=source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==expected
    gpu_released=Path('runs/road-g2/study_complete.json').is_file()
    if source_ok and gpu_released:break
    print(json.dumps({'canonical_data_verified':source_ok,'g2_jobs_finished':gpu_released}),flush=True)
    time.sleep(30)
else:raise SystemExit('Readiness timeout; no R2 training launched')
env=os.environ.copy();env.update(PYTHONPATH='src',OMP_NUM_THREADS='4')
subprocess.run([sys.executable,'scripts/prepare_data.py','--source',str(source),
                '--output','data/processed/most224','--length','224'],env=env,check=True)
subprocess.run([sys.executable,'scripts/run_representation_study.py','--name','representation-r2-long',
                '--data','data/processed/most224','--steps','4000'],env=env,check=True)
subprocess.run([sys.executable,'scripts/summarize_representation_study.py','--study','runs/representation-r2-long',
                '--output','docs/results/representation-r2-long'],env=env,check=True)
