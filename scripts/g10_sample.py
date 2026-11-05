from pathlib import Path
import sys,json,numpy as np
from g9_sample import run
cfg=json.loads(Path('runs/long-path-e3/seed42/config.json').read_text())
with np.load('g10-data/requests.npz') as z:
    d={k:z[k] for k in z.files}
assert d['xy'].shape[1:]==(1,2)
run(int(sys.argv[1]),(cfg,cfg['dataset']['source_protocol'],d),(76,),Path('g10-data'),noise_offset=3100000)
