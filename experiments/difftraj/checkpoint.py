"""Replay the released Chengdu sampling cells in an upstream checkout."""
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(exist_ok=True, parents=True)
os.chdir(root)
sys.path.insert(0, str(root))
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.environ['MPLBACKEND'] = 'Agg'
cells = json.loads(Path('process/Traj_Generation.ipynb').read_text())['cells']
env = {'__name__': '__main__'}
for cell in cells:
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source']).replace('utils.Traj_UNe import', 'utils.Traj_UNet import').replace('from tqdm.notebook import tqdm','from tqdm import tqdm').replace("torch.load('./model.pt')","torch.load('./model.pt', weights_only=True)").replace("plt.savefig('Chengdu_traj.png')", f"plt.savefig({str(out/'chengdu.png')!r})")
        exec(compile(code, 'author_notebook', 'exec'), env)
trips = env['Gen_traj']
points = np.concatenate(trips)
assert len(trips) == 500 and np.isfinite(points).all()
summary = dict(seed=42, trips=len(trips), points=len(points), median_samples=float(np.median([len(x) for x in trips])), mean_lon_lat=points.mean(0).tolist())
(out/'summary.json').write_text(json.dumps(summary, indent=2))
print(summary)
