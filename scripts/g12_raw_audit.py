"""Descriptive frozen coordinate-classifier control on identical new SUMO IDs."""
import json
import numpy as np
import torch
from pathlib import Path
from cnr.representations import RepresentationClassifier
from g12_fit import metrics
@torch.no_grad()
def run():
    q=np.load('g12-data/requests.npz');r=np.load('g12-data/references.npz')
    assert np.array_equal(q['ids'],r['ids'])
    x=torch.tensor(r['xy'],device='cuda')
    result=[]
    for seed in [42,43,44]:
        p=Path(f'runs/long-selection-e4/tcn-seed{seed}')
        model=RepresentationClassifier('xy_motion_tcn',json.loads((p/'config.json').read_text())['stats']).cuda().eval()
        model.load_state_dict(torch.load(p/'latest.pt',weights_only=True)['model'])
        result.append(metrics(torch.cat([model(b) for b in x.split(32)]).cpu().numpy(),q['labels'],range(1,6)))
    Path('g12-data/raw_audit.json').write_text(json.dumps(result,indent=1))
if __name__=='__main__':torch.set_num_threads(2);run()
