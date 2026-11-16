"""Matched one-step speed-target errors and frozen-checkpoint training receipts."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from five_class_train import make_model

root=Path('/workspace/cnr-experiments/five-class-difftraj-200k-v1')
data=Path('/workspace/cnr-experiments/difftraj-five-40k-v1')
rows=np.flatnonzero(np.load(data/'validation/index.npz')['classes']=='bus')
assert len(rows)==1000
def tensor(x):
    return torch.as_tensor(x,dtype=torch.float32,device='cuda')
xy=tensor(np.load(data/'validation/traj.npy')[rows]).transpose(1,2)
h=tensor(np.load(data/'validation/head.npy')[rows])
v=tensor(np.load(root/'prepared/validation/v.npy')[rows])
dt=tensor(np.load(root/'prepared/validation/dt.npy')[rows])[:,None]
std=tensor(np.load(data/'normalization.npz')['std'])[None,:,None]
noise=torch.randn(xy.shape,generator=torch.Generator().manual_seed(116)).cuda()
a=(1-torch.linspace(.0001,.05,500,device='cuda')).cumprod(0)
result={}
with torch.inference_mode():
    for arm in ('A','B'):
        for seed in (42,43):
            run=root/f'runs/bus-{arm}-{seed}'
            model=make_model(arm).eval()
            model.load_state_dict(torch.load(run/'final.pt',weights_only=True,map_location='cpu')['weights'])
            log=json.loads((run/'epochs.jsonl').read_text().splitlines()[-1])
            entry=dict(final_noise_loss=log['noise_loss'],final_weighted_speed_loss=log['weighted_speed_loss'],levels={})
            for level in (10,100,250,400):
                alpha=a[level]
                z=alpha.sqrt()*xy+(1-alpha).sqrt()*noise
                pred=torch.cat([model(z[i:i+64],torch.full((len(z[i:i+64]),),level,device='cuda'),h[i:i+64],torch.ones(len(z[i:i+64]),device='cuda',dtype=torch.long)) for i in range(0,len(z),64)])
                clean=(z-(1-alpha).sqrt()*pred)/alpha.sqrt()
                speed=(clean.diff(dim=2)*std).norm(dim=1)/dt
                entry['levels'][level]=dict(log_speed_loss=F.smooth_l1_loss(speed.log1p(),v.log1p()).item(),speed_mae=(speed-v).abs().mean().item(),weight=.05*alpha.item())
            result[f'{arm}-{seed}']=entry
print(json.dumps(result,indent=2))
