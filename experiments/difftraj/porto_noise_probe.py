"""Compare noise prediction from two finished checkpoints on identical development inputs."""
import sys,json
import numpy as np
import torch
from porto import model,schedule
d=np.load('prepared.npz')
ids=np.flatnonzero(d['split']==8)[:64]
x=torch.tensor(((d['xy'][ids]-d['mean'])/d['std']).astype('f')).cuda().transpose(1,2)
h=torch.tensor(d['h'][ids]).cuda()
alpha=(1-schedule()).cumprod(0)
torch.manual_seed(457)
e=torch.randn_like(x)
out={}
for ck in sys.argv[1:]:
    net=model().eval()
    net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['weights'])
    out[ck]={}
    with torch.no_grad():
        for level in [0,50,250,495]:
            a=alpha[level];z=a.sqrt()*x+(1-a).sqrt()*e
            pred=net(z,torch.full((len(x),),level,device='cuda'),h)
            diff=(pred-e)
            out[ck][level]=dict(noise_mse=float(diff.square().mean()),clean_rmse_standardized=float((diff.square().mean()*(1-a)/a).sqrt()),mean_pred=float(pred.mean()),sd_pred=float(pred.std()))
print(json.dumps(out,indent=2))
