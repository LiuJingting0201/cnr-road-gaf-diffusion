"""Held-out reconstruction at known noise levels plus an exact-noise DDIM control."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import torch
from utils.config import args
from utils.Traj_UNet import Guide_UNet
from utils.utils import p_xt
cfg=SimpleNamespace(**{k:SimpleNamespace(**v) for k,v in args.items()})
net=Guide_UNet(cfg).cuda().eval()
net.load_state_dict(torch.load('B-2000.pt',weights_only=True))
d=np.load('pilot.npz')
xy=torch.tensor(d['xy'],device='cuda')
mask=d['train']
mu=xy[mask].mean((0,2))[None,:,None]
sd=xy[mask].std((0,2))[None,:,None]
ids=np.flatnonzero(~mask)[:32]
x=(xy[ids]-mu)/sd
h=torch.tensor(d['h'][ids],device='cuda')
torch.manual_seed(42)
noise=torch.randn_like(x)
beta=torch.linspace(.0001,.05,500,device='cuda')
ab=(1-beta).cumprod(0)
summary={}
with torch.no_grad():
    for level in [50,250,495]:
        t=torch.full((len(x),),level,device='cuda')
        z=ab[level].sqrt()*x+(1-ab[level]).sqrt()*noise
        pred=net(z,t,h)
        clean=(z-(1-ab[level]).sqrt()*pred)/ab[level].sqrt()
        err=(clean-x)*sd
        summary[level]=dict(noise_mse=float((noise-pred).square().mean()),position_rmse_m=float(err.square().mean().sqrt()),trip_center_rmse_m=float(err.mean(-1).square().mean().sqrt()))
    oracle=z
    for step in reversed(range(0,500,5)):
        t=torch.full((len(x),),step,device='cuda')
        eps=(oracle-ab[step].sqrt()*x)/(1-ab[step]).sqrt()
        oracle=p_xt(oracle,eps,t,torch.full_like(t,max(-1,step-5)),beta,0.)
    summary['oracle_position_rmse_m']=float(((oracle-x)*sd).square().mean().sqrt())
print(json.dumps(summary,indent=2))
