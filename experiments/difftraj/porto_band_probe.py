"""Counterfactual sampler diagnosis, old/retrained predictors above or below a noise level."""
import json
import numpy as np
import torch
from porto import model,schedule,arc_length
from utils.utils import p_xt,resample_trajectory
d=np.load('prepared.npz')
ids=np.random.default_rng(145).choice(np.flatnonzero(d['split']==8),64,replace=False)
raw=np.load('observations.npz')
ends=np.r_[0,raw['length'].cumsum()]
delta=raw['delta']
true=np.array([arc_length(delta[ends[i]:ends[i+1]].cumsum(0)*1e-6) for i in ids])
models=[]
for ck in ['../difftraj-nov11/porto/checkpoint.pt','../porto-cutoff-v1/long/checkpoint.pt']:
    net=model().eval()
    net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['weights'])
    models.append(net)
beta=schedule()
h=torch.tensor(d['h'][ids]).cuda()
out={}
with torch.no_grad():
    for hi,lo in [(0,0),(1,1),(0,1),(1,0)]:
        torch.manual_seed(173)
        lengths=[]
        for offset,batch in enumerate(h.split(16)):
            z=torch.randn(len(batch),2,200,device='cuda')
            for step in reversed(range(0,500,5)):
                t=torch.full((len(z),),step,device='cuda')
                z=p_xt(z,models[hi if step>=250 else lo](z,t,batch),t,torch.full_like(t,max(-1,step-5)),beta,0.)
            tracks=z.transpose(1,2).cpu().numpy()*d['std']+d['mean']
            for j,g in enumerate(tracks):
                lengths.append(arc_length(resample_trajectory(g,int(d['sizes'][ids[offset*16+j]]))))
        out[f'high{hi}-low{lo}']=dict(median_length_ratio=float(np.median(lengths/true)),median_generated_km=float(np.median(lengths)/1000),mean_excess_km=float(np.mean(lengths-true)/1000))
print(json.dumps(out,indent=2))
