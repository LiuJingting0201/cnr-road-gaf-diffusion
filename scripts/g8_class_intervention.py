"""Within-path, within-noise bus vs car intervention. Development only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from g8_common import assets,Scores,tensor,reference
from cnr.path_diffusion import PathDenoiser,decode
from cnr.topology import LaneConsistency
from cnr.data import CLASSES

@torch.no_grad()
def run(seed):
    cfg,meta,d=assets();sc=Scores(meta)
    matcher=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64)
    model=PathDenoiser().cuda().eval()
    model.load_state_dict(torch.load(f'runs/long-path-e3/seed{seed}/latest.pt',weights_only=True)['ema'])
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    times=torch.linspace(999,0,100).long().tolist()
    def valid(x,c):
        g=matcher.match(x,CLASSES[c])
        return g['graph_consistent'] and not g['candidate_truncation_warning']
    rows=[]
    for i in np.flatnonzero(np.isin(d['labels'],[1,3])):
        path=d['route'][i];row=dict(condition=int(i),permitted=[valid(path,c) for c in [1,3]])
        rows.append(row)
        if not all(row['permitted']):continue
        start=tensor(d['xy'][i,0]).expand(16,2);route=tensor(path).expand(16,-1,-1)
        length=tensor(d['route_length'][i]).expand(16)
        cond=((route[:,::8]-start[:,None])/128).transpose(1,2)
        lab=torch.tensor([1]*8+[3]*8,device='cuda')
        rng=torch.Generator(device='cuda').manual_seed(82001+int(i))
        z=torch.randn(8,1,511,device='cuda',generator=rng).repeat(2,1,1)
        for j,t in enumerate(times):
            eps=model(z,torch.full((16,),t,device='cuda'),lab,(start-tensor(cfg['dataset']['mean']))/tensor(cfg['dataset']['std']),cond)
            clean=(z-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            ap=alpha[times[j+1]] if j<99 else 1.
            z=ap**.5*clean+(1-ap)**.5*eps
        xy,p=decode(z,cfg['log_increment_mean'],cfg['log_increment_std'],route,length,start)
        x=xy.cpu().numpy();aud=sc.raw(x);teacher,ok,_=sc.image(x)
        for c,idx in [(1,slice(0,8)),(3,slice(8,16))]:
            v=np.linalg.norm(np.diff(x[idx],axis=1),axis=-1)/.25
            row[CLASSES[c]]=dict(travel=(v.sum(1)*.25).tolist(),stop=(v<.1).mean(1).tolist(),
                tcn_bus=aud[idx,1].tolist(),gaf_bus=teacher[idx,1].tolist(),encodable=int(ok[idx].sum()),
                valid=sum(valid(y,c) for y in x[idx]),caps=int((p[idx,-1]>=length[idx]).sum().item()))
    real=reference(cfg,d)
    obs={}
    for c in [1,3]:
        v=np.linalg.norm(np.diff(real[d['labels']==c],axis=1),axis=-1)/.25
        obs[CLASSES[c]]=dict(travel=float(np.mean(v.sum(1)*.25)),stop=float(np.mean(v<.1)))
    Path('pilot-output/intervention.json' if seed==42 else f'pilot-output/intervention-{seed}.json').write_text(json.dumps(dict(complete=True,rows=rows,observed=obs),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=42)
    torch.set_num_threads(2);run(parser.parse_args().seed)
