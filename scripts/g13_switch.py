"""Paired motion interventions on new SUMO starts. Never read SUMO futures."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from g8_common import tensor
from cnr.path_diffusion import PathDenoiser,decode
from cnr.topology import LaneConsistency
from cnr.data import CLASSES

@torch.no_grad()
def run(seed):
    root=Path('g13-data');q=np.load(root/'requests.npz')
    cfg=json.loads(Path('runs/long-path-e3/seed42/config.json').read_text())
    graph=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64)
    def valid(x,c):
        g=graph.match(x,CLASSES[c]);return bool(g['graph_consistent'] and not g['candidate_truncation_warning'])
    model=PathDenoiser().cuda().eval()
    model.load_state_dict(torch.load(f'runs/long-path-e3/seed{seed}/latest.pt',weights_only=True)['ema'])
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    times=torch.linspace(999,0,100).long().tolist();lab=torch.tensor([1]*8+[3]*8,device='cuda')
    rows=[];example=None
    for i in np.flatnonzero(np.isin(q['labels'],[1,3])):
        path=q['route'][i];entry=dict(id=str(q['ids'][i]),index=int(i),episode=int(q['episode'][i]),permitted=[valid(path,c) for c in [1,3]])
        rows.append(entry)
        if not all(entry['permitted']):continue
        start=tensor(q['xy'][i,0]).expand(16,2);route=tensor(path).expand(16,-1,-1);L=tensor(q['route_length'][i]).expand(16)
        cond=((route[:,::8]-start[:,None])/128).transpose(1,2)
        rng=torch.Generator(device='cuda').manual_seed(1043201+int(i))
        z=torch.randn(8,1,511,device='cuda',generator=rng).repeat(2,1,1)
        for j,t in enumerate(times):
            eps=model(z,torch.full_like(lab,t),lab,(start-tensor(cfg['dataset']['mean']))/tensor(cfg['dataset']['std']),cond)
            clean=(z-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            ap=alpha[times[j+1]] if j<99 else 1.
            z=ap**.5*clean+(1-ap)**.5*eps
        xy,p=decode(z,cfg['log_increment_mean'],cfg['log_increment_std'],route,L,start)
        x=xy.cpu().numpy()
        for c,sl in [(1,slice(0,8)),(3,slice(8,16))]:
            v=np.linalg.norm(np.diff(x[sl],axis=1),axis=-1)/.25
            entry[CLASSES[c]]=dict(travel=(v.sum(1)*.25).tolist(),stop=(v<.1).mean(1).tolist(),
              valid=sum(valid(y,c) for y in x[sl]),caps=int((p[sl,-1]>=L[sl]).sum().item()))
        if seed==42 and example is None:
            example=dict(condition=int(i),xy=x[[0,8]].tolist(),route=path.tolist(),slot=0,seed=42)
    if example:(root/'example.json').write_text(json.dumps(example))
    (root/f'switch{seed}.json').write_text(json.dumps(dict(seed=seed,rows=rows)))
if __name__=='__main__':torch.set_num_threads(2);run(int(sys.argv[1]))
