"""E4r: change only candidate-route diversity; reuse trained diffusion and K=8 conditions."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from cnr.data import CLASSES
from cnr.path_diffusion import PathDenoiser,decode
from cnr.route_paths import RoutePaths

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
torch.set_num_threads(4)
old=Path(f'runs/long-path-e3/seed{a.seed}');cfg=json.loads((old/'config.json').read_text())
data=Path(cfg['data']);va=np.load(data/'val.npz');reference=np.load(old/'candidates.npz')
assert cfg['dataset']['length']==512
out=Path(f'runs/route-diverse-e4r/seed{a.seed}');out.mkdir(parents=True,exist_ok=False)
rows={int(ix):j for j,ix in enumerate(va['source_index'])}
index=np.array([rows[int(ix)] for ix in reference['source_index']])
labels=reference['labels'];starts=va['xy'][index,0]
assert np.array_equal(labels,va['labels'][index])
matcher=RoutePaths('data/maps/most.net.xml',radius=2.)
lane_index={lane['id']:i for i,lane in enumerate(matcher.road.lanes)}
prior=json.loads((data/'route_prior.json').read_text())
counts={(r['class'],lane_index[r['from']],lane_index[r['to']]):r['count'] for r in prior}
rng=np.random.default_rng(20261107);routes=[];lengths=[];fallback=[]
for j,(start,label) in enumerate(zip(starts,labels)):
    planned=matcher.sample(start,CLASSES[int(label)],counts,rng,budget=4096.)
    failed=planned is None
    if failed:planned=va['route'][index[j]],float(va['route_length'][index[j]])
    route,length=planned;routes.append(route);lengths.append(length);fallback.append(failed)
routes=np.array(routes,dtype=np.float32);lengths=np.array(lengths,dtype=np.float32)
groups=[np.flatnonzero(reference['condition_id']==c) for c in np.unique(reference['condition_id'])]
assert all(len(g)==8 for g in groups)
unique_routes=[len({hashlib.sha256(routes[i].tobytes()).hexdigest() for i in g}) for g in groups]
checkpoint=old/'latest.pt';model=PathDenoiser().cuda().eval()
model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['ema'])
to=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
mean=to(cfg['dataset']['mean']);std=to(cfg['dataset']['std'])
alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
times=torch.linspace(999,0,100).long().tolist();torch.manual_seed(2026)
all_xy=[];all_progress=[]
with torch.no_grad():
    for i in range(0,len(labels),32):
        route=to(routes[i:i+32]);length=to(lengths[i:i+32]);start=to(starts[i:i+32]);label=torch.as_tensor(labels[i:i+32],device='cuda')
        condition=((route[:,::8]-start[:,None])/128).transpose(1,2)
        sample=torch.randn(len(label),1,511,device='cuda')
        for j,t in enumerate(times):
            eps=model(sample,torch.full((len(label),),t,device='cuda'),label,(start-mean)/std,condition)
            clean=(sample-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            prev=alpha[times[j+1]] if j+1<len(times) else 1.
            sample=prev**.5*clean+(1-prev)**.5*eps
        xy,progress=decode(sample,cfg['log_increment_mean'],cfg['log_increment_std'],route,length,start)
        assert torch.isfinite(xy).all()
        all_xy.append(xy.cpu().numpy());all_progress.append(progress.cpu().numpy())
xy=np.concatenate(all_xy);progress=np.concatenate(all_progress)
np.savez_compressed(out/'candidates.npz',xy=xy,progress=progress,labels=labels,source_index=reference['source_index'],condition_id=reference['condition_id'],route_length=lengths,route=routes)
newcfg=cfg|{'steps':0,'sampling_only':True,'pretrained_steps':20000,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'route_sampling_seed':20261107,'noise_seed':2026,'scope':'Same 33 validation start/class conditions, same K=8 and DDIM noise order as E3. Each draw independently samples a legal path from training-only transition prior; no future validation route. No retraining.'}
(out/'config.json').write_text(json.dumps(newcfg,indent=2))
(out/'complete.json').write_text(json.dumps({'complete':True,'samples':len(labels),'route_planning_fallback':int(sum(fallback)),
    'mean_unique_routes_per_pool':float(np.mean(unique_routes)),'identical_route_pool_fraction':float(np.mean(np.array(unique_routes)==1)),
    'route_end_capped_fraction':float(np.mean(progress[:,-1]>=lengths)),'next':'Same-pool original GAF/TCN/random/shuffled evaluation; no downstream gain claim'},indent=2))
