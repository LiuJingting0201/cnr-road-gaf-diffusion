"""Trajectory-state trigger with within-class permuted stopping times as placebo."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from g8_common import Scores,tensor
from g9_sample import weights,turn
from cnr.path_diffusion import PathDenoiser,decode

@torch.no_grad()
def run(seed):
    root=Path('g11-data');dest=root/f'{seed}.npz'
    if dest.exists():raise FileExistsError(dest)
    q=np.load(root/'requests.npz');cfg=json.loads(Path('runs/long-path-e3/seed42/config.json').read_text())
    scores=Scores(cfg['dataset']['source_protocol']);model=PathDenoiser().cuda().eval()
    model.load_state_dict(torch.load(f'runs/long-path-e3/seed{seed}/latest.pt',weights_only=True)['ema'])
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    times=torch.linspace(999,0,100).long().tolist()
    total=len(q['labels'])*3;stages=np.full(total,100,int);tri=torch.triu_indices(8,8,1,device='cuda')
    conditions=np.repeat(np.arange(len(q['labels'])),3);labels=q['labels'][conditions]
    keys=['mid','terminal','placebo','placebo_terminal'];out={k:[] for k in keys};cap={k:[] for k in keys}
    unenc=0
    for phase in ['detect','state','shuffle']:
        if phase=='shuffle':
            assigned=stages.copy();perm=np.random.default_rng(69512)
            for c in np.unique(labels):
                for rep in range(3):
                    ix=np.flatnonzero((labels==c)&(np.arange(total)%3==rep));assigned[ix]=perm.permutation(stages[ix])
        else:assigned=stages
        for begin in range(0,total,8):
            ix=np.arange(begin,min(total,begin+8));ii=conditions[ix];b=len(ix)
            route=tensor(q['route'][ii]).repeat_interleave(8,0);start=tensor(q['xy'][ii,0]).repeat_interleave(8,0)
            L=tensor(q['route_length'][ii]).repeat_interleave(8,0);lab=torch.tensor(labels[ix],device='cuda').repeat_interleave(8)
            cond=((route[:,::8]-start[:,None])/128).transpose(1,2);sn=(start-tensor(cfg['dataset']['mean']))/tensor(cfg['dataset']['std'])
            gens=[torch.Generator(device='cuda').manual_seed(791003+int(k)) for k in ix]
            noise=lambda:torch.cat([torch.randn(8,1,511,device='cuda',generator=g) for g in gens])
            z=noise();innovation=noise();term=z.clone()
            rg=[np.random.default_rng(905831+int(k)) for k in ix];grids=[(np.arange(8)+r.random())/8 for r in rg];slots=[int(r.integers(8)) for r in rg]
            prev=None;prior=torch.zeros(b,device='cuda',dtype=torch.bool);branched=False
            def clean(x,t):
                e=model(x,torch.full_like(lab,t),lab,sn,cond)
                return e,(x-(1-alpha[t]).sqrt()*e)/alpha[t].sqrt()
            def coords(x):return decode(x,cfg['log_increment_mean'],cfg['log_increment_std'],route,L,start)
            def parents(x,k):
                nonlocal unenc
                p,ok,_=scores.image(x.cpu().numpy());unenc+=int(sum(~ok))
                return np.minimum(7,np.searchsorted(weights(p,ok,labels[ix[k]]).cumsum(),grids[k]))
            for j,t in enumerate(times):
                e,h=clean(z,t);ap=alpha[times[j+1]] if j<99 else 1.;adv=ap**.5*h+(1-ap)**.5*e
                if phase=='detect':
                    if (j+1)%5==0 and j<99:
                        p=coords(h)[1].minimum(L[:,None]).reshape(b,8,-1)
                        if prev is not None:
                            drift=((p-prev).square().mean(-1)).sqrt().median(-1).values
                            sep=(p[:,tri[0]]-p[:,tri[1]]).square().mean(-1).sqrt().median(-1).values
                            ok=(sep>0)&(drift<sep);hit=(ok&prior).cpu().numpy()&(stages[ix]==100)
                            stages[ix[hit]]=j+1;prior=ok
                        prev=p
                    z=adv
                    if (stages[ix]<100).all():break
                    continue
                if branched:et,ht=clean(term,t);advt=ap**.5*ht+(1-ap)**.5*et
                else:et,ht=e,h;advt=adv.clone()
                chosen=np.flatnonzero((assigned[ix]==j+1)&(j<99))
                if len(chosen):
                    branched=True;sig=((1-ap)/(1-alpha[t])*(1-alpha[t]/ap)).sqrt()
                    xs=coords(h)[0];base=ap.sqrt()*h+(1-ap-sig.square()).clamp_min(0).sqrt()*e
                    for k in chosen:
                        sl=slice(8*k,8*k+8);ind=parents(xs[sl],k)
                        adv[sl]=base[sl][ind]+sig*innovation[sl]
                        advt[sl]=ap.sqrt()*ht[sl]+(1-ap-sig.square()).clamp_min(0).sqrt()*et[sl]+sig*innovation[sl]
                z,term=adv,advt
            if phase=='detect':continue
            xx,pr=coords(z);xt,pt=coords(term)
            for k in range(b):
                sl=slice(8*k,8*k+8);slot=slots[k];kterm=8*k+parents(xt[sl],k)[slot]
                kmid=kterm if assigned[ix[k]]==100 else 8*k+slot
                for key,x,p,kid in [('mid' if phase=='state' else 'placebo',xx,pr,kmid),('terminal' if phase=='state' else 'placebo_terminal',xt,pt,kterm)]:
                    out[key].append(x[kid].cpu().numpy());cap[key].append(bool(p[kid,-1]>=L[kid]))
        print(seed,phase,'finished',flush=True)
    np.savez_compressed(dest,labels=q['labels'],stage=stages.reshape(-1,3),placebo_stage=assigned.reshape(-1,3),burden=[turn(r,l) for r,l in zip(q['route'],q['route_length'])],failures=unenc,**{k:np.asarray(v).reshape(-1,3,512,2) for k,v in out.items()},**{'cap_'+k:np.array(v).reshape(-1,3) for k,v in cap.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);torch.set_num_threads(2);run(p.parse_args().seed)
