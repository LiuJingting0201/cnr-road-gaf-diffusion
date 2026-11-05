"""Stage-matched, decision-rule-matched guidance on saved development routes."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from g8_common import assets,Scores,tensor
from cnr.path_diffusion import PathDenoiser,decode

OUT=Path('g9-output')
def weights(prob,ok,c):
    w=np.maximum(prob[:,c],1e-6) if np.all(ok) else np.ones(8)
    w=w/w.sum()
    return 1/8+min(1.,np.sqrt((1/6-1/8)/max(np.sum((w-1/8)**2),1e-15)))*(w-1/8)
def turn(path,L):
    d=np.linspace(0,L,len(path))
    q=np.unique(np.r_[np.arange(0,min(1000,L),20),min(1000,L)])
    v=np.diff(np.array([np.interp(q,d,path[:,j]) for j in range(2)]).T,axis=0)
    v=v[np.linalg.norm(v,axis=1)>1e-6]
    return float(abs(np.diff(np.unwrap(np.arctan2(v[:,1],v[:,0])))).sum())
@torch.no_grad()
def run(seed):
    torch.set_num_threads(2)
    cfg,meta,d=assets();sc=Scores(meta);model=PathDenoiser().cuda().eval()
    model.load_state_dict(torch.load(f'runs/long-path-e3/seed{seed}/latest.pt',weights_only=True)['ema'])
    times=torch.linspace(999,0,100).long().tolist()
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device='cuda'),0)
    turn_burden=[turn(r,l) for r,l in zip(d['route'],d['route_length'])]
    for stage in [40,76,91]:
        dest=OUT/f'{seed}-{stage}.npz'
        if dest.exists():raise FileExistsError(dest)
        rows={k:[] for k in ['mid','terminal','maximum','identity']}
        cap={k:[] for k in rows};fails=[0,0]
        for i,label in enumerate(d['labels']):
            start=tensor(d['xy'][i,0]).expand(24,2);route=tensor(d['route'][i]).expand(24,-1,-1)
            length=tensor(d['route_length'][i]).expand(24);cond=((route[:,::8]-start[:,None])/128).transpose(1,2)
            lab=torch.full((24,),int(label),device='cuda');rng=np.random.default_rng(92237+i)
            grid=(np.arange(8)[None,:]+rng.random((3,1)))/8;slot=rng.integers(0,8,3)
            gen=torch.Generator(device='cuda').manual_seed(93711+i)
            z=torch.randn(24,1,511,device='cuda',generator=gen);innov=torch.randn(z.shape,device='cuda',generator=gen)
            def predict(x,j):
                t=times[j];eps=model(x,torch.full_like(lab,t),lab,(start-tensor(cfg['dataset']['mean']))/tensor(cfg['dataset']['std']),cond)
                return eps,(x-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            def coords(x):return decode(x,cfg['log_increment_mean'],cfg['log_increment_std'],route,length,start)
            def choose(x):
                prob,ok,_=sc.image(x.cpu().numpy())
                parents=[];best=[]
                for rep in range(3):
                    ix=slice(8*rep,8*rep+8)
                    parents.extend(8*rep+np.minimum(7,np.searchsorted(weights(prob[ix],ok[ix],label).cumsum(),grid[rep])))
                    best.append(8*rep+(int(np.argmax(prob[ix,label])) if ok[ix].all() else slot[rep]))
                return np.array(parents),np.array(best),int((~ok).sum())
            for j in range(stage):
                ee,cc=predict(z,j);at=alpha[times[j]];ap=alpha[times[j+1]]
                if j<stage-1:z=ap.sqrt()*cc+(1-ap).sqrt()*ee
            parent,_,n=choose(coords(cc)[0]);fails[0]+=n
            sig=((1-ap)/(1-at)*(1-at/ap)).sqrt();common=ap.sqrt()*cc+(1-ap-sig.square()).sqrt()*ee
            outslot=8*np.arange(3)+slot
            for mid in [False,True]:
                zz=(common[parent] if mid else common)+sig*innov
                for j in range(stage,100):
                    ee2,cc2=predict(zz,j);a=alpha[times[j+1]] if j<99 else 1.
                    zz=a**.5*cc2+(1-a)**.5*ee2
                x,p=coords(zz)
                picks={'mid':outslot} if mid else {'identity':outslot}
                if not mid:
                    term,highest,n=choose(x);fails[1]+=n
                    picks.update(terminal=term[outslot],maximum=highest)
                for k,ix in picks.items():
                    rows[k].append(x[ix].cpu().numpy());cap[k].append((p[ix,-1]>=length[ix]).cpu().numpy())
        np.savez_compressed(dest,labels=d['labels'],burden=turn_burden,encoding_failures=fails,**rows,**{'cap_'+k:v for k,v in cap.items()})
        print(seed,stage,'saved',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);run(parser.parse_args().seed)
