"""Fixed-noise endpoint interventions using training-supported alternate conditions."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from porto import model,schedule,grid,p_xt
seed=sys.argv[1]; base=Path('/workspace/cnr-experiments/taxi10k-base')
p=np.load(base/'prepared.npz');raw=np.load(base/'observations.npz')
h=p['h'];xy=p['xy'];lo,hi=p['lo'],p['hi'];w=(hi-lo)/16
def center(c):return lo+(np.stack([c//16,c%16],axis=-1)+.5)*w
cells=h[:,6:].astype(int)
ends=np.r_[0,raw['length'].cumsum()]
native=np.array([np.cumsum(a,axis=0)[[0,-1]] for a in np.split(raw['delta'],ends[1:-1])])*float(raw['planar_unit_m'])
def mismatch(v):return int(np.sum(grid(v,lo,hi)!=cells))
roundtrip=(((xy-p['mean'])/p['std']).astype('f')*p['std']+p['mean'])
audit=dict(native=mismatch(native),prepared=mismatch(xy[:,[0,-1]]),roundtrip=mismatch(roundtrip[:,[0,-1]]),total=int(cells.size))
net=model().eval();net.load_state_dict(torch.load(base/seed/'checkpoint.pt',weights_only=True)['weights'])
beta=schedule()
def sample(head,z):
    with torch.no_grad():
        for t0 in reversed(range(0,500,5)):
            t=torch.full((len(z),),t0,device='cuda')
            z=p_xt(z,net(z,t,head),t,torch.full_like(t,max(-1,t0-5)),beta,0.)
    if '--map-examples' in sys.argv:
        return z.transpose(1,2).cpu().numpy()*p['std']+p['mean']
    return z[:,:,[0,-1]].transpose(1,2).cpu().numpy()*p['std']+p['mean']
train=np.flatnonzero(p['split']<8);test=np.flatnonzero(p['split']==9)
freq=np.bincount(cells[train,0]*256+cells[train,1],minlength=65536)
def support(i):return freq[cells[i,0]*256+cells[i,1]]
results={}
for k,name in enumerate(['origin','destination']):
    ids=[];donors=[]
    for i in test:
        eligible=train[(cells[train,1-k]==cells[i,1-k])&(support(train)>=2)]
        eligible=eligible[np.linalg.norm(center(cells[eligible,k])-center(cells[i,k]),axis=1)>=1500]
        if not len(eligible):continue
        gap=np.abs(h[eligible,1:6]-h[i,1:6]); clock=np.abs(h[eligible,0]-h[i,0]);clock=np.minimum(clock,288-clock)
        mask=(gap.max(1)<=.75)&(clock<=36)
        if not mask.any():continue
        cost=gap.sum(1)+clock/36
        j=eligible[np.argmin(np.where(mask,cost,np.inf))]
        ids.append(i);donors.append(j)
    ids=np.array(ids); donors=np.array(donors)
    if '--map-examples' in sys.argv:
        if name=='origin':continue
        ids=ids[:32];donors=donors[:32]
    orig=[];alt=[]
    for start in range(0,len(ids),32):
        ix=ids[start:start+32];jx=donors[start:start+32]
        a=torch.tensor(h[ix]).cuda();b=a.clone();b[:,6+k]=torch.tensor(h[jx,6+k]).cuda()
        torch.manual_seed(1800+start)
        z=torch.randn(len(ix),2,200,device='cuda')
        orig.extend(sample(a,z.clone()));alt.extend(sample(b,z.clone()))
    orig=np.array(orig);alt=np.array(alt)
    if '--map-examples' in sys.argv:
        np.savez(f'map-{seed}.npz',before=orig[:3],after=alt[:3],reference=xy[ids[:3]],A=center(cells[ids[:3],k]),B=center(cells[donors[:3],k]),O=center(cells[ids[:3],1-k]),width=w)
        continue
    A=center(cells[ids,k]);B=center(cells[donors,k]);v=B-A;u=alt[:,k]-orig[:,k]
    ratio=(u*v).sum(1)/(v*v).sum(1); drift=np.linalg.norm(alt[:,1-k]-orig[:,1-k],axis=1)
    closer=np.linalg.norm(alt[:,k]-B,axis=1)<np.linalg.norm(orig[:,k]-B,axis=1)
    hit=grid(alt[:,k],lo,hi)==cells[donors,k]
    results[name]=dict(n=len(ids),median_target_m=float(np.median(np.linalg.norm(v,axis=1))),fraction_forward=float((ratio>0).mean()),median_projected_fraction=float(np.median(ratio)),median_moved_m=float(np.median(np.linalg.norm(u,axis=1))),median_other_end_drift_m=float(np.median(drift)),fraction_closer=float(closer.mean()),new_cell_hit=float(hit.mean()))
    np.savez(f'{seed}-{name}.npz',A=A[:6],B=B[:6],before=orig[:6,k],after=alt[:6,k])
if '--map-examples' not in sys.argv:Path(f'{seed}.json').write_text(json.dumps(dict(coding=audit,results=results),indent=2))
