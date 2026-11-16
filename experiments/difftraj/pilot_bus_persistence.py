"""Development pilot adding physical-time continuous low-speed window statistics."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from five_class_evaluate import features, motion_scores, paper_scores, occupancy, read_observations
from probe_sampling_guidance import DATA, OLD, QUANTILES, make_model, p_xt, resample_trajectory
from bus_low_speed_spells import describe

def windows(speed, dt):
    # Each sample's time step differs. Group windows with the same interval count.
    output=[]
    for seconds in (10,20,30):
        widths=torch.ceil(seconds/dt).long().clamp(1,199)
        stat=torch.zeros_like(dt)
        for width in torch.unique(widths).tolist():
            mask=widths==width
            peaks=F.max_pool1d(speed[mask,None],width,stride=1)
            stat[mask]=torch.sigmoid((1-peaks)/.25).mean((1,2))
        output.append(stat)
    return torch.stack(output,dim=1)

def run(seed):
    from bus_low_speed_spells import ids,obs,dur
    from five_class_evaluate import stratify
    with np.load(DATA/'validation/index.npz') as z:
        pool,_=stratify(z['classes'])
        prev=pool[np.isin(z['classes'][pool],['bus','bicycle'])]
        allrows=np.flatnonzero(np.isin(z['classes'],['bus','bicycle']))
        rows=np.r_[prev,np.setdiff1d(allrows,prev)]
        select=np.array([np.flatnonzero(z['ids'][rows]==k)[0] for k in ids])
    rng=np.random.default_rng(42)
    noise=np.concatenate((rng.standard_normal((len(pool),2,200),dtype='f')[np.isin(pool,prev)],
                          rng.standard_normal((800,2,200),dtype='f')))[select]
    head=np.load(DATA/'validation/head.npy')[rows[select]]
    train=np.load(DATA/'train/head.npy')
    with np.load(DATA/'normalization.npz') as z: scale=dict(z)
    bus=np.flatnonzero(np.load(OLD/'prepared/train/y.npy')==1)
    _,near=cKDTree(train[bus][:,[2,5]]).query(head[:,[2,5]],k=32)
    tracks=np.load(DATA/'train/traj.npy',mmap_mode='r')
    asf=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    sd=asf(scale['std'])[None,:,None]
    qtar=[]; wtar=[]
    for sub in np.array_split(bus[near],32):
        flat=sub.ravel()
        step=(train[flat,2]*scale['header_std'][1]+scale['header_mean'][1])/199
        v=np.linalg.norm(np.diff(tracks[flat],axis=1)*scale['std'],axis=2)/step[:,None]
        qtar.extend(np.median(np.quantile(v,QUANTILES,axis=1).T.reshape(-1,32,5),axis=1))
        wtar.extend(windows(asf(v),asf(step)).reshape(-1,32,3).median(1).values.cpu().numpy())
    qt,wt=asf(np.asarray(qtar)),asf(np.asarray(wtar))
    beta=torch.linspace(.0001,.05,500,device='cuda'); alpha=(1-beta).cumprod(0)
    net=make_model('C').eval()
    net.load_state_dict(torch.load(OLD/f'runs/all-C-{seed}/final.pt',map_location='cpu',weights_only=True)['weights'])
    output=[]
    for i in range(0,len(head),64):
        sl=slice(i,i+64)
        z,h,dt=asf(noise[sl]),asf(head[sl]),asf(dur[sl]/199)
        y=torch.ones(len(z),device='cuda',dtype=torch.long)
        for step in reversed(range(0,500,5)):
            t=torch.full_like(y,step)
            with torch.no_grad(): eps=net(z,t,h,y)
            a=alpha[step]
            clean=(((z-(1-a).sqrt()*eps)/a.sqrt())*sd).detach().requires_grad_()
            v=clean.diff(dim=2).norm(dim=1)/dt[:,None]
            sorted_v=v.sort(dim=1).values.log1p()
            bands=torch.stack([sorted_v[:,round(q*198)-5:round(q*198)+6].mean(1) for q in QUANTILES],1)
            energy=F.smooth_l1_loss(bands,qt[sl].log1p(),reduction='sum')/5
            energy+=F.smooth_l1_loss(windows(v,dt),wt[sl],reduction='sum')/3
            grad,=torch.autograd.grad(energy,clean)
            direction=grad/grad.norm(dim=1,keepdim=True).amax(dim=2,keepdim=True).clamp_min(1.e-8)
            direction[:,:,[0,-1]]=0
            eps=eps+(a/(1-a)).sqrt()*2*a*direction/sd
            z=p_xt(z,eps,t,torch.full_like(t,max(-1,step-5)),beta,0.)
        output.append(z.transpose(1,2).cpu().numpy()*scale['std']+scale['mean'])
    xy=np.concatenate(output); f,v=features(xy,dur); r,rv=features(obs,dur)
    raw=read_observations(DATA,ids); ii=np.arange(len(xy))
    occ=lambda x:occupancy(x,scale['lo'],scale['hi'])
    bins=np.quantile(train[:,1].astype('d')*scale['header_std'][0]+scale['header_mean'][0],np.arange(1,20)/20)
    result=dict(spells=describe(xy)[1],motion=motion_scores(r,f,rv,v,ii),paper=paper_scores(occ(raw),occ([resample_trajectory(x,int(d)+1) for x,d in zip(xy,dur)]),ii,ii,bins))
    Path(f'persistence-{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    import sys
    run(int(sys.argv[1]))
