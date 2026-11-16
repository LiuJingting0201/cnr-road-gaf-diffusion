"""Paired bus/bicycle sampling on the pre-existing 113 development pairs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from five_class_train import make_model
from five_class_evaluate import (stratify, match_bus_bicycle, features, motion_scores,
    read_observations, occupancy, paper_scores, FEATURES)
from utils.utils import p_xt, resample_trajectory

DATA = Path('/workspace/cnr-experiments/difftraj-five-40k-v1')
OLD = Path('/workspace/cnr-experiments/five-class-difftraj-200k-v1')
QUANTILES = [.1, .25, .5, .75, .9]


def speed_targets(head, labels, scale):
    # Similar-duration and similar-mean-speed training requests within the target class.
    train = np.load(DATA/'train/head.npy')
    classes = np.load(OLD/'prepared/train/y.npy')
    tracks = np.load(DATA/'train/traj.npy', mmap_mode='r')
    targets = np.empty((len(head), len(QUANTILES)))
    for label in np.unique(labels):
        rows = np.flatnonzero(classes == label)
        mask = labels == label
        tree = cKDTree(train[rows][:, [2, 5]])
        _, near = tree.query(head[mask][:, [2, 5]], k=32)
        for i, neighbors in zip(np.flatnonzero(mask), rows[near]):
            dt = (train[neighbors, 2]*scale['header_std'][1]+scale['header_mean'][1])/199
            v = np.linalg.norm(np.diff(tracks[neighbors], axis=1)*scale['std'], axis=2)/dt[:, None]
            targets[i] = np.median(np.quantile(v, QUANTILES, axis=1), axis=1)
    return targets


def draw(net, kind, head, label, noise, scale):
    beta = torch.linspace(.0001, .05, 500, device='cuda')
    alpha = (1-beta).cumprod(0)
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device='cuda')
    sd = tensor(scale['std'])[None, :, None]
    dt = tensor((head[:, 2]*scale['header_std'][1]+scale['header_mean'][1])/199)
    if kind == 'speed':
        targets = tensor(speed_targets(head, label, scale))
    output = []
    for start in range(0, len(head), 64):
        ix = slice(start,start+64)
        h, y, z = tensor(head[ix]), torch.as_tensor(label[ix], device='cuda'), tensor(noise[ix])
        for step in reversed(range(0,500,5)):
            t = torch.full((len(y),), step, device='cuda')
            with torch.no_grad():
                eps = net(z,t,h,y)
                if kind == 'class':
                    eps += 0.5*(eps-net(z,t,h,1-y))
            a = alpha[step]
            if kind == 'speed':
                # Differentiate only w.r.t. estimated clean positions in metres.
                clean = (((z-(1-a).sqrt()*eps)/a.sqrt())*sd).detach().requires_grad_()
                v = clean.diff(dim=2).norm(dim=1)/dt[ix,None]
                # Smooth distribution objective: use finite-width order-statistic bands
                # rather than a derivative concentrated at a single quantile.
                sorted_v = torch.log1p(v.sort(dim=1).values)
                bands = torch.stack([sorted_v[:,round(q*198)-5:round(q*198)+6].mean(1) for q in QUANTILES], dim=1)
                energy = torch.nn.functional.smooth_l1_loss(bands, torch.log1p(targets[ix]), reduction='none').mean(1)
                grad, = torch.autograd.grad(energy.sum(), clean)
                direction = grad / grad.norm(dim=1, keepdim=True).amax(dim=2,keepdim=True).clamp_min(1.e-8)
                direction[:,:,0] = 0
                direction[:,:,-1] = 0
                # Cap displacement in the clean estimate at two metres, attenuated by signal power.
                eps = eps + (a/(1-a)).sqrt()*2*a*direction/sd
            z = p_xt(z,eps,t,torch.full_like(t,max(-1,step-5)),beta,0.)
        output.append(z.transpose(1,2).cpu().numpy()*scale['std']+scale['mean'])
    return np.concatenate(output)


def main(seed, kind):
    folder=DATA/'validation'
    with np.load(folder/'index.npz') as z:
        ids, labels = z['ids'],z['classes']
    with np.load(DATA/'normalization.npz') as z:
        scale=dict(z)
    head=np.load(folder/'head.npy')
    pool,_=stratify(labels)
    bike,bus=match_bus_bicycle(head[pool], labels[pool])
    slots=np.r_[bike,bus]
    assert len(bike)==113
    rows=pool[slots]
    y=(labels[rows]=='bus').astype('i8')
    noise=np.random.default_rng(42).standard_normal((len(pool),2,200),dtype=np.float32)[slots]
    ck=torch.load(OLD/f'runs/all-C-{seed}/final.pt',map_location='cpu',weights_only=True)
    net=make_model('C').eval()
    net.load_state_dict(ck['weights'])
    obs=np.load(folder/'traj.npy',mmap_mode='r')[rows]*scale['std']+scale['mean']
    raw=read_observations(DATA,ids[rows])
    duration=np.array([len(p)-1 for p in raw])
    known,known_v=features(obs,duration)
    tr=np.load(DATA/'train/head.npy',mmap_mode='r')[:,1]
    bins=np.unique(np.quantile(tr.astype('d')*scale['header_std'][0]+scale['header_mean'][0],np.arange(1,20)/20))
    occ=occupancy(raw,scale['lo'],scale['hi'])
    result={}
    for method in ('base',kind):
        direct=draw(net,method,head[rows],y,noise,scale)
        flip=draw(net,method,head[rows],1-y,noise,scale)
        f,v=features(direct,duration)
        g,_=features(flip,duration)
        gen_occ=occupancy([resample_trajectory(p,int(d)+1) for p,d in zip(direct,duration)],scale['lo'],scale['hi'])
        delta=((g-f)*np.where(y[:,None]==0,1,-1)).mean(0)
        result[method]=dict(contrast=dict(zip(FEATURES,delta.tolist())),
            observed_contrast=dict(zip(FEATURES,(known[y==1]-known[y==0]).mean(0).tolist())),
            spatial=paper_scores(occ,gen_occ,np.arange(len(y)),np.arange(len(y)),bins),
            motion={k:motion_scores(known,f,known_v,v,np.flatnonzero(y==i)) for i,k in enumerate(('bicycle','bus'))})
    Path(f'{kind}-{seed}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('seed',type=int)
    parser.add_argument('kind',choices=['class','speed'])
    opt=parser.parse_args()
    main(opt.seed,opt.kind)
