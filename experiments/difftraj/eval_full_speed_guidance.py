"""Reuse the locked speed sampler on all validation bicycles and buses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from five_class_evaluate import (stratify, match_bus_bicycle, read_observations,
                                 features, occupancy, motion_scores, paper_scores)
from probe_sampling_guidance import DATA, OLD, draw, make_model
from utils.utils import resample_trajectory


def evaluate(seed, method):
    valid = DATA/'validation'
    with np.load(valid/'index.npz') as z:
        ids, labels = z['ids'], z['classes']
    with np.load(DATA/'normalization.npz') as z:
        scale = dict(z)
    head = np.load(valid/'head.npy')
    pool, _ = stratify(labels)
    b, t = match_bus_bicycle(head[pool], labels[pool])
    pilot = pool[np.r_[b,t]]
    previous = pool[np.isin(labels[pool], ['bicycle','bus'])]
    missing = np.setdiff1d(np.flatnonzero(np.isin(labels,['bicycle','bus'])), previous)
    rows = np.r_[previous,missing]
    rng = np.random.default_rng(42)
    old_noise = rng.standard_normal((len(pool),2,200),dtype='f')
    noise = np.concatenate((old_noise[np.isin(labels[pool],['bicycle','bus'])],
                            rng.standard_normal((len(missing),2,200),dtype='f')))
    y = (labels[rows]=='bus').astype('i8')
    assert len(np.unique(ids[rows])) == 2000 and y.sum()==1000 and len(pilot)==226
    ck = torch.load(OLD/f'runs/all-C-{seed}/final.pt',map_location='cpu',weights_only=True)
    net = make_model('C').eval()
    net.load_state_dict(ck['weights'])
    xy = draw(net,method,head[rows],y,noise,scale)
    raw = read_observations(DATA,ids[rows])
    dur = np.asarray([len(p)-1 for p in raw])
    obs = np.load(valid/'traj.npy',mmap_mode='r')[rows]*scale['std']+scale['mean']
    ref,rv = features(obs,dur)
    gen,gv = features(xy,dur)
    ro = occupancy(raw,scale['lo'],scale['hi'])
    go = occupancy([resample_trajectory(p,int(d)+1) for p,d in zip(xy,dur)],scale['lo'],scale['hi'])
    h = np.load(DATA/'train/head.npy',mmap_mode='r')[:,1].astype('d')
    bins = np.unique(np.quantile(h*scale['header_std'][0]+scale['header_mean'][0],np.arange(1,20)/20))
    result = dict(seed=seed,method=method,noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                  id_sha256=hashlib.sha256('\n'.join(ids[rows]).encode()).hexdigest(),scores={})
    for group, keep in [('full',np.ones(len(y),bool)),('outside_pilot',~np.isin(rows,pilot))]:
        result['scores'][group]={}
        for name, mask in [('bus',y==1),('bicycle',y==0),('combined',np.ones(len(y),bool))]:
            idx=np.flatnonzero(keep & mask)
            result['scores'][group][name]=dict(motion=motion_scores(ref,gen,rv,gv,idx),
                                                 paper=paper_scores(ro,go,idx,idx,bins))
    np.savez_compressed(f'draws-{method}-{seed}.npz',ids=ids[rows],xy=xy.astype('f'))
    Path(f'{method}-{seed}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('seed',type=int,choices=(42,43))
    p.add_argument('method',choices=('base','speed'))
    a=p.parse_args()
    evaluate(a.seed,a.method)
