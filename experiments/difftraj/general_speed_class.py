"""Common five-mode test of supplied vehicle labels and sampling-time speed guidance."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
import torch
from five_class_evaluate import (MODES,stratify,features,motion_scores,occupancy,paper_scores,
    read_observations,string_digest,array_digest)
from probe_sampling_guidance import draw,make_model,DATA,OLD
from utils.utils import resample_trajectory


def request_features(h):
    # Five training-standardised continuous descriptors, circular departure time,
    # and explicit coordinates for the origin and destination cells.
    angle=h[:,0]*2*np.pi/288
    cells=h[:,6:8].astype(int)
    return np.column_stack((h[:,1:6], np.sin(angle),np.cos(angle),
                            (cells//16)/4,(cells%16)/4))


def run(seed,method):
    norm=dict(np.load(DATA/'normalization.npz'))
    tr=np.load(DATA/'train/head.npy')
    classes=np.load(DATA/'train/index.npz')['classes']
    val=DATA/'validation'
    index=np.load(val/'index.npz')
    rows,_=stratify(index['classes'])
    ids=index['ids'][rows]; labels=index['classes'][rows]
    actual=np.searchsorted(MODES,labels)
    h=np.load(val/'head.npy')[rows]
    _,near=cKDTree(request_features(tr)).query(request_features(h),k=32)
    rng=np.random.default_rng(11161)
    # One fixed Monte Carlo draw from an empirical class distribution conditional on the header.
    inferred=np.searchsorted(MODES,classes[near[np.arange(len(rows)),rng.integers(32,size=len(rows))]])
    noise=np.random.default_rng(42).standard_normal((len(rows),2,200),dtype='f')
    path=OLD/f'runs/all-C-{seed}'
    net=make_model('C').eval()
    net.load_state_dict(torch.load(path/'final.pt',weights_only=True,map_location='cpu')['weights'])
    ref=np.load(val/'traj.npy',mmap_mode='r')[rows]*norm['std']+norm['mean']
    raw=read_observations(DATA,ids)
    duration=np.asarray([len(x)-1 for x in raw])
    f0,v0=features(ref,duration)
    bins=np.quantile(tr[:,1].astype('d')*norm['header_std'][0]+norm['header_mean'][0],np.arange(1,20)/20)
    def occ(x):return occupancy(x,norm['lo'],norm['hi'])
    o=occ(raw)
    archive=json.loads((path/'eval/metrics.json').read_text())
    assert archive['noise_sha256']==array_digest(noise) and archive['request_sha256']==string_digest(ids)
    counts=np.zeros((5,5),int); np.add.at(counts,(actual,inferred),1)
    outcome=dict(seed=seed,method=method,classes=list(MODES),class_confusion=counts.tolist(),
                 noise_sha256=array_digest(noise),ids_sha256=string_digest(ids),results={})
    for tag,requested in [('provided',actual),('header_only',inferred)]:
        if method=='base' and tag=='provided':
            with np.load(path/'eval/draws.npz') as d:
                assert np.array_equal(d['ids'],ids)
                xy=d['xy_m'].astype('d')
        else:
            xy=draw(net,method,h,requested,noise,norm).astype('f').astype('d')
        gg=occ([resample_trajectory(x,int(n)+1) for x,n in zip(xy,duration)])
        f1,v1=features(xy,duration)
        scores={}
        for k,mode in enumerate(MODES):
            ix=np.flatnonzero(actual==k)
            score=motion_scores(f0,f1,v0,v1,ix)
            score['accel_W1_mps2']=wasserstein_distance((np.diff(v0[ix])/ (duration[ix,None]/199)).ravel(),
                                                       (np.diff(v1[ix])/ (duration[ix,None]/199)).ravel())
            score['paper']=paper_scores(o,gg,ix,ix,bins)
            scores[mode]=score
        outcome['results'][tag]=scores
    dest=Path(f'{method}-{seed}.json')
    dest.write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('seed',type=int,choices=(42,43))
    p.add_argument('method',choices=('speed','base'))
    a=p.parse_args();run(a.seed,a.method)
