"""Generate untimed route descriptors from frozen coordinate DiffTraj samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

BINS=16
DIM=BINS+1


def describe(routes):
    """Log polyline length and unsigned turns on 16 consecutive equal-distance bins."""
    result=np.empty((len(routes),DIM),dtype='f4')
    for i,xy in enumerate(routes):
        seg=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        arc=np.r_[0.,seg.cumsum()]
        if not np.isfinite(arc[-1]) or arc[-1]<=0:
            raise ValueError('Invalid generated route')
        s=np.linspace(0,arc[-1],BINS*4+2)
        points=np.stack([np.interp(s,arc,axis) for axis in xy.T],axis=1)
        delta=np.diff(points,axis=0)
        a,b=delta[:-1],delta[1:]
        turn=np.arctan2(abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]),(a*b).sum(1))
        result[i,0]=np.log(arc[-1])
        result[i,1:]=turn.reshape(BINS,4).sum(1)
    return result


def generate(a):
    import torch
    from five_class_train import make_model, CLASSES
    from five_class_evaluate import sample
    net=make_model('C').eval()
    ck=torch.load(a.base/f'coordinate-{a.seed}/final.pt',weights_only=True,map_location='cpu')
    assert ck['seed']==a.seed and ck['epoch']==10
    net.load_state_dict(ck['weights'])
    norm=dict(np.load(a.data/'normalization.npz'))
    for split in ('train','validation'):
        h=np.load(a.data/split/'head.npy',mmap_mode='r')
        with np.load(a.data/split/'index.npz') as z:
            classes=np.searchsorted(CLASSES,z['classes'])
        path=a.out/split
        path.mkdir(parents=True,exist_ok=True)
        for j,start in enumerate(range(0,len(h),512)):
            if j%2!=a.part: continue
            dest=path/f'{start:07}.npy'
            if dest.exists(): continue
            group=slice(start,start+512)
            noise=np.random.default_rng([17119,0 if split=='train' else 1,start]).standard_normal((len(h[group]),2,200),dtype='f4')
            xy=sample(net,h[group].copy(),classes[group],noise,norm,256)
            f=describe(xy)
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as stream: np.save(stream,f)
            os.replace(tmp,dest)
            if j%20 in (0,1): print(split,start,len(h),flush=True)


def combine(a):
    for split,n in [('train',200000),('validation',5000)]:
        chunks=[np.load(a.out/split/f'{i:07}.npy') for i in range(0,n,512)]
        value=np.concatenate(chunks)
        assert value.shape==(n,DIM) and np.isfinite(value).all()
        if split=='train':
            mean,sd=value.mean(0,dtype='d'),value.std(0,dtype='d')
            assert np.all(sd>0)
            np.savez(a.out/'scale.npz',mean=mean,sd=sd)
        np.save(a.out/(split+'.npy'),((value-mean)/sd).astype('f4'))
    receipt=dict(seed=a.seed,descriptor='log_length_16_unsigned_turns',training_rows=200000,source_manifest=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),training_feature_sha256=hashlib.sha256((a.out/'train.npy').read_bytes()).hexdigest())
    (a.out/'manifest.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('generate','combine'))
    for key in ('base','data','out'): p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--part',type=int,choices=(0,1))
    a=p.parse_args()
    (generate if a.action=='generate' else combine)(a)
