"""Speed-only angular images and train-defined complete-trip motion-mode labels."""
from pathlib import Path
import json
import numpy as np
import torch
from g22_diffusion import ROOT as TRIPS

OUT=Path("g23-data")


def speed_series(t,xy):
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy.astype(float),axis=0),axis=1))]
    ts=t-t[0]
    sec=np.arange(int(np.floor(ts[-1]))+1)
    v=np.diff(np.interp(sec,ts,d))
    if len(v)<5:raise ValueError("Too short for four-second variation")
    profile=np.interp(np.linspace(0,len(v)-1,128),np.arange(len(v)),v)
    return profile.astype(np.float32),np.array([v.mean(),abs(v[4:]-v[:-4]).mean()])


def mode(stats,threshold):
    band=np.searchsorted(threshold["speed_tertiles"],stats[...,0])
    return 2*band+(stats[...,1]>np.array(threshold["variation_medians"])[band])


def speed_images(v):
    # Absolute speed scale, shared by all participants. No XY or participant input.
    u=(v/35.).clamp(0,1);b=(1-u.square()).clamp_min(0).sqrt()
    bins=(u*8).long().clamp_max(7)
    edge=bins[:,:-1]*8+bins[:,1:]
    counts=v.new_zeros(len(v),64)
    counts.scatter_add_(1,edge,torch.ones_like(edge,dtype=v.dtype))
    counts=counts.reshape(-1,8,8)
    trans=counts/counts.sum(2,keepdim=True).clamp_min(1)
    mtf=trans[torch.arange(len(v),device=v.device)[:,None,None],bins[:,:,None],bins[:,None,:]]
    return torch.stack([u[:,:,None]*u[:,None,:]-b[:,:,None]*b[:,None,:],
                        b[:,:,None]*u[:,None,:]-u[:,:,None]*b[:,None,:],mtf],1)


def read(split):
    rows=[]
    for f in sorted((TRIPS/split).glob("*/quarter.npz")):
        with np.load(f) as q,np.load(f.parent/"trips.npz") as z:
            assert np.array_equal(q["ids"],z["ids"])
            off=q["offset"];tx=q["txy"];paths=z["paths"]
            for i,ident in enumerate(q["ids"]):
                arr=tx[off[i]:off[i+1]]
                # FCD observations at 0.25 s; duration and endpoints are not supplied to the CNN.
                profile,stat=speed_series(arr[:,0],arr[:,1:])
                rows.append((ident,z["labels"][i],z["length"][i],profile,stat))
    ids,labs,length,profiles,stats=map(np.array,zip(*rows))
    return dict(ids=ids,participant=labs,length=length,v=profiles,stats=stats)


def prepare():
    OUT.mkdir(exist_ok=True)
    tr=read("train");m=read("monitor")
    q=np.quantile(tr["stats"][:,0],[1/3,2/3])
    band=np.searchsorted(q,tr["stats"][:,0])
    threshold=dict(speed_tertiles=q.tolist(),variation_medians=[float(np.median(tr["stats"][band==c,1])) for c in range(3)])
    for name,d in [("train",tr),("monitor",m)]:
        y=mode(d["stats"],threshold).astype(np.int64)
        np.savez(OUT/(name+".npz"),**d,y=y)
    y=mode(tr["stats"],threshold)
    threshold["training_mode_by_participant"]=np.bincount(6*tr["participant"]+y,minlength=48).reshape(8,6).tolist()
    (OUT/"definition.json").write_text(json.dumps(threshold,indent=2))


if __name__=="__main__":prepare()
