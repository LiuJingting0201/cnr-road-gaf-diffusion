"""Complete-trip image inputs; the phase grid preserves whole-trip duration."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import torch
from g19_gaf import images
from g22_diffusion import ROOT

def dynamics(t,xy):
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy.astype(float),axis=0),axis=1))]
    center=np.linspace(0,t[-1],127)
    def speed(at):
        lo=np.clip(at-.5,0,t[-1]);hi=np.clip(at+.5,0,t[-1])
        return (np.interp(hi,t,arc)-np.interp(lo,t,arc))/np.maximum(hi-lo,1e-6)
    lo=np.maximum(center-.5,0);hi=np.minimum(center+.5,t[-1])
    return np.stack([speed(center),(speed(hi)-speed(lo))/np.maximum(hi-lo,1e-6)]).astype("float32")


def fields(va):
    v=(va[:,0]/32).clamp(0,1)
    a=torch.tanh(va[:,1]/2)
    p=(1-v.square()).clamp_min(0).sqrt()
    b=(1-a.square()).clamp_min(0).sqrt()
    return torch.stack([v[:,:,None]*v[:,None,:]-p[:,:,None]*p[:,None,:],
                        a[:,:,None]*a[:,None,:]-b[:,:,None]*b[:,None,:],
                        b[:,:,None]*a[:,None,:]-a[:,:,None]*b[:,None,:]],1)


def episodes(file):
    path=Path(file)
    out=path.parent/"image-features.npz"
    if out.exists(): return
    with np.load(file) as z:
        xy=z["xy"];offset=z["offset"];T=z["duration"];paths=z["paths"];ids=z["ids"];labels=z["labels"]
    whole=[];short=[];va=[];has=[]
    for j,dur in enumerate(T):
        raw=xy[offset[j]:offset[j+1]]
        # FCD at one-second cadence; endpoints are specified by the request.
        x=np.vstack([paths[j,0],raw,paths[j,-1]])
        t=np.linspace(0,dur,len(x))
        whole.append(x)
        va.append(dynamics(t,x))
        has.append(dur>=127.75)
        times=(dur-127.75)/2+np.arange(128)
        short.append(np.stack([np.interp(times,t,coord) for coord in x.T],axis=-1))
    rgb=[];ok=[];sub=[];sok=[]
    for i in range(0,len(whole),16):
        im,v=images(whole[i:i+16]);rgb.extend(im);ok.extend(v)
        im,v=images(short[i:i+16]);sub.extend(im);sok.extend(v)
    temp=out.with_name("image.tmp.npz")
    np.savez_compressed(temp,ids=ids,labels=labels,full=rgb,short=sub,va=va,ok=ok,sok=sok,eligible=has)
    temp.replace(out)


if __name__=="__main__":
    paths=[p for split in ("train","monitor","evaluation") for p in sorted((ROOT/split).glob("*/trips.npz"))]
    with ProcessPoolExecutor(max_workers=10) as pool:
        list(pool.map(episodes,paths))
    (ROOT/"images-ready.txt").write_text(f"{len(paths)} episodes\n")
