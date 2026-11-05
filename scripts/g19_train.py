"""Matched warm-start eight-class diffusion fine-tunes and fixed-pool sampling."""
import argparse
import copy
import hashlib
import json
import time
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser
from cnr.route_paths import decode_progress_numpy
from g16_train import regularize_start,tensor
from g16_data import GRID,ORIGINAL
from g19_field import ROOT,DATA,MapField
from pathlib import Path

OLD=Path("../g17-job/g17-data")
STEPS=10000


def fit(arm,seed,out):
    torch.manual_seed(seed)
    model=PathDenoiser(classes=8).cuda()
    model.load_state_dict(torch.load(OLD/f"C-{seed}/ema.pt",map_location="cuda",weights_only=True))
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    with np.load(DATA/"expanded.npz") as d:
        labels=d["label"]
        target=tensor(d["target"] if arm=="plain" else np.load(ROOT/"road-target.npy"))[:,None]-1
        prefix=tensor(d["route"]).transpose(1,2)/64
        start=regularize_start(tensor(d["start"]),json.loads((ORIGINAL/"dataset.json").read_text()))
    rng=np.random.default_rng(seed+19000)
    chosen=rng.integers(0,8,size=(STEPS,64))
    queue=np.empty_like(chosen)
    for c in range(8):
        mask=chosen==c
        queue[mask]=rng.choice(np.flatnonzero(labels==c),mask.sum())
    queue=torch.from_numpy(queue).cuda()
    labs=torch.from_numpy(labels).cuda()
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device="cuda"),0)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    begin=time.monotonic()
    with (out/"fit.jsonl").open("w",buffering=1) as stream:
        for i,ix in enumerate(queue,1):
            x=target[ix]
            t=torch.randint(1000,(len(ix),),device="cuda")
            noise=torch.randn_like(x)
            a=alpha[t,None,None]
            loss=(model(a.sqrt()*x+(1-a).sqrt()*noise,t,labs[ix],start[ix],prefix[ix])-noise).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
            if i==1 or i%500==0:
                stream.write(json.dumps(dict(step=i,loss=loss.item(),seconds=time.monotonic()-begin))+"\n")
    torch.save(ema.state_dict(),out/"ema.tmp")
    (out/"ema.tmp").replace(out/"ema.pt")
    return ema


@torch.no_grad()
def generate(model,q,field,meta):
    paths,length,starts=[tensor(q[k]) for k in ("route","length","start")]
    labs=torch.from_numpy(q["labels"]).cuda()
    u=tensor(GRID)[None,:]*(paths.shape[1]-1)/length[:,None]
    lo=u.long().clamp(0,paths.shape[1]-2)
    left=paths.gather(1,lo[:,:,None].expand(-1,-1,2))
    right=paths.gather(1,(lo+1)[:,:,None].expand(-1,-1,2))
    prefix=((left+(right-left)*(u-lo)[:,:,None])-starts[:,None]).transpose(1,2)/64
    alpha=torch.cumprod(1-torch.linspace(1e-4,.02,1000,device="cuda"),0)
    times=torch.linspace(999,0,100).long().tolist()
    noise=[]
    for ident in q["ids"]:
        key=int.from_bytes(hashlib.sha256(str(ident).encode()).digest()[:4],"little")
        noise.append(torch.randn(4,1,511,device="cuda",generator=torch.Generator(device="cuda").manual_seed(key)))
    all_noise=torch.cat(noise)
    motion=[]
    for ix in torch.arange(len(labs)*4,device="cuda").split(64):
        ixq=ix//4
        z=all_noise[ix]
        for j,t in enumerate(times):
            eps=model(z,torch.full_like(labs[ixq],t),labs[ixq],regularize_start(starts[ixq],meta),prefix[ixq])
            clean=(z-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            prev=alpha[times[j+1]] if j+1<len(times) else 1.
            z=prev**.5*clean+(1-prev)**.5*eps
        motion.append(torch.expm1((z[:,0]+1).clamp(0,5)).cpu().numpy())
    inc=np.concatenate(motion).reshape(-1,4,511)
    xy,caps=[],[]
    for i,(path,L,start,c) in enumerate(zip(q["route"],q["length"],q["start"],q["labels"])):
        pos=np.pad(np.cumsum(inc[i],axis=-1),((0,0),(1,0)))
        if field is not None:
            grid,trans=field.coordinate(path,L,c)
            caps.append(pos[:,-1]>=trans[-1])
            pos=np.interp(pos,trans,grid)
        else:
            caps.append(pos[:,-1]>=L)
        xy.append(np.stack([decode_progress_numpy(path,L,p,start) for p in pos]))
    xy=np.array(xy,dtype=np.float32)
    assert np.isfinite(xy).all()
    return dict(ids=q["ids"],xy=xy,caps=np.asarray(caps))


def main(arm,seed):
    torch.set_num_threads(2)
    out=ROOT/f"{arm}-{seed}"
    out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():
        return
    if (out/"ema.pt").exists():
        model=PathDenoiser(classes=8).cuda().eval()
        model.load_state_dict(torch.load(out/"ema.pt",weights_only=True,map_location="cuda"))
    else:
        model=fit(arm,seed,out)
    field=MapField(np.load(ROOT/"factor.npy")) if arm=="road" else None
    meta=json.loads((ORIGINAL/"dataset.json").read_text())
    for domain,filename in [("g14","g14-requests.npz"),("g15","requests.npz")]:
        with np.load(OLD/filename) as q:
            result=generate(model,q,field,meta)
            np.savez_compressed(out/f"{domain}.npz",**result)
    (out/"complete.json").write_text(json.dumps(dict(arm=arm,seed=seed,steps=STEPS)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=["plain","road"])
    p.add_argument("seed",type=int)
    a=p.parse_args()
    main(a.arm,a.seed)
