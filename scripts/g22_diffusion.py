"""Draw a complete trip: jointly diffuse duration and normalized within-trip progress."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser

ROOT=Path("g22-data")
STEPS=12000
K=4


def load(split):
    rows=[]
    for file in sorted((ROOT/split).glob("*/trips.npz")):
        with np.load(file) as d:
            rows.append({k:d[k] for k in ("ids","labels","paths","length","duration","progress")})
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def physical_stats(tr):
    median=[np.median(tr["length"][tr["labels"]==c]/tr["duration"][tr["labels"]==c]) for c in range(8)]
    residual=np.log(tr["duration"]*np.array(median)[tr["labels"]]/tr["length"])
    motion=np.log1p(256*np.maximum(0,np.diff(tr["progress"],axis=1)))
    return dict(median_speed=list(map(float,median)),duration_mean=float(residual.mean()),duration_sd=float(residual.std()),
                increment_mean=float(motion.mean()),increment_sd=float(motion.std()),
                start_mean=tr["paths"][:,0].mean(0).tolist(),start_std=np.maximum(tr["paths"][:,0].std(0),1).tolist())


def targets(tr,stats):
    center=np.array(stats["median_speed"])[tr["labels"]]
    time_z=(np.log(tr["duration"]*center/tr["length"])-stats["duration_mean"])/stats["duration_sd"]
    arc_z=(np.log1p(256*np.maximum(0,np.diff(tr["progress"],axis=1)))-stats["increment_mean"])/stats["increment_sd"]
    return np.c_[time_z,arc_z].astype("float32")


def inputs(rows,stats):
    path=rows["paths"][:,::16]
    start=path[:,0]
    return (torch.as_tensor(rows["labels"],device="cuda"),
            torch.as_tensor((start-np.array(stats["start_mean"]))/stats["start_std"],device="cuda",dtype=torch.float32),
            torch.as_tensor((path-start[:,None])/1500,device="cuda").transpose(1,2))


def fit(seed,out,tr,stats):
    torch.manual_seed(seed)
    model=PathDenoiser(classes=8).cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    target=torch.from_numpy(targets(tr,stats)).cuda()[:,None]
    label,start,path=inputs(tr,stats)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    ab=torch.cumprod(1-torch.linspace(.0001,.02,1000,device="cuda"),0)
    rng=np.random.default_rng(seed+22000)
    classes=rng.integers(0,8,(STEPS,64))
    indices=np.empty_like(classes)
    for c in range(8):
        mask=classes==c
        indices[mask]=rng.choice(np.flatnonzero(tr["labels"]==c),mask.sum())
    beg=time.monotonic()
    with (out/"training.jsonl").open("w",buffering=1) as s:
        for step,ix in enumerate(torch.from_numpy(indices).cuda(),1):
            x=target[ix]
            t=torch.randint(1000,(len(ix),),device="cuda")
            noise=torch.randn_like(x)
            a=ab[t,None,None]
            error=(model(a.sqrt()*x+(1-a).sqrt()*noise,t,label[ix],start[ix],path[ix])-noise).square()
            loss=error[:,:,1:].mean()+.5*error[:,:,0].mean()
            if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
            if step==1 or step%1000==0:
                s.write(json.dumps(dict(step=step,loss=loss.item(),elapsed=time.monotonic()-beg))+"\n")
    torch.save(ema.state_dict(),out/"model.tmp")
    (out/"model.tmp").replace(out/"model.pt")
    return ema


@torch.no_grad()
def sample(model,requests,stats,seed):
    # Only route, start, class and known route length are model inputs.
    label,start,path=inputs(requests,stats)
    ab=torch.cumprod(1-torch.linspace(.0001,.02,1000,device="cuda"),0)
    times=torch.linspace(999,0,100).long().tolist()
    noise=[]
    for ident in requests["ids"]:
        key=int.from_bytes(hashlib.sha256(f"G22:{ident}:{seed}".encode()).digest()[:4],"little")
        noise.append(torch.randn(K,1,257,device="cuda",generator=torch.Generator(device="cuda").manual_seed(key)))
    noise=torch.cat(noise)
    latent=[]
    for ix in torch.arange(len(noise),device="cuda").split(128):
        idx=ix//K
        z=noise[ix]
        for j,t in enumerate(times):
            pred=model(z,torch.full_like(label[idx],t),label[idx],start[idx],path[idx])
            clean=(z-(1-ab[t]).sqrt()*pred)/ab[t].sqrt()
            ap=ab[times[j+1]] if j+1<len(times) else 1
            z=ap**.5*clean+(1-ap)**.5*pred
        latent.append(z[:,0].cpu().numpy())
    latent=np.concatenate(latent).reshape(-1,K,257)
    if not np.isfinite(latent).all():raise ValueError("Nonfinite sample")
    logtime=latent[:,:,0]*stats["duration_sd"]+stats["duration_mean"]
    T=requests["length"][:,None]/np.array(stats["median_speed"])[requests["labels"],None]*np.exp(np.clip(logtime,-8,8))
    lower=np.maximum(1.,requests["length"]/55)[:,None]
    bounded=(T<lower)|(T>7200)
    T=np.clip(T,lower,7200)
    increments=np.expm1(np.clip(latent[:,:,1:]*stats["increment_sd"]+stats["increment_mean"],0,5))
    flat=increments.sum(-1)<1e-8
    increments[flat]=1
    p=np.pad(np.cumsum(increments.astype(float),axis=-1),((0,0),(0,0),(1,0)))
    p/=p[:,:,-1:]
    return dict(ids=requests["ids"],duration=T,progress=p.astype("float32"),bounded=bounded,flat=flat)


def positions(path,progress):
    return np.stack([np.interp(progress,np.linspace(0,1,len(path)),v) for v in path.T],axis=-1)


def decode(path,duration,profile,dt=.25):
    times=np.r_[np.arange(0,duration,dt),duration]
    arc=np.interp(times/duration,np.linspace(0,1,len(profile)),profile)
    return times,positions(path,arc)


def main(seed):
    torch.set_num_threads(2)
    out=ROOT/f"model-{seed}"
    out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():return
    tr=load("train");stats=physical_stats(tr)
    (out/"stats.json").write_text(json.dumps(stats,indent=2))
    if (out/"model.pt").exists():
        model=PathDenoiser(classes=8).cuda().eval()
        model.load_state_dict(torch.load(out/"model.pt",weights_only=True,map_location="cuda"))
    else:model=fit(seed,out,tr,stats)
    ev=load("evaluation")
    req={k:ev[k] for k in ("ids","labels","paths","length")}
    gen=sample(model,req,stats,seed)
    np.savez_compressed(out/"samples.npz",**gen)
    (out/"complete.json").write_text(json.dumps(dict(seed=seed,steps=STEPS,requests=len(ev["ids"]),samples=len(ev["ids"])*K,
                                                     bound_count=int(gen["bounded"].sum()),flat=int(gen["flat"].sum()))))


if __name__=="__main__":
    main(int(sys.argv[1]))
