"""Joint diffusion over the trip clock, block travel and within-block distance allocation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from cnr.path_diffusion import PathDenoiser
from g22_diffusion import K, STEPS, inputs, load, physical_stats
from g23_select import targets as requested_modes

OUT=Path("g26-data")
SPEED=Path("../g23-job/g23-data")
ARMS=("H","HM","HV","HG")


class Denoiser(PathDenoiser):
    def __init__(self):
        super().__init__(classes=8)
        self.mode=nn.Embedding(6,128)
        nn.init.zeros_(self.mode.weight)

    def forward(self,x,t,label,start,route,mode=None):
        cond=self.route_encoder(route)
        if mode is not None:
            cond=cond+self.mode(mode)
        return self.backbone(x,t,label,start,cond)


def pack(d):
    stats=physical_stats(d)
    middle=np.maximum(0,np.diff(d["progress"].astype(float),axis=1)).reshape(-1,16,16)
    coarse=middle.sum(-1)
    fine=np.divide(middle,coarse[:,:,None],out=np.zeros_like(middle),where=coarse[:,:,None]>1e-12)
    c=np.log1p(16*coarse)
    f=np.log1p(16*fine)
    stats.update(coarse_mean=float(c.mean()),coarse_sd=float(c.std()),fine_mean=float(f.mean()),fine_sd=float(f.std()))
    c=(c-stats["coarse_mean"])/stats["coarse_sd"]
    f=(f-stats["fine_mean"])/stats["fine_sd"]
    resid=np.log(d["duration"]*np.array(stats["median_speed"])[d["labels"]]/d["length"])
    clock=(resid-stats["duration_mean"])/stats["duration_sd"]
    z=np.c_[clock,np.concatenate([c[:,:,None],f],axis=2).reshape(-1,272)]
    return z.astype("float32"),stats


def normalize_fractions(v):
    total=v.sum(-1,keepdim=True)
    return torch.where(total>1e-8,v/total.clamp_min(1e-8),torch.full_like(v,1/v.shape[-1]))


def unpack(z,L,labels,stats):
    cap_low=(L/55).clamp_min(1)
    T=L/z.new_tensor(stats["median_speed"])[labels]*torch.exp((z[:,0]*stats["duration_sd"]+stats["duration_mean"]).clamp(-8,8))
    capped=(T<cap_low)|(T>7200)
    T=torch.minimum(torch.maximum(T,cap_low),T.new_tensor(7200))
    body=z[:,1:].reshape(-1,16,17)
    coarse=normalize_fractions(torch.expm1((body[:,:,0]*stats["coarse_sd"]+stats["coarse_mean"]).clamp(0,5)))
    fine=normalize_fractions(torch.expm1((body[:,:,1:]*stats["fine_sd"]+stats["fine_mean"]).clamp(0,5)))
    delta=(coarse[:,:,None]*fine).reshape(-1,256)
    progress=nn.functional.pad(delta.cumsum(1),(1,0))
    progress=progress/progress[:,-1:].clamp_min(1e-8)
    return T,progress,capped


def normalized_speed(T,P,L):
    # The same 128 relative positions as the complete-trip speed reader.
    grid=(T.floor().clamp_min(1)-1)[:,None]*torch.linspace(0,1,128,device=T.device)
    left=grid.floor()
    def arc(t):
        pos=(256*t/T[:,None]).clamp(0,256)
        lo=pos.long().clamp_max(255)
        x=P.gather(1,lo)
        return L[:,None]*(x+(pos-lo)*(P.gather(1,lo+1)-x))
    a,b,c=[arc(left+k) for k in (0,1,2)]
    return ((b-a+(grid-left)*(c-2*b+a))/35).clamp(0,1-1e-5)


def discrepancy(x,y,arm):
    if arm=="HV":
        return (x-y).square().mean(1)
    def field(u):
        v=(1-u*u).clamp_min(1e-6).sqrt()
        return (u[:,:,None]*u[:,None,:]-v[:,:,None]*v[:,None,:],
                v[:,:,None]*u[:,None,:]-u[:,:,None]*v[:,None,:])
    a,b=field(x);c,d=field(y)
    return .5*((a-c).square()+(b-d).square()).mean((1,2))


def fit(arm,seed,out):
    tr=load("train")
    raw,stats=pack(tr)
    with np.load(SPEED/"train.npz") as m:
        order={ident:i for i,ident in enumerate(m["ids"])}
        ix=[order[i] for i in tr["ids"]]
        assert np.array_equal(tr["labels"],m["participant"][ix])
        mode=torch.from_numpy(m["y"][ix]).cuda()
        actual=torch.from_numpy(m["v"][ix]).cuda().div(35).clamp(0,1-1e-5)
    torch.manual_seed(seed);net=Denoiser().cuda()
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    y,start,path=inputs(tr,stats)
    z=torch.from_numpy(raw).cuda()
    L=torch.tensor(tr["length"],device="cuda",dtype=torch.float32)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
    ab=torch.cumprod(1-torch.linspace(.0001,.02,1000,device="cuda"),0)
    rng=np.random.default_rng(seed+22000)
    classes=rng.integers(0,8,(STEPS,64))
    batches=np.empty_like(classes)
    for c in range(8):
        mask=classes==c
        batches[mask]=rng.choice(np.flatnonzero(tr["labels"]==c),mask.sum())
    scale=1.
    if arm in ("HV","HG"):
        cal=np.random.default_rng(26042).choice(len(y),1024)
        terms=[discrepancy(actual[block],actual[np.roll(cal,1)[j*64:(j+1)*64]],arm).mean().item()
               for j,block in enumerate(np.array_split(cal,16))]
        scale=max(float(np.mean(terms)),1e-5)
    begin=time.monotonic()
    with (out/"fit.jsonl").open("w",buffering=1) as stream:
        for step,ix in enumerate(torch.from_numpy(batches).cuda(),1):
            target=z[ix]
            t=torch.randint(1000,(len(ix),),device="cuda")
            noise=torch.randn_like(target)
            a=ab[t,None]
            noisy=a.sqrt()*target+(1-a).sqrt()*noise
            pred=net(noisy[:,None],t,y[ix],start[ix],path[ix],mode[ix] if arm!="H" else None)[:,0]
            err=(pred-noise).square()
            groups=err[:,1:].reshape(-1,16,17)
            loss=.5*err[:,0].mean()+.5*groups[:,:,0].mean()+groups[:,:,1:].mean()
            if arm in ("HV","HG"):
                estimate=(noisy-(1-a).sqrt()*pred)/a.sqrt()
                T,P,_=unpack(estimate,L[ix],y[ix],stats)
                v=normalized_speed(T,P,L[ix])
                aux=(a[:,0]*discrepancy(v,actual[ix],arm)).mean()/scale
                loss+=.1*aux
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite objective")
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
            with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.001)
            if step==1 or step%1000==0:
                stream.write(json.dumps(dict(step=step,loss=loss.item(),seconds=time.monotonic()-begin))+"\n")
    (out/"stats.json").write_text(json.dumps(dict(stats=stats,aux_normalizer=scale),indent=2))
    torch.save(ema.state_dict(),out/"model.tmp");(out/"model.tmp").replace(out/"model.pt")
    return ema,stats


@torch.no_grad()
def sample(model,arm,req,stats,seed,requested=None):
    y,start,path=inputs(req,stats)
    L=torch.tensor(req["length"],device="cuda",dtype=torch.float32)
    if requested is None:
        with np.load(SPEED/"train.npz") as tr:
            desired,_=requested_modes(tr,req["labels"],req["length"],req["ids"],seed)
    else:
        desired=np.asarray(requested,dtype=np.int64)
        if desired.shape!=(len(y),) or np.any((desired<0)|(desired>5)):
            raise ValueError("Expected one supported mode per request")
    mode=torch.from_numpy(desired).cuda()
    noise=[]
    for ident in req["ids"]:
        key=int.from_bytes(hashlib.sha256(f"G22:{ident}:{seed}".encode()).digest()[:4],"little")
        rng=torch.Generator(device="cuda").manual_seed(key)
        basic=torch.randn(K,257,device="cuda",generator=rng)
        coarse=torch.randn(K,16,1,device="cuda",generator=rng)
        noise.append(torch.cat([basic[:,:1],torch.cat([coarse,basic[:,1:].reshape(K,16,16)],2).reshape(K,-1)],1))
    noise=torch.cat(noise)
    ab=torch.cumprod(1-torch.linspace(.0001,.02,1000,device="cuda"),0)
    times=torch.linspace(999,0,100).long().tolist()
    clocks=[];profiles=[];caps=[]
    for ix in torch.arange(len(noise),device="cuda").split(128):
        idx=ix//K;z=noise[ix,None]
        for j,t in enumerate(times):
            pred=model(z,torch.full_like(y[idx],t),y[idx],start[idx],path[idx],mode[idx] if arm!="H" else None)
            clean=(z-(1-ab[t]).sqrt()*pred)/ab[t].sqrt()
            prev=ab[times[j+1]] if j+1<len(times) else 1
            z=prev**.5*clean+(1-prev)**.5*pred
        T,P,cap=unpack(z[:,0],L[idx],y[idx],stats)
        clocks.append(T.cpu().numpy());profiles.append(P.cpu().numpy());caps.append(cap.cpu().numpy())
    p=np.concatenate(profiles).reshape(-1,4,257)
    assert np.isfinite(p).all() and (np.diff(p,axis=-1)>=-1e-7).all() and np.allclose(p[:,:,-1],1)
    return dict(ids=req["ids"],duration=np.concatenate(clocks).reshape(-1,4),progress=p,bounded=np.concatenate(caps).reshape(-1,4),target=desired)


def main(arm,seed):
    assert arm in ARMS
    torch.set_num_threads(2)
    out=OUT/f"{arm}-{seed}"
    out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():return
    if (out/"model.pt").exists():
        model=Denoiser().cuda().eval();model.load_state_dict(torch.load(out/"model.pt",weights_only=True))
        stats=json.loads((out/"stats.json").read_text())["stats"]
    else:model,stats=fit(arm,seed,out)
    for _ in range(600):
        if (OUT/"ready.txt").exists():break
        time.sleep(2)
    with np.load(OUT/"requests.npz") as req:result=sample(model,arm,req,stats,seed)
    np.savez_compressed(out/"samples.npz",**result)
    (out/"complete.json").write_text(json.dumps(dict(arm=arm,seed=seed,steps=STEPS,requests=len(result["ids"]),caps=int(result["bounded"].sum()))))


if __name__=="__main__":main(sys.argv[1],int(sys.argv[2]))
