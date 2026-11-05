"""Speed-field gradient guidance. The spatial recognition network is never loaded here."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser
from g22_diffusion import inputs,K
from g23_fit import classifier
from g23_modes import speed_images
from g23_select import targets

OUT=Path("g24-data")
OLD=Path("../g22-job/g22-data")
SPEED=Path("../g23-job/g23-data")


def progress_and_clock(clean,length,label,stats):
    med=clean.new_tensor(stats["median_speed"])[label]
    T=(length/med)*torch.exp((clean[:,0]*stats["duration_sd"]+stats["duration_mean"]).clamp(-8,8))
    T=torch.minimum(torch.maximum(T,(length/55).clamp_min(1)),T.new_tensor(7200.))
    increments=torch.expm1((clean[:,1:]*stats["increment_sd"]+stats["increment_mean"]).clamp(0,5))
    increments=torch.where(increments.sum(1,keepdim=True)<1e-8,torch.ones_like(increments),increments)
    p=torch.nn.functional.pad(increments.cumsum(1),(1,0))
    return T,p/p[:,-1:].clamp_min(1e-8)


def soft_input(clean,length,label,stats):
    # Use planned arc metres. Spatial geometry and the other classifier are absent.
    T,p=progress_and_clock(clean,length,label,stats)
    index=(torch.floor(T).clamp_min(1)-1)[:,None]*torch.linspace(0,1,128,device=T.device)[None]
    left=index.floor()
    def arc(t):
        u=(256*t/T[:,None]).clamp(0,256)
        lo=u.long().clamp_max(255)
        x=p.gather(1,lo)
        return length[:,None]*(x+(u-lo)*(p.gather(1,lo+1)-x))
    a,b,c=[arc(left+k) for k in (0,1,2)]
    speed=(b-a)+(index-left)*(c-2*b+a)
    # Keep the derivative finite at the unit-circle endpoint.
    return speed_images(speed.clamp(0,35*(1-1e-5)))


def direction(clean,length,label,stats,target,reader):
    with torch.enable_grad():
        x=clean.detach().requires_grad_()
        logits=reader(soft_input(x,length,label,stats))
        score=logits.log_softmax(1).gather(1,target[:,None]).sum()
        grad=torch.autograd.grad(score,x)[0]
        norm=grad.square().mean(1,keepdim=True).sqrt()
        if not torch.isfinite(norm).all():raise ValueError("Nonfinite speed gradient")
        return grad/norm.clamp_min(1e-7)


def run(arm,seed):
    torch.set_num_threads(2)
    output=OUT/f"{arm}-{seed}.npz"
    if output.exists():return
    with np.load(OUT/"requests.npz") as d:req={k:d[k] for k in d.files}
    stats=json.loads((OLD/f"model-{seed}/stats.json").read_text())
    den=PathDenoiser(classes=8).cuda().eval()
    den.load_state_dict(torch.load(OLD/f"model-{seed}/model.pt",weights_only=True,map_location="cuda"))
    den.requires_grad_(False)
    reader=None
    if arm!="plain":
        reader=classifier().eval()
        reader.load_state_dict(torch.load(SPEED/"speed-mode.pt",weights_only=True,map_location="cuda"))
        reader.requires_grad_(False)
    with np.load(SPEED/"train.npz") as tr:
        desired,_=targets(tr,req["labels"],req["length"],req["ids"],seed)
    use=desired if arm!="permuted" else np.random.default_rng(24000+seed).permutation(desired)
    mode=torch.from_numpy(use).cuda()
    labs,starts,paths=inputs(req,stats)
    length=torch.tensor(req["length"],device="cuda",dtype=torch.float32)
    ab=torch.cumprod(1-torch.linspace(.0001,.02,1000,device="cuda"),0)
    times=torch.linspace(999,0,100).long().tolist()
    noise=[]
    for ident in req["ids"]:
        key=int.from_bytes(hashlib.sha256(f"G22:{ident}:{seed}".encode()).digest()[:4],"little")
        noise.append(torch.randn(K,1,257,device="cuda",generator=torch.Generator(device="cuda").manual_seed(key)))
    noise=torch.cat(noise)
    samples=[]
    begin=time.monotonic()
    with torch.no_grad():
        for ix in torch.arange(len(noise),device="cuda").split(32):
            idx=ix//K
            z=noise[ix]
            for j,t in enumerate(times):
                eps=den(z,torch.full_like(labs[idx],t),labs[idx],starts[idx],paths[idx])
                alpha=ab[t]
                clean=(z-(1-alpha).sqrt()*eps)/alpha.sqrt()
                if reader is not None:
                    grad=direction(clean[:,0],length[idx],labs[idx],stats,mode[idx],reader)
                    eps=eps-.05*4*alpha*(1-alpha)*grad[:,None]
                    clean=(z-(1-alpha).sqrt()*eps)/alpha.sqrt()
                prev=ab[times[j+1]] if j+1<len(times) else 1.
                z=prev**.5*clean+(1-prev)**.5*eps
            samples.append(z[:,0].cpu().numpy())
    latent=np.concatenate(samples).reshape(-1,K,257)
    if not np.isfinite(latent).all():raise ValueError("Nonfinite latent")
    L=req["length"][:,None]
    T=L/np.array(stats["median_speed"])[req["labels"],None]*np.exp(np.clip(latent[:,:,0]*stats["duration_sd"]+stats["duration_mean"],-8,8))
    lower=np.maximum(1.,L/55)
    cap=(T<lower)|(T>7200)
    T=np.clip(T,lower,7200)
    inc=np.expm1(np.clip(latent[:,:,1:]*stats["increment_sd"]+stats["increment_mean"],0,5))
    flat=inc.sum(-1)<1e-8
    inc[flat]=1
    p=np.pad(np.cumsum(inc.astype(float),axis=-1),((0,0),(0,0),(1,0)))
    p/=p[:,:,-1:]
    tmp=output.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,ids=req["ids"],duration=T,progress=p.astype("float32"),bounded=cap,flat=flat,
                        target=desired,guided_target=use,seconds=time.monotonic()-begin)
    tmp.replace(output)


if __name__=="__main__":run(sys.argv[1],int(sys.argv[2]))
