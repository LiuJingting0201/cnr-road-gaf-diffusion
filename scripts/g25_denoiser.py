"""Measure speed-gradient alignment with the denoiser's bus/car switch at fixed state."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser
from cnr.topology import LaneConsistency
from g22_diffusion import inputs
from g23_fit import classifier
from g24_sample import direction

BASE=Path("g25-data")
OLD=Path("../g24-job/g24-data")
MODELS=Path("../g22-job/g22-data")


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    q=np.load(OLD/"requests.npz")
    graph=LaneConsistency("data/maps/most.net.xml",radius=5.,max_candidates=64)
    selected=[]
    for i in np.flatnonzero(np.isin(q["labels"],[1,3])):
        valid=[]
        for name in ("bus","private_car"):
            r=graph.match(q["paths"][i],name)
            valid.append(r["graph_consistent"] and not r["candidate_truncation_warning"])
        if all(valid):selected.append(i)
    req={k:q[k][selected] for k in q.files}
    reader=classifier().eval()
    reader.load_state_dict(torch.load("../g23-job/g23-data/speed-mode.pt",weights_only=True))
    reader.requires_grad_(False)
    stats=[]
    for seed in (42,43):
        source=MODELS/f"model-{seed}"
        cfg=json.loads((source/"stats.json").read_text())
        den=PathDenoiser(classes=8).cuda().eval()
        den.load_state_dict(torch.load(source/"model.pt",weights_only=True))
        den.requires_grad_(False)
        label,start,path=inputs(req,cfg)
        other=torch.where(label==1,3,1)
        L=torch.as_tensor(req["length"],device="cuda",dtype=torch.float32)
        desired=torch.from_numpy(np.load(OLD/f"guided-{seed}.npz")["guided_target"][selected]).cuda()
        perm=torch.from_numpy(np.load(OLD/f"permuted-{seed}.npz")["guided_target"][selected]).cuda()
        ab=torch.cumprod(1-torch.linspace(.0001,.02,1000,device="cuda"),0)
        times=torch.linspace(999,0,100).long().tolist()
        noise=[]
        for ident in req["ids"]:
            key=int.from_bytes(hashlib.sha256(f"G22:{ident}:{seed}".encode()).digest()[:4],"little")
            noise.append(torch.randn(4,1,257,device="cuda",generator=torch.Generator(device="cuda").manual_seed(key)))
        noise=torch.cat(noise)
        rows=[]
        for ix in torch.arange(len(noise),device="cuda").split(32):
            idx=ix//4;z=noise[ix];record=0;mass=0
            for j,t in enumerate(times):
                step=torch.full_like(label[idx],t)
                eps=den(z,step,label[idx],start[idx],path[idx])
                clean=(z-(1-ab[t]).sqrt()*eps)/ab[t].sqrt()
                if j%10==5:
                    alt=den(z,step,other[idx],start[idx],path[idx])
                    shift=(alt-eps)[:,0]
                    weight=4*ab[t]*(1-ab[t])
                    terms=[]
                    for target in (desired,perm):
                        grad=direction(clean[:,0],L[idx],label[idx],cfg,target[idx],reader)
                        for region in (slice(None),slice(1,None)):
                            terms.append(torch.nn.functional.cosine_similarity(grad[:,region],shift[:,region],dim=1))
                    record=record+weight*torch.stack(terms,1)
                    mass=mass+weight
                prev=ab[times[j+1]] if j+1<len(times) else 1.
                z=prev**.5*clean+(1-prev)**.5*eps
            rows.append((record/mass).cpu().numpy())
        stats.append(np.concatenate(rows).reshape(-1,4,4).mean(1))
    arr=np.mean(stats,axis=0)
    lab=req["labels"];ep=np.unique([s.split("_")[0] for s in req["ids"]],return_inverse=True)[1]
    n=ep.max()+1;cnt=np.vstack([np.ones(n),np.random.default_rng(25197).multinomial(n,np.ones(n)/n,2000)])
    curves=[]
    for c in (1,3):
        use=lab==c
        sums=np.stack([arr[use&(ep==e)].sum(0) for e in range(n)])
        curve=(cnt@sums)/(cnt@np.bincount(ep[use],minlength=n))[:,None]
        curves.append(curve)
    avg=np.mean(curves,axis=0)
    out=dict(attempted=int(np.isin(q["labels"],[1,3]).sum()),eligible=len(lab),by_class=[int(sum(lab==c)) for c in (1,3)])
    for k,v in zip(["correct_full","correct_progress","permuted_full","permuted_progress","delta_full","delta_progress"],
                   [*avg.T,avg[:,0]-avg[:,2],avg[:,1]-avg[:,3]]):
        out[k]=dict(mean=float(v[0]),ci=np.quantile(v[1:],[.025,.975]).tolist())
    (BASE/"denoiser.json").write_text(json.dumps(out,indent=2))


if __name__=="__main__":main()
