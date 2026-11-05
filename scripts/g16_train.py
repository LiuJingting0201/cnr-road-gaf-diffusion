"""Matched optimization on original, original + SUMO, and eight-class training sets."""
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser, decode
from g16_data import BASE, GRID, ORIGINAL

STEPS = int(os.getenv("CNR_G16_STEPS", 3000))
K = 4


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32, device="cuda")


def batches(arm, seed):
    sources = ["original"] if arm == "A" else ["original", "expanded"]
    features = ["target", "route", "start", "label"]
    columns = {key:[] for key in features}
    pools = {}
    offset = 0
    for source in sources:
        with np.load(BASE / (source + ".npz")) as arr:
            labels = arr["label"]
            for c in np.unique(labels):
                members = np.flatnonzero(labels == c)
                if arm == "D" and source == "expanded":
                    members = np.random.default_rng(162000 + int(c)).choice(members, 85, replace=False)
                pools[source, c] = offset + members
            for key in features:
                columns[key].append(arr[key])
            offset += len(labels)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0 if arm == "C" else 1, 8 if arm == "C" else 6, (STEPS,64))
    source_mask = rng.random(labels.shape) < .5
    indices = np.empty(labels.shape, dtype=int)
    for c in np.unique(labels):
        expanded = (source_mask if c in range(1,6) else np.ones_like(source_mask)) & (arm != "A")
        for source, where in [("original", ~expanded), ("expanded", expanded)]:
            use = where & (labels == c)
            if use.any():
                indices[use] = rng.choice(pools[source,c], size=use.sum())
    dev = {key:torch.from_numpy(np.concatenate(value)).cuda() for key,value in columns.items()}
    return dev, torch.from_numpy(indices).cuda()


def regularize_start(x, meta):
    return (x - tensor(meta["mean"])) / tensor(meta["std"])


def fit(arm, seed, out):
    torch.manual_seed(seed)
    model = PathDenoiser(classes=8 if arm=="C" else 6).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    dev, indices = batches(arm, seed)
    meta = json.loads((ORIGINAL / "dataset.json").read_text())
    start = regularize_start(dev["start"], meta)
    route = dev["route"].transpose(1,2) / 64
    alpha = torch.cumprod(1 - torch.linspace(1e-4,.02,1000,device="cuda"),0)
    begin = time.monotonic()
    with (out / "fit.jsonl").open("w", buffering=1) as stream:
        for step, ix in enumerate(indices, 1):
            target = dev["target"][ix,None] - 1
            t = torch.randint(1000,(len(ix),),device="cuda")
            noise = torch.randn_like(target)
            a = alpha[t,None,None]
            eps = model(a.sqrt()*target + (1-a).sqrt()*noise, t, dev["label"][ix], start[ix], route[ix])
            loss = (eps-noise).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.001)
            if step == 1 or step % 500 == 0:
                stream.write(json.dumps(dict(step=step,loss=loss.item(),seconds=time.monotonic()-begin))+"\n")
    torch.save(ema.state_dict(),out/"ema.pt")
    return ema


@torch.no_grad()
def sample(model, req, filename, meta=None):
    if meta is None:
        meta = json.loads((ORIGINAL / "dataset.json").read_text())
    route, length, start = [tensor(req[k]) for k in ("route","length","start")]
    label = torch.as_tensor(req["labels"],device="cuda")
    # Spatial sampling uses the actual planned path length, independent of the reference motion.
    pos = tensor(GRID)[None,:] * (route.shape[1]-1) / length[:,None]
    lo = pos.long().clamp(0,route.shape[1]-2)
    left = route.gather(1,lo[:,:,None].expand(-1,-1,2))
    right = route.gather(1,(lo+1)[:,:,None].expand(-1,-1,2))
    prefix = ((left+(right-left)*(pos-lo)[:,:,None])-start[:,None]).transpose(1,2) / 64
    alpha = torch.cumprod(1 - torch.linspace(1e-4,.02,1000,device="cuda"),0)
    times = torch.linspace(999,0,100).long().tolist()
    import hashlib
    # Seed by request ID so a shared request receives identical noise in all arms.
    noise = []
    for ident in req["ids"]:
        seed = int.from_bytes(hashlib.sha256(str(ident).encode()).digest()[:4],"little")
        rng = torch.Generator(device="cuda").manual_seed(seed)
        noise.append(torch.randn(K,1,511,device="cuda",generator=rng))
    noise = torch.cat(noise)
    all_xy, caps = [], []
    indices = torch.arange(len(label),device="cuda").repeat_interleave(K)
    for i, ix in enumerate(indices.split(64)):
        offset = 64*i
        z = noise[offset:offset+len(ix)]
        for j,t in enumerate(times):
            eps = model(z,torch.full_like(label[ix],t),label[ix],regularize_start(start[ix],meta),prefix[ix])
            clean = (z-(1-alpha[t]).sqrt()*eps)/alpha[t].sqrt()
            ap = alpha[times[j+1]] if j+1<len(times) else 1
            z = ap**.5 * clean + (1-ap)**.5 * eps
        xy, progress = decode(z,1.,1.,route[ix],length[ix],start[ix])
        if not torch.isfinite(xy).all():
            raise RuntimeError("Nonfinite sample")
        all_xy.append(xy.cpu().numpy())
        caps.append((progress[:,-1]>=length[ix]).cpu().numpy())
    np.savez_compressed(filename,ids=req["ids"],xy=np.concatenate(all_xy).reshape(-1,K,512,2),
                        caps=np.concatenate(caps).reshape(-1,K))


def main(arm, seed):
    torch.set_num_threads(2)
    out = BASE / f"{arm}-{seed}"
    out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():
        return
    if (out/"ema.pt").exists():
        model = PathDenoiser(classes=8 if arm=="C" else 6).cuda().eval()
        model.load_state_dict(torch.load(out/"ema.pt",weights_only=True))
    else:
        model = fit(arm,seed,out)
    request_files = {"g15": BASE/"requests.npz"}
    if (BASE/"g14-requests.npz").exists():
        request_files["g14"] = BASE/"g14-requests.npz"
    for domain, request_file in request_files.items():
        with np.load(request_file) as q:
            use = np.isin(q["labels"], range(8) if arm=="C" else range(1,6))
            sample(model,{k:q[k][use] for k in q.files},out/(domain+".npz"))
    # Original development queries use only the stored start and *planned* map paths.
    with np.load(ORIGINAL/"val.npz") as q:
        sample(model,dict(route=q["route"],length=q["route_length"],start=q["xy"][:,0],
                          labels=q["labels"],ids=q["source_index"]),out/"cnr.npz")
    (out/"complete.json").write_text(json.dumps(dict(arm=arm,seed=seed,steps=STEPS,K=K)))


if __name__ == "__main__":
    main(sys.argv[1],int(sys.argv[2]))
