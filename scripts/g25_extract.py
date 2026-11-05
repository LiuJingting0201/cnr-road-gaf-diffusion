"""Extract two frozen penultimate representations for matched full trips."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from g12_fit import features
from g19_gaf import images
from g22_diffusion import decode
from g23_modes import speed_images,speed_series,mode

BASE=Path("g25-data")
DEF=json.loads(Path("../g23-job/g23-data/definition.json").read_text())


def inputs(batch):
    rgb,ok=images([z[:,1:] for z in batch])
    pairs=[speed_series(arr[:,0],arr[:,1:]) for arr in batch]
    return rgb,ok,np.stack([p for p,s in pairs]),np.stack([s for p,s in pairs])


def network(path,n):
    net=resnet18(weights=None);net.fc=nn.Linear(512,n)
    net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    head=net.fc.cuda().eval()
    net.fc=nn.Identity()
    return net.cuda().eval(),head


def observed(files):
    ids=[];labs=[];lens=[];ends=[];clocks=[];tracks=[]
    for f in files:
        with np.load(f) as r,np.load(f.parent/"trips.npz") as d:
            assert np.array_equal(r["ids"],d["ids"])
            ids.extend(r["ids"]);labs.extend(d["labels"]);lens.extend(d["length"])
            ends.extend(d["paths"][:,[0,-1]].reshape(-1,4));clocks.extend(d["duration"])
            arr,off=r["txy"],r["offset"]
            tracks.extend([arr[a:b] for a,b in zip(off,off[1:])])
    return dict(ids=np.array(ids),label=np.array(labs),length=np.array(lens),ends=np.array(ends),clock=np.array(clocks)),tracks


def generated(arm,seed):
    root=Path("../g24-job/g24-data")
    with np.load(root/"requests.npz") as q,np.load(root/f"{arm}-{seed}.npz") as d:
        assert np.array_equal(q["ids"],d["ids"])
        tracks=[]
        for path,ts,profiles in zip(q["paths"],d["duration"],d["progress"]):
            for duration,profile in zip(ts,profiles):
                t,x=decode(path,duration,profile)
                tracks.append(np.c_[t,x])
        meta=dict(ids=np.repeat(q["ids"],4),slot=np.tile(np.arange(4),len(q["ids"])),target=np.repeat(d["target"],4),
                  label=np.repeat(q["labels"],4),length=np.repeat(q["length"],4),clock=d["duration"].ravel(),
                  ends=np.repeat(q["paths"][:,[0,-1]].reshape(-1,4),4,axis=0))
    return meta,tracks


@torch.no_grad()
def write(out,meta,tracks,net,head,speed,shead,pool):
    zz=[];ss=[];pc=[];ps=[];flags=[];stats=[]
    for start in range(0,len(tracks),128):
        batches=[tracks[i:i+16] for i in range(start,min(start+128,len(tracks)),16)]
        for im,ok,v,s in pool.map(inputs,batches):
            z=net(features(torch.from_numpy(im).cuda()))
            h=speed(speed_images(torch.from_numpy(v).cuda()))
            zz.append(z.cpu().numpy());ss.append(h.cpu().numpy())
            pc.append(head(z).softmax(1).cpu().numpy());ps.append(shead(h).softmax(1).cpu().numpy())
            flags.extend(ok);stats.extend(s)
    np.savez_compressed(out,**meta,Z=np.concatenate(zz),S=np.concatenate(ss),class_p=np.concatenate(pc),mode_p=np.concatenate(ps),
                        valid=flags,phys=stats,mode=mode(np.array(stats),DEF))
    print(out,len(tracks),sum(flags),flush=True)


if __name__=="__main__":
    BASE.mkdir(exist_ok=True);torch.set_num_threads(1)
    net,head=network("../g22-job/g22-data/cnn-native-gaf/model.pt",8)
    speed,shead=network("../g23-job/g23-data/speed-mode.pt",6)
    root=Path("../g22-job/g22-data")
    allfiles={k:sorted((root/k).glob("*/quarter.npz")) for k in ("train","monitor")}
    allfiles["g23"]=sorted(Path("../g23-job/g22-data/fresh").glob("*/quarter.npz"))
    with ProcessPoolExecutor(max_workers=12) as pool:
        for name,files in allfiles.items():
            out=BASE/(name+".npz")
            if not out.exists():
                write(out,*observed(files),net,head,speed,shead,pool)
        root=Path("../g24-job/g24-data")
        if not (BASE/"g24.npz").exists():
            with np.load(root/"requests.npz") as q,np.load(root/"references.npz") as r:
                assert np.array_equal(q["ids"],r["ids"])
                arr,off=r["txy"],r["offset"]
                meta=dict(ids=q["ids"],label=q["labels"],length=q["length"],clock=r["duration"],ends=q["paths"][:,[0,-1]].reshape(-1,4))
                write(BASE/"g24.npz",meta,[arr[a:b] for a,b in zip(off,off[1:])],net,head,speed,shead,pool)
        for name in ("plain","guided","permuted"):
            for seed in (42,43):
                out=BASE/f"{name}-{seed}.npz"
                if not out.exists():write(out,*generated(name,seed),net,head,speed,shead,pool)
    (BASE/"features-ready.txt").write_text("complete")
