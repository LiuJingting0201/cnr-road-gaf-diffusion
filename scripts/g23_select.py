"""Select complete candidates before consulting new reference motion."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from g12_fit import features
from g19_gaf import images
from g22_diffusion import decode
from g23_fit import classifier
from g23_modes import OUT,mode,speed_series,speed_images


def streams(task):
    paths,times,profiles=task
    xy=[];speeds=[];stats=[]
    for path,clock,rows in zip(paths,times,profiles):
        for T,p in zip(clock,rows):
            t,x=decode(path,T,p)
            v,s=speed_series(t,x)
            xy.append(x);speeds.append(v);stats.append(s)
    rgb,ok=images(xy)
    return rgb,ok,np.array(speeds),np.array(stats)


def targets(tr,labels,lengths,ids,seed):
    aware=[];blind=[]
    for c,L,ident in zip(labels,lengths,ids):
        n=int.from_bytes(hashlib.sha256(f"mode:{seed}:{ident}".encode()).digest()[:4],"little")
        u=np.random.default_rng(n).random()
        for group,allowed in [(aware,tr["participant"]==c),(blind,np.ones(len(tr["y"]),bool))]:
            ix=np.flatnonzero(allowed)
            near=ix[np.argsort(abs(np.log(tr["length"][ix]/L)),kind="stable")[:64]]
            prob=np.bincount(tr["y"][near],minlength=6)/len(near)
            group.append(int(np.searchsorted(prob.cumsum(),u)))
    return np.array(aware),np.array(blind)


def rank(v):
    # Average ranks for tied values; use the shared random order only to break final ties.
    return np.sum(v[:,None]>v[None,:],axis=1)+.5*np.sum(v[:,None]==v[None,:],axis=1)


@torch.no_grad()
def run():
    torch.set_num_threads(2)
    speed=classifier().eval()
    speed.load_state_dict(torch.load(OUT/"speed-mode.pt",weights_only=True,map_location="cuda"))
    spatial=resnet18(weights=None);spatial.fc=torch.nn.Linear(512,8)
    spatial.load_state_dict(torch.load("../g22-job/g22-data/cnn-native-gaf/model.pt",weights_only=True,map_location="cpu"))
    spatial.cuda().eval()
    keys=("ids","labels","paths","length")
    arrays={k:[] for k in keys}
    for f in sorted(Path("g22-data/fresh").glob("*/trips.npz")):
        with np.load(f) as d:
            for k in keys:arrays[k].append(d[k])
    q={k:np.concatenate(v) for k,v in arrays.items()}
    threshold=json.loads((OUT/"definition.json").read_text())
    tr=np.load(OUT/"train.npz")
    for seed in (42,43):
        p=np.load(OUT/f"pool-{seed}.npz")
        assert np.array_equal(q["ids"],p["ids"])
        desired,blind=targets(tr,q["labels"],q["length"],q["ids"],seed)
        selected={k:[] for k in ("uniform","spatial","speed","speed_blind","joint","numeric")}
        failed=0
        jobs=[(q["paths"][i:i+4],p["duration"][i:i+4],p["progress"][i:i+4]) for i in range(0,len(q["ids"]),4)]
        # Bounded futures to avoid buffering an entire corpus of encoded images in memory.
        with ProcessPoolExecutor(max_workers=10) as pool:
            for base in range(0,len(jobs),30):
                for offset,(rgb,ok,v,stats) in enumerate(pool.map(streams,jobs[base:base+30])):
                    sp=spatial(features(torch.from_numpy(rgb).cuda())).softmax(1).cpu().numpy()
                    mo=speed(speed_images(torch.from_numpy(v).cuda())).softmax(1).cpu().numpy()
                    truth=mode(stats,threshold)
                    failed+=int((~ok).sum())
                    for j in range(len(rgb)//4):
                        i=4*(base+offset)+j
                        perm=np.random.default_rng(int.from_bytes(hashlib.sha256(f"slot:{seed}:{q['ids'][i]}".encode()).digest()[:4],"little")).permutation(4)
                        group=slice(j*4,j*4+4)
                        cs=sp[group,q["labels"][i]]
                        cs=np.where(ok[group],cs,-1.)
                        ms=mo[group,desired[i]]
                        b=mo[group,blind[i]]
                        # Numeric oracle for the operational mode only. It uses no reference future.
                        z=truth[group]
                        cost=abs(z//2-desired[i]//2)+(z%2!=desired[i]%2)*.5
                        for name,score in [("uniform",np.zeros(4)),("spatial",cs),("speed",ms),("speed_blind",b),
                                           ("joint",rank(cs)+rank(ms)),("numeric",-cost)]:
                            selected[name].append(int(perm[np.argmax(score[perm])]))
        np.savez_compressed(OUT/f"selected-{seed}.npz",ids=q["ids"],desired=desired,blind=blind,failures=failed,**selected)
    (OUT/"selection-ready.txt").write_text("complete")


if __name__=="__main__":run()
