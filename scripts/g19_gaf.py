"""Train eight-class feedback on the inherited, frozen Hilbert image representation."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18
from cnr.original_gaf import trajectory_sequence, encode_frozen
from g12_fit import features
from g16_data import csv_tracks

ROOT=Path("g19-data")
from functools import lru_cache

@lru_cache(maxsize=1)
def original_metadata():
    return json.loads(Path("../runs/long-path-e3/seed42/config.json").read_text())["dataset"]["source_protocol"]


def images(xy, meta=None):
    if meta is None:
        meta=original_metadata()
    seq,valid=[],[]
    for track in xy:
        try:
            h,_=trajectory_sequence(track,np.array(meta["xy_min"]),np.array(meta["xy_max"]),eps=meta["hilbert_eps"])
            valid.append(True)
        except ValueError:
            h=np.zeros(224)
            valid.append(False)
        seq.append(h)
    rgb,_=encode_frozen(seq,np.array(meta["channel_min"]),np.array(meta["channel_max"]),batch=16)
    return rgb,np.array(valid)


def build():
    rng=np.random.default_rng(19042)
    chosen=[set(rng.choice(8000,1152,replace=False)) for _ in range(8)]
    counts=np.zeros(8,int)
    tracks,labels,ids=[],[],[]
    for ident,c,xy in csv_tracks("train"):
        if counts[c] in chosen[c]:
            tracks.append(xy)
            labels.append(c)
            ids.append(ident)
        counts[c]+=1
    assert (counts==8000).all() and len(set(ids))==9216
    groups=[tracks[i:i+16] for i in range(0,len(tracks),16)]
    path=ROOT/"images.npy"
    im=np.lib.format.open_memmap(path,mode="w+",dtype=np.uint8,shape=(len(tracks),224,224,3))
    flags=[]
    with ProcessPoolExecutor(max_workers=10) as pool:
        for i,(rgb,ok) in enumerate(pool.map(images,groups)):
            im[i*16:i*16+len(rgb)]=rgb
            flags.extend(ok)
    im.flush()
    labels=np.array(labels)
    monitor=np.zeros(len(labels),bool)
    for c in range(8):
        monitor[rng.choice(np.flatnonzero(labels==c),128,replace=False)]=True
    np.savez(ROOT/"image-index.npz",ids=ids,labels=labels,valid=flags,monitor=monitor)


def model():
    net=resnet18(weights=None)
    net.fc=torch.nn.Linear(512,8)
    return net.cuda()


def train():
    torch.set_num_threads(2)
    torch.manual_seed(19042)
    net=model()
    old=torch.load("../g12-data/reg-scorer42.pt",weights_only=True,map_location="cpu")
    del old["fc.weight"],old["fc.bias"]
    net.load_state_dict(old,strict=False)
    index=np.load(ROOT/"image-index.npz")
    im=np.load(ROOT/"images.npy",mmap_mode="r")
    y=index["labels"]
    fit=np.flatnonzero(index["valid"]&~index["monitor"])
    monitor=np.flatnonzero(index["valid"]&index["monitor"])
    rng=np.random.default_rng(19042)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
    history=[]
    for epoch in range(5):
        net.train()
        order=rng.permutation(fit)
        for ix in np.array_split(order,int(np.ceil(len(order)/64))):
            logits=net(features(torch.from_numpy(im[ix]).cuda()))
            loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y[ix]).cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred=np.concatenate([net(features(torch.from_numpy(im[ix]).cuda())).argmax(1).cpu().numpy()
                                 for ix in np.array_split(monitor,16)])
        history.append(float(np.mean(pred==y[monitor])))
        print("gaf epoch",epoch+1,history[-1],flush=True)
    torch.save(net.state_dict(),ROOT/"gaf8.pt")
    (ROOT/"gaf-fit.json").write_text(json.dumps(dict(history=history,fit=len(fit),monitor=len(monitor),
        failures=int((~index["valid"]).sum()),cm=np.bincount(8*y[monitor]+pred,minlength=64).reshape(8,8).tolist()),indent=2))


if __name__=="__main__":
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/"image-index.npz").exists():
        build()
    train()
