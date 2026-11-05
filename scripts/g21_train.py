"""Short-window participant recognition with original images and two motion CNNs."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from g12_fit import features

ROOT=Path(os.getenv("CNR_CLASSIFIER_DATA","g21-data"))
EPOCHS={"original":16,"physical":20,"temporal":30}


def motion(v):
    second=v[:,:508].reshape(-1,127,4).mean(-1)
    accel=torch.diff(second,dim=1,prepend=second[:,:1])
    return second,accel


def angular(v):
    speed,acc=motion(v)
    # Fixed scales preserve absolute units across trajectories. These are motion images,
    # distinct from the inherited time-ordered spatial Hilbert representation.
    u=(speed/32).clamp(0,1)
    a=torch.tanh(acc/2)
    p=(1-u.square()).clamp_min(0).sqrt()
    b=(1-a.square()).clamp_min(0).sqrt()
    return torch.stack([u[:,:,None]*u[:,None,:]-p[:,:,None]*p[:,None,:],
                        a[:,:,None]*a[:,None,:]-b[:,:,None]*b[:,None,:],
                        b[:,:,None]*a[:,None,:]-a[:,:,None]*b[:,None,:]],1)


class Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(3,48,7,padding=3),nn.BatchNorm1d(48),nn.SiLU(),
            nn.Conv1d(48,64,5,padding=2),nn.BatchNorm1d(64),nn.SiLU(),
            nn.Conv1d(64,96,5,padding=4,dilation=2),nn.BatchNorm1d(96),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(96*2+9*4+4,192),nn.BatchNorm1d(192),nn.SiLU(),
                                nn.Dropout(.15),nn.Linear(192,8))

    def forward(self,v):
        s,a=motion(v)
        raw=torch.diff(v,dim=1,prepend=v[:,:1])*4
        seq=torch.stack([v/16,torch.asinh(raw/2),torch.asinh(torch.diff(raw,dim=1,prepend=raw[:,:1])/2)],1)
        h=self.conv(seq)
        grid=[0,.02,.1,.25,.5,.75,.9,.98,1]
        fields=[]
        for arr in [v/16,torch.asinh(raw/2),s/16,torch.asinh(a/2)]:
            order=arr.sort(1).values
            fields.append(order[:,[round(q*(arr.shape[1]-1)) for q in grid]])
        extra=torch.stack([v.mean(1)/16,v.std(1)/8,(v<.3).float().mean(1),a.abs().mean(1)],1)
        return self.head(torch.cat([h.mean(-1),h.amax(-1),*fields,extra],1))


def build(arm,initialize=True):
    if arm=="temporal":
        return Temporal()
    net=resnet18(weights=None)
    if initialize:
        weights=torch.load(Path.home()/".cache/torch/hub/checkpoints/resnet18-f37072fd.pth",
                           map_location="cpu",weights_only=True)
        net.load_state_dict(weights)
    net.fc=nn.Linear(512,8)
    return net


class Inputs:
    def __init__(self,folder,arm):
        i=np.load(folder/"index.npz")
        self.ids,self.labels,self.valid=i["ids"],i["labels"],i["valid"]
        self.arm=arm
        if arm=="original":
            self.data=np.load(folder/"images.npy",mmap_mode="r")
        else:
            self.data=torch.from_numpy(np.load(folder/"speed.npy")).cuda()

    def take(self,ix):
        if self.arm=="original":
            return features(torch.from_numpy(self.data[ix]).cuda()).contiguous(memory_format=torch.channels_last)
        v=self.data[ix]
        return angular(v).contiguous(memory_format=torch.channels_last) if self.arm=="physical" else v


def scores(labels,prob):
    pred=prob.argmax(1)
    cm=np.bincount(8*labels+pred,minlength=64).reshape(8,8)
    f=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(accuracy=float(np.trace(cm)/len(labels)),macro_f1=float(f.mean()),
                weighted_f1=float(np.average(f,weights=cm.sum(1))),cm=cm.tolist())


@torch.no_grad()
def predict(net,inputs,ix):
    net.eval()
    out=[]
    for pos in range(0,len(ix),192):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out.append(net(inputs.take(ix[pos:pos+192])).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def train(arm,seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    torch.backends.cudnn.benchmark=True
    out=ROOT/f"{arm}-{seed}"
    out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():
        return
    tr=Inputs(ROOT/"train",arm)
    episodes=np.array([int(s.split("_")[0]) for s in tr.ids])
    monitor=np.isin(episodes,np.unique(episodes)[-8:])
    fit=np.flatnonzero(tr.valid&~monitor)
    held=np.flatnonzero(tr.valid&monitor)
    net=build(arm).cuda()
    if arm!="temporal":
        net=net.to(memory_format=torch.channels_last)
    rates={"original":2e-4,"physical":4e-4,"temporal":1e-3}
    opt=torch.optim.AdamW(net.parameters(),lr=rates[arm],weight_decay=.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS[arm],eta_min=rates[arm]/20)
    best=-1
    hist=[]
    begin=time.monotonic()
    for epoch in range(1,EPOCHS[arm]+1):
        net.train()
        order=rng.permutation(fit)
        for ix in np.array_split(order,int(np.ceil(len(order)/128))):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=net(tr.take(ix))
                loss=nn.functional.cross_entropy(logits.float(),torch.from_numpy(tr.labels[ix]).cuda(),label_smoothing=.025)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5)
            opt.step()
        sched.step()
        prob=predict(net,tr,held)
        row=scores(tr.labels[held],prob)
        record=dict(epoch=epoch,accuracy=row["accuracy"],macro_f1=row["macro_f1"],seconds=time.monotonic()-begin)
        hist.append(record)
        print(record,flush=True)
        if row["macro_f1"]>best:
            best=row["macro_f1"]
            selected=epoch
            torch.save(net.state_dict(),out/"model.pt")
            np.savez(out/"monitor.npz",prob=prob,ids=tr.ids[held],labels=tr.labels[held])
    net.load_state_dict(torch.load(out/"model.pt",map_location="cuda",weights_only=True))
    va=Inputs(ROOT/"validation",arm)
    ix=np.flatnonzero(va.valid)
    prob=predict(net,va,ix)
    np.savez(out/"validation.npz",prob=prob,ids=va.ids[ix],labels=va.labels[ix])
    (out/"complete.json").write_text(json.dumps(dict(selected_epoch=selected,fit=len(fit),monitor=len(held),
        validation=scores(va.labels[ix],prob),history=hist),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=EPOCHS)
    p.add_argument("seed",type=int)
    a=p.parse_args()
    train(a.arm,a.seed)
