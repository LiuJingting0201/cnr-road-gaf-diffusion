"""Fit original full-trip Hilbert and alternative physical-image CNNs."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torchvision.models import resnet18
from g12_fit import features
from g21_train import scores
from g22_images import ROOT,fields


def dataset(split,arm):
    arrays={k:[] for k in ("input","ids","labels")}
    filename="quarter-images.npz" if arm=="native-gaf" else "image-features.npz"
    for p in sorted((ROOT/split).glob("*/"+filename)):
        with np.load(p) as d:
            keep=d["ok"] if arm.startswith("full") or arm=="native-gaf" else (d["sok"] if arm=="short-paired" else np.ones(len(d["ids"]),bool))
            if arm.endswith("paired"): keep=keep & d["eligible"] & d["ok"] & d["sok"]
            key="va" if arm=="motion" else ("short" if arm=="short-paired" else "full")
            arrays["input"].append(d[key][keep])
            for k in ("ids","labels"): arrays[k].append(d[k][keep])
    return {k:np.concatenate(v) for k,v in arrays.items()}


def train(arm):
    torch.set_num_threads(2);torch.manual_seed(2242)
    torch.backends.cudnn.benchmark=True
    out=ROOT/("cnn-"+arm)
    out.mkdir(exist_ok=True)
    train=dataset("train",arm);monitor=dataset("monitor",arm)
    net=resnet18(weights=None);net.fc=torch.nn.Linear(512,8)
    stem="physical" if arm=="motion" else "original"
    net.load_state_dict(torch.load(f"../g21-job/g21-data/{stem}-42/model.pt",map_location="cpu",weights_only=True))
    net.cuda().to(memory_format=torch.channels_last)
    transform=fields if arm=="motion" else features
    def batch(x):
        return transform(torch.from_numpy(x).cuda()).contiguous(memory_format=torch.channels_last)
    def predict(d):
        net.eval();pred=[]
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            for i in range(0,len(d["ids"]),128):
                pred.append(net(batch(d["input"][i:i+128])).float().softmax(1).cpu().numpy())
        return np.concatenate(pred)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,18,eta_min=1e-5)
    rng=np.random.default_rng(2242);best=-1
    for epoch in range(1,19):
        net.train();order=rng.permutation(len(train["ids"]))
        for ix in np.array_split(order,int(np.ceil(len(order)/64))):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=net(batch(train["input"][ix])).float()
                loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(train["labels"][ix]).cuda(),label_smoothing=.025)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        scheduler.step()
        pr=predict(monitor);score=scores(monitor["labels"],pr)
        print(epoch,score["macro_f1"],flush=True)
        if score["macro_f1"]>best:
            selected=epoch;best=score["macro_f1"];torch.save(net.state_dict(),out/"model.pt")
            np.savez(out/"monitor.npz",ids=monitor["ids"],labels=monitor["labels"],prob=pr)
    net.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    ev=dataset("evaluation",arm);pr=predict(ev)
    np.savez(out/"evaluation.npz",ids=ev["ids"],labels=ev["labels"],prob=pr)
    (out/"complete.json").write_text(json.dumps(dict(arm=arm,selected=selected,fit=len(train["ids"]),monitor=len(monitor["ids"]),
        evaluated=len(ev["ids"]),results=scores(ev["labels"],pr)),indent=2))


if __name__=="__main__":train(sys.argv[1])
