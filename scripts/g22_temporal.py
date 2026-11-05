"""A variable-length trip classifier: shared temporal convolution and duration-weighted pooling."""
import json
from pathlib import Path
import numpy as np
import torch
from g21_train import Temporal,scores
from g22_diffusion import ROOT


def blocks(xy):
    x=xy.astype(float)
    v=4*np.linalg.norm(np.diff(x,axis=0),axis=1)
    # Evenly cover the entire trip with comparable-sized physical-time blocks.
    # Short final/slow trips retain absolute speeds, time interpolation changes only the grid.
    parts=np.array_split(v,max(1,int(np.ceil(len(v)/511))))
    rows=np.stack([np.interp(np.linspace(0,len(p)-1,511),np.arange(len(p)),p) for p in parts])
    return rows.astype(np.float32),np.array([len(p)/len(v) for p in parts],np.float32)


def dataset(split):
    ids,labels,sequence=[],[],[]
    for f in sorted((ROOT/split).glob("*/quarter.npz")):
        with np.load(f) as hi, np.load(f.parent/"trips.npz") as trips:
            assert np.array_equal(hi["ids"],trips["ids"])
            ids.extend(trips["ids"]);labels.extend(trips["labels"])
            tx,off=hi["txy"],hi["offset"]
            for a,b in zip(off,off[1:]):
                sequence.append(blocks(tx[a:b,1:]))
    return np.array(ids),np.array(labels),sequence


def trip_logits(model,seq):
    v=torch.from_numpy(np.concatenate([x for x,w in seq])).cuda()
    w=torch.from_numpy(np.concatenate([w for x,w in seq])).cuda()
    i=torch.from_numpy(np.repeat(np.arange(len(seq)),[len(x) for x,w in seq])).cuda()
    logits=model(v).float()*w[:,None]
    return torch.zeros(len(seq),8,device="cuda").index_add(0,i,logits)


def main():
    torch.set_num_threads(2);torch.manual_seed(2242)
    out=ROOT/"cnn-native-full"
    out.mkdir(exist_ok=True)
    ids,y,seq=dataset("train")
    mid,ym,sm=dataset("monitor")
    model=Temporal().cuda()
    model.load_state_dict(torch.load("../g21-job/g21-data/temporal-42/model.pt",weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=1e-5)
    rng=np.random.default_rng(2242);best=-1
    def predict(s):
        model.eval()
        with torch.no_grad():
            return np.concatenate([trip_logits(model,s[i:i+64]).softmax(1).cpu().numpy() for i in range(0,len(s),64)])
    for epoch in range(1,25):
        model.train()
        for ix in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/48))):
            logits=trip_logits(model,[seq[i] for i in ix])
            loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y[ix]).cuda(),label_smoothing=.025)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        sched.step()
        p=predict(sm);r=scores(ym,p);print(epoch,r["macro_f1"],flush=True)
        if r["macro_f1"]>best:
            best=r["macro_f1"];selected=epoch
            torch.save(model.state_dict(),out/"model.pt")
            np.savez(out/"monitor.npz",ids=mid,labels=ym,prob=p)
    model.load_state_dict(torch.load(out/"model.pt",weights_only=True))
    ids,y,s=dataset("evaluation");p=predict(s)
    np.savez(out/"evaluation.npz",ids=ids,labels=y,prob=p)
    (out/"complete.json").write_text(json.dumps(dict(fit=len(seq),monitor=len(sm),selected=selected,evaluated=len(y),results=scores(y,p)),indent=2))


if __name__=="__main__":main()
