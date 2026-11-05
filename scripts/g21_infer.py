"""Classify a five-column file of 512-point tracks with three distinct CNN representations."""
import argparse
import csv
import json
from itertools import groupby
from pathlib import Path
import numpy as np
import torch
from g19_gaf import images
from g21_train import build,angular,features

NAMES=["pedestrian","bus","motorcycle","private_car","bycicle","taxi_or_uber","truck","delivery"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",required=True,type=Path);p.add_argument("--csv",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    meta=json.loads((a.assets/"encoding.json").read_text());selection=json.loads((a.assets/"selection.json").read_text())
    models={}
    for arm in ("original","physical","temporal"):
        models[arm]=[]
        for seed in (42,43):
            net=build(arm,False).cuda().eval()
            net.load_state_dict(torch.load(a.assets/f"{arm}-{seed}.pt",weights_only=True,map_location="cuda"))
            models[arm].append(net)
    with a.csv.open() as src,a.out.open("x") as target:
        for ident,rows in groupby(csv.DictReader(src),key=lambda r:r["id"]):
            r=list(rows);xy=np.array([[q["x"],q["y"]] for q in r],float);t=np.array([q["time"] for q in r],float)
            if xy.shape!=(512,2) or not np.allclose(np.diff(t),.25):raise ValueError("Expected 512 consecutive 0.25s observations")
            rgb,ok=images(xy[None],meta)
            v=torch.tensor(4*np.linalg.norm(np.diff(xy,axis=0),axis=1)[None],dtype=torch.float32,device="cuda")
            inp=dict(original=features(torch.from_numpy(rgb).cuda()),physical=angular(v),temporal=v)
            probs={}
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                for arm in models:
                    probs[arm]=np.mean([net(inp[arm]).float().softmax(1)[0].cpu().numpy() for net in models[arm]],axis=0)
            motion=probs[selection["motion"]];w=selection["weight"] if ok[0] else 0
            probs["fusion"]=w*probs["original"]+(1-w)*motion
            target.write(json.dumps(dict(id=ident,encoding_ok=bool(ok[0]),predicted={k:NAMES[v.argmax()] for k,v in probs.items()},
                         probabilities={k:v.tolist() for k,v in probs.items()}))+"\n")


if __name__=="__main__":main()
