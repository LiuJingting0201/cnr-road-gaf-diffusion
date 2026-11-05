"""Score the same four completed journeys separately by requested type and speed mode."""
import argparse
import csv
from itertools import groupby
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from g12_fit import features
from g16_data import NAMES
from g19_gaf import images
from g23_modes import speed_series,speed_images
from g23_select import targets,rank


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--mode",choices=NAMES,required=True)
    p.add_argument("--route-length",type=float,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--request-id",required=True)
    a=p.parse_args()
    ids=[];xy=[];v=[]
    with a.csv.open() as f:
        for ident,rows in groupby(csv.DictReader(f),key=lambda r:r["id"]):
            r=list(rows);t=np.array([float(row["time"]) for row in r])
            x=np.array([[float(row["x"]),float(row["y"])] for row in r])
            if r[0]["type"]!=a.mode or not np.all(np.diff(t)>0):
                raise ValueError("The CSV must contain ordered candidates for the requested class.")
            ids.append(ident);xy.append(x);v.append(speed_series(t,x)[0])
    if len(xy)!=4 or len(set(ids))!=4:raise ValueError("Expected a single four-candidate pool.")
    c=NAMES.index(a.mode)
    with np.load(a.assets/"mode-prior.npz") as prior:
        z,_=targets(prior,[c],[a.route_length],[a.request_id],a.seed)
    spatial=resnet18(weights=None);spatial.fc=torch.nn.Linear(512,8)
    spatial.load_state_dict(torch.load(a.assets/"gaf.pt",weights_only=True,map_location="cpu"))
    speed=resnet18(weights=None);speed.fc=torch.nn.Linear(512,6)
    speed.load_state_dict(torch.load(a.assets/"speed-mode.pt",weights_only=True,map_location="cpu"))
    spatial.cuda().eval();speed.cuda().eval()
    im,ok=images(xy,json.loads((a.assets/"encoding.json").read_text()))
    with torch.no_grad():
        cs=spatial(features(torch.from_numpy(im).cuda())).softmax(1)[:,c].cpu().numpy()
        ms=speed(speed_images(torch.from_numpy(np.array(v)).cuda())).softmax(1)[:,int(z[0])].cpu().numpy()
    cs[~ok]=-1
    import hashlib
    key=int.from_bytes(hashlib.sha256(f"slot:{a.seed}:{a.request_id}".encode()).digest()[:4],"little")
    perm=np.random.default_rng(key).permutation(4)
    pick=lambda score:ids[int(perm[np.argmax(score[perm])])]
    print(json.dumps(dict(target_participant=a.mode,target_speed_mode=int(z[0]),ids=ids,
        class_scores=cs.tolist(),motion_scores=ms.tolist(),spatial_pick=pick(cs),speed_pick=pick(ms),
        joint_pick=pick(rank(cs)+rank(ms))),indent=2))


if __name__=="__main__":main()
