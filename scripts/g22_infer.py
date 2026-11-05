"""Generate complete OD candidates and score the entire journeys, including arrival."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.path_diffusion import PathDenoiser
from cnr.route_paths import RoutePaths
from g16_data import NAMES
from g12_fit import features
from g19_gaf import images
from g21_train import Temporal
from g22_diffusion import sample,decode
from g22_temporal import blocks,trip_logits
from g22_route import from_endpoints


def generate(assets,mode,seed,route,start,dest):
    device="cuda"
    c=NAMES.index(mode)
    if route:
        with np.load(route) as r:
            path,length=r["path"],float(r["length"])
            if int(r["label"])!=c:raise ValueError("Example route is for a different class")
    else:
        matcher=RoutePaths(str(assets/"most.net.xml"),radius=2.)
        path,length=from_endpoints(matcher,c,start,dest)
    path=path.astype(np.float32)
    m=PathDenoiser(classes=8).to(device).eval()
    m.load_state_dict(torch.load(assets/"diffusion.pt",map_location=device,weights_only=True))
    req=dict(ids=[f"OD_{seed}_{mode}"],labels=np.array([c]),paths=path[None],length=np.array([length]))
    result=sample(m,req,json.loads((assets/"stats.json").read_text()),seed)
    trips=[decode(path,t,p) for t,p in zip(result["duration"][0],result["progress"][0])]
    g=resnet18(weights=None);g.fc=torch.nn.Linear(512,8);g.to(device).eval()
    g.load_state_dict(torch.load(assets/"gaf.pt",map_location=device,weights_only=True))
    tc=Temporal().to(device).eval();tc.load_state_dict(torch.load(assets/"temporal.pt",map_location=device,weights_only=True))
    im,ok=images([xy for _,xy in trips],json.loads((assets/"encoding.json").read_text()))
    with torch.no_grad():
        pg=g(features(torch.from_numpy(im).to(device))).softmax(1).cpu().numpy()[:,c]
        pt=trip_logits(tc,[blocks(xy) for _,xy in trips]).softmax(1).cpu().numpy()[:,c]
    valid=np.flatnonzero(ok)
    choice=int(valid[np.argmax(pg[valid])]) if len(valid) else None
    meta=dict(seed=seed,mode=mode,planned_start=path[0].tolist(),planned_destination=path[-1].tolist(),
              route_length_m=length,durations_s=result["duration"][0].tolist(),clock_bound=result["bounded"][0].tolist(),
              flat_fallback=result["flat"][0].tolist(),gaf_scores=pg.tolist(),gaf_valid=ok.tolist(),
              temporal_scores=pt.tolist(),gaf_choice=choice,temporal_choice=int(np.argmax(pt)))
    return trips,meta


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--assets",type=Path,required=True);p.add_argument("--mode",choices=NAMES,required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--route",type=Path)
    p.add_argument("--start",nargs=2,type=float);p.add_argument("--destination",nargs=2,type=float)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if not a.route and (a.start is None or a.destination is None):p.error("Specify both --start and --destination, or --route")
    trips,meta=generate(a.assets,a.mode,a.seed,a.route,a.start,a.destination)
    with a.out.open("x") as f:
        w=csv.writer(f);w.writerow(["time","id","x","y","type"])
        for k,(times,xy) in enumerate(trips):
            w.writerows((f"{t:.8f}",f"G22_{a.mode}_{a.seed}_{k}",f"{x:.4f}",f"{y:.4f}",a.mode)
                         for t,(x,y) in zip(times,xy))
    print(json.dumps(meta))
