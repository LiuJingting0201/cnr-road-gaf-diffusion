"""Generate a whole trip with training-time multiscale diffusion and an optional mode."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from cnr.route_paths import RoutePaths
from g16_data import NAMES
from g22_diffusion import decode
from g22_route import from_endpoints
from g23_select import targets
from g26_train import Denoiser,sample

MODES=[f"{band}-{pattern}" for band in ("low","middle","high") for pattern in ("steady","variable")]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",type=Path,required=True);p.add_argument("--mode",choices=NAMES,required=True)
    p.add_argument("--arm",choices=("H","HM","HV","HG"),default="HG")
    p.add_argument("--motion-mode",choices=MODES);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--route",type=Path);p.add_argument("--start",nargs=2,type=float);p.add_argument("--destination",nargs=2,type=float)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    c=NAMES.index(a.mode)
    if a.route:
        with np.load(a.route) as data:
            if data["label"]!=c:raise ValueError("Incompatible route")
            route,L=data["path"],float(data["length"])
    else:
        if a.start is None or a.destination is None:p.error("Choose a route or both endpoints")
        network=RoutePaths(str(a.assets/"most.net.xml"),radius=2.)
        route,L=from_endpoints(network,c,a.start,a.destination)
    req=dict(ids=np.array([f"OD_{a.seed}_{a.mode}"]),labels=np.array([c]),paths=route.astype(np.float32)[None],length=np.array([L]))
    with np.load(a.assets/"mode-prior.npz") as prior:
        wanted,_=targets(prior,req["labels"],req["length"],req["ids"],a.seed)
        if a.motion_mode is not None:
            if a.arm=="H":raise ValueError("The H control has no mode condition.")
            wanted[0]=MODES.index(a.motion_mode)
            if np.sum((prior["y"]==wanted[0])&(prior["participant"]==c))<40:
                raise ValueError("Explicit mode lacks the predeclared training support for this class.")
    den=Denoiser().cuda().eval()
    den.load_state_dict(torch.load(a.assets/f"{a.arm}.pt",weights_only=True,map_location="cuda"))
    stats=json.loads((a.assets/f"{a.arm}.json").read_text())["stats"]
    result=sample(den,a.arm,req,stats,a.seed,wanted)
    with a.out.open("x") as f:
        w=csv.writer(f);w.writerow(["time","id","x","y","type"])
        for k,(T,P) in enumerate(zip(result["duration"][0],result["progress"][0])):
            t,x=decode(route,T,P)
            w.writerows((f"{s:.8f}",f"{req['ids'][0]}_{k}",f"{v:.4f}",f"{z:.4f}",a.mode) for s,(v,z) in zip(t,x))
    print(json.dumps(dict(arm=a.arm,requested_mode=MODES[wanted[0]] if a.arm!="H" else None,
                          length_m=L,duration_s=result["duration"][0].tolist(),bounded=result["bounded"][0].tolist())))


if __name__=="__main__":main()
