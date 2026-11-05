"""Generate four diffusion trajectories and identify the original Hilbert–GAF choice."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser
from cnr.route_paths import RoutePaths
from g12_fit import features
from g16_data import NAMES,walk
from g19_gaf import images,model as scorer_model
from g19_field import MapField
from g19_train import generate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--assets",required=True,type=Path)
    parser.add_argument("--mode",required=True,choices=NAMES)
    parser.add_argument("--x",required=True,type=float)
    parser.add_argument("--y",required=True,type=float)
    parser.add_argument("--seed",required=True,type=int)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assets=args.assets
    network=assets/"most.net.xml"
    matcher=RoutePaths(str(network),radius=2.)
    start=np.array([args.x,args.y])
    if args.mode=="pedestrian":
        plan=walk(matcher,start)
    else:
        prior=json.loads((assets/"route_prior.json").read_text())
        counts={(r["class"],matcher.ids[r["from"]],matcher.ids[r["to"]]):r["count"] for r in prior}
        plan=matcher.sample(start,args.mode,counts,np.random.default_rng(args.seed),budget=4096.)
    if plan is None or plan[1]<64:
        raise ValueError("No compatible route at the requested start.")
    path,length=plan[:2]
    field=MapField(np.load(assets/"factor.npy"),network)
    cn=NAMES.index(args.mode)
    ident=f"G19_{args.mode}_{args.seed}"
    req=dict(route=path[None],length=np.array([length],np.float32),start=start[None],
             labels=np.array([cn]),ids=[ident])
    den=PathDenoiser(classes=8).cuda().eval()
    den.load_state_dict(torch.load(assets/"road.pt",weights_only=True,map_location="cuda"))
    generated=generate(den,req,field,json.loads((assets/"normalization.json").read_text()))
    xy=generated["xy"][0]
    net=scorer_model().eval()
    net.load_state_dict(torch.load(assets/"gaf8.pt",weights_only=True,map_location="cuda"))
    rgb,ok=images(xy,json.loads((assets/"encoding.json").read_text()))
    with torch.no_grad():
        scores=net(features(torch.from_numpy(rgb).cuda())).softmax(1)[:,cn].cpu().numpy()
    supported=np.flatnonzero(ok)
    selected=int(supported[np.argmax(scores[supported])]) if len(supported) else 0
    with args.out.open("x") as stream:
        writer=csv.writer(stream)
        writer.writerow(["time","id","x","y","type"])
        for i,track in enumerate(xy):
            writer.writerows((t*.25,f"{ident}_{i}",f"{x:.3f}",f"{y:.3f}",args.mode)
                             for t,(x,y) in enumerate(track))
    print(json.dumps(dict(gaf_selected_id=f"{ident}_{selected}",encodable=ok.tolist(),
                          capped=generated["caps"][0].tolist(),target_scores=scores.tolist())))


if __name__=="__main__":
    main()
