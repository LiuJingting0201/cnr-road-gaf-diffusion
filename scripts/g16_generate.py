"""Generate four timed trajectories from one start, class and random seed."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from cnr.path_diffusion import PathDenoiser
from cnr.route_paths import RoutePaths
from g16_data import NAMES, walk
from g16_train import sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--mode", choices=NAMES, required=True)
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    matcher = RoutePaths(str(a.assets/"most.net.xml"),radius=2.)
    start = np.array([a.x,a.y])
    if a.mode=="pedestrian":
        plan = walk(matcher,start)
    else:
        prior = json.loads((a.assets/"route_prior.json").read_text())
        counts = {(r["class"],matcher.ids[r["from"]],matcher.ids[r["to"]]):r["count"] for r in prior}
        plan = matcher.sample(start,a.mode,counts,np.random.default_rng(a.seed),budget=4096.)
    if plan is None or plan[1]<64:
        raise ValueError("No supported path at this start; choose a permitted lane within 2 m.")
    route, length = plan[:2]
    meta = json.loads((a.assets/"normalization.json").read_text())
    model = PathDenoiser(classes=len(NAMES)).cuda().eval()
    model.load_state_dict(torch.load(a.assets/"ema.pt",weights_only=True,map_location="cuda"))
    ident = f"g16_{a.mode}_{a.seed}"
    with TemporaryDirectory() as tmp:
        f = Path(tmp)/"sample.npz"
        sample(model,dict(route=route[None],length=[length],start=start[None],
                          labels=[NAMES.index(a.mode)],ids=[ident]),f,meta)
        result = np.load(f)
        with a.out.open("x") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time","id","x","y","type"))
            for slot,xy in enumerate(result["xy"][0]):
                writer.writerows((f"{i*.25:.2f}",f"{ident}_{slot}",f"{x:.3f}",f"{y:.3f}",a.mode)
                                 for i,(x,y) in enumerate(xy))
        print(json.dumps(dict(route_length_m=length,capped=result["caps"].tolist())))


if __name__=="__main__":
    main()
