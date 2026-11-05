"""Secondary map check and frozen spatial-reader score on one predeclared draw per route."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import shapely
import torch
from torchvision.models import resnet18
from cnr.topology import LaneConsistency
from g12_fit import features
from g16_data import NAMES
from g19_gaf import images
from g22_diffusion import decode
from g26_eval import plain
from g26_train import ARMS,OUT


def episode(job):
    arm,ids,labels,paths,clock,profile=job
    map_matcher=LaneConsistency("data/maps/most.net.xml",radius=5,max_candidates=64)
    tracks=[];passed=[]
    for c,path,t,p in zip(labels,paths,clock,profile):
        _,xy=decode(path,t,p)
        if c==0:
            valid=np.max(shapely.distance(shapely.points(xy),shapely.LineString(path)))<.02
        else:
            r=map_matcher.match(xy,NAMES[c])
            valid=r["graph_consistent"] and not r["candidate_truncation_warning"]
        passed.append(valid);tracks.append(xy)
    tables=[images(tracks[i:i+16]) for i in range(0,len(tracks),16)]
    return arm,ids,labels,passed,np.concatenate([v[0] for v in tables]),np.concatenate([v[1] for v in tables])


@torch.no_grad()
def run():
    torch.set_num_threads(1)
    with np.load(OUT/"requests.npz") as d:req={k:d[k] for k in d.files}
    net=resnet18(weights=None);net.fc=torch.nn.Linear(512,8)
    net.load_state_dict(torch.load("../g22-job/g22-data/cnn-native-gaf/model.pt",map_location="cpu",weights_only=True))
    net.cuda().eval()
    jobs=[]
    eps=np.array([int(s.split("_")[0]) for s in req["ids"]])
    for arm in ("G22",*ARMS):
        p=plain(req,42) if arm=="G22" else np.load(OUT/f"{arm}-42/samples.npz")
        assert np.array_equal(p["ids"],req["ids"])
        for ep in np.unique(eps):
            ix=eps==ep
            jobs.append((arm,req["ids"][ix],req["labels"][ix],req["paths"][ix],p["duration"][ix,0],p["progress"][ix,0]))
    out={}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for arm,ids,y,valid,rgb,ok in pool.map(episode,jobs):
            logits=net(features(torch.from_numpy(rgb).cuda()))
            probs=logits.softmax(1).cpu().numpy()[np.arange(len(y)),y]
            a=out.setdefault(arm,defaultdict(list))
            for c,good,p,support in zip(y,valid,probs,ok):
                a[NAMES[c]].append([int(good),float(p) if support else None])
    summary={}
    for arm,table in out.items():
        summary[arm]={n:dict(graph_or_ped_pass=sum(i for i,p in v),attempts=len(v),
                             spatial_score=float(np.mean([p for _,p in v if p is not None])),
                             encoded=sum(p is not None for _,p in v)) for n,v in table.items()}
    (OUT/"checks.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":run()
