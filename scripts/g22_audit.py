"""Audit the actual emitted quarter-second sample at fixed seed/slot, one per route."""
import json
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import shapely
from cnr.topology import LaneConsistency
from g16_data import NAMES
from g22_diffusion import ROOT,decode


def audit_episode(file):
    graph=LaneConsistency("data/maps/most.net.xml",radius=5,max_candidates=64)
    generated=np.load(ROOT/"model-42/samples.npz")
    index={ident:i for i,ident in enumerate(generated["ids"])}
    with np.load(file) as q:
        ids,labs,paths=q["ids"],q["labels"],q["paths"]
    T,P=generated["duration"],generated["progress"]
    results={}
    for ident,c,path in zip(ids,labs,paths):
        i=index[ident]
        t,x=decode(path,T[i,0],P[i,0])
        if c==0:
            line=shapely.LineString(path)
            good=bool(np.max(shapely.distance(shapely.points(x),line))<=.02)
        else:
            r=graph.match(x,NAMES[c]);good=r["graph_consistent"] and not r["candidate_truncation_warning"]
        row=results.setdefault(NAMES[c],[0,0])
        row[0]+=int(good);row[1]+=1
    return results


if __name__=="__main__":
    files=sorted((ROOT/"evaluation").glob("*/trips.npz"))
    with ProcessPoolExecutor(max_workers=4) as pool:r=list(pool.map(audit_episode,files))
    summary={name:np.sum([v[name] for v in r],axis=0).tolist() for name in NAMES}
    (ROOT/"graph-audit.json").write_text(json.dumps(summary,indent=2))
