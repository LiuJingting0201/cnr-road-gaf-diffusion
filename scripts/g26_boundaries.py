"""Post-hoc check: compare speed discontinuity at internal mass-allocation boundaries."""
import json
from pathlib import Path
import numpy as np
from g26_train import OUT,ARMS
from g26_eval import plain


def diagnostic(P,T,L):
    v=256*L[...,None]*np.diff(P.astype(float),axis=-1)/T[...,None]
    changes=abs(np.diff(v,axis=-1))
    boundary=np.arange(1,256)%16==0
    return np.stack([changes[...,boundary].mean(-1),changes[...,~boundary].mean(-1)],axis=-1)


def main():
    with np.load(OUT/"requests.npz") as z:q={k:z[k] for k in z.files}
    byid={}
    for f in sorted(Path("g22-data/fresh").glob("*/trips.npz")):
        with np.load(f) as d:
            vals=diagnostic(d["progress"],d["duration"],d["length"])
            byid.update(zip(d["ids"],vals))
    modes={"reference":np.stack([byid[i] for i in q["ids"]])}
    for arm in ("G22",*ARMS):
        vals=[]
        for seed in (42,43):
            d=plain(q,seed) if arm=="G22" else np.load(OUT/f"{arm}-{seed}/samples.npz")
            vals.append(diagnostic(d["progress"],d["duration"],q["length"][:,None]).mean(1))
        modes[arm]=np.mean(vals,axis=0)
    results={}
    for arm,v in modes.items():
        avg=np.mean([v[q["labels"]==c].mean(0) for c in range(8)],axis=0)
        results[arm]=dict(boundary_mps=float(avg[0]),interior_mps=float(avg[1]),ratio=float(avg[0]/avg[1]))
    (OUT/"boundaries.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":main()
