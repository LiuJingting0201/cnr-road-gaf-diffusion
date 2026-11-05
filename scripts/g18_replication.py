"""Same spatial bend audit on the G14 independent development requests."""
import argparse
import json
from pathlib import Path
import numpy as np
from g18_motion import measure, describe

NAMES=["pedestrian","bus","motorcycle","private_car","bycicle","taxi_or_uber","truck","delivery"]


def audit(root,domain):
    prefix="g14-" if domain=="g14" else ""
    req=np.load(root/(prefix+"requests.npz"))
    ref=np.load(root/(prefix+"references.npz"))
    assert np.array_equal(req["ids"],ref["ids"])
    obs=[measure(x) for x in ref["xy"]]
    out={}
    for arm in (("A","B","C") if domain=="g14" else ("C",)):
        pools=[]
        for seed in [42,43]:
            s=np.load(root/f"{arm}-{seed}"/(domain+".npz"))
            assert np.array_equal(req["ids"],s["ids"])
            pools.append(s["xy"])
        x=np.concatenate(pools,axis=1)
        rows={}
        for c in np.unique(req["labels"]):
            pairs=[];episodes=[]
            for i in np.flatnonzero(req["labels"]==c):
                _,reference=obs[i]
                gaps=[measure(t)[1] for t in x[i]]
                available=[g for g in gaps if g is not None]
                if reference is not None and available:
                    pairs.append((reference,np.mean(available),len(available)))
                    episodes.append(req["episodes"][i])
            if not pairs:
                rows[NAMES[c]]={"requests":0}
                continue
            a=np.array(pairs)
            e=np.array(episodes)
            u=np.unique(e)
            draws=np.random.default_rng(18172).multinomial(len(u),np.ones(len(u))/len(u),2000)
            sums=np.array([(a[e==ep,1]-a[e==ep,0]).sum() for ep in u])
            sizes=np.array([(e==ep).sum() for ep in u])
            rows[NAMES[c]]=dict(requests=len(a),episodes=len(u),reference=float(a[:,0].mean()),
                generated=float(a[:,1].mean()),supported_candidates=int(a[:,2].sum()),
                diff_ci=np.quantile((draws@sums)/(draws@sizes),[.025,.975]).tolist())
        out[arm]=rows
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps({k:audit(a.source,k) for k in ("g14","g15")},indent=2))
