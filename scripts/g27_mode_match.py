"""Summarize requested-mode realization without using a classifier's score."""
import json
from pathlib import Path
import numpy as np
from g22_diffusion import decode
from g23_eval import phys
from g23_modes import mode
from g26_train import ARMS,OUT


if __name__=="__main__":
    with np.load(OUT/"requests.npz") as q:
        labs=q["labels"];paths=q["paths"];ids=q["ids"]
    ep=np.unique([s.split("_")[0] for s in ids],return_inverse=True)[1]
    cnt=np.vstack([np.ones(8),np.random.default_rng(2731).multinomial(8,np.ones(8)/8,2000)])
    defs=json.loads(Path("../g23-job/g23-data/definition.json").read_text())
    curves={}
    for arm in ARMS:
        score=[]
        for seed in (42,43):
            d=np.load(OUT/f"{arm}-{seed}/samples.npz")
            assert np.array_equal(ids,d["ids"])
            met=np.array([[phys(*decode(p,t,u)) for t,u in zip(T,P)] for p,T,P in zip(paths,d["duration"],d["progress"])])
            hits=(mode(met[:,:,[3,2]],defs)==d["target"][:,None]).mean(1)
            for c in range(8):
                use=labs==c;n=np.bincount(ep[use],minlength=8)
                sums=np.bincount(ep[use],weights=hits[use],minlength=8)
                score.append((cnt@sums)/(cnt@n))
        curves[arm]=np.mean(score,axis=0)
    result={k:float(v[0]) for k,v in curves.items()}
    for a,b in [("HM","H"),("HG","HM"),("HG","HV")]:
        diff=curves[a]-curves[b];result[a+"-"+b]=dict(delta=float(diff[0]),ci=np.quantile(diff[1:],[.025,.975]).tolist())
    (OUT/"mode-match.json").write_text(json.dumps(result,indent=2))
