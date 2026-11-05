"""Compare the old block model and continuous model on identical new requests."""
import json
from pathlib import Path
import numpy as np
import torch
from cnr.episode_w1 import episode_w1
from g22_diffusion import decode
from g23_eval import phys
from g26_train import OUT,Denoiser,sample


def run():
    torch.set_num_threads(2)
    with np.load(OUT/"requests.npz") as d:req={k:d[k] for k in d.files}
    r=np.load(OUT/"references.npz");assert np.array_equal(r["ids"],req["ids"])
    off=r["offset"];arr=r["txy"]
    ref=np.array([phys(arr[a:b,0],arr[a:b,1:]) for a,b in zip(off,off[1:])])
    ref[:,1]=r["duration"]
    ep=np.unique([i.split("_")[0] for i in req["ids"]],return_inverse=True)[1]
    cnt=np.vstack([np.ones(8),np.random.default_rng(27106).multinomial(8,np.ones(8)/8,2000)])
    scores={}
    for arm in ("H","HG"):
        vals={}
        for layout in ("flat","blocks"):
            curves=[]
            for seed in (42,43):
                folder=OUT/f"{arm}-{seed}"
                if layout=="flat":
                    pool=np.load(folder/"samples.npz")
                else:
                    other=Path("../g26-job/g26-data")/f"{arm}-{seed}"
                    model=Denoiser().cuda().eval()
                    model.load_state_dict(torch.load(other/"model.pt",map_location="cuda",weights_only=True))
                    pool=sample(model,arm,req,json.loads((other/"stats.json").read_text())["stats"],seed)
                X=np.array([[phys(*decode(path,t,p)) for t,p in zip(times,profiles)]
                            for path,times,profiles in zip(req["paths"],pool["duration"],pool["progress"])])
                estimate=np.zeros((len(cnt),3))
                for c in range(8):
                    ids=req["labels"]==c
                    for j in range(3):
                        estimate[:,j]+=episode_w1(X[ids,:,j],ref[ids,j,None],ep[ids],False,cnt)/8
                curves.append(estimate)
            vals[layout]=np.mean(curves,axis=0)
        delta=vals["flat"]-vals["blocks"]
        scores[arm]=dict(flat=vals["flat"][0].tolist(),blocks=vals["blocks"][0].tolist(),
                         delta=delta[0].tolist(),ci=np.quantile(delta[1:],[.025,.975],axis=0).T.tolist())
    (OUT/"layouts.json").write_text(json.dumps(scores,indent=2))


if __name__=="__main__":run()
