"""Score complete quarter-second candidates with the deployable native GAF model."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import torch
from g12_fit import features
from g19_gaf import images
from g21_train import scores
from g22_diffusion import ROOT,load,decode
from g22_eval import classifier,duration_stats


def encode_task(args):
    path,times,profiles=args
    return images([decode(path,t,p)[1] for t,p in zip(times,profiles)])


if __name__=="__main__":
    torch.set_num_threads(1)
    ev=load("evaluation");net=classifier("native-gaf")
    stats={}
    for seed in [42,43]:
        folder=ROOT/f"model-{seed}"
        pred=np.load(folder/"samples.npz");old=np.load(folder/"whole-picks.npz")
        assert np.array_equal(old["ids"],pred["ids"])
        T,P=pred["duration"],pred["progress"]
        picks=[];failed=0
        with ProcessPoolExecutor(max_workers=8) as pool:
            for j in range(0,len(T),16):
                jobs=[(ev["paths"][i],T[i],P[i]) for i in range(j,min(j+16,len(T)))]
                tables=list(pool.map(encode_task,jobs))
                im=np.concatenate([v[0] for v in tables])
                with torch.no_grad():prob=net(features(torch.from_numpy(im).cuda())).softmax(1).cpu().numpy()
                for i,(_,ok) in enumerate(tables):
                    idx=j+i;c=ev["labels"][idx]
                    good=np.flatnonzero(ok);failed+=int((~ok).sum())
                    pick=good[np.argmax(prob[i*4+good,c])] if len(good) else old["uniform"][idx]
                    picks.append(int(pick))
        np.savez(folder/"native-picks.npz",ids=pred["ids"],gaf=picks,temporal=old["temporal"],uniform=old["uniform"])
        stats[seed]=dict(enc_failures=failed,pools=len(T),changed_vs_uniform=int(np.sum(picks!=old["uniform"])),
                         duration_w1=duration_stats(ev["duration"],T[np.arange(len(T)),picks,None],ev["labels"])["macro_w1_s"])
    (ROOT/"native-feedback.json").write_text(json.dumps(stats,indent=2))
