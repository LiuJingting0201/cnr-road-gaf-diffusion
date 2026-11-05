"""Post-hoc clock/profile swaps isolate which change the frozen class reader sees."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from g12_fit import features
from g19_gaf import images
from g22_diffusion import decode
from g24_eval import OUT,macro_mean,interval


def encode(task):
    path,T,P=task
    return images([decode(path,t,p)[1] for t,p in zip(T,P)])


@torch.no_grad()
def run():
    torch.set_num_threads(2)
    reader=resnet18(weights=None);reader.fc=torch.nn.Linear(512,8)
    reader.load_state_dict(torch.load("../g22-job/g22-data/cnn-native-gaf/model.pt",weights_only=True,map_location="cpu"))
    reader.cuda().eval()
    q=np.load(OUT/"requests.npz");lab=q["labels"]
    ep=np.unique([i.split("_")[0] for i in q["ids"]],return_inverse=True)[1]
    cnt=np.vstack([np.ones(8),np.random.default_rng(2419).multinomial(8,np.ones(8)/8,2000)])
    means={k:[] for k in ("plain","clock_only","profile_only","both")}
    for seed in [42,43]:
        b=np.load(OUT/f"plain-{seed}.npz");g=np.load(OUT/f"guided-{seed}.npz")
        for k,clock,profile in [("plain",b,b),("clock_only",g,b),("profile_only",b,g),("both",g,g)]:
            jobs=list(zip(q["paths"],clock["duration"],profile["progress"]))
            probs=[]
            with ProcessPoolExecutor(max_workers=8) as pool:
                for start in range(0,len(jobs),16):
                    tables=list(pool.map(encode,jobs[start:start+16]))
                    rgb=np.concatenate([im for im,_ in tables]);assert all(ok.all() for _,ok in tables)
                    pr=reader(features(torch.from_numpy(rgb).cuda())).softmax(1).cpu().numpy().reshape(-1,4,8)
                    probs.extend(pr[j,:,lab[start+j]].mean() for j in range(len(tables)))
            means[k].append(macro_mean(np.asarray(probs),lab,ep,cnt))
    base=np.mean(means["plain"],axis=0)
    result={k:interval(np.mean(v,axis=0)-base) for k,v in means.items()}
    (OUT/"swap.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":run()
