"""Select a whole-trip scorer on monitor episodes and score held-out full-trip candidates."""
import json
from pathlib import Path
import numpy as np
import torch
from g21_train import Temporal,scores
from g22_diffusion import ROOT,load,decode
from g22_eval import classifier,score_batch,duration_stats
from g22_temporal import blocks,trip_logits


def align(filename,arm,ids):
    z=np.load(ROOT/f"cnn-{arm}"/filename)
    lookup={v:i for i,v in enumerate(ids)}
    p=np.full((len(ids),8),np.nan)
    p[[lookup[v] for v in z["ids"]]]=z["prob"]
    return p


def combine(original,motion,w):
    p=motion.copy();valid=np.isfinite(original).all(1)
    p[valid]=w*original[valid]+(1-w)*p[valid]
    return p


@torch.no_grad()
def run():
    m=load("monitor")
    mon={k:align("monitor.npz",k,m["ids"]) for k in ("full","motion","native-full")}
    branch=max(("motion","native-full"),key=lambda k:scores(m["labels"],mon[k])["macro_f1"])
    weight=float(max(np.linspace(0,1,11),key=lambda w:scores(m["labels"],combine(mon["full"],mon[branch],w))["macro_f1"]))
    ev=load("evaluation")
    ps={k:align("evaluation.npz",k,ev["ids"]) for k in mon}
    ps["fusion"]=combine(ps["full"],ps[branch],weight)
    results={}
    for arm,p in ps.items():
        ok=np.isfinite(p).all(1)
        results[arm]=dict(eligible=int(ok.sum()),**scores(ev["labels"][ok],p[ok]))
    torch.set_num_threads(2)
    nets=[classifier("full"),classifier("motion")]
    native=Temporal().cuda().eval()
    native.load_state_dict(torch.load(ROOT/"cnn-native-full/model.pt",weights_only=True))
    ranking={}
    for seed in (42,43):
        folder=ROOT/f"model-{seed}"
        d=np.load(folder/"samples.npz")
        prior=np.load(folder/"selection.npz")
        T=d["duration"];P=d["progress"];picks=[]
        for i in range(0,len(T),8):
            seq=[]
            for j in range(i,min(i+8,len(T))):
                for k in range(4):
                    _,xy=decode(ev["paths"][j],T[j,k],P[j,k],.25)
                    seq.append(blocks(xy))
            pt=trip_logits(native,seq).softmax(1).cpu().numpy()
            xs=[];tt=[]
            for j in range(i,min(i+8,len(T))):
                for k in range(4):
                    t,xy=decode(ev["paths"][j],T[j,k],P[j,k],1.)
                    xs.append(xy.astype(np.float32));tt.append(t)
            pg,pm,ok=score_batch(nets,xs,tt)
            pg[~ok]=np.nan
            pf=combine(pg,pt if branch=="native-full" else pm,weight)
            for local,j in enumerate(range(i,min(i+8,len(T)))):
                sl=slice(local*4,(local+1)*4);c=ev["labels"][j]
                picks.append((int(pt[sl,c].argmax()),int(pf[sl,c].argmax())))
        picks=np.array(picks)
        np.savez(folder/"whole-picks.npz",ids=d["ids"],uniform=prior["uniform"],gaf=prior["gaf"],
                 temporal=picks[:,0],fused=picks[:,1])
        ranking[seed]={}
        for key,pick in [("uniform",prior["uniform"]),("gaf",prior["gaf"]),("temporal",picks[:,0]),("fused",picks[:,1])]:
            ranking[seed][key]=duration_stats(ev["duration"],T[np.arange(len(T)),pick,None],ev["labels"])["macro_w1_s"]
    (ROOT/"whole-scorer.json").write_text(json.dumps(dict(motion_branch=branch,gaf_weight=weight,classification=results,
                                                         candidate_duration_w1=ranking),indent=2))


if __name__=="__main__":run()
