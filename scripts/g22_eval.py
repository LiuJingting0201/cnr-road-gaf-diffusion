"""Evaluate real durations, full-trip physics, and fixed-pool whole-trip classification."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torchvision.models import resnet18
from g12_fit import features
from g16_data import NAMES
from g21_train import scores
from g22_diffusion import ROOT,load,decode,positions,K
from g22_images import dynamics,fields,images


def empirical(tr,ev,seed):
    rng=np.random.default_rng(22000+seed)
    times=[];profiles=[]
    for lab,L in zip(ev["labels"],ev["length"]):
        ix=np.flatnonzero(tr["labels"]==lab)
        closest=ix[np.argsort(abs(np.log(tr["length"][ix]/L)))[:32]]
        draw=rng.choice(closest,K)
        times.append(tr["duration"][draw]*L/tr["length"][draw])
        profiles.append(tr["progress"][draw])
    return np.array(times),np.array(profiles)


def duration_stats(real,times,labels):
    byclass={}
    for c in np.unique(labels):
        mask=labels==c
        byclass[NAMES[c]]=dict(requests=int(mask.sum()),observed_median_s=float(np.median(real[mask])),
            generated_median_s=float(np.median(times[mask])),duration_w1_s=float(wasserstein_distance(real[mask],times[mask].ravel())))
    return dict(classes=byclass,macro_w1_s=float(np.mean([r["duration_w1_s"] for r in byclass.values()])))


def signals(t,xy):
    # Uniform seconds over the full trip. The final fractional second is excluded from speed summaries.
    sec=np.arange(0,np.floor(t[-1])+1)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy.astype(float),axis=0),axis=1))]
    v=np.diff(np.interp(sec,t,arc))
    return [float(v.mean()),float(abs(np.diff(v)).mean()),float(np.mean(v<.3))]


def reference_signals(split):
    rows=[]
    for file in sorted((ROOT/split).glob("*/trips.npz")):
        with np.load(file) as d:
            xy,paths,off,T=d["xy"],d["paths"],d["offset"],d["duration"]
            for i,t in enumerate(T):
                x=np.vstack([paths[i,0],xy[off[i]:off[i+1]],paths[i,-1]])
                rows.append(signals(np.linspace(0,t,len(x)),x))
    return np.array(rows)


def classifier(arm):
    m=resnet18(weights=None);m.fc=torch.nn.Linear(512,8)
    m.load_state_dict(torch.load(ROOT/f"cnn-{arm}"/"model.pt",weights_only=True,map_location="cpu"))
    return m.cuda().eval()


def classify_report():
    reports={a:json.loads((ROOT/f"cnn-{a}"/"complete.json").read_text()) for a in ("full","motion","full-paired","short-paired")}
    mons={a:np.load(ROOT/f"cnn-{a}/monitor.npz") for a in ("full","motion")}
    cur=mons["full"]
    idx={s:i for i,s in enumerate(mons["motion"]["ids"])}
    ix=[idx[s] for s in cur["ids"]]
    pm=mons["motion"]["prob"][ix]
    w=float(max(np.linspace(0,1,11),key=lambda x:scores(cur["labels"],x*cur["prob"]+(1-x)*pm)["macro_f1"]))
    evs={a:np.load(ROOT/f"cnn-{a}/evaluation.npz") for a in ("full","motion")}
    pm=evs["motion"]["prob"].copy()
    idx={s:i for i,s in enumerate(evs["motion"]["ids"])}
    ix=[idx[s] for s in evs["full"]["ids"]]
    pm[ix]=w*evs["full"]["prob"]+(1-w)*pm[ix]
    reports["fusion"]=dict(weight_full_gaf=w,results=scores(evs["motion"]["labels"],pm))
    return reports,w


@torch.no_grad()
def score_batch(nets,paths,times_list):
    rgb,ok=images(paths)
    vel=np.stack([dynamics(t,x) for t,x in zip(times_list,paths)])
    p1=nets[0](features(torch.from_numpy(rgb).cuda())).softmax(1).cpu().numpy()
    p2=nets[1](fields(torch.from_numpy(vel).cuda())).softmax(1).cpu().numpy()
    return p1,p2,ok


def go():
    torch.set_num_threads(2)
    report,w=classify_report()
    nets=[classifier("full"),classifier("motion")]
    ev=load("evaluation");tr=load("train")
    ref=reference_signals("evaluation")
    rng=np.random.default_rng(22435)
    counts=rng.multinomial(4,np.ones(4)/4,2000)
    eps=np.unique([int(s.split("_")[0]) for s in ev["ids"]],return_inverse=True)[1]
    result=dict(classifiers=report,arms={})
    durations={}
    for arm in ("empirical","diffusion"):
        arm_scores=[]
        for seed in (42,43):
            if arm=="empirical":
                T,P=empirical(tr,ev,seed);caps=flat=0
            else:
                d=np.load(ROOT/f"model-{seed}/samples.npz")
                assert np.array_equal(d["ids"],ev["ids"])
                T,P=d["duration"],d["progress"]
                caps=int(d["bounded"].sum());flat=int(d["flat"].sum())
            durations[arm,seed]=T
            sel=np.zeros((len(T),3),int)
            phys=np.empty((len(T),K,3));max_end=0.;bad=0
            for i in range(0,len(T),8):
                xs=[];ts=[]
                for j in range(i,min(i+8,len(T))):
                    for k in range(K):
                        t,x=decode(ev["paths"][j],T[j,k],P[j,k],1.)
                        max_end=max(max_end,float(np.linalg.norm(x[-1]-ev["paths"][j,-1])))
                        phys[j,k]=signals(t,x)
                        ts.append(t);xs.append(x.astype("float32"))
                if arm=="diffusion":
                    p1,p2,ok=score_batch(nets,xs,ts)
                    bad+=int((~ok).sum())
                    for z,j in enumerate(range(i,min(i+8,len(T)))):
                        sl=slice(z*K,(z+1)*K)
                        perm=np.random.default_rng(22101+j+seed).permutation(K)
                        valid=perm[ok[sl][perm]]
                        uniform=perm[0]
                        c=ev["labels"][j]
                        pick=valid[np.argmax(p1[sl,c][valid])] if len(valid) else uniform
                        fusion=p2[sl,c].copy()
                        fusion[valid]=w*p1[sl,c][valid]+(1-w)*p2[sl,c][valid]
                        sel[j]=[uniform,pick,perm[np.argmax(fusion[perm])]]
            stats=duration_stats(ev["duration"],T,ev["labels"])
            stats.update(bound_count=caps,flat=flat,max_endpoint_m=max_end)
            stats["motion_w1"]=[float(np.mean([wasserstein_distance(ref[ev["labels"]==c,l],phys[ev["labels"]==c,:,l].ravel()) for c in range(8)])) for l in range(3)]
            if arm=="diffusion":
                stats["encoding_failures"]=bad
                stats["selected_duration_w1"]={}
                for k,name in enumerate(("uniform","full_gaf","fusion")):
                    chosen=T[np.arange(len(T)),sel[:,k]]
                    stats["selected_duration_w1"][name]=duration_stats(ev["duration"],chosen[:,None],ev["labels"])["macro_w1_s"]
                np.savez(ROOT/f"model-{seed}/selection.npz",ids=ev["ids"],uniform=sel[:,0],gaf=sel[:,1],fusion=sel[:,2])
            arm_scores.append(stats)
        result["arms"][arm]=arm_scores
    # Primary duration uncertainty uses class-macro W1, bootstrapping whole episodes.
    from cnr.episode_w1 import episode_w1
    cnt=np.vstack([np.ones(4),counts])
    curve={}
    for arm in ("empirical","diffusion"):
        curve[arm]=sum(episode_w1(durations[arm,s][ev["labels"]==c],ev["duration"][ev["labels"]==c,None],
                        eps[ev["labels"]==c],False,cnt) for s in (42,43) for c in range(8))/16
    delta=curve["diffusion"]-curve["empirical"]
    result["duration_delta_vs_empirical"]={"mean":float(delta[0]),"ci":np.quantile(delta[1:],[.025,.975]).tolist()}
    (ROOT/"results.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":go()
