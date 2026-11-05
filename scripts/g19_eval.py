"""Select without futures, then evaluate the same pools against physical motion."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from g12_fit import features
from g16_data import NAMES
from g18_motion import measure,describe
from g19_gaf import images,model
from g19_field import ROOT
from g19_train import OLD


@torch.no_grad()
def select(net,xy,ids,labels,seed):
    scores=[]
    for block in np.array_split(xy.reshape(-1,512,2),int(np.ceil(len(xy)*4/32))):
        rgb,ok=images(block)
        prob=net(features(torch.from_numpy(rgb).cuda())).softmax(1).cpu().numpy()
        prob[~ok]=np.nan
        scores.extend(prob)
    scores=np.array(scores).reshape(len(xy),4,8)
    chosen=[]
    for i,(ident,label) in enumerate(zip(ids,labels)):
        key=int.from_bytes(hashlib.sha256(f"{ident}:{seed}:g19".encode()).digest()[:4],"little")
        perm=np.random.default_rng(key).permutation(4)
        valid=perm[np.isfinite(scores[i,perm,label])]
        chosen.append([int(valid[0]),int(valid[np.argmax(scores[i,valid,label])])] if len(valid) else [int(perm[0])]*2)
    return np.array(chosen),int(np.isnan(scores[:,:,0]).sum())


def measures(x):
    v,bend=measure(x)
    cruise=describe(v,.5)[0]
    travel=np.linalg.norm(np.diff(x.astype(float),axis=0),axis=-1).sum()
    return [np.nan if bend is None else bend,cruise,travel]


def paired(ref,first,second,labs,ep):
    """Macro absolute discrepancy of class means, on common support in both arms."""
    rng=np.random.default_rng(19067)
    n=len(np.unique(ep))
    weights=np.vstack([np.ones(n),rng.multinomial(n,np.ones(n)/n,2000)])
    estimates=[]
    support={}
    for c in np.unique(labs):
        keep=(labs==c)&np.isfinite(first)&np.isfinite(second)&np.isfinite(ref)
        support[NAMES[c]]=int(keep.sum())
        if not keep.any():
            continue
        totals=np.bincount(ep[keep],minlength=n)
        den=weights@totals
        errors=[]
        for trial in (first,second):
            s=np.bincount(ep[keep],weights=(trial-ref)[keep],minlength=n)
            errors.append(abs(weights@s)/np.where(den,den,np.nan))
        estimates.append(errors)
    # Drop unsupported bootstrap resamples rather than dropping a class within a draw.
    avg=np.mean(estimates,axis=0)
    delta=avg[1]-avg[0]
    valid=delta[1:][np.isfinite(delta[1:])]
    return dict(first=float(avg[0,0]),second=float(avg[1,0]),delta=float(delta[0]),
                ci=np.quantile(valid,[.025,.975]).tolist(),supported_draws=len(valid),requests=support)


def supported_mean(values,axis):
    count=np.isfinite(values).sum(axis=axis)
    return np.nansum(values,axis=axis)/np.where(count,count,np.nan)


def domain_eval(domain,net):
    prefix="g14-" if domain=="g14" else ""
    q=np.load(OLD/(prefix+"requests.npz"))
    labs=q["labels"]
    ep=np.unique(q["episodes"],return_inverse=True)[1]
    outs={}
    overview={}
    for arm in ("plain","road"):
        groups=[]
        for seed in (42,43):
            pool=np.load(ROOT/f"{arm}-{seed}"/(domain+".npz"))
            assert np.array_equal(q["ids"],pool["ids"])
            sel,failed=select(net,pool["xy"],pool["ids"],labs,seed)
            x=pool["xy"]
            rows=np.array([[measures(candidate) for candidate in row] for row in x])
            ix=np.arange(len(labs))
            groups.append(dict(all= supported_mean(rows,1),uniform=rows[ix,sel[:,0]],gaf=rows[ix,sel[:,1]],
                               travel=rows[:,:,2],failed=failed,caps=int(pool["caps"].sum()),
                               changed=int((sel[:,0]!=sel[:,1]).sum())))
            np.savez_compressed(ROOT/f"{arm}-{seed}"/(domain+"-picks.npz"),ids=q["ids"],uniform=sel[:,0],gaf=sel[:,1])
        outs[arm]={k:supported_mean(np.stack([g[k] for g in groups]),0) for k in ("all","gaf","uniform")}
        overview[arm]={k:sum(g[k] for g in groups) for k in ("failed","caps","changed")}
        overview[arm]["candidates"]=len(labs)*8
        overview[arm]["travel"]=np.stack([g["travel"] for g in groups])
    ref_file=np.load(OLD/(prefix+"references.npz"))
    assert np.array_equal(ref_file["ids"],q["ids"])
    ref=np.array([measures(x) for x in ref_file["xy"]])
    for arm in overview:
        travel=overview[arm].pop("travel")
        overview[arm]["macro_travel_w1"]=float(np.mean([wasserstein_distance(t[labs==c].ravel(),ref[labs==c,2]) for t in travel for c in np.unique(labs)]))
    metrics={}
    for j,name in enumerate(("bend_gap_mps","longest_steady_s")):
        cmp=lambda x,y:paired(ref[:,j],x[:,j],y[:,j],labs,ep)
        metrics[name]=dict(road_vs_plain=cmp(outs["plain"]["all"],outs["road"]["all"]),
            gaf_vs_uniform={a:cmp(outs[a]["uniform"],outs[a]["gaf"]) for a in outs})
    byclass={}
    for c in np.unique(labs):
        keep=(labs==c)&np.isfinite(ref[:,0])&np.isfinite(outs["plain"]["all"][:,0])&np.isfinite(outs["road"]["all"][:,0])
        if keep.any():
            byclass[NAMES[c]]=[float(v[keep,0].mean()) for v in (ref,outs["plain"]["all"],outs["road"]["all"])]
    return dict(metrics=metrics,pools=overview,class_bend_ref_plain_road=byclass)


def graph_check(args):
    import shapely
    from cnr.topology import LaneConsistency
    arm,seed,domain=args
    prefix="g14-" if domain=="g14" else ""
    q=np.load(OLD/(prefix+"requests.npz"))
    d=np.load(ROOT/f"{arm}-{seed}"/(domain+".npz"))
    graph=LaneConsistency("data/maps/most.net.xml",radius=5.,max_candidates=64)
    byclass={}
    for xs,c,lane in zip(d["xy"],q["labels"],q["lane"]):
        row=byclass.setdefault(NAMES[c],[0,0])
        for x in xs:
            if c==0:
                valid=np.max(shapely.distance(shapely.points(x),graph.road.shapes[lane]))<=2.01
            else:
                r=graph.match(x,NAMES[c])
                valid=r["graph_consistent"] and not r["candidate_truncation_warning"]
            row[0]+=int(valid)
            row[1]+=1
    return "-".join(map(str,args)),byclass


if __name__=="__main__":
    import sys
    if sys.argv[1]=="graph":
        tasks=[(arm,s,d) for arm in ("plain","road") for s in (42,43) for d in ("g14","g15")]
        with ProcessPoolExecutor(max_workers=8) as pool:
            result=dict(pool.map(graph_check,tasks))
    else:
        torch.set_num_threads(2)
        net=model().eval()
        net.load_state_dict(torch.load(ROOT/"gaf8.pt",weights_only=True,map_location="cuda"))
        result={d:domain_eval(d,net) for d in ("g14","g15")}
    (ROOT/(sys.argv[1]+".json")).write_text(json.dumps(result,indent=2,allow_nan=False))
