"""Audit classifier heads versus held-out latent structure. No new generator fits."""
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE=Path("g25-data")
MOTOR=[1,2,3,5,6,7]


def get(name):
    with np.load(BASE/(name+".npz")) as z:
        keep=z["valid"]
        return {k:z[k][keep] for k in z.files}


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-10)


def cka(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    return float(np.linalg.norm(a.T@b)**2/(np.linalg.norm(a.T@a)*np.linalg.norm(b.T@b)))


def covariates(d):
    return np.c_[np.eye(8)[d["label"]],np.log(d["length"]),np.log(d["clock"]),d["phys"][:,0],d["ends"]]


def aligned(a,b,lab,ep,seed):
    # Null retains class frequencies, episode, and both marginal feature distributions.
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero((lab==c)&(ep==e)) for c in np.unique(lab) for e in np.unique(ep)]
    null=[]
    for _ in range(199):
        perm=np.arange(len(a))
        for g in groups:perm[g]=rng.permutation(g)
        null.append(cka(a,b[perm]))
    observed=cka(a,b)
    return dict(cka=observed,null_median=float(np.median(null)),null_q95=float(np.quantile(null,.95)),
                permutation_p=(1+sum(n>=observed for n in null))/200)


def compare_representations(tr,d):
    y=d["label"]
    ep=np.array([int(k.split("_")[0]) for k in d["ids"]])
    projections={}
    for key in ("Z","S"):
        pc=PCA(n_components=32,svd_solver="full").fit(tr[key])
        train=pc.transform(tr[key]).astype(float)
        score=pc.transform(d[key]).astype(float)
        means=np.stack([train[tr["label"]==c].mean(0) for c in range(8)])
        scale=StandardScaler().fit(covariates(tr))
        model=Ridge(alpha=1.).fit(scale.transform(covariates(tr)),train)
        projections[key]=dict(raw=score,within=score-means[y],adjusted=score-model.predict(scale.transform(covariates(d))),
                              variance=float(pc.explained_variance_ratio_.sum()))
    result={"pca_variance":{k:v["variance"] for k,v in projections.items()}}
    for name,mask in [("all",np.ones(len(y),bool)),("motor",np.isin(y,MOTOR))]:
        result[name]={}
        for mode in ("raw","within","adjusted"):
            result[name][mode]=aligned(projections["Z"][mode][mask],projections["S"][mode][mask],y[mask],ep[mask],2525)
        labs=np.unique(y[mask])
        a,b=[np.stack([projections[k]["raw"][y==c].mean(0) for c in labs]) for k in ("Z","S")]
        rng=np.random.default_rng(25032)
        null=np.array([cka(a,b[rng.permutation(len(b))]) for _ in range(999)])
        result[name]["centroids"]=dict(cka=cka(a,b),perm_q95=float(np.quantile(null,.95)),p=(1+int(sum(null>=cka(a,b))))/1000)
    return result


def classify(y,p):
    pred=p.argmax(1)
    return dict(accuracy=float(np.mean(pred==y)),macro_f1=float(f1_score(y,pred,average="macro")),
                motor_top1=float(np.mean(pred[np.isin(y,MOTOR)]==y[np.isin(y,MOTOR)])))


def vectors(tr):
    center=tr["Z"].mean(0)
    z=unit(tr["Z"]-center)
    classes=unit(np.stack([z[tr["label"]==c].mean(0) for c in range(8)]))
    directions={}
    for c in range(8):
        for m in range(6):
            pos=(tr["label"]==c)&(tr["mode"]==m)
            neg=(tr["label"]==c)&(tr["mode"]!=m)
            if min(pos.sum(),neg.sum())>=20:
                directions[c,m]=unit(z[pos].mean(0)-z[neg].mean(0))
    return center,classes,directions


def validate_modes(d,center,dirs):
    z=unit(d["Z"]-center)
    rows={}
    for (c,m),axis in dirs.items():
        use=d["label"]==c
        positive=d["mode"][use]==m
        if min(positive.sum(),sum(~positive))>=5:
            rows[f"{c}-{m}"]=float(roc_auc_score(positive,z[use]@axis))
    return dict(n=len(rows),mean_auc=float(np.mean(list(rows.values()))),aucs=rows)


def scores_for(d,scaler,probe,center,classes,dirs):
    y=d["label"];idx=np.arange(len(y));z=unit(d["Z"]-center)
    sim=z@classes.T
    other=sim.copy();other[idx,y]=-np.inf
    motif=np.full(len(y),np.nan)
    for (c,m),axis in dirs.items():
        mask=(y==c)&(d["target"]==m)
        motif[mask]=z[mask]@axis
    return np.c_[d["class_p"][idx,y],probe.predict_proba(scaler.transform(d["Z"]))[idx,y],
                 sim[idx,y]-np.max(other,axis=1),motif]


def macro_boot(x,lab,episode,counts):
    rows=[];support={}
    for c in np.unique(lab):
        keep=(lab==c)&np.isfinite(x)
        support[str(c)]=int(sum(keep))
        if not keep.any():continue
        n=np.bincount(episode[keep],minlength=counts.shape[1])
        s=np.bincount(episode[keep],weights=x[keep],minlength=counts.shape[1])
        denom=counts@n
        rows.append((counts@s)/np.where(denom,denom,np.nan))
    return np.mean(rows,axis=0),support


def main():
    tr,mon,ext,ref=[get(k) for k in ("train","monitor","g23","g24")]
    scaler=StandardScaler().fit(tr["Z"])
    best=-1;chosen=None
    for C in [.01,.1,1.]:
        model=LogisticRegression(C=C,class_weight="balanced",max_iter=1200,solver="lbfgs")
        model.fit(scaler.transform(tr["Z"]),tr["label"])
        metric=classify(mon["label"],model.predict_proba(scaler.transform(mon["Z"])))["macro_f1"]
        if metric>best:probe=model;best=metric;chosen=C
    center,classes,dirs=vectors(tr)
    report=dict(chosen_C=chosen,monitor_macro_f1=best,mode_axes=len(dirs),reference={})
    for name,d in [("g23",ext),("g24",ref)]:
        report["reference"][name]=dict(n=len(d["ids"]),head=classify(d["label"],d["class_p"]),
            probe=classify(d["label"],probe.predict_proba(scaler.transform(d["Z"]))),
            alignment=compare_representations(tr,d),within_mode_auc=validate_modes(d,center,dirs))
    values={}
    for arm in ("plain","guided","permuted"):
        curves=[]
        for seed in (42,43):
            d=get(f"{arm}-{seed}");b=get(f"plain-{seed}")
            assert np.array_equal(d["ids"],b["ids"]) and np.array_equal(d["slot"],b["slot"]) and np.array_equal(d["target"],b["target"])
            s=scores_for(d,scaler,probe,center,classes,dirs)
            ep=np.unique([i.split("_")[0] for i in d["ids"]],return_inverse=True)[1]
            n=ep.max()+1;counts=np.vstack([np.ones(n),np.random.default_rng(2419).multinomial(n,np.ones(n)/n,2000)])
            rows=[]
            for j in range(4):
                v,support=macro_boot(s[:,j],d["label"],ep,counts);rows.append(v)
                if j==3:report.setdefault("target_mode_support_by_seed",{})[seed]=support
            curves.append(rows)
        values[arm]=np.mean(curves,axis=0)
    report["generation"]={}
    names=["frozen_head","spatial_probe_probability","class_margin_cosine","within_class_target_mode_direction"]
    for a,b in [("guided","plain"),("permuted","plain"),("guided","permuted")]:
        report["generation"][a+"-"+b]={k:dict(delta=float(v[0]),ci=np.nanquantile(v[1:],[.025,.975]).tolist(),
              supported_draws=int(sum(np.isfinite(v[1:])))) for k,v in zip(names,values[a]-values[b])}
    previous=json.loads(Path("../g24-job/g24-data/results.json").read_text())["differences"]["guided-plain"]["scores"][0]["estimate"]
    assert abs(report["generation"]["guided-plain"]["frozen_head"]["delta"]-previous)<1e-4
    (BASE/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":main()
