"""Independent evaluation of class-conditional latent distribution fidelity."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist,pdist
from sklearn.decomposition import PCA
from g25_audit import BASE,get


def gram(x,y,bandwidth):
    return np.exp(-cdist(x,y,"sqeuclidean")/(2*bandwidth**2))


def run():
    tr,ref=get("train"),get("g24")
    labels=ref["label"]
    ep=np.unique([i.split("_")[0] for i in ref["ids"]],return_inverse=True)[1]
    cnt=np.vstack([np.ones(8),np.random.default_rng(2419).multinomial(8,np.ones(8)/8,2000)])
    results={}
    for key in ("Z","S"):
        pc=PCA(n_components=32,svd_solver="full").fit(tr[key])
        train=pc.transform(tr[key]).astype(float);real=pc.transform(ref[key]).astype(float)
        rng=np.random.default_rng(25504)
        bandwidth=[]
        for c in range(8):
            members=np.flatnonzero(tr["label"]==c)
            subset=rng.choice(members,min(256,len(members)),replace=False)
            sigma=np.median(pdist(train[subset]))
            if sigma<=0:raise ValueError("Degenerate kernel bandwidth")
            bandwidth.append(float(sigma))
        curves={}
        class_curves={}
        for arm in ("plain","guided","permuted"):
            curve=np.zeros((8,len(cnt)))
            for seed in (42,43):
                d=get(f"{arm}-{seed}")
                assert np.array_equal(d["ids"],np.repeat(ref["ids"],4))
                x=pc.transform(d[key]).astype(float).reshape(len(labels),4,-1)
                for c in range(8):
                    take=labels==c;n=take.sum();r=real[take];y=x[take].reshape(n*4,-1);s=bandwidth[c]
                    yy=gram(y,y,s).reshape(n,4,n,4).mean((1,3))
                    yr=gram(y,r,s).reshape(n,4,n).mean(1)
                    contrasts=yy+gram(r,r,s)-yr-yr.T
                    weights=cnt[:,ep[take]].copy();weights/=weights.sum(1,keepdims=True)
                    curve[c]+=np.einsum("ij,ij->i",weights@contrasts,weights)/2
            class_curves[arm]=curve
            curves[arm]=curve.mean(0)
        result=dict(bandwidth=bandwidth,estimates={a:float(v[0]) for a,v in curves.items()})
        for a,b in [("guided","plain"),("guided","permuted"),("permuted","plain")]:
            delta=curves[a]-curves[b]
            result[a+"-"+b]=dict(delta=float(delta[0]),ci=np.quantile(delta[1:],[.025,.975]).tolist())
        delta=class_curves["guided"]-class_curves["plain"]
        result["class_differences"]=delta[:,0].tolist()
        motor=delta[[1,2,3,5,6,7]].mean(0)
        result["six_motor_delta"]=dict(delta=float(motor[0]),ci=np.quantile(motor[1:],[.025,.975]).tolist())
        results[key]=result
    (BASE/"distribution.json").write_text(json.dumps(results,indent=2))


if __name__=="__main__":run()
