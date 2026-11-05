"""Lock spatial grid, motion branch and probability fusion on internal episodes only."""
import json
from pathlib import Path
import numpy as np
import torch
from g19_gaf import images,original_metadata
from g21_train import ROOT,build,features,angular,scores

CONFIGS={"hilbert":("g21-data","original"),"fine":("g21-fine","original"),
         "physical":("g21-data","physical"),"temporal":("g21-data","temporal")}


def ensemble(key,split,ids):
    root,arm=CONFIGS[key]
    byid={v:i for i,v in enumerate(ids)}
    out=np.full((len(ids),8),np.nan)
    vals=[]
    for seed in (42,43):
        z=np.load(Path(root)/f"{arm}-{seed}"/(split+".npz"))
        ix=np.array([byid[x] for x in z["ids"]])
        vals.append(z["prob"])
    out[ix]=np.mean(vals,axis=0)
    return out


def summary(y,p):
    ok=np.isfinite(p).all(1)
    ans=scores(y[ok],p[ok])
    ans.update(eligible=int(ok.sum()),total=len(y),accuracy_all=float((p[ok].argmax(1)==y[ok]).sum()/len(y)))
    return ans


def combine(img,mot,weight):
    out=mot.copy()
    ok=np.isfinite(img).all(1)
    out[ok]=weight*img[ok]+(1-weight)*mot[ok]
    return out


def external(key,xy):
    root,arm=CONFIGS[key]
    meta=original_metadata()
    if key=="fine":meta=dict(meta,hilbert_eps=meta["hilbert_eps"]/16)
    prob=[]
    for seed in (42,43):
        net=build(arm,False).cuda().eval()
        net.load_state_dict(torch.load(Path(root)/f"{arm}-{seed}"/"model.pt",weights_only=True))
        rows=[]
        for pos in range(0,len(xy),16):
            x=xy[pos:pos+16]
            if arm=="original":
                rgb,ok=images(x,meta)
                inputs=features(torch.from_numpy(rgb).cuda())
            else:
                ok=np.ones(len(x),bool)
                v=torch.from_numpy((4*np.linalg.norm(np.diff(x.astype(float),axis=1),axis=-1)).astype("float32")).cuda()
                inputs=angular(v) if arm=="physical" else v
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                data=net(inputs).float().softmax(1).cpu().numpy()
            data[~ok]=np.nan
            rows.extend(data)
        prob.append(rows)
    return np.mean(prob,axis=0)


def go():
    idx=np.load(ROOT/"train/index.npz")
    ep=np.array([int(s.split("_")[0]) for s in idx["ids"]])
    select=np.isin(ep,np.unique(ep)[-8:])
    ids,y=idx["ids"][select],idx["labels"][select]
    mon={k:ensemble(k,"monitor",ids) for k in CONFIGS}
    spatial=max(("hilbert","fine"),key=lambda k:summary(y,mon[k])["macro_f1"])
    motion=max(("physical","temporal"),key=lambda k:summary(y,mon[k])["macro_f1"])
    w=max(np.linspace(0,1,11),key=lambda v:summary(y,combine(mon[spatial],mon[motion],v))["macro_f1"])
    selected=dict(spatial=spatial,motion=motion,weight=float(w))
    mon["fusion"]=combine(mon[spatial],mon[motion],w)
    res=dict(selection=selected,monitor={k:summary(y,p) for k,p in mon.items()})
    idx=np.load(ROOT/"validation/index.npz");ids,y=idx["ids"],idx["labels"]
    val={k:ensemble(k,"validation",ids) for k in CONFIGS}
    val["fusion"]=combine(val[spatial],val[motion],w)
    base=Path("../g17-job/g17-data")
    current=np.load(base/"requests.npz")
    used=np.isin(ids,current["ids"])
    assert used.sum()==len(current["ids"])
    res["validation"]={k:summary(y,p) for k,p in val.items()}
    res["same_175"]={k:summary(y[used],p[used]) for k,p in val.items()}
    # Eight possible outputs even though G14 ground truth contains five classes.
    r=np.load(base/"g14-references.npz");q=np.load(base/"g14-requests.npz")
    assert np.array_equal(q["ids"],r["ids"])
    ext={k:external(k,r["xy"]) for k in CONFIGS}
    ext["fusion"]=combine(ext[spatial],ext[motion],w)
    res["g14"]={k:summary(q["labels"],p) for k,p in ext.items()}
    for k in res["g14"]:
        cm=np.array(res["g14"][k]["cm"])
        res["g14"][k]["macro_f1_supported"]=float(np.mean((2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1))[1:6]))
    (ROOT/"results.json").write_text(json.dumps(res,indent=2))


if __name__=="__main__":
    torch.set_num_threads(2)
    go()
