"""Evaluate feedback only after selections are written. All controls use the same pools."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.episode_w1 import episode_w1
from g12_fit import features
from g16_data import NAMES
from g19_gaf import images
from g22_diffusion import decode
from g23_fit import classifier,metrics
from g23_modes import OUT,mode,speed_series,speed_images


def phys(t,xy):
    t=t-t[0]
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy.astype(float),axis=0),axis=1))]
    v=np.diff(np.interp(np.arange(int(np.floor(t[-1]))+1),t,arc))
    steady=(v[1:]>=.3)&(abs(np.diff(v))<.5)
    edges=np.diff(np.r_[0,steady.astype(int),0])
    longest=max(np.flatnonzero(edges==-1)-np.flatnonzero(edges==1),default=0)
    return [longest,t[-1],np.mean(abs(v[4:]-v[:-4])),np.mean(v)]


def ref_data():
    ids=[];labels=[];paths=[];observed=[];targets=[];speeds=[];xy=[]
    for f in sorted(Path("g22-data/fresh").glob("*/quarter.npz")):
        with np.load(f) as q,np.load(f.parent/"trips.npz") as d:
            assert np.array_equal(q["ids"],d["ids"])
            off=q["offset"];tx=q["txy"]
            ids.extend(q["ids"]);labels.extend(d["labels"]);paths.extend(d["paths"])
            for j,arrive in enumerate(d["duration"]):
                arr=tx[off[j]:off[j+1]]
                t=arr[:,0].astype(float);x=arr[:,1:]
                obs=phys(t,x);obs[1]=arrive
                observed.append(obs);v,s=speed_series(t,x)
                targets.append(s);speeds.append(v);xy.append(x)
    return np.array(ids),np.array(labels),np.array(paths),np.array(observed),np.array(targets),np.array(speeds),xy


def cm(y,p,n):
    mat=np.bincount(n*y+p,minlength=n*n).reshape(n,n)
    return dict(accuracy=float(np.trace(mat)/len(y)),macro_f1=float(np.mean(2*np.diag(mat)/np.maximum(1,mat.sum(0)+mat.sum(1)))),cm=mat.tolist())


def aux(xy,y,v,z):
    spatial=resnet18(weights=None);spatial.fc=torch.nn.Linear(512,8)
    spatial.load_state_dict(torch.load("../g22-job/g22-data/cnn-native-gaf/model.pt",weights_only=True,map_location="cpu"))
    spatial.cuda().eval();mo=classifier().eval()
    mo.load_state_dict(torch.load(OUT/"speed-mode.pt",weights_only=True))
    sp=[];truth=[];speed=[]
    with torch.no_grad(),ProcessPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(y),16*24):
            batches=[xy[i:i+16] for i in range(offset,min(len(y),offset+16*24),16)]
            for k,(im,ok) in enumerate(pool.map(images,batches)):
                i=offset+k*16
                pred=spatial(features(torch.from_numpy(im).cuda())).argmax(1).cpu().numpy()
                sp.extend(pred[ok]);truth.extend(y[i:i+len(im)][ok])
                speed.extend(mo(speed_images(torch.from_numpy(v[i:i+len(im)]).cuda())).argmax(1).cpu().numpy())
    return dict(spatial_class=cm(np.array(truth),np.array(sp),8),spatial_supported=len(truth),
                speed_pattern=cm(z,np.array(speed),6))


def freq_tv(a,b,e,counts):
    samples=np.array([(np.bincount(a[e==i],minlength=6)-np.bincount(b[e==i],minlength=6)) for i in range(counts.shape[1])])
    n=np.bincount(e,minlength=counts.shape[1])
    return np.sum(abs(counts@samples),axis=1)/(2*(counts@n))


def run():
    torch.set_num_threads(2)
    ids,labs,paths,ref,sref,v,xy=ref_data()
    th=json.loads((OUT/"definition.json").read_text());rz=mode(sref,th)
    classifier_eval=aux(xy,labs,v,rz)
    ep=np.unique([int(i.split("_")[0]) for i in ids],return_inverse=True)[1]
    assert len(np.unique(ep))==8
    mult=np.vstack([np.ones(8),np.random.default_rng(230552).multinomial(8,np.ones(8)/8,2000)])
    methods=["uniform","spatial","speed","speed_blind","joint","numeric"]
    boot={m:np.zeros((len(mult),4)) for m in methods}
    byclass={m:{} for m in methods}
    changes={m:0 for m in methods}
    matching={m:0 for m in methods}
    caps=bad=0
    for seed in (42,43):
        p=np.load(OUT/f"pool-{seed}.npz");pick=np.load(OUT/f"selected-{seed}.npz")
        assert np.array_equal(ids,p["ids"]) and np.array_equal(ids,pick["ids"])
        caps+=int(p["bounded"].sum());bad+=int(pick["failures"])
        outcomes=[]
        for path,clock,rows in zip(paths,p["duration"],p["progress"]):
            outcomes.append([phys(*decode(path,t,prog)) for t,prog in zip(clock,rows)])
        outcomes=np.asarray(outcomes)
        for m in methods:
            slot=pick[m]
            gen=outcomes[np.arange(len(ids)),slot]
            z=mode(gen[:,[3,2]],th)
            changes[m]+=int(np.sum(slot!=pick["uniform"]))
            matching[m]+=int(np.sum(z==pick["desired"]))
            for c,name in enumerate(NAMES):
                mask=labs==c
                curves=[episode_w1(gen[mask,j,None],ref[mask,j,None],ep[mask],False,mult) for j in range(3)]
                curves.append(freq_tv(z[mask],rz[mask],ep[mask],mult))
                arr=np.stack(curves,axis=1)
                boot[m]+=arr/16
                if name not in byclass[m]:byclass[m][name]=np.zeros(4)
                byclass[m][name]+=arr[0]/2
    differences={}
    def diff(a,b):
        d=boot[a]-boot[b]
        differences[a+"-"+b]=dict(delta=d[0].tolist(),ci=np.quantile(d[1:],[.025,.975],axis=0).T.tolist())
    for m in methods[1:]:diff(m,"uniform")
    for a,b in [("speed","spatial"),("speed","speed_blind"),("speed","numeric"),("joint","speed"),("joint","spatial")]:
        diff(a,b)
    result=dict(requests=len(ids),classes=np.bincount(labs).tolist(),episodes=8,seeds=[42,43],candidates=int(len(ids)*8),
        caps=caps,encoding_failures=bad,classifier_eval=classifier_eval,changes_vs_uniform=changes,
        matches_class_directed_target=matching,columns=["longest_steady_W1_s","duration_W1_s","lag4_W1_mps","class_mode_frequency_TV"],
        observed_mode_by_participant=np.bincount(6*labs+rz,minlength=48).reshape(8,6).tolist(),
        macro={k:a[0].tolist() for k,a in boot.items()},
        by_class={m:{k:v.tolist() for k,v in d.items()} for m,d in byclass.items()},differences=differences)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":run()
