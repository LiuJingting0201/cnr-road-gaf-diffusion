"""Equal-budget generation evaluation. Reference futures enter only after sampling."""
import json
from pathlib import Path
import numpy as np
import torch
from cnr.episode_w1 import episode_w1
from cnr.path_diffusion import PathDenoiser
from g16_data import NAMES
from g22_diffusion import decode,sample as flat_sample
from g23_modes import mode
from g23_eval import phys
from g26_train import OUT,ARMS,STEPS,K,Denoiser,sample


def plain(req,seed):
    old=Path("../g22-job/g22-data")/f"model-{seed}"
    net=PathDenoiser(classes=8).cuda().eval()
    net.load_state_dict(torch.load(old/"model.pt",weights_only=True,map_location="cuda"))
    return flat_sample(net,req,json.loads((old/"stats.json").read_text()),seed)


def evaluate(req):
    r=np.load(OUT/"references.npz")
    assert np.array_equal(r["ids"],req["ids"])
    off=r["offset"];arr=r["txy"]
    reference=np.array([phys(arr[a:b,0],arr[a:b,1:]) for a,b in zip(off,off[1:])])
    reference[:,1]=r["duration"]
    lab=req["labels"];ep=np.unique([i.split("_")[0] for i in req["ids"]],return_inverse=True)[1]
    cnt=np.vstack([np.ones(8),np.random.default_rng(260133).multinomial(8,np.ones(8)/8,2000)])
    thresholds=json.loads(Path("../g23-job/g23-data/definition.json").read_text())
    estimates={};tables={}
    for arm in ("G22",*ARMS):
        curves=[];class_stats=[];caps=0;deviation=0.;target_match=[];elapsed=[]
        for seed in (42,43):
            if arm=="G22":
                pool=plain(req,seed)
                desired=np.load(OUT/f"HM-{seed}/samples.npz")["target"]
            else:
                root=OUT/f"{arm}-{seed}";pool=np.load(root/"samples.npz")
                desired=pool["target"]
                elapsed.append(json.loads((root/"fit.jsonl").read_text().splitlines()[-1])["seconds"])
            assert np.array_equal(req["ids"],pool["ids"])
            caps+=int(pool["bounded"].sum())
            T,P=pool["duration"],pool["progress"]
            values=[]
            for i,path in enumerate(req["paths"]):
                row=[]
                for t,p in zip(T[i],P[i]):
                    ticks,x=decode(path,t,p)
                    deviation=max(deviation,float(np.linalg.norm(x[-1]-path[-1])))
                    row.append(phys(ticks,x))
                values.append(row)
            values=np.array(values)
            rr=mode(values[:,:,[3,2]],thresholds)
            target_match.append(float((rr==desired[:,None]).mean()))
            byclass=[];est=np.zeros((len(cnt),3))
            for c in range(8):
                use=lab==c
                vals=np.stack([episode_w1(values[use,:,j],reference[use,j,None],ep[use],False,cnt) for j in range(3)],axis=1)
                byclass.append(vals[0]);est+=vals/8
            curves.append(est);class_stats.append(byclass)
        estimates[arm]=np.mean(curves,axis=0)
        tables[arm]=dict(macro=estimates[arm][0].tolist(),byclass=dict(zip(NAMES,np.mean(class_stats,axis=0).tolist())),
                         caps=caps,max_endpoint_error_m=deviation,mode_match=float(np.mean(target_match)),fit_elapsed_s=elapsed)
    diff={}
    for a,b in [("H","G22"),("HM","H"),("HV","HM"),("HG","HM"),("HG","HV"),("HG","G22")]:
        delta=estimates[a]-estimates[b]
        diff[a+"-"+b]=dict(delta=delta[0].tolist(),ci=np.quantile(delta[1:],[.025,.975],axis=0).T.tolist())
    return dict(requests=len(lab),counts=np.bincount(lab,minlength=8).tolist(),episodes=8,candidates_per_arm=len(lab)*8,
                columns=["longest_steady_W1_s","travel_time_W1_s","four_second_change_W1_mps"],arms=tables,differences=diff)


def intervention(req):
    # Per requested class use the most-supported pair that shares its speed band.
    train=np.load("../g23-job/g23-data/train.npz")
    counts=np.bincount(6*train["participant"]+train["y"],minlength=48).reshape(8,3,2)
    pair=counts.min(2).argmax(1)
    eligible=counts.min(2).max(1)>=40
    keep=eligible[req["labels"]]
    req={k:v[keep] for k,v in req.items()}
    low=2*pair[req["labels"]]
    output={}
    for arm in ("HM","HV","HG"):
        all_delta=[]
        for seed in (42,43):
            root=OUT/f"{arm}-{seed}"
            net=Denoiser().cuda().eval()
            net.load_state_dict(torch.load(root/"model.pt",weights_only=True,map_location="cuda"))
            stats=json.loads((root/"stats.json").read_text())["stats"]
            draws=[sample(net,arm,req,stats,seed,low+z) for z in (0,1)]
            delta=[]
            for i,path in enumerate(req["paths"]):
                values=[]
                for d in draws:
                    values.append(np.mean([phys(*decode(path,T,P)) for T,P in zip(d["duration"][i],d["progress"][i])],axis=0))
                delta.append(values[1]-values[0])
            all_delta.append(delta)
        d=np.mean(all_delta,axis=0);y=req["labels"]
        ep=np.unique([i.split("_")[0] for i in req["ids"]],return_inverse=True)[1]
        cnt=np.vstack([np.ones(8),np.random.default_rng(260134).multinomial(8,np.ones(8)/8,2000)])
        curves=[]
        for c in np.unique(y):
            sums=np.stack([d[(y==c)&(ep==e)].sum(0) for e in range(8)])
            n=np.bincount(ep[y==c],minlength=8)
            curves.append((cnt@sums)/(cnt@n)[:,None])
        mean=np.mean(curves,axis=0)
        output[arm]=dict(delta=mean[0].tolist(),ci=np.quantile(mean[1:],[.025,.975],axis=0).T.tolist(),
                         byclass={NAMES[c]:d[y==c].mean(0).tolist() for c in np.unique(y)})
    return dict(requests=len(y),selected_band={n:int(pair[c]) for c,n in enumerate(NAMES) if eligible[c]},
                columns=["longest_steady_s","duration_s","four_second_change_mps","mean_speed_mps"],models=output)


if __name__=="__main__":
    torch.set_num_threads(2)
    with np.load(OUT/"requests.npz") as data:req={k:data[k] for k in data.files}
    (OUT/"results.json").write_text(json.dumps(evaluate(req),indent=2))
    (OUT/"switch.json").write_text(json.dumps(intervention(req),indent=2))
