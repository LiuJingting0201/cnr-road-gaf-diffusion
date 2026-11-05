"""Frozen cross-reader only after sampling finishes; paired episode evaluation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import torch
from torchvision.models import resnet18
from cnr.episode_w1 import episode_w1
from g12_fit import features
from g16_data import NAMES
from g19_gaf import images
from g22_diffusion import decode
from g23_eval import phys
from g23_fit import classifier
from g23_modes import mode,speed_images,speed_series
from g24_sample import OUT,SPEED


def encode(args):
    path,T,P=args
    xy=[];profile=[];metric=[]
    for dur,p in zip(T,P):
        t,x=decode(path,dur,p)
        xy.append(x)
        profile.append(speed_series(t,x)[0])
        metric.append(phys(t,x))
    rgb,ok=images(xy)
    return rgb,ok,np.array(profile),np.array(metric)


def interval(x):
    return dict(estimate=float(x[0]),ci=np.quantile(x[1:],[.025,.975]).tolist())


def macro_mean(x,y,ep,cnt):
    curves=[]
    for label in np.unique(y):
        mask=y==label
        tot=np.bincount(ep[mask],minlength=cnt.shape[1])
        val=np.bincount(ep[mask],weights=x[mask],minlength=cnt.shape[1])
        curves.append((cnt@val)/(cnt@tot))
    return np.mean(curves,axis=0)


@torch.no_grad()
def run():
    torch.set_num_threads(2)
    q=np.load(OUT/"requests.npz");labels=q["labels"]
    ep=np.unique([str(i).split("_")[0] for i in q["ids"]],return_inverse=True)[1];n=int(ep.max()+1)
    cnt=np.vstack([np.ones(n),np.random.default_rng(2419).multinomial(n,np.ones(n)/n,2000)])
    spatial=resnet18(weights=None);spatial.fc=torch.nn.Linear(512,8)
    spatial.load_state_dict(torch.load("../g22-job/g22-data/cnn-native-gaf/model.pt",weights_only=True,map_location="cpu"))
    spatial.cuda().eval();reader=classifier().eval()
    reader.load_state_dict(torch.load(SPEED/"speed-mode.pt",weights_only=True,map_location="cuda"))
    threshold=json.loads((SPEED/"definition.json").read_text())
    refs=np.load(OUT/"references.npz")
    assert np.array_equal(q["ids"],refs["ids"])
    txy=refs["txy"];off=refs["offset"]
    truth=np.array([phys(txy[a:b,0],txy[a:b,1:]) for a,b in zip(off,off[1:])])
    truth[:,1]=refs["duration"]
    stages={};audit={}
    for arm in ("plain","guided","permuted"):
        audit[arm]=dict(encoding_failures=0,bounds=0,flat=0,seconds=[],mean_abs_time_shift_s=[],mean_abs_profile_shift=[])
        for seed in (42,43):
            d=np.load(OUT/f"{arm}-{seed}.npz")
            source=np.load(OUT/f"plain-{seed}.npz")
            assert np.array_equal(d["ids"],q["ids"]) and np.array_equal(d["target"],source["target"])
            assert np.array_equal(np.sort(d["guided_target"]),np.sort(d["target"]))
            audit[arm]["seconds"].append(float(d["seconds"]))
            audit[arm]["bounds"]+=int(d["bounded"].sum());audit[arm]["flat"]+=int(d["flat"].sum())
            audit[arm]["mean_abs_time_shift_s"].append(float(abs(d["duration"]-source["duration"]).mean()))
            audit[arm]["mean_abs_profile_shift"].append(float(abs(d["progress"]-source["progress"]).mean()))
            pp=[];hits=[];ss=[];ms=[];modehit=[];valid=[]
            jobs=list(zip(q["paths"],d["duration"],d["progress"]))
            with ProcessPoolExecutor(max_workers=8) as pool:
                for start in range(0,len(jobs),16):
                    entries=list(pool.map(encode,jobs[start:start+16]))
                    rgb=np.concatenate([v[0] for v in entries])
                    profiles=np.concatenate([v[2] for v in entries])
                    p=spatial(features(torch.from_numpy(rgb).cuda())).softmax(1).cpu().numpy().reshape(-1,4,8)
                    s=reader(speed_images(torch.from_numpy(profiles).cuda())).softmax(1).cpu().numpy().reshape(-1,4,6)
                    for j,(_,ok,_,vals) in enumerate(entries):
                        i=start+j;c=labels[i];target=d["target"][i]
                        valid.extend(ok)
                        pp.append(p[j,:,c]);hits.append(p[j].argmax(1)==c)
                        ss.append(s[j,:,target]);ms.append(vals)
                        modehit.append(mode(vals[:,[3,2]],threshold)==target)
            audit[arm]["encoding_failures"]+=int(len(valid)-sum(valid))
            # A valid score for every sample is required for this paired primary endpoint.
            if not all(valid):raise ValueError("Encoding failures need an explicit common-support analysis")
            scores=np.stack([pp,hits,ss,modehit],axis=-1)
            physvals=np.array(ms)
            slots=np.argmax(ss,axis=1)
            ii=np.arange(len(slots))
            for name,x,vals in [(arm,scores,physvals),(arm+"_terminal",scores[ii,slots,None],physvals[ii,slots,None])]:
                if name not in stages:stages[name]=dict(scores=[],physics=[],by_class=[])
                stages[name]["scores"].append(np.stack([macro_mean(x[:,:,j].mean(1),labels,ep,cnt) for j in range(4)]))
                stages[name]["by_class"].append([float(x[labels==c,:,0].mean()) for c in range(8)])
                w=[]
                for j in range(3):
                    w.append(sum(episode_w1(vals[labels==c,:,j],truth[labels==c,j,None],ep[labels==c],False,cnt) for c in range(8))/8)
                stages[name]["physics"].append(np.stack(w))
    summary={};diff={}
    for k,st in stages.items():
        for key in ("scores","physics","by_class"):st[key]=np.mean(st[key],axis=0)
        summary[k]=dict(scores=st["scores"][:,0].tolist(),physics=st["physics"][:,0].tolist(),
                        spatial_probability_by_class=dict(zip(NAMES,st["by_class"])))
    for a,b in [("guided","plain"),("permuted","plain"),("guided","permuted"),("plain_terminal","plain"),
                ("guided","plain_terminal"),("guided_terminal","plain_terminal")]:
        diff[a+"-"+b]={k:[interval(v) for v in stages[a][k]-stages[b][k]] for k in ("scores","physics")}
    result=dict(n=len(labels),episodes=n,draws_per_arm=len(labels)*4*2,score_columns=["spatial_target_probability","spatial_top1",
        "speed_target_probability","actual_speed_mode_match"],physics_columns=["longest_steady_W1_s","trip_time_W1_s","four_second_change_W1_mps"],
        stages=summary,differences=diff,audit=audit)
    (OUT/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":run()
