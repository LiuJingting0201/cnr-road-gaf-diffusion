"""Independent raw-series downstream utility; fixed synthetic counts and fits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from g33_train import digest,save
from g33_evaluate import read_pool
from g35c_teachers import load_dataset
from g33_teachers import metrics

ARMS=('B','F-SV','J-S','J-V','J-SV','J-RAW')
REGIMES=('real',)+tuple(f'only-{a}' for a in ('empirical',*ARMS))+tuple(f'plus-{a}' for a in ('empirical',*ARMS))


def rank(value):return hashlib.sha256(('G40utility:'+str(value)).encode()).digest()


def series(t,xy):
    t=np.asarray(t,float);xy=np.asarray(xy,float);t=t-t[0]
    if len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(xy).all():raise ValueError('Invalid observed series')
    q=np.linspace(0,t[-1],257)
    v=np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(t)
    return np.stack([np.interp(q,t,xy[:,0]),np.interp(q,t,xy[:,1]),
                     np.interp(q,(t[1:]+t[:-1])/2,v)]).astype(np.float32)


def prepare(root,data,out):
    out.mkdir(parents=True,exist_ok=False)
    try:
        closed=json.loads((root/'complete.json').read_text());assert closed['complete'] and closed['route_duration_equal_across_arms']
        _,raw=load_dataset(data);train=np.flatnonzero(raw['partition']=='train');realix=[]
        for c in range(5):
            eligible=sorted([i for i in train if raw['labels'][i]==c],key=lambda i:rank(raw['ids'][i]))
            seen=set();chosen=[]
            for i in eligible:
                if raw['parent_ids'][i] in seen:continue
                seen.add(raw['parent_ids'][i]);chosen.append(i)
                if len(chosen)==200:break
            assert len(chosen)==200,'Insufficient distinct real train parents'
            realix.extend(chosen)
        real=[]
        for i in realix:
            a,b=raw['offset'][i:i+2];real.append(series(raw['time'][a:b],raw['xy'][a:b]))
        real=np.stack(real);mean=real[:,:2].mean((0,2));sd=np.maximum(real[:,:2].std((0,2)),1.)
        def normalize(x):
            x=x.copy();x[:,:2]=(x[:,:2]-mean[None,:,None])/sd[None,:,None];x[:,2]/=35.;return x
        y=raw['labels'][realix];assert np.array_equal(y,np.repeat(np.arange(5),200))
        rng=np.random.default_rng(40017);empix=np.concatenate([rng.choice(np.flatnonzero(y==c),200,replace=True) for c in range(5)])
        artifacts={};parts={'real':normalize(real),'empirical':normalize(real[empix])};realparents=set(raw['parent_ids'][realix])
        del raw,real
        trainparents=set();testparents=set();request_meta={}
        for e in range(400201,400209):
            rows=json.loads((root/str(e)/'requests/requests.json').read_text())['requests']
            for r in rows:
                request_meta[r['id']]=r
                if r['label']!=0:(trainparents if e<=400204 else testparents).add(r['source_parent'])
        assert not trainparents & testparents
        candidates={a:{} for a in ARMS}
        for a in ARMS:
            for e in range(400201,400205):
                for seed in (42,43):
                    folder=root/str(e)/'generated'/f'{a}-{seed}';rec=json.loads((folder/'complete.json').read_text())
                    assert rec['complete'] and digest(folder/'candidates.json')==rec['artifacts']['candidates.json']
                    for r in json.loads((folder/'candidates.json').read_text())['rows']:
                        key=f'{e}:{seed}:{r["id"]}';candidates[a][key]=(folder,r['index'],r['label'])
        common=set.intersection(*(set(candidates[a]) for a in ARMS));selected=[]
        for c in range(5):
            eligible=sorted([k for k in common if candidates['B'][k][2]==c],key=rank)
            assert len(eligible)>=200,'Insufficient shared synthetic identities'
            selected.extend(eligible[:200])
        for a in ARMS:
            values=[None]*1000;folders={v[0] for k,v in candidates[a].items() if k in set(selected)}
            for folder in folders:
                rec=json.loads((folder/'complete.json').read_text());assert digest(folder/'samples.npz')==rec['artifacts']['samples.npz']
                with np.load(folder/'samples.npz',allow_pickle=False) as f:
                    offsets,txy_all=f['offsets'],f['txy']
                for j,k in enumerate(selected):
                    where,i,label=candidates[a][k]
                    assert label==y[j]
                    if where!=folder:continue
                    lo,hi=offsets[i:i+2];txy=txy_all[lo:hi];values[j]=series(txy[:,0],txy[:,1:])
            assert all(v is not None for v in values);parts[a]=normalize(np.stack(values))
        for name,x in parts.items():
            np.savez_compressed(out/f'{name}.npz',x=x,y=y)
            artifacts[name+'.npz']={'sha256':digest(out/f'{name}.npz')}
        test=[];labels=[];episodes=[];parents=[];ids=[];support={}
        for e in range(400205,400209):
            receipt,f=read_pool(root/str(e)/'reference','references.npz');support[str(e)]={'admitted':len(f['ids']),'requested':100}
            for i,ident in enumerate(f['ids']):
                r=request_meta[str(ident)];assert int(f['labels'][i])==r['label']
                assert r['source_parent'] not in realparents
                lo,hi=f['offsets'][i:i+2];txy=f['txy'][lo:hi];test.append(series(txy[:,0],txy[:,1:]));labels.append(r['label'])
                episodes.append(e);parents.append(r['source_parent']);ids.append(str(ident))
        np.savez_compressed(out/'evaluation.npz',x=normalize(np.stack(test)),y=np.array(labels),episode=np.array(episodes),source_parent=np.array(parents),ids=np.array(ids))
        artifacts['evaluation.npz']={'sha256':digest(out/'evaluation.npz')}
        np.savez_compressed(out/'identities.npz',real_parent=np.array(sorted(realparents)),synthetic_identity=np.array(selected))
        save(out/'complete.json',{'complete':True,'real_n':1000,'synthetic_n_each':1000,'classes':5,'evaluation_support':support,
            'mean_xy':mean.tolist(),'sd_xy':sd.tolist(),'normalizer_real_train_only':True,'nonbus_condition_parent_disjoint':True,
            'shared_bus_templates':True,'source_G40_sha256':digest(root/'complete.json'),'artifacts':artifacts,
            'evaluation_loaded_for_training':False})
    except BaseException as e:save(out/'failed.json',{'error':repr(e)});raise


class Block(nn.Module):
    def __init__(self,d):
        super().__init__();self.conv=nn.Conv1d(64,64,3,padding=d,dilation=d);self.norm=nn.GroupNorm(8,64)
    def forward(self,x):return x+F.silu(self.norm(self.conv(x)))


class Evaluator(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Conv1d(3,64,1);self.blocks=nn.Sequential(*[Block(d) for d in (1,2,4,8)]);self.fc=nn.Linear(128,5)
    def forward(self,x):
        h=self.blocks(self.input(x));return self.fc(torch.cat([h.mean(-1),h.amax(-1)],dim=1))


def load(cache,name):
    rec=json.loads((cache/'complete.json').read_text());assert rec['complete'] and digest(cache/(name+'.npz'))==rec['artifacts'][name+'.npz']['sha256']
    with np.load(cache/(name+'.npz'),allow_pickle=False) as f:return {k:f[k] for k in f.files}


def fit(cache,regime,seed,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
        torch.backends.cudnn.benchmark=False
        if regime=='real':data=load(cache,'real')
        else:
            kind,source=regime.split('-',1);data=load(cache,source)
            if kind=='plus':
                real=load(cache,'real');data={k:np.concatenate([real[k],data[k]]) for k in ('x','y')}
        model=Evaluator().cuda();initial=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
        x=torch.as_tensor(data['x'],device='cuda');y=torch.as_tensor(data['y'],device='cuda');groups=[np.flatnonzero(data['y']==c) for c in range(5)]
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);rng=np.random.default_rng(seed)
        save(out/'config.json',{'regime':regime,'seed':seed,'steps':2000,'rows':len(x),'parameters':sum(p.numel() for p in model.parameters()),
            'initial_sha256':initial,'source_sha256':digest(__file__),'cache_sha256':digest(cache/'complete.json'),'evaluation_used_for_selection':False})
        with (out/'training.jsonl').open('w',buffering=1) as f:
            for step in range(1,2001):
                labels=rng.integers(0,5,64);ix=np.array([rng.choice(groups[c]) for c in labels])
                logits=model(x[ix]);loss=F.cross_entropy(logits,y[ix]);assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if step==1 or step%100==0:
                    row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start};f.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        torch.save(model.state_dict(),out/'model.pt');model.eval()
        # Holdout is loaded only after all updates and final checkpoint closure.
        evaluation=load(cache,'evaluation');logits=[]
        with torch.no_grad():
            for i in range(0,len(evaluation['x']),64):logits.append(model(torch.as_tensor(evaluation['x'][i:i+64],device='cuda')).cpu().numpy())
        z=np.concatenate(logits);labels=evaluation['y'];per={}
        for e in np.unique(evaluation['episode']):
            ix=evaluation['episode']==e;per[str(e)]=metrics(labels[ix],z[ix])
        nonbus=labels!=0;counts=Counter(evaluation['source_parent'][nonbus]);weight=np.array([1/counts[p] if c!=0 else 0 for p,c in zip(evaluation['source_parent'],labels)])
        cm=np.bincount(5*labels+z.argmax(1),weights=weight,minlength=25).reshape(5,5)
        f1=2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1e-12)
        np.savez_compressed(out/'predictions.npz',logits=z,**{k:evaluation[k] for k in ('y','episode','source_parent','ids')})
        save(out/'complete.json',{'complete':True,'regime':regime,'seed':seed,'seconds':time.monotonic()-start,'pooled':metrics(labels,z),
            'per_episode':per,'nonbus_source_parent_equal_macro_f1':float(f1[1:].mean()),'nonbus_weighted_confusion':cm.tolist(),
            'bus_f1':metrics(labels,z)['per_class_f1'][0],'model_sha256':digest(out/'model.pt'),
            'predictions_sha256':digest(out/'predictions.npz'),'GAF_or_teacher_weights_used':False})
    except BaseException as e:save(out/'failed.json',{'error':repr(e),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--root',type=Path);p.add_argument('--data',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--regime',choices=REGIMES);p.add_argument('--seed',type=int,choices=[4001,4002]);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.data,a.output)
    else:fit(a.cache,a.regime,a.seed,a.output)
