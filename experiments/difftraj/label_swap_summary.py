from pathlib import Path
from itertools import combinations
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

base=Path('/workspace/cnr-experiments/beijing-geolife-difftraj')
data=np.load(base/'pilot.npz')
ids=np.flatnonzero(~data['train'])
classes=list(data['classes'])
label=data['y'][ids]; h=data['h'][ids]; dt=data['dt'][ids]
def features(x):
    speed=np.linalg.norm(np.diff(x,axis=2),axis=1)/dt[:,None]
    return np.column_stack((speed.mean(1),np.quantile(speed,.9,axis=1),(speed<1).mean(1),np.abs(np.diff(speed,axis=1)).mean(1)/dt))
real=features(data['xy'][ids])
def avg(x):
    return x.mean(0).tolist()
out=dict(features=['mean_speed_mps','p90_speed_mps','slow_fraction','mean_abs_accel_mps2'],modes=classes,real={m:avg(real[label==i]) for i,m in enumerate(classes)},runs={},matches={})
pairs={}
for a,b in combinations(range(4),2):
    ia,ib=[np.flatnonzero(label==k) for k in (a,b)]
    x,y=h[ia,None],h[None,ib]
    num=np.abs(x[:,:,1:6]-y[:,:,1:6])
    clock=np.abs(x[:,:,0]-y[:,:,0]); clock=np.minimum(clock,288-clock)
    od=np.concatenate([np.abs(x[:,:,6:]//16-y[:,:,6:]//16),np.abs(x[:,:,6:]%16-y[:,:,6:]%16)],axis=2)
    ok=(num.max(2)<=.75)&(clock<=36)&(od.max(2)<=2)
    cost=num.sum(2)+clock/36+od.sum(2)/2
    i,j=linear_sum_assignment(np.where(ok,cost,1e7))
    keep=ok[i,j]; i,j=ia[i[keep]],ib[j[keep]]
    name=f'{classes[b]} minus {classes[a]}'
    pairs[name]=(a,b,np.r_[i,j])
    out['matches'][name]=dict(n=len(i),exact=int(np.all(x==y,axis=2).sum()),real=avg(real[j]-real[i]) if len(i) else None)
for arm in ('C','VC'):
    for seed in (42,43):
        old=np.load(f'/workspace/cnr-experiments/road-check-v1/bj-{arm}-{seed}/private-{arm}-8000.npz')
        f=[]; err=0
        for k,m in enumerate(classes):
            d=np.load(f'{arm}-{seed}-{m}/private-{arm}-8000.npz')
            assert np.array_equal(d['ids'],ids) and np.array_equal(d['dt'],dt)
            x=d['generated']; mask=label==k
            err=max(err,float(np.abs(x[mask]-old['generated'][mask]).max()))
            f.append(features(x))
        assert err<0.05,err
        f=np.asarray(f)
        r=dict(original_replay_max_m=err,by_label={m:avg(f[k]) for k,m in enumerate(classes)},median_within_request_range=np.median(np.ptp(f,axis=0),axis=0).tolist(),pairs={})
        for name,(a,b,ix) in pairs.items():
            diff=f[b]-f[a]
            r['pairs'][name]=dict(all=avg(diff),matched=avg(diff[ix]) if len(ix) else None)
        out['runs'][f'{arm}-{seed}']=r
Path('label-swap.json').write_text(json.dumps(out,indent=2,allow_nan=False))
