from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

root=Path('/workspace/cnr-experiments')
d=np.load(root/'beijing-geolife-difftraj/pilot.npz')
ids=np.flatnonzero(~d['train']); h=d['h'][ids]; dt=d['dt'][ids]
labels=d['classes'][d['y'][ids]]
ia,ib=[np.flatnonzero(labels==k) for k in ('bike','bus')]
x,y=h[ia,None],h[None,ib]
num=np.abs(x[:,:,1:6]-y[:,:,1:6])
clock=np.abs(x[:,:,0]-y[:,:,0]);clock=np.minimum(clock,288-clock)
od=np.concatenate([np.abs(x[:,:,6:]//16-y[:,:,6:]//16),np.abs(x[:,:,6:]%16-y[:,:,6:]%16)],axis=2)
ok=(num.max(2)<=.75)&(clock<=36)&(od.max(2)<=2)
i,j=linear_sum_assignment(np.where(ok,num.sum(2)+clock/36+od.sum(2)/2,1e7))
keep=ok[i,j];a,b=ia[i[keep]],ib[j[keep]]
assert len(a)==94
def features(pos):
    v=np.linalg.norm(np.diff(pos,axis=2),axis=1)/dt[:,None]
    slow=v<1
    bouts=(~slow[:,:-2]&slow[:,1:-1]&slow[:,2:]).sum(1)*600/(199*dt)
    return np.column_stack((v.mean(1),np.quantile(v,.9,axis=1),slow.mean(1)*100,np.abs(np.diff(v,axis=1)).mean(1)/dt,bouts)),v
ref,rv=features(d['xy'][ids])
truth=(ref[b]-ref[a]).mean(0)
out=dict(pairs=len(a),features=['mean_mps','p90_mps','slow_percentage_points','mean_abs_acceleration','slow_entries_per_10min'],real=truth.tolist(),arms={})
for name in ['full','no_mean','no_distance_cues']:
    for seed in [42,43]:
        vs=[]; fs=[]
        for mode in ['bike','bus']:
            f=root/f'bj-label-swap-01/C-{seed}-{mode}/private-C-8000.npz' if name=='full' else Path(f'{name}-{seed}/{mode}.npz')
            draw=np.load(f)
            assert np.array_equal(draw['ids'],ids) and np.array_equal(draw['dt'],dt)
            feat,v=features(draw['generated']);fs.append(feat);vs.append(v)
        diff=(fs[1]-fs[0])[np.r_[a,b]].mean(0)
        w1={mode:float(wasserstein_distance(rv[labels==mode].ravel(),v[labels==mode].ravel())) for mode,v in zip(['bike','bus'],vs)}
        out['arms'][f'{name}-{seed}']=dict(delta=diff.tolist(),abs_error=np.abs(diff-truth).tolist(),original_label_speed_w1=w1)
Path('summary.json').write_text(json.dumps(out,indent=2))
