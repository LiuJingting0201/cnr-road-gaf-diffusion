"""Re-score saved long-trip Porto draws at Turin's evaluation size, without refitting."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import jensenshannon
from porto import arc_length,grid,resample_trajectory
root=Path('/workspace/cnr-experiments')
n=json.loads(Path('split.json').read_text())['held_out']
results={}
for seed,path in [(42,root/'porto-metric-review-01/42-long-on-long'),(43,root/'porto-long-repeat43/score')]:
    data=np.load(path/'prepared.npz');raw=np.load(path/'observations.npz');draw=np.load(path/'draws.npz')
    sizes=raw['length'];ends=np.r_[0,sizes.cumsum()];delta=raw['delta']
    bins=np.load(root/'porto-stabilized-v1/shared-bins.npz')['long']
    ids=draw['ids'];lo,hi=data['lo'],data['hi']
    ref=[delta[ends[i]:ends[i+1]].cumsum(0)*1e-6 for i in ids]
    gen=[resample_trajectory(t,int(sizes[i])) for t,i in zip(draw['xy'],ids)]
    def features(trips):
        cells=[grid(t,lo,hi) for t in trips]
        return np.array([np.bincount(c,minlength=256) for c in cells]),np.array([c[0]*256+c[-1] for c in cells]),np.digitize([arc_length(t) for t in trips],bins)
    a,b=map(features,[ref,gen]);rows=[]
    for rep in range(50):
        idx=np.random.default_rng(1300+rep).choice(len(ids),n,replace=False)
        def hist(f):
            return f[0][idx].sum(0),np.bincount(f[1][idx],minlength=65536),np.bincount(f[2][idx],minlength=len(bins)+1)
        ha,hb=hist(a),hist(b)
        top=lambda h:set(np.argsort(-h,kind='stable')[:20])
        rows.append([float(jensenshannon(x,y)**2) for x,y in zip(ha,hb)]+[len(top(ha[0])&top(hb[0]))/20])
    results[str(seed)]=np.quantile(rows,[.1,.5,.9],axis=0).tolist()
Path('porto-equal-n.json').write_text(json.dumps(dict(n=n,quantiles=[.1,.5,.9],metrics=['density','od','length','pattern'],results=results),indent=2))
