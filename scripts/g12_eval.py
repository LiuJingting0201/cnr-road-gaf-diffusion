"""Compare candidate selection using observed movement only, paired by simulation."""
import json
from pathlib import Path
import numpy as np
from cnr.data import CLASSES
from cnr.topology import LaneConsistency
from cnr.episode_w1 import episode_w1
base=Path('g12-data');q=np.load(base/'requests.npz');r=np.load(base/'references.npz');z=np.load(base/'rank.npz');info=json.loads((base/'scorer.json').read_text())
assert np.array_equal(q['ids'],r['ids'])
y=q['labels'];ep=q['episode']-260301
cnt=np.vstack([np.ones(8),np.random.default_rng(75342).multinomial(8,np.ones(8)/8,2000)])
m=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64);cache={}
def motion(x):
    v=np.linalg.norm(np.diff(x,axis=1),axis=-1)/.25
    return v,np.diff(v)/.25,v.sum(1,keepdims=True)*.25,(v<.1).mean(1,keepdims=True)
real=motion(r['xy']);draws={};out={}
for k in z.files:
    x=z[k];assert np.isfinite(x).all()
    val=np.zeros((len(cnt),4));vs=motion(x);valid=0
    for c in np.unique(y):
        ix=y==c
        for j in range(4):
            val[:,j]+=episode_w1(vs[j][ix],real[j][ix],ep[ix],j==3,cnt)/len(np.unique(y))
    for c,path in zip(y,x):
        key=(int(c),path.tobytes())
        if key not in cache:
            g=m.match(path,CLASSES[c]);cache[key]=g['graph_consistent'] and not g['candidate_truncation_warning']
        valid+=int(cache[key])
    draws[k]=val
    within=np.array([d[k] for d in info['within_pool']]);wm=np.stack([within[y==c].mean(0) for c in np.unique(y)]).mean(0)
    out[k]=dict(quality=val[0].tolist(),valid=valid,caps=info['caps'][k],within_vs_uniform=wm.tolist())
for k in ['selected','calibrated','original']:
    for other in ['random','original']:
        if k==other:continue
        delta=draws[k]-draws[other]
        out[f'{k}_minus_{other}']=dict(delta=delta[0].tolist(),ci=np.percentile(delta[1:],[2.5,97.5],axis=0).T.tolist())
out['same_pick_calibrated_vs_selected']=int(sum(d['indices']['calibrated']==d['indices']['selected'] for d in info['within_pool']))
out['n']=len(y)
(base/'quality.json').write_text(json.dumps(out,indent=1))
