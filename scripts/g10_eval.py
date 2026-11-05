"""Paired episode bootstrap; a traffic episode, not an output trace, is the unit."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from cnr.topology import LaneConsistency
from cnr.data import CLASSES
root=Path('g10-data');q=np.load(root/'requests.npz');r=np.load(root/'references.npz')
assert np.array_equal(q['ids'],r['ids'])
ds=[np.load(root/f'{s}-76.npz') for s in [42,43,44]]
ep=q['episode']-260101;y=q['labels'];bur=ds[0]['burden'];high=np.zeros(len(y),bool)
for c in np.unique(y):
    ix=np.where(y==c)[0];high[ix[np.argsort(bur[ix],kind='stable')[len(ix)//2:]]]=True
# One row per bootstrap draw, each column the multiplicity of a complete episode.
cnt=np.vstack([np.ones(8),np.random.default_rng(34176).multinomial(8,np.ones(8)/8,2000)])
def metrics(x):
    v=np.linalg.norm(np.diff(x,axis=-2),axis=-1)/.25
    return [v,np.diff(v,axis=-1)/.25,v.sum(-1)*.25,(v<.1).mean(-1)]
def episode_w1(a,b,e,stop):
    n=np.bincount(e,minlength=8)
    denom=cnt@n
    if denom.min()==0: raise ValueError('Unsupported bootstrap group')
    if stop:
        diff=np.bincount(e,weights=a.mean(1)-b.mean(1),minlength=8)
        return abs(cnt@diff/denom)
    grid=np.unique(np.r_[a.ravel(),b.ravel()]);width=np.diff(grid);cdf=np.zeros((8,len(width)))
    for k in np.unique(e):
        aa=np.sort(a[e==k].ravel());bb=np.sort(b[e==k].ravel())
        cdf[k]=n[k]*(np.searchsorted(aa,grid[:-1],'right')/len(aa)-np.searchsorted(bb,grid[:-1],'right')/len(bb))
    w=cnt/denom[:,None]
    z=np.concatenate([np.abs(w[i:i+16]@cdf)@width for i in range(0,len(w),16)])
    assert abs(z[0]-wasserstein_distance(a.ravel(),b.ravel()))<.001
    return z
truth=metrics(r['xy'][:,None]);matcher=LaneConsistency('data/maps/most.net.xml',radius=5.,max_candidates=64)
out={'N':len(y),'n_high':int(high.sum()),'classes':{},'bootstrap_unit':'8 independent SUMO traffic episodes','metrics':['speed W1','acc W1','travel W1','stop fraction abs error']}
variants=['mid','terminal'];draws={};cache={}
for c in range(1,6):
    out['classes'][CLASSES[c]]=[int(sum((y==c)&m)) for m in [high,~high]]
for arm in variants:
    valid=0;vs=[];div=[];caps=0
    for d in ds:
        x=d[arm];assert x.shape==(len(y),3,512,2) and np.isfinite(x).all()
        vs.append(metrics(x));caps+=int(d['cap_'+arm].sum())
        div.append(np.mean([np.linalg.norm(x[:,a]-x[:,b],axis=-1).mean() for a,b in [(0,1),(0,2),(1,2)]]))
        for row,label in zip(x,y):
            for path in row:
                k=(int(label),hashlib.sha256(path.tobytes()).digest())
                if k not in cache:
                    g=matcher.match(path,CLASSES[int(label)])
                    cache[k]=bool(g['graph_consistent'] and not g['candidate_truncation_warning'])
                valid+=int(cache[k])
    out[arm]={'graph_pass':valid,'denominator':len(y)*9,'caps':caps,'diversity':float(np.mean(div))}
    for group,mask in [('high',high),('low',~high),('all',np.ones(len(y),bool))]:
        est=np.zeros((len(cnt),4))
        for c in np.unique(y):
            ix=(y==c)&mask;n=sum(ix)
            for v in vs:
                for j in range(4):
                    est[:,j]+=episode_w1(v[j][ix].reshape(n,-1),truth[j][ix].reshape(n,-1),ep[ix],j==3)/(3*len(np.unique(y)))
        out[arm][group]=est[0].tolist();draws[arm,group]=est
for name in ['high','low','all','interaction']:
    def delta(g):return draws['mid',g]-draws['terminal',g]
    diff=delta(name) if name!='interaction' else delta('high')-delta('low')
    out[name]={'delta':diff[0].tolist(),'CI':np.percentile(diff[1:],[2.5,97.5],axis=0).T.tolist()}
out['encoding_failures']=[d['encoding_failures'].tolist() for d in ds]
(root/'results.json').write_text(json.dumps(out,indent=2))
