"""Prespecified descriptive route-burden strata; same episode draws as primary."""
from pathlib import Path
import numpy as np,json
from cnr.episode_w1 import episode_w1
from g11_eval import metrics
base=Path('g11-data');q=np.load(base/'requests.npz');y=q['labels'];ep=q['episode']-260201
ds=[np.load(base/f'{s}.npz') for s in (42,43,44)]
r=metrics(np.load(base/'references.npz')['xy'][:,None])
hi=np.zeros(len(y),bool)
for c in np.unique(y):
    ix=np.flatnonzero(y==c);hi[ix[np.argsort(ds[0]['burden'][ix],kind='stable')[len(ix)//2:]]]=True
counts=np.vstack([np.ones(8),np.random.default_rng(53415).multinomial(8,np.ones(8)/8,2000)])
out={}
for group,mask in [('higher',hi),('lower',~hi)]:
    out[group]={};scores={}
    for arm in ['mid','terminal','placebo','placebo_terminal']:
        a=np.zeros((len(counts),4))
        for d in ds:
            v=metrics(d[arm])
            for c in np.unique(y):
                ix=(y==c)&mask;n=sum(ix)
                for j in range(4):
                    a[:,j]+=episode_w1(v[j][ix].reshape(n,-1),r[j][ix].reshape(n,-1),ep[ix],j==3,counts)/15
        scores[arm]=a
    active=scores['mid']-scores['terminal'];control=scores['placebo']-scores['placebo_terminal']
    for key,v in [('adaptive',active),('timing_interaction',active-control)]:
        out[group][key]={'point':v[0].tolist(),'CI':np.percentile(v[1:],[2.5,97.5],axis=0).T.tolist()}
(base/'groups.json').write_text(json.dumps(out,indent=1))
