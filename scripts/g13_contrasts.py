"""Secondary contrasts on their own common-route populations."""
import json
from pathlib import Path
import numpy as np
base=Path('docs/results/g13-shared-road');primary=json.loads((base/'summary.json').read_text());out={'bus':primary}
for c,name in [(2,'motorcycle'),(4,'bycicle'),(5,'taxi_or_uber')]:
    files=[json.loads((base/f'class{c}-switch{s}.json').read_text())['rows'] for s in [42,43,44]]
    assert all([(r['id'],r['permitted']) for r in rows]==[(r['id'],r['permitted']) for r in files[0]] for rows in files)
    accepted=[[r for r in rows if all(r['permitted'])] for rows in files]
    ep=np.array([r['episode'] for r in accepted[0]])-260401
    res={'attempts':len(files[0]),'routes':len(ep),'by_episode':np.bincount(ep,minlength=8).tolist(),'valid':sum(r[k]['valid'] for rows in accepted for r in rows for k in [name,'private_car']),'caps':sum(r[k]['caps'] for rows in accepted for r in rows for k in [name,'private_car'])}
    w=np.random.default_rng(530711).multinomial(8,np.ones(8)/8,2000)[:,ep]
    assert w.sum(1).min()>0
    for k in ['travel','stop']:
        a=np.array([[[np.mean(r[x][k]) for x in [name,'private_car']] for r in rows] for rows in accepted]);d=a[:,:,0]-a[:,:,1];v=d.mean(0)
        res[k]=dict(means=a.mean((0,1)).tolist(),difference=float(v.mean()),seed_differences=d.mean(1).tolist(),ci=np.percentile(w@v/w.sum(1),[2.5,97.5]).tolist())
    out[name]=res
(base/'contrasts.json').write_text(json.dumps(out,indent=1))
for k,v in out.items():print(k,v['routes'],v['valid'],v['caps'],[round(v[j]['difference'],3) for j in ['travel','stop']],v['travel']['ci'],v['stop']['ci'])
