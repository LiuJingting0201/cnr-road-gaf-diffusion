import os
from pathlib import Path
import json,numpy as np
from collections import Counter
from cnr.route_paths import RoutePaths
from cnr.data import CLASSES
base=Path(os.getenv('CNR_RUN_ROOT','g10-data'));matcher=RoutePaths('data/maps/most.net.xml',radius=2.)
idx={a['id']:i for i,a in enumerate(matcher.road.lanes)}
prior=json.loads(Path('data/processed/route512/route_prior.json').read_text())
counts={(r['class'],idx[r['from']],idx[r['to']]):r['count'] for r in prior}
rows=[];coverage={}
for seed in range(int(os.getenv('CNR_FIRST_SEED',260101)),int(os.getenv('CNR_FIRST_SEED',260101))+8):
    src=np.load(base/str(seed)/'eligible.npz'); rng=np.random.default_rng(seed+1470000)
    tally=Counter();failure=0
    for ident,c,track in zip(src['ids'],src['labels'],src['xy']):
        if tally[int(c)]>=6:continue
        start=track[0].copy()
        plan=matcher.sample(start,CLASSES[int(c)],counts,rng,budget=4096.)
        if plan is None:failure+=1;continue
        path,length=plan;tally[int(c)]+=1;rows.append((ident,int(c),start,path,length,seed,track))
    coverage[str(seed)]={'selected':dict(tally),'planning_failures_before_quota':failure}
(base/'coverage.json').write_text(json.dumps(coverage,indent=1))
ids,labels,starts,routes,lengths,seeds,real=map(np.array,zip(*rows))
np.savez_compressed(base/'requests.npz',ids=ids,labels=labels,xy=starts[:,None].astype('float32'),route=routes.astype('float32'),route_length=lengths.astype('float32'),episode=seeds)
np.savez_compressed(base/'references.npz',ids=ids,xy=real.astype('float32'))
