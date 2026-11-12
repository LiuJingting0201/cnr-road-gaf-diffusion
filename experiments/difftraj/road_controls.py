"""Positive, timing and map-coverage controls using the exact matcher and distance code."""
import json
from pathlib import Path
import numpy as np
from road_match import init, match
d=np.load('tracks/Porto-real.npz')
s=np.load('tracks/Porto-real-status.npz')
idx=next(i for i in np.flatnonzero(s['passed']) if d['ends'][i+1]-d['ends'][i]>=10)
a,b=d['ends'][idx:idx+2]
p=d['points'][a:b]
init(Path('Porto.json').resolve())
results={}
for label,coords,dt in [('original',p,15.),('very_fast',p,0.015),('outside',p+[20.,0.],15.)]:
    row=match((coords,dt,'taxi'))
    results[label]={k:row[k] for k in ('n','linked','reasonable','walk','passed')}
assert results['original']['passed']==1
assert results['very_fast']['walk']==1 and results['very_fast']['passed']==0
assert results['outside']['passed']==results['outside']['linked']==0
Path('metrics/controls.json').write_text(json.dumps(results,indent=2))
