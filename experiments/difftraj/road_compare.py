"""Coverage and paired counts using private trip statuses, export aggregates only."""
import json
from pathlib import Path
import numpy as np
bounds={'Beijing':(116.08,39.68,116.77,40.18),'Porto':(-8.82,40.90,-8.2,41.42)}
report={}
for file in sorted(Path('metrics').glob('*.json')):
    name=file.stem
    if not name.startswith(('Porto-','Beijing-')): continue
    city=name.split('-')[0]
    d=np.load('tracks/'+name+'.npz')
    s=np.load('tracks/'+name+'-status.npz')
    p=d['points']; ends=d['ends']; x0,y0,x1,y1=bounds[city]
    inside=((p[:,0]>=x0)&(p[:,0]<=x1)&(p[:,1]>=y0)&(p[:,1]<=y1))
    contained=np.logical_and.reduceat(inside,ends[:-1])
    out=dict(points_outside=int((~inside).sum()),trips_outside=int((~contained).sum()),in_bounds=int(contained.sum()),
             in_bounds_pass=int(s['passed'][contained].sum()))
    if city=='Beijing' and name!='Beijing-real':
        baseline=np.load('tracks/Beijing-B-'+name.split('-')[-1]+'-status.npz')
        assert np.array_equal(s['intervals'],baseline['intervals'])
        out['paired_improved_intervals_trips']=int((s['reasonable']>baseline['reasonable']).sum())
        out['paired_worsened_intervals_trips']=int((s['reasonable']<baseline['reasonable']).sum())
    report[name]=out
Path('metrics/coverage.json').write_text(json.dumps(report,indent=2))
