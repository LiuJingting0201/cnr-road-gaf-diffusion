"""Localize real-data calibration failures without changing matcher parameters."""
import json
from pathlib import Path
import numpy as np
from cnr.topology import LaneConsistency

root=Path('docs/results/topology-t1');r=json.loads((root/'results.json').read_text())
real=np.load('data/processed/most64/val.npz');m=LaneConsistency('data/maps/most.net.xml',radius=r['config']['radius_m'])
failures=[]
for row in r['calibration']['rows']:
    if row['graph_consistent']:continue
    x=real['xy'][row['index']];groups,_=m.candidates(x,row['class']);bad=[]
    for t in range(1,len(x)):
        budget=min(m.route_cap,m.route_factor*np.linalg.norm(x[t]-x[t-1])+m.route_slack)
        pairs=[]
        for src,s,sd in groups[t-1]:
            for dst,d,dd in groups[t]:
                length=m.route_distance(src,s,dst,d,row['class'])
                pairs.append({'from_lane':m.road.lanes[src]['id'],'to_lane':m.road.lanes[dst]['id'],
                    'from_attributes':m.road.lanes[src],'to_attributes':m.road.lanes[dst],
                    'from_progress':s,'to_progress':d,'from_distance':sd,'to_distance':dd,
                    'route_distance':float(length) if np.isfinite(length) else None,
                    'forward_successors':[m.road.lanes[i]['id'] for i in m.successors[src]],
                    'fits_budget':bool(length<=budget)})
        if pairs and not any(v['fits_budget'] for v in pairs):
            bad.append({'step':t,'from_xy':x[t-1].tolist(),'to_xy':x[t].tolist(),'budget':float(budget),'pairs':pairs})
    failures.append({'validation_index':row['index'],'class':row['class'],'minimum_breaks':row['minimum_incompatible_transitions'],
                     'locally_incompatible_steps':bad})
(root/'failure_diagnostics.json').write_text(json.dumps(failures,indent=2))
print(json.dumps(failures,indent=2))
